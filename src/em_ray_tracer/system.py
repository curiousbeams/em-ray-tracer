"""The forward model: an ordered set of components along the optic axis."""

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Any, Self

import torch
from torch import nn

from em_ray_tracer.components.base import ComponentBase
from em_ray_tracer.optimize.mixin import OptimizerMixin
from em_ray_tracer.transfer_map import TransferMap, drift_map

_TOL = 1e-15


@dataclass(frozen=True)
class Gap:
    """Field-free space between ``z0`` and ``z1``: a drift for every solver."""

    z0: torch.Tensor
    z1: torch.Tensor


@dataclass(frozen=True)
class Element:
    """A component, spanning ``component.z_start`` to ``component.z_end``."""

    component: ComponentBase


Segment = Gap | Element


class OpticalSystem(nn.Module, OptimizerMixin):
    """Components placed along z, the forward model shared by every solver.

    Solvers never look at the components directly. They walk :meth:`segments`, an ordered
    list of :class:`Gap` and :class:`Element`, and each solver decides how to cross each kind.
    That is what makes solvers interchangeable.

    Components are stored in an ``nn.ModuleDict`` keyed by name, so parameters are named
    ``components.<name>.<parameter>`` (e.g. ``components.C1.power``) in ``state_dict`` and for
    ``torch.func.functional_call``. Names are fixed once a component is added.

    The system owns the optimizer (``OptimizerMixin``). Parameters are grouped under one key
    by default, or by component name for per-component learning rates::

        system.set_optimizer({"C1": OptimizerParams.Adam(lr=1e-2),
                              "C2": OptimizerParams.Adam(lr=1e-3)})
    """

    def __init__(self, components: Iterable[ComponentBase] = ()):
        nn.Module.__init__(self)
        OptimizerMixin.__init__(self)
        self.components = nn.ModuleDict()
        for component in components:
            self.add(component)

    # region --- entry points ---

    @classmethod
    def from_components(cls, *components: ComponentBase) -> Self:
        return cls(components)

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> Self:
        """Rebuild from :meth:`to_config` output: ``{"components": [{"type": ..., ...}]}``."""
        return cls(ComponentBase.from_config(c) for c in config["components"])

    def to_config(self) -> dict[str, Any]:
        return {"components": [c.config() for c in self]}

    # endregion --- entry points ---

    # region --- container ---

    def add(self, component: ComponentBase) -> Self:
        if component.name is None:
            base = type(component).__name__
            k = sum(type(c) is type(component) for c in self)
            while f"{base}{k}" in self.names:
                k += 1
            component.name = f"{base}{k}"
        name = component.name
        if name in self.components:
            raise ValueError(f"A component named {name!r} already exists")
        if not name or "." in name or hasattr(self.components, name):
            raise ValueError(f"{name!r} can't be a component name (empty, has '.', or reserved)")
        self.components[name] = component
        return self

    def remove(self, name: str) -> ComponentBase:
        return self.components.pop(name)

    @property
    def names(self) -> list[str]:
        """Component names in the order they were added."""
        return list(self.components.keys())

    def __getitem__(self, key: str | int) -> ComponentBase:
        """By name, or by position along z (``system[0]`` is the most upstream)."""
        if isinstance(key, int):
            return self.ordered()[key]
        if key not in self.components:
            raise KeyError(f"No component named {key!r}; have {self.names}")
        return self.components[key]

    def __iter__(self) -> Iterator[ComponentBase]:
        return iter(self.ordered())

    def __len__(self) -> int:
        return len(self.components)

    def ordered(self) -> list[ComponentBase]:
        """Components sorted by position (ties keep insertion order)."""
        return sorted(self.components.values(), key=lambda c: float(c.z_start.detach()))

    # endregion --- container ---

    # region --- geometry ---

    @property
    def z_last(self) -> torch.Tensor:
        """The downstream end of the last component."""
        if not len(self):
            raise ValueError("The system has no components")
        return max((c.z_end for c in self.components.values()), key=lambda z: float(z.detach()))

    def segments(
        self, z_start: torch.Tensor | float, z_end: torch.Tensor | float | None = None
    ) -> list[Segment]:
        """The schedule from ``z_start`` to ``z_end`` (default: the last component).

        Components wholly outside the span are skipped. Gaps are always emitted, even at zero
        length, so gradients reach every position. Overlapping extended components raise:
        superposing overlapping fields is not implemented yet.
        """
        z_start = torch.as_tensor(z_start, dtype=torch.float64)
        z_end = self.z_last if z_end is None else torch.as_tensor(z_end, dtype=torch.float64)
        lo, hi = float(z_start.detach()), float(z_end.detach())

        segments: list[Segment] = []
        cursor = z_start
        for c in self.ordered():
            c0, c1 = float(c.z_start.detach()), float(c.z_end.detach())
            if c1 < lo - _TOL or c0 > hi + _TOL:
                continue
            if c0 < lo - _TOL or c1 > hi + _TOL:
                raise ValueError(f"{c.name} [{c0}, {c1}] straddles the span [{lo}, {hi}]")
            if c0 < float(cursor.detach()) - _TOL:
                raise ValueError(
                    f"{c.name} starts at {c0}, inside the previous component (ends at "
                    f"{float(cursor.detach())}). Overlapping fields are not supported yet."
                )
            segments.append(Gap(cursor, c.z_start))
            segments.append(Element(c))
            cursor = c.z_end
        segments.append(Gap(cursor, z_end))
        return segments

    # endregion --- geometry ---

    def forward(self, rays, solver=None, z_end=None, record: bool = False):
        """Trace ``rays`` (default solver: first-order transfer matrices) and return the
        ``Trace``. This makes the system usable with ``torch.func`` (``functional_call``,
        ``vmap``, ``jacrev``) as a function of its parameters."""
        from em_ray_tracer.solvers.base import SolverBase
        from em_ray_tracer.solvers.params import SolverParams

        solver = SolverBase.from_params(solver or SolverParams.TransferMatrix())
        return solver.trace(self, rays, z_end=z_end, record=record)

    # region --- first-order optics ---

    def transfer_map(
        self,
        z_start: torch.Tensor | float,
        z_end: torch.Tensor | float | None = None,
        order: int = 1,
        voltage: float | None = None,
    ) -> TransferMap:
        """The composed map from ``z_start`` to ``z_end``. Every component must have one."""
        total = None
        for seg in self.segments(z_start, z_end):
            match seg:
                case Gap(z0, z1):
                    m = drift_map(z1 - z0)
                case Element(c):
                    m = c.transfer_map(order=order, voltage=voltage)
                    if m is None:
                        raise ValueError(f"{c.name} is not a polynomial map; trace rays instead")
            total = m if total is None else m @ total
        return total

    def gaussian_image(
        self, z_object: torch.Tensor | float, voltage: float | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Paraxial image of the object plane ``z_object``, as ``(z_image, magnification)``.

        Closed form, so it is exact and differentiable. A round system with image rotation
        theta maps complex (w, omega) as w' = e^{i theta} (A w + B omega), omega' =
        e^{i theta} (C w + D omega). A further drift d images when B + d D = 0.

        The magnification is complex, e^{i theta} (A + d C): its modulus is the scale and its
        argument the image rotation (Larmor rotation for magnetic lenses; 0 otherwise).
        Astigmatic systems have separate x and y image planes, and this is not defined for
        them.
        """
        A, B, C, D = self._complex_blocks(self.transfer_map(z_object, voltage=voltage))
        d = (-B / D).real
        return self.z_last + d, A + d * C

    def focal_length(self, voltage: float | None = None) -> torch.Tensor:
        """Complex image-side focal length -1 / (e^{i theta} C). Unchanged by drifts before
        or after. Its modulus is |f| and its argument is the image rotation (mod pi)."""
        first = min(float(c.z_start.detach()) for c in self.components.values())
        _, _, C, _ = self._complex_blocks(self.transfer_map(first, voltage=voltage))
        return -1.0 / C

    @staticmethod
    def _complex_blocks(m: TransferMap) -> tuple[torch.Tensor, ...]:
        """e^{i theta} (A, B, C, D) from the x-columns of a round system's map."""
        return tuple(
            torch.complex(m.entry(row, col), m.entry(row.replace("x", "y"), col))
            for row, col in (("x", "x"), ("x", "x'"), ("x'", "x"), ("x'", "x'"))
        )

    # endregion --- first-order optics ---

    # region --- optimization ---

    def get_optimization_parameters(self) -> dict[str, list[torch.Tensor]]:
        learnable = {c.name: [p for p in c.parameters() if p.requires_grad] for c in self}
        keys = set(self.optimizer_params)
        if keys == {self.DEFAULT_OPTIMIZER_KEY}:
            params = [p for ps in learnable.values() for p in ps]
            return {self.DEFAULT_OPTIMIZER_KEY: params} if params else {}
        return {k: learnable[k] for k in keys if learnable.get(k)}

    @property
    def learnable(self) -> dict[str, torch.Tensor]:
        return {f"{c.name}.{n}": p for c in self for n, p in c.learnable.items()}

    # endregion --- optimization ---

    def __repr__(self) -> str:
        rows = "\n".join(f"  {c!r}" for c in self)
        return f"{type(self).__name__}(\n{rows}\n)"
