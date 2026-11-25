"""The forward model: an ordered set of components along the optic axis."""

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from functools import cached_property
from typing import Any, Literal, Self

import torch
from torch import nn

from em_ray_tracer.components.base import ComponentBase
from em_ray_tracer.constants import DTYPE
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
    """A component crossed as one action: a thin component, one kick of a split
    ``DoubleDeflector``, or an extended map component that nothing overlaps."""

    component: ComponentBase


@dataclass(frozen=True, eq=False)
class FieldSpan:
    """``z0`` to ``z1``, crossed by integrating the superposition of ``components``.

    Every field component whose extent covers the interval contributes, in the lab frame:

    - B adds linearly, B(z) = sum_c B_c(z - z_c), and so does each of its derivatives;
    - phi adds linearly too. Each profile is the potential relative to ground and vanishes
      outside its extent, so the beam potential is V + sum_c phi_c(z - z_c). Everything
      nonlinear in phi (gamma, phi_hat, the paraxial coefficients) is computed from the sum,
      never summed per component;
    - the off-axis (Laplace) expansion is linear in the axial profile, so expanding the summed
      derivatives is exact.

    Superposing profiles computed separately assumes the sources don't interact: no saturated
    iron shared between pole pieces, no electrode that changes another lens's potential.
    """

    z0: torch.Tensor
    z1: torch.Tensor
    components: tuple[ComponentBase, ...]

    @classmethod
    def of(cls, component: ComponentBase) -> Self:
        """The span of one field component on its own, over its whole extent."""
        return cls(component.z_start, component.z_end, (component,))

    def axial_derivative(self, n: int, z: torch.Tensor) -> dict[str, torch.Tensor]:
        """The n-th derivative of the summed axial profiles at the lab position ``z``."""
        total: dict[str, torch.Tensor] = {}
        for c in self.components:
            for kind, value in c.axial_derivative(n, z).items():
                total[kind] = value if kind not in total else total[kind] + value
        return total

    # the AxialField interface, in lab z, so a LaplaceExpansion can expand the summed profiles
    derivative = axial_derivative

    @cached_property
    def kinds(self) -> tuple[str, ...]:
        """The field kinds present ("magnetic", "electrostatic"), from one probe in the span."""
        middle = ((self.z0 + self.z1) / 2).detach().reshape(1)
        return tuple(self.axial_derivative(0, middle))

    @property
    def whole(self) -> ComponentBase | None:
        """The component, if the span is one component over its whole extent; else ``None``."""
        if len(self.components) != 1:
            return None
        (c,) = self.components
        if abs(_value(self.z0) - _value(c.z_start)) > _TOL:
            return None
        if abs(_value(self.z1) - _value(c.z_end)) > _TOL:
            return None
        return c

    def transfer_map(
        self, order: int = 1, voltage: float | None = None, r_interest: float | None = None
    ) -> TransferMap:
        """The span's map. A whole component keeps its own (misalignment included); a partial
        or superposed span is mapped by tracing the principal rays through it."""
        if (c := self.whole) is not None:
            return c.transfer_map(order=order, voltage=voltage, r_interest=r_interest)
        if any(c.is_misaligned for c in self.components):
            raise NotImplementedError(
                f"Superposed or clipped field components must be aligned: "
                f"{[c.name for c in self.components]}"
            )
        from em_ray_tracer.solvers.ode import span_transfer_map

        return span_transfer_map(self, order=order, voltage=voltage, r_interest=r_interest)


Segment = Gap | Element | FieldSpan


def _value(z: torch.Tensor | float) -> float:
    return float(z.detach()) if isinstance(z, torch.Tensor) else float(z)


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

        A sweep over breakpoints: the span's ends, every field component's start and end, and
        every thin action. Between consecutive breakpoints the space is a :class:`Gap` if no
        field covers it and a :class:`FieldSpan` of the fields that do. Each thin action is an
        :class:`Element` at its own z. So fields superpose, and thin components (apertures,
        planes, the two kicks of a ``DoubleDeflector``) can sit inside fields.

        - Breakpoints are the components' own tensors, and segments are emitted even at zero
          length, so gradients reach every position.
        - A field that reaches past either end of the span is clipped to it, so rays can start
          or stop inside a field.
        - An extended component that can't be split (``thin_parts()`` is ``None``) is crossed
          whole; anything inside it, or a span that cuts it, raises.
        - Thin components outside the span are skipped.
        """
        z_start = torch.as_tensor(z_start, dtype=torch.float64)
        z_end = self.z_last if z_end is None else torch.as_tensor(z_end, dtype=torch.float64)
        lo, hi = _value(z_start), _value(z_end)

        # (value, tensor, kind, payload); sorted by value, ties keep insertion order
        marks: list[tuple[float, torch.Tensor, str, Any]] = []
        active: list[ComponentBase] = []  # fields that start before the span
        for c in self.ordered():
            c0, c1 = _value(c.z_start), _value(c.z_end)
            if c1 < lo - _TOL or c0 > hi + _TOL:
                continue
            if c.is_field:
                if c1 <= lo + _TOL or c0 >= hi - _TOL:
                    continue  # touches the span at one end only
                if c0 < lo - _TOL:
                    active.append(c)
                else:
                    marks.append((c0, c.z_start, "on", c))
                if c1 <= hi + _TOL:
                    marks.append((c1, c.z_end, "off", c))
                continue
            parts = c.thin_parts()
            if parts is None:
                if c0 < lo - _TOL or c1 > hi + _TOL:
                    raise ValueError(
                        f"{c.name} [{c0}, {c1}] straddles the span [{lo}, {hi}]; an extended "
                        "component that can't be split must lie wholly inside it"
                    )
                marks.append((c0, c.z_start, "whole", c))
                marks.append((c1, c.z_end, "whole_end", c))
                continue
            for part in parts:
                pz = _value(part.z_start)
                if lo - _TOL <= pz <= hi + _TOL:
                    marks.append((pz, part.z_start, "thin", part))
        marks.sort(key=lambda m: m[0])

        segments: list[Segment] = []
        cursor = z_start
        inside: ComponentBase | None = None  # an unsplittable component being crossed
        for value, z, kind, payload in [*marks, (hi, z_end, "end", None)]:
            if inside is not None:
                if kind == "whole_end" and payload is inside:
                    cursor, inside = z, None
                    continue
                if value < _value(inside.z_end) - _TOL or kind in ("on", "off"):
                    name = getattr(payload, "name", "the end of the span")
                    raise ValueError(
                        f"{name} at {value} lies inside {inside.name} "
                        f"[{_value(inside.z_start)}, {_value(inside.z_end)}]. Overlapping "
                        "an extended component that can't be split is not supported."
                    )
            segments.append(FieldSpan(cursor, z, tuple(active)) if active else Gap(cursor, z))
            cursor = z
            match kind:
                case "on":
                    active.append(payload)
                case "off":
                    active.remove(payload)
                case "thin":
                    segments.append(Element(payload))
                case "whole":
                    if active:
                        raise ValueError(
                            f"{payload.name} lies inside the field of "
                            f"{[f.name for f in active]}. Overlapping an extended component "
                            "that can't be split is not supported."
                        )
                    segments.append(Element(payload))
                    inside = payload
        return segments

    # endregion --- geometry ---

    def forward(self, rays, solver=None, z_end=None, record: bool | Literal["light"] = False):
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
        r_interest: float | None = None,
    ) -> TransferMap:
        """The composed map from ``z_start`` to ``z_end``. Every component must have one."""
        total = None
        for seg in self.segments(z_start, z_end):
            match seg:
                case Gap(z0, z1):
                    m = drift_map(z1 - z0)
                case Element(c):
                    m = c.transfer_map(order=order, voltage=voltage, r_interest=r_interest)
                    if m is None:
                        raise ValueError(f"{c.name} is not a polynomial map; trace rays instead")
                case FieldSpan():
                    m = seg.transfer_map(order=order, voltage=voltage, r_interest=r_interest)
            total = m if total is None else m @ total
        return total

    def gaussian_image(
        self, z_object: torch.Tensor | float, voltage: float | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Paraxial image of the object plane ``z_object``, as ``(z_image, magnification)``.

        only accurate for field systems when the gaussian plane is past the end of the optical system.

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

    def focal_length(self, voltage: float | None = None, **kwargs) -> torch.Tensor:
        """Complex image-side focal length -1 / (e^{i theta} C). Unchanged by drifts before
        or after. Its modulus is |f| and its argument is the image rotation (mod pi).

        parameters
        ---------------
        order : int , default 1
            specifies the order of the (non)paraxial tracer
        voltage : float | None, default None
            specifies the voltage at which the transfermap is constructed
        r_interest : float | None , default None
            specifies the scale at which the transfermap is most accurate for non-linear components
        """
        first = min(float(c.z_start.detach()) for c in self.components.values())
        _, _, C, _ = self._complex_blocks(self.transfer_map(first, voltage=voltage, **kwargs))
        return -1.0 / C

    @staticmethod
    def _complex_blocks(m: TransferMap) -> tuple[torch.Tensor, ...]:
        """e^{i theta} (A, B, C, D) from the x-columns of a round system's map."""
        return tuple(
            torch.complex(m.entry(row, col), m.entry(row.replace("x", "y"), col))
            for row, col in (("x", "x"), ("x", "x'"), ("x'", "x"), ("x'", "x'"))
        )

    # endregion --- first-order optics ---

    #  region --- fields ---

    def fields(self, z_start: torch.Tensor | float, z_end: torch.Tensor | float):
        """The superposed axial fields from ``z_start`` to ``z_end``, as ``(z, phi, B)``.

        Field spans are sampled at 1000 points each and hold the summed profiles; gaps and thin
        components contribute zeros at their ends.
        """
        phi_space = []
        b_space = []
        z_space = []
        for seg in self.segments(z_start=z_start, z_end=z_end):
            match seg:
                case Gap(z0, z1):
                    z_space.append(torch.stack([z0, z1]).to(DTYPE))
                    phi_space.append(torch.zeros(2, dtype=DTYPE))
                    b_space.append(torch.zeros(2, dtype=DTYPE))
                case Element(c):
                    z_space.append(torch.stack([c.z_start, c.z_end]).to(DTYPE))
                    phi_space.append(torch.zeros(2, dtype=DTYPE))
                    b_space.append(torch.zeros(2, dtype=DTYPE))
                case FieldSpan(z0, z1):
                    z_local = torch.linspace(z0, z1, 1000, dtype=DTYPE)
                    deriv = seg.axial_derivative(0, z_local)
                    zeros = torch.zeros_like(z_local)
                    z_space.append(z_local)
                    phi_space.append(deriv.get("electrostatic", zeros))
                    b_space.append(deriv.get("magnetic", zeros))
        z_space = torch.cat(z_space)
        phi_space = torch.cat(phi_space)
        b_space = torch.cat(b_space)
        return z_space, phi_space, b_space

    # endregion --- fields ---

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
