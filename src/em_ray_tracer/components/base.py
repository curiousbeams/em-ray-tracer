"""Base class for everything placed along the optic axis."""

from abc import ABC, abstractmethod
from typing import Any, ClassVar, Self

import torch
from torch import nn

from em_ray_tracer.rays import Rays
from em_ray_tracer.transfer_map import TransferMap, translation_map
from em_ray_tracer.utils import as_parameter


class ComponentBase(nn.Module, ABC):
    """An element of the optical system.

    Every physical quantity is an ``nn.Parameter`` that starts frozen; ``learn("power")``
    makes it optimizable. Position ``z`` is a parameter too: drifts are the gaps between component positions, so moving a lens is
    differentiable instead of snapping to a z-grid.

    The constructor takes the canonical representation and is the one to call directly.
    Classmethods (``ThinLens.from_focal_length``, ``DoubleDeflector.from_excitations``, ...) convert
    from other representations.

    Subclasses implement :meth:`local_transfer_map`, the map in the element's own aligned frame.
    Misalignment (``shift``, ``tilt``) is applied on top by conjugation, T(s) M T(s)^-1, so an
    element is written once and works aligned or not. Elements that are not polynomial maps
    (apertures, biprisms) also override :meth:`apply`.

    Parameters
    ----------
    z : float
        Axial position [m]. Extended elements are centred on it.
    name : str | None
        Unique name within the system; generated from the class name if omitted.
    shift : tuple[float, float]
        Transverse displacement (dx, dy) of the element's axis [m].
    tilt : tuple[float, float]
        Tilt (tx, ty) of the element's axis, as slopes.
    """

    #: Names of constructor arguments (beyond z/name/shift/tilt) that ``config()`` records.
    config_fields: ClassVar[tuple[str, ...]] = ()
    _registry: ClassVar[dict[str, type["ComponentBase"]]] = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        ComponentBase._registry[cls.__name__] = cls

    def __init__(
        self,
        *,
        z: float,
        name: str | None = None,
        shift: tuple[float, float] = (0.0, 0.0),
        tilt: tuple[float, float] = (0.0, 0.0),
    ):
        super().__init__()
        self.name = name
        self.z = as_parameter(z)
        self.shift = as_parameter(shift)
        self.tilt = as_parameter(tilt)

    # region --- geometry ---

    @property
    def z_start(self) -> torch.Tensor:
        return self.z

    @property
    def z_end(self) -> torch.Tensor:
        return self.z

    @property
    def length(self) -> torch.Tensor:
        return self.z_end - self.z_start

    # endregion --- geometry ---

    # region --- learnable parameters ---

    def learn(self, *names: str) -> Self:
        """Unfreeze the named parameters (e.g. ``lens.learn("power", "z")``)."""
        for name in names:
            self._named_parameter(name).requires_grad_(True)
        return self

    def freeze(self, *names: str) -> Self:
        """Freeze the named parameters, or all of them if none are named."""
        params = [self._named_parameter(n) for n in names] if names else self.parameters()
        for p in params:
            p.requires_grad_(False)
        return self

    @property
    def learnable(self) -> dict[str, nn.Parameter]:
        return {n: p for n, p in self.named_parameters() if p.requires_grad}

    def _named_parameter(self, name: str) -> nn.Parameter:
        params = dict(self.named_parameters())
        if name not in params:
            raise KeyError(f"{type(self).__name__} has no parameter {name!r}; has {list(params)}")
        return params[name]

    # endregion --- learnable parameters ---

    # region --- optics ---

    @abstractmethod
    def local_transfer_map(
        self, order: int = 1, voltage: float | None = None
    ) -> TransferMap | None:
        """The map across the element in its own aligned frame, or ``None`` if the element
        is not a polynomial map of the ray state (e.g. a biprism)."""

    def transfer_map(self, order: int = 1, voltage: float | None = None) -> TransferMap | None:
        """The map in the lab frame: :meth:`local_transfer_map` conjugated by misalignment."""
        local = self.local_transfer_map(order=order, voltage=voltage)
        if local is None or not self.is_misaligned:
            return local
        sx, sy = self.shift
        tx, ty = self.tilt
        rel_in = self.z_start - self.z
        rel_out = self.z_end - self.z
        to_local = translation_map(-(sx + tx * rel_in), -(sy + ty * rel_in), -tx, -ty)
        to_lab = translation_map(sx + tx * rel_out, sy + ty * rel_out, tx, ty)
        return to_lab @ local @ to_local

    @property
    def is_misaligned(self) -> bool:
        if self.shift.requires_grad or self.tilt.requires_grad:
            return True
        return bool(torch.any(self.shift != 0) or torch.any(self.tilt != 0))

    def apply(self, rays: Rays, order: int = 1) -> Rays:
        """Propagate ``rays`` from ``z_start`` to ``z_end`` through this element."""
        m = self.transfer_map(order=order, voltage=rays.voltage)
        if m is None:
            raise NotImplementedError(f"{type(self).__name__} must implement apply()")
        return rays.replace(state=m.apply(rays.state), z=self.z_end)

    # endregion --- optics ---

    # region --- config ---

    def config(self) -> dict[str, Any]:
        """A plain-dict description that ``OpticalSystem.from_config`` can rebuild."""

        def plain(value):
            return value.detach().cpu().tolist() if isinstance(value, torch.Tensor) else value

        out = {"type": type(self).__name__, "name": self.name, "z": plain(self.z)}
        if self.is_misaligned:
            out |= {"shift": plain(self.shift), "tilt": plain(self.tilt)}
        return out | {f: plain(getattr(self, f)) for f in self.config_fields}

    @staticmethod
    def from_config(config: dict[str, Any]) -> "ComponentBase":
        config = dict(config)
        kind = config.pop("type")
        if kind not in ComponentBase._registry:
            raise ValueError(f"Unknown component type {kind!r}")
        return ComponentBase._registry[kind](**config)

    # endregion --- config ---

    def extra_repr(self) -> str:
        fields = ", ".join(
            f"{f}={getattr(self, f).tolist()}"
            for f in self.config_fields
            if isinstance(getattr(self, f), torch.Tensor)
        )
        learn = f", learn={list(self.learnable)}" if self.learnable else ""
        return f"name={self.name!r}, z={self.z.item():.6g}, {fields}{learn}"
