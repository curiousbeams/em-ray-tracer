"""Thin (zero-length) elements, plus the double deflector built from two of them.

Drifts are not components: they are the gaps between component positions, filled in by
``OpticalSystem.segments``.
"""

from typing import Literal, Self

import torch

from em_ray_tracer.components.base import ComponentBase
from em_ray_tracer.constants import lorentz_factor
from em_ray_tracer.rays import XP, YP, Rays
from em_ray_tracer.transfer_map import (
    TransferMap,
    deflection_map,
    drift_map,
    quadrupole_map,
    thin_lens_map,
)
from em_ray_tracer.utils import as_parameter


class ThinLens(ComponentBase):
    """Ideal round thin lens.

    Parameterized by ``power`` = 1/f rather than f, so the lens can pass smoothly through zero
    strength and change sign during optimization. ``rotation`` is the image rotation a
    magnetic lens adds (its Larmor angle); 0 for electrostatic lenses.
    """

    config_fields = ("power", "rotation")

    def __init__(self, power: float, *, z: float, rotation: float = 0.0, **kwargs):
        super().__init__(z=z, **kwargs)
        self.power = as_parameter(power)
        self.rotation = as_parameter(rotation)

    @classmethod
    def from_focal_length(cls, focal_length: float, *, z: float, **kwargs) -> Self:
        return cls(power=1.0 / focal_length, z=z, **kwargs)

    @property
    def focal_length(self) -> torch.Tensor:
        return 1.0 / self.power

    def local_transfer_map(self, order: int = 1, voltage: float | None = None) -> TransferMap:
        # An ideal thin lens is exactly linear, so the map is the same at every order.
        return thin_lens_map(self.power, self.rotation)


class Quadrupole(ComponentBase):
    """Thin quadrupole (or stigmator): focusing ``power`` along ``orientation``, defocusing
    at right angles to it. Produces pure two-fold astigmatism (A1 / C12)."""

    config_fields = ("power", "orientation")

    def __init__(self, power: float, *, z: float, orientation: float = 0.0, **kwargs):
        super().__init__(z=z, **kwargs)
        self.power = as_parameter(power)
        self.orientation = as_parameter(orientation)

    def local_transfer_map(self, order: int = 1, voltage: float | None = None) -> TransferMap:
        return quadrupole_map(self.power, self.orientation)


DeflectorKind = Literal["magnetic", "electrostatic", "achromatic"]


def dispersion_coefficient(kind: DeflectorKind, voltage: float | None) -> float:
    """d(ln alpha)/d(delta) for a deflector, with delta = (E - E0)/E0 the kinetic-energy
    deviation. Zero when the kind is achromatic or the beam voltage is unknown."""
    if kind == "achromatic" or voltage is None:
        return 0.0
    gamma = lorentz_factor(voltage)
    if kind == "magnetic":  # alpha ~ 1/p
        return -gamma / (1 + gamma)
    if kind == "electrostatic":  # alpha ~ 1/(p v)
        return -(gamma**2 + 1) / (gamma * (gamma + 1))
    raise ValueError(f"Unknown deflector kind {kind!r}")


class Deflector(ComponentBase):
    """Thin dipole deflector: a constant slope kick ``deflection`` = (alpha_x, alpha_y).

    The kick sits in the homogeneous column of the transfer matrix. Off-energy rays are
    deflected by alpha (1 + k delta), with k from :func:`dispersion_coefficient`, which is a
    first-order entry M[x', delta]. It needs ``Rays.voltage``; without it the deflector acts
    as achromatic.
    """

    config_fields = ("deflection", "kind")

    def __init__(
        self,
        deflection: tuple[float, float],
        *,
        z: float,
        kind: DeflectorKind = "magnetic",
        **kwargs,
    ):
        super().__init__(z=z, **kwargs)
        self.deflection = as_parameter(deflection)
        self.kind = kind

    def local_transfer_map(self, order: int = 1, voltage: float | None = None) -> TransferMap:
        k = dispersion_coefficient(self.kind, voltage)
        return deflection_map(self.deflection[0], self.deflection[1], dispersion=k)


class DoubleDeflector(ComponentBase):
    """Two deflectors at ``z`` and ``z_2`` that tilt and shift the beam about a pivot plane.

    It is parameterized by what you control, not by the coil excitations. At the ``pivot``
    plane, every ray leaves displaced by ``beam_shift`` and tilted by ``tilt`` compared with no
    deflection. The two excitations follow from these (see :attr:`excitations`). The
    change of variables is linear and invertible, so nothing is lost. Its benefit is that a
    pure pivot, where the beam tilts but does not move at the pivot, is ``learn("tilt")`` with
    ``beam_shift`` left frozen, and it holds exactly at every optimizer step. Build one from
    measured excitations with :meth:`from_excitations`.

    An extended element: it occupies [z, z_2]. The ``pivot`` is usually below it (e.g. the
    specimen plane). The pivot condition assumes the space from ``z_2`` to the pivot is
    field-free, and it holds for the reference energy: an off-energy ray is deflected by
    a (1 + k delta), as for a single deflector.
    """

    config_fields = ("tilt", "beam_shift", "pivot", "z_2", "kind")

    def __init__(
        self,
        tilt: tuple[float, float],
        *,
        z: float,
        z_2: float,
        pivot: float,
        beam_shift: tuple[float, float] = (0.0, 0.0),
        kind: DeflectorKind = "magnetic",
        **kwargs,
    ):
        super().__init__(z=z, **kwargs)
        if z_2 <= z:
            raise ValueError(f"z_2 ({z_2}) must lie below z ({z})")
        self.tilt = as_parameter(tilt)
        self.beam_shift = as_parameter(beam_shift)
        self.pivot = as_parameter(pivot)
        self.z_2 = as_parameter(z_2)
        self.kind = kind

    @classmethod
    def from_excitations(
        cls,
        deflection: tuple[float, float],
        deflection_2: tuple[float, float],
        *,
        z: float,
        z_2: float,
        pivot: float,
        **kwargs,
    ) -> Self:
        """From the two deflection angles, e.g. calibrated coil excitations.

        The pair leaves slope a1 + a2 and, at the pivot, displacement
        a1 (z_2 - z) + (a1 + a2)(pivot - z_2).
        """
        a1 = torch.as_tensor(deflection, dtype=torch.float64)
        a2 = torch.as_tensor(deflection_2, dtype=torch.float64)
        tilt = a1 + a2
        beam_shift = a1 * (z_2 - z) + tilt * (pivot - z_2)
        return cls(tilt, z=z, z_2=z_2, pivot=pivot, beam_shift=beam_shift, **kwargs)

    @property
    def excitations(self) -> tuple[torch.Tensor, torch.Tensor]:
        """The two deflection angles (a1, a2). They invert :meth:`from_excitations`:
        a1 = (beam_shift - tilt (pivot - z_2)) / (z_2 - z) and a2 = tilt - a1."""
        a1 = (self.beam_shift - self.tilt * (self.pivot - self.z_2)) / (self.z_2 - self.z)
        return a1, self.tilt - a1

    @property
    def z_end(self) -> torch.Tensor:
        return self.z_2

    def local_transfer_map(self, order: int = 1, voltage: float | None = None) -> TransferMap:
        k = dispersion_coefficient(self.kind, voltage)
        a1, a2 = self.excitations
        first = deflection_map(a1[0], a1[1], dispersion=k)
        second = deflection_map(a2[0], a2[1], dispersion=k)
        return second @ drift_map(self.z_2 - self.z) @ first


class Biprism(ComponentBase):
    """Electrostatic biprism: a wire that kicks rays by ``deflection`` toward it (for a
    positive wire voltage), whichever side they pass.

    The kick direction depends on sign(x), so it is not a polynomial map: ``transfer_map``
    is ``None`` and a system containing one has no overall matrix. Rays that hit the wire
    (|u| < ``wire_radius``) are marked dead.
    """

    config_fields = ("deflection", "orientation", "wire_radius")

    def __init__(
        self,
        deflection: float,
        *,
        z: float,
        orientation: float = 0.0,
        wire_radius: float = 0.0,
        **kwargs,
    ):
        super().__init__(z=z, **kwargs)
        self.deflection = as_parameter(deflection)
        self.orientation = as_parameter(orientation)
        self.wire_radius = as_parameter(wire_radius)

    def local_transfer_map(self, order: int = 1, voltage: float | None = None) -> None:
        return None

    def apply(self, rays: Rays, order: int = 1) -> Rays:
        # u: coordinate perpendicular to the wire, measured from the (possibly shifted) wire
        c, s = torch.cos(self.orientation), torch.sin(self.orientation)
        u = (rays.x - self.shift[0]) * c + (rays.y - self.shift[1]) * s
        kick = -self.deflection * torch.sign(u)
        state = rays.state.clone()
        state[..., XP] = state[..., XP] + kick * c
        state[..., YP] = state[..., YP] + kick * s
        alive = rays.alive & (u.abs() >= self.wire_radius)
        return rays.replace(state=state, alive=alive, z=self.z_end)


class Aperture(ComponentBase):
    """Circular aperture of ``radius``. Rays are never removed: a hard aperture clears
    ``Rays.alive`` for rays outside it, so the bundle keeps a static shape.

    With ``edge_width`` > 0 the edge is soft. Rays within the band
    ``radius +- edge_width / 2`` keep a fraction of their ``weight``: a smoothstep that goes
    from 1 inside to 0 outside, linear in r^2 (i.e. in area). That makes the transmitted
    weight, and every weighted quantity downstream (spot size, current, aberration fits), a
    smooth function of ``radius``, so ``learn("radius")`` gets useful gradients. Rays beyond
    the band carry zero weight and are marked dead.

    The gradient only comes from rays inside the band. Pick ``edge_width`` wide enough to
    hold several rays, e.g. a few percent of the beam radius with ``Rays.from_disk``. A hard
    edge (the default) is exact and is what a simulation should use once the radius is fixed.

    The state is unchanged, so the transfer map is the identity and a system with apertures
    still has an overall matrix.
    """

    config_fields = ("radius", "edge_width")

    def __init__(self, radius: float, *, z: float, edge_width: float = 0.0, **kwargs):
        super().__init__(z=z, **kwargs)
        if edge_width < 0 or (edge_width > 0 and edge_width >= 2 * radius):
            raise ValueError(f"edge_width must be in [0, 2 radius), got {edge_width}")
        self.radius = as_parameter(radius)
        self.edge_width = float(edge_width)

    def transmission(self, rays: Rays) -> torch.Tensor:
        """Per-ray transmitted fraction (0 or 1 for a hard edge)."""
        r2 = (rays.x - self.shift[0]) ** 2 + (rays.y - self.shift[1]) ** 2
        if self.edge_width == 0:
            return (r2 <= self.radius**2).to(rays.weight.dtype)
        outer2 = (self.radius + self.edge_width / 2) ** 2
        band = 2 * self.radius * self.edge_width  # outer^2 - inner^2
        t = ((outer2 - r2) / band).clamp(0, 1)
        return t * t * (3 - 2 * t)

    def local_transfer_map(self, order: int = 1, voltage: float | None = None) -> TransferMap:
        return TransferMap.identity(dtype=self.z.dtype, device=self.z.device)

    def apply(self, rays: Rays, order: int = 1) -> Rays:
        t = self.transmission(rays)
        alive = rays.alive & (t > 0)
        if self.edge_width == 0:
            return rays.replace(alive=alive, z=self.z_end)
        return rays.replace(weight=rays.weight * t, alive=alive, z=self.z_end)


class Plane(ComponentBase):
    """A named observation plane. It does nothing to the rays; the solver records the bundle
    here under ``trace.planes[name]``. Make ``z`` learnable to optimize where to look."""

    def __init__(self, *, z: float, name: str | None = None):
        super().__init__(z=z, name=name)

    def local_transfer_map(self, order: int = 1, voltage: float | None = None) -> TransferMap:
        return TransferMap.identity(dtype=self.z.dtype, device=self.z.device)

    def apply(self, rays: Rays, order: int = 1) -> Rays:
        return rays.replace(z=self.z_end)
