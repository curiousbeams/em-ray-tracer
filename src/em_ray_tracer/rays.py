"""The ray bundle: a batch of electron trajectories at one axial plane."""

import dataclasses
import math
from dataclasses import dataclass
from typing import Self

import torch

from em_ray_tracer.constants import DTYPE

# Column layout of ``Rays.state``. Transfer maps act on the homogeneous vector
# [x, x', y, y', delta, 1]; the trailing 1 is added inside ``TransferMap.apply`` and never
# stored, so it cannot be overwritten.
X, XP, Y, YP, DELTA = range(5)
STATE_DIM = 5
MAP_DIM = STATE_DIM + 1
STATE_NAMES = ("x", "x'", "y", "y'", "delta")


@dataclass
class Rays:
    """A batch of rays at a common axial position ``z``.

    Attributes
    ----------
    state : torch.Tensor
        ``(..., N, 5)`` tensor of ``[x, x', y, y', delta]``: transverse positions [m],
        slopes dx/dz and dy/dz, and the fractional energy deviation
        ``delta = (E - E0) / E0`` relative to the reference energy set by ``voltage``.
    z : torch.Tensor
        Axial position of the plane [m]. May carry gradients (e.g. a learnable plane).
    alive : torch.Tensor
        ``(..., N)`` bool mask. Apertures clear it instead of dropping rays, so every tensor
        keeps a static shape and the whole bundle stays one batched operation.
    weight : torch.Tensor
        ``(..., N)`` per-ray weight (intensity), used by spot-size objectives and fits.
    voltage : float | None
        Reference accelerating voltage [V] of the beam. Components whose action depends on
        the beam energy (field lenses, dispersive deflectors) read it from here.
    """

    state: torch.Tensor
    z: torch.Tensor | float = 0.0
    alive: torch.Tensor | None = None
    weight: torch.Tensor | None = None
    voltage: float | None = None

    def __post_init__(self):
        state = torch.as_tensor(self.state)
        if not state.is_floating_point():
            state = state.to(DTYPE)
        if state.ndim < 2 or state.shape[-1] != STATE_DIM:
            raise ValueError(
                f"state must have shape (..., N, {STATE_DIM}), got {tuple(state.shape)}"
            )
        self.state = state
        if isinstance(self.z, torch.Tensor):
            self.z = self.z.to(dtype=state.dtype, device=state.device)
        else:
            self.z = torch.tensor(float(self.z), dtype=state.dtype, device=state.device)
        batch_shape = state.shape[:-1]
        if self.alive is None:
            self.alive = torch.ones(batch_shape, dtype=torch.bool, device=state.device)
        if self.weight is None:
            self.weight = torch.ones(batch_shape, dtype=state.dtype, device=state.device)

    # region --- entry points ---

    @classmethod
    def from_state(
        cls,
        state: torch.Tensor,
        z: float | torch.Tensor = 0.0,
        voltage: float | None = None,
    ) -> Self:
        """Rays from an explicit ``(..., N, 4)`` or ``(..., N, 5)`` state tensor.

        A 4-column state ``[x, x', y, y']`` is padded with ``delta = 0``.
        """
        state = torch.as_tensor(state, dtype=DTYPE)
        if state.shape[-1] == STATE_DIM - 1:
            state = torch.cat([state, torch.zeros_like(state[..., :1])], dim=-1)
        return cls(state=state, z=z, voltage=voltage)

    @classmethod
    def from_fan(
        cls,
        semiangle: float,
        num_rays: int,
        x0: float = 0.0,
        z: float = 0.0,
        voltage: float | None = None,
    ) -> Self:
        """A meridional (x-z) fan of rays from one point, slopes spanning +-tan(semiangle).

        The 2D ray diagram is just this bundle with y = y' = 0; there is no separate 2D code
        path.
        """
        slopes = torch.linspace(-math.tan(semiangle), math.tan(semiangle), num_rays, dtype=DTYPE)
        state = torch.zeros(num_rays, STATE_DIM, dtype=DTYPE)
        state[:, X] = x0
        state[:, XP] = slopes
        return cls(state=state, z=z, voltage=voltage)

    @classmethod
    def from_parallel(
        cls,
        radius: float,
        num_rays: int,
        z: float = 0.0,
        voltage: float | None = None,
    ) -> Self:
        """A meridional bundle of rays parallel to the axis, x spanning +-radius."""
        state = torch.zeros(num_rays, STATE_DIM, dtype=DTYPE)
        state[:, X] = torch.linspace(-radius, radius, num_rays, dtype=DTYPE)
        return cls(state=state, z=z, voltage=voltage)

    @classmethod
    def from_cone(
        cls,
        semiangle: float,
        n_azimuthal: int = 16,
        n_shells: int = 3,
        origin: tuple[float, float] = (0.0, 0.0),
        z: float = 0.0,
        voltage: float | None = None,
    ) -> Self:
        """Concentric hollow cones of rays from one point, the outermost at ``semiangle``.

        Shells rather than a filled cone because spherical aberration scales with the cube of
        the entry angle; coloring by shell is what makes it visible. The rays are *not*
        uniform in solid angle: every ray has weight 1, but outer shells stand for more of the
        beam. Use :meth:`from_disk` for anything weighted by current.
        """
        shells = torch.arange(1, n_shells + 1, dtype=DTYPE) / n_shells * math.tan(semiangle)
        phi = torch.arange(n_azimuthal, dtype=DTYPE) * (2 * math.pi / n_azimuthal)
        slope, azimuth = torch.meshgrid(shells, phi, indexing="ij")
        n = slope.numel()
        state = torch.zeros(n, STATE_DIM, dtype=DTYPE)
        state[:, X] = origin[0]
        state[:, Y] = origin[1]
        state[:, XP] = (slope * torch.cos(azimuth)).reshape(-1)
        state[:, YP] = (slope * torch.sin(azimuth)).reshape(-1)
        return cls(state=state, z=z, voltage=voltage)

    @classmethod
    def from_disk(
        cls,
        semiangle: float,
        num_rays: int = 1000,
        origin: tuple[float, float] = (0.0, 0.0),
        z: float = 0.0,
        voltage: float | None = None,
    ) -> Self:
        """A filled cone from one point, sampled uniformly in solid angle (sunflower spiral).

        Every ray carries an equal share of the current, so summed weights are currents
        (``Trace.transmission``) and weighted spot sizes are intensity-weighted. Each ray also
        sits at a distinct radius, so the current through an aperture grows smoothly with its
        radius, rather than in jumps as it crosses whole shells of :meth:`from_cone`.
        """
        k = torch.arange(num_rays, dtype=DTYPE)
        slope = torch.sqrt((k + 0.5) / num_rays) * math.tan(semiangle)
        azimuth = k * math.pi * (3 - math.sqrt(5))  # golden angle
        state = torch.zeros(num_rays, STATE_DIM, dtype=DTYPE)
        state[:, X] = origin[0]
        state[:, Y] = origin[1]
        state[:, XP] = slope * torch.cos(azimuth)
        state[:, YP] = slope * torch.sin(azimuth)
        return cls(state=state, z=z, voltage=voltage)

    @classmethod
    def from_object_points(
        cls,
        points: torch.Tensor,
        semiangle: float,
        n_azimuthal: int = 16,
        n_shells: int = 3,
        z: float = 0.0,
        voltage: float | None = None,
    ) -> Self:
        """A cone (see :meth:`from_cone`) from each of ``points`` (P, 2), concatenated.

        This is the bundle an off-axial aberration fit needs: it varies both the object
        position w and the slope omega.
        """
        points = torch.as_tensor(points, dtype=DTYPE)
        cones = [
            cls.from_cone(semiangle, n_azimuthal, n_shells, origin=(float(p[0]), float(p[1])))
            for p in points
        ]
        state = torch.cat([c.state for c in cones], dim=0)
        return cls(state=state, z=z, voltage=voltage)

    @classmethod
    def principal(cls, z: float = 0.0, voltage: float | None = None) -> Self:
        """The four unit rays g_x, h_x, g_y, h_y (unit position or unit slope in x and y).

        Traced through a linear system, their final states are the columns of its matrix.
        """
        state = torch.zeros(4, STATE_DIM, dtype=DTYPE)
        state[:, :4] = torch.eye(4, dtype=DTYPE)
        return cls(state=state, z=z, voltage=voltage)

    # endregion --- entry points ---

    # region --- views ---

    @property
    def num_rays(self) -> int:
        return self.state.shape[-2]

    @property
    def x(self) -> torch.Tensor:
        return self.state[..., X]

    @property
    def xp(self) -> torch.Tensor:
        return self.state[..., XP]

    @property
    def y(self) -> torch.Tensor:
        return self.state[..., Y]

    @property
    def yp(self) -> torch.Tensor:
        return self.state[..., YP]

    @property
    def delta(self) -> torch.Tensor:
        return self.state[..., DELTA]

    @property
    def w(self) -> torch.Tensor:
        """Complex transverse position x + iy."""
        return torch.complex(self.x, self.y)

    @property
    def slope(self) -> torch.Tensor:
        """Complex slope omega = x' + iy'."""
        return torch.complex(self.xp, self.yp)

    # endregion --- views ---

    # region --- methods ---

    def replace(self, **changes) -> Self:
        """A copy with the given fields replaced (tensors are shared, not cloned).

        Delegates to ``dataclasses.replace``, which re-runs ``__init__``/``__post_init__``.
        """
        return dataclasses.replace(self, **changes)

    def to(self, device: str | torch.device) -> Self:
        return self.replace(
            state=self.state.to(device),
            z=self.z.to(device),
            alive=self.alive.to(device),
            weight=self.weight.to(device),
        )

    def detach(self) -> Self:
        return self.replace(
            state=self.state.detach(),
            z=self.z.detach(),
            weight=self.weight.detach(),
        )

    def clone(self) -> Self:
        return self.replace(
            state=self.state.clone(),
            z=self.z.clone(),
            alive=self.alive.clone(),
            weight=self.weight.clone(),
        )

    # endregion --- methods ---
