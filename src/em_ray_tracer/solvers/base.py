"""Solver interface and the trace result it returns."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Self

import torch

from em_ray_tracer.components.base import ComponentBase
from em_ray_tracer.components.thin import Plane
from em_ray_tracer.constants import DTYPE
from em_ray_tracer.rays import Rays, X, Y
from em_ray_tracer.solvers.params import SolverParams, SolverParamsType
from em_ray_tracer.transfer_map import TransferMap, drift_map

if TYPE_CHECKING:
    from em_ray_tracer.system import FieldSpan, OpticalSystem


@dataclass
class Trajectory:
    """Snapshots of the bundle along z: ``z`` (K,), ``state`` (K, ..., N, 5),
    ``alive`` (K, ..., N).

    With the transfer-matrix solver the snapshots are at segment boundaries and rays are
    straight lines between them, so linear interpolation is exact. ODE solvers record every
    step.
    """

    z: torch.Tensor
    state: torch.Tensor
    alive: torch.Tensor
    _polc: tuple[torch.Tensor, torch.Tensor] | None = None

    @classmethod
    def from_snapshots(cls, snapshots: list[Rays]) -> Self:
        return cls(
            z=torch.stack([r.z for r in snapshots]),
            state=torch.stack([r.state for r in snapshots]),
            alive=torch.stack([r.alive for r in snapshots]),
        )

    def plane_of_least_confusion(self, domain: tuple[float, float] | None = None):
        if self._polc is None or domain is not None:
            from em_ray_tracer.interpolation import b_spline

            domain = list(domain)
            if domain is None:
                domain = (self.z[0], self.z[-1])
            if domain[0] is None:
                domain[0] = self.z[0]
            if domain[1] is None:
                domain[1] = self.z[-1]

            x, y, z = self.state[..., X], self.state[..., Y], self.z

            dz_set = z[1:] - z[:-1]
            asc = torch.cat([torch.tensor([True], device=dz_set.device), (dz_set > 0)])
            dz_set = dz_set[dz_set != 0]
            dz = torch.min(dz_set)
            x, y, z = x[asc], y[asc], z[asc]

            keep = (z > domain[0]) & (z < domain[1])
            inner = torch.argwhere(keep)
            keep[inner[0] - 1] = True
            keep[inner[-1] + 1] = True
            x, y, z = x[keep], y[keep], z[keep]

            x_spline = b_spline(z, x, order=1, window_size=2)
            y_spline = b_spline(z, y, order=1, window_size=2)
            n_points = (domain[1] - domain[0]) / dz + 1
            z_domain = torch.linspace(domain[0], domain[1], int(n_points), dtype=DTYPE)

            r2 = torch.var(x_spline(0, z_domain), dim=-1) + torch.var(
                y_spline(0, z_domain), dim=-1
            )
            polc_arg = torch.argmin(r2)

            # extra refinement
            z_domain2 = torch.linspace(
                z_domain[polc_arg - 1], z_domain[polc_arg + 1], int(n_points), dtype=DTYPE
            )
            r2 = torch.var(x_spline(0, z_domain2), dim=-1) + torch.var(
                y_spline(0, z_domain2), dim=-1
            )
            polc_arg = torch.argmin(r2)

            self._polc = z_domain2[polc_arg], torch.sqrt(r2[polc_arg])
        return self._polc


@dataclass
class Trace:
    """Result of tracing a bundle through a system.

    Attributes
    ----------
    initial, final : Rays
        The bundle at the start and at the end of the span.
    planes : dict[str, Rays]
        The bundle at every ``Plane`` component, keyed by name.
    transfer_map : TransferMap | None
        The composed first-order map over the span, or ``None`` if some element has no map
        (a biprism, or a field lens under an ODE solver).
    trajectory : Trajectory | None
        Snapshots along z, when traced with ``record=True``.
    """

    initial: Rays
    final: Rays
    planes: dict[str, Rays]
    transfer_map: TransferMap | None = None
    trajectory: Trajectory | None = None

    def __getitem__(self, plane: str) -> Rays:
        return self.planes[plane]

    def transmission(self, plane: str | None = None) -> torch.Tensor:
        """Fraction of the input weight that reaches ``plane`` (default: the end) alive.

        With rays that sample the beam uniformly (``Rays.from_disk``), this is the probe
        current as a fraction of the source current. It is differentiable through soft-edged
        apertures.
        """
        out = self.final if plane is None else self.planes[plane]
        total = (self.initial.weight * self.initial.alive).sum(-1)
        return (out.weight * out.alive).sum(-1) / total


class SolverBase(ABC):
    """Propagates a ray bundle through an ``OpticalSystem``.

    The loop over the system's segments lives here. Gaps are drifts for every solver, which is
    exact at every order in slope coordinates. Subclasses decide only how to cross an element,
    in :meth:`propagate`.
    """

    @classmethod
    def from_params(cls, params: "SolverParamsType | dict | SolverBase") -> "SolverBase":
        """Build a solver from its config (see ``SolverParams``)."""
        from em_ray_tracer.solvers.ode import ODESolver
        from em_ray_tracer.solvers.transfer_matrix import TransferMatrixSolver

        if isinstance(params, SolverBase):
            return params
        if isinstance(params, dict):
            params = SolverParams.parse_dict(params)
        match params:
            case SolverParams.TransferMatrix(order=order):
                return TransferMatrixSolver(order=order)
            case SolverParams.Paraxial(integrator=integrator):
                return ODESolver.paraxial(integrator=integrator)
            case SolverParams.NonParaxial(expansion_order=order, integrator=integrator):
                return ODESolver.nonparaxial(expansion_order=order, integrator=integrator)
            case SolverParams.Lorentz():
                raise NotImplementedError("The time-domain Lorentz solver is not implemented yet")
            case _:
                raise TypeError(f"Unknown solver params {params!r}")

    @abstractmethod
    def propagate(
        self, component: ComponentBase, rays: Rays, snapshots: list[Rays] | None = None
    ) -> tuple[Rays, TransferMap | None, list[Rays] | None]:
        """Cross ``component``; return the rays at its exit and its map (``None`` if it has
        none)."""

    @abstractmethod
    def cross(
        self, span: "FieldSpan", rays: Rays, snapshots: list[Rays] | None = None
    ) -> tuple[Rays, TransferMap | None, list[Rays] | None]:
        """Cross ``span``, the superposed fields of one or more field components; return the
        rays at ``span.z1`` and the span's map (``None`` if the solver makes none)."""

    def trace(
        self,
        system: "OpticalSystem",
        rays: Rays,
        *,
        z_end: torch.Tensor | float | None = None,
        record: bool | Literal["light"] = False,
    ) -> Trace:
        """Trace ``rays`` from ``rays.z`` to ``z_end`` (default: the last component).
        Use record = "light" if you only want the edges of elements tracked"""
        from em_ray_tracer.system import Element, FieldSpan, Gap

        current = rays
        snapshots = [rays] if record else []
        planes: dict[str, Rays] = {}
        total: TransferMap | None = TransferMap.identity(rays.state.dtype, rays.state.device)

        for seg in system.segments(rays.z, z_end):
            match seg:
                case Gap(z0, z1):
                    m = drift_map(z1 - z0)
                    current = current.replace(state=m.apply(current.state), z=z1)
                    if record:
                        snapshots.append(current)
                case Element(component):
                    if record is True:
                        current, m, snapshots = self.propagate(
                            component, current, snapshots=snapshots
                        )
                    elif record == "light":
                        current, m, _ = self.propagate(component, current)
                        snapshots.append(current)
                    else:
                        current, m, _ = self.propagate(component, current)
                    if isinstance(component, Plane):
                        planes[component.name] = current
                case FieldSpan():
                    if record is True:
                        current, m, snapshots = self.cross(seg, current, snapshots=snapshots)
                    elif record == "light":
                        current, m, _ = self.cross(seg, current)
                        snapshots.append(current)
                    else:
                        current, m, _ = self.cross(seg, current)
            if total is not None and m is not None and not (m.higher or total.higher):
                total = m @ total
            else:
                total = None

        return Trace(
            initial=rays,
            final=current,
            planes=planes,
            transfer_map=total,
            trajectory=Trajectory.from_snapshots(snapshots) if record else None,
        )
