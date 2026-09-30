"""Solver interface and the trace result it returns."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Self

import torch

from em_ray_tracer.components.base import ComponentBase
from em_ray_tracer.components.thin import Plane
from em_ray_tracer.rays import Rays
from em_ray_tracer.solvers.params import SolverParams, SolverParamsType
from em_ray_tracer.transfer_map import TransferMap, drift_map

if TYPE_CHECKING:
    from em_ray_tracer.system import OpticalSystem


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

    @classmethod
    def from_snapshots(cls, snapshots: list[Rays]) -> Self:
        return cls(
            z=torch.stack([r.z for r in snapshots]),
            state=torch.stack([r.state for r in snapshots]),
            alive=torch.stack([r.alive for r in snapshots]),
        )


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
    def propagate(self, component: ComponentBase, rays: Rays) -> tuple[Rays, TransferMap | None]:
        """Cross ``component``; return the rays at its exit and its map (``None`` if it has
        none)."""

    def trace(
        self,
        system: "OpticalSystem",
        rays: Rays,
        *,
        z_end: torch.Tensor | float | None = None,
        record: bool = False,
    ) -> Trace:
        """Trace ``rays`` from ``rays.z`` to ``z_end`` (default: the last component)."""
        from em_ray_tracer.system import Element, Gap

        current = rays
        snapshots = [rays] if record else []
        planes: dict[str, Rays] = {}
        total: TransferMap | None = TransferMap.identity(rays.state.dtype, rays.state.device)

        for seg in system.segments(rays.z, z_end):
            match seg:
                case Gap(z0, z1):
                    m = drift_map(z1 - z0)
                    current = current.replace(state=m.apply(current.state), z=z1)
                case Element(component):
                    current, m = self.propagate(component, current)
                    if isinstance(component, Plane):
                        planes[component.name] = current
            if total is not None and m is not None and not (m.higher or total.higher):
                total = m @ total
            else:
                total = None
            if record:
                snapshots.append(current)

        return Trace(
            initial=rays,
            final=current,
            planes=planes,
            transfer_map=total,
            trajectory=Trajectory.from_snapshots(snapshots) if record else None,
        )
