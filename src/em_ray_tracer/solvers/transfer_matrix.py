"""Transfer-matrix (polynomial map) tracing."""

from em_ray_tracer.components.base import ComponentBase
from em_ray_tracer.rays import Rays
from em_ray_tracer.solvers.base import SolverBase
from em_ray_tracer.system import FieldSpan
from em_ray_tracer.transfer_map import TransferMap


class TransferMatrixSolver(SolverBase):
    """Propagate element by element with each component's transfer map of ``order``.

    For higher order, consider specifying r_interest which contains the axial distance of interest

    Rays are pushed through each map in turn rather than through a single composed system
    matrix. That way non-polynomial elements (apertures, biprisms) interleave naturally, and
    higher-order maps work before their truncated composition is implemented. A field lens
    contributes its thick-lens map, which is a stub for now (see ``FieldLens``).
    """

    def __init__(self, order: int = 1, r_interest: float | None = None):
        if not isinstance(order, int):
            raise TypeError(f"order must be of type int, is now {type(order)}")
        if order < 1:
            raise ValueError("order must be >= 1")
        self.order = order
        self.r_interest = r_interest

    def propagate(
        self, component: ComponentBase, rays: Rays, snapshots: list[Rays] | None = None
    ) -> tuple[Rays, TransferMap | None, list[Rays] | None]:
        m = component.transfer_map(
            order=self.order, voltage=rays.voltage, r_interest=self.r_interest
        )
        current = component.apply(rays, order=self.order, r_interest=self.r_interest)
        if snapshots is not None:
            snapshots.append(current)
        return current, m, snapshots

    def cross(
        self, span: FieldSpan, rays: Rays, snapshots: list[Rays] | None = None
    ) -> tuple[Rays, TransferMap | None, list[Rays] | None]:
        """Cross the superposed fields of ``span`` with its map (``FieldSpan.transfer_map``)."""
        m = span.transfer_map(order=self.order, voltage=rays.voltage, r_interest=self.r_interest)
        current = rays.replace(state=m.apply(rays.state), z=span.z1)
        if snapshots is not None:
            snapshots.append(current)
        return current, m, snapshots

    def __repr__(self) -> str:
        return f"TransferMatrixSolver(order={self.order})"
