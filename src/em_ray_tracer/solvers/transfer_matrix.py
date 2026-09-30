"""Transfer-matrix (polynomial map) tracing."""

from em_ray_tracer.components.base import ComponentBase
from em_ray_tracer.rays import Rays
from em_ray_tracer.solvers.base import SolverBase
from em_ray_tracer.transfer_map import TransferMap


class TransferMatrixSolver(SolverBase):
    """Propagate element by element with each component's transfer map of ``order``.

    Rays are pushed through each map in turn rather than through a single composed system
    matrix. That way non-polynomial elements (apertures, biprisms) interleave naturally, and
    higher-order maps work before their truncated composition is implemented. A field lens
    contributes its thick-lens map, which is a stub for now (see ``FieldLens``).
    """

    def __init__(self, order: int = 1):
        if order < 1:
            raise ValueError("order must be >= 1")
        self.order = order

    def propagate(self, component: ComponentBase, rays: Rays) -> tuple[Rays, TransferMap | None]:
        m = component.transfer_map(order=self.order, voltage=rays.voltage)
        return component.apply(rays, order=self.order), m

    def __repr__(self) -> str:
        return f"TransferMatrixSolver(order={self.order})"
