"""The top-level class tying a system, a solver and a ray bundle together.

Build it from its parts with ``from_models``, then ``trace``, ``fit_aberrations`` or
``optimize``.
"""

from collections.abc import Sequence
from typing import Self

import torch

from em_ray_tracer.aberrations.basis import AberrationBasis
from em_ray_tracer.aberrations.coefficients import AberrationCoefficients
from em_ray_tracer.optimize.objectives import Evaluation, Objective
from em_ray_tracer.optimize.params import (
    OptimizerParams,
    OptimizerParamsType,
    SchedulerParamsType,
)
from em_ray_tracer.rays import Rays
from em_ray_tracer.solvers.base import SolverBase, Trace
from em_ray_tracer.solvers.params import SolverParams, SolverParamsType
from em_ray_tracer.system import OpticalSystem


class RayTracer:
    """A system, a solver, and the input ray bundle.

    Swapping the solver is a one-argument change:
    ``RayTracer.from_models(system, rays, solver=SolverParams.Paraxial())``.
    """

    def __init__(
        self,
        *,
        system: OpticalSystem,
        rays: Rays,
        solver: SolverBase,
        z_end: float | None = None,
        verbose: bool = True,
    ):
        self.system = system
        self.rays = rays
        self.solver = solver
        self.z_end = z_end
        self.verbose = verbose
        self.losses: list[float] = []
        self.lrs: list[float] = []

    @classmethod
    def from_models(
        cls,
        system: OpticalSystem,
        rays: Rays,
        solver: SolverBase | SolverParamsType | dict | None = None,
        z_end: float | None = None,
        verbose: bool = True,
    ) -> Self:
        """``solver`` defaults to the first-order transfer-matrix solver."""
        solver = SolverBase.from_params(solver or SolverParams.TransferMatrix())
        return cls(system=system, rays=rays, solver=solver, z_end=z_end, verbose=verbose)

    # region --- forward ---

    def trace(self, record: bool = False) -> Trace:
        return self.solver.trace(self.system, self.rays, z_end=self.z_end, record=record)

    def gaussian_image(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Paraxial ``(z_image, magnification)`` of the plane the rays start from."""
        return self.system.gaussian_image(self.rays.z, voltage=self.rays.voltage)

    def fit_aberrations(
        self,
        basis: AberrationBasis | None = None,
        plane: str | None = None,
        trace: Trace | None = None,
    ) -> AberrationCoefficients:
        """Fit ``basis`` (default: axial to third order) to the rays at ``plane``."""
        trace = trace or self.trace()
        return AberrationCoefficients.from_trace(trace, basis or AberrationBasis.axial(), plane)

    # endregion --- forward ---

    # region --- optimization ---

    def loss(self, objectives: Sequence[Objective], trace: Trace | None = None) -> torch.Tensor:
        """Summed objectives. The rays are traced only if some objective needs them (or
        ``trace`` is given), and shared quantities are computed once; see ``Evaluation``."""
        evaluation = Evaluation(self, trace)
        return sum(objective(evaluation) for objective in objectives)

    def optimize(
        self,
        objectives: Objective | Sequence[Objective],
        num_iters: int,
        optimizer_params: OptimizerParamsType | dict | None = None,
        scheduler_params: SchedulerParamsType | dict | None = None,
        reset: bool = False,
    ) -> Self:
        """Minimize the summed ``objectives`` over the system's learnable parameters.

        Mark parameters first, e.g. ``system["C2"].learn("power")``. The optimizer defaults
        to LBFGS, which suits the few smooth parameters of lens design. Calling again
        continues with the same optimizer state unless new params or ``reset`` are passed.
        """
        if isinstance(objectives, Objective):
            objectives = [objectives]
        if not self.system.learnable:
            raise ValueError("Nothing to optimize: mark parameters with component.learn(...)")

        if reset or optimizer_params is not None or not self.system.has_optimizer():
            self.system.set_optimizer(optimizer_params or OptimizerParams.LBFGS())
            self.system.set_scheduler(scheduler_params, num_iter=num_iters)
        elif scheduler_params is not None:
            self.system.set_scheduler(scheduler_params, num_iter=num_iters)

        def closure() -> torch.Tensor:
            self.system.zero_optimizer_grad()
            loss = self.loss(objectives)
            loss.backward()
            return loss

        report_every = max(1, num_iters // 10)
        for i in range(num_iters):
            loss = float(self.system.step_optimizer(closure).detach())
            self.system.step_scheduler(loss)
            self.losses.append(loss)
            self.lrs.append(self.system.get_current_lr())
            if self.verbose and (i % report_every == 0 or i == num_iters - 1):
                print(f"iter {i:4d}  loss {loss:.4e}  lr {self.lrs[-1]:.2e}")
        return self

    # endregion --- optimization ---

    def __repr__(self) -> str:
        return (
            f"RayTracer(solver={self.solver!r}, rays={self.rays.num_rays}, "
            f"system={self.system.names})"
        )
