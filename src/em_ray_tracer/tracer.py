"""The top-level class tying a system, a solver and a ray bundle together.

Build it from its parts with ``from_models``, then ``trace``, ``fit_aberrations`` or
``optimize``.
"""

import warnings
from collections.abc import Sequence
from typing import Literal, Self

import torch

from em_ray_tracer.aberrations.basis import AberrationBasis
from em_ray_tracer.aberrations.coefficients import AberrationCoefficients
from em_ray_tracer.aberrations.rotation import larmor_fit
from em_ray_tracer.optimize.objectives import Evaluation, Objective
from em_ray_tracer.optimize.params import (
    OptimizerParams,
    OptimizerParamsType,
    SchedulerParamsType,
)
from em_ray_tracer.rays import Rays, X, Y
from em_ray_tracer.solvers.base import SolverBase, Trace
from em_ray_tracer.solvers.params import SolverParams, SolverParamsType
from em_ray_tracer.system import OpticalSystem


def _mean_r2(x : torch.Tensor, y : torch.Tensor):
    r2=torch.var(x)+torch.var(y)
    return r2




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

    def trace(self, record: bool | Literal["light"] = False) -> Trace:
        return self.solver.trace(self.system, self.rays, z_end=self.z_end, record=record)

    def gaussian_image(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Paraxial ``(z_image, magnification)`` of the plane the rays start from."""
        return self.system.gaussian_image(self.rays.z, voltage=self.rays.voltage)
    
    def plane_of_least_confusion(self, z_initial: float|torch.Tensor|None = None, z_tol : float = 1e-8, add_plane = False):
        from em_ray_tracer.components.thin import Plane
        if z_initial is None:
            z_initial=self.gaussian_image()[0]


        self.system.add(Plane(name="POLC", z=float(z_initial)))
        plane=self.system["POLC"]
        z=plane.z
        if z.requires_grad is False:
            z.requires_grad_(True)
            internal_grad=True
        else:
            internal_grad=False

        def closure():
            state=self.trace().planes["POLC"].state
            x=state[...,X]
            y=state[...,Y]
            r2=_mean_r2(x,y)
            z.grad=torch.autograd.grad(r2,z)[0]
            return r2
        
        plane_opt=torch.optim.LBFGS([z],lr=1,max_iter=50,tolerance_grad=z_tol,tolerance_change=1e-14,line_search_fn="strong_wolfe")
        plane_opt.step(closure)

        if z.grad is not None and z.grad.abs()>z_tol :
            warnings.warn("POLC may not have converged")
        if internal_grad:
            z.requires_grad_(False)
        if add_plane is False:
            self.system.remove("POLC")
        
        return z.detach().clone(), plane




    def fit_aberrations(
        self,
        basis: AberrationBasis | None = None,
        plane: str | None = None,
        trace: Trace | None = None,
        rotation: torch.Tensor | Literal["larmor","standard"] = "standard"
    ) -> AberrationCoefficients:
        """Fit ``basis`` (default: axial to third order) to the rays at ``plane``. 
        The rotation is removed beforehand either by direct fitting using "standard" 
        or by calculating the larmor rotation from the b-field (first order) using "larmor" """
        trace = trace or self.trace()
        if rotation == "larmor":
            z_out=trace.final.z if plane is None else trace.planes[plane].z
            z_set, phi_set, b_set=self.system.fields(z_start=trace.initial.z, z_end=z_out)
            if not b_set.any():
                rotation="standard"
                warnings.warn("system has no magnetic field to retrieve larmor rotation from, " \
                "used standard rotation fitter instead")
            else :
                voltage=trace.initial.voltage
                #z_imag, magnification=self.system.gaussian_image(z_object=trace.initial.z,voltage=voltage)
                #argument=magnification.angle()
                rotation=larmor_fit(z_set,phi_set,b_set,voltage=voltage)
                #print(f"from the gaussian {argument} rad at z_img {z_i} was found while from larmor fit {rotation} rad was found")
        return AberrationCoefficients.from_trace(trace, basis or AberrationBasis.axial(), plane,rotation=rotation)

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
