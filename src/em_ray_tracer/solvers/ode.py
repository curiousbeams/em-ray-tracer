"""ODE ray tracing through field components. The equations of motion are STUBS.

The solver is fully wired: gaps are exact drifts, thin elements act as they do under the
transfer-matrix solver, and field components are integrated with the chosen integrator. Only
the right-hand sides (``ParaxialEquations.bind``, ``NonParaxialEquations.bind``) remain, so a
system of thin elements already traces identically under either solver.
"""

import math
from abc import ABC, abstractmethod
from typing import Self

from em_ray_tracer.components.base import ComponentBase
from em_ray_tracer.components.field_lenses import FieldComponent
from em_ray_tracer.rays import Rays
from em_ray_tracer.solvers.base import SolverBase
from em_ray_tracer.solvers.integrators import RHS, rk4_step
from em_ray_tracer.solvers.params import IntegratorParams, IntegratorParamsType
from em_ray_tracer.transfer_map import TransferMap


class EquationsOfMotion(ABC):
    """Right-hand side of the ray equation in the z-domain, on the state [x, x', y, y', delta].

    ``bind`` fixes the component and beam voltage and returns ``f(z, state) -> d state / dz``
    (with d delta / dz = 0), in the lab frame, with ``z`` absolute. Evaluate axial-field
    derivatives once per ``z`` (a scalar) and broadcast them over the rays.
    """

    @abstractmethod
    def bind(self, component: FieldComponent, voltage: float) -> RHS: ...


class ParaxialEquations(EquationsOfMotion):
    """The linearized ray equation.

    In the frame rotating with the Larmor angle (Hawkes & Kasper), with gamma the Lorentz
    factor and phi_hat the relativistic potential (``constants``):

        r'' + (gamma phi' / 2 phi_hat) r' + ((gamma phi'' + eta^2 B^2) / 4 phi_hat) r = 0

    Implement it in the lab frame instead, so the Larmor rotation appears as x-y coupling and
    the output is directly comparable with the transfer-matrix solver. With s = eta/sqrt(phi_hat):

        x'' = -(gamma phi'/2 phi_hat) x' - (gamma phi''/4 phi_hat) x - s (B y' + B' y / 2)
        y'' = -(gamma phi'/2 phi_hat) y' - (gamma phi''/4 phi_hat) y + s (B x' + B' x / 2)

    The magnetic signs follow ``equationsOfMotion`` in the em-widgets kit (kit/nonparaxial.js).
    Substituting x + iy = w exp(i theta), with theta' = s B / 2, recovers the rotating-frame
    equation above. Check against the Glaser closed form (tests/test_stubs.py).
    """

    def bind(self, component: FieldComponent, voltage: float) -> RHS:
        raise NotImplementedError("ParaxialEquations.bind")


class NonParaxialEquations(EquationsOfMotion):
    """The coupled 3D ray equations, with the off-axis field from ``LaplaceExpansion`` to
    ``expansion_order``.

    Port ``equationsOfMotion`` from the em-widgets kit (kit/nonparaxial.js). That version
    leaves out the kinematic factor rho = sqrt(1 + x'^2 + y'^2), which contributes third-order
    terms of its own. Decide explicitly whether to keep it: truncating the field and
    linearizing the kinematics are separate approximations.
    """

    def __init__(self, expansion_order: int = 3):
        self.expansion_order = expansion_order

    def bind(self, component: FieldComponent, voltage: float) -> RHS:
        raise NotImplementedError("NonParaxialEquations.bind")


class ODESolver(SolverBase):
    """Integrate the ray equation through field components with a chosen integrator."""

    def __init__(self, equations: EquationsOfMotion, integrator: IntegratorParamsType):
        self.equations = equations
        self.integrator = integrator

    @classmethod
    def paraxial(cls, integrator: IntegratorParamsType | None = None) -> Self:
        return cls(ParaxialEquations(), integrator or IntegratorParams.RK4())

    @classmethod
    def nonparaxial(
        cls, expansion_order: int = 3, integrator: IntegratorParamsType | None = None
    ) -> Self:
        return cls(NonParaxialEquations(expansion_order), integrator or IntegratorParams.RK4())

    def propagate(self, component: ComponentBase, rays: Rays) -> tuple[Rays, TransferMap | None]:
        if isinstance(component, FieldComponent):
            return self.integrate(component, rays), None
        return component.apply(rays), component.transfer_map(voltage=rays.voltage)

    def integrate(self, component: FieldComponent, rays: Rays) -> Rays:
        if rays.voltage is None:
            raise ValueError("Field components need the beam energy: set Rays(voltage=...)")
        f = self.equations.bind(component, rays.voltage)
        z0, z1 = component.z_start, component.z_end
        match self.integrator:
            case IntegratorParams.RK4(dz=dz):
                n = max(1, math.ceil(float((z1 - z0).detach()) / dz))
                h = (z1 - z0) / n
                state = rays.state
                for i in range(n):
                    state = rk4_step(f, z0 + i * h, state, h)
            case _:
                raise NotImplementedError(f"{type(self.integrator).__name__} integration")
        return rays.replace(state=state, z=z1)

    def __repr__(self) -> str:
        return f"ODESolver({type(self.equations).__name__}, {self.integrator})"
