"""ODE ray tracing through field components. The non-paraxial equations of motion are a STUB.

The solver is fully wired: gaps are exact drifts, thin elements act as they do under the
transfer-matrix solver, and field components are integrated with the chosen integrator. The
paraxial equations are implemented; only ``NonParaxialEquations.bind`` remains. A system of
thin elements traces identically under either solver.
"""

import math
from abc import ABC, abstractmethod
from typing import Self

import torch

from em_ray_tracer.components.base import ComponentBase
from em_ray_tracer.components.field_lenses import FieldComponent
from em_ray_tracer.constants import ETA, lorentz_factor, relativistic_potential
from em_ray_tracer.rays import DTYPE, XP, YP, Rays, X, Y
from em_ray_tracer.solvers.base import SolverBase
from em_ray_tracer.solvers.integrators import RHS, rk4_step
from em_ray_tracer.solvers.params import IntegratorParams, IntegratorParamsType
from em_ray_tracer.transfer_map import TransferMap


def state_derivative(xp, xpp, yp, ypp) -> torch.Tensor:
    """d/dz [x, x', y, y', delta] from its components; delta is constant."""
    return torch.stack([xp, xpp, yp, ypp, torch.zeros_like(xp)], dim=-1)


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
    equation above. Checked against the Glaser closed form (tests/test_stubs.py) and, for
    electrostatic lenses and the sense of the rotation, in tests/test_paraxial.py.
    """

    def bind(self, component: FieldComponent, voltage: float) -> RHS:
        match component.field.kind:
            case "electrostatic":

                def f(z: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
                    # the field's profile is per volt of the beam's accelerating voltage
                    phi = voltage + component.axial_derivative(0, z)
                    dphi = component.axial_derivative(1, z)
                    d2phi = component.axial_derivative(2, z)
                    gamma, phi_hat = lorentz_factor(phi), relativistic_potential(phi)
                    damping = gamma * dphi / (2 * phi_hat)
                    focusing = gamma * d2phi / (4 * phi_hat)
                    x, xp, y, yp = state[..., X], state[..., XP], state[..., Y], state[..., YP]
                    return state_derivative(
                        xp, -damping * xp - focusing * x, yp, -damping * yp - focusing * y
                    )

            case "magnetic":
                s = ETA / math.sqrt(relativistic_potential(voltage))

                def f(z: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
                    B = component.axial_derivative(0, z)
                    dB = component.axial_derivative(1, z)
                    x, xp, y, yp = state[..., X], state[..., XP], state[..., Y], state[..., YP]
                    return state_derivative(
                        xp, -s * (B * yp + dB * y / 2), yp, s * (B * xp + dB * x / 2)
                    )

            case "static-electromagnetic":
                def f(z: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
                    # the field's profile is per volt of the beam's accelerating voltage
                    phi = voltage + component.axial_derivative(0, z, kind_requested="electrostatic")
                    dphi = component.axial_derivative(1, z, kind_requested="electrostatic")
                    d2phi = component.axial_derivative(2, z, kind_requested="electrostatic")
                    gamma, phi_hat = lorentz_factor(phi), relativistic_potential(phi)
                    
                    B = component.axial_derivative(0, z, kind_requested="magnetic")
                    dB = component.axial_derivative(1, z, kind_requested="magnetic")

                    s = ETA / math.sqrt(phi_hat)
                    damping = gamma * dphi / (2 * phi_hat)
                    focusing = gamma * d2phi / (4 * phi_hat)
                    x, xp, y, yp = state[..., X], state[..., XP], state[..., Y], state[..., YP]

                    dxpdz=-damping * xp - focusing * x - s * (B * yp + dB * y / 2)
                    dypdz=-damping * yp - focusing * y + s * (B * xp + dB * x / 2)
                    
                    return state_derivative(
                        xp, dxpdz, yp, dypdz
                    )


            case kind:
                raise TypeError(f"No paraxial equations for a {kind!r} field")
        return f


class NonParaxialEquations(EquationsOfMotion):
    """The coupled 3D ray equations, with the off-axis field from ``LaplaceExpansion`` to
    ``expansion_order``.

    Port ``equationsOfMotion`` from the em-widgets kit (kit/nonparaxial.js). That version
    leaves out the kinematic factor rho = sqrt(1 + x'^2 + y'^2), which contributes third-order
    terms of its own. Decide explicitly whether to keep it: truncating the field and
    linearizing the kinematics are separate approximations.
    """

    def __init__(self, expansion_order: int = 3):
        if not isinstance(expansion_order,int) or (expansion_order <= 0):
            raise ValueError(f"The expansion order must be an integer and greater then 0, it is now {expansion_order,type(expansion_order)}")
        self.expansion_order = expansion_order
        

    def bind(self, component: FieldComponent, voltage: float) -> RHS:
        order=self.expansion_order
        def phi(z,x,y):
            phi_val=torch.ones_like(x)*voltage #to offset up to electron level
            u=x**2+y**2
            for k in range((order-1)//2+1):
                phi_val+=(-1/4)**k * 1/(math.factorial(k))**2 *component.axial_derivative(2*k,z,kind_requested="electrostatic")*u**k
            return phi_val
        
        def phi_x(z,x,y):
            phi_val=torch.zeros_like(x)
            u=x**2+y**2
            for k in range((order+1)//2+1):
                if k !=0: 
                    phi_val+=(-1/4)**k * 1/(math.factorial(k))**2 *component.axial_derivative(2*k,z,kind_requested="electrostatic") * (2*x*k*u**(k-1))
            return phi_val

        def phi_y(z,x,y):
            phi_val=torch.zeros_like(x)
            u=x**2+y**2
            for k in range((order+1)//2+1):
                if k !=0: 
                    phi_val+=(-1/4)**k * 1/(math.factorial(k))**2 *component.axial_derivative(2*k,z,kind_requested="electrostatic") * (2*y*k*u**(k-1))
            return phi_val

        def phi_z(z,x,y):
            phi_val=torch.zeros_like(x)
            u=x**2+y**2
            for k in range((order+2)//2):
                phi_val+=(-1/4)**k * 1/(math.factorial(k))**2 *component.axial_derivative(2*k+1,z,kind_requested="electrostatic")*u**k
            return phi_val

        def B_z(z,x,y):
            b_val=torch.zeros_like(x)
            u=x**2+y**2
            for k in range((order-1)//2+1):
                b_val+=(-1/4)**k * 1/(math.factorial(k))**2 *component.axial_derivative(2*k,z,kind_requested="magnetic")*u**k
            return b_val

        def B_x(z,x,y):
            b_val=torch.zeros_like(x)
            u=x**2+y**2
            for k in range((order-1)//2+1):
                b_val+=-(-1/4)**k * 1/(math.factorial(k))**2 *component.axial_derivative(2*k+1,z,kind_requested="magnetic")*u**(k)*x/(2*k+2)
            return b_val

        def B_y(z,x,y):
            b_val=torch.zeros_like(x)
            u=x**2+y**2
            for k in range((order-1)//2+1):
                b_val+=-(-1/4)**k * 1/(math.factorial(k))**2 *component.axial_derivative(2*k+1,z,kind_requested="magnetic")*u**(k)*y/(2*k+2)
            return b_val


        match component.field.kind:
            case "electrostatic":
                def f(z: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
                    x, xp, y, yp = state[..., X], state[..., XP], state[..., Y], state[..., YP]

                    w=torch.sqrt(1+xp**2+yp**2)
                    p=phi(z,x,y)
                    gamma=lorentz_factor(p)
                    phi_hat=relativistic_potential(p)

                    p_z=phi_z(z,x,y)
                    p_x=phi_x(z,x,y)
                    p_y=phi_y(z,x,y)
                    
                    
                    
                    dxpdz=w**2*(gamma/(2*phi_hat))*(p_x-xp*p_z)
                    dypdz=w**2*(gamma/(2*phi_hat))*(p_y-yp*p_z)
                    
                    return state_derivative(
                        xp, dxpdz, yp, dypdz
                    )

            case "magnetic":
                p=torch.as_tensor(voltage,dtype=DTYPE)
                phi_hat=relativistic_potential(p)
                s = ETA / torch.sqrt(phi_hat)

                def f(z: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
                    x, xp, y, yp = state[..., X], state[..., XP], state[..., Y], state[..., YP]

                    w=torch.sqrt(1+xp**2+yp**2)
                    
                    b_z=B_z(z,x,y) #B
                    b_x=B_x(z,x,y)  #-x*dB/2
                    b_y=B_y(z,x,y)
                    
                    
                    
                    dxpdz= + s*w*(b_y*(1+xp**2)-b_z*yp-xp*yp*b_x)
                    dypdz= - s*w*(b_x*(1+yp**2)-b_z*xp-xp*yp*b_y)

                    return state_derivative(
                        xp, dxpdz, yp, dypdz
                    )


            case "static-electromagnetic":                
                def f(z: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
                    x, xp, y, yp = state[..., X], state[..., XP], state[..., Y], state[..., YP]

                    
                    w=torch.sqrt(1+xp**2+yp**2)
                    p=phi(z,x,y)
                    gamma=lorentz_factor(p)
                    phi_hat=relativistic_potential(p)
                    s = ETA / torch.sqrt(phi_hat)

                    p_z=phi_z(z,x,y)
                    p_x=phi_x(z,x,y)
                    p_y=phi_y(z,x,y)

                    b_z=B_z(z,x,y)
                    b_x=B_x(z,x,y)
                    b_y=B_y(z,x,y)
                    
                    
                    
                    dxpdz=w**2*(gamma/(2*phi_hat))*(p_x-xp*p_z) + s*w*(b_y*(1+xp**2)-b_z*yp-xp*yp*b_x)
                    dypdz=w**2*(gamma/(2*phi_hat))*(p_y-yp*p_z) - s*w*(b_x*(1+yp**2)-b_z*xp-xp*yp*b_y)

                    return state_derivative(
                        xp, dxpdz, yp, dypdz
                    )


            case kind:
                raise TypeError(f"No paraxial equations for a {kind!r} field")
        return f


        


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
