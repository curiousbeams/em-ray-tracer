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
from em_ray_tracer.constants import DTYPE, ETA, lorentz_factor, relativistic_potential
from em_ray_tracer.rays import XP, YP, Rays, X, Y
from em_ray_tracer.solvers.base import SolverBase
from em_ray_tracer.solvers.integrators import RHS, rk4_step
from em_ray_tracer.solvers.params import IntegratorParams, IntegratorParamsType
from em_ray_tracer.transfer_map import TransferMap


def state_derivative(xp, xpp, yp, ypp) -> torch.Tensor:
    """d/dz [x, x', y, y', delta] from its components; delta is constant."""
    return torch.stack([xp, xpp, yp, ypp, torch.zeros_like(xp)], dim=-1)

class _HalfStepTable:
    """(S, S', D, L) on the RK4 half-step grid z_start + k h/2, k = 0 .. 2n."""
    def __init__(self, z_start: float, step: float, n_steps: int, values: torch.Tensor):
        self.z_start, self.step, self.n_steps = z_start, step, n_steps
        self.values = values  # (2n + 1, 4), used for batched lookups
        self.rows = values.unbind(0)

    def index(self, z) -> torch.Tensor:
        """Half-step grid index of z (long tensor, same shape as z; 0-d for a scalar)."""
        z = torch.as_tensor(z, dtype=DTYPE).detach()
        k = (z - self.z_start) / (self.step / 2)
        k_round = torch.round(k)
        bad = ((k - k_round).abs() > 1e-6) | (k_round < 0) | (k_round > 2 * self.n_steps)
        if bad.any():
            z_bad = z[bad].flatten()[0]  # first offending value, for the message
            raise ValueError(f"z = {float(z_bad):.6g} m is not on the half-step grid")
        return k_round.long()

    def at(self, z) -> torch.Tensor:
        """Rows (S, S', D, L) at z: shape (4,) for scalar z, (*z.shape, 4) for batched z."""
        idx = self.index(z)
        if idx.ndim == 0:
            return self.rows[int(idx)]
        return self.values[idx]  # advanced indexing; differentiable


class EquationsOfMotion(ABC):
    """Right-hand side of the ray equation in the z-domain, on the state [x, x', y, y', delta].

    ``bind`` fixes the component and beam voltage and returns ``f(z, state) -> d state / dz``
    (with d delta / dz = 0), in the lab frame, with ``z`` absolute. Evaluate axial-field
    derivatives once per ``z`` (a scalar) and broadcast them over the rays.
    """

    @abstractmethod
    def bind(self, component: FieldComponent, voltage: float) : ...

    def __call__(self,z: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        return self.f(z, state)


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
    def __init__(self):
        self.f: RHS | None = None
        self.component: FieldComponent | None = None
        self.voltage = None
        self.table: _HalfStepTable | None = None

    def coefficients(self, z: torch.Tensor) -> torch.Tensor:
        """(S, S', D, L) at z, stacked on a last axis of 4. S and S' include the field amplitude.

        Deliberately not under torch.no_grad(): the graph back to the field values is kept.
        Works for a 0-d z (on-the-fly) and for a whole grid (table build).
        """
        component : FieldComponent = self.component
        zero = torch.zeros_like(z)
        b = db = dphi = d2phi = zero
        phi = self.voltage + zero
        d0=component.axial_derivative(0,z)
        d1=component.axial_derivative(1,z)
        for key in d0.keys():
            match key:
                case "electrostatic":
                    phi = phi + d0["electrostatic"]
                    dphi = d1["electrostatic"]
                    d2phi = component.axial_derivative(2, z)["electrostatic"]
                case "magnetic":
                    b = d0["magnetic"]
                    db = d1["magnetic"]
                case _:
                    raise Warning("Unknown field present : ",key)
            

        gamma, phi_hat = lorentz_factor(phi), relativistic_potential(phi)
        s = ETA / phi_hat.sqrt()
        return torch.stack(
            [s * b, s * db, gamma * dphi / (2 * phi_hat), gamma * d2phi / (4 * phi_hat)], dim=-1
        )

    def _row(self, z) -> torch.Tensor:
        if self.table is not None:
            return self.table.at(z)
        return self.coefficients(torch.as_tensor(z, dtype=DTYPE))  # fallback: on the fly

    def bind(self, component: FieldComponent, voltage: float):
        self.component, self.voltage, self.table = component, voltage, None

        def f(z: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
            S, dS, D, L = self._row(z).unbind(-1)
            x, xp, y, yp = state[..., X], state[..., XP], state[..., Y], state[..., YP]
            return state_derivative(
                xp, -D * xp - L * x - (S * yp + dS * y / 2),
                yp, -D * yp - L * y + (S * xp + dS * x / 2),
            )

        self.f = f
    
        
    def pre_calc(self, n_steps: int) -> None:
        """Tabulate the coefficients on z_start + k*step/2, k = 0 .. 2*n_steps, for fixed-step RK4."""
        z_start=self.component.z_start
        z_end=self.component.z_end
        step=(z_end-z_start)/n_steps
        grid = z_start + (step / 2) * torch.arange(2 * n_steps + 1, dtype=DTYPE)
        self.table = _HalfStepTable(z_start, step, n_steps, self.coefficients(grid))


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
        self.f: RHS | None = None

    def bind(self, component: FieldComponent, voltage: float):
        order=self.expansion_order
        component.axial_laplace(order)      
        self.component=component

        def f(z: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
            x, xp, y, yp = state[..., X], state[..., XP], state[..., Y], state[..., YP]

            
            w=torch.sqrt(1+xp**2+yp**2)                    

            E,B,p=self.component.field_values(x,y,z,voltage=voltage)

            e_x=E[...,0]
            e_y=E[...,1]
            e_z=E[...,2]
            
            b_x=B[...,0]
            b_y=B[...,1]
            b_z=B[...,2]

            gamma=lorentz_factor(p)
            phi_hat=relativistic_potential(p)
            s = ETA / torch.sqrt(phi_hat)

            
            
            dxpdz=w**2*(gamma/(2*phi_hat))*(e_x-xp*e_z) + s*w*(b_y*(1+xp**2)-b_z*yp-xp*yp*b_x)
            dypdz=w**2*(gamma/(2*phi_hat))*(e_y-yp*e_z) - s*w*(b_x*(1+yp**2)-b_z*xp-xp*yp*b_y)

            return state_derivative(xp, dxpdz, yp, dypdz)
        self.f=f

    def pre_calc(self,n_steps: int):
        self.component.laplace.pre_calc(n_steps=n_steps)
    


        


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

    def propagate(self, component: ComponentBase, rays: Rays, snapshots : list[Rays]|None=None) -> tuple[Rays, TransferMap | None,list[Rays]|None]:
        if isinstance(component, FieldComponent):
            return self.integrate(component, rays,snapshots=snapshots)
        current=component.apply(rays)
        if snapshots is not None : 
            snapshots.append(current)
        return current, component.transfer_map(voltage=rays.voltage),snapshots

    def integrate(self, component: FieldComponent, rays: Rays, snapshots : list[Rays]|None=None) -> tuple[Rays, None, list[Rays] | None]:
        if rays.voltage is None:
            raise ValueError("Field components need the beam energy: set Rays(voltage=...)")
        self.equations.bind(component, rays.voltage)
        f=self.equations
        z0, z1 = component.z_start, component.z_end
        match self.integrator:
            case IntegratorParams.RK4(dz=dz):
                n = max(1, math.ceil(float((z1 - z0).detach()) / dz))
                h = (z1 - z0) / n
                state = rays.state
                f.pre_calc(n_steps=n)
                for i in range(n):
                    state = rk4_step(f, z0 + i * h, state, h)
                    if snapshots is not None : 
                        snapshots.append(rays.replace(state=state, z=z0+(i+1)*h))
            case _:
                raise NotImplementedError(f"{type(self.integrator).__name__} integration")
        return rays.replace(state=state, z=z1), None, snapshots

    def __repr__(self) -> str:
        return f"ODESolver({type(self.equations).__name__}, {self.integrator})"
