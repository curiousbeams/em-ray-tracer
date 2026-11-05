"""Off-axis expansion of an axisymmetric field from its axial profile. STUB.

Laplace's equation fixes the off-axis field term by term from the axial profile F(z):

    F_z = sum_{n=0..N}  (-1)^n / (2^(2n) (n!)^2)       r^(2n)   F^(2n)(z)
    F_x = sum_{n=1..N}  (-1)^n / (2^(2n-1) (n-1)! n!)  x r^(2n-2) F^(2n-1)(z)
    F_y = the same with y in place of x

Truncating at N leaves exactly one unpaired term in the divergence,
(-1)^N / (2^(2N) (N!)^2) r^(2N) F^(2N+1)(z); asserting that identity is a good test.

Order 1 is the paraxial field and order 3 is where spherical aberration appears. Truncating the
longitudinal sum one order below the transverse one gives the strictly linearized field (the
``r^2 B''`` term in B_z is most of a magnetic lens's spherical aberration). Reference
implementation: ``laplaceExpansion`` in the em-widgets kit (kit/nonparaxial.js).
"""

import torch
import math

from em_ray_tracer.fields.base import AxialField, Field3D


class LaplaceExpansion(Field3D):
    """The 3D field of ``axial`` expanded to ``order`` (transverse) and
    ``longitudinal_order`` (defaults to ``order``). Phi can also be returned using phi = True.
    The convention was used such that for order k the highest r term present in the force field expansion 
    is r^k and in the energy r^k+1 
    """

    def __init__(self, axial: AxialField, order: int = 3, longitudinal_order: int | None = None):
        super().__init__()
        self.axial: AxialField = axial
        self.order = order
        self.longitudinal_order = order if longitudinal_order is None else longitudinal_order

    # --- magnetic field components ---
    def B_z(self,x,y,z):
        b_val=torch.zeros_like(x)
        u=x**2+y**2
        for k in range((self.longitudinal_order)//2+1):
            b_val+=(-1/4)**k * 1/(math.factorial(k))**2 *self.axial.derivative(2*k,z,kind_requested="magnetic")*u**k
        return b_val

    def B_x(self,x,y,z):
        b_val=torch.zeros_like(x)
        u=x**2+y**2
        for k in range((self.order-1)//2+1):
            b_val+=-(-1/4)**k * 1/(math.factorial(k))**2 *self.axial.derivative(2*k+1,z,kind_requested="magnetic")*u**(k)*x/(2*k+2)
        return b_val

    def B_y(self,x,y,z):
        b_val=torch.zeros_like(x)
        u=x**2+y**2
        for k in range((self.order-1)//2+1):
            b_val+=-(-1/4)**k * 1/(math.factorial(k))**2 *self.axial.derivative(2*k+1,z,kind_requested="magnetic")*u**(k)*y/(2*k+2)
        return b_val


    # --- electric field components ---
    def phi_z(self,x,y,z):
        phi_val=torch.zeros_like(x)
        u=x**2+y**2
        for k in range((self.longitudinal_order)//2+1):
            phi_val+=(-1/4)**k * 1/(math.factorial(k))**2 *self.axial.derivative(2*k+1,z,kind_requested="electrostatic")*u**k
        return phi_val
    
    def phi_x(self,x,y,z):
        phi_val=torch.zeros_like(x)
        u=x**2+y**2
        for k in range((self.order+1)//2+1):
            if k !=0: 
                phi_val+=(-1/4)**k * 1/(math.factorial(k))**2 *self.axial.derivative(2*k,z,kind_requested="electrostatic") * (2*x*k*u**(k-1))
        return phi_val

    def phi_y(self,x,y,z):
        phi_val=torch.zeros_like(x)
        u=x**2+y**2
        for k in range((self.order+1)//2+1):
            if k !=0: 
                phi_val+=(-1/4)**k * 1/(math.factorial(k))**2 *self.axial.derivative(2*k,z,kind_requested="electrostatic") * (2*y*k*u**(k-1))
        return phi_val

    #--- absolute value electric field ---
    def phi(self,x,y,z,voltage : float):
        phi_val=torch.ones_like(x)*voltage #to offset up to electron level
        u=x**2+y**2
        for k in range((self.order+1)//2+1):
            phi_val+=(-1/4)**k * 1/(math.factorial(k))**2 *self.axial.derivative(2*k,z,kind_requested="electrostatic")*u**k
        return phi_val
        

    
    def evaluate(
        self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor, voltage: float|None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        kind=self.axial.kind
        match kind:
            case "magnetic":
                E=torch.zeros(len(x),3)
                B=torch.stack((self.B_x(x,y,z),self.B_y(x,y,z),self.B_z(x,y,z)),dim=-1)
            case "electrostatic":
                E=torch.stack((self.phi_x(x,y,z),self.phi_y(x,y,z),self.phi_z(x,y,z)),dim=-1)
                B=torch.zeros(len(x),3)
            case "static-electromagnetic":
                E=torch.stack((self.phi_x(x,y,z),self.phi_y(x,y,z),self.phi_z(x,y,z)),dim=-1)
                B=torch.stack((self.B_x(x,y,z),self.B_y(x,y,z),self.B_z(x,y,z)),dim=-1)
        if voltage != None:
            phi_abs=self.phi(x,y,z,voltage)
            return (E,B,phi_abs)
        else:
            return (E,B)
