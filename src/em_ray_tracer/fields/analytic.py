"""Analytic axial fields: Glaser (magnetic) and Schiske (electrostatic). STUBS.

Both profiles are a constant times 1/(a^2 + z^2), whose n-th derivative has an exact closed
form. Writing 1/(a^2 + z^2) = (1/2ia)[1/(z - ia) - 1/(z + ia)] and differentiating each pole:

    f^(n)(z) = (-1)^(n+1) n! sin((n+1) psi) / (a rho^(n+1)),
    rho = |z - ia|,  psi = arg(z - ia) = atan2(-a, z).

It is exact at every order and costs the same at every order, where finite differences at the
seventh derivative lose all their digits. Reference implementation: ``poleDerivative`` in the
em-widgets kit (kit/paraxial.js), with its tests in test/paraxial.test.js.
"""

import torch
import math

from em_ray_tracer.fields.base import AxialField
from em_ray_tracer.utils import as_parameter
import sympy as sp


class GlaserField(AxialField):
    """Glaser's bell-shaped field, B(z) = B0 / (1 + ((z - z0)/a)^2).

    Not an exact coil geometry, but it reproduces a pole-piece lens well enough that
    analytical aberration theory is built on it, and its paraxial ray equation has a closed-form
    solution. With k^2 = eta^2 B0^2 a^2 / (4 phi_hat) and omega = sqrt(1 + k^2),
    substituting z = a cot(theta), r = u / sin(theta) gives u'' + omega^2 u = 0. Use this to
    test the paraxial solver.

    Parameters
    ----------
    B0 : float
        Peak axial field [T].
    a : float
        Half-width at half-maximum [m].
    extent : float
        Integration half-span in units of ``a``; the field is 1% of its peak at 10a.
    """

    kind = "magnetic"

    def __init__(self, B0: float, a: float, extent: float = 10.0):
        """z was decided to be relative to the component.z,
        initially there were two z0 positions that could be disagreeing in case one of the two is optimized"""
        super().__init__()
        self.b0 = as_parameter(B0)
        self.a = as_parameter(a)
        self.extent = extent
        self.ddict={}

        self.sym_z, self.sym_b0, self.sym_a=sp.symbols("z b0 a",real=True)
        self.args=(self.sym_z, self.sym_b0, self.sym_a)
        self.sym_b=self.sym_b0/(1+((self.sym_z)/self.sym_a)**2) #analytical glaser profile

        


    @property
    def half_extent(self) -> float:
        return float(self.extent * self.a)

    def derivative(self, n: int, z: torch.Tensor) -> torch.Tensor:
        """B0 a^2 f^(n)(z), with f^(n) the pole derivative in the module docstring."""
        if n in self.ddict.keys(): #no unnecessary repeated calculations
            nth_numeric=self.ddict[n]
        else:
            nth_expression=sp.diff(self.sym_b,self.sym_z,n)
            nth_numeric=sp.lambdify(self.args,nth_expression,"torch")
            self.ddict[n]=nth_numeric

        return nth_numeric(z,self.b0,self.a)
        


class SchiskeField(AxialField):
    """Schiske's model of an einzel lens, phi(z) = phi0 (1 - k^2 / (1 + (z/a)^2)).

    The electron decelerates through the middle electrode and re-accelerates, so there is no
    net change of energy. ``phi0`` is the beam's accelerating voltage, read from
    ``Rays.voltage`` at trace time; the field itself stores only the shape (k, a).

    Parameters
    ----------
    k : float
        Strength, 0 to 1 (the potential dips to phi0 (1 - k^2)).
    a : float
        Half-width [m].
    extent : float
        Integration half-span in units of ``a``.
    """

    kind = "electrostatic"

    def __init__(self, k: float, a: float, extent: float = 10.0):
        super().__init__()
        if abs(k)>=1:raise ValueError(f"Rays can't pass this lens, with |k|={abs(k)}>=1 the ray would reflect and not transmit.")
        self.k = as_parameter(k)
        self.a = as_parameter(a)
        self.extent = extent
        self.ddict={}

        self.sym_z, self.sym_k, self.sym_a=sp.symbols("z k a",real=True)
        self.args=(self.sym_z, self.sym_k, self.sym_a)
        self.sym_phi = (1 - self.sym_k**2 / (1 + (self.sym_z/self.sym_a)**2)) #analytical schiske profile w/out voltage

    @property
    def half_extent(self) -> float:
        return float(self.extent * self.a)

    def derivative(self, n: int, z: torch.Tensor) -> torch.Tensor:
        """(n == 0) phi0 - phi0 k^2 a^2 f^(n)(z). Note: returns the profile per volt of phi0."""
        if n in self.ddict.keys(): #no unnecessary repeated calculations
                nth_numeric=self.ddict[n]
        else:           
            nth_expression=sp.diff(self.sym_phi,self.sym_z,n)
            nth_numeric=sp.lambdify(self.args,nth_expression,"torch")
            self.ddict[n]=nth_numeric

        return nth_numeric(z,self.k,self.a) #voltage dependance implemented inside (non)paraxial

