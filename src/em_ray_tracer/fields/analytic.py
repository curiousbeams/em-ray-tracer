"""Analytic axial fields: Glaser (magnetic) and Schiske (electrostatic).

Both profiles are a constant times f(z) = 1 / (a^2 + z^2), so every derivative the ray
equations need is a multiple of f^(n), which has an exact closed form. Split f into two simple
poles,

    f(z) = (1 / 2ia) [1 / (z - ia) - 1 / (z + ia)],

and differentiate each n times with d^n/dz^n (z - c)^-1 = (-1)^n n! (z - c)^-(n+1). The two
terms are complex conjugates, so f^(n)(z) = ((-1)^n n! / a) Im[(z - ia)^-(n+1)]. Writing
z - ia = rho exp(i psi):

    f^(n)(z) = (-1)^(n+1) n! sin((n+1) psi) / (a rho^(n+1)),
    rho = |z - ia|,  psi = arg(z - ia) = atan2(-a, z).

It is exact at every order and costs the same at every order, where finite differences at the
seventh derivative lose all their digits. The same formula is ``poleDerivative`` in the
em-widgets kit (kit/paraxial.js).

To cross-check it symbolically (sympy is not a dependency of the package):

    import sympy as sp
    z, a = sp.symbols("z a", positive=True)
    n = 7
    closed = (-1) ** (n + 1) * sp.factorial(n) * sp.sin((n + 1) * sp.atan2(-a, z)) / (
        a * sp.sqrt(z**2 + a**2) ** (n + 1)
    )
    error = closed - sp.diff(1 / (a**2 + z**2), z, n)
    points = (-2, sp.Rational(3, 10), 5)
    [sp.simplify(error.subs({z: zz, a: sp.Rational(3, 4)})) for zz in points]  # [0, 0, 0]

The tests check it against nested autograd instead (tests/test_fields.py).
"""

import math
from typing import Callable, Literal

import torch

from em_ray_tracer.fields.base import AxialField
from em_ray_tracer.utilities.utils import as_parameter


def pole_derivative(n: int, z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
    """The n-th derivative of 1 / (a^2 + z^2) at ``z`` (module docstring)."""
    z = torch.as_tensor(z, dtype=a.dtype, device=a.device)
    rho = torch.hypot(z, a)
    psi = torch.atan2(-a, z)
    return (-1) ** (n + 1) * math.factorial(n) * torch.sin((n + 1) * psi) / (a * rho ** (n + 1))

# region --- standard ---
class GlaserField(AxialField):
    """Glaser's bell-shaped field, B(z) = B0 / (1 + (z/a)^2).

    Not an exact coil geometry, but it reproduces a pole-piece lens well enough that
    analytical aberration theory is built on it, and its paraxial ray equation has a closed-form
    solution. With k^2 = eta^2 B0^2 a^2 / (4 phi_hat) and omega = sqrt(1 + k^2),
    substituting z = a cot(theta), r = u / sin(theta) gives u'' + omega^2 u = 0. Use this to
    test the paraxial solver.

    ``z`` is measured from the lens centre, the ``z`` of the component that holds the field.
    The field has no position of its own, so a learnable lens position can't disagree with it.

    Parameters
    ----------
    B0 : float
        Peak axial field [T].
    a : float
        Half-width at half-maximum [m].
    extent : float
        Integration half-span in units of ``a``; the field is 1% of its peak at 10a.
    """


    def __init__(self, B0: float, a: float, extent: float = 10.0):
        super().__init__()
        self.B0 = as_parameter(B0)
        self.a = as_parameter(a)
        self.extent = extent

    @property
    def half_extent(self) -> float:
        return float(self.extent * self.a)

    def derivative(self, n: int, z: torch.Tensor) -> dict:
        """The n-th derivative of B(z): B0 a^2 f^(n)(z)."""
        return {"magnetic" : self.B0 * self.a**2 * pole_derivative(n, z, self.a)}


class SchiskeField(AxialField):
    """Schiske's model of an einzel lens, phi(z) = phi0 (1 - k^2 / (1 + (z/a)^2)).

    The electron decelerates through the middle electrode and re-accelerates, so there is no
    net change of energy. ``phi0`` is the beam's accelerating voltage, read from
    ``Rays.voltage`` at trace time; the field itself stores only the shape (k, a), and
    ``derivative`` returns the profile per volt of phi0.

    Parameters
    ----------
    k : float
        Strength, |k| < 1 (the potential dips to phi0 (1 - k^2)).
    a : float
        Half-width [m].
    extent : float
        Integration half-span in units of ``a``.
    """


    def __init__(self, k: float, a: float, voltage: float, extent: float = 10.0):
        super().__init__()
        if abs(k) >= 1:
            raise ValueError(
                f"|k| = {abs(k)} >= 1: the potential reaches zero on the axis and the electrons "
                "are reflected"
            )
        self.k = as_parameter(k)
        self.a = as_parameter(a)
        self.voltage=as_parameter(voltage)
        self.extent = extent

    @property
    def half_extent(self) -> float:
        return float(self.extent * self.a)

    def derivative(self, n: int, z: torch.Tensor) -> dict:
        """The n-th derivative of phi(z) / phi0: (n == 0) - k^2 a^2 f^(n)(z)."""
        dip = -(self.k**2) * self.a**2 * pole_derivative(n, z, self.a)
        return {"electrostatic" : self.voltage*dip}

# endregion --- standard ---
# region --- arbitrary ---

class AxialAnalytical(AxialField):
    """On-axis profile given as a function of a single (0-d) z.

    ``axial`` maps a 0-d tensor z to a 0-d tensor (single field kind) or to a
    tensor of shape (K,) (one entry per name in ``fieldkind``, same order).
    It must be written with torch ops only (no .item(), no in-place ops).
    """
    def __init__(self, axial:Callable[[torch.Tensor],torch.Tensor],fieldkind : Literal["electrostatic"]|Literal["magnetic"]|tuple[str,...], half_extent : float) -> None:
        super().__init__()
        self.dlist=[axial]
        self.fieldkind=(fieldkind,) if isinstance(fieldkind,str) else tuple(fieldkind)
        self._half_extent=half_extent

    def derivative(self, n: int, z: torch.Tensor) -> dict:
        """The n-th z-derivative of the axial profile at ``z`` (n = 0 is the profile)."""
        while len(self.dlist) <= n:
            self.dlist.append(torch.func.jacfwd(self.dlist[-1]))
        val=torch.vmap(self.dlist[n])(z.reshape(-1))
        val=val.reshape(*z.shape,-1)
        if val.shape[-1] != len(self.fieldkind):
            raise ValueError(f"axial returned {val.shape[-1]} component(s) but fieldkind has {len(self.fieldkind)}")
        return {name:val[...,i] for i,name in enumerate(self.fieldkind) }


    @property
    def half_extent(self) -> float:
        """Half-width [m] beyond which the field is negligible; sets the integration span."""
        return self._half_extent



