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

import torch

from em_ray_tracer.fields.base import AxialField, FieldKind
from em_ray_tracer.utils import as_parameter


def pole_derivative(n: int, z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
    """The n-th derivative of 1 / (a^2 + z^2) at ``z`` (module docstring)."""
    z = torch.as_tensor(z, dtype=a.dtype, device=a.device)
    rho = torch.hypot(z, a)
    psi = torch.atan2(-a, z)
    return (-1) ** (n + 1) * math.factorial(n) * torch.sin((n + 1) * psi) / (a * rho ** (n + 1))


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

    kind = "magnetic"

    def __init__(self, B0: float, a: float, extent: float = 10.0):
        super().__init__()
        self.B0 = as_parameter(B0)
        self.a = as_parameter(a)
        self.extent = extent

    @property
    def half_extent(self) -> float:
        return float(self.extent * self.a)

    def derivative(self, n: int, z: torch.Tensor, kind_requested: FieldKind|None = None) -> torch.Tensor:
        """The n-th derivative of B(z): B0 a^2 f^(n)(z)."""
        if kind_requested and kind_requested != self.kind:
            raise ValueError(f"Glaser field only contains a magnetic field, not a field of type {kind_requested}")
        return self.B0 * self.a**2 * pole_derivative(n, z, self.a)


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

    kind = "electrostatic"

    def __init__(self, k: float, a: float, extent: float = 10.0):
        super().__init__()
        if abs(k) >= 1:
            raise ValueError(
                f"|k| = {abs(k)} >= 1: the potential reaches zero on the axis and the electrons "
                "are reflected"
            )
        self.k = as_parameter(k)
        self.a = as_parameter(a)
        self.extent = extent

    @property
    def half_extent(self) -> float:
        return float(self.extent * self.a)

    def derivative(self, n: int, z: torch.Tensor, kind_requested: FieldKind|None = None) -> torch.Tensor:
        """The n-th derivative of phi(z) / phi0: (n == 0) - k^2 a^2 f^(n)(z)."""
        if kind_requested and kind_requested != self.kind:
            raise ValueError(f"Schiske field only contains an electrostatic field, not a field of type {kind_requested}")
        dip = -(self.k**2) * self.a**2 * pole_derivative(n, z, self.a)
        return 1 + dip if n == 0 else dip
