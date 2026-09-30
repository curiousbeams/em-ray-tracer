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

from em_ray_tracer.fields.base import AxialField
from em_ray_tracer.utils import as_parameter


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
        super().__init__()
        self.B0 = as_parameter(B0)
        self.a = as_parameter(a)
        self.extent = extent

    @property
    def half_extent(self) -> float:
        return float(self.extent * self.a)

    def derivative(self, n: int, z: torch.Tensor) -> torch.Tensor:
        """B0 a^2 f^(n)(z), with f^(n) the pole derivative in the module docstring."""
        raise NotImplementedError("GlaserField.derivative: port poleDerivative (kit/paraxial.js)")


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
        self.k = as_parameter(k)
        self.a = as_parameter(a)
        self.extent = extent

    @property
    def half_extent(self) -> float:
        return float(self.extent * self.a)

    def derivative(self, n: int, z: torch.Tensor) -> torch.Tensor:
        """(n == 0) phi0 - phi0 k^2 a^2 f^(n)(z). Note: returns the profile per volt of phi0."""
        raise NotImplementedError("SchiskeField.derivative: port schiskeField (kit/paraxial.js)")
