"""Physical constants (SI) and the relativistic helpers every ray equation needs.

Units are SI throughout the package: metres, volts, tesla, radians. Slopes (x', y') are
dimensionless dx/dz, not angles; the two agree to first order.
"""

import math

import torch

E_CHARGE = 1.602176634e-19  # |e| [C]
M_E = 9.1093837015e-31  # electron rest mass [kg]
C_LIGHT = 299_792_458.0  # [m/s]
REST_ENERGY_EV = M_E * C_LIGHT**2 / E_CHARGE  # m c^2 / e [V], ~511 kV

ETA = math.sqrt(E_CHARGE / (2 * M_E))
"""sqrt(|e| / 2m), the constant the magnetic ray equation carries [C^0.5 kg^-0.5]."""

DTYPE = torch.float64
"""Default dtype. Third-order ray aberrations sit ~1e-6 below the paraxial terms, which is
too close to float32 resolution to fit them reliably."""


def relativistic_potential(voltage):
    """The relativistically corrected accelerating potential phi_hat = phi (1 + e phi / 2 m c^2).

    Every ray equation should use this rather than the bare potential; at 200 kV the
    difference is ~20%. Works on floats and tensors.
    """
    return voltage * (1 + voltage / (2 * REST_ENERGY_EV))


def lorentz_factor(voltage):
    """gamma = 1 + e phi / m c^2 for an electron accelerated through ``voltage``."""
    return 1 + voltage / REST_ENERGY_EV
