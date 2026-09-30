"""Physics-agnostic integrator steps on batched tensors.

Each step advances ``y`` (any shape, e.g. (..., N, 5)) by one step ``h`` of the independent
variable. Every ray in the bundle is advanced by one tensor operation, and the right-hand side
can evaluate the axial field once per z and broadcast it over rays, which is the torch
counterpart of the em-widgets kit's ``memoiseDerivatives``.
"""

from collections.abc import Callable

import torch

RHS = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def rk4_step(f: RHS, z: torch.Tensor, y: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
    """One classical fourth-order Runge-Kutta step of dy/dz = f(z, y)."""
    k1 = f(z, y)
    k2 = f(z + h / 2, y + (h / 2) * k1)
    k3 = f(z + h / 2, y + (h / 2) * k2)
    k4 = f(z + h, y + h * k3)
    return y + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)


def velocity_verlet_step(
    accel: RHS, z: torch.Tensor, pos: torch.Tensor, vel: torch.Tensor, h: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """One velocity-Verlet step of pos'' = accel(z, pos).

    Second-order and symplectic only because ``accel`` does not depend on ``vel``. Ray
    equations with magnetic fields or electrostatic damping do depend on it, so they lose both
    properties; see ``IntegratorParams``.
    """
    a0 = accel(z, pos)
    pos_next = pos + h * vel + (h * h / 2) * a0
    a1 = accel(z + h, pos_next)
    return pos_next, vel + (h / 2) * (a0 + a1)


def dopri45_step(f: RHS, z, y, h, rtol: float, atol: float):
    """Adaptive Dormand-Prince 4(5) step. Returns (y_next, h_used, h_suggested)."""
    raise NotImplementedError("dopri45_step")


def boris_push(position, velocity, E, B, dt, charge_over_mass):
    """Time-domain Boris push for the Lorentz force (energy-conserving for pure B)."""
    raise NotImplementedError("boris_push")
