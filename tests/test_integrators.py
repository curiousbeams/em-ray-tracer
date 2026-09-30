"""The generic integrator steps converge at their stated order."""

import math

import torch

from em_ray_tracer.solvers.integrators import rk4_step, velocity_verlet_step


def harmonic_error(steps: int, method: str) -> float:
    """Integrate x'' = -x over [0, 1] from (1, 0); return |x(1) - cos(1)|.

    Not over a whole period: there, symmetric phase errors cancel and inflate the order.
    """
    h = torch.tensor(1.0 / steps, dtype=torch.float64)
    z = torch.tensor(0.0, dtype=torch.float64)
    if method == "rk4":
        y = torch.tensor([1.0, 0.0], dtype=torch.float64)
        for i in range(steps):
            y = rk4_step(lambda _, s: torch.stack([s[1], -s[0]]), z + i * h, y, h)
        return abs(float(y[0]) - math.cos(1.0))
    pos, vel = torch.tensor(1.0, dtype=torch.float64), torch.tensor(0.0, dtype=torch.float64)
    for i in range(steps):
        pos, vel = velocity_verlet_step(lambda _, p: -p, z + i * h, pos, vel, h)
    return abs(float(pos) - math.cos(1.0))


def test_rk4_is_fourth_order():
    ratio = harmonic_error(20, "rk4") / harmonic_error(40, "rk4")
    assert 14 < ratio < 18


def test_velocity_verlet_is_second_order():
    ratio = harmonic_error(20, "verlet") / harmonic_error(40, "verlet")
    assert 3.5 < ratio < 4.5
