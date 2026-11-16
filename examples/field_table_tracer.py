"""A paraxial ray tracer that reads its field coefficients from RK4's half-step grid.

Fixed-step RK4 evaluates the right-hand side only at z_0 + k h/2 (``field_table_timing.py``
explains why, and what precomputing saves). The paraxial ray equations depend on the field
only through four functions of z,

    x'' = -D x' - L x - B0 (S y' + S' y / 2)
    y'' = -D y' - L y + B0 (S x' + S' x / 2)

with S = s B / B0, S' = s B' / B0, s = eta / sqrt(phi_hat), D = gamma phi' / (2 phi_hat) and
L = gamma phi'' / (4 phi_hat). ``coefficients`` computes them on any tensor of z.
``CoefficientTable`` calls it once on the whole grid, and its ``at(z)`` returns the row for z,
so the right-hand side is per-ray arithmetic only. The lens is a Glaser magnetic field and a
Schiske einzel field with a common centre; the table holds their combined effect and does not
record which field each part came from.

Two more things are batched:

- rays: the bundle's state is (N, 5), and each RK4 step moves every ray in one tensor operation;
- excitations: ``focusing_excitation`` traces 64 values of B0 at once (state (64, 2), in the
  frame rotating with the Larmor angle), keeps the bracket where a parallel beam starts to cross
  the axis by ``FOCUS``, and repeats on that bracket.

The script checks that the bundle focuses at ``FOCUS``, that the same equations evaluated on the
fly give the same trace, and that ``ODESolver.paraxial`` gives the same trace for each field on
its own.

Run: uv run python examples/field_table_tracer.py
"""

import math
import time
from dataclasses import dataclass
from typing import Self

import torch

from em_ray_tracer import FieldLens, OpticalSystem, Rays
from em_ray_tracer.constants import DTYPE, ETA, lorentz_factor, relativistic_potential
from em_ray_tracer.fields.analytic import GlaserField, SchiskeField
from em_ray_tracer.rays import XP, YP, X, Y
from em_ray_tracer.solvers.integrators import rk4_step
from em_ray_tracer.solvers.ode import ODESolver, state_derivative
from em_ray_tracer.solvers.params import IntegratorParams

VOLTAGE = 1e4  # beam voltage [V]
A = 4e-3  # half-width of both fields [m]
FOCUS = 20e-3  # where the parallel beam should first cross the axis [m]
DZ = 10e-6  # RK4 step [m]

MAGNETIC = GlaserField(B0=1.0, a=A)  # per tesla: the right-hand side scales it by B0
ELECTROSTATIC = SchiskeField(k=0.5, a=A, voltage=VOLTAGE)
Z_START, Z_END = -MAGNETIC.half_extent, MAGNETIC.half_extent


def coefficients(z: torch.Tensor, magnetic=MAGNETIC, electrostatic=ELECTROSTATIC) -> torch.Tensor:
    """(S, S', D, L) at z, stacked on a last axis of 4. Either field may be None."""
    zero = torch.zeros_like(z)
    b = db = dphi = d2phi = zero
    phi = VOLTAGE + zero
    with torch.no_grad():
        if magnetic is not None:
            b, db = magnetic.derivative(0, z), magnetic.derivative(1, z)
        if electrostatic is not None:
            phi = phi + electrostatic.derivative(0, z)
            dphi, d2phi = electrostatic.derivative(1, z), electrostatic.derivative(2, z)
    phi_hat, gamma = relativistic_potential(phi), lorentz_factor(phi)
    s = ETA / phi_hat.sqrt()
    damping, lens = gamma * dphi / (2 * phi_hat), gamma * d2phi / (4 * phi_hat)
    return torch.stack([s * b, s * db, damping, lens], dim=-1)


@dataclass
class CoefficientTable:
    """``coefficients`` on the half-step grid start + k h/2, k = 0 .. 2n."""

    start: float
    step: float  # the RK4 step h
    n_steps: int
    values: torch.Tensor  # (2n + 1, 4)

    @classmethod
    def build(cls, z_start: float, z_end: float, dz: float, **fields) -> Self:
        n = max(1, math.ceil((z_end - z_start) / dz))
        h = (z_end - z_start) / n
        grid = z_start + (h / 2) * torch.arange(2 * n + 1, dtype=DTYPE)
        return cls(z_start, h, n, coefficients(grid, **fields))

    def index(self, z: float) -> int:
        k = (float(z) - self.start) / (self.step / 2)
        if abs(k - round(k)) > 1e-6:
            raise ValueError(f"z = {float(z):.6g} m is not on the half-step grid")
        return round(k)

    def at(self, z: float) -> torch.Tensor:
        return self.values[self.index(z)]


def lab_frame(at, B0):
    """d/dz [x, x', y, y', delta] in the lab frame, where the Larmor rotation shows as x-y
    coupling. ``at(z)`` returns (S, S', D, L) at z."""

    def f(z, state):
        S, dS, D, L = at(z).unbind(-1)
        x, xp, y, yp = state[..., X], state[..., XP], state[..., Y], state[..., YP]
        xpp = -D * xp - L * x - B0 * (S * yp + dS * y / 2)
        ypp = -D * yp - L * y + B0 * (S * xp + dS * x / 2)
        return state_derivative(xp, xpp, yp, ypp)

    return f


def larmor_radius(at, B0):
    """d/dz [r, r'] in the frame rotating with the Larmor angle, where x and y decouple. B0 may
    be a tensor of shape (M,), traced as M lenses at once with state (M, 2)."""

    def f(z, state):
        S, _, D, L = at(z).unbind(-1)
        r, rp = state[..., 0], state[..., 1]
        return torch.stack([rp, -D * rp - (L + (B0 * S) ** 2 / 4) * r], dim=-1)

    return f


def integrate(f, state, table: CoefficientTable, n_steps=None, record=False):
    """Fixed-step RK4 from the table's start, with the package's step."""
    n_steps = table.n_steps if n_steps is None else n_steps
    states = [state]
    for i in range(n_steps):
        state = rk4_step(f, table.start + i * table.step, state, table.step)
        if record:
            states.append(state)
    return torch.stack(states) if record else state


def focusing_excitation(table, z_focus, low=0.0, high=0.5, rounds=6, batch=64) -> float:
    """The B0 [T] at which a parallel beam first crosses the axis at ``z_focus``.

    Each round traces ``batch`` values of B0 across the bracket at once, keeps the neighbours
    around the first one whose ray has crossed by ``z_focus``, and repeats on that bracket.
    """
    k = table.index(z_focus)
    if k % 2:
        raise ValueError("z_focus must fall on a full RK4 step")
    for _ in range(rounds):
        B0 = torch.linspace(low, high, batch, dtype=DTYPE)
        start = torch.zeros(batch, 2, dtype=DTYPE)
        start[:, 0] = 1.0
        r = integrate(larmor_radius(table.at, B0), start, table, n_steps=k // 2)[:, 0]
        crossed = torch.nonzero(r <= 0)
        if len(crossed) == 0 or int(crossed[0]) == 0:
            raise ValueError(f"the focus at z_focus needs a B0 outside [{low}, {high}] T")
        first = int(crossed[0])
        low, high = float(B0[first - 1]), float(B0[first])
    return (low + high) / 2


def timed(fn):
    start = time.perf_counter()
    out = fn()
    return time.perf_counter() - start, out


t_build, table = timed(lambda: CoefficientTable.build(Z_START, Z_END, DZ))
t_focus, B0 = timed(lambda: focusing_excitation(table, FOCUS))
print(
    f"{table.n_steps} RK4 steps; table of {len(table.values)} rows built in {t_build:.3f} s;"
    f" B0 = {B0:.6f} T found in {t_focus:.1f} s"
)

# A bundle in the x-z plane, x from -1 to 1 mm, which the lens rotates out of that plane. The
# field at the start is still 1% of its peak, so a ray parallel in the lab frame would carry
# angular momentum and miss the axis. Give each ray the slope of one that entered from
# field-free space instead: it turns at the Larmor rate, theta' = B0 S / 2.
state = Rays.from_parallel(1e-3, 9).state.clone()
state[:, YP] = B0 * table.values[0, 0] / 2 * state[:, X]
rays = Rays.from_state(state, z=Z_START, voltage=VOLTAGE)
t_table, trace = timed(lambda: integrate(lab_frame(table.at, B0), rays.state, table, record=True))
at_focus = trace[table.index(FOCUS) // 2]
spot = float(torch.hypot(at_focus[:, X], at_focus[:, Y]).max())
print(f"largest |r| at z = {FOCUS * 1e3:g} mm: {spot:.1e} m, from a 1 mm bundle")


def on_the_fly(z):
    return coefficients(torch.as_tensor(z, dtype=DTYPE))


t_fly, fly = timed(lambda: integrate(lab_frame(on_the_fly, B0), rays.state, table))
print(
    f"tabulated: {t_table:.2f} s; on the fly: {t_fly:.2f} s;"
    f" largest difference in the final state {float((fly - trace[-1]).abs().max()):.1e}"
)

package = ODESolver.paraxial(IntegratorParams.RK4(dz=DZ))
checks = (
    ("Glaser", FieldLens.glaser(B0=B0, a=A, z=0.0), dict(electrostatic=None)),
    ("Schiske", FieldLens(ELECTROSTATIC, z=0.0), dict(magnetic=None)),
)
for name, lens, only in checks:
    alone = CoefficientTable.build(Z_START, Z_END, DZ, **only)
    ours = integrate(lab_frame(alone.at, B0), rays.state, alone)
    with torch.no_grad():
        theirs = package.trace(OpticalSystem.from_components(lens), rays).final.state
    print(f"{name} alone vs ODESolver.paraxial: {float((ours - theirs).abs().max()):.1e}")
