"""How much a non-paraxial trace gains from precomputing its axial field on RK4's grid.

The ray equations need the axial field only as a function of z, and fixed-step RK4 asks for it
at points known before the loop starts. A step of size h from z_i calls the right-hand side at
z_i, twice at z_i + h/2 (k2 and k3 differ in the state, not in z), and at z_i + h, where the
next step starts. Over n steps that is 4n calls but only 2n + 1 distinct points, z_0 + k h/2.

So every axial derivative can be evaluated once on that grid, in one batched call, and the
right-hand side can index the result. The values are the ones the field would have returned
at the same z, so the trace is unchanged to rounding. Most of the gain is not the factor of
two: evaluating a field at one z is almost all fixed cost (Python calls and small tensor
operations, and for a spline the knot search and recursion), so one call on the whole grid
costs about as much as a handful of single-point calls.

The script traces one bundle through a sampled Glaser + Schiske lens at third order twice:
once with ``NonParaxialEquations`` evaluating the spline at every call, and once with the same
equations reading ``TabulatedField``, which has the axial field's ``derivative`` interface and
indexes the half-step grid. It prints the cost of one derivative at one point and on the whole
grid, both trace times and the largest difference between the traces. At the default 10 um
step the on-the-fly trace takes close to a minute. ``field_table_tracer.py`` builds a paraxial
tracer around the same idea.

Run: uv run python examples/field_table_timing.py [dz]
"""

import math
import sys
import time

import torch

from em_ray_tracer import FieldLens, OpticalSystem, Rays
from em_ray_tracer.constants import DTYPE
from em_ray_tracer.fields import SampledAxialField
from em_ray_tracer.solvers.ode import ODESolver
from em_ray_tracer.solvers.params import IntegratorParams

VOLTAGE = 1e4  # beam voltage [V]
ORDER = 3  # expansion order of the non-paraxial equations
DZ = float(sys.argv[1]) if len(sys.argv) > 1 else 10e-6  # RK4 step [m]
KINDS = ("magnetic", "electrostatic")

# a Glaser and a Schiske profile, sampled on 10^4 points and fitted with a spline
glaser = FieldLens.glaser(B0=0.1, a=4e-3, z=0.03)
schiske = FieldLens.schiske(k=0.8, a=4e-3, voltage=VOLTAGE, z=-0.02)
z_samples = torch.linspace(-60e-3, 60e-3, 10**4, dtype=DTYPE)
with torch.no_grad():
    samples = torch.stack(
        [glaser.axial_derivative(0, z_samples), schiske.axial_derivative(0, z_samples)]
    )
field = SampledAxialField(
    z_samples, samples, KINDS, grounded_check=False, extrapolate="clamp", window_size=10, order=7
)
lens = FieldLens(field, z=0.06)


def half_step_grid(field, n_steps: int) -> torch.Tensor:
    """The 2n + 1 points where n RK4 steps across the field evaluate it, in its own coordinate."""
    h = 2 * float(field.half_extent) / n_steps
    return -float(field.half_extent) + (h / 2) * torch.arange(2 * n_steps + 1, dtype=DTYPE)


class TabulatedField:
    """An axial field's derivatives on the half-step grid, behind the field's own interface.

    ``derivative(n, z, kind_requested)`` returns the stored value at the grid point z and raises
    anywhere else: the table serves fixed-step RK4 only, where every z is known in advance.
    """

    def __init__(self, field, n_steps: int, max_derivative: int):
        self.kind, self.half_extent = field.kind, field.half_extent
        self.start = -float(field.half_extent)
        self.spacing = float(field.half_extent) / n_steps  # h / 2
        grid = half_step_grid(field, n_steps)
        with torch.no_grad():
            self.values = {
                (n, kind): field.derivative(n, grid, kind_requested=kind)
                for n in range(max_derivative + 1)
                for kind in KINDS
            }

    def derivative(self, n: int, z, kind_requested=None) -> torch.Tensor:
        k = (z - self.start) / self.spacing
        idx=torch.round(k)
        for i,ki in enumerate(k):
            if abs(round(float(ki))-ki) > 1e-8:
                raise ValueError(f"z = {float(z[i]):.6g} m is not on the half-step grid")
        
        return self.values[n, kind_requested][idx.int()]


def per_call(fn, repeat: int) -> float:
    fn()  # warm-up
    start = time.perf_counter()
    for _ in range(repeat):
        fn()
    return (time.perf_counter() - start) / repeat


def timed(fn):
    start = time.perf_counter()
    out = fn()
    return time.perf_counter() - start, out


# the same step count as ODESolver.integrate
n_steps = max(1, math.ceil(float(lens.z_end - lens.z_start) / DZ))
grid = half_step_grid(field, n_steps)
print(f"{n_steps} RK4 steps: {4 * n_steps} right-hand-side calls at {len(grid)} distinct z")

with torch.no_grad():
    point = torch.tensor(0.01, dtype=DTYPE)
    one = per_call(lambda: field.derivative(3, point, kind_requested="magnetic"), 200)
    whole = per_call(lambda: field.derivative(3, grid, kind_requested="magnetic"), 5)
print(
    f"B''' at one point: {one * 1e3:.3f} ms; on all {len(grid)} points: {whole * 1e3:.3f} ms"
    f" (the cost of {whole / one:.0f} single-point calls)"
)

rays = Rays.from_parallel(1e-3, 8, z=float(lens.z_start), voltage=VOLTAGE)
solver = ODESolver.nonparaxial(ORDER, IntegratorParams.RK4(dz=DZ))
with torch.no_grad():
    t_fly, fly = timed(lambda: solver.trace(OpticalSystem.from_components(lens), rays))
    # LaplaceExpansion at order N needs derivatives up to N + 1 (of phi)
    t_build, table = timed(lambda: TabulatedField(field, n_steps, ORDER + 1))
    tabulated = FieldLens(table, z=float(lens.z))
    t_table, tab = timed(lambda: solver.trace(OpticalSystem.from_components(tabulated), rays))

print(f"on the fly: {t_fly:6.2f} s")
print(f"tabulated:  {t_table:6.2f} s, plus {t_build:.2f} s to build the table")
difference = float((fly.final.state - tab.final.state).abs().max())
print(f"largest difference in the final state: {difference:.1e}")
