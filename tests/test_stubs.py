"""The tests the student's field and ODE work has to pass.

Each one is marked xfail(strict=True, raises=NotImplementedError). They fail now because
the stubs raise. Once a stub is implemented, a passing test turns into an XPASS, which
strict mode reports as a failure: that is the cue to delete the marker. The expected values
come from the em-widgets kit tests (test/paraxial.test.js, test/nonparaxial.test.js), made
relativistic.
"""

import math

import pytest
import torch

from em_ray_tracer import FieldLens, ODESolver, OpticalSystem, Plane, Rays, TransferMatrixSolver
from em_ray_tracer.constants import ETA, relativistic_potential
from em_ray_tracer.fields import GlaserField, LaplaceExpansion

stub = pytest.mark.xfail(raises=NotImplementedError, strict=True, reason="stub")

GLASER = dict(B0=0.01, a=0.0075)
VOLTAGE = 1000.0


def glaser_exact(B0, a, voltage, z0, r0, rp0):
    """Closed-form paraxial ray through a Glaser field: with z = a cot(theta) and
    r = u / sin(theta), u'' + omega^2 u = 0, omega^2 = 1 + eta^2 B0^2 a^2 / (4 phi_hat)."""
    K = ETA**2 * B0**2 / (4 * relativistic_potential(voltage))
    omega = math.sqrt(1 + K * a * a)
    t0 = math.atan2(a, z0)
    u0 = r0 * math.sin(t0)
    du0 = -a * rp0 / math.sin(t0) + r0 * math.cos(t0)

    def r(z):
        t = math.atan2(a, z)
        u = u0 * math.cos(omega * (t - t0)) + du0 / omega * math.sin(omega * (t - t0))
        return u / math.sin(t)

    return r


@stub
def test_glaser_derivatives_match_finite_differences():
    field = GlaserField(**GLASER)
    z = torch.linspace(-0.05, 0.05, 101, dtype=torch.float64)
    h = 1e-6
    fd = (field.derivative(0, z + h) - field.derivative(0, z - h)) / (2 * h)
    torch.testing.assert_close(field.derivative(1, z), fd, rtol=1e-6, atol=1e-9)


@stub
def test_paraxial_glaser_matches_closed_form():
    # compare in the rotating frame: the radius of an x-plane ray is rotation-invariant
    lens = FieldLens.glaser(**GLASER, z=0.0)
    system = OpticalSystem.from_components(lens, Plane(z=0.1, name="out"))
    z0 = float(lens.z_start)
    rays = Rays.from_state(torch.tensor([[1e-3, 0, 0, 0], [0, 1e-3, 0, 0]]), z=z0, voltage=VOLTAGE)
    out = ODESolver.paraxial().trace(system, rays).planes["out"]
    for i, (r0, rp0) in enumerate([(1e-3, 0.0), (0.0, 1e-3)]):
        exact = glaser_exact(**GLASER, voltage=VOLTAGE, z0=z0, r0=r0, rp0=rp0)(0.1)
        assert torch.hypot(out.x[i], out.y[i]).item() == pytest.approx(abs(exact), rel=1e-8)


@stub
def test_weak_glaser_lens_approaches_thin_lens_power():
    # 1/f -> eta^2 B0^2 pi a / (8 phi_hat) as B0 -> 0
    a, B0 = 0.0075, 1e-3
    lens = FieldLens.glaser(B0=B0, a=a, z=0.0)
    system = OpticalSystem.from_components(lens)
    expected = ETA**2 * B0**2 * math.pi * a / (8 * relativistic_potential(VOLTAGE))
    focal = system.focal_length(voltage=VOLTAGE)
    assert 1 / focal.abs().item() == pytest.approx(expected, rel=0.02)


@stub
def test_thick_lens_matrix_matches_ode_trace():
    lens = FieldLens.glaser(**GLASER, z=0.0)
    system = OpticalSystem.from_components(lens, Plane(z=0.1, name="out"))
    rays = Rays.from_cone(1e-3, z=float(lens.z_start), voltage=VOLTAGE)
    tm = TransferMatrixSolver().trace(system, rays).final.state
    ode = ODESolver.paraxial().trace(system, rays).final.state
    torch.testing.assert_close(tm, ode, rtol=1e-8, atol=1e-12)


@stub
def test_laplace_expansion_divergence_identity():
    # truncating at N leaves div F = (-1)^N / (2^(2N) (N!)^2) r^(2N) F^(2N+1)(z)
    field = GlaserField(**GLASER)
    expansion = LaplaceExpansion(field, order=3)
    x = torch.tensor([1e-3], dtype=torch.float64, requires_grad=True)
    y = torch.tensor([5e-4], dtype=torch.float64, requires_grad=True)
    z = torch.tensor([2e-3], dtype=torch.float64, requires_grad=True)
    _, B = expansion.evaluate(x, y, z)
    div = sum(
        torch.autograd.grad(B[..., i].sum(), v, create_graph=True)[0]
        for i, v in enumerate((x, y, z))
    )
    r2 = x**2 + y**2
    expected = -1 / (2**6 * 36) * r2**3 * field.derivative(7, z)
    torch.testing.assert_close(div, expected, rtol=1e-6, atol=0)
