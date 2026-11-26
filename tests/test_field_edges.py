"""Field components end smoothly, and the Larmor angle is the integral of their summed field."""

import math

import pytest
import torch

from em_ray_tracer import FieldLens, ODESolver, OpticalSystem, Plane, Rays
from em_ray_tracer.constants import DTYPE, ETA, relativistic_potential
from em_ray_tracer.fields.taper import SMOOTHNESS, smoothstep
from em_ray_tracer.solvers.params import IntegratorParams

VOLTAGE = 1e4
GLASER = dict(B0=0.05, a=4e-3)
RK4 = IntegratorParams.RK4(dz=10e-6)


# --- the window ---


def test_window_is_a_smooth_step():
    t = torch.tensor([-0.5, 0.0, 0.5, 1.0, 1.5], dtype=DTYPE)
    torch.testing.assert_close(smoothstep(0, t), torch.tensor([0, 0, 0.5, 1, 1], dtype=DTYPE))
    ends = torch.tensor([0.0, 1.0], dtype=DTYPE)
    for m in range(1, SMOOTHNESS + 1):
        assert smoothstep(m, ends).abs().max() == 0, m


def test_window_derivatives_match_autograd_everywhere():
    # includes t = 1/2, where the two halves of S meet
    t = torch.linspace(0.0, 1.0, 41, dtype=DTYPE, requires_grad=True)
    expected = smoothstep(0, t)
    for m in range(1, 9):
        (expected,) = torch.autograd.grad(expected.sum(), t, create_graph=True)
        scale = float(expected.detach().abs().max())
        torch.testing.assert_close(smoothstep(m, t), expected, rtol=1e-9, atol=1e-9 * scale)


@pytest.mark.parametrize("n", range(1, 8))
def test_tapered_derivatives_match_autograd(n):
    lens = FieldLens.glaser(**GLASER, z=0.01)
    z = torch.linspace(0.01 + 0.02, 0.01 + 0.0399, 23, dtype=DTYPE, requires_grad=True)
    expected = lens.axial_derivative(0, z)["magnetic"]
    for _ in range(n):
        (expected,) = torch.autograd.grad(expected.sum(), z, create_graph=True)
    got = lens.axial_derivative(n, z)["magnetic"]
    scale = float(expected.detach().abs().max())
    torch.testing.assert_close(got, expected, rtol=1e-8, atol=1e-8 * scale)


def test_profile_is_untouched_in_the_core_and_zero_at_the_extent():
    lens = FieldLens.glaser(**GLASER, z=0.0)
    core = (1 - lens.taper) * lens.field.half_extent
    z = torch.linspace(-core, core, 101, dtype=DTYPE)
    torch.testing.assert_close(
        lens.axial_derivative(0, z)["magnetic"],
        lens.field.derivative(0, z)["magnetic"],
        rtol=0,
        atol=0,
    )
    ends = torch.tensor([lens.z_start, lens.z_end], dtype=DTYPE)
    assert lens.axial_derivative(0, ends)["magnetic"].abs().max() == 0


# --- why: a cut-off field gives rays a spurious rotation ---


@pytest.mark.parametrize("order", [None, 3])
@pytest.mark.parametrize("taper, meridional", [(0.4, True), (0.0, False)])
def test_rays_from_the_axis_leave_a_lens_without_angular_momentum(taper, meridional, order):
    """A ray from a point on the axis has no angular momentum about it, so once it is back in
    field-free space, x y' - y x' = 0. A field cut off at 1% of its peak breaks that."""
    lens = FieldLens.glaser(**GLASER, z=0.0, taper=taper)
    system = OpticalSystem.from_components(lens, Plane(z=0.06, name="out"))
    rays = Rays.from_state(torch.tensor([[0, 1e-3, 0, 0]], dtype=DTYPE), z=-0.06, voltage=VOLTAGE)
    solver = ODESolver.paraxial(RK4) if order is None else ODESolver.nonparaxial(order, RK4)
    with torch.no_grad():
        out = solver.trace(system, rays).planes["out"]
    skew = float(
        (out.x * out.yp - out.y * out.xp).abs()
        / (torch.hypot(out.x, out.y) * torch.hypot(out.xp, out.yp))
    )
    if meridional:
        assert skew < 1e-10
    else:
        assert skew > 1e-3


# --- the Larmor angle ---


def test_larmor_angle_matches_the_closed_form():
    lens = FieldLens.glaser(**GLASER, z=0.0, taper=0.0)
    s = ETA / math.sqrt(relativistic_potential(VOLTAGE))
    a, b0 = GLASER["a"], GLASER["B0"]
    expected = s * b0 * a / 2 * (math.atan(10) - math.atan(-10))
    theta = OpticalSystem.from_components(lens).larmor_angle(lens.z_start, lens.z_end, VOLTAGE)
    assert float(theta) == pytest.approx(expected, rel=1e-12)


def test_larmor_angle_ignores_thin_elements_and_is_not_wrapped():
    """Thin elements inside and between fields split the spans but add no rotation; a strong
    lens turns the image past pi, and the angle keeps it."""
    lenses = [FieldLens.glaser(B0=0.4, a=4e-3, z=z) for z in (0.0, 0.05)]
    bare = OpticalSystem.from_components(*lenses)
    planes = [Plane(z=z, name=f"P{i}") for i, z in enumerate((-0.01, 0.0, 0.02, 0.025, 0.051))]
    busy = OpticalSystem.from_components(*lenses, *planes)
    theta = bare.larmor_angle(-0.04, 0.09, VOLTAGE)
    assert float(theta) > math.pi
    # equal up to the quadrature error of splitting the spans differently
    torch.testing.assert_close(busy.larmor_angle(-0.04, 0.09, VOLTAGE), theta, rtol=1e-9, atol=0)


def test_larmor_angle_is_the_image_rotation_of_a_traced_ray():
    lenses = [FieldLens.glaser(B0=0.4, a=4e-3, z=z) for z in (0.0, 0.05)]
    system = OpticalSystem.from_components(*lenses, Plane(z=0.09, name="out"))
    ray = Rays.from_state(torch.tensor([[0, 1e-5, 0, 0]], dtype=DTYPE), z=-0.04, voltage=VOLTAGE)
    out = ODESolver.paraxial(RK4).trace(system, ray).planes["out"]
    # zero angular momentum: the slope's azimuth is the Larmor angle, modulo the crossings (pi)
    azimuth = math.atan2(float(out.yp[0]), float(out.xp[0]))
    theta = float(system.larmor_angle(-0.04, 0.09, VOLTAGE))
    difference = (azimuth - theta + math.pi / 2) % math.pi - math.pi / 2
    assert abs(difference) < 1e-8
