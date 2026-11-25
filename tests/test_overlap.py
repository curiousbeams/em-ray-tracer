"""Overlapping components: superposed fields, thin elements inside fields, split deflectors."""

import math

import pytest
import torch

from em_ray_tracer import (
    Aperture,
    DoubleDeflector,
    FieldLens,
    ODESolver,
    OpticalSystem,
    Plane,
    Rays,
    ThinLens,
    TransferMatrixSolver,
)
from em_ray_tracer.constants import DTYPE, ETA, relativistic_potential
from em_ray_tracer.solvers.params import IntegratorParams
from em_ray_tracer.system import Element, FieldSpan, Gap
from em_ray_tracer.transfer_map import deflection_map, drift_map
from tests.test_stubs import glaser_exact
from tests.test_utilities.glaser_reference import GlaserLens, glaser_c30

VOLTAGE = 1e4
GLASER = dict(B0=0.05, a=4e-3)
RK4 = IntegratorParams.RK4(dz=10e-6)


def solvers():
    return {
        "paraxial": ODESolver.paraxial(RK4),
        "nonparaxial": ODESolver.nonparaxial(3, RK4),
        "transfer_matrix": TransferMatrixSolver(),
    }


def cone(z=-0.05):
    return Rays.from_cone(semiangle=4e-3, n_azimuthal=6, n_shells=3, z=z, voltage=VOLTAGE)


def final(solver, *components, rays=None):
    system = OpticalSystem.from_components(*components, Plane(z=0.1, name="out"))
    with torch.no_grad():
        return solver.trace(system, cone() if rays is None else rays).final.state


# --- the schedule ---


def test_schedule_superposes_overlapping_fields_and_splits_at_thin_elements():
    a = FieldLens.glaser(**GLASER, z=0.0, name="A")
    b = FieldLens.glaser(**GLASER, z=0.02, name="B")
    system = OpticalSystem.from_components(a, b, Plane(z=0.01, name="P"))
    kinds = []
    for seg in system.segments(-0.06, 0.08):
        match seg:
            case Gap():
                kinds.append("gap")
            case Element(c):
                kinds.append(c.name)
            case FieldSpan(z0, z1, comps):
                if float(z1 - z0) > 0:
                    kinds.append("+".join(c.name for c in comps))
    assert kinds == ["gap", "A", "A+B", "P", "A+B", "B", "gap"]


def test_fields_reaching_past_the_span_are_clipped():
    lens = FieldLens.glaser(**GLASER, z=0.0)
    (span,) = [
        s
        for s in OpticalSystem.from_components(lens).segments(-0.01, 0.01)
        if isinstance(s, FieldSpan)
    ]
    assert (float(span.z0), float(span.z1)) == (-0.01, 0.01)


# --- superposition ---


@pytest.mark.parametrize("name", ["paraxial", "nonparaxial", "transfer_matrix"])
def test_two_half_lenses_at_one_place_equal_the_whole(name):
    solver = solvers()[name]
    half = dict(GLASER, B0=GLASER["B0"] / 2)
    whole = final(solver, FieldLens.glaser(**GLASER, z=0.0))
    halves = final(solver, FieldLens.glaser(**half, z=0.0), FieldLens.glaser(**half, z=0.0))
    torch.testing.assert_close(halves, whole, rtol=1e-12, atol=1e-18)


@pytest.mark.parametrize("name", ["paraxial", "nonparaxial"])
def test_glaser_plus_schiske_equals_one_profile_with_both(name):
    B0, a, k = GLASER["B0"], GLASER["a"], 0.6

    def both(z):
        bell = 1 / (1 + (z / a) ** 2)
        return torch.stack([B0 * bell, -VOLTAGE * k**2 * bell])

    combined = FieldLens.from_axial_analytical(
        axial=both, fieldkind=("magnetic", "electrostatic"), z=0.0, half_extent=10 * a
    )
    separate = (
        FieldLens.glaser(**GLASER, z=0.0),
        FieldLens.schiske(k=k, a=a, voltage=VOLTAGE, z=0.0),
    )
    solver = solvers()[name]
    torch.testing.assert_close(
        final(solver, *separate), final(solver, combined), rtol=1e-10, atol=1e-16
    )


def test_overlapping_tails_trace_alike_under_both_solvers():
    lenses = (FieldLens.glaser(**GLASER, z=0.0), FieldLens.glaser(**GLASER, z=0.012))
    ode = final(ODESolver.paraxial(), *lenses)
    tm = final(TransferMatrixSolver(), *lenses)
    torch.testing.assert_close(tm, ode, rtol=1e-10, atol=1e-16)


def test_superposed_misaligned_fields_raise_under_transfer_matrices():
    lenses = (
        FieldLens.glaser(**GLASER, z=0.0, shift=(1e-4, 0.0)),
        FieldLens.glaser(**GLASER, z=0.012),
    )
    with pytest.raises(NotImplementedError, match="aligned"):
        final(TransferMatrixSolver(), *lenses)


# --- thin elements inside fields ---


def test_plane_inside_a_lens_records_the_closed_form_ray():
    lens = FieldLens.glaser(**GLASER, z=0.0)
    z0, zp = float(lens.z_start), 3e-3
    system = OpticalSystem.from_components(lens, Plane(z=zp, name="inside"))
    state = torch.tensor([[1e-3, 0, 0, 0], [1e-3, -1e-3, 5e-3, 1e-3]], dtype=DTYPE)
    rays = Rays.from_state(state, z=z0, voltage=VOLTAGE)
    at = ODESolver.paraxial(RK4).trace(system, rays).planes["inside"]
    # the closed form works in the Larmor frame; see test_paraxial_glaser_matches_closed_form
    s = ETA / math.sqrt(relativistic_potential(VOLTAGE))
    rate = s * GLASER["B0"] / (1 + (z0 / GLASER["a"]) ** 2) / 2
    for i, (x0, xp0, y0, yp0) in enumerate(state.tolist()):
        w0, wp0 = complex(x0, y0), complex(xp0, yp0)
        exact = glaser_exact(**GLASER, voltage=VOLTAGE, z0=z0, r0=w0, rp0=wp0 - 1j * rate * w0)
        r = torch.hypot(at.x[i], at.y[i]).item()
        assert r == pytest.approx(abs(exact(zp)), rel=1e-9)


@pytest.mark.parametrize("name", ["paraxial", "nonparaxial", "transfer_matrix"])
def test_a_plane_inside_a_lens_does_not_change_the_trace(name):
    solver = solvers()[name]
    lens = FieldLens.glaser(**GLASER, z=0.0)
    without = final(solver, lens)
    with_plane = final(solver, lens, Plane(z=1.234e-3, name="inside"))
    torch.testing.assert_close(with_plane, without, rtol=1e-9, atol=1e-15)


def test_continuing_from_a_plane_inside_a_lens_equals_one_trace():
    solver = ODESolver.nonparaxial(3, RK4)
    lens = FieldLens.glaser(**GLASER, z=0.0)
    system = OpticalSystem.from_components(
        lens, Plane(z=2e-3, name="inside"), Plane(z=0.1, name="out")
    )
    with torch.no_grad():
        full = solver.trace(system, cone())
        rest = solver.trace(system, full.planes["inside"]).final  # starts inside the field
    torch.testing.assert_close(rest.state, full.final.state, rtol=0, atol=0)


def test_aperture_inside_a_lens_clips_where_it_is():
    lens = FieldLens.glaser(**GLASER, z=0.0)
    rays = Rays.from_parallel(2e-3, 21, z=float(lens.z_start), voltage=VOLTAGE)
    solver = ODESolver.paraxial(RK4)
    probe = OpticalSystem.from_components(lens, Plane(z=0.0, name="centre"))
    radius_at_centre = solver.trace(probe, rays).planes["centre"]
    r = torch.hypot(radius_at_centre.x, radius_at_centre.y)
    clip = float(r.median())
    system = OpticalSystem.from_components(lens, Aperture(radius=clip, z=0.0))
    alive = solver.trace(system, rays).final.alive
    torch.testing.assert_close(alive, r <= clip)
    assert 0 < int(alive.sum()) < len(alive)


# --- gradients: a learnable plane inside a lens (finding the image inside the field) ---


def _image_inside_lens():
    """The C30 geometry: the first image of a point object lies inside the Glaser field."""
    B0, a, zc = 1.0, 1e-3, 60e-3
    ref = glaser_c30(GlaserLens(B0, a, zc, 0.0, 2 * zc), VOLTAGE, 0.0)
    lens = FieldLens.glaser(B0=B0, a=a, z=zc, extent=60)
    rays = Rays.from_cone(semiangle=1e-3, n_azimuthal=8, n_shells=3, voltage=VOLTAGE)
    return lens, rays, ref["z_img"]


def _spread(solver, system, rays, plane, alpha=1e-3):
    """RMS spot at ``plane``, over alpha^2. Tracing stops at the plane: the field beyond it is
    clipped off the span, which halves the work."""
    at = solver.trace(system, rays, z_end=plane.z).planes["image"]
    return (at.x.var() + at.y.var()) / alpha**2


IMAGE_RK4 = IntegratorParams.RK4(dz=20e-6)  # the image comes out within 1e-10 m


def test_plane_position_gradient_inside_a_lens_matches_finite_differences():
    lens, rays, z_img = _image_inside_lens()
    solver = ODESolver.paraxial(IMAGE_RK4)

    def spread(z):
        plane = Plane(z=z, name="image").learn("z")
        system = OpticalSystem.from_components(lens, plane)
        return _spread(solver, system, rays, plane), plane.z

    value, z = spread(z_img - 0.5e-3)
    (grad,) = torch.autograd.grad(value, z)
    h = 1e-7
    with torch.no_grad():
        fd = (spread(z_img - 0.5e-3 + h)[0] - spread(z_img - 0.5e-3 - h)[0]) / (2 * h)
    assert float(grad) == pytest.approx(float(fd), rel=1e-6)


def test_optimizing_a_plane_inside_a_lens_finds_the_gaussian_image():
    lens, rays, z_img = _image_inside_lens()
    plane = Plane(z=z_img - 0.5e-3, name="image").learn("z")
    system = OpticalSystem.from_components(lens, plane)
    solver = ODESolver.paraxial(IMAGE_RK4)
    optimizer = torch.optim.LBFGS(
        [plane.z],
        max_iter=50,
        tolerance_grad=1e-14,
        tolerance_change=1e-20,
        line_search_fn="strong_wolfe",
    )

    def closure():
        optimizer.zero_grad()
        loss = _spread(solver, system, rays, plane)
        loss.backward()
        return loss

    optimizer.step(closure)
    assert float(plane.z.detach()) == pytest.approx(z_img, abs=1e-9)


# --- double deflectors ---


def test_double_deflector_around_a_thin_lens_is_kick_drift_lens_drift_kick():
    dd = DoubleDeflector((2e-3, -1e-3), z=0.0, z_2=0.04, pivot=0.1, beam_shift=(1e-5, 0.0))
    lens = ThinLens.from_focal_length(0.05, z=0.01)
    system = OpticalSystem.from_components(dd, lens)
    rays = cone(z=0.0)
    out = TransferMatrixSolver().trace(system, rays).final.state
    (a1, a2) = dd.excitations
    expected = (
        deflection_map(*a2)
        @ drift_map(0.03)
        @ lens.transfer_map()
        @ drift_map(0.01)
        @ deflection_map(*a1)
    )
    torch.testing.assert_close(out, expected.apply(rays.state), rtol=1e-12, atol=1e-18)


def test_double_deflector_inside_a_lens_field():
    dd = DoubleDeflector((2e-3, -1e-3), z=-0.01, z_2=0.004, pivot=0.1).learn("tilt")
    lens = FieldLens.glaser(**GLASER, z=0.0)
    ode = final(ODESolver.paraxial(), dd, lens)
    tm = final(TransferMatrixSolver(), dd, lens)
    torch.testing.assert_close(tm, ode, rtol=1e-10, atol=1e-16)
    system = OpticalSystem.from_components(dd, lens, Plane(z=0.1, name="out"))
    ODESolver.paraxial().trace(system, cone()).final.x.sum().backward()
    assert dd.tilt.grad is not None and dd.tilt.grad.abs().sum() > 0


# --- the module graph stays a tree ---


def test_system_is_a_plain_module_tree_after_a_nonparaxial_trace():
    system = OpticalSystem.from_components(FieldLens.glaser(**GLASER, z=0.0))
    with torch.no_grad():
        ODESolver.nonparaxial(3).trace(system, cone())
    repr(system)
    system.state_dict()
    system.to(DTYPE)
