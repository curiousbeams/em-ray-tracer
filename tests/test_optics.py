"""First-order optics: transfer maps, imaging, deflection, misalignment, apertures."""

import math

import pytest
import torch

from em_ray_tracer import (
    Aperture,
    Biprism,
    Deflector,
    DoubleDeflector,
    ODESolver,
    OpticalSystem,
    Plane,
    Quadrupole,
    Rays,
    RayTracer,
    ThinLens,
    TransferMap,
    TransferMatrixSolver,
)
from em_ray_tracer.components.thin import dispersion_coefficient
from em_ray_tracer.constants import lorentz_factor
from em_ray_tracer.transfer_map import drift_map, thin_lens_map

torch.manual_seed(0)


def random_rays(n=20, scale=1e-3):
    return Rays.from_state(torch.randn(n, 4, dtype=torch.float64) * scale)


def test_drift_lens_drift_is_the_matrix_product():
    system = OpticalSystem.from_components(ThinLens(power=0.5, z=3.0))
    m = system.transfer_map(0.0, 9.0)
    expected = drift_map(6.0).matrix @ thin_lens_map(0.5).matrix @ drift_map(3.0).matrix
    torch.testing.assert_close(m.matrix, expected)


def test_thin_lens_imaging_equation():
    # u = 3, f = 2  ->  v = 6, M = -v/u = -2
    system = OpticalSystem.from_components(ThinLens.from_focal_length(2.0, z=3.0))
    z_image, magnification = system.gaussian_image(0.0)
    torch.testing.assert_close(z_image, torch.tensor(9.0, dtype=torch.float64))
    torch.testing.assert_close(magnification, torch.tensor(-2.0 + 0j, dtype=torch.complex128))

    # and the traced rays from one object point meet at that image point
    rays = Rays.from_cone(semiangle=1e-2, origin=(1e-4, -2e-4))
    trace = TransferMatrixSolver().trace(system, rays, z_end=z_image)
    torch.testing.assert_close(trace.final.x, torch.full_like(trace.final.x, -2e-4))
    torch.testing.assert_close(trace.final.y, torch.full_like(trace.final.y, 4e-4))


def test_batched_trace_matches_per_ray():
    system = OpticalSystem.from_components(
        ThinLens(power=0.3, z=1.0, rotation=0.2), Quadrupole(power=0.05, z=2.0, orientation=0.4)
    )
    rays = random_rays()
    batched = TransferMatrixSolver().trace(system, rays, z_end=4.0).final.state
    single = torch.cat(
        [
            TransferMatrixSolver()
            .trace(system, Rays(rays.state[i : i + 1]), z_end=4.0)
            .final.state
            for i in range(rays.num_rays)
        ]
    )
    torch.testing.assert_close(batched, single)


def test_trace_map_equals_system_map():
    system = OpticalSystem.from_components(
        ThinLens(power=0.3, z=1.0), Deflector((1e-3, 0.0), z=1.5), Aperture(1.0, z=2.0)
    )
    trace = TransferMatrixSolver().trace(system, random_rays(), z_end=4.0)
    torch.testing.assert_close(trace.transfer_map.matrix, system.transfer_map(0.0, 4.0).matrix)
    torch.testing.assert_close(trace.final.state, trace.transfer_map.apply(trace.initial.state))


def test_deflector_then_drift_shifts_every_ray():
    alpha, L = 2e-3, 0.5
    system = OpticalSystem.from_components(Deflector((alpha, -alpha), z=0.0))
    rays = random_rays()
    out = TransferMatrixSolver().trace(system, rays, z_end=L).final
    torch.testing.assert_close(out.x, rays.x + L * rays.xp + alpha * L)
    torch.testing.assert_close(out.y, rays.y + L * rays.yp - alpha * L)
    torch.testing.assert_close(out.xp, rays.xp + alpha)


def test_double_deflector_pivots_on_the_specimen_plane():
    tilt = (3e-3, -1e-3)
    dd = DoubleDeflector(tilt, z=0.0, z_2=0.05, pivot=0.2)
    system = OpticalSystem.from_components(dd, Plane(z=0.2, name="specimen"))
    axial = Rays.from_state(torch.zeros(1, 4, dtype=torch.float64))
    at_specimen = TransferMatrixSolver().trace(system, axial).planes["specimen"]
    torch.testing.assert_close(at_specimen.x, torch.zeros(1, dtype=torch.float64))
    torch.testing.assert_close(at_specimen.y, torch.zeros(1, dtype=torch.float64))
    torch.testing.assert_close(at_specimen.xp, torch.tensor([tilt[0]], dtype=torch.float64))
    torch.testing.assert_close(at_specimen.yp, torch.tensor([tilt[1]], dtype=torch.float64))


def test_double_deflector_excitations_roundtrip():
    dd = DoubleDeflector((3e-3, -1e-3), z=0.0, z_2=0.05, pivot=0.2, beam_shift=(1e-5, 2e-5))
    a1, a2 = dd.excitations
    rebuilt = DoubleDeflector.from_excitations(a1, a2, z=0.0, z_2=0.05, pivot=0.2)
    torch.testing.assert_close(rebuilt.tilt, dd.tilt)
    torch.testing.assert_close(rebuilt.beam_shift, dd.beam_shift)
    torch.testing.assert_close(rebuilt.transfer_map().matrix, dd.transfer_map().matrix)


def test_learning_tilt_keeps_the_pivot():
    # steer the beam onto a target slope; the position at the pivot must stay fixed throughout
    dd = DoubleDeflector((0.0, 0.0), z=0.0, z_2=0.05, pivot=0.2).learn("tilt")
    system = OpticalSystem.from_components(dd, Plane(z=0.2, name="specimen"))
    axial = Rays.from_state(torch.zeros(1, 4, dtype=torch.float64))
    optimizer = torch.optim.Adam([dd.tilt], lr=1e-4)
    for _ in range(20):
        optimizer.zero_grad()
        out = TransferMatrixSolver().trace(system, axial).planes["specimen"]
        loss = ((out.xp - 2e-3) ** 2).sum()
        loss.backward()
        optimizer.step()
        assert out.x.abs().item() < 1e-18
    assert dd.tilt[0].item() > 1e-3


def test_shifted_lens_is_centred_lens_plus_deflection():
    P, s = 0.4, (1e-3, -2e-3)
    shifted = ThinLens(power=P, z=0.0, shift=s).transfer_map()
    equivalent = Deflector((P * s[0], P * s[1]), z=0.0, kind="achromatic").transfer_map()
    torch.testing.assert_close(shifted.matrix, (equivalent @ thin_lens_map(P)).matrix)


def test_tilted_thin_lens_is_unchanged_at_first_order():
    P = 0.4
    tilted = ThinLens(power=P, z=0.0, tilt=(1e-3, 2e-3)).transfer_map()
    torch.testing.assert_close(tilted.matrix, thin_lens_map(P).matrix)


@pytest.mark.parametrize("kind", ["magnetic", "electrostatic"])
def test_deflector_dispersion(kind):
    alpha, voltage, delta = 5e-3, 200e3, 1e-3
    gamma = lorentz_factor(voltage)
    expected_k = (
        -gamma / (1 + gamma) if kind == "magnetic" else -(gamma**2 + 1) / (gamma * (gamma + 1))
    )
    assert dispersion_coefficient(kind, voltage) == pytest.approx(expected_k)

    rays = Rays.from_state(
        torch.tensor([[0, 0, 0, 0, delta]], dtype=torch.float64), voltage=voltage
    )
    out = (
        OpticalSystem([Deflector((alpha, 0.0), z=0.0, kind=kind)])
        .transfer_map(-1.0, 0.0, voltage=voltage)
        .apply(rays.state)
    )
    torch.testing.assert_close(
        out[0, 1], torch.tensor(alpha * (1 + expected_k * delta), dtype=torch.float64)
    )
    # non-relativistic limits: 1/2 for magnetic, 1 for electrostatic
    assert dispersion_coefficient(kind, 1.0) == pytest.approx(
        -0.5 if kind == "magnetic" else -1.0, rel=1e-5
    )


def test_aperture_clears_alive_but_keeps_shape():
    system = OpticalSystem.from_components(Aperture(radius=1e-3, z=1.0, shift=(5e-4, 0.0)))
    rays = Rays.from_parallel(radius=2e-3, num_rays=9)
    out = TransferMatrixSolver().trace(system, rays).final
    assert out.state.shape == rays.state.shape
    expected = (rays.x - 5e-4).abs() <= 1e-3
    assert torch.equal(out.alive, expected)
    torch.testing.assert_close(out.state, rays.state)


def test_biprism_deflects_toward_the_wire_and_has_no_map():
    system = OpticalSystem.from_components(Biprism(deflection=1e-3, z=0.0, wire_radius=1e-6))
    rays = Rays.from_state(
        torch.tensor([[-1e-4, 0, 0, 0], [0.0, 0, 0, 0], [1e-4, 0, 0, 0]], dtype=torch.float64)
    )
    trace = TransferMatrixSolver().trace(system, rays)
    torch.testing.assert_close(
        trace.final.xp, torch.tensor([1e-3, 0.0, -1e-3], dtype=torch.float64)
    )
    assert trace.final.alive.tolist() == [True, False, True]
    assert trace.transfer_map is None


def test_magnetic_lens_rotation_rotates_the_image():
    # u = 2, f = 1  ->  v = 2, M = -1, rotated by theta
    theta = 0.3
    system = OpticalSystem.from_components(ThinLens.from_focal_length(1.0, z=2.0, rotation=theta))
    z_image, magnification = system.gaussian_image(0.0)
    torch.testing.assert_close(z_image, torch.tensor(4.0, dtype=torch.float64))
    expected_m = -complex(math.cos(theta), math.sin(theta))
    torch.testing.assert_close(magnification, torch.tensor(expected_m, dtype=torch.complex128))

    rays = Rays.from_cone(1e-2, origin=(1e-4, 0.0))
    out = TransferMatrixSolver().trace(system, rays, z_end=z_image).final
    torch.testing.assert_close(out.w, torch.full_like(out.w, expected_m * 1e-4))
    focal = system.focal_length()
    torch.testing.assert_close(focal.abs(), torch.tensor(1.0, dtype=torch.float64))


def test_ode_solver_matches_transfer_matrix_on_thin_elements():
    system = OpticalSystem.from_components(
        ThinLens(power=0.3, z=1.0), Deflector((1e-3, 0.0), z=1.5), Aperture(2e-3, z=2.0)
    )
    rays = random_rays()
    tm = RayTracer.from_models(system, rays).trace()
    ode = RayTracer.from_models(system, rays, solver=ODESolver.paraxial()).trace()
    torch.testing.assert_close(ode.final.state, tm.final.state)
    assert torch.equal(ode.final.alive, tm.final.alive)


def test_transfer_map_compose_and_affine_roundtrip():
    a = TransferMap(thin_lens_map(0.2).matrix @ drift_map(1.0).matrix)
    b = TransferMap.from_affine(a.linear, a.offset)
    torch.testing.assert_close(a.matrix, b.matrix)
    assert a.entry("x", "x'") == pytest.approx(1.0)


def test_soft_aperture_transmission_profile():
    R, w = 1e-3, 1e-4
    aperture = Aperture(radius=R, z=0.0, edge_width=w)
    x = torch.tensor([0.0, R - w, R, R + w], dtype=torch.float64)
    rays = Rays.from_state(torch.stack([x, *[torch.zeros_like(x)] * 3], dim=-1))
    out = aperture.apply(rays)
    assert out.weight[:2].tolist() == [1.0, 1.0]
    assert out.weight[2].item() == pytest.approx(0.5, abs=w / R)
    assert out.weight[3].item() == 0.0
    assert out.alive.tolist() == [True, True, True, False]
    with pytest.raises(ValueError, match="edge_width"):
        Aperture(radius=R, z=0.0, edge_width=2 * R)


def test_soft_aperture_current_and_its_gradient():
    # a uniform cone through an aperture at distance L passes (R / L tan(alpha))^2 of the current
    L, alpha = 0.1, 1e-2
    beam = L * math.tan(alpha)
    aperture = Aperture(radius=0.6 * beam, z=L, edge_width=0.02 * beam).learn("radius")
    system = OpticalSystem.from_components(aperture)
    trace = TransferMatrixSolver().trace(system, Rays.from_disk(alpha, num_rays=4000))
    transmission = trace.transmission()
    assert transmission.item() == pytest.approx(0.36, abs=1e-3)
    transmission.backward()
    assert aperture.radius.grad.item() == pytest.approx(2 * 0.6 / beam, rel=1e-3)
