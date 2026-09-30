"""Aberration basis and differentiable fit."""

import math

import pytest
import torch

from em_ray_tracer import (
    AberrationBasis,
    AberrationCoefficients,
    AberrationTarget,
    Deflector,
    OpticalSystem,
    Plane,
    Quadrupole,
    Rays,
    RayTracer,
    ThinLens,
)

CONE = dict(semiangle=1e-2, n_azimuthal=16, n_shells=4)


def fit_at(system, rays, basis=None, plane="detector"):
    return RayTracer.from_models(system, rays).fit_aberrations(basis, plane=plane)


def test_defocus_fits_as_C10():
    dz = 0.37
    system = OpticalSystem.from_components(Plane(z=dz, name="detector"))
    coefs = fit_at(system, Rays.from_cone(**CONE))
    assert coefs["C10"].item() == pytest.approx(dz, rel=1e-12)
    for name, value in coefs.values.items():
        if name != "C10":
            assert value.abs().item() < 1e-12, name
    assert coefs.residual_rms.item() < 1e-15


def test_quadrupole_fits_as_C12():
    # source at 0, quadrupole P at L, detector at L + D: w = (L + D) omega - P L D conj(omega)
    P, L, D = 0.2, 0.5, 1.5
    system = OpticalSystem.from_components(
        Quadrupole(power=P, z=L), Plane(z=L + D, name="detector")
    )
    coefs = fit_at(system, Rays.from_cone(**CONE))
    assert coefs["C10"].item() == pytest.approx(L + D, rel=1e-12)
    assert coefs.magnitude("C12").item() == pytest.approx(P * L * D, rel=1e-12)
    assert coefs.angle("C12").item() == pytest.approx(math.pi / 2, rel=1e-12)
    polar = coefs.to_polar()
    assert set(polar) >= {"C10", "C12", "phi12", "C21", "phi21", "C30", "C32", "phi34"}


def test_deflection_goes_to_shift_not_C10():
    alpha, dz = 3e-3, 0.4
    system = OpticalSystem.from_components(
        Deflector((alpha, 0.0), z=0.1), Plane(z=dz, name="detector")
    )
    coefs = fit_at(system, Rays.from_cone(**CONE))
    assert coefs["C10"].item() == pytest.approx(dz, rel=1e-12)
    assert coefs["shift"].item() == pytest.approx(complex(alpha * (dz - 0.1)), rel=1e-12)


def test_C10_is_differentiable_in_the_plane_position():
    detector = Plane(z=0.37, name="detector").learn("z")
    system = OpticalSystem.from_components(ThinLens(power=1.0, z=0.2), detector)
    fit_at(system, Rays.from_cone(**CONE))["C10"].backward()
    # a further drift dz adds dz * M[x', x'] = dz * (1 - 0.2 P) to C10
    assert detector.z.grad.item() == pytest.approx(0.8, rel=1e-10)


def test_optimize_detector_onto_the_focus():
    # object at 0, f = 1 at z = 2: the image is at 4
    detector = Plane(z=3.5, name="detector").learn("z")
    system = OpticalSystem.from_components(ThinLens.from_focal_length(1.0, z=2.0), detector)
    tracer = RayTracer.from_models(system, Rays.from_cone(**CONE), verbose=False)
    tracer.optimize(AberrationTarget("C10", 0.0, plane="detector"), num_iters=5)
    assert detector.z.item() == pytest.approx(4.0, abs=1e-9)


def test_round_lens_fit_at_the_gaussian_image():
    system = OpticalSystem.from_components(ThinLens.from_focal_length(2.0, z=3.0))
    points = torch.tensor([[0.0, 0.0], [1e-4, 0.0], [0.0, 2e-4], [-1e-4, -1e-4]])
    rays = Rays.from_object_points(points, **CONE)
    z_image, magnification = system.gaussian_image(0.0)
    system.add(Plane(z=z_image.item(), name="detector"))
    basis = AberrationBasis.round_lens(order=3)
    coefs = fit_at(system, rays, basis)
    assert coefs["magnification"].item() == pytest.approx(magnification.item(), rel=1e-10)

    # Coefficients carry units (m / m^3 for distortion), so compare each term's contribution
    # to the ray positions, not its raw value, against the 1e-4 m field of view.
    omega = rays.slope - rays.slope.mean()
    w = rays.w - rays.w.mean()
    for name in ["C10", "C30", "coma", "field_astigmatism", "field_curvature", "distortion"]:
        term = basis[name]
        size = max(col.abs().max() for col in term.columns(omega, w))
        assert (coefs[name].abs() * size).item() < 1e-15, name


@pytest.mark.parametrize("seed", [0, 1])
def test_synthetic_axial_coefficients_roundtrip(seed):
    """The basis and fit recover arbitrary third-order coefficients exactly."""
    gen = torch.Generator().manual_seed(seed)
    basis = AberrationBasis.axial(order=3)
    rays = Rays.from_cone(**CONE)
    omega = rays.slope
    truth = {}
    w_out = torch.zeros_like(omega)
    for term in basis:
        re, im = torch.randn(2, generator=gen, dtype=torch.float64)
        value = re if term.is_real else torch.complex(re, im)
        truth[term.name] = value
        cols = term.columns(omega, torch.zeros_like(omega))
        w_out = w_out + re * cols[0] + (im * cols[1] if not term.is_real else 0)
    coefs = AberrationCoefficients.fit(basis, omega, torch.zeros_like(omega), w_out)
    for name, value in truth.items():
        torch.testing.assert_close(coefs[name], value, rtol=1e-8, atol=1e-10)


def test_polar_convention():
    # chi = 1/4 C30 |omega|^4 -> dw = C30 |omega|^2 omega
    # chi = 1/2 C12 alpha^2 cos(2 (phi - phi12)) -> dw = C12 e^{2i phi12} conj(omega)
    C30, C12, phi12 = 1.2, 0.03, 0.4
    omega = Rays.from_cone(**CONE).slope
    w_out = (
        C30 * omega.abs() ** 2 * omega
        + C12 * complex(math.cos(2 * phi12), math.sin(2 * phi12)) * omega.conj()
    )
    coefs = AberrationCoefficients.fit(
        AberrationBasis.axial(3), omega, torch.zeros_like(omega), w_out
    )
    polar = coefs.to_polar()
    assert polar["C30"] == pytest.approx(C30, rel=1e-8)
    assert polar["C12"] == pytest.approx(C12, rel=1e-8)
    assert polar["phi12"] == pytest.approx(phi12, rel=1e-8)


def test_rescaled_refers_to_image_side_slopes():
    omega = Rays.from_cone(**CONE).slope
    w_out = 2.0 * omega.abs() ** 2 * omega
    coefs = AberrationCoefficients.fit(
        AberrationBasis.axial(3), omega, torch.zeros_like(omega), w_out
    )
    # with omega' = m omega, dw = (2 / m^3) |omega'|^2 omega'
    assert coefs.rescaled(slope_scale=2.0)["C30"].item() == pytest.approx(0.25, rel=1e-8)


def test_meridional_fan_cannot_see_y_astigmatism():
    system = OpticalSystem.from_components(Plane(z=1.0, name="detector"))
    with pytest.raises(ValueError, match="C12"):
        fit_at(system, Rays.from_fan(1e-2, 11), AberrationBasis.axial(order=1))


def test_full_basis_has_no_conjugate_duplicates():
    basis = AberrationBasis.full(order=3)
    keys = {t.powers for t in basis}
    for a, b, c, d in keys:
        if (b, a, d, c) != (a, b, c, d):
            assert (b, a, d, c) not in keys
