"""Paraxial ray equations through field lenses (ODESolver.paraxial).

The Glaser closed form, which checks the magnetic equations, is in tests/test_stubs.py.
"""

import math

import pytest
import torch

from em_ray_tracer import FieldLens, ODESolver, OpticalSystem, Plane, Rays
from em_ray_tracer.constants import (
    DTYPE,
    ETA,
    REST_ENERGY_EV,
    lorentz_factor,
    relativistic_potential,
)

A = 0.0075


def trace_through(lens, state, voltage):
    """The rays at the end of the lens's span, traced from its start."""
    system = OpticalSystem.from_components(lens, Plane(z=float(lens.z_end), name="out"))
    rays = Rays.from_state(
        torch.tensor(state, dtype=DTYPE), z=float(lens.z_start), voltage=voltage
    )
    return ODESolver.paraxial().trace(system, rays).planes["out"]


def test_electrostatic_wronskian_is_conserved():
    # sqrt(phi_hat) (x1 x2' - x2 x1') is invariant under the paraxial equation. A Schiske
    # lens's span is symmetric, so phi_hat is the same at both ends, and so is the Wronskian.
    lens = FieldLens.schiske(k=0.6, a=A, z=0.0)
    out = trace_through(lens, [[1e-4, 0, 0, 0], [0, 1e-3, 0, 0]], voltage=30e3)
    wronskian = out.x[0] * out.xp[1] - out.x[1] * out.xp[0]
    assert wronskian.item() == pytest.approx(1e-4 * 1e-3, rel=1e-9)


def test_weak_schiske_lens_power():
    # With R = r phi_hat^(1/4) the paraxial equation becomes R'' + T R = 0, where
    # T = (3/16) (gamma phi' / phi_hat)^2 - phi'^2 / (4 E0 phi_hat) and E0 = m c^2 / e. In a weak
    # lens R stays nearly constant, so a ray that enters parallel leaves with
    # -x'/x = integral of T + (1/2) phi_hat'/phi_hat at the exit. The last term is there
    # because the span ends where the field has not quite vanished.
    voltage = 30e3
    lens = FieldLens.schiske(k=0.1, a=A, z=0.0)
    out = trace_through(lens, [[1e-3, 0, 0, 0]], voltage)
    z = torch.linspace(float(lens.z_start), float(lens.z_end), 200_001, dtype=DTYPE)
    with torch.no_grad():
        phi = voltage * lens.axial_derivative(0, z)
        dphi = voltage * lens.axial_derivative(1, z)
    gamma, phi_hat = lorentz_factor(phi), relativistic_potential(phi)
    T = 3 / 16 * (gamma * dphi / phi_hat) ** 2 - dphi**2 / (4 * REST_ENERGY_EV * phi_hat)
    expected = torch.trapezoid(T, z) + (gamma * dphi / phi_hat)[-1] / 2
    assert -out.xp[0].item() / 1e-3 == pytest.approx(expected.item(), rel=2e-3)


def test_magnetic_rotation_follows_larmor_angle():
    # The closed-form test compares |w| only, so it can't see the sense of the rotation. A ray
    # in a meridional plane of the rotating frame (w' = i theta' w at the start) ends at the
    # azimuth theta = (s / 2) integral of B dz.
    voltage, B0 = 1000.0, 1e-3  # weak enough that the ray doesn't cross the axis in the span
    lens = FieldLens.glaser(B0=B0, a=A, z=0.0)
    s = ETA / math.sqrt(relativistic_potential(voltage))
    rate = s * lens.axial_derivative(0, lens.z_start).item() / 2
    out = trace_through(lens, [[1e-3, 0, 0, rate * 1e-3]], voltage)
    u0, u1 = float(lens.z_start) / A, float(lens.z_end) / A
    theta = s * B0 * A / 2 * (math.atan(u1) - math.atan(u0))
    assert math.atan2(out.y[0].item(), out.x[0].item()) == pytest.approx(theta, rel=1e-8)
