"""Independent reference values for the third-order spherical aberration of a Glaser bell lens.

Nothing in here uses em_ray_tracer: only numpy and scipy. Two routes are provided so they can
check each other:

* ``glaser_c30``  : the aberration integral, Cs = -4 * integral of the fourth-order term of the
                    reduced Lagrangian over the paraxial object ray (derived below).
* ``trace_cs``    : brute-force ray tracing with exact kinematics and the off-axis field series
                    to high order, extrapolated to zero aperture.

Setup: pure magnetic lens, B(z) = B0 / (1 + ((z - zc)/a)^2) inside [z_start, z_end] and zero
outside (the same truncation a finite component has). Object point on the axis at z_obj, the
rays leave it with slope alpha. Lengths in metres, B in tesla, voltage in volts.

Derivation (pure magnetic, phi_hat constant, canonical angular momentum zero, which holds for a
point object on the axis). In the frame rotating with the Larmor angle the radial motion follows
from the reduced Lagrangian (up to the constant factor sqrt(phi_hat))

    L = sqrt(1 + r'^2) * sqrt(1 - q A^2),   q = eta^2 / phi_hat,   A = r B/2 - r^3 B''/16 + ...

Order 2 gives the paraxial equation r'' + (q B^2 / 4) r = 0. Order 4 is

    F4 = -r'^4/8 - (q B^2/16) r^2 r'^2 + (q B B''/32 - q^2 B^4/128) r^4

First-order perturbation theory with the Green function of the paraxial equation gives, for the
displacement at the Gaussian image plane, delta = M * Cs * alpha^3 with the object-referred

    Cs = -4 * integral_{z_obj}^{z_img} F4(g, g') dz
       = integral [ g'^4/2 + (q B^2/4) g^2 g'^2 - (q B B''/8) g^4 + (q^2 B^4/32) g^4 ] dz

where g is the paraxial ray with g(z_obj) = 0, g'(z_obj) = 1, and M the magnification. The
image-referred coefficient (angle measured at the image side) is M^4 * Cs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp

from em_ray_tracer.constants import ETA, relativistic_potential


@dataclass(frozen=True)
class GlaserLens:
    B0: float
    a: float
    zc: float
    z_start: float
    z_end: float

    def inside(self, z: float) -> bool:
        return self.z_start <= z <= self.z_end

    def deriv(self, n: int, z):
        """n-th z-derivative of the axial profile, zero outside the component."""
        z = np.asarray(z, dtype=float)
        zeta = (z - self.zc) / self.a
        # 1/(1+zeta^2) = Im[1/(zeta - i)],  d^n/dzeta^n -> (-1)^n n! / (zeta - i)^(n+1)
        val = np.imag((-1.0) ** n * math.factorial(n) / (zeta - 1j) ** (n + 1))
        val = self.B0 * val / self.a**n
        return np.where((z >= self.z_start) & (z <= self.z_end), val, 0.0)


def glaser_c30(lens: GlaserLens, voltage: float, z_obj: float, *, rtol: float = 1e-12) -> dict:
    """Aberration-integral value of Cs for a point object on the axis at z_obj."""
    q = ETA**2 / relativistic_potential(voltage)
    if z_obj >= lens.z_end:
        raise ValueError("object must be upstream of the end of the component")

    # field-free drift from the object up to the start of the field
    z0 = max(z_obj, lens.z_start)
    d = z0 - z_obj
    g, gp, c, cp, integral = d, 1.0, 1.0, 0.0, 0.5 * d

    def rhs(z, y):
        g, gp, c, cp, _ = y
        B, B2 = float(lens.deriv(0, z)), float(lens.deriv(2, z))
        qb2 = q * B * B
        return [
            gp, -0.25 * qb2 * g, cp, -0.25 * qb2 * c,
            0.5 * gp**4 + 0.25 * qb2 * g**2 * gp**2 - 0.125 * q * B * B2 * g**4
            + (q * q / 32.0) * B**4 * g**4,
        ]

    def hits_axis(z, y):
        return y[0]
    hits_axis.terminal = True
    hits_axis.direction = -1

    sol = solve_ivp(rhs, (z0, lens.z_end), [g, gp, c, cp, integral], method="DOP853",
                    rtol=rtol, atol=1e-20, events=hits_axis)
    g, gp, c, cp, integral = sol.y[:, -1]
    if sol.t_events[0].size:
        z_img = float(sol.t_events[0][0])            # image plane inside the field region
    else:
        if gp >= 0:
            raise ValueError("no real image: the object ray never crosses the axis")
        z_img = lens.z_end - g / gp                   # exact: straight line after the lens
        dz = z_img - lens.z_end
        integral += 0.5 * gp**4 * dz
        c, cp = c + cp * dz, cp
    magnification = float(c)
    cs = float(integral)
    return {"Cs_object": cs, "Cs_image": magnification**4 * cs, "M": magnification,
            "z_img": float(z_img)}


def _series_field(lens: GlaserLens, x, y, z, nmax=4):
    """B_x, B_y, B_z from the axial profile, Laplace series to r^(2*nmax)."""
    r2 = x * x + y * y
    bz = np.zeros_like(r2)
    bt = np.zeros_like(r2)       # B_x = x * bt, B_y = y * bt
    for n in range(nmax + 1):
        bz += (-1.0) ** n * r2**n * lens.deriv(2 * n, z) / (4.0**n * math.factorial(n) ** 2)
        if n >= 1:
            bt += ((-1.0) ** n / (2.0 ** (2 * n - 1) * math.factorial(n - 1) * math.factorial(n))
                   * r2 ** (n - 1) * lens.deriv(2 * n - 1, z))
    return x * bt, y * bt, bz


def trace_cs(lens: GlaserLens, voltage: float, z_obj: float, z_img: float, magnification: float,
             alphas=None, nmax: int = 4) -> float:
    """Zero-aperture Cs from exact-kinematics tracing through the series field.

    Integrates the exact Lorentz equation in z (state: x, y, ux = x'/rho, uy = y'/rho, plus the
    paraxial Larmor angle to fix the sign of the radial displacement), reads the radial
    displacement at z_img, forms delta / (M alpha^3) and Richardson-extrapolates in alpha^2.
    """
    kappa = ETA / math.sqrt(relativistic_potential(voltage))
    if alphas is None:
        # keep the ray height at the lens centre below ~1% of a, so the Richardson step only has
        # to remove the fifth-order term
        s0 = 0.01 * lens.a / max(abs(lens.zc - z_obj), lens.a)
        alphas = (s0, s0 / 2)

    def rhs(z, s):
        x, y, ux, uy, _ = s
        uz = math.sqrt(1.0 - ux * ux - uy * uy)
        bx, by, bz = (float(v) for v in _series_field(lens, np.array(x), np.array(y), z, nmax))
        xp, yp = ux / uz, uy / uz
        return [xp, yp, kappa * (by - bz * yp), kappa * (-bx + bz * xp),
                0.5 * kappa * float(lens.deriv(0, z))]

    def delta(alpha):
        rho = math.sqrt(1.0 + alpha * alpha)
        ux, uy = alpha / rho, 0.0
        x = y = theta = 0.0
        z = z_obj
        if z_obj < lens.z_start:                      # exact drift
            dz = lens.z_start - z_obj
            uz = math.sqrt(1 - ux * ux - uy * uy)
            x, y, z = x + ux / uz * dz, y + uy / uz * dz, lens.z_start
        z_stop = min(z_img, lens.z_end)
        sol = solve_ivp(rhs, (z, z_stop), [x, y, ux, uy, theta], method="DOP853",
                        rtol=1e-13, atol=1e-22)
        x, y, ux, uy, theta = sol.y[:, -1]
        if z_img > z_stop:                            # exact drift to the image plane
            uz = math.sqrt(1 - ux * ux - uy * uy)
            dz = z_img - z_stop
            x, y = x + ux / uz * dz, y + uy / uz * dz
        return (x * math.cos(theta) + y * math.sin(theta)) / (magnification * alpha**3)

    (a1, a2) = alphas
    d1, d2 = delta(a1), delta(a2)
    r = (a1 / a2) ** 2
    return float((r * d2 - d1) / (r - 1.0))        # removes the alpha^2 (fifth-order) term