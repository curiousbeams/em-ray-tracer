"""Hybrid hexapole corrector: magnetic Glaser lenses (fields) and hexapoles as thin slices.

Probe-forming column at 200 kV, z down:
  point source (z = 0) -> collimator C -> hexapole H1 -> transfer doublet TL1, TL2 (TL2
  reversed, so their Larmor rotations cancel) -> hexapole H2 -> objective OL -> probe.

The Glaser fields have their default extent (+-10a, tapered to zero over the outer 40%), and
their tails overlap where lenses are close. The lens strengths are solved on the package's own
summed, tapered profile so that the source is imaged to the crossover between the transfer
lenses and to the probe, and H1's centre is imaged onto H2's.

Each hexapole is ``n`` thin kicks spread over its length, d(omega) = -(K/n) conj(w)^2 with a
complex K (strength and orientation). H2's K is solved so that the second-order three-fold
astigmatism A2 cancels at the probe; the common scale is then set so that C30 vanishes for the
40 mm hexapoles. The thin hexapole is the one piece the package doesn't have yet.

The figure compares five columns (no corrector, H1 alone, and H1 + H2 at 0, 20 and 40 mm) with
the rays, the summed axial field, and the spot at the probe. A2 cancels for any hexapole
length, but the negative C30 that cancels the round lenses' Cs comes only from the hexapoles'
finite length: the thin pair leaves C30 where it was.

Run (about 80 s): uv run python examples/hexapole_corrector.py [output.png]
"""

import cmath
import math
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Rectangle

from em_ray_tracer import AberrationBasis, FieldLens, OpticalSystem, Plane, Rays, RayTracer
from em_ray_tracer.aberrations.coefficients import AberrationCoefficients
from em_ray_tracer.components.base import ComponentBase
from em_ray_tracer.constants import DTYPE, ETA, relativistic_potential
from em_ray_tracer.rays import XP, YP, X, Y
from em_ray_tracer.solvers.ode import ODESolver
from em_ray_tracer.solvers.params import IntegratorParams
from em_ray_tracer.visualization import plot_rays

MM = 1e-3
VOLTAGE = 200e3
Q = ETA**2 / relativistic_potential(VOLTAGE)  # r'' = -(Q B^2 / 4) r in the Larmor frame
DZ = 20e-6
SOURCE_ANGLE = 3e-3  # source semi-angle [rad]
Z_END = 310 * MM  # traced span: 0 to here

# name: (centre, half-width a) [m]
LENSES = {
    "C": (35 * MM, 3 * MM),
    "TL1": (130 * MM, 4 * MM),
    "TL2": (210 * MM, 4 * MM),
    "OL": (300 * MM, 3 * MM),
}
CROSSOVER = 170 * MM  # the source's intermediate image, between the transfer lenses
H1_Z, H2_Z = 90 * MM, 250 * MM  # conjugate through the transfer doublet
PROBE = 302.5 * MM
LENGTH = 40 * MM  # hexapole length
SLICES = 16

INK, INK_2, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"
RAY = "#2a78d6"
HEXAPOLE = "#eb6834"


class ThinHexapole(ComponentBase):
    """A thin magnetic hexapole kick, d(omega) = -strength conj(w)^2, with a complex strength
    K e^{3 i phi} (phi the orientation). The piece the package is missing."""

    def __init__(self, strength: complex, *, z: float, **kwargs):
        super().__init__(z=z, **kwargs)
        self.strength = complex(strength)

    def local_transfer_map(self, order=1, voltage=None):
        return None  # nonlinear: crossed by ``apply``

    def apply(self, rays, **kwargs):
        s = rays.state.clone()
        w = torch.complex(s[..., X], s[..., Y])
        kick = -self.strength * w.conj() ** 2
        s[..., XP] = s[..., XP] + kick.real
        s[..., YP] = s[..., YP] + kick.imag
        return rays.replace(state=s, z=self.z_end)


# region --- design: paraxial radial equation on the summed field ---


def unit_profiles(z):
    """Each lens's tapered axial B per tesla of B0 at z (TL2's sign included)."""
    out = {}
    for name, (zc, a) in LENSES.items():
        sign = -1.0 if name == "TL2" else 1.0
        lens = FieldLens.glaser(B0=sign, a=a, z=zc)
        with torch.no_grad():
            out[name] = lens.axial_derivative(0, z)["magnetic"]
    return out


def radial(B0, z0, z1, r0, rp0, n=12000):
    """RK4 of r'' = -(Q B^2 / 4) r from z0 to z1, batched over any tensor entry of B0. The
    profiles are tabulated once on the half-step grid."""
    h = (z1 - z0) / n
    grid = z0 + (h / 2) * torch.arange(2 * n + 1, dtype=DTYPE)
    unit = unit_profiles(grid)
    b = sum(B0[name] * unit[name].unsqueeze(-1) for name in LENSES)  # (2n + 1, batch)
    accel = -(Q * b * b / 4)
    batch = next((v for v in B0.values() if torch.is_tensor(v)), torch.zeros(1, dtype=DTYPE))
    r = torch.full_like(batch, r0)
    rp = torch.full_like(batch, rp0)
    for i in range(n):
        a0, a1, a2 = accel[2 * i], accel[2 * i + 1], accel[2 * i + 2]
        k1r, k1v = rp, a0 * r
        k2r, k2v = rp + h / 2 * k1v, a1 * (r + h / 2 * k1r)
        k3r, k3v = rp + h / 2 * k2v, a1 * (r + h / 2 * k2r)
        k4r, k4v = rp + h * k3v, a2 * (r + h * k3r)
        r = r + h / 6 * (k1r + 2 * k2r + 2 * k3r + k4r)
        rp = rp + h / 6 * (k1v + 2 * k2v + 2 * k3v + k4v)
    return r, rp


def first_root(f, lo, hi, rounds=4, batch=40):
    """The first sign change of f over [lo, hi], refined by batched grids (f takes a tensor)."""
    for _ in range(rounds):
        grid = torch.linspace(lo, hi, batch, dtype=DTYPE)
        v = f(grid)
        k = int(torch.nonzero(torch.sign(v) != torch.sign(v[0]))[0])
        lo, hi = float(grid[k - 1]), float(grid[k])
    return (lo + hi) / 2


def design(iterations=3):
    B0 = {"C": 0.26, "TL1": 0.213, "TL2": 0.213, "OL": 1.056}
    for _ in range(iterations):
        B0["C"] = first_root(
            lambda g: radial({**B0, "C": g}, 0.0, CROSSOVER, 0.0, 1.0)[0], 0.18, 0.35
        )
        tl = first_root(
            lambda g: radial({**B0, "TL1": g, "TL2": g}, H1_Z, H2_Z, 0.0, 1.0)[0], 0.15, 0.3
        )
        B0["TL1"] = B0["TL2"] = tl
        B0["OL"] = first_root(lambda g: radial({**B0, "OL": g}, 0.0, PROBE, 0.0, 1.0)[0], 0.9, 1.2)
    return B0


# endregion --- design ---


def lenses(B0):
    out = []
    for name, (zc, a) in LENSES.items():
        sign = -1.0 if name == "TL2" else 1.0
        out.append(FieldLens.glaser(B0=sign * B0[name], a=a, z=zc, name=name))
    return out


def hexapole(K, centre, length, n, label):
    if K == 0:
        return []
    n = n if length > 0 else 1
    zs = [centre] if n == 1 else [centre - length / 2 + length * (i + 0.5) / n for i in range(n)]
    return [ThinHexapole(K / n, z=z, name=f"{label}_{i}") for i, z in enumerate(zs)]


def column(B0, K1=0, K2=0, length=LENGTH, n=SLICES):
    return OpticalSystem.from_components(
        *lenses(B0),
        *hexapole(K1, H1_Z, length, n, "H1"),
        *hexapole(K2, H2_Z, length, n, "H2"),
        Plane(z=CROSSOVER, name="crossover"),
        Plane(z=PROBE, name="probe"),
    )


SOLVER = ODESolver.nonparaxial(3, IntegratorParams.RK4(dz=DZ))
PARAXIAL = ODESolver.paraxial(IntegratorParams.RK4(dz=DZ))
BASIS = AberrationBasis.axial(order=5)
FIT_RAYS = Rays.from_cone(semiangle=SOURCE_ANGLE, n_azimuthal=24, n_shells=6, voltage=VOLTAGE)


def references(B0):
    """Image rotation and slope scale at each plane, from one paraxial ray: arg and modulus of
    omega_plane / omega_source. The rotation is the full angle (inversion included), so a
    positive Cs fits as a positive C30."""
    one = Rays.from_state(torch.tensor([[0, 1e-5, 0, 0]], dtype=DTYPE), voltage=VOLTAGE)
    with torch.no_grad():
        trace = PARAXIAL.trace(column(B0), one, z_end=Z_END)
    out = {}
    for plane in ("crossover", "probe"):
        at = trace.planes[plane]
        omega = complex(float(at.xp[0]), float(at.yp[0])) / 1e-5
        spot = float(torch.hypot(at.x, at.y)[0]) / 1e-5
        out[plane] = (cmath.phase(omega), abs(omega), spot)
    return out


def fit(system, ref, rays=FIT_RAYS):
    """Coefficients at the crossover and the probe, referred to the slopes there."""
    with torch.no_grad():
        trace = RayTracer.from_models(
            system, rays, solver=SOLVER, z_end=PROBE, verbose=False
        ).trace()
    out = {}
    for plane in ("crossover", "probe"):
        rotation, scale, _ = ref[plane]
        c = AberrationCoefficients.from_rays(
            BASIS, trace.initial, trace.planes[plane], rotation=torch.tensor(rotation, dtype=DTYPE)
        )
        out[plane] = c.rescaled(slope_scale=scale)
    return out


def solve_corrector(B0, ref, length, n):
    """H2's complex strength per unit H1 strength, cancelling A2 at the probe (A2 is linear in
    each hexapole's strength)."""
    unit = 1e3
    a1 = complex(fit(column(B0, unit, 0, length, n), ref)["probe"]["C23"])
    a2 = complex(fit(column(B0, 0, unit, length, n), ref)["probe"]["C23"])
    return -a1 / a2


def main(output=None):
    B0 = design()
    print("lens B0 [T]:", {k: round(v, 5) for k, v in B0.items()})
    ref = references(B0)
    for plane, (rot, scale, spot) in ref.items():
        print(
            f"{plane:9s}: rotation {rot:+.4f} rad, slope scale {scale:.4f}, "
            f"paraxial spot / source slope {spot:.1e} m (0 at an exact image)"
        )

    round_only = fit(column(B0), ref)
    c0 = float(round_only["probe"]["C30"])
    k2 = {(LENGTH, SLICES): solve_corrector(B0, ref, LENGTH, SLICES)}
    t1 = 1e4
    c1 = float(fit(column(B0, t1, t1 * k2[LENGTH, SLICES]), ref)["probe"]["C30"])
    t_star = t1 * math.sqrt(c0 / (c0 - c1))
    print(
        f"probe C30: round lenses {c0 / MM:+.4f} mm; with K = {t1:.0e}: {c1 / MM:+.4f} mm "
        f"-> K* = {t_star:.4g} m^-2, H2/H1 = {abs(k2[LENGTH, SLICES]):.4f} at "
        f"{math.degrees(cmath.phase(k2[LENGTH, SLICES])) / 3:+.2f} deg"
    )

    configs = [
        ("No corrector", dict()),
        ("H1 only, 40 mm", dict(K1=t_star, length=LENGTH)),
        ("H1 + H2, thin", dict(length=0.0, n=1)),
        ("H1 + H2, 20 mm", dict(length=20 * MM)),
        ("H1 + H2, 40 mm", dict(length=LENGTH)),
    ]
    rows = []
    for title, spec in configs:
        if "K1" not in spec and title != "No corrector":
            key = (spec["length"], spec.get("n", SLICES))
            if key not in k2:
                k2[key] = solve_corrector(B0, ref, *key)
            spec = dict(spec, K1=t_star, K2=t_star * k2[key])
        system = column(B0, **spec)
        coeffs = fit(system, ref)
        rows.append((title, spec, system, coeffs))
        p, x = coeffs["probe"], coeffs["crossover"]
        print(
            f"{title:16s} crossover |A2| {abs(complex(x['C23'])):.3g} m | probe "
            f"C30 {float(p['C30']) / MM:+.4f} mm, |A2| {abs(complex(p['C23'])) * 1e9:8.2f} nm, "
            f"C50 {float(p['C50']) / MM:+8.1f} mm, resid {float(p.residual_rms):.1e} m"
        )
    figure(B0, rows, output)


# region --- figure ---


def fmt_length(value: float) -> str:
    """A length in the unit that keeps 1-3 digits before the point; fit noise below 1 pm is 0."""
    if abs(value) < 1e-12:
        return "0"
    for unit, scale in (("m", 1.0), ("mm", 1e-3), ("µm", 1e-6), ("nm", 1e-9), ("pm", 1e-12)):
        if abs(value) >= scale or unit == "pm":
            return f"{value / scale:.3g} {unit}"


def figure(B0, rows, output):
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.edgecolor": INK_2,
            "axes.labelcolor": INK,
            "xtick.color": INK_2,
            "ytick.color": INK_2,
            "text.color": INK,
        }
    )
    n = len(rows)
    fig = plt.figure(figsize=(3.0 * (n + 1), 13.5), facecolor=SURFACE)
    outer = fig.add_gridspec(2, 1, height_ratios=[3.2, 1], hspace=0.16)
    widths = [0.8] + [1] * n
    top = outer[0].subgridspec(1, n + 1, width_ratios=widths, wspace=0.12)
    bottom = outer[1].subgridspec(1, n + 1, width_ratios=widths, wspace=0.42)

    # fields column: summed B(z) and the hexapole positions
    axf = fig.add_subplot(top[0], facecolor=SURFACE)
    z, _, b = column(B0).fields(0.0, Z_END)
    axf.plot(b.numpy() / float(b.abs().max()), z.numpy() * 1e3, color=INK, lw=1.2)
    axf.axvline(0, color=INK_2, lw=0.5)
    for centre, label in ((H1_Z, "H1"), (H2_Z, "H2")):
        axf.add_patch(
            Rectangle(
                (-1, (centre - LENGTH / 2) * 1e3),
                2,
                LENGTH * 1e3,
                color=HEXAPOLE,
                alpha=0.18,
                lw=0,
            )
        )
        axf.text(
            -0.95, centre * 1e3, label, color=HEXAPOLE, va="center", fontsize=9, fontweight="bold"
        )
    for name, (zc, _) in LENSES.items():
        axf.text(-0.55, zc * 1e3, name, ha="center", va="center", fontsize=8.5, color=INK_2)
    axf.set_xlim(-1, 1.05)
    axf.set_ylim(Z_END * 1e3, 0)
    axf.set_xlabel("B / max |B|")
    axf.set_ylabel("z [mm]")
    axf.set_title("Axial field (summed)", fontsize=10)

    # ray columns
    plot_rays_in = Rays.from_cone(
        semiangle=SOURCE_ANGLE, n_azimuthal=12, n_shells=3, voltage=VOLTAGE
    )
    traces = []
    for _, _, system, _ in rows:
        with torch.no_grad():
            traces.append(
                RayTracer.from_models(
                    system, plot_rays_in, solver=SOLVER, z_end=Z_END, verbose=False
                ).trace(record=True)
            )
    reach = float(traces[0].trajectory.state[..., X].abs().max()) * 1e3 * 1.35
    for i, ((title, spec, _, coeffs), trace) in enumerate(zip(rows, traces), start=1):
        ax = fig.add_subplot(top[i], sharey=axf, facecolor=SURFACE)
        for zc, _ in LENSES.values():
            ax.axhline(zc * 1e3, color=INK_2, lw=0.5, ls=":", alpha=0.6)
        plot_rays(
            trace,
            system=None,
            ax=ax,
            color=RAY,
            linewidth=0.6,
            off_axis_scale="mm",
            on_axis_scale="mm",
            invert_orientation=True,
        )
        ax.set_ylabel("")
        length = spec.get("length", LENGTH)
        for centre, key in ((H1_Z, "K1"), (H2_Z, "K2")):
            if spec.get(key):
                h = max(length, 0.8 * MM) * 1e3
                ax.add_patch(
                    Rectangle(
                        (-reach, centre * 1e3 - h / 2),
                        2 * reach,
                        h,
                        color=HEXAPOLE,
                        alpha=0.18,
                        lw=0,
                    )
                )
        for zz in (CROSSOVER, PROBE):
            ax.axhline(zz * 1e3, color=INK_2, ls="--", lw=0.7)
        ax.set_xlim(-reach, reach)
        ax.set_title(title, fontsize=10)
        ax.tick_params(labelleft=False)
        a2x = abs(complex(coeffs["crossover"]["C23"]))
        ax.text(
            0.03,
            CROSSOVER * 1e3 - 1.5,
            f"crossover\n|A2| {fmt_length(a2x)}",
            transform=ax.get_yaxis_transform(),
            fontsize=7.5,
            color=INK_2,
            va="bottom",
        )

    # spot diagrams at the probe
    spot_rays = Rays.from_disk(semiangle=SOURCE_ANGLE, num_rays=900, voltage=VOLTAGE)
    shells = (spot_rays.slope.abs() / SOURCE_ANGLE).numpy()
    cmap = plt.get_cmap("Blues")
    for i, (_, _, system, coeffs) in enumerate(rows, start=1):
        with torch.no_grad():
            at = (
                RayTracer.from_models(system, spot_rays, solver=SOLVER, z_end=PROBE, verbose=False)
                .trace()
                .planes["probe"]
            )
        x, y = at.x.numpy() * 1e9, at.y.numpy() * 1e9
        x, y = x - x.mean(), y - y.mean()
        rms = float(np.sqrt((x**2 + y**2).mean()))
        ax = fig.add_subplot(bottom[i], facecolor=SURFACE)
        ax.scatter(x, y, s=2.5, c=cmap(0.35 + 0.65 * shells), lw=0)
        lim = max(np.abs(x).max(), np.abs(y).max()) * 1.15
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect("equal")
        ax.set_xlabel("x [nm]")
        if i == 1:
            ax.set_ylabel("y [nm]")
        p = coeffs["probe"]
        ax.set_title(
            f"C30 {fmt_length(float(p['C30']))} · |A2| {fmt_length(abs(complex(p['C23'])))}"
            f"\nRMS spot {rms:.3g} nm",
            fontsize=8.5,
        )
    axn = fig.add_subplot(bottom[0])
    axn.axis("off")
    axn.text(
        0,
        0.95,
        "Probe spot\n(x–y at the probe plane,\nrays coloured by\naperture angle)\n\n"
        "C30, A2 fitted at the\nprobe, referred to the\nprobe semi-angle",
        va="top",
        fontsize=8.5,
        color=INK_2,
    )

    if output:
        fig.savefig(output, dpi=170, bbox_inches="tight", facecolor=SURFACE)
        print(f"saved {output}")
    else:
        plt.show()


# endregion --- figure ---

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
