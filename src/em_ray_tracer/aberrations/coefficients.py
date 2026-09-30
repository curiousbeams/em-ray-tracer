"""Fitted aberration coefficients and their entry points."""

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Self

import torch

from em_ray_tracer.aberrations.basis import AberrationBasis
from em_ray_tracer.aberrations.fit import fit_coefficients
from em_ray_tracer.rays import Rays

if TYPE_CHECKING:
    from em_ray_tracer.solvers.base import Trace


@dataclass
class AberrationCoefficients:
    """Aberration coefficients fitted to traced rays.

    Convention: the deviations are the ray positions at the reference plane, and the
    variables are the *input* (object-side) slope omega and position w, measured from a
    reference ray (by default the mean of the alive input rays). A drift of dz past the focus
    therefore fits as C10 = dz. To refer the coefficients to image-side slopes (as a probe's
    Cs usually is), use :meth:`rescaled` with the angular magnification.

    ``values`` hold complex tensors for terms with an azimuthal orientation and real tensors
    otherwise. All of them carry gradients back to the system parameters.
    """

    values: dict[str, torch.Tensor]
    basis: AberrationBasis
    residual_rms: torch.Tensor

    # region --- entry points ---

    @classmethod
    def fit(
        cls,
        basis: AberrationBasis,
        omega: torch.Tensor,
        w: torch.Tensor,
        w_out: torch.Tensor,
        weight: torch.Tensor | None = None,
    ) -> Self:
        """Fit from complex input slopes, input positions and output positions."""
        values, residual = fit_coefficients(basis, omega, w, w_out, weight)
        return cls(values=values, basis=basis, residual_rms=residual)

    @classmethod
    def from_rays(
        cls,
        basis: AberrationBasis,
        rays_in: Rays,
        rays_out: Rays,
        reference: tuple[complex, complex] | None = None,
    ) -> Self:
        """Fit output rays against the input rays they came from.

        ``reference`` is ``(w_ref, omega_ref)``; it defaults to the mean input position and
        slope of the rays that reach the output alive. Dead rays get zero weight.
        """
        alive = rays_in.alive & rays_out.alive
        weight = rays_out.weight * alive
        if reference is None:
            norm = weight.sum(-1, keepdim=True)
            w_ref = (rays_in.w * weight).sum(-1, keepdim=True) / norm
            omega_ref = (rays_in.slope * weight).sum(-1, keepdim=True) / norm
        else:
            w_ref, omega_ref = reference
        return cls.fit(
            basis,
            omega=rays_in.slope - omega_ref,
            w=rays_in.w - w_ref,
            w_out=rays_out.w,
            weight=weight,
        )

    @classmethod
    def from_trace(
        cls,
        trace: "Trace",
        basis: AberrationBasis,
        plane: str | None = None,
        reference: tuple[complex, complex] | None = None,
    ) -> Self:
        """Fit the rays at ``plane`` (default: the end of the trace) against the input rays."""
        rays_out = trace.final if plane is None else trace.planes[plane]
        return cls.from_rays(basis, trace.initial, rays_out, reference)

    # endregion --- entry points ---

    def __getitem__(self, name: str) -> torch.Tensor:
        return self.values[name]

    def magnitude(self, name: str) -> torch.Tensor:
        return self.values[name].abs()

    def angle(self, name: str) -> torch.Tensor:
        """Orientation phi_nm = arg(c) / m of an axial term (0 for round terms)."""
        term = self.basis[name]
        value = self.values[name]
        if term.multiplicity == 0 or not value.is_complex():
            return torch.zeros_like(value.real)
        return value.angle() / term.multiplicity

    def to_polar(self) -> dict[str, float]:
        """Axial terms in polar notation: ``{"C10": ..., "C12": ..., "phi12": ...}``.

        Units are those of the system (m); wave-optics codes usually expect Angstrom. Check
        the sign of C10 against their convention before passing it on (abTEM's C10 is
        -defocus).
        """
        out = {}
        for term in self.basis:
            if not term.is_axial or not term.name.startswith("C"):
                continue
            n, m = term.name[1], term.name[2]
            value = self.values[term.name]
            if term.multiplicity == 0:
                out[f"C{n}{m}"] = float(value.real)
            else:
                out[f"C{n}{m}"] = float(value.abs())
                out[f"phi{n}{m}"] = float(self.angle(term.name))
        return out

    def rescaled(self, slope_scale: float = 1.0, position_scale: float = 1.0) -> Self:
        """Coefficients against omega' = slope_scale * omega and w' = position_scale * w.

        A term whose deviation has degree p in omega and q in w is divided by
        slope_scale^p position_scale^q. With the angular magnification as ``slope_scale``,
        this refers the coefficients to the image-side slope.
        """
        values = {}
        for term in self.basis:
            a, b, c, d = term.powers
            factor = slope_scale ** (a + b - 1) * position_scale ** (c + d)
            values[term.name] = self.values[term.name] / factor
        return type(self)(values=values, basis=self.basis, residual_rms=self.residual_rms)

    def __repr__(self) -> str:
        def fmt(v: torch.Tensor) -> str:
            v = v.detach()
            if v.is_complex():
                return f"{float(v.abs()):.6g} at {math.degrees(float(v.angle())):.1f} deg"
            return f"{float(v):.6g}"

        rows = "\n".join(f"  {name:>18s}: {fmt(v)}" for name, v in self.values.items())
        return (
            f"AberrationCoefficients(residual_rms={float(self.residual_rms.detach()):.3g}\n"
            f"{rows}\n)"
        )
