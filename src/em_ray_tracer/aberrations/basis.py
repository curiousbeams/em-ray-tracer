"""Aberration bases: terms of the aberration function chi and the ray deviations they produce.

Each term is ``chi_k = Re[c_k * norm_k * omega^a conj(omega)^b w^c conj(w)^d]`` with
omega = x' + iy' the (object-side) slope, w = x + iy the object position, and c_k a complex
coefficient (real when the monomial is real). The ray deviation at the reference plane is

    delta_w = grad_omega chi = 2 d chi / d conj(omega)
            = c d_omegabar(f) + conj(c) conj(d_omega(f))

which is real-linear in (Re c, Im c). Fitting chi rather than free polynomials in delta_w
builds in the constraint that ray aberrations derive from one function, as they do for
Hamiltonian optics. It is also why B2 appears in two deviation monomials with tied
coefficients.

Axial terms use the polar notation C_nm, phi_nm common in wave-optics codes (e.g. abTEM):
chi = sum 1/(n+1) C_nm alpha^(n+1) cos(m (phi - phi_nm)), which is a term with
powers ((n+1-m)/2, (n+1+m)/2, 0, 0), norm 1/(n+1), and c = C_nm exp(i m phi_nm).

Off-axial (Seidel, third order) terms and the deviations they give for a real coefficient:

    field_curvature    chi = 1/2 c |omega|^2 |w|^2         dw = c |w|^2 omega
    field_astigmatism  chi = Re[1/2 c conj(omega)^2 w^2]   dw = c conj(omega) w^2
    coma               chi = Re[c |omega|^2 conj(omega) w] dw = c (2 |omega|^2 w + omega^2 conj(w))
    distortion         chi = Re[c conj(omega) |w|^2 w]     dw = c |w|^2 w

Imaginary parts are the anisotropic aberrations of magnetic lenses. Rotational symmetry
allows only terms with (a - b) + (c - d) = 0 in chi.

Every basis also has ``shift`` (dw = c, the chief-ray offset, which absorbs deflections) and,
when w varies, ``magnification`` (dw = c w, the complex magnification including rotation).
"""

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from itertools import product
from typing import Self

import torch


@dataclass(frozen=True)
class AberrationTerm:
    name: str
    powers: tuple[int, int, int, int]
    norm: float = 1.0

    @property
    def is_real(self) -> bool:
        """The monomial is real, so only Re(c) enters chi."""
        a, b, c, d = self.powers
        return a == b and c == d

    @property
    def multiplicity(self) -> int:
        """Azimuthal multiplicity m of an axial term (|b - a|)."""
        return abs(self.powers[1] - self.powers[0])

    @property
    def is_axial(self) -> bool:
        return self.powers[2] == 0 and self.powers[3] == 0

    def columns(self, omega: torch.Tensor, w: torch.Tensor) -> list[torch.Tensor]:
        """Deviation columns for Re(c) and, unless the term is real, Im(c)."""
        a, b, c, d = self.powers
        rest = w**c * w.conj() ** d * self.norm
        zero = torch.zeros_like(omega)
        d_bar = b * omega**a * omega.conj() ** (b - 1) * rest if b > 0 else zero
        d_om = a * omega ** (a - 1) * omega.conj() ** b * rest if a > 0 else zero
        col_re = d_bar + d_om.conj()
        if self.is_real:
            return [col_re]
        return [col_re, 1j * (d_bar - d_om.conj())]


SHIFT = AberrationTerm("shift", (0, 1, 0, 0))
MAGNIFICATION = AberrationTerm("magnification", (0, 1, 1, 0))


def axial_term(n: int, m: int) -> AberrationTerm:
    """The polar-notation term C_nm (order n, multiplicity m)."""
    if (n + 1 - m) % 2 or m > n + 1:
        raise ValueError(f"No axial aberration C{n}{m}")
    return AberrationTerm(f"C{n}{m}", ((n + 1 - m) // 2, (n + 1 + m) // 2, 0, 0), 1 / (n + 1))


class AberrationBasis:
    """An ordered set of :class:`AberrationTerm` to fit. Build one with a classmethod."""

    def __init__(self, terms: Sequence[AberrationTerm]):
        names = [t.name for t in terms]
        if len(set(names)) != len(names):
            raise ValueError(f"Duplicate term names in {names}")
        self.terms = tuple(terms)

    @classmethod
    def axial(cls, order: int = 3) -> Self:
        """Axial (field-independent) aberrations up to ``order``: C10, C12, C21, C23, C30, ...

        Fit it to rays from a single object point (``Rays.from_cone``).
        """
        terms = [SHIFT]
        for n in range(1, order + 1):
            terms += [axial_term(n, m) for m in range(n + 1, -1, -2)][::-1]
        return cls(terms)

    @classmethod
    def round_lens(cls, order: int = 3) -> Self:
        """Terms a rotationally symmetric system allows, axial and off-axial (Seidel at order 3).

        Fit it to rays from several object points (``Rays.from_object_points``).
        """
        if order not in (1, 3):
            raise NotImplementedError("round_lens basis is implemented for orders 1 and 3")
        terms = [SHIFT, MAGNIFICATION, axial_term(1, 0)]
        if order == 3:
            terms += [
                axial_term(3, 0),
                AberrationTerm("coma", (1, 2, 1, 0)),
                AberrationTerm("field_astigmatism", (0, 2, 2, 0), 0.5),
                AberrationTerm("field_curvature", (1, 1, 1, 1), 0.5),
                AberrationTerm("distortion", (0, 1, 2, 1)),
            ]
        return cls(terms)

    @classmethod
    def full(cls, order: int = 3) -> Self:
        """Every chi monomial whose deviation has degree <= ``order`` in (omega, w).

        For systems without symmetry (misaligned, multipole). Named ``o{a}{b}w{c}{d}``.
        """
        terms = []
        for a, b, c, d in product(range(order + 2), repeat=4):
            if not 1 <= a + b + c + d <= order + 1 or a + b == 0:
                continue
            if (b, a, d, c) < (a, b, c, d):  # Re[c f] and Re[c' conj(f)] are the same term
                continue
            terms.append(AberrationTerm(f"o{a}{b}w{c}{d}", (a, b, c, d)))
        return cls(terms)

    @property
    def names(self) -> list[str]:
        return [t.name for t in self.terms]

    def __iter__(self) -> Iterator[AberrationTerm]:
        return iter(self.terms)

    def __len__(self) -> int:
        return len(self.terms)

    def __getitem__(self, name: str) -> AberrationTerm:
        return self.terms[self.names.index(name)]

    def design_matrix(
        self, omega: torch.Tensor, w: torch.Tensor
    ) -> tuple[torch.Tensor, list[tuple[str, bool]]]:
        """Complex ``(N, P)`` design matrix and, per column, ``(term name, is_imag_part)``."""
        columns, index = [], []
        for term in self.terms:
            cols = term.columns(omega, w)
            columns += cols
            index += [(term.name, i == 1) for i in range(len(cols))]
        return torch.stack(columns, dim=-1), index

    def __repr__(self) -> str:
        return f"AberrationBasis({self.names})"
