"""Sampled fields from FEM simulations (COMSOL, FEMM, ...). STUBS.

Two cases, in order of how soon they are likely to be needed:

- ``SampledAxialField``: an on-axis profile B_z(z) or phi(z) exported from an axisymmetric FEM
  model. Differentiate it with a smoothing spline or a Hermite-function fit, not finite
  differences: the non-paraxial expansion needs derivatives up to order 2N + 1.
- ``FieldMap3D``: E and B on a regular 3D grid, for elements without rotational symmetry
  (multipoles, misaligned pole pieces). Interpolate with
  ``torch.nn.functional.grid_sample`` (trilinear, differentiable), and trace it with the
  time-domain Lorentz solver (``SolverParams.Lorentz``, Boris pusher).
"""

from typing import Self, Tuple

import numpy as np
import torch

from em_ray_tracer.constants import DTYPE
from em_ray_tracer.fields.base import AxialField, Field3D, FieldKind


def cheb_derivative_coeffs(c):
    """c: (..., order+1) -> coefficients of d/dx in the same Chebyshev basis (last entry 0)."""
    max_order = c.shape[-1] - 1
    d_coeff = torch.zeros_like(c)
    for k in range(max_order, 0, -1):
        higher = d_coeff[..., k + 1] if k + 1 <= max_order else 0.0
        d_coeff[..., k - 1] = higher + 2 * k * c[..., k]
    d_coeff[..., 0] = d_coeff[..., 0] / 2
    return d_coeff



class SampledAxialField(AxialField):
    """An axial profile sampled at ``z_samples``."""

    def __init__(self, z_samples: torch.Tensor, values: torch.Tensor, kind_set: FieldKind|Tuple[FieldKind,...]):
        super().__init__()
        values = torch.as_tensor(np.asarray(values),dtype=DTYPE)
        z_samples=torch.as_tensor(z_samples,dtype=DTYPE)

        # --- normalize kind_set to shape (FieldKind,) ---
        if not isinstance(kind_set,tuple):
            kind_set=(kind_set,)


        # --- format check ---
        if  len(z_samples) != values.shape[-1]:
            raise TypeError(f"shape mismatch between sampled z and field values, got {len(z_samples)} and {values.shape[-1]} respectively")

        #normalize all values to shape (n_fields,n_z)
        if values.shape==z_samples.shape :
            values=values.unsqueeze(0)
        if values.shape[0] != len(kind_set): 
            raise ValueError(f"number of field kinds specified does not agree with number of field values provided, values is {values.shape[0]} and kind_set is {len(kind_set)}")

        
        # --- determine ComponentFieldKind ---
        match sorted(kind_set):
            case ("magnetic",) | ("electrostatic",):
                componentkind=kind_set[0]
            case ("electrostatic","magnetic"):
                componentkind="static-electromagnetic"           
            case _:
                raise ValueError(f"unrecognized fields or combination of fields: {kind_set}")

        
        # --- setup dictionaries for easy indexing and field derivatives ---
        self._row={kind:index for index, kind in enumerate(kind_set)}
        
        self.smoothed={}
        for field_key in kind_set:
            self.smoothed[field_key]=None



        self.register_buffer("z_samples", z_samples)
        self.register_buffer("values", values)
        self.kind_set=kind_set
        self.kind=componentkind

                


    @classmethod
    def from_file(cls, path: str, kind_set: FieldKind | Tuple[FieldKind, ...], delimiter: str | None = None) -> Self:
        """this method accepts column txt or csv files with (z [m], values ... ), e.g. a COMSOL line-graph export.\n
        For multiple field kinds ensure that kind_set consists of a tuple of field names in the same order as the respective columns, example:
                kind_set=("magnetic","electrostatic")
        """
        data = np.loadtxt(fname=path, delimiter=delimiter)
        z_samples=data[...,0]
        field_samples=data[...,1:]

        #invert field_samples from shape (n_z,n_fields) to (n_fields,n_z)
        field_samples = [list(row) for row in zip(*field_samples)]
        return cls(z_samples, field_samples, kind_set)


    @property
    def half_extent(self) -> float:
        return float((self.z_samples[-1] - self.z_samples[0]) / 2)


    # ---------- B-spline internals ----------
    @staticmethod
    def _basis_funs(t: torch.Tensor, x: torch.Tensor, p: int):
        """Values of the p+1 nonzero B-splines at each x.
        Returns span index k (m,) and values (m, p+1) for B_{k-p..k, p}(x)."""
        n_basis = len(t) - p - 1
        k = (torch.searchsorted(t, x, right=True) - 1).clamp(p, n_basis - 1)

        N = torch.ones(len(x), 1, dtype=x.dtype, device=x.device)
        left = [None] * (p + 1)
        right = [None] * (p + 1)
        for j in range(1, p + 1):
            left[j] = x - t[k + 1 - j]
            right[j] = t[k + j] - x
            saved = torch.zeros_like(x)
            cols = []
            for r in range(j):
                temp = N[:, r] / (right[r + 1] + left[j - r])
                cols.append(saved + right[r + 1] * temp)
                saved = left[j - r] * temp
            cols.append(saved)
            N = torch.stack(cols, dim=1)
        return k, N

    @staticmethod
    def _design_matrix(t: torch.Tensor, x: torch.Tensor, p: int) -> torch.Tensor:
        n_basis = len(t) - p - 1
        k, N = SampledAxialField._basis_funs(t, x, p)
        cols = (k - p).unsqueeze(1) + torch.arange(p + 1, device=x.device)
        B = torch.zeros(len(x), n_basis, dtype=x.dtype, device=x.device)
        return B.scatter(1, cols, N)

    @staticmethod
    def _make_knots(z: torch.Tensor, p: int, n_spans: int) -> torch.Tensor:
        """Clamped knot vector; interior knots placed at equal sample-count
        interval. It is assumed the input data has more points near features/non-linear sections"""
        idx = torch.linspace(0, len(z) - 1, n_spans + 1, dtype=z.dtype, device=z.device)[1:-1]
        lo, hi = idx.floor().int(), idx.ceil().int()
        frac = idx - lo
        interior = z[lo] * (1 - frac) + z[hi] * frac
        return torch.cat([z[:1].expand(p + 1), interior, z[-1:].expand(p + 1)])

    # ---------- B-spline main ----------
    @staticmethod
    def smooth_spline(z: torch.Tensor, field_values: torch.Tensor,
                      order: int = 5, window_size: int = 10,
                      lam: float = 0.0, penalty_order: int = 2,
                      extrapolate: str = "error"):
        """
        Penalized B-spline fit. Returns f(n, z0) = n-th derivative at (batched) z0.

        order        spline degree p
        window_size  approx. samples per knot span (>= order + 1)
        lam          P-spline smoothing strength (0 = plain regression spline)
        extrapolate  'error' | 'clamp' | 'polynomial'
        """
        p : int = order

        # --- setup checks ---
        if window_size < p + 1:
            raise ValueError("window_size must be at least order + 1")
        if torch.any(z[1:] <= z[:-1]):
            raise ValueError("z must be strictly ascending")
        if extrapolate not in ("error", "clamp", "polynomial"):
            raise ValueError("extrapolate must be 'error', 'clamp' or 'polynomial'")

        squeeze = (field_values.ndim == 1)
        y = field_values.unsqueeze(-1) if squeeze else field_values

        n_spans = max(1, len(z) // window_size)
        t = SampledAxialField._make_knots(z, p, n_spans) 
        n_basis = len(t) - p - 1
        if n_basis > len(z):
            raise ValueError("more basis functions than samples, increase window_size")

        # --- calculation of coefficients ---
        B = SampledAxialField._design_matrix(t, z, p)
        A = B.T @ B
        if lam > 0:
            D = torch.diff(torch.eye(n_basis, dtype=z.dtype, device=z.device),
                           n=penalty_order, dim=0)
            A = A + lam * D.T @ D
        L, info = torch.linalg.cholesky_ex(A)
        if info.item() != 0:
            raise ValueError(
                "Normal equations are singular/ill-conditioned: too few samples "
                "per knot span (Schoenberg-Whitney). Increase window_size, "
                "set lam > 0, or use float64.")
        coeff = torch.cholesky_solve(B.T @ y, L)          # (n_basis, C)

        lo, hi = t[0], t[-1]

        def generalized_derivative(n: int, z0) -> torch.Tensor:
            z0 = torch.as_tensor(z0, dtype=z.dtype, device=z.device)
            scalar = (z0.ndim == 0)
            z0 = z0.reshape(-1)     #flattens

            if ((z0 < lo) | (z0 > hi)).any():
                if extrapolate == "error":
                    raise ValueError("z0 outside sample range")
                if extrapolate == "clamp":
                    z0 = z0.clamp(lo, hi)

            if n > p:
                out = torch.zeros(len(z0), y.shape[-1], dtype=z.dtype, device=z.device)
            else:
                # derivative of a spline, convert coefficients to lower degree
                tn, cn, pn = t, coeff, p
                for _ in range(n):
                    dt = (tn[pn + 1: pn + len(cn)] - tn[1: len(cn)]).unsqueeze(-1)
                    cn = pn * (cn[1:] - cn[:-1]) / dt
                    tn = tn[1:-1]
                    pn -= 1
                k, Nv = SampledAxialField._basis_funs(tn, z0, pn)
                idx = (k - pn).unsqueeze(1) + torch.arange(pn + 1, device=z.device)
                out = (Nv.unsqueeze(-1) * cn[idx]).sum(dim=1)   # (m, C)

            if squeeze:
                out = out[:, 0]
            return out[0] if scalar else out

        return generalized_derivative
    


    def derivative(self, n: int, z: torch.Tensor, kind_requested: FieldKind|None = None) -> torch.Tensor:
        if kind_requested is None :
            if len(self.kind_set)!=1:
                raise ValueError(f"kind_requested is required when multiple fields are defined: {self.kind_set}")
            kind_requested=self.kind_set[0]
        
        if kind_requested not in self._row.keys():
            raise TypeError(f"Requested kind not in set of fields defined, requested kind: {kind_requested}, set of fields: {self.kind_set}")
        
        if not self.smoothed[kind_requested]:
            kind_index=self._row[kind_requested]
            self.smoothed[kind_requested]=self.smooth_spline(self.z_samples,self.values[kind_index])
        
        return self.smoothed[kind_requested](n,z)


class FieldMap3D(Field3D):
    """E and B sampled on a regular grid with the given ``origin`` and ``spacing`` [m]."""

    def __init__(
        self,
        E: torch.Tensor | None,
        B: torch.Tensor | None,
        origin: tuple[float, float, float],
        spacing: tuple[float, float, float],
    ):
        super().__init__()
        self.register_buffer("E", None if E is None else torch.as_tensor(E, dtype=DTYPE))
        self.register_buffer("B", None if B is None else torch.as_tensor(B, dtype=DTYPE))
        self.origin = origin
        self.spacing = spacing

    @classmethod
    def from_npz(cls, path: str) -> Self:
        """An ``.npz`` with arrays ``E`` and/or ``B`` of shape (nx, ny, nz, 3), plus ``origin``
        and ``spacing``."""
        data = np.load(path)
        return cls(
            E=data.get("E"),
            B=data.get("B"),
            origin=tuple(data["origin"]),
            spacing=tuple(data["spacing"]),
        )

    def evaluate(
        self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError("FieldMap3D.evaluate: interpolate with F.grid_sample")
