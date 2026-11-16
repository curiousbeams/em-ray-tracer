
"""Namespace holding different interpolation methods used throughout em-ray-tracer,
they allow compatibility with pytorch. \n
Supported interpolation methods:\n
    B-spline
"""
from dataclasses import dataclass

import torch


# ---------- B-spline internals ----------
def b_spline_basis_funs(t: torch.Tensor, x: torch.Tensor, p: int):
    """Values of the p+1 nonzero B-splines at each x.
    Returns span index k (m,) and values (m, p+1) for B_{k-p..k, p}(x)."""
    n_basis = len(t) - p - 1
    k = (torch.searchsorted(t, x.contiguous(), right=True) - 1).clamp(p, n_basis - 1)

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


def b_spline_design_matrix(t: torch.Tensor, x: torch.Tensor, p: int) -> torch.Tensor:
    n_basis = len(t) - p - 1
    k, N = b_spline_basis_funs(t, x, p)
    cols = (k - p).unsqueeze(1) + torch.arange(p + 1, device=x.device)
    B = torch.zeros(len(x), n_basis, dtype=x.dtype, device=x.device)
    return B.scatter(1, cols, N)


def b_spline_make_knots(z: torch.Tensor, p: int, n_spans: int) -> torch.Tensor:
    """Clamped knot vector; interior knots placed at equal sample-count
    interval. It is assumed the input data has more points near features/non-linear sections"""
    idx = torch.linspace(0, len(z) - 1, n_spans + 1, dtype=z.dtype, device=z.device)[1:-1]
    lo, hi = idx.floor().int(), idx.ceil().int()
    frac = idx - lo
    interior = z[lo] * (1 - frac) + z[hi] * frac
    return torch.cat([z[:1].expand(p + 1), interior, z[-1:].expand(p + 1)])

# ---------- B-spline main ----------
@dataclass
class b_spline:
    """
    B-spline fit. Returns f(n, z0) = n-th derivative at (batched) z0.

    order        spline degree p
    window_size  approx. samples per knot span (>= order + 1)
    lam          P-spline smoothing strength (0 = plain regression spline)
    extrapolate  'error' | 'clamp' | 'polynomial'
    """
    z: torch.Tensor
    field_values: torch.Tensor
    order: int = 5
    window_size: int = 10
    lam: float = 0.0
    penalty_order: int = 2
    extrapolate: str = "error"

    # --- setup checks ---
    def __post_init__(self):
    

        if self.window_size < self.order + 1:
            raise ValueError("window_size must be at least order + 1")
        if torch.any(self.z[1:] <= self.z[:-1]):
            raise ValueError("z must be strictly ascending")
        if self.extrapolate not in ("error", "clamp", "polynomial"):
            raise ValueError("extrapolate must be 'error', 'clamp' or 'polynomial'")

        self.squeeze = (self.field_values.ndim == 1)
        y = self.field_values.unsqueeze(-1) if self.squeeze else self.field_values

        n_spans = max(1, len(self.z) // self.window_size)
        t = b_spline_make_knots(self.z, self.order, n_spans) 
        n_basis = len(t) - self.order - 1
        if n_basis > len(self.z):
            raise ValueError("more basis functions than samples, increase window_size")

        # --- calculation of coefficients ---
        B = b_spline_design_matrix(t, self.z, self.order)
        A = B.T @ B
        if self.lam > 0:
            D = torch.diff(torch.eye(n_basis, dtype=self.z.dtype, device=self.z.device),
                            n=self.penalty_order, dim=0)
            A = A + self.lam * D.T @ D
        L, info = torch.linalg.cholesky_ex(A)
        if info.item() != 0:
            raise ValueError(
                "Normal equations are singular/ill-conditioned: too few samples "
                "per knot span (Schoenberg-Whitney). Increase window_size, "
                "set lam > 0 or use float64.")
        coeff = torch.cholesky_solve(B.T @ y, L)          # (n_basis, C)

        self.lo, self.hi = t[0], t[-1]
        self.y, self.t = y, t
        self.coeff=coeff

    def __call__(self,n: int, z0) -> torch.Tensor:
        z0 = torch.as_tensor(z0, dtype=self.z.dtype, device=self.z.device)
        scalar = (z0.ndim == 0)
        z0 = z0.reshape(-1)     #flattens

        if ((z0 < self.lo) | (z0 > self.hi)).any():
            if self.extrapolate == "error":
                raise ValueError("z0 outside sample range")
            if self.extrapolate == "clamp":
                z0 = z0.clamp(self.lo, self.hi)
            if self.extrapolate == "polynomial":
                raise NotImplementedError()

        if n > self.order:
            out = torch.zeros(len(z0), self.y.shape[-1], dtype=self.z.dtype, device=self.z.device)
        else:
            # derivative of a spline, convert coefficients to lower degree
            tn, cn, pn = self.t, self.coeff, self.order
            for _ in range(n):
                dt = (tn[pn + 1: pn + len(cn)] - tn[1: len(cn)]).unsqueeze(-1)
                cn = pn * (cn[1:] - cn[:-1]) / dt
                tn = tn[1:-1]
                pn -= 1
            k, Nv = b_spline_basis_funs(tn, z0, pn)
            idx = (k - pn).unsqueeze(1) + torch.arange(pn + 1, device=self.z.device)
            out = (Nv.unsqueeze(-1) * cn[idx]).sum(dim=1)   # (m, C)

        if self.squeeze:
            out = out[:, 0]
        return out[0] if scalar else out
