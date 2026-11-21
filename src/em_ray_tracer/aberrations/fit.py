"""Differentiable least-squares fit of an aberration basis to traced rays."""

from typing import Literal

import torch

from em_ray_tracer.aberrations.basis import AberrationBasis
from em_ray_tracer.aberrations.rotation import fit_rotation
from em_ray_tracer.constants import DTYPE


def fit_coefficients(
    basis: AberrationBasis,
    omega: torch.Tensor,
    w: torch.Tensor,
    w_out: torch.Tensor,
    weight: torch.Tensor | None = None,
    rotation: torch.Tensor | Literal["standard"] = "standard",
    rcond: float = 1e-10,
    spread_tol: tuple[float,float]|None = None
) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    """Fit ``w_out ~ sum_k delta_w_k(omega, w)`` for the terms of ``basis``.

    Real and imaginary parts are stacked into one real least-squares problem (the model is
    real-linear, not complex-linear, in the coefficients). The problem is solved by QR on
    column-normalized data, which is differentiable, so fitted coefficients can be
    optimization targets. Columns are normalized because a constant shift and a cubic in
    10 mrad slopes differ by ~1e6 in scale.

    Returns ``(values, residual_rms)``: complex coefficients (real tensors for real terms),
    and the weighted RMS of |w_out - model|.

    Raises
    ------
    ValueError
        If the rays do not constrain some term, e.g. a meridional fan (y = y' = 0) cannot
        separate x- from y-astigmatism. The message names the terms.
    """
    if rotation == "standard":    
        rotation = fit_rotation(w,omega,w_out,weight=weight,spread_tol=spread_tol)
    else :
        rotation = torch.as_tensor(rotation,dtype=DTYPE).detach()
    
    w_out=w_out*torch.exp(-1j*rotation) #de-rotation
    A, index = basis.design_matrix(omega, w)
    if weight is None:
        weight = torch.ones(w_out.shape, dtype=A.real.dtype, device=A.device)
    sw = weight.sqrt().unsqueeze(-1)
    A_real = torch.cat([A.real * sw, A.imag * sw], dim=-2)
    y = torch.cat([w_out.real * sw[..., 0], w_out.imag * sw[..., 0]], dim=-1)

    scale = A_real.detach().norm(dim=-2).clamp_min(torch.finfo(A_real.dtype).tiny)
    Q, R = torch.linalg.qr(A_real / scale)
    diag = R.diagonal(dim1=-2, dim2=-1).abs().detach()
    bad = (diag < rcond * diag.max()).nonzero().flatten().tolist()
    if bad:
        parts = [f"{index[i][0]} ({'imag' if index[i][1] else 'real'})" for i in bad]
        raise ValueError(f"The rays do not constrain {parts}; use a fuller bundle or basis.")
    x = torch.linalg.solve_triangular(R, (Q.mT @ y).unsqueeze(-1), upper=True).squeeze(-1)
    x = x / scale

    residual = y - A_real @ x
    n = residual.shape[-1] // 2
    residual_rms = torch.sqrt(
        (residual[..., :n] ** 2 + residual[..., n:] ** 2).sum(-1)
        / weight.sum(-1).clamp_min(1e-300)
    )

    re: dict[str, torch.Tensor] = {}
    im: dict[str, torch.Tensor] = {}
    for (name, is_imag), xi in zip(index, x.unbind(-1)):
        (im if is_imag else re)[name] = xi
    values = {name: torch.complex(re[name], im[name]) if name in im else re[name] for name in re}
    return values, residual_rms, rotation

    
    

