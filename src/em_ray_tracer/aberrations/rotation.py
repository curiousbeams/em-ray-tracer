"""Image rotation fitted from the rays (the paraxial Larmor angle is
``OpticalSystem.larmor_angle``)."""

import torch


def _wrap_pi(theta):
    """Wrap to [-pi/2, pi/2), period pi."""
    return torch.remainder(theta + torch.pi / 2, torch.pi) - torch.pi / 2


# region --- direct fit ---
@torch.no_grad()
def fit_rotation(
    w: torch.Tensor,
    omega: torch.Tensor,
    w_out: torch.Tensor,
    weight: torch.Tensor | None = None,
    spread_tol: tuple[float, float] | None = None,
) -> torch.Tensor:
    if weight is None:
        weight = torch.ones_like(w.real)
    rho, nw, no = _joint_conditioning(w, omega, weight)

    if spread_tol is None:
        spread_tol = (1e-10, 1e-10)

    w_ok = nw > spread_tol[0]
    o_ok = no > spread_tol[1]

    if w_ok and o_ok and (rho < 0.9).all():
        theta = _fit_rotation_joint(w, omega, w_out, weight)
    elif o_ok and not w_ok:
        theta = _phase(omega, w_out, weight)
    elif w_ok and not o_ok:
        theta = _phase(w, w_out, weight)
    else:
        raise ValueError("Neither w nor omega has spread; the rotation is not determined.")

    return theta


def _fit_rotation_joint(w, omega, w_out, weight):
    sw = weight.sqrt().unsqueeze(-1)
    A = torch.stack([w, omega, torch.ones_like(w)], dim=-1) * sw  # (N, 3)
    c = torch.linalg.lstsq(A, (w_out.unsqueeze(-1) * sw)).solution.squeeze(-1)
    s = c[..., 0] ** 2 + c[..., 1] ** 2
    return 0.5 * torch.atan2(s.imag, s.real)  # defined modulo pi


def _joint_conditioning(w, omega, weight):
    def norm(x):
        return (x.abs().pow(2) * weight).sum(-1).sqrt()

    norm_w, norm_o, nweight = norm(w), norm(omega), norm(torch.ones_like(omega.real))
    cross = (w * omega.conj() * weight).sum(-1).abs()
    rho = cross / (norm_w * norm_o).clamp_min(1e-300)
    return rho, norm_w / nweight, norm_o / nweight


def _phase(ref, w_out, weight):
    z = (w_out * ref.conj() * weight).sum(-1)
    return _wrap_pi(torch.atan2(z.imag, z.real))


# endregion --- direct fit ---
