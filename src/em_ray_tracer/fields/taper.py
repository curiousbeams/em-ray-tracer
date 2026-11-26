"""Smooth ends for field components.

A field that is cut off where it is still nonzero (a Glaser field at +-10a is 1% of its peak)
is not a field Maxwell's equations allow: the step in B needs a radial field of zero width to
go with it. The ray equations see only the step, so a ray crossing it gains a spurious
rotation, and a ray from a point on the axis no longer returns to the axis.

``FieldComponent`` therefore multiplies its profile by a window that is 1 in the core and falls
smoothly to 0 at the component's extent. The window is the C^N smoothstep

    S(t) = (1/B) integral_0^t u^N (1 - u)^N du,   B = N!^2 / (2N + 1)!,

so every derivative up to the N-th is continuous at both ends; the order-7 expansion needs
derivatives up to the 8th. It is evaluated in product form rather than as one degree-(2N + 1)
polynomial, whose alternating coefficients lose most of their digits near t = 1.
"""

from math import comb, factorial

import torch

SMOOTHNESS = 9  # S is C^9: derivatives 1..9 vanish at both ends
_NORM = factorial(2 * SMOOTHNESS + 1) / factorial(SMOOTHNESS) ** 2  # 1 / B


def _falling(n: int, k: int) -> int:
    """n! / (n - k)!, the k-th derivative's coefficient of x^n."""
    return factorial(n) // factorial(n - k)


def _step(u: torch.Tensor) -> torch.Tensor:
    """S(u) for 0 <= u <= 1/2, from its expansion about 0 (no cancellation there)."""
    N = SMOOTHNESS
    acc = torch.zeros_like(u)
    for i in range(N + 1):
        acc = acc + comb(N, i) * (-1) ** i * u ** (N + 1 + i) / (N + 1 + i)
    return _NORM * acc


def smoothstep(m: int, t: torch.Tensor) -> torch.Tensor:
    """The m-th derivative of S at ``t``; S is 0 for t <= 0 and 1 for t >= 1."""
    N = SMOOTHNESS
    inside = (t > 0) & (t < 1)
    u = t.clamp(0.0, 1.0)
    if m == 0:
        # each half from its own end; torch.where sends the gradient through one branch only
        # (a min(u, 1 - u) would split it at u = 1/2)
        return torch.where(u <= 0.5, _step(u), 1 - _step(1 - u))
    # S^(m) = (1/B) d^(m-1)/dt^(m-1) [t^N (1 - t)^N], by Leibniz
    acc = torch.zeros_like(u)
    for j in range(m):
        k = m - 1 - j  # derivatives on (1 - t)^N
        if j > N or k > N:
            continue
        acc = acc + (
            comb(m - 1, j)
            * _falling(N, j)
            * _falling(N, k)
            * (-1) ** k
            * u ** (N - j)
            * (1 - u) ** (N - k)
        )
    return torch.where(inside, _NORM * acc, torch.zeros_like(acc))
