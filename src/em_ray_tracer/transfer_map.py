"""Polynomial transfer maps on the ray state, plus builders for the standard elements.

A first-order map is a 6x6 matrix acting on the homogeneous vector [x, x', y, y', delta, 1].
The trailing 1 lets constant kicks (deflectors, misaligned elements, beam shift/tilt) live in
the last column, so every first-order element, aligned or not, composes by plain matrix
multiplication. This is TEMGYM's 5x5 [x, x', y, y', 1] convention plus the energy deviation.

Higher orders (TRANSPORT convention) are stored in ``higher``: ``higher[0]`` is the
second-order tensor T[i, j, k], ``higher[1]`` the third-order U[i, j, k, l], and so on. They act
on the physical 5-vector only (no homogeneous 1), so the constant and linear parts stay in the
matrix and there is no double counting:

    out_i = M[i, :] @ [s, 1] + sum_jk T[i, j, k] s_j s_k + sum_jkl U[i, j, k, l] s_j s_k s_l
"""

from collections.abc import Sequence
from typing import Self

import torch

from em_ray_tracer.constants import DTYPE
from em_ray_tracer.rays import DELTA, MAP_DIM, STATE_DIM, STATE_NAMES, XP, YP, X, Y

ONE = STATE_DIM  # index of the homogeneous coordinate
_INDEX = {name: i for i, name in enumerate(STATE_NAMES)} | {"1": ONE}


class TransferMap:
    """A truncated polynomial map of the ray state (first order: a 6x6 homogeneous matrix).

    Parameters
    ----------
    matrix : torch.Tensor
        ``(..., 6, 6)`` homogeneous first-order matrix. Leading batch dimensions (e.g. a sweep
        over lens strengths) broadcast against the rays.
    higher : Sequence[torch.Tensor]
        Terms of order 2, 3, ... acting on the 5-vector ``[x, x', y, y', delta]``.
    """

    def __init__(self, matrix: torch.Tensor, higher: Sequence[torch.Tensor] = ()):
        if matrix.shape[-2:] != (MAP_DIM, MAP_DIM):
            raise ValueError(f"matrix must be (..., {MAP_DIM}, {MAP_DIM}), got {matrix.shape}")
        self.matrix = matrix
        self.higher = tuple(higher)

    @classmethod
    def identity(cls, dtype: torch.dtype = DTYPE, device=None) -> Self:
        return cls(torch.eye(MAP_DIM, dtype=dtype, device=device))

    @classmethod
    def from_affine(cls, linear: torch.Tensor, offset: torch.Tensor) -> Self:
        """Build from the rayTEM-style pair r' = linear @ r + offset."""
        batch = torch.broadcast_shapes(linear.shape[:-2], offset.shape[:-1])
        matrix = torch.zeros(*batch, MAP_DIM, MAP_DIM, dtype=linear.dtype, device=linear.device)
        matrix[..., :STATE_DIM, :STATE_DIM] = linear
        matrix[..., :STATE_DIM, ONE] = offset
        matrix[..., ONE, ONE] = 1
        return cls(matrix)

    @property
    def order(self) -> int:
        return 1 + len(self.higher)

    @property
    def linear(self) -> torch.Tensor:
        return self.matrix[..., :STATE_DIM, :STATE_DIM]

    @property
    def offset(self) -> torch.Tensor:
        return self.matrix[..., :STATE_DIM, ONE]

    def entry(self, row: str, col: str) -> torch.Tensor:
        """Matrix element by name, e.g. ``entry("x", "x'")`` or ``entry("x'", "1")``."""
        return self.matrix[..., _INDEX[row], _INDEX[col]]

    def apply(self, state: torch.Tensor) -> torch.Tensor:
        """Map a ``(..., N, 5)`` state tensor."""
        out = state @ self.linear.mT + self.offset.unsqueeze(-2)
        for degree, tensor in enumerate(self.higher, start=2):
            letters = "jklmnop"[:degree]
            operands = ",".join(f"...N{c}" for c in letters)
            out = out + torch.einsum(f"...i{letters},{operands}->...Ni", tensor, *[state] * degree)
        return out

    def compose(self, first: "TransferMap") -> "TransferMap":
        """The map ``self o first``: apply ``first``, then ``self``."""
        if self.higher or first.higher:
            raise NotImplementedError(
                "Truncated composition of higher-order maps is not implemented yet. Trace "
                "element by element instead (TransferMatrixSolver does), or implement the "
                "order-2 composition rule here: T = M2 T1 + T2 (M1 x M1)."
            )
        return TransferMap(self.matrix @ first.matrix)

    def __matmul__(self, first: "TransferMap") -> "TransferMap":
        return self.compose(first)

    def __repr__(self) -> str:
        return f"TransferMap(order={self.order}, batch_shape={tuple(self.matrix.shape[:-2])})"


# region --- element builders ---
# All builders take floats or tensors (optionally batched); gradients flow through every entry.


def _ref(*values) -> tuple[torch.dtype, torch.device | None]:
    for v in values:
        if isinstance(v, torch.Tensor):
            return (v.dtype if v.is_floating_point() else DTYPE), v.device
    return DTYPE, None


def _scalar(v, dtype, device) -> torch.Tensor:
    """A (possibly batched) scalar, shaped to broadcast against (..., 6, 6)."""
    return torch.as_tensor(v, dtype=dtype, device=device)[..., None, None]


def _unit(i: int, j: int, dtype, device) -> torch.Tensor:
    e = torch.zeros(MAP_DIM, MAP_DIM, dtype=dtype, device=device)
    e[i, j] = 1
    return e


def drift_map(length) -> TransferMap:
    """Field-free drift. Exact at every order: with slopes as coordinates a drift is linear."""
    dtype, device = _ref(length)
    eye = torch.eye(MAP_DIM, dtype=dtype, device=device)
    L = _scalar(length, dtype, device)
    return TransferMap(eye + L * (_unit(X, XP, dtype, device) + _unit(Y, YP, dtype, device)))


def rotation_map(angle) -> TransferMap:
    """Rotate positions and slopes about the optic axis by ``angle`` (x toward y)."""
    dtype, device = _ref(angle)
    theta = torch.as_tensor(angle, dtype=dtype, device=device)
    c, s = _scalar(torch.cos(theta), dtype, device), _scalar(torch.sin(theta), dtype, device)

    def u(i, j):
        return _unit(i, j, dtype, device)

    fixed = u(DELTA, DELTA) + u(ONE, ONE)
    diag = u(X, X) + u(Y, Y) + u(XP, XP) + u(YP, YP)
    skew = u(Y, X) - u(X, Y) + u(YP, XP) - u(XP, YP)
    return TransferMap(fixed + c * diag + s * skew)


def thin_lens_map(power, rotation=0.0) -> TransferMap:
    """Ideal round thin lens of ``power`` = 1/f, followed by an image rotation (magnetic lens)."""
    dtype, device = _ref(power, rotation)
    eye = torch.eye(MAP_DIM, dtype=dtype, device=device)
    P = _scalar(power, dtype, device)
    lens = TransferMap(eye - P * (_unit(XP, X, dtype, device) + _unit(YP, Y, dtype, device)))
    if isinstance(rotation, torch.Tensor) or rotation != 0.0:
        return rotation_map(rotation) @ lens
    return lens


def quadrupole_map(power, orientation=0.0) -> TransferMap:
    """Thin quadrupole: focusing ``power`` in x, defocusing in y, rotated by ``orientation``."""
    dtype, device = _ref(power, orientation)
    eye = torch.eye(MAP_DIM, dtype=dtype, device=device)
    P = _scalar(power, dtype, device)
    quad = TransferMap(eye - P * (_unit(XP, X, dtype, device) - _unit(YP, Y, dtype, device)))
    if isinstance(orientation, torch.Tensor) or orientation != 0.0:
        return rotation_map(orientation) @ quad @ rotation_map(-orientation)
    return quad


def deflection_map(deflection_x, deflection_y, dispersion=0.0) -> TransferMap:
    """Constant slope kick (alpha_x, alpha_y), in the homogeneous column.

    ``dispersion`` is d(ln alpha)/d(delta): an off-energy ray is deflected by
    alpha (1 + dispersion * delta). It is -gamma/(1+gamma) for a magnetic deflector
    (alpha ~ 1/p) and -(gamma^2+1)/(gamma(gamma+1)) for an electrostatic one (alpha ~ 1/pv).
    """
    dtype, device = _ref(deflection_x, deflection_y, dispersion)
    eye = torch.eye(MAP_DIM, dtype=dtype, device=device)
    ax = _scalar(deflection_x, dtype, device)
    ay = _scalar(deflection_y, dtype, device)
    k = _scalar(dispersion, dtype, device)
    matrix = (
        eye
        + ax * (_unit(XP, ONE, dtype, device) + k * _unit(XP, DELTA, dtype, device))
        + ay * (_unit(YP, ONE, dtype, device) + k * _unit(YP, DELTA, dtype, device))
    )
    return TransferMap(matrix)


def translation_map(dx=0.0, dy=0.0, dxp=0.0, dyp=0.0) -> TransferMap:
    """Shift positions by (dx, dy) and slopes by (dxp, dyp)."""
    dtype, device = _ref(dx, dy, dxp, dyp)
    eye = torch.eye(MAP_DIM, dtype=dtype, device=device)
    terms = [(dx, X), (dy, Y), (dxp, XP), (dyp, YP)]
    matrix = eye
    for value, row in terms:
        matrix = matrix + _scalar(value, dtype, device) * _unit(row, ONE, dtype, device)
    return TransferMap(matrix)


# endregion --- element builders ---
