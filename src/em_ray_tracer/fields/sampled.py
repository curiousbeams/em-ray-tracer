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

from typing import Self

import numpy as np
import torch

from em_ray_tracer.constants import DTYPE
from em_ray_tracer.fields.base import AxialField, Field3D, FieldKind


class SampledAxialField(AxialField):
    """An axial profile sampled at ``z_samples``."""

    def __init__(self, z_samples: torch.Tensor, values: torch.Tensor, kind: FieldKind):
        super().__init__()
        self.register_buffer("z_samples", torch.as_tensor(z_samples, dtype=DTYPE))
        self.register_buffer("values", torch.as_tensor(values, dtype=DTYPE))
        self._kind = kind

    @property
    def kind(self) -> FieldKind:  # type: ignore[override]
        return self._kind

    @classmethod
    def from_file(cls, path: str, kind: FieldKind, delimiter: str | None = None) -> Self:
        """Two-column text file (z [m], value [T or V]), e.g. a COMSOL line-graph export."""
        data = np.loadtxt(path, delimiter=delimiter)
        return cls(data[:, 0], data[:, 1], kind)

    @property
    def half_extent(self) -> float:
        return float((self.z_samples[-1] - self.z_samples[0]) / 2)

    def derivative(self, n: int, z: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("SampledAxialField.derivative: fit a smooth interpolant first")


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
