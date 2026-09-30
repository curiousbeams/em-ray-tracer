"""Field interfaces: axial profiles of round lenses, and full 3D fields."""

from abc import ABC, abstractmethod
from typing import ClassVar, Literal

import torch
from torch import nn

FieldKind = Literal["magnetic", "electrostatic"]


class AxialField(nn.Module, ABC):
    """The on-axis profile of a rotationally symmetric lens: B_z(0, 0, z) [T] for a magnetic
    lens, or the potential phi(0, 0, z) [V] for an electrostatic one.

    The paraxial ray equation needs the profile and its first two derivatives. The
    non-paraxial expansion of order N (see ``fields.expansion``) needs derivatives up to
    2N + 1, which is why the interface is ``derivative(n, z)`` rather than the profile alone.
    Profile parameters (B0, a, ...) are ``nn.Parameter`` so they can be optimized.
    """

    kind: ClassVar[FieldKind]

    @abstractmethod
    def derivative(self, n: int, z: torch.Tensor) -> torch.Tensor:
        """The n-th z-derivative of the axial profile at ``z`` (n = 0 is the profile)."""

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.derivative(0, z)

    @property
    @abstractmethod
    def half_extent(self) -> float:
        """Half-width [m] beyond which the field is negligible; sets the integration span."""


class Field3D(nn.Module, ABC):
    """A full electromagnetic field, E(x, y, z) [V/m] and B(x, y, z) [T]."""

    @abstractmethod
    def evaluate(
        self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(E, B)``, each shaped ``(*x.shape, 3)``."""
