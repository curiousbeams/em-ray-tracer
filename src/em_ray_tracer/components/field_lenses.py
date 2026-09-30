"""Extended (thick) elements defined by a field rather than a matrix. The transfer map is a STUB.

A ``FieldLens`` occupies [z - half_extent, z + half_extent]. Solvers handle it in two ways:

- ``ODESolver`` integrates the ray equation through the field directly.
- ``TransferMatrixSolver`` asks for ``transfer_map``, the thick-lens matrix. At first order
  this means integrating the principal rays g (g = 1, g' = 0) and h (h = 0, h' = 1) with the
  paraxial solver: their final values are the columns of the radial 2x2 block. A magnetic
  lens then adds its Larmor rotation, theta = integral of eta B / (2 sqrt(phi_hat)) dz, which
  couples x and y in the lab frame. Higher orders need either differential algebra (TEMGYM
  Advanced's approach) or a polynomial fit to traced rays.

The field's own z coordinate is measured from the lens centre ``z``.
"""

from typing import Self

import torch

from em_ray_tracer.components.base import ComponentBase
from em_ray_tracer.fields.analytic import GlaserField, SchiskeField
from em_ray_tracer.fields.base import AxialField
from em_ray_tracer.fields.sampled import SampledAxialField
from em_ray_tracer.transfer_map import TransferMap


class FieldComponent(ComponentBase):
    """Base for elements with an extent and a field to integrate through."""

    def __init__(self, field: AxialField, *, z: float, **kwargs):
        super().__init__(z=z, **kwargs)
        self.field = field

    @property
    def z_start(self) -> torch.Tensor:
        return self.z - self.field.half_extent

    @property
    def z_end(self) -> torch.Tensor:
        return self.z + self.field.half_extent

    def config(self):
        raise NotImplementedError("config() for field components is not implemented yet")


class FieldLens(FieldComponent):
    """A round lens defined by its axial field (magnetic or electrostatic)."""

    @classmethod
    def glaser(cls, B0: float, a: float, *, z: float, extent: float = 10.0, **kwargs) -> Self:
        """Magnetic lens with Glaser's bell-shaped field (see :class:`GlaserField`)."""
        return cls(GlaserField(B0=B0, a=a, extent=extent), z=z, **kwargs)

    @classmethod
    def schiske(cls, k: float, a: float, *, z: float, extent: float = 10.0, **kwargs) -> Self:
        """Electrostatic einzel lens with Schiske's potential (see :class:`SchiskeField`)."""
        return cls(SchiskeField(k=k, a=a, extent=extent), z=z, **kwargs)

    @classmethod
    def from_axial_samples(cls, z_samples, values, *, kind: str, z: float, **kwargs) -> Self:
        """From an on-axis profile exported by an FEM model (``z_samples`` relative to ``z``)."""
        return cls(SampledAxialField(z_samples, values, kind), z=z, **kwargs)

    def local_transfer_map(self, order: int = 1, voltage: float | None = None) -> TransferMap:
        raise NotImplementedError(
            "FieldLens.local_transfer_map: integrate the principal rays g, h with the paraxial "
            "ODE (see the module docstring), or trace this lens with ODESolver instead."
        )
