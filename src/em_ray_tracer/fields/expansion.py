"""Off-axis expansion of an axisymmetric field from its axial profile. STUB.

Laplace's equation fixes the off-axis field term by term from the axial profile F(z):

    F_z = sum_{n=0..N}  (-1)^n / (2^(2n) (n!)^2)       r^(2n)   F^(2n)(z)
    F_x = sum_{n=1..N}  (-1)^n / (2^(2n-1) (n-1)! n!)  x r^(2n-2) F^(2n-1)(z)
    F_y = the same with y in place of x

Truncating at N leaves exactly one unpaired term in the divergence,
(-1)^N / (2^(2N) (N!)^2) r^(2N) F^(2N+1)(z); asserting that identity is a good test.

Order 1 is the paraxial field and order 3 is where spherical aberration appears. Truncating the
longitudinal sum one order below the transverse one gives the strictly linearized field (the
``r^2 B''`` term in B_z is most of a magnetic lens's spherical aberration). Reference
implementation: ``laplaceExpansion`` in the em-widgets kit (kit/nonparaxial.js).
"""

import torch

from em_ray_tracer.fields.base import AxialField, Field3D


class LaplaceExpansion(Field3D):
    """The 3D field of ``axial`` expanded to ``order`` (transverse) and
    ``longitudinal_order`` (defaults to ``order``)."""

    def __init__(self, axial: AxialField, order: int = 3, longitudinal_order: int | None = None):
        super().__init__()
        self.axial = axial
        self.order = order
        self.longitudinal_order = order if longitudinal_order is None else longitudinal_order

    def evaluate(
        self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError("LaplaceExpansion.evaluate: port laplaceExpansion")
