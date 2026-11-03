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
from em_ray_tracer.interpolation import b_spline


class SampledAxialField(AxialField):
    """An axial profile sampled at ``z_samples``."""

    def __init__(self, z_samples: torch.Tensor, values: torch.Tensor, kind_set: FieldKind|Tuple[FieldKind,...],grounded_check = True):
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

        # --- check grounded fields ---
        ignorable=torch.max(values,dim=-1).values/1000
        if grounded_check and (any(values[:,0]>ignorable) or any(values[:,-1]>ignorable)):
            raise ValueError("\nThe edges of supplied fields must go to 0\n"
                             f"     the lower is now: {values[:,0]}\n"
                             f"     the upper is now: {values[:,-1]}\n"
                             "if you are aware of the error then this check can be ignored by passing grounded_check = False")


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
    def from_file(cls, path: str, kind_set: FieldKind | Tuple[FieldKind, ...], delimiter: str | None = None, grounded_check = True) -> Self:
        """this method accepts column txt or csv files with (z [m], values ... ), e.g. a COMSOL line-graph export.\n
        For multiple field kinds ensure that kind_set consists of a tuple of field names in the same order as the respective columns, example:
                kind_set=("magnetic","electrostatic")
        """
        data = np.loadtxt(fname=path, delimiter=delimiter)
        z_samples=data[...,0]
        field_samples=data[...,1:]

        #invert field_samples from shape (n_z,n_fields) to (n_fields,n_z)
        field_samples = [list(row) for row in zip(*field_samples)]
        return cls(z_samples, field_samples, kind_set, grounded_check = grounded_check)


    @property
    def half_extent(self) -> float:
        return float((self.z_samples[-1] - self.z_samples[0]) / 2)
    


    def derivative(self, n: int, z: torch.Tensor, kind_requested: FieldKind|None = None) -> torch.Tensor:
        if kind_requested is None :
            if len(self.kind_set)!=1:
                raise ValueError(f"kind_requested is required when multiple fields are defined: {self.kind_set}")
            kind_requested=self.kind_set[0]
        
        if kind_requested not in self._row.keys():
            raise TypeError(f"Requested kind not in set of fields defined, requested kind: {kind_requested}, set of fields: {self.kind_set}")
        
        if not self.smoothed[kind_requested]:
            kind_index=self._row[kind_requested]
            self.smoothed[kind_requested]=b_spline(self.z_samples,self.values[kind_index])
        
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
