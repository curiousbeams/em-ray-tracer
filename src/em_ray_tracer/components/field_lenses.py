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

from typing import Callable, Literal, Self

import torch

from em_ray_tracer.components.base import ComponentBase
from em_ray_tracer.fields.analytic import AxialAnalytical, GlaserField, SchiskeField
from em_ray_tracer.fields.base import AxialField, FieldKind
from em_ray_tracer.fields.expansion import LaplaceExpansion
from em_ray_tracer.fields.sampled import SampledAxialField
from em_ray_tracer.rays import STATE_DIM, Rays
from em_ray_tracer.system import OpticalSystem
from em_ray_tracer.transfer_map import TransferMap
from em_ray_tracer.utilities.utils import as_parameter


class FieldComponent(ComponentBase):
    """Base for elements with an extent and a field to integrate through."""

    def __init__(self, field: AxialField, *, z: float, **kwargs):
        super().__init__(z=z, **kwargs)
        self.field = field
        self.z_start_rel_cut=None
        self.z_end_rel_cut=None
        self.laplace=None

    @property
    def z_start(self) -> torch.Tensor:
        if self.z_start_rel_cut:
            return self.z + self.z_start_rel_cut
        return self.z - self.field.half_extent

    @property
    def z_end(self) -> torch.Tensor:
        if self.z_end_rel_cut:
            return self.z + self.z_end_rel_cut
        return self.z + self.field.half_extent

    @z_start.setter
    def z_start(self,value) :
        self.z_start_rel_cut=as_parameter(value)-self.z

    @z_end.setter
    def z_end(self,value) :
        self.z_end_rel_cut=as_parameter(value)-self.z


    def axial_derivative(self, n: int, z: torch.Tensor | float) -> dict:
        """The n-th derivative of the axial profile at the absolute position ``z`` (n = 0 is
        the profile). The field's own coordinate is measured from the lens centre."""
        return self.field.derivative(n, z - self.z)

    def axial_laplace(self, order: int = 3, longitudinal_order: int | None = None) -> Callable:
            """The n-th order laplace expansion of the axial profile at the absolute z position within the complete system.
            The field's own coordinate is measured from the lens centre.\n
            Returns a function which takes (x,y,z) and returns the E, B fields\n
            If the optional voltage is specified it returns E, B, phi_absolute instead"""

            self.laplace=LaplaceExpansion(self.field, component=self, order = order, longitudinal_order = longitudinal_order)

    def field_values(self,x,y,z,voltage = None) -> tuple[torch.Tensor,torch.Tensor]|tuple[torch.Tensor,torch.Tensor,torch.Tensor]:
        """returns E, B \n
        If voltage (of the electron) is also specified it returns E, B, Phi instead"""
        if not self.laplace:
            raise RuntimeError("call axial_laplace first before using the call function")
        z_field=z-self.z
        return self.laplace.evaluate(x,y,z_field,voltage=voltage)
        
    def config(self):
        raise NotImplementedError("config() for field components is not implemented yet")


class FieldLens(FieldComponent):
    """A round lens defined by its axial field (magnetic or electrostatic)."""

    @classmethod
    def glaser(cls, B0: float, a: float, *, z: float, extent: float = 10.0, **kwargs) -> Self:
        """Magnetic lens with Glaser's bell-shaped field (see :class:`GlaserField`)."""
        return cls(GlaserField(B0=B0, a=a, extent=extent), z=z, **kwargs)

    @classmethod
    def schiske(cls, k: float, a: float, voltage: float, *, z: float, extent: float = 10.0, **kwargs) -> Self:
        """Electrostatic einzel lens with Schiske's potential (see :class:`SchiskeField`)."""
        return cls(SchiskeField(k=k, a=a, extent=extent,voltage=voltage), z=z, **kwargs)

    @classmethod
    def from_axial_samples(cls, z_samples, values, *, kind_set: FieldKind|tuple[FieldKind,...], z: float, grounded_check = True,**kwargs) -> Self:
        """From an on-axis profile exported by an FEM model (``z_samples`` relative to ``z``)."""
        return cls(SampledAxialField(z_samples, values, kind_set,grounded_check=grounded_check), z=z, **kwargs)

    @classmethod
    def from_file_with_axial_samples(cls, path: str, kind_set: FieldKind | tuple[FieldKind, ...],z: float, delimiter: str | None = None, grounded_check = True, B_kwargs: dict = {},**kwargs) -> Self:
        """Build a lens from axial field samples stored in a text or CSV file.

        The file must have columns ``(z [m], values, ...)``, e.g. a COMSOL
        line-graph export. A B-spline is fitted to each value column so that
        smooth axial derivatives are available.

        Parameters
        ----------
        path : str
            Path to the column text or CSV file.
        kind_set : FieldKind or tuple of FieldKind
            Field kind of each value column, in the same order as the columns.
            For several kinds pass a tuple, e.g.
            ``("magnetic", "electrostatic")``.
        z : float
            Position of the lens along the optical axis [m].
        delimiter : str, optional
            Column delimiter used in the file.
        grounded_check : bool, default True
            Whether to check that the field is grounded at the ends of the
            sampled range.
        B_kwargs : dict
            Passed to the B-spline fit. 
            See "Other Parameters" below which adjust parameters related to the underlying B-spline interpolation.
        
        Other parameters
        ----------
        order : int, default 5
            Spline degree p.
            lower the order for fewer sample points
        window_size : int, default 10
            Approximate number of samples per knot span (>= ``order + 1``).
        lam : float, default 0
            P-spline smoothing strength (0 = plain regression spline).
        extrapolate : {'error', 'clamp', 'polynomial'}, default error
            Behaviour outside the sampled z-range.

        Returns
        -------
        Self
            The lens, whose axial derivatives follow the fitted spline.
        """
        return cls(SampledAxialField.from_file(path, kind_set, delimiter, grounded_check = grounded_check,**B_kwargs),z=z,**kwargs)

    @classmethod 
    def from_axial_analytical(cls,
                              axial:Callable[[torch.Tensor],torch.Tensor],
                              fieldkind : Literal["electrostatic"]|Literal["magnetic"]|tuple[str,...],
                              z : float,
                              half_extent : float,
                              **kwargs
                              ) -> Self:
        """On-axis profile given as a function of a single (0-d) z.

        ``axial`` maps a 0-d tensor z to a 0-d tensor (single field kind) or to a
        tensor of shape (K,) (one entry per name in ``fieldkind``, same order).
        It must be written with torch ops only (no .item(), no in-place ops).
        """
        return cls(AxialAnalytical(axial=axial,fieldkind=fieldkind,half_extent=half_extent),z=z,**kwargs)

    def local_transfer_map(self, order: int = 1, voltage: float | None = None) -> TransferMap:
        """Constructs local transfer map from the for principal rays.
        order 1 uses paraxial while higher orders use non-paraxial laplace expansion, all use RK4.
        
        Higher order transfer maps may benefit from adjusting r in order to
        asses around a relevant order of magnitude and prevent blow up."""
        from em_ray_tracer.solvers.ode import ODESolver
        
        
        principal=Rays.principal(z=float(self.z_start),voltage=voltage,r_interest=self.r_interest)
        system=OpticalSystem.from_components(self)
        if self.r_interest is not None:
            scale=self.r_interest
        else:
            scale=1

        
        if order==1:
            out_rays=ODESolver.paraxial().trace(system, principal).final.state
        if order > 1:
            out_rays=ODESolver.nonparaxial(expansion_order=order).trace(system, principal).final.state

        out_states=torch.cat([out_rays[:,:STATE_DIM]/scale,torch.zeros(STATE_DIM).unsqueeze(0)])
        local_map=TransferMap.from_affine(out_states.mT,torch.zeros(5))
        return local_map
        
