import torch

from em_ray_tracer.components import FieldLens, Plane
from em_ray_tracer.rays import Rays
from em_ray_tracer.system import OpticalSystem
from em_ray_tracer.visualization import Standardplots


def analytic(z : torch.Tensor):
    phi0=8e3
    Q=1e3
    K=1
    width=10e-3
    phi=phi0*(torch.exp(K*torch.arctan(Q*(z+width/2)))-torch.exp(K*torch.arctan(Q*(z-width/2))))
    return phi

lens=FieldLens.from_axial_analytical(axial=analytic,fieldkind="electrostatic",half_extent=50e-3,z=50e-3)
detector=Plane(z=100e-3,name="detector")
rays=Rays.from_disk(semiangle=4e-3,num_rays=256,voltage=1e4)
system=OpticalSystem.from_components(lens,detector)

Standardplots.multi_order(system=system,rays=rays,highest_order=5,show_fields=True,POLC_domain=(0.04,None),step_size=1e-5)
