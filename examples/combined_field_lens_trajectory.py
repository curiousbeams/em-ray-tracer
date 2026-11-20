
from time import time

import numpy as np
import torch

from em_ray_tracer.components import FieldLens
from em_ray_tracer.components.thin import Aperture, Plane, ThinLens
from em_ray_tracer.rays import Rays
from em_ray_tracer.system import OpticalSystem
from em_ray_tracer.visualization import Standardplots

volt=1e4
GLASER = dict(B0=0.1, a=4e-3)
SCHISKE= dict(k=0.7, a=4e-3, voltage= volt )
delimiter=","

# --- create sample field file ---
lens_B=FieldLens.glaser(**GLASER,z=0)
lens_E=FieldLens.schiske(**SCHISKE,z=-20e-3)
z_sample_space=torch.linspace(-60e-3,60e-3,10**4)

B_sampled=lens_B.axial_derivative(0,z_sample_space)["magnetic"]
B_sampled=B_sampled# + max(B_sampled)*torch.rand_like(B_sampled)*1e-3 #to see the effect of noise
Phi_sampled=lens_E.axial_derivative(0,z_sample_space)["electrostatic"]
Phi_sampled=Phi_sampled #+ max(Phi_sampled)*torch.rand_like(Phi_sampled)*1e-3

np.savetxt("examples/combined_field_lens_sampled.csv",torch.stack([z_sample_space,B_sampled,Phi_sampled],dim=-1),delimiter=delimiter)



# --- read from file and setup system ---
t0=time()

B_kwargs= {"window_size" : 14,"order" : 7,"lam" : 0} #Parameters for B-spline -- set lam=0.1 for smoothing of noise
fieldlens=FieldLens.from_file_with_axial_samples(
    "examples/combined_field_lens_sampled.csv",
    kind_set=("magnetic","electrostatic"),z=60e-3,name="F0",
    delimiter=delimiter,grounded_check=False, 
    B_kwargs=B_kwargs) 

t1=time()
print(f"spline fitting time:{round(t1-t0,2)}")

thinlens=ThinLens.from_focal_length(10e-3,z=150e-3,name="T0")
aperture=Aperture(radius=2e-3,z=130e-3,name="A0")

plane=Plane(z=180e-3,name="detector")


system=OpticalSystem.from_components(fieldlens,thinlens,aperture,plane)

rays=Rays.from_parallel(radius=2e-3,num_rays=32,voltage=volt)
#rays=Rays.from_disk(semiangle=0.01,num_rays=32,voltage=volt)


Standardplots.multi_order(system=system,rays=rays,highest_order=5,step_size=1e-5,
                          show_fields=True,
                          fname="examples/Combined_lens_multi-order.png")