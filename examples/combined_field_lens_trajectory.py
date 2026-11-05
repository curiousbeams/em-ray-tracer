
from concurrent.futures import ProcessPoolExecutor
from time import time

import matplotlib.pyplot as plt
import numpy as np
import torch

from em_ray_tracer.components import FieldLens
from em_ray_tracer.rays import Rays
from em_ray_tracer.solvers.ode import ODESolver
from em_ray_tracer.solvers.params import IntegratorParams
from em_ray_tracer.system import OpticalSystem
from em_ray_tracer.tracer import RayTracer
from em_ray_tracer.visualization import plot_rays

GLASER = dict(B0=0.1, a=4e-3)
SCHISKE= dict(k=0.8, a=4e-3, voltage= 1e4 )
delimiter=","

# --- create sample field file ---
lens_B=FieldLens.glaser(**GLASER,z=30e-3)
lens_E=FieldLens.schiske(**SCHISKE,z=-20e-3)
z_sample_space=torch.linspace(-60e-3,60e-3,10**4)

B_sampled=lens_B.axial_derivative(0,z_sample_space)
B_sampled=B_sampled# + max(B_sampled)*torch.rand_like(B_sampled)*1e-3 #to see the effect of noise
Phi_sampled=lens_E.axial_derivative(0,z_sample_space)
Phi_sampled=Phi_sampled #+ max(Phi_sampled)*torch.rand_like(Phi_sampled)*1e-3

np.savetxt("examples/combined_field_lens_sampled.csv",torch.stack([z_sample_space,B_sampled,Phi_sampled],dim=-1),delimiter=delimiter)



# --- read from file and setup system ---
t0=time()
lens=FieldLens.from_file_with_axial_samples(
    "examples/combined_field_lens_sampled.csv",
    kind_set=("magnetic","electrostatic"),z=60e-3,
    delimiter=delimiter,grounded_check=False,
    extrapolate="clamp",window_size=10,order=7,lam=0) #Parameters for B-spline -- set lam=0.1 for smoothing of noise

system=OpticalSystem.from_components(lens)
r=1e-3 #mm
rays=Rays.from_parallel(radius=r,num_rays=8,voltage=1e4)

# --- read field values ---
z_space=torch.linspace(0,140e-3,10**3)
phi_space=lens.axial_derivative(0,z_space,kind_requested="electrostatic")
b_space=lens.axial_derivative(0,z_space,kind_requested="magnetic")

# --- normalize field values
phi_space=phi_space/max(abs(phi_space))
b_space=b_space/max(abs(b_space))
Phi_sampled=Phi_sampled/max(abs(Phi_sampled))
B_sampled=B_sampled/max(abs(B_sampled))


# --- setup plots ---
N=5
step_size=1e-3
fig, axis=plt.subplots(1,N+1,sharey=True,figsize=(19.3,15),constrained_layout=True)

# --- Phi/B field values ---
#axis[0].plot(Phi_sampled,z_sample_space+60e-3,"k.",label="Phi_samples")
#axis[0].plot(B_sampled,z_sample_space+60e-3,"k.",label="B_samples")
axis[0].plot(phi_space,z_space,"r--",label="Phi_splined")
axis[0].plot(b_space,z_space,"b--",label="B_z_splined")
axis[0].set_xlabel("normalized fields")
axis[0].set_ylabel("z[m]")
axis[0].invert_yaxis()
axis[0].set_title("on axis fields")
axis[0].legend()

# --- solver ---
def run_case(i):
    torch.set_num_threads(1)
    if i==0: 
        solver=ODESolver.paraxial(integrator=IntegratorParams.RK4(dz=step_size))
    else: 
        solver=ODESolver.nonparaxial(expansion_order=2*i-1,integrator=IntegratorParams.RK4(dz=step_size))
    
    trace=RayTracer(system=system,rays=rays,solver=solver,z_end=140e-3).trace(record=True)
    return trace

# --- seperate tasks across multiple cores ---
with ProcessPoolExecutor(max_workers=8) as ex:
    traces=list(ex.map(run_case,range(N)))

# --- plot ---

for i,trace in enumerate(traces):
    if i == 0 :
        axis[i+1].set_title("paraxial")
    else : 
        axis[i+1].set_title(f"non paraxial order: {2*i-1}")
    
    axis[i+1]=plot_rays(trace,color="b",ax=axis[i+1],off_axis_scale="mm",on_axis_scale=None,invert_orientation=True)
    axis[i+1].set_xlim(-3*r*1e3,3*r*1e3)
    axis[i+1].plot()

t1=time()
print("run time:",round(t1-t0,1),"s")

plt.savefig(fname="examples/Combined_lens_multi-order.png",dpi=300,bbox_inches="tight")


