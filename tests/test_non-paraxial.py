"""Non paraxial ray equations through field lenses (ODESolver.nonparaxial).

"""

import torch

from em_ray_tracer import FieldLens, ODESolver, OpticalSystem, Plane, Rays

volt = 3e3
GLASER=dict(B0=1,a=0.03)

    

def test_order1_equals_paraxial():
    """checks wether paraxial and non-paraxial roughly agree with eachother"""
    lens=FieldLens.glaser(**GLASER,z=0)
    
    system=OpticalSystem.from_components(lens,Plane(z=float(lens.z_end),name="out"))
    rays=Rays.from_state(torch.tensor([[1e-3,0,0,0],[0,1e-3,0,0],[0,0,1e-3,0],[0,0,0,1e-3]]),lens.z_start,voltage=volt)
    
    out_paraxial=ODESolver.paraxial().trace(system, rays).planes["out"]
    out_nonparaxial=ODESolver.nonparaxial(expansion_order=3).trace(system, rays).planes["out"]
    torch.testing.assert_close(out_nonparaxial.w,out_paraxial.w,rtol=1e-1,atol=1e-6)
