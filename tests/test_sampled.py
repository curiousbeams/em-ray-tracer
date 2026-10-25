""" """
import numpy
import torch
import pytest

from em_ray_tracer.fields.sampled import SampledAxialField
from em_ray_tracer.components.field_lenses import FieldLens
from em_ray_tracer.constants import DTYPE

# --- General setup ---

GLASER = dict(B0=0.01, a=0.0075)
SCHISKE= dict(k=0.5, a=0.0075)
VOLTAGE = 1000.0
z0=1
delimiter=","

# --- create sample field ---
lens_B=FieldLens.glaser(**GLASER,z=z0)
lens_E=FieldLens.schiske(**SCHISKE,z=z0)
z_space=torch.linspace(lens_B.z_start,lens_B.z_end,10**4)#still needs fix

B_sampled=lens_B.axial_derivative(0,z_space)
E_sampled=lens_E.axial_derivative(0,z_space)

# --- end ---

def test_sampled_axial_from_file():
    #create sample file:
    numpy.savetxt("tests/testdata_glaserschiske.csv",torch.stack([z_space,B_sampled,E_sampled],dim=-1),delimiter=delimiter)

    #extract sample field:
    axial=SampledAxialField.from_file("tests/testdata_glaserschiske.csv",("magnetic","electrostatic"),delimiter=delimiter)
    #check match
    torch.testing.assert_close([lens_E.z_start,lens_E.z_end], [lens_B.z_start,lens_B.z_end])
    torch.testing.assert_close(axial.values[0], B_sampled)
    torch.testing.assert_close(axial.z_samples, z_space.to(dtype=DTYPE))

def test_sampled_derivatives():
    n : int =1
    z_test=torch.linspace(lens_B.z_start,lens_B.z_end,300)
    axial=FieldLens.from_axial_samples(z_samples=z_space-z0,values=[E_sampled,B_sampled],kind_set=("electrostatic","magnetic"),z=z0) 
    B_dsample = axial.axial_derivative(n,z_test,kind_requested="magnetic")
    E_dsample = axial.axial_derivative(n,z_test,kind_requested="electrostatic")

    torch.testing.assert_close(B_dsample,lens_B.axial_derivative(n,z_test))
    torch.testing.assert_close(E_dsample,lens_E.axial_derivative(n,z_test))

    
    

