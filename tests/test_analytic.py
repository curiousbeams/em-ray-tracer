
import pytest
import torch

from em_ray_tracer.components.field_lenses import FieldLens
from em_ray_tracer.constants import DTYPE

# --- General setup ---
ORDERS=[0,1,2,3,4]
GLASER = dict(B0=0.1, a=4e-3)
SCHISKE= dict(k=0.5, a=4e-3, voltage= 1e4 )
VOLTAGE = 1e4
z0=-4e-3
z_space=torch.linspace(-10*4e-3,10*4e-3,100,dtype=DTYPE)
def glaser_profile(B0, a):
    def B(z):
        return B0 / (1 + (z / a) ** 2)
    return B


def schiske_profile(k, a, voltage):
    def phi(z):
        return -voltage * k**2 / (1 + (z / a) ** 2)
    return phi

# --- initial calculation ---
glaser=FieldLens.glaser(**GLASER,z=z0)
b=glaser_profile(**GLASER)
b_analyt=FieldLens.from_axial_analytical(axial=b,fieldkind="magnetic",z=z0,half_extent=10*4e-3)

schiske=FieldLens.schiske(**SCHISKE,z=z0)
phi=schiske_profile(**SCHISKE)
phi_analyt=FieldLens.from_axial_analytical(axial=phi,fieldkind="electrostatic",z=z0,half_extent=10*4e-3)


@pytest.mark.parametrize("order",ORDERS)
def test_schiske(order):
    ana=phi_analyt.axial_derivative(order,z_space)
    base=schiske.axial_derivative(order,z_space)
    torch.testing.assert_close(ana,base)


@pytest.mark.parametrize("order",ORDERS)
def test_glaser(order):
    ana=b_analyt.axial_derivative(order,z_space)
    base=glaser.axial_derivative(order,z_space)
    torch.testing.assert_close(ana,base)
    


