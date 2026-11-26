""" """

import numpy
import torch

from em_ray_tracer.components.field_lenses import FieldLens
from em_ray_tracer.constants import DTYPE
from em_ray_tracer.fields.sampled import SampledAxialField

# --- General setup ---

GLASER = dict(B0=0.1, a=4e-3)
SCHISKE = dict(k=0.5, a=4e-3, voltage=1e4)
VOLTAGE = 1e4
delimiter = ","
z0 = 0

# --- create sample field ---
lens_B = FieldLens.glaser(**GLASER, z=0)
lens_E = FieldLens.schiske(**SCHISKE, z=0)
z_space = torch.linspace(lens_B.z_start, lens_B.z_end, 10**4)

# the fields' own profiles: the lenses built from the samples taper them, as lens_B and lens_E do
B_sampled = lens_B.field.derivative(0, z_space - lens_B.z)["magnetic"]
E_sampled = lens_E.field.derivative(0, z_space - lens_E.z)["electrostatic"]

# --- end ---


def test_sampled_axial_from_file():
    # create sample file:
    numpy.savetxt(
        "tests/test_utilities/testdata_glaserschiske.csv",
        torch.stack([z_space, B_sampled, E_sampled], dim=-1),
        delimiter=delimiter,
    )

    # extract sample field:
    axial = SampledAxialField.from_file(
        "tests/test_utilities/testdata_glaserschiske.csv",
        ("magnetic", "electrostatic"),
        delimiter=delimiter,
        grounded_check=False,
    )
    # check match
    torch.testing.assert_close([lens_E.z_start, lens_E.z_end], [lens_B.z_start, lens_B.z_end])
    torch.testing.assert_close(axial.values[0], B_sampled)
    torch.testing.assert_close(axial.z_samples, z_space.to(dtype=DTYPE))


def test_sampled_derivatives():
    n: int = 1
    z_test = torch.linspace(lens_B.z_start, lens_B.z_end, 300)
    axial = FieldLens.from_axial_samples(
        z_samples=z_space - z0,
        values=[E_sampled, B_sampled],
        kind_set=("electrostatic", "magnetic"),
        z=z0,
        grounded_check=False,
    )
    # the spline against the analytic profiles, before the lenses' tapers
    B_dsample = axial.field.derivative(n, z_test - z0)["magnetic"]
    E_dsample = axial.field.derivative(n, z_test - z0)["electrostatic"]

    torch.testing.assert_close(
        B_dsample, lens_B.field.derivative(n, z_test - lens_B.z)["magnetic"]
    )
    torch.testing.assert_close(
        E_dsample, lens_E.field.derivative(n, z_test - lens_E.z)["electrostatic"]
    )
