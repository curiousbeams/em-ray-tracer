"""Analytic axial fields: derivatives, parameter names, and persistence."""

import copy
import io

import pytest
import torch

from em_ray_tracer import FieldLens
from em_ray_tracer.fields import GlaserField, SchiskeField

B0, K, A = 0.01, 0.6, 0.0075
Z = torch.linspace(-0.05, 0.05, 41, dtype=torch.float64)


def autograd_derivative(profile, n, z):
    """The n-th derivative of a scalar profile by nested autograd, at each point of z."""
    f = profile
    for _ in range(n):
        f = torch.func.grad(f)
    return torch.func.vmap(f)(z)


@pytest.mark.parametrize("n", range(8))
@pytest.mark.parametrize(
    "field, profile",
    [
        (GlaserField(B0=B0, a=A), lambda z: B0 / (1 + (z / A) ** 2)),
        (SchiskeField(k=K, a=A), lambda z: 1 - K**2 / (1 + (z / A) ** 2)),
    ],
    ids=["glaser", "schiske"],
)
def test_derivatives_match_autograd(field, profile, n):
    # up to the 7th derivative, which the 3rd-order Laplace expansion needs
    expected = autograd_derivative(profile, n, Z)
    scale = expected.abs().max().item()
    torch.testing.assert_close(
        field.derivative(n, Z).detach(), expected, rtol=1e-9, atol=1e-12 * scale
    )


def test_parameters_keep_their_documented_names():
    assert dict(GlaserField(B0=B0, a=A).named_parameters()).keys() == {"B0", "a"}
    assert dict(SchiskeField(k=K, a=A).named_parameters()).keys() == {"k", "a"}
    lens = FieldLens.glaser(B0=B0, a=A, z=0.0).learn("field.B0")
    assert lens.field.B0.requires_grad


@pytest.mark.parametrize(
    "lens",
    [FieldLens.glaser(B0=B0, a=A, z=0.0), FieldLens.schiske(k=K, a=A, z=0.0)],
    ids=["glaser", "schiske"],
)
def test_field_lenses_copy_and_save_after_use(lens):
    before = lens.axial_derivative(3, Z).detach()
    buffer = io.BytesIO()
    torch.save(lens, buffer)
    buffer.seek(0)
    for clone in (copy.deepcopy(lens), torch.load(buffer, weights_only=False)):
        torch.testing.assert_close(clone.axial_derivative(3, Z).detach(), before)


def test_schiske_rejects_a_reflecting_strength():
    with pytest.raises(ValueError, match="reflected"):
        SchiskeField(k=1.0, a=A)
