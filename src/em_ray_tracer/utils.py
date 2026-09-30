"""Small helpers shared across modules."""

from typing import Any

import torch
from torch import nn

from em_ray_tracer.constants import DTYPE


def as_parameter(value: Any) -> nn.Parameter:
    """A frozen float64 parameter. Opt in to optimization with ``component.learn(name)``."""
    tensor = torch.as_tensor(value, dtype=DTYPE).detach().clone()
    return nn.Parameter(tensor, requires_grad=False)
