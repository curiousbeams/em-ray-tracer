"""Optimization targets, as dataclasses. Each returns a scalar loss.

Objectives read what they need from an :class:`Evaluation`, which computes the ray trace, the
paraxial image and aberration fits on first use and shares them. One loss evaluation (one
optimizer closure call) therefore traces the rays at most once, and not at all when every
objective works from the transfer map (``ImagePlane``, ``Magnification``, ``FocalLength``).

Losses are plain squared errors, multiplied by ``weight``. They are not normalized, so set
the weights when combining targets of different units (e.g. metres and radians).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

from em_ray_tracer.aberrations.basis import AberrationBasis
from em_ray_tracer.aberrations.coefficients import AberrationCoefficients

if TYPE_CHECKING:
    from em_ray_tracer.rays import Rays
    from em_ray_tracer.solvers.base import Trace
    from em_ray_tracer.tracer import RayTracer


class Evaluation:
    """The quantities objectives read, each computed on first use and then reused.

    A new ``Evaluation`` is made for every loss evaluation, because the parameters change
    between them. Nothing is cached across evaluations.
    """

    def __init__(self, tracer: "RayTracer", trace: "Trace | None" = None):
        self.tracer = tracer
        self._trace = trace
        self._image: tuple[torch.Tensor, torch.Tensor] | None = None
        self._focal_length: torch.Tensor | None = None
        self._fits: dict[tuple, AberrationCoefficients] = {}

    @property
    def trace(self) -> "Trace":
        if self._trace is None:
            self._trace = self.tracer.trace()
        return self._trace

    def rays(self, plane: str | None = None) -> "Rays":
        """The traced bundle at ``plane`` (default: the end of the trace)."""
        return self.trace.final if plane is None else self.trace.planes[plane]

    @property
    def gaussian_image(self) -> tuple[torch.Tensor, torch.Tensor]:
        """``(z_image, magnification)``, from the transfer map; no rays are traced."""
        if self._image is None:
            self._image = self.tracer.gaussian_image()
        return self._image

    @property
    def focal_length(self) -> torch.Tensor:
        if self._focal_length is None:
            self._focal_length = self.tracer.system.focal_length(voltage=self.tracer.rays.voltage)
        return self._focal_length

    def aberrations(
        self, basis: AberrationBasis, plane: str | None = None
    ) -> AberrationCoefficients:
        """The fit of ``basis`` at ``plane``, shared by all targets with the same basis and
        plane."""
        key = (basis.terms, plane)
        if key not in self._fits:
            self._fits[key] = AberrationCoefficients.from_trace(self.trace, basis, plane)
        return self._fits[key]


@dataclass(kw_only=True)
class Objective(ABC):
    weight: float = 1.0

    def __call__(self, evaluation: Evaluation) -> torch.Tensor:
        return self.weight * self.loss(evaluation)

    @abstractmethod
    def loss(self, evaluation: Evaluation) -> torch.Tensor: ...


@dataclass
class RayTarget(Objective):
    """Mean squared distance of the alive rays at ``plane`` from ``target`` (x, or (x, y))."""

    target: float | tuple[float, float]
    plane: str | None = None

    def loss(self, evaluation):
        rays = evaluation.rays(self.plane)
        target = torch.as_tensor(self.target, dtype=rays.state.dtype)
        coords = rays.state[..., :1] if target.ndim == 0 else rays.state[..., [0, 2]]
        err = ((coords - target) ** 2).sum(-1)
        weight = rays.weight * rays.alive
        return (err * weight).sum() / weight.sum()


@dataclass
class SpotSize(Objective):
    """Squared difference between the RMS spot radius at ``plane`` and ``target``."""

    target: float = 0.0
    plane: str | None = None

    def loss(self, evaluation):
        rays = evaluation.rays(self.plane)
        weight = rays.weight * rays.alive
        centre_x = (rays.x * weight).sum() / weight.sum()
        centre_y = (rays.y * weight).sum() / weight.sum()
        r2 = (rays.x - centre_x) ** 2 + (rays.y - centre_y) ** 2
        rms = torch.sqrt((r2 * weight).sum() / weight.sum())
        return (rms - self.target) ** 2


@dataclass
class Transmission(Objective):
    """Fraction of the beam reaching ``plane`` (probe current / source current).

    With ``at_least=True`` (the default) it penalizes only a shortfall below ``target``, which
    expresses a minimum probe current. The penalty is soft: a stronger ``weight`` holds the
    minimum more tightly against competing objectives. Use rays that sample the beam uniformly
    (``Rays.from_disk``) and a soft-edged ``Aperture`` for smooth gradients.
    """

    target: float
    plane: str | None = None
    at_least: bool = True

    def loss(self, evaluation):
        shortfall = self.target - evaluation.trace.transmission(self.plane)
        if self.at_least:
            shortfall = torch.relu(shortfall)
        return shortfall**2


@dataclass
class ImagePlane(Objective):
    """Place the paraxial image of the object plane (the rays' start) at ``target`` z."""

    target: float

    def loss(self, evaluation):
        z_image, _ = evaluation.gaussian_image
        return (z_image - self.target) ** 2


@dataclass
class Magnification(Objective):
    """Paraxial magnification at the Gaussian image plane.

    The magnification is complex (its argument is the image rotation). A real ``target``
    therefore also asks for zero rotation. Set ``ignore_rotation`` to match |M| only, which is
    usually what you want for magnetic lenses.
    """

    target: complex
    ignore_rotation: bool = False

    def loss(self, evaluation):
        _, magnification = evaluation.gaussian_image
        if self.ignore_rotation:
            return (magnification.abs() - abs(self.target)) ** 2
        return (magnification - self.target).abs() ** 2


@dataclass
class FocalLength(Objective):
    """System focal length (its modulus, so the image rotation of magnetic lenses is free)."""

    target: float

    def loss(self, evaluation):
        return (evaluation.focal_length.abs() - self.target) ** 2


@dataclass
class AberrationTarget(Objective):
    """|c - target|^2 for one fitted aberration coefficient (e.g. ``"C30"``)."""

    name: str
    target: complex = 0.0
    basis: AberrationBasis = field(default_factory=AberrationBasis.axial)
    plane: str | None = None

    def loss(self, evaluation):
        coefficients = evaluation.aberrations(self.basis, self.plane)
        return (coefficients[self.name] - self.target).abs() ** 2
