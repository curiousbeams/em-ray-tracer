"""Optimizing lens parameters with RayTracer.optimize.

1. The legacy v2 optimizer: find the focal length that puts a ray at a target height.
2. A design problem: choose two lens powers for a given magnification and image plane.
3. Aperture size: the smallest spot that still delivers a minimum probe current.

Run: uv run python examples/optimize_lenses.py
"""

import math

import torch

from em_ray_tracer import (
    Aperture,
    ImagePlane,
    Magnification,
    OpticalSystem,
    OptimizerParams,
    Plane,
    Rays,
    RayTarget,
    RayTracer,
    SchedulerParams,
    SpotSize,
    ThinLens,
    Transmission,
)

# 1. Legacy v2: a ray at x = 10 through one lens should land at x = -6.4502 on a screen at
#    z = 12. The legacy grid snapped the lens from z = 5 to grid point 42 of 100 (z = 5.0909);
#    that is where the target number comes from, so the lens goes there too. The target was
#    computed in float32, so f = 4.2 is only recovered to ~1e-7.
lens = ThinLens.from_focal_length(5.0, z=42 * 12 / 99, name="L").learn("power")
system = OpticalSystem.from_components(lens, Plane(z=12.0, name="screen"))
rays = Rays.from_state(torch.tensor([[10.0, 0.0, 0.0, 0.0]]))

tracer = RayTracer.from_models(system, rays, verbose=False)
tracer.optimize(
    RayTarget(-6.450220584869385, plane="screen"),
    num_iters=200,
    optimizer_params=OptimizerParams.Adam(lr=1e-2),
    scheduler_params=SchedulerParams.Plateau(patience=20, cooldown=0),
)
print(f"Adam:  f = {lens.focal_length.item():.6f} (legacy target 4.2)\n")

# The optimizer state lives on the system, so a new optimizer is requested
# explicitly rather than continuing the Adam run above.
with torch.no_grad():
    lens.power.fill_(1 / 5.0)
RayTracer.from_models(system, rays, verbose=False).optimize(
    RayTarget(-6.450220584869385, "screen"), num_iters=5, optimizer_params=OptimizerParams.LBFGS()
)
print(f"LBFGS: f = {lens.focal_length.item():.9f}\n")

# 2. Two lenses at fixed positions; find powers giving M = -0.25 with the image at z = 6.
system = OpticalSystem.from_components(
    ThinLens(power=0.4, z=2.0, name="C1").learn("power"),
    ThinLens(power=1.0, z=5.0, name="C2").learn("power"),
)
tracer = RayTracer.from_models(system, Rays.from_cone(semiangle=1e-2), verbose=False)
tracer.optimize([Magnification(target=-0.25), ImagePlane(target=6.0)], num_iters=5)
z_image, magnification = tracer.gaussian_image()
print(
    f"C1 f = {system['C1'].focal_length.item():.4f} m, C2 f = {system['C2'].focal_length.item():.4f} m"
    f" -> image at {z_image.item():.6f} m, M = {magnification.real.item():.6f}"
)

# 3. A probe-forming aperture: a larger aperture passes more current but gives a larger spot.
#    Only first-order optics exist so far, so the blur here is defocus (the specimen sits
#    1 mm past focus). Once third-order maps exist, the same two objectives trade spherical
#    aberration blur (~ Cs alpha^3) against current (~ alpha^2).
#    Rays.from_disk samples the cone uniformly, so summed ray weight is current. The soft
#    edge makes that current a smooth function of the radius.
L, alpha = 0.1, 10e-3
beam_radius = L * math.tan(alpha)  # at the aperture
aperture = Aperture(radius=0.9 * beam_radius, z=L, edge_width=0.02 * beam_radius, name="OA")
aperture.learn("radius")
system = OpticalSystem.from_components(
    aperture,
    ThinLens.from_focal_length(0.1, z=0.2, name="OL"),
    Plane(z=0.401, name="specimen"),
)
tracer = RayTracer.from_models(system, Rays.from_disk(alpha, num_rays=2000), verbose=False)
tracer.optimize(
    [SpotSize(plane="specimen"), Transmission(0.25, plane="specimen")],
    num_iters=10,
    optimizer_params=OptimizerParams.LBFGS(),
)
current = tracer.trace().transmission("specimen")
print(
    f"\naperture radius {aperture.radius.item() * 1e3:.4f} mm "
    f"(analytic {0.5 * beam_radius * 1e3:.4f} mm), probe current {current.item():.4f} of source"
)
