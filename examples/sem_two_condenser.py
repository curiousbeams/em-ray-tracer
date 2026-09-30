"""SEM column with two condensers and an objective: the legacy v2 demo on the new API.

Run: uv run python examples/sem_two_condenser.py [output.png]
"""

import math
import sys

import matplotlib.pyplot as plt

from em_ray_tracer import Aperture, OpticalSystem, Plane, Rays, RayTracer, ThinLens
from em_ray_tracer.visualization import plot_rays

system = OpticalSystem.from_components(
    ThinLens.from_focal_length(0.2, z=0.2, name="C1"),
    Aperture(radius=0.03, z=0.6, name="A1"),
    ThinLens.from_focal_length(0.07, z=0.8, name="C2"),
    Aperture(radius=0.05, z=1.3, name="A2"),
    ThinLens.from_focal_length(0.14, z=1.5, name="OL"),
    Plane(z=2.0, name="detector"),
)
rays = Rays.from_fan(semiangle=math.radians(15), num_rays=100)
tracer = RayTracer.from_models(system, rays)

trace = tracer.trace(record=True)
z_image, magnification = tracer.gaussian_image()
print(system)
print(f"image of the source at z = {z_image.item():.4f} m, M = {magnification.item():.4g}")
print(f"{int(trace.final.alive.sum())} of {rays.num_rays} rays reach the detector")

ax = plot_rays(trace, system)
ax.set_ylim(-0.25, 0.25)
ax.set_title("SEM: two condensers + objective")
if len(sys.argv) > 1:
    plt.savefig(sys.argv[1], dpi=150, bbox_inches="tight")
else:
    plt.show()
