# em-ray-tracer

Differentiable electron ray tracing in torch. It uses first-order transfer matrices today; higher-order maps, paraxial and non-paraxial ODE solvers for analytic fields (Glaser, Schiske) and FEM field maps are planned. Aberration coefficients are fitted from the traced rays, and every result can be optimized with respect to lens parameters.

## Install

Uses [uv](https://docs.astral.sh/uv/):

```bash
uv sync --all-groups          # package + test and dev tools
uv sync --extra viz           # optional pyvista for 3D plots
uv run pytest
```

## Quick start

```python
from em_ray_tracer import *

system = OpticalSystem.from_components(
    ThinLens.from_focal_length(0.2, z=0.2, name="C1"),
    Aperture(radius=0.03, z=0.6),
    ThinLens.from_focal_length(0.07, z=0.8, name="C2"),
    Plane(z=1.2, name="detector"),
)
tracer = RayTracer.from_models(system, Rays.from_cone(semiangle=10e-3))

trace = tracer.trace(record=True)
z_image, magnification = tracer.gaussian_image()
coefs = tracer.fit_aberrations(AberrationBasis.axial(order=3), plane="detector")

system["C2"].learn("power")
tracer.optimize(ImagePlane(target=1.2), num_iters=10)
```

New to the code? Work through [docs/walkthrough.md](docs/walkthrough.md), a self-guided tour with questions and answers. See `examples/` for runnable scripts, [docs/architecture.md](docs/architecture.md) for the design reference, and [docs/roadmap.md](docs/roadmap.md) for where the project is headed. The original scripts are kept for comparison in `scripts/legacy/`.
