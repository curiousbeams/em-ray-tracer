"""Differentiable electron ray tracing in torch.

Three swappable parts:

- the forward model (``OpticalSystem`` of ``components``)
- the solver (``SolverParams`` -> transfer maps or ODE integration)
- the analysis (``AberrationBasis`` -> ``AberrationCoefficients``)

``RayTracer`` ties them together and optimizes the system's learnable parameters.
"""

from em_ray_tracer.rays import Rays as Rays
from em_ray_tracer.transfer_map import TransferMap as TransferMap
from em_ray_tracer.components import (
    Aperture as Aperture,
    Biprism as Biprism,
    ComponentBase as ComponentBase,
    Deflector as Deflector,
    DoubleDeflector as DoubleDeflector,
    FieldLens as FieldLens,
    Plane as Plane,
    Quadrupole as Quadrupole,
    ThinLens as ThinLens,
)
from em_ray_tracer.system import OpticalSystem as OpticalSystem
from em_ray_tracer.solvers import (
    IntegratorParams as IntegratorParams,
    ODESolver as ODESolver,
    SolverBase as SolverBase,
    SolverParams as SolverParams,
    Trace as Trace,
    TransferMatrixSolver as TransferMatrixSolver,
)
from em_ray_tracer.aberrations import (
    AberrationBasis as AberrationBasis,
    AberrationCoefficients as AberrationCoefficients,
)
from em_ray_tracer.optimize import (
    AberrationTarget as AberrationTarget,
    FocalLength as FocalLength,
    ImagePlane as ImagePlane,
    Magnification as Magnification,
    OptimizerParams as OptimizerParams,
    RayTarget as RayTarget,
    SchedulerParams as SchedulerParams,
    SpotSize as SpotSize,
    Transmission as Transmission,
)
from em_ray_tracer.tracer import RayTracer as RayTracer
