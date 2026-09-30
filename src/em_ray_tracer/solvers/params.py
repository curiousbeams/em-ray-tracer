"""Solver and integrator configs as dataclasses, in the style of ``OptimizerParams``.

>>> SolverParams.TransferMatrix(order=1)
>>> SolverParams.Paraxial(integrator=IntegratorParams.RK4(dz=1e-5))
>>> SolverParams.parse_dict({"name": "nonparaxial", "expansion_order": 3,
...                          "integrator": {"name": "rk4", "dz": 1e-5}})

``SolverBase.from_params`` turns any of these into a solver.
"""

from dataclasses import dataclass, field
from typing import ClassVar

from em_ray_tracer.optimize.params import _parse


class IntegratorParams:
    """Numerical integrators for the ODE solvers.

    RK4 and DormandPrince45 suit the z-domain ray equations. VelocityVerlet is second-order
    and symplectic only when the force does not depend on velocity. The magnetic force does,
    and so does the damping term (phi'/2phi) r' of the electrostatic z-domain equation, so
    Verlet is not recommended for either. For time-domain Lorentz tracing through field maps
    use Boris, the standard energy-conserving pusher for magnetic fields.
    """

    @dataclass
    class RK4:
        """Classical fixed-step Runge-Kutta, step ``dz`` [m]."""

        dz: float = 1e-5
        _name: ClassVar[str] = "rk4"

    @dataclass
    class DormandPrince45:
        """Adaptive embedded RK4(5) with error control."""

        rtol: float = 1e-10
        atol: float = 1e-14
        dz_max: float | None = None
        _name: ClassVar[str] = "dopri45"

    @dataclass
    class VelocityVerlet:
        """Fixed-step velocity Verlet; see the class docstring for when it is appropriate."""

        dz: float = 1e-5
        _name: ClassVar[str] = "velocity_verlet"

    @dataclass
    class Boris:
        """Time-domain Boris pusher for the full Lorentz force, step ``dt`` [s]."""

        dt: float = 1e-14
        _name: ClassVar[str] = "boris"

    @classmethod
    def parse_dict(cls, d: dict) -> "IntegratorParamsType":
        return _parse(cls, d)


IntegratorParamsType = (
    IntegratorParams.RK4
    | IntegratorParams.DormandPrince45
    | IntegratorParams.VelocityVerlet
    | IntegratorParams.Boris
)


def _integrator(value) -> IntegratorParamsType:
    return IntegratorParams.parse_dict(value) if isinstance(value, dict) else value


class SolverParams:
    """How rays are propagated through the system: the choice of model and its accuracy."""

    @dataclass
    class TransferMatrix:
        """Polynomial transfer maps of the given ``order`` (1 = linear, paraxial)."""

        order: int = 1
        _name: ClassVar[str] = "transfer_matrix"

    @dataclass
    class Paraxial:
        """Integrate the linearized (paraxial) ray equation through field components."""

        integrator: IntegratorParamsType = field(default_factory=IntegratorParams.RK4)
        _name: ClassVar[str] = "paraxial"

        def __post_init__(self):
            self.integrator = _integrator(self.integrator)

    @dataclass
    class NonParaxial:
        """Integrate the coupled 3D ray equations with the off-axis field expanded to
        ``expansion_order`` (3 is where spherical aberration appears)."""

        expansion_order: int = 3
        integrator: IntegratorParamsType = field(default_factory=IntegratorParams.RK4)
        _name: ClassVar[str] = "nonparaxial"

        def __post_init__(self):
            self.integrator = _integrator(self.integrator)

    @dataclass
    class Lorentz:
        """Time-domain Lorentz-force tracing through full 3D fields (e.g. FEM maps)."""

        integrator: IntegratorParamsType = field(default_factory=IntegratorParams.Boris)
        _name: ClassVar[str] = "lorentz"

        def __post_init__(self):
            self.integrator = _integrator(self.integrator)

    @classmethod
    def parse_dict(cls, d: dict) -> "SolverParamsType":
        return _parse(cls, d)


SolverParamsType = (
    SolverParams.TransferMatrix
    | SolverParams.Paraxial
    | SolverParams.NonParaxial
    | SolverParams.Lorentz
)
