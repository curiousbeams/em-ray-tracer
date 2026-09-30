"""Entry points: config dataclasses, dict parsing, system round trips, ray constructors."""

import pytest
import torch

from em_ray_tracer import (
    Aperture,
    Deflector,
    DoubleDeflector,
    IntegratorParams,
    ODESolver,
    OpticalSystem,
    OptimizerParams,
    Plane,
    Rays,
    SchedulerParams,
    SolverBase,
    SolverParams,
    ThinLens,
    TransferMatrixSolver,
)


def test_solver_params_parse_and_build():
    params = SolverParams.parse_dict(
        {"name": "paraxial", "integrator": {"name": "rk4", "dz": 1e-6}}
    )
    assert params == SolverParams.Paraxial(integrator=IntegratorParams.RK4(dz=1e-6))
    solver = SolverBase.from_params(params)
    assert isinstance(solver, ODESolver) and solver.integrator.dz == 1e-6

    tm = SolverBase.from_params({"name": "transfer_matrix", "order": 1})
    assert isinstance(tm, TransferMatrixSolver) and tm.order == 1
    with pytest.raises(NotImplementedError):
        SolverBase.from_params(SolverParams.Lorentz())


def test_optimizer_and_scheduler_params_parse():
    assert OptimizerParams.parse_dict({"type": "Adam", "lr": 0.1}) == OptimizerParams.Adam(lr=0.1)
    assert SchedulerParams.parse_dict({}) == SchedulerParams.NoneScheduler()
    with pytest.raises(ValueError, match="Unknown"):
        OptimizerParams.parse_dict({"name": "rmsprop"})


def test_system_config_roundtrip():
    system = OpticalSystem.from_components(
        ThinLens(power=0.5, z=1.0, rotation=0.1, name="C1"),
        Deflector((1e-3, -2e-3), z=1.5, kind="electrostatic"),
        DoubleDeflector((1e-3, 0.0), z=2.0, z_2=2.1, pivot=2.5, beam_shift=(0.0, 2e-5)),
        Aperture(radius=1e-3, z=2.2, shift=(1e-4, 0.0), edge_width=1e-5),
        Plane(z=3.0, name="detector"),
    )
    rebuilt = OpticalSystem.from_config(system.to_config())
    assert rebuilt.names == system.names
    torch.testing.assert_close(
        rebuilt.transfer_map(0.0, 3.0, voltage=200e3).matrix,
        system.transfer_map(0.0, 3.0, voltage=200e3).matrix,
    )


def test_names_are_unique_and_generated():
    system = OpticalSystem.from_components(ThinLens(power=1, z=1), ThinLens(power=1, z=2))
    assert system.names == ["ThinLens0", "ThinLens1"]
    with pytest.raises(ValueError, match="already exists"):
        system.add(ThinLens(power=1, z=3, name="ThinLens0"))


def test_overlapping_extended_components_raise():
    system = OpticalSystem.from_components(
        DoubleDeflector((0, 0), z=1.0, z_2=2.0, pivot=3.0), ThinLens(power=1.0, z=1.5)
    )
    with pytest.raises(ValueError, match="Overlapping"):
        system.segments(0.0)


def test_ray_constructors():
    cone = Rays.from_cone(semiangle=1e-2, n_azimuthal=8, n_shells=3)
    assert cone.state.shape == (24, 5)
    assert cone.slope.abs().max().item() == pytest.approx(torch.tan(torch.tensor(1e-2)).item())
    assert Rays.from_state(torch.zeros(3, 4)).state.shape == (3, 5)
    assert Rays.from_object_points(torch.zeros(2, 2), 1e-2, 8, 3).num_rays == 48
    assert bool(cone.alive.all()) and cone.weight.shape == (24,)


def test_principal_rays_trace_out_the_matrix_columns():
    system = OpticalSystem.from_components(
        ThinLens(power=0.7, z=1.0, rotation=0.3), Deflector((1e-3, 0.0), z=1.2)
    )
    trace = TransferMatrixSolver().trace(system, Rays.principal(), z_end=2.0)
    linear = trace.transfer_map.linear[:4, :4]
    offset = trace.transfer_map.offset[:4]
    torch.testing.assert_close(trace.final.state[:, :4] - offset, linear.T)


def test_parameters_are_named_by_component():
    system = OpticalSystem.from_components(
        ThinLens(power=1, z=5, name="C2"), ThinLens(power=1, z=1, name="C1")
    )
    assert "components.C1.power" in dict(system.named_parameters())
    assert system.names == ["C2", "C1"]  # insertion order
    assert system[0].name == "C1"  # integer index: position along z
    system.remove("C2")
    assert system.names == ["C1"]
    for bad in ["", "a.b", "keys"]:
        with pytest.raises(ValueError, match="component name"):
            system.add(ThinLens(power=1, z=2, name=bad))
