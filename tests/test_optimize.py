"""Gradients through the forward model, and the optimization loop."""

import math

import pytest
import torch
from torch.func import functional_call

from em_ray_tracer import (
    AberrationBasis,
    AberrationCoefficients,
    AberrationTarget,
    Aperture,
    Deflector,
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


def test_gradcheck_lens_power_and_position():
    system = OpticalSystem.from_components(ThinLens(power=0.5, z=1.0), Plane(z=3.0, name="out"))
    rays = Rays.from_state(torch.randn(6, 4, dtype=torch.float64) * 1e-3)

    def f(power, z):
        params = {"components.ThinLens0.power": power, "components.ThinLens0.z": z}
        return functional_call(system, params, (rays,)).planes["out"].state

    inputs = (
        torch.tensor(0.5, dtype=torch.float64, requires_grad=True),
        torch.tensor(1.0, dtype=torch.float64, requires_grad=True),
    )
    assert torch.autograd.gradcheck(f, inputs)


def test_gradcheck_deflection_and_misalignment():
    system = OpticalSystem.from_components(
        ThinLens(power=0.5, z=1.0), Deflector((1e-3, 0.0), z=2.0), Plane(z=3.0, name="out")
    )
    rays = Rays.from_state(torch.randn(6, 4, dtype=torch.float64) * 1e-3)

    def f(shift, deflection):
        params = {
            "components.ThinLens0.shift": shift,
            "components.Deflector0.deflection": deflection,
        }
        return functional_call(system, params, (rays,)).planes["out"].state

    inputs = (
        torch.tensor([1e-4, -2e-4], dtype=torch.float64, requires_grad=True),
        torch.tensor([1e-3, 5e-4], dtype=torch.float64, requires_grad=True),
    )
    assert torch.autograd.gradcheck(f, inputs)


def legacy_system():
    """The legacy v2 optimizer's setup. Its 100-point grid over 12 m snapped the lens at z = 5
    onto grid point 42, i.e. z = 42 * 12/99 = 5.0909, which is what its target -6.4502...
    encodes for f = 4.2."""
    lens = ThinLens.from_focal_length(5.0, z=42 * 12 / 99).learn("power")
    system = OpticalSystem.from_components(lens, Plane(z=12.0, name="screen"))
    rays = Rays.from_state(torch.tensor([[10.0, 0.0, 0.0, 0.0]], dtype=torch.float64))
    return system, rays


def test_legacy_optimizer_regression_lbfgs():
    system, rays = legacy_system()
    tracer = RayTracer.from_models(system, rays, verbose=False)
    tracer.optimize(RayTarget(-6.450220584869385, plane="screen"), num_iters=5)
    assert system["ThinLens0"].focal_length.item() == pytest.approx(4.2, rel=1e-6)
    assert tracer.losses[-1] < 1e-12


def test_legacy_optimizer_regression_adam_from_dicts():
    system, rays = legacy_system()
    tracer = RayTracer.from_models(system, rays, verbose=False)
    tracer.optimize(
        RayTarget(-6.450220584869385, plane="screen"),
        num_iters=400,
        optimizer_params={"name": "adam", "lr": 1e-2},
        scheduler_params={"name": "plateau", "patience": 20, "cooldown": 0},
    )
    assert tracer.losses[-1] < 1e-3 * tracer.losses[0]
    assert system["ThinLens0"].focal_length.item() == pytest.approx(4.2, rel=1e-2)


def test_place_the_image_by_moving_the_lens():
    # f = 1, object at 0: the image is at 5 when the lens is at u with u + u/(u-1) = 5.
    # Start at u = 3 (image at 4.5), on the same side of the u = f singularity as the root.
    lens = ThinLens.from_focal_length(1.0, z=3.0).learn("z")
    system = OpticalSystem.from_components(lens)
    tracer = RayTracer.from_models(system, Rays.from_cone(1e-2), verbose=False)
    tracer.optimize(ImagePlane(target=5.0), num_iters=10)
    z_image, _ = tracer.gaussian_image()
    assert z_image.item() == pytest.approx(5.0, abs=1e-9)
    u = lens.z.item()
    assert u + u / (u - 1) == pytest.approx(5.0, abs=1e-9)


def test_two_lens_magnification_with_per_component_learning_rates():
    # Solution: P1 = 1/6, P2 = 7/6 (a virtual intermediate image 3 m before C1). Start from
    # P1 = 0.4, P2 = 1, which keeps that intermediate image virtual throughout.
    system = OpticalSystem.from_components(
        ThinLens(power=0.4, z=2.0, name="C1").learn("power"),
        ThinLens(power=1.0, z=5.0, name="C2").learn("power"),
    )
    tracer = RayTracer.from_models(system, Rays.from_cone(1e-2), verbose=False)
    tracer.optimize(
        [Magnification(target=-0.25), ImagePlane(target=6.0)],
        num_iters=30,
        optimizer_params={"C1": OptimizerParams.LBFGS(), "C2": OptimizerParams.LBFGS()},
        scheduler_params=SchedulerParams.NoneScheduler(),
    )
    z_image, magnification = tracer.gaussian_image()
    assert z_image.item() == pytest.approx(6.0, abs=1e-6)
    assert magnification.item() == pytest.approx(-0.25, abs=1e-6)
    assert system["C1"].power.item() == pytest.approx(1 / 6, abs=1e-6)
    assert system["C2"].power.item() == pytest.approx(7 / 6, abs=1e-6)


def test_optimize_without_learnable_parameters_raises():
    system = OpticalSystem.from_components(ThinLens(power=1.0, z=1.0))
    tracer = RayTracer.from_models(system, Rays.from_cone(1e-2), verbose=False)
    with pytest.raises(ValueError, match="learn"):
        tracer.optimize(ImagePlane(target=5.0), num_iters=1)


def test_smallest_aperture_that_passes_the_minimum_current():
    """Spot size at a defocused plane grows with the aperture, so the optimum is the smallest
    aperture that still passes 25% of the current: R* = L tan(alpha) sqrt(0.25)."""
    L, alpha = 0.1, 1e-2
    beam = L * math.tan(alpha)
    aperture = Aperture(radius=0.9 * beam, z=L, edge_width=0.02 * beam).learn("radius")
    system = OpticalSystem.from_components(
        aperture, ThinLens.from_focal_length(0.1, z=0.2), Plane(z=0.401, name="specimen")
    )
    tracer = RayTracer.from_models(system, Rays.from_disk(alpha, num_rays=2000), verbose=False)
    tracer.optimize(
        [SpotSize(plane="specimen"), Transmission(0.25, plane="specimen")], num_iters=10
    )
    assert aperture.radius.item() == pytest.approx(0.5 * beam, rel=1e-3)
    assert tracer.trace().transmission("specimen").item() == pytest.approx(0.25, abs=1e-4)


def test_loss_traces_only_when_needed_and_shares_work(monkeypatch):
    system = OpticalSystem.from_components(
        ThinLens.from_focal_length(2.0, z=3.0), Plane(z=9.0, name="detector")
    )
    tracer = RayTracer.from_models(system, Rays.from_cone(1e-2), verbose=False)
    calls = {"trace": 0, "image": 0, "fit": 0}

    def counted(name, fn):
        def wrapped(*args, **kwargs):
            calls[name] += 1
            return fn(*args, **kwargs)

        return wrapped

    monkeypatch.setattr(tracer, "trace", counted("trace", tracer.trace))
    monkeypatch.setattr(tracer, "gaussian_image", counted("image", tracer.gaussian_image))
    fit = AberrationCoefficients.from_trace
    monkeypatch.setattr(AberrationCoefficients, "from_trace", counted("fit", fit))

    # map-only objectives: no rays traced, one image computation shared by both
    tracer.loss([ImagePlane(target=9.0), Magnification(target=-2.0)])
    assert calls == {"trace": 0, "image": 1, "fit": 0}

    # ray objectives share one trace; targets with the same basis and plane share one fit
    tracer.loss(
        [
            SpotSize(plane="detector"),
            RayTarget(0.0, plane="detector"),
            AberrationTarget("C10"),
            AberrationTarget("C30"),
            AberrationTarget("C10", basis=AberrationBasis.axial(order=1)),
        ]
    )
    assert calls == {"trace": 1, "image": 1, "fit": 2}
