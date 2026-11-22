"""Does the aberration fitter reproduce the theoretical C30 of a Glaser bell lens?

Two parts:

1. ``test_reference_*``: self-checks of ``glaser_reference`` (aberration integral against a
   brute-force exact-kinematics trace). They need only numpy and scipy, and they guard the
   reference itself so a failure in part 2 can be blamed on the tracer or fitter.
2. ``test_fitted_c30_*``: your package. Skipped if ``em_ray_tracer`` is not importable.

ADJUST the block marked CONFIG (object position, CONE, import paths, convention) to your setup.
"""


import pytest
import torch

from tests.test_utilities.glaser_reference import GlaserLens, glaser_c30, trace_cs

# ----------------------------------------------------------------------------- CONFIG
FACTOR_A=60
B0 = 1.0                 # T
A = 1e-3                 # m, half width of the bell
ZC = FACTOR_A*A              # m, lens centre (FieldLens.glaser(z=...))
VOLTAGE = 1e4            # V
Z_OBJECT = ZC - FACTOR_A * A   # m, where the cone of rays starts. SET THIS to your Rays.from_cone origin.
HALF_RANGE = FACTOR_A * A      # m, field extent on each side of the centre. Must match the component.
CONE = dict(semiangle=1e-3, n_azimuthal=16, n_shells=4)
                         # Keep the semi-angle small; the fit should include C50.
CONVENTION = None #"image"     # "object": delta = M Cs alpha^3 ; "image": delta = Cs alpha_i^3 (= M^4 Cs)
RTOL = 1e-3              # tolerance on |C30|. Tighten after the halving test (see notes below).
ORDERS = (3, 5, 7)       # expansion orders to test
# ----------------------------------------------------------------------------------------


def _lens() -> GlaserLens:
    return GlaserLens(B0, A, ZC, ZC - HALF_RANGE, ZC + HALF_RANGE)


def _theory() -> dict:
    return glaser_c30(_lens(), VOLTAGE, Z_OBJECT)


# ---------------------------------------------------------------- part 1: the reference
@pytest.mark.parametrize("z_obj_in_a", [-FACTOR_A,])
def test_reference_integral_matches_exact_trace(z_obj_in_a):
    """Aberration integral vs ray tracing through the series field with exact kinematics."""
    lens = _lens()
    z_obj = Z_OBJECT#ZC + z_obj_in_a * A
    ref = glaser_c30(lens, VOLTAGE, z_obj)
    traced = trace_cs(lens, VOLTAGE, z_obj, ref["z_img"], ref["M"])
    #print(trace_cs(lens, VOLTAGE, z_obj, ref["z_img"], ref["M"]))
    assert traced == pytest.approx(ref["Cs_object"], rel=1e-6)


def test_reference_is_positive_and_consistent():
    ref = _theory()
    assert ref["Cs_object"] > 0 and ref["Cs_image"] > 0          # Scherzer
    assert ref["Cs_image"] == pytest.approx(ref["M"] ** 4 * ref["Cs_object"], rel=1e-12)


# ---------------------------------------------------------------- part 2: your package
def _fit(order: int,z_img=None,rotation = "larmor") -> dict:
    #pytest.importorskip("em_ray_tracer")   # only the package tests skip, not the reference checks
    # ADJUST these imports to your package layout.
    from em_ray_tracer.aberrations.basis import AberrationBasis
    from em_ray_tracer.components.field_lenses import FieldLens
    from em_ray_tracer.rays import Rays
    from em_ray_tracer.solvers.base import SolverBase
    from em_ray_tracer.solvers.params import SolverParams
    from em_ray_tracer.system import OpticalSystem
    from em_ray_tracer.tracer import RayTracer

    glaser = FieldLens.glaser(B0=B0, a=A, z=ZC,extent=FACTOR_A)
    if z_img:
        glaser.z_end_rel_cut=z_img-ZC
    system = OpticalSystem.from_components(glaser)
    solver = SolverBase.from_params(params=SolverParams.NonParaxial(expansion_order=order))
    rays = Rays.from_cone(**CONE, voltage=VOLTAGE)
    final=RayTracer.from_models(system=system, rays=rays, solver=solver).fit_aberrations(basis=AberrationBasis.axial(order=7),rotation=rotation)
    return final

@pytest.mark.parametrize("order", ORDERS)
def test_fitted_c30_matches_theory(order):
    ref = _theory()
    z_img=ref["z_img"]
    expected = ref["M"]*(ref["Cs_image"] if CONVENTION == "image" else ref["Cs_object"])
    fitted = float(_fit(order,z_img)["C30"])
    msg = (
        f"fitted C30 = {fitted:.8g}, theory ({CONVENTION}-referred) = {expected:.8g}, "
        f"ratio = {fitted / expected:.6g}\n"
        f"theory: Cs_object = {ref['Cs_object']:.8g} m, Cs_image = {ref['Cs_image']:.8g} m, "
        f"M = {ref['M']:.6g}, z_img = {ref['z_img']:.8g} m\n"
        "A ratio near 1/M, M, M^4 or a power of 10 usually means a convention or unit "
        "mismatch (object vs image side, mm vs m), not a physics error."
    )
    assert abs(fitted) == pytest.approx(abs(expected), rel=RTOL), msg


@pytest.mark.parametrize("order", ORDERS)
def test_fitted_c30_is_positive(order):
    """Scherzer: round, static, space-charge-free lenses have positive Cs. A sign flip here
    points at the fitter's sign convention (for example abTEM-style) or a field-sign error."""
    ref=_theory()
    M=ref["M"]
    assert float(_fit(order)["C30"]/M) > 0


def test_c30_independent_of_expansion_order():
    """The third-order coefficient cannot depend on the field order beyond 3."""
    c = [abs(float(_fit(o)["C30"])) for o in ORDERS]
    assert c[1] == pytest.approx(c[0], rel=RTOL)
    assert c[2] == pytest.approx(c[1], rel=RTOL)

@pytest.mark.parametrize("order",ORDERS)
def test_larmor_theta_matches_direct_fit(order):
    tol=1e-2
    
    theta_direct=_fit(order,rotation="standard").rotation
    theta_larmor=_fit(order,rotation="larmor").rotation
    d = theta_larmor - theta_direct
    # defined modulo pi in the joint fit, so compare on the circle of period pi
    found=d.abs()
    assert found < tol