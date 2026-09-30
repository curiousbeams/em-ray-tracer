# Architecture

New to the code? Start with the self-guided [walkthrough](walkthrough.md), then use this page as the reference.

This package replaces the scripts in `scripts/legacy/` with one structure that stays the same as the physics grows. It will grow from thin-lens matrices to higher-order maps, then to integrating analytic fields, and eventually to FEM field maps. The code is split into three parts that can each be swapped without touching the others:

| Part | Question it answers | Where |
|---|---|---|
| **Forward model** | What does the electron pass through? | `components/`, `fields/`, `system.py` |
| **Solver** | How is a ray moved through it, and how accurately? | `solvers/` |
| **Analysis** | Which aberration coefficients describe the result? | `aberrations/` |

Two supporting pieces sit alongside them: `rays.py` (the ray bundle) and `optimize/` (optimizers and objectives). `RayTracer` (`tracer.py`) ties everything together:

```python
from em_ray_tracer import *

system = OpticalSystem.from_components(
    ThinLens.from_focal_length(2.0, z=3.0, name="L1"),
    Aperture(radius=5e-2, z=4.0),
    Plane(z=9.0, name="detector"),
)
rays = Rays.from_cone(semiangle=10e-3, n_azimuthal=16, n_shells=3)
tracer = RayTracer.from_models(system, rays, solver=SolverParams.TransferMatrix(order=1))

trace = tracer.trace(record=True)  # Trace: initial/final rays, planes, map, trajectory
coefs = tracer.fit_aberrations(AberrationBasis.axial(order=3), plane="detector")

system["L1"].learn("power")  # opt a parameter in to optimization
tracer.optimize(Magnification(target=-2.0), num_iters=10)
```

Everything is torch and float64, so any number you compute can be a loss.

## Design rules

- **One class per swappable thing.** Examples: `ThinLens` vs `FieldLens`, `TransferMatrixSolver` vs `ODESolver`, `AberrationBasis.axial` vs `.round_lens`.
- **Classmethods for alternative entry points; a plain `__init__` for the canonical one.** `ThinLens(power=...)` is canonical, and `ThinLens.from_focal_length(f)` converts from the familiar form. Likewise `DoubleDeflector(tilt, pivot=..., beam_shift=...)` is canonical because tilt and shift are what you optimize, and `DoubleDeflector.from_excitations(a1, a2, ...)` converts from coil excitations. The stored parameters are the ones you want to control, so that constraints are exact: a pure pivot is `learn("tilt")` with `beam_shift` frozen.
- **Dataclasses for configuration.** `SolverParams`, `IntegratorParams`, `OptimizerParams` and `SchedulerParams` are namespaces of `@dataclass`es with a `_name` and a `parse_dict`. A config can therefore be a dataclass or a plain dict (e.g. from YAML):
  ```python
  SolverParams.parse_dict({"name": "paraxial", "integrator": {"name": "rk4", "dz": 1e-6}})
  ```
- **Parameters are `nn.Parameter`s that start frozen.** `component.learn("power", "z")` opts them in. The system owns the optimizer through `OptimizerMixin`, which also supports the LBFGS closure.

## The ray state

`Rays.state` is `(..., N, 5)`: `[x, x', y, y', delta]`.
- Slopes are dx/dz, not angles.
- `delta = (E - E0)/E0` is the fractional energy deviation from `Rays.voltage`.
- Rays are never removed. Apertures clear `Rays.alive`, so every tensor keeps its shape and the whole bundle moves as one batched operation.
- `Rays.weight` is each ray's share of the current. Spot sizes, `Trace.transmission` (probe current / source current) and aberration fits are all weighted by it. For current to be meaningful the rays must sample the beam uniformly: `Rays.from_disk` does, and `Rays.from_cone` (shells, for visualization and fits) does not.

There is one Cartesian code path; a 2D (meridional) diagram is just `y = y' = 0` (`Rays.from_fan`).

First-order maps are **6×6 homogeneous matrices** acting on `[x, x', y, y', delta, 1]`. This is TEMGYM's 5×5 `[x, x', y, y', 1]` convention (and the one in `first_order_v1_3D_class.py`) plus an energy row and column. The constant 1 is what lets these live in the last column of the matrix:
- deflectors
- misaligned elements
- beam shift and tilt

A 4×4 matrix cannot represent them because they do not depend on the ray.

With the 1 in place:
- **Composition is plain matrix multiplication.** rayTEM keeps `(M, a)` pairs instead and has to compose them as `(M2 M1, M2 a1 + a2)`.
- **Misalignment is conjugation.** A lens shifted by `s` is `T(s) M T(-s)`, which equals the centred lens plus a deflection `P s` (tested).
- **Deflector dispersion is first order.** It is the entry `M[x', delta]`: −γ/(1+γ)·α for a magnetic deflector, −(γ²+1)/(γ(γ+1))·α for an electrostatic one.
- **Round-lens chromatic aberration (x·delta, x'·delta) is second order.** It belongs in the second-order map.

### The first-order maps

Every map acts on the column vector $r = (x,\ x',\ y,\ y',\ \delta,\ 1)^T$, so rows are outputs and columns are inputs in that order. Each map has the block form

$$
M = \begin{pmatrix} A & b \\ 0 & 1 \end{pmatrix},
\qquad
M_2 M_1 = \begin{pmatrix} A_2 A_1 & A_2 b_1 + b_2 \\ 0 & 1 \end{pmatrix},
$$

where $A$ is the $5\times5$ linear part and $b$ the constant kick (`TransferMap.linear`, `TransferMap.offset`). In every map below the $\delta$ row is $(0,0,0,0,1,0)$, because no element here changes the beam energy.

**Drift** over a length $L$ (`drift_map`). It is exact at every order, because a straight line is linear in slopes:

$$
D(L) = \begin{pmatrix}
1 & L & 0 & 0 & 0 & 0 \\
0 & 1 & 0 & 0 & 0 & 0 \\
0 & 0 & 1 & L & 0 & 0 \\
0 & 0 & 0 & 1 & 0 & 0 \\
0 & 0 & 0 & 0 & 1 & 0 \\
0 & 0 & 0 & 0 & 0 & 1
\end{pmatrix}
$$

**Rotation** about the axis by $\theta$ (`rotation_map`), with $c = \cos\theta$, $s = \sin\theta$:

$$
R(\theta) = \begin{pmatrix}
c & 0 & -s & 0 & 0 & 0 \\
0 & c & 0 & -s & 0 & 0 \\
s & 0 & c & 0 & 0 & 0 \\
0 & s & 0 & c & 0 & 0 \\
0 & 0 & 0 & 0 & 1 & 0 \\
0 & 0 & 0 & 0 & 0 & 1
\end{pmatrix}
$$

**Thin lens** of power $P = 1/f$ (`thin_lens_map`):

$$
L(P) = \begin{pmatrix}
1 & 0 & 0 & 0 & 0 & 0 \\
-P & 1 & 0 & 0 & 0 & 0 \\
0 & 0 & 1 & 0 & 0 & 0 \\
0 & 0 & -P & 1 & 0 & 0 \\
0 & 0 & 0 & 0 & 1 & 0 \\
0 & 0 & 0 & 0 & 0 & 1
\end{pmatrix}
$$

A magnetic lens adds its image rotation (the Larmor angle $\theta$; $0$ for an electrostatic lens), applied after the focusing:

$$
R(\theta)\,L(P) = \begin{pmatrix}
c & 0 & -s & 0 & 0 & 0 \\
-Pc & c & Ps & -s & 0 & 0 \\
s & 0 & c & 0 & 0 & 0 \\
-Ps & s & -Pc & c & 0 & 0 \\
0 & 0 & 0 & 0 & 1 & 0 \\
0 & 0 & 0 & 0 & 0 & 1
\end{pmatrix}
$$

**Thin quadrupole** of power $P$, focusing along the azimuth $\psi$ (`quadrupole_map`). It is built as $R(\psi)\,Q\,R(-\psi)$, with $C = \cos 2\psi$ and $S = \sin 2\psi$:

$$
\begin{pmatrix}
1 & 0 & 0 & 0 & 0 & 0 \\
-PC & 1 & -PS & 0 & 0 & 0 \\
0 & 0 & 1 & 0 & 0 & 0 \\
-PS & 0 & PC & 1 & 0 & 0 \\
0 & 0 & 0 & 0 & 1 & 0 \\
0 & 0 & 0 & 0 & 0 & 1
\end{pmatrix}
$$

**Deflector** $(\alpha_x, \alpha_y)$ with dispersion $k = \mathrm{d}\ln\alpha / \mathrm{d}\delta$ (`deflection_map`). For a magnetic deflector $k = -\gamma/(1+\gamma)$; for an electrostatic one $k = -(\gamma^2+1)/(\gamma(\gamma+1))$. Both are negative: a faster electron is deflected less.

$$
K(\alpha_x, \alpha_y) = \begin{pmatrix}
1 & 0 & 0 & 0 & 0 & 0 \\
0 & 1 & 0 & 0 & k\alpha_x & \alpha_x \\
0 & 0 & 1 & 0 & 0 & 0 \\
0 & 0 & 0 & 1 & k\alpha_y & \alpha_y \\
0 & 0 & 0 & 0 & 1 & 0 \\
0 & 0 & 0 & 0 & 0 & 1
\end{pmatrix}
$$

A **double deflector** is $K_2\, D(z_2 - z_1)\, K_1$.

**Translation** (`translation_map`), used for misalignment:

$$
T(d_x, d_y, d_{x'}, d_{y'}) = \begin{pmatrix}
1 & 0 & 0 & 0 & 0 & d_x \\
0 & 1 & 0 & 0 & 0 & d_{x'} \\
0 & 0 & 1 & 0 & 0 & d_y \\
0 & 0 & 0 & 1 & 0 & d_{y'} \\
0 & 0 & 0 & 0 & 1 & 0 \\
0 & 0 & 0 & 0 & 0 & 1
\end{pmatrix}
$$

**Misaligned element.** An element shifted by $s$ and tilted by $t$ acts in its own frame. For a thin element this is (`ComponentBase.transfer_map`)

$$
M_\text{lab} = T(s, t)\; M_\text{local}\; T(-s, -t).
$$

For a thin lens shifted by $s_x$, this gives $x' \to x' - P(x - s_x)$: the centred lens plus a deflection $P s_x$.

Higher orders follow the TRANSPORT convention. They are stored in `TransferMap.higher` as tensors T[i,j,k], U[i,j,k,l], … acting on the 5-vector (not the homogeneous one), so nothing is counted twice. `TransferMap.apply` already evaluates them. Truncated *composition* is not written yet, which is why the solver applies element maps one after another instead of multiplying out a single system map.

## Forward model

`OpticalSystem` is an ordered set of components. **Drifts are not components**: they are the gaps between positions, so moving a lens is differentiable. In the legacy code, a lens snapped to the z-grid and had zero position gradient.

Solvers never touch components directly. They walk `system.segments(z_start, z_end)`, a list of:

- `Gap(z0, z1)`: field-free space, which every solver treats as a drift. This is exact at every order, because a straight line is linear in slope coordinates.
- `Element(component)`: a thin element (`z_start == z_end`) or an extended one (`DoubleDeflector`, `FieldLens`).

A component implements `local_transfer_map(order, voltage)`, its map in its own aligned frame. The base class adds misalignment. Elements that are not polynomial maps override `apply(rays)`:
- **Aperture:** a mask; its map is the identity. With `edge_width > 0` the edge is soft: rays in the band `radius ± edge_width/2` keep a smoothstep fraction of their weight. That makes current and weighted spot sizes differentiable in `radius`, so the aperture can be optimized, e.g. the smallest spot subject to a minimum current (`SpotSize` + `Transmission(target, at_least=True)`; see `examples/optimize_lenses.py`). The gradient comes only from rays inside the band, so it should hold several of them. Use a hard edge for plain simulation.
- **Biprism:** a kick depending on sign(x); it has no map, so a system containing one has `trace.transfer_map = None`.

## Solvers

`SolverBase.trace` owns the loop over segments and records `Plane`s, the composed map, and optionally the trajectory. A solver only decides how to cross an element:

| Element | `TransferMatrixSolver(order)` | `ODESolver(equations, integrator)` |
|---|---|---|
| thin | its map | its map / `apply` |
| field (`FieldComponent`) | its thick-lens map | integrate `equations.bind(component, voltage)` with the integrator |

So a system of thin elements traces identically under both solvers (tested), and a `FieldLens` works under either once its stubs are filled in.

## Aberrations

A basis is a list of terms of the aberration function, `chi = Re[c · norm · omega^a conj(omega)^b w^c conj(w)^d]`:
- `omega = x' + i y'` is the input slope
- `w = x + i y` is the input position

The fit matches ray positions at a plane to `dw = 2 ∂chi/∂conj(omega)` by QR least squares, which is differentiable. Fitting chi rather than free polynomials ties together, for example, the two deviation monomials of B2.

- **Axial terms** use the polar notation C_nm, φ_nm of wave-optics codes, so `coefs.to_polar()` returns `{"C10", "C12", "phi12", ...}` (in metres; check the sign convention of C10 before passing it to a wave-optics code).
- **Off-axial terms** (`round_lens(order=3)`) are the Seidel set, whose imaginary parts are the anisotropic aberrations of magnetic lenses.
- **Every basis includes `shift`**, which absorbs deflections so they do not leak into C10.

**Convention:** the coefficients relate *image-plane positions* to *input slopes and positions*. A drift `dz` past a focus fits as `C10 = dz`. To refer them to image-side slopes (as a probe's Cs is quoted), use `coefs.rescaled(slope_scale=angular_magnification)`.

## Units and numerics

- SI throughout (m, V, T). The em-widgets kit uses SI too; the rest of that kit uses Å.
- float64 by default: third-order ray aberrations sit ~1e-6 below the paraxial terms.
- Use the relativistic potential `phi_hat = phi (1 + e phi / 2mc²)` in every ray equation (`constants.relativistic_potential`). The em-widgets kit is non-relativistic, which is about 20% off at 200 kV.
- Integrators: RK4 or adaptive DOPRI45 for the z-domain equations. Velocity-Verlet is second-order and symplectic only for velocity-independent forces. The magnetic force and the electrostatic damping term (φ'/2φ) r' both depend on velocity, so it is not a good fit for either. For time-domain Lorentz tracing through field maps, use the Boris pusher.
- LBFGS is the default optimizer. Its tolerances are absolute and tightened for SI-scale losses: a 2 µm error is a loss of 4e-12.

## What is stubbed, in suggested order

The stubs raise `NotImplementedError`, and `tests/test_stubs.py` holds the tests they must pass, marked `xfail(strict=True)`. When you implement one and its test starts passing, strict mode fails the run; delete that test's marker. The reference implementations are in the em-widgets kit (`kit/paraxial.js`, `kit/nonparaxial.js`), which has its own tests.

1. **`GlaserField.derivative` / `SchiskeField.derivative`** (`fields/analytic.py`). This is the closed-form nth derivative of 1/(a²+z²); the formula is in the module docstring.
2. **`ParaxialEquations.bind`** (`solvers/ode.py`). This is the lab-frame paraxial equation, written out in the docstring. It is checked against the Glaser closed form.
3. **`FieldLens.local_transfer_map`** (`components/field_lenses.py`). Integrate the principal rays g and h, then add the Larmor rotation. The test is that the TM solver matches the ODE solver.
4. **Second-order maps.** `TransferMap.compose` for `higher`, and second-order maps for field lenses. Chromatic aberration lands here.
5. **`LaplaceExpansion.evaluate` and `NonParaxialEquations.bind`**, the third-order field and the full equations. The payoff is fitting C30 of a Glaser lens and comparing it with its known closed form.
6. **`SampledAxialField` and `FieldMap3D`**, plus the Lorentz/Boris solver, for FEM fields.

## Adding things

- **A component:** subclass `ComponentBase`. Store physical quantities with `as_parameter`, list the constructor arguments in `config_fields` so `OpticalSystem.to_config()` round-trips, and implement `local_transfer_map`. Override `apply` for anything that is not a polynomial map.
- **A solver:** subclass `SolverBase` and implement `propagate(component, rays) -> (rays, map | None)`. Then add a `SolverParams` dataclass and a `case` in `SolverBase.from_params`.
- **An objective:** subclass `Objective` (a dataclass) and implement `loss(evaluation)`. Read everything through the `Evaluation`: `evaluation.rays(plane)`, `.trace`, `.gaussian_image`, `.focal_length`, `.aberrations(basis, plane)`. It computes each on first use and shares it across the objectives in one loss evaluation. Rays are traced only if an objective asks for them, and never for objectives that work from the transfer map.
- **Tests:** `uv run pytest`. `torch.func.functional_call(system, {"components.C1.power": p}, (rays,))` treats the system as a pure function of its parameters, which is how `test_optimize.py` runs `gradcheck`. Parameters are named `components.<component name>.<parameter>`, because components are stored in an `nn.ModuleDict` keyed by name.
