# Roadmap: toward a GPT-class differentiable tracer

[architecture.md](architecture.md) covers the near term: the stubs to fill in next. This page is the long view. It uses the commercial General Particle Tracer (GPT) as the reference point and asks two questions:
- What would it take to trace electrons as completely as GPT does?
- Where can a differentiable torch code do *better*, rather than just reproduce it?

Space charge is out of scope.

The short answer:
- **The biggest gaps are in the physics engine:** exact equations of motion, general 3D fields and full 3D element placement.
- **Aberration analysis uses the same formalism as GPT's.** It needs extending, not redesigning.
- **Optimization and hardware are where we're already ahead:** exact gradients and GPU batching.

---

## 1. What GPT does

- **Equations of motion.** The full relativistic Lorentz force, dp/dt = q(E + v×B) with p = γmv, integrated in *time* for all particles together. There is no paraxial approximation and no transfer matrix anywhere; screens record where particles cross a fixed z.
- **Integrator.** A 5th-order embedded Runge–Kutta with adaptive steps. The error is controlled separately on positions and on γβ, and one "accuracy" setting chooses the number of significant digits.
- **Geometry.** Every element has its own frame, r_world = M r_element + o, where M is any rotation, built by chaining shifts and roll/pitch/yaw. Fields are computed in the element's frame, rotated back, and summed over all elements.
- **Fields.**
  - Analytic elements: Gaussian axial profiles with off-axis expansions; magnetic multipoles to 7th order with Enge-function fringe fields; solenoids, quadrupoles, sextupoles and so on.
  - 1D on-axis maps, which need high-order derivatives for accurate off-axis fields.
  - 2D (r, z) and 3D maps, with trilinear interpolation.
  - A built-in 3D boundary-element solver.
- **Aberration analysis.** A least-squares fit of the ray deviations, Δw = ∂(ψ + ψ̄)/∂ᾱ with ψ = C δᶜ αʲ ᾱᵏ βˡ β̄ᵐ, up to 7th order. On top of the basic fit it has:
  - "rotational" terms (the same columns × i), which are needed to fit beams that rotate correctly;
  - an energy order on every term (Cc is one of these);
  - statistics: the R² of the fit adjusted for the number of terms, and an rms error and p-value for each coefficient, with insignificant terms filtered out;
  - coefficients divided by the magnification and reported under literature names with prefactors (C1, C3, A1, coma, …).
- **Optimization.** Newton–Raphson or Powell's method with finite-difference derivatives, or a genetic algorithm needing hundreds of runs.

## 2. Where we stand

| Area | GPT | em-ray-tracer | Gap |
|---|---|---|---|
| Equations of motion | Full Lorentz, time domain, adaptive RK45 | First-order maps; ODE solver wired, equations stubbed | Large |
| Element placement | Full 3D frames | z position + small-angle shift/tilt | Medium |
| Fields | Superposed analytic elements, maps, BEM | Interfaces and stubs; no overlap allowed | Large |
| Aberration fit | χ basis to 7th order, energy orders, rotational terms, statistics | χ basis to 3rd order | Small–medium: **same formalism** |
| Optimization | Finite differences, genetic search | Exact autodiff gradients, LBFGS/Adam | **We're ahead** |
| Platform | C, CPU | torch: GPU, batched, differentiable | **We're ahead** |

Our χ terms ω^a ω̄^b w^c w̄^d with Δw = 2∂χ/∂ω̄ are GPT's ψ_jklm, so the aberration module grows rather than changes.

---

## 3. Milestones, in priority order

### M1. Analytic round lenses and the z-domain ODE (in progress)

The stubs listed in [architecture.md](architecture.md#what-is-stubbed-in-suggested-order):
- Glaser/Schiske derivatives;
- the paraxial and non-paraxial equations of motion;
- the thick-lens map from the principal rays.

Validate against the Glaser closed forms: trajectory, focal length, Cs, Cc.

### M2. `MultipoleField`: one element for every static lens (best value for time)

A single field element defined by its on-axis multipole strengths a_ν(z), ν = 0…6, each normal or skew. The off-axis field follows from Laplace's equation:

$$
V_\nu = \sum_{\lambda \ge 0} \frac{(-1)^\lambda\, \nu!}{\lambda!\,(\lambda+\nu)!}\left(\frac{w\bar w}{4}\right)^{\lambda}\operatorname{Re}\!\left(a_\nu^{(2\lambda)}(z)\,\bar w^{\nu}\right), \qquad w = x + iy .
$$

What each ν covers:

| ν | Element |
|---|---|
| 0 | round lenses, electrostatic and magnetic (via a scalar potential) |
| 1 | deflectors |
| 2 | quadrupoles and stigmators |
| 3, 4 | hexapole and octupole correctors |

Notes:
- **It replaces the `LaplaceExpansion` stub,** which is its ν = 0 case. Build them as one piece of work.
- **Profiles for a_ν(z):** Glaser, Gaussian, Enge-function fringe fields, or tabulated from an external on-axis multipole export.
- **Exact derivatives:** closed form for the analytic profiles, a smooth fit for tabulated ones.
- **Learnable** strengths, plus `shift`/`tilt` like any other component.
- **Traced by the M1 non-paraxial ODE.** Correctors and parasitic multipoles then become real physics rather than first-order kicks, and the existing `Quadrupole` and `Deflector` become its thin-lens limits.
- **Tests:**
  - ν = 0 reproduces the Glaser closed forms;
  - the fields are divergence- and curl-free to the truncation order;
  - a short quadrupole's fitted C12 approaches the thin-lens value;
  - a hexapole produces a threefold A2.

### M3. Aberration analysis at GPT's level (can run in parallel)

Files: `aberrations/basis.py`, `fit.py`, `coefficients.py`.

1. **Energy order** c on every term: columns multiplied by δᶜ.
2. **Rotational terms:** columns multiplied by i. For round terms this releases the imaginary parts we currently drop.
3. **Statistics from the QR factors:** covariance, rms errors, t- and p-values (scipy's Student-t), adjusted R², condition number, and per-term blur. Filtering insignificant terms is a model-selection step kept outside the gradient; the final fit remains differentiable.
4. **Conventions:**
   - `referred_to_object()`, which divides by the complex magnification;
   - rotation of the frame;
   - a name map to GPT's names and prefactors (C1↔C10, C3↔C30, A1↔C12, …), pinned by tests;
   - a helper to convert between angles and slopes (section 6).
5. **Field decomposition:** the same fit, run on (x, y) → (E_x, E_y), decomposes a field into multipoles.
6. **Orders up to 7.**

### M4. External fields and full 3D placement

We rely on externally computed fields (COMSOL, FEMM, …) rather than our own field solver; see section 4.

- **Loaders:**
  - 1D on-axis profiles and on-axis multipole exports, which feed `MultipoleField` directly;
  - 2D (r, z) maps;
  - 3D maps.
- **Smooth (C²) interpolation:** cubic or B-spline, or tricubic in 3D. Trilinear interpolation is only continuous, not smooth, and that shows up as noise in high-order aberrations.
- **`Pose`:** a full rigid transform (roll/pitch/yaw + offset, chained) for field components, differentiable so misalignments can be fitted.
- **`FieldSuperposition`:** posed fields summed over a support window. This lifts the "no overlap" restriction for ODE solvers.

### M5. Exact tracers

- **`ZDomainLorentz`:**
  - exact equations of motion with z as the independent variable;
  - adaptive DOPRI5(4) with `rtol`/`atol`;
  - exact screens.

  The natural choice for columns, where every electron moves forward.
- **`TimeDomainLorentz`:**
  - state (r, γβ) and adaptive DOPRI5;
  - interpolated output between steps;
  - differentiable screen crossings (a bisection without gradients, then one Newton step, as the image-plane finder does);
  - the Boris pusher as an option.

  Needed only for guns, mirrors, and particles that start at rest.
- **Memory for backpropagation:** checkpointing every N steps.

### M6. Design and calibration

- **Digital twin:** fit excitations and misalignments to measurements, handing the beam to a wave-optics probe simulator to compare with Ronchigrams and probe images.
- **Tolerancing:** the Jacobian d(aberrations)/d(misalignments) in one backward pass.
- **Trade-offs:** Pareto fronts from batched (vmapped) objective-weight sweeps.

### Deferred

Our own field solver (section 4), space charge, and time-dependent (RF) fields.

### Validation ladder

1. Glaser closed forms (M1, M2).
2. An einzel-lens benchmark: fully specified electrode geometry, an externally computed field, and a reference run in GPT where available.
3. Shared test cases traced in both codes.

---

## 4. A boundary-element solver: analysis and decision

**Mathematically it's simple.**
- Flat triangles each carry a constant surface charge σ, and the potential is matched at the triangle centres: V = Mσ.
- The self-terms of M are analytic, nearby pairs use Gaussian quadrature, and the system is solved iteratively.
- Each electrode is solved once at 1 V with the others grounded, so real voltages are a linear rescaling and their gradients are free.
- The core is a few hundred lines of torch.

**The hard parts are operational:**
- mesh quality (graded, near-equilateral triangles);
- scale (ppm precision at 10⁵–10⁶ triangles needs a fast multipole method);
- accurate fields close to electrode surfaces and edges, where beams pass through apertures;
- saturated iron, which is nonlinear.

**Decision: external fields (M4) for now.** If we come back to it, the best-value entry point is an **axisymmetric** solver for round electrostatic lenses. It needs a few hundred ring elements and a dense solve that works with autograd, and it gives gradients with respect to electrode *shape*. A 3D solver would come after that, matrix-free on the GPU, with meshing delegated to gmsh and an existing library for very large problems.

---

## 5. Where a differentiable code can beat GPT

- **Exact gradients and Jacobians** of image plane, magnification, C3, Cc and so on with respect to every excitation, position and misalignment, in one backward pass. A finite-difference workflow needs one run per variable; genetic search needs hundreds.
- **Taylor-mode maps.** Nested forward-mode derivatives (`torch.func.jacfwd` / `jvp`) of a *single* traced reference ray give the exact Taylor coefficients of the map, with no fitting noise and no particle-count statistics. This complements fitting, which still handles apertures, noise and non-polynomial effects.
- **GPU batching** over rays and over parameter sweeps (`vmap`).
- **Open source and Python,** so it plugs into wave-optics and ptychography pipelines.

To be realistic: GPT represents decades of engineering in robustness and breadth. We aim at what matters for electron-microscope columns (static fields, forward-moving beams, aberration-centred design and calibration), where gradients are a qualitative advantage.

---

## 6. Conventions to settle before comparing numbers with GPT

| Quantity | GPT | em-ray-tracer | Action |
|---|---|---|---|
| Independent variable | time t | z | M5 tracers |
| Transverse direction | α = atan(v_x/v_z) per component | slope ω = dx/dz | Conversion helper (below) |
| Energy deviation | (γ − γ₀)/(γ − 1) = (E − E₀)/E | (E − E₀)/E₀ | Same at first order; 2nd-order energy terms differ |
| Coefficient scaling | divided by M (object side), prefactors (C1 ×2, C3 ×4) | image displacement against input slope | `referred_to_object()` and a name map (M3) |
| Beam rotation | frame rotated to follow the beam | complex magnification | Frame rotation (M3) |
| Units | m | m | Same |
| Placement | frame (M, o), chained | z + shift/tilt | `Pose` (M4) |
| Overlapping fields | summed | not allowed | `FieldSuperposition` (M4) |
| Field interpolation | trilinear | not implemented | C² interpolation (M4) |
| Accuracy control | significant digits on x and γβ | fixed-step RK4 | DOPRI5 with rtol/atol (M5) |

**Angles vs slopes.** Taking atan component by component differs from the slope at third order. It isn't rotationally symmetric either:

$$
\alpha \approx \omega - \tfrac14\,|\omega|^2\omega - \tfrac1{12}\,\bar\omega^{3}.
$$

The relation is checked numerically; the error is fifth order. So away from focus, a defocus C1 appears in angle-based fits as an extra C3 of C1/4 and a spurious fourfold term of C1/12. At focus (C1 ≈ 0) the two conventions agree to third order.
