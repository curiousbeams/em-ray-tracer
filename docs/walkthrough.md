# Code walkthrough: a self-guided tour

This guide walks through the package in ten steps, in the order the files depend on each other: each step only uses ideas from the steps before it. For each step:

1. Open the files listed under **Read** and skim them.
2. Check each point under **What to notice** against the code.
3. Answer the **Questions** yourself, in writing, before opening **Answers**.

If an answer surprises you, rerun the relevant test or open a Python shell (`uv run python`) and try it. [architecture.md](architecture.md) covers the same ground as a reference once you're done.

| Step | Read | Idea it introduces |
|---|---|---|
| 1 | `rays.py` | The ray state, and classmethod entry points |
| 2 | `transfer_map.py` | 6×6 homogeneous maps: deflections live in the last column |
| 3 | `components/base.py`, `utils.py` | Frozen parameters and `learn()`; misalignment added around each element |
| 4 | `components/thin.py` | The concrete elements, including two that aren't matrices |
| 5 | `system.py` | The forward model: drifts as gaps between components; closed-form image plane |
| 6 | `solvers/params.py` → `base.py` → `transfer_matrix.py` | Solver configs as dataclasses; one shared tracing loop |
| 7 | `field_lenses.py`, `fields/`, `solvers/ode.py`, `tests/test_stubs.py` | The stubs: where the next work goes |
| 8 | `aberrations/`: basis → fit → coefficients | Fitting the aberration function χ in C_nm notation |
| 9 | `optimize/`: params → mixin → objectives | Optimizers, schedulers and objectives; the shared per-evaluation cache |
| 10 | `tracer.py`, then `examples/` | The top-level class and the optimization loop |

---

## Step 1: the ray bundle

**Read:** [src/em_ray_tracer/rays.py](../src/em_ray_tracer/rays.py)

This is the one data structure every other module passes around: a batch of rays at one axial position z.

**What to notice**

1. **The state layout.** `Rays.state` has shape `(..., N, 5)` with columns `[x, x′, y, y′, δ]` (the constants `X, XP, Y, YP, DELTA` at the top of the file).
   - Slopes are dx/dz, not angles.
   - δ = (E − E₀)/E₀ is the fractional energy deviation from `Rays.voltage`.
   - The `...` leading dimensions are optional batch dimensions, e.g. one bundle per value in a sweep of lens strengths.
   - The transfer matrices are 6×6, but the constant 1 they act on is **not stored** here; it's added inside the matrix step (step 2).
2. **`Rays` is a `@dataclass` built with its plain constructor.** `__post_init__` tidies the inputs: it forces float64, turns `z` into a tensor, and fills in default `alive` and `weight`.
3. **Rays are never removed.** Two parallel arrays record what apertures do:
   - `alive` (bool): the ray was blocked.
   - `weight` (float): its share of the current, which soft apertures reduce smoothly.

   Keeping every ray means every tensor keeps its shape, so the whole bundle moves through each element in one batched operation.
4. **Classmethod entry points**, each for a different purpose:

   | Constructor | What it gives | Used for |
   |---|---|---|
   | `from_state` | Your own tensor. A 4-column input gets δ = 0 | Anything custom |
   | `from_fan` | A 2D fan of slopes in the x–z plane | Ray diagrams (there's no separate 2D code) |
   | `from_parallel` | Parallel rays across ±radius | Ray diagrams |
   | `from_cone` | Rings of slopes, equally spaced in angle | Diagrams and aberration fits. **Not** uniform in solid angle |
   | `from_disk` | A filled cone, uniform in solid angle (sunflower spiral) | Anything weighted by current |
   | `from_object_points` | A cone from each of several object points | Off-axial aberration fits |
   | `principal` | Four unit rays (unit x, x′, y, y′) | Reading off a system's matrix |

5. **Complex views:** `w = x + iy` and `slope = x′ + iy′`, used by the aberration fits in step 8.
6. **`replace()` shares tensors; it doesn't copy them.** It calls `dataclasses.replace`, which builds a new `Rays` through the constructor without copying the tensors. Every element returns `rays.replace(state=...)`, which is why gradients flow unbroken from source to detector.

**Questions**

1. The transfer matrices act on `[x, x′, y, y′, δ, 1]`. Why doesn't `Rays.state` store that constant 1?
2. A ray hits a hard aperture. What happens to its `state`, `alive`, `weight` and `z`? What would break if it were removed from the tensor instead?
3. You want the fraction of the source current that passes an aperture. `from_cone` or `from_disk`? What exactly goes wrong with the other?
4. How do you reproduce a 2D (r, r′) meridional ray diagram with this class?
5. Predict the shape and dtype of `Rays.from_state(torch.zeros(3, 4)).state`.
6. What does tracing `Rays.principal()` through a purely linear system give you, and why is that useful?

<details><summary><b>Answers</b></summary>

1. **So it can't be corrupted, and the stored state stays physical.** If the 1 were a stored column, anything that writes to or rescales `state` could silently break every deflection: an element that multiplies the state, or a hand-built tensor that forgets the 1. It's added inside `TransferMap.apply` instead. The higher-order terms (step 2) also act on the 5 physical coordinates only.
2. **`state` is unchanged, `alive` becomes False, and `z` moves to the aperture.** A hard aperture leaves `weight` alone, because everything downstream uses `weight * alive`. Removing the ray would break two things:
   - **Batching:** the tensor shape would change from element to element, and differently for each setting in a parameter sweep.
   - **Input–output pairing:** the aberration fit pairs row *i* of the input with row *i* of the output, and dropping rows breaks that correspondence.
3. **`from_disk`.** Each ring of `from_cone` has the same number of rays, so ray density per solid angle goes as 1/α and the centre is over-counted: the current through an aperture of semi-angle a would come out ∝ a instead of ∝ a². Also, all rays in a ring share one radius, so transmission jumps as an aperture edge crosses each ring, and the gradient is zero in between.
4. **`Rays.from_fan(semiangle, n)`** for a fan from one point, and `Rays.from_parallel(radius, n)` for a parallel bundle. Both are the 3D rays with y = y′ = 0. Note that `from_fan` takes radians.
5. **`(3, 5)`, float64.** The δ column is zero-padded, and `from_state` casts to float64 even though `torch.zeros` defaults to float32.
6. **The g and h rays (unit position, unit slope) in x and in y,** which any paraxial ray is a linear combination of. Traced through a linear system, their final states are **the columns of the system's matrix**, which is how a field lens's matrix will be built (step 7). There are four rays, not two, because a magnetic lens rotates x into y. With deflectors present, the final states are the columns plus the constant offset.

</details>

---

## Step 2: transfer maps

**Read:** [src/em_ray_tracer/transfer_map.py](../src/em_ray_tracer/transfer_map.py). The matrices are also written out in [architecture.md](architecture.md#the-first-order-maps).

**What to notice**

1. **The convention (module docstring).** A first-order map is a 6×6 matrix on `[x, x′, y, y′, δ, 1]`. Higher orders are stored separately in `higher` as T[i,j,k], U[i,j,k,l], …, the convention the TRANSPORT beam-optics code uses, and act on the 5-vector only. The constant and linear parts therefore live only in the matrix.
2. **`TransferMap`** holds `matrix` and `higher`. `linear` is the 5×5 block and `offset` the constant column. `entry("x", "x'")` reads an element by name, so no code depends on remembering index numbers.
3. **`apply`** computes `state @ linear.T + offset`: exactly the 6×6 product with the 1 appended, without building the 6-vector. It also evaluates any higher-order terms.
4. **`compose` and `@`** follow matrix order: `second @ first` means apply `first`, then `second`. They raise an error for higher-order maps (see question 6).
5. **The builders** (`drift_map`, `rotation_map`, `thin_lens_map`, `quadrupole_map`, `deflection_map`, `translation_map`) are pure functions written as `identity + value × unit_matrix`, never by assigning into a matrix. That gives two things:
   - **Batching:** a batched `power` of shape `(B,)` produces a `(B, 6, 6)` stack of matrices through broadcasting.
   - **`torch.func` compatibility:** `vmap` rejects writing batched values into an unbatched tensor.
6. **Three builders are worth reading closely:**
   - `deflection_map`: the kick α goes in the constant column, and energy dispersion k·α in the δ column.
   - `thin_lens_map`: applies rotation R(θ) after the focusing. This is how a magnetic lens's Larmor rotation appears.
   - `quadrupole_map`: R(ψ) · Q · R(−ψ). Rotate into the element's own frame, act, rotate back. Step 3 uses the same pattern for misalignment.

**Questions**

1. Describe the 6×6 matrix of a deflector (αx, αy), ignoring dispersion. Why can no 4×4 matrix on (x, x′, y, y′) represent it?
2. Rays leave a source, drift a distance L, then pass a thin lens of power P. Write the map from the source to just after the lens. Which factor goes on the left, and how do you apply it to a state tensor?
3. `apply` never appends the 1. Where does the deflector's kick enter the calculation?
4. For a magnetic deflector at 200 kV, `dispersion` = −γ/(1+γ) ≈ −0.58. Is a ray with δ > 0 deflected more or less than α? Why, physically?
5. Why is the drift exact at *any* order here? What would change if the state used angles θ instead of slopes?
6. `apply` handles second-order terms, but `compose` raises an error for them. Why is composition harder?

<details><summary><b>Answers</b></summary>

1. **The identity, plus αx in row x′ and αy in row y′ of the last (constant) column.** The kick doesn't depend on the ray; it's an affine map r′ = M r + a. No linear map of (x, x′, y, y′) produces a constant, so a 4×4 matrix can't represent it.
2. **`thin_lens_map(P) @ drift_map(L)`,** with the element met last on the left. `@` combines maps; to act on rays, call `(lens @ drift).apply(state)`. In practice `system.transfer_map(z0, z1)` builds this for you from the positions.
3. **In `offset`,** the column `matrix[:5, 5]` that multiplies the constant 1. `apply` adds it directly.
4. **Less:** k < 0, so α(1 + kδ) < α. Magnetic deflection is α = eBL/p, and a higher-energy electron has more momentum, so it's stiffer. This is the first-order dispersion an energy filter uses.
5. **Because the coordinate is the slope dx/dz,** a straight line is exactly linear: x = x₀ + x′L. With angles it would be x₀ + L tan θ = x₀ + Lθ + Lθ³/3 + …, so every drift would add spurious "aberrations" purely from the choice of coordinates. That's also why `from_cone` uses tan(semiangle).
6. **Truncation and feed-down.** Composing (M₂, T₂) after (M₁, T₁) gives a second-order part M₂T₁ + T₂(M₁ ⊗ M₁), plus third- and fourth-order leftovers that must be cut off. Worse, a constant offset a in the first map (a deflection or misalignment) makes T₂(a + M₁s)(a + M₁s) produce *new* linear and constant terms. Physically, a misaligned lens with spherical aberration produces coma and astigmatism. `apply` avoids all of this because it just evaluates the polynomial at the ray's numbers.

</details>

---

## Step 3: the component contract

**Read:** [src/em_ray_tracer/utils.py](../src/em_ray_tracer/utils.py), then [src/em_ray_tracer/components/base.py](../src/em_ray_tracer/components/base.py)

**What to notice**

1. **`as_parameter`** wraps every physical quantity as a float64 `nn.Parameter` with `requires_grad=False`. Being a parameter, it automatically:
   - appears in `.parameters()` and `state_dict`;
   - moves with `.to(device)`;
   - can be swapped out by `torch.func.functional_call`.

   Starting frozen means nothing is optimized unless you say so.
2. **Every component has `z`, `shift` and `tilt` parameters.** `z_start` and `z_end` both return `z` for thin elements; extended elements override them.
3. **`learn(*names)` / `freeze(*names)`** find parameters by name through `named_parameters()`, so nested names work: `learn("field.B0")`. They return `self`, so you can write `ThinLens(...).learn("power")` inline.
4. **`local_transfer_map` vs `transfer_map`: the central design point.** A subclass writes only the physics of the *aligned* element in `local_transfer_map`. The base class's `transfer_map` adds misalignment by changing frame:
   - `to_local` shifts the ray by −(s + t·Δz) in position and −t in slope, into the element's frame;
   - the element acts;
   - `to_lab` shifts back.

   Aligned elements skip this entirely (`is_misaligned`). `local_transfer_map` returns `None` for an element that isn't a polynomial map.
5. **`apply`** by default applies the map and moves `z` to `z_end`. Elements with side effects override it: masks, weights, or kicks that depend on sign(x).
6. **Energy comes from the beam:** `voltage` arrives from `Rays.voltage` and is never stored on the component.
7. **Config round trip.**
   - `__init_subclass__` records every subclass by class name.
   - `config()` writes out the fields listed in `config_fields`.
   - `ComponentBase.from_config` looks the class up by the saved `"type"` string and calls its constructor.

**Questions**

1. After `lens = ThinLens(power=0.5, z=1.0)`, which parameters does `lens` have, and which require gradients? What changes after `lens.learn("z")`?
2. A new component writes only `local_transfer_map`. How does it support `shift` and `tilt`? Name one advantage of this split.
3. A thin lens of power P is shifted by (sₓ, 0). What does the frame change do to x′? What simpler combination of elements is that equivalent to?
4. When does `transfer_map` return `None`? What must such a component override, and what happens to the system's overall matrix?
5. You add `Stigmator(strength, *, z, orientation=0.0)`. What must you do so `OpticalSystem.from_config(system.to_config())` rebuilds it?
6. Why is `voltage` an argument to `transfer_map` rather than stored on the component?

<details><summary><b>Answers</b></summary>

1. **Five parameters, all frozen:** `z`, `shift` (2 values), `tilt` (2 values), `power`, `rotation`. After `learn("z")`, `z` requires gradients and the lens position is optimizable. Moving it changes the drift before *and* after it, because both are differences of positions.
2. **The base-class `transfer_map` wraps `local_transfer_map` in the frame change.** Advantages:
   - Each element's physics is written once, aligned, and misalignment works for every element, including future ones.
   - Learnable misalignment (for alignment and calibration fits) comes free.
   - Aligned elements skip the extra step, so it costs nothing.
3. **x′ → x′ − P(x − sₓ) = (x′ − Px) + Psₓ:** the centred lens plus a constant deflection P·sₓ. A ray through the displaced lens centre (x = sₓ) isn't bent, which is the physical check. See `test_shifted_lens_is_centred_lens_plus_deflection`.
4. **Only when `local_transfer_map` returns `None`,** for a non-polynomial element such as the biprism. An aligned element still returns its map; `is_misaligned` only skips the frame change. Such an element must override `apply`. The trace's `transfer_map` becomes `None`, and `system.transfer_map()` raises an error.
5. Three things; registration itself is automatic through `__init_subclass__`:
   - set `config_fields = ("strength", "orientation")`;
   - store those names as attributes (`self.strength = as_parameter(strength)`, …);
   - make `__init__` accept them as keywords and pass `**kwargs` to the base class for `name`, `shift` and `tilt`.
6. **Energy is a property of the beam, not the lens.** The same lens acts differently at 60 kV and 300 kV, and differently on each off-energy ray. Storing it per component would duplicate it and let components disagree.

</details>

---

## Step 4: the concrete elements

**Read:** [src/em_ray_tracer/components/thin.py](../src/em_ray_tracer/components/thin.py)

**What to notice**

1. **`ThinLens`**:
   - It stores `power` = 1/f; `from_focal_length` is the alternative entry point, and `focal_length` is computed from the power.
   - `rotation` is the Larmor rotation of a magnetic lens.
   - Its map is exact at every order: an ideal thin lens has no aberrations.
2. **`Quadrupole`** produces pure two-fold astigmatism (C12). A stigmator is one of these with a chosen `orientation`.
3. **`Deflector`.** `kind` is a plain string setting, not a parameter. `dispersion_coefficient` computes k from `Rays.voltage`; without a voltage, the deflector acts as achromatic.
4. **`DoubleDeflector`**, an *extended* element spanning [z, z₂] (it overrides `z_end`):
   - **It stores what you control:** `tilt` and `beam_shift` at a `pivot` plane.
   - **The coil excitations are computed** by the `excitations` property, and `from_excitations` converts back from measured values.
   - **The change of variables is linear and invertible, so nothing is lost.** A pure pivot is `learn("tilt")` with `beam_shift` frozen, and it holds exactly at every optimizer step.
5. **`Biprism`** is non-polynomial:
   - `local_transfer_map` returns `None`.
   - `apply` kicks each ray by −α·sign(u), where u is its distance from the wire, and marks rays that hit the wire as dead.
   - Having no map, it handles `shift` itself.
6. **`Aperture`**:
   - `transmission` is 0 or 1 for a hard edge. With `edge_width > 0` it's a smooth step, linear in r², across the band radius ± edge_width/2.
   - Its map is the identity.
   - `apply` changes `alive`, and, for a soft edge, also multiplies `weight`.
7. **`Plane`** does nothing to the rays. The solver records the bundle there (step 6). Its `z` can be learned, e.g. to move a detector onto the focus.

**Questions**

1. Why store `power` rather than focal length? What goes wrong when optimizing f directly if the best answer is a very weak lens, or needs a sign change?
2. After `dd = DoubleDeflector((β, 0), z=0, z_2=0.05, pivot=0.2)` and `dd.learn("tilt")`, does the beam stay centred at the pivot throughout an optimization? What if you had instead stored the two excitations and learned them?
3. Why can't the biprism have a transfer map? Name two consequences for a system that contains one.
4. In a soft aperture, which rays contribute to d(transmission)/d(radius)? What goes wrong if `edge_width` is small compared with the spacing between rays? What units is `edge_width` in?
5. A `ThinLens` and an `Aperture` sit at exactly the same z. In what order are they applied, and does the order change the result? When *would* order at the same z matter?
6. You build a system of `ThinLens`es only and fit C30 at the Gaussian image. What do you get, and what in the roadmap changes that?

<details><summary><b>Answers</b></summary>

1. **Optimizing f is badly behaved in three ways:**
   - a weak lens needs f → ∞, so the optimum sits at infinity;
   - a sign change means jumping from f = +∞ to f = −∞;
   - the map depends on 1/f, so d(loss)/df ∝ 1/f² vanishes for weak lenses and blows up for strong ones.

   With power, the map is linear in P and zero strength is an ordinary point.
2. **Yes: `beam_shift` stays frozen and the excitations are recomputed from `tilt` at every step,** so the pivot condition holds exactly (tested in `test_learning_tilt_keeps_the_pivot`). With the excitations as the stored parameters, the optimizer would move them freely and the pivot would drift. The lesson: **store the quantities you want to control**, as with power vs f.
3. **Its kick is −α·sign(u), a discontinuous function; no polynomial of any order represents it.** Consequences:
   - it must override `apply`;
   - the system has no overall matrix: `trace.transfer_map` is `None`, and `system.transfer_map()` / `gaussian_image()` raise, so `ImagePlane`, `Magnification` and `FocalLength` can't be used across a biprism.
4. **Only rays inside the band radius ± edge_width/2.** Rays further in have weight exactly 1 and rays further out exactly 0, so neither changes when the radius moves. If the band holds zero or one ray, transmission becomes a staircase and the gradient is mostly zero with spikes, so the optimizer stalls. Units: metres, a transverse length in the aperture plane, like `radius`.
5. **In the order you added them** (`OpticalSystem.ordered()` uses a stable sort on z). Here it doesn't matter: a thin element changes only slopes, as a function of position (x′ += f(x)), and two such kicks commute; the aperture only reads positions. It matters when an element also rotates positions: a thin magnetic lens with `rotation ≠ 0` next to a quadrupole.
6. **C30 ≈ 0 to round-off:** ideal thin lenses are exactly linear. Spherical aberration will first appear with field lenses integrated non-paraxially (item 5 of the roadmap in architecture.md), and later in the higher-order maps derived from them.

</details>

---

## Step 5: the optical system

**Read:** [src/em_ray_tracer/system.py](../src/em_ray_tracer/system.py)

**What to notice**

1. **`Gap` and `Element`** are the only two things a solver ever sees.
2. **Components live in an `nn.ModuleDict` keyed by name.** That's what makes `parameters()`, `state_dict()` and `.to(device)` work, and it names parameters `components.<name>.<parameter>`, e.g. `components.C1.power`.
3. **Entry points and container:**
   - `from_components`, `from_config` / `to_config`;
   - `add` gives unnamed components names like `ThinLens0`, and rejects duplicate or invalid names;
   - `system["C1"]` looks up by name; `system[0]` is the most upstream component.
4. **`ordered()`** sorts by `z_start`, detached from the gradient, on every call. If a lens moves past another during optimization, the next trace uses the new order.
5. **`segments()`, the heart of the file.** The loop is `Gap(cursor, c.z_start)`, then `Element(c)`, then `cursor = c.z_end`, then a final gap to `z_end`:
   - Gap lengths are differences of `z` parameters, which is why positions are differentiable.
   - Gaps are emitted even at zero length.
   - Components outside the span are skipped.
   - A component that straddles an end of the span, or overlaps another, raises an error.
6. **`forward(rays)`** traces with the default matrix solver, so `torch.func` tools work on the system.
7. **Closed-form optics:**
   - `transfer_map` multiplies the element maps together.
   - `gaussian_image` packs the map's x-columns into complex numbers e^{iθ}(A, B, C, D), sets d = Re(−B/D), and returns the complex magnification e^{iθ}(A + dC).
   - `focal_length` is complex in the same way.

   These are only meaningful for round systems.
8. **`get_optimization_parameters`** returns one group holding every learnable parameter by default. If `optimizer_params` is keyed by component names, it returns one group per component.

**Questions**

1. The rays start at z = 0; there's a `ThinLens` at 1, an `Aperture` at 2 and a `Plane` at 3. Write out `segments(0.0)`. Which tensors is the first gap's length computed from?
2. Why emit zero-length gaps instead of skipping them?
3. During optimization, lens A (z = 2.0) moves past lens B (z = 2.5). What happens to the order in the next trace? Does the gradient "know" that a swap is possible? Is the loss continuous across it?
4. Why does `gaussian_image` combine entries into complex numbers instead of reading M[x, x]? What would M[x, x] + d·M[x′, x] return for a magnetic lens that rotates the image by θ?
5. In `functional_call(system, {"components.C1.power": p}, (rays,))`, which component's power is replaced? What would a name like `components.0.power` have been fragile against?
6. You want C1 to learn with a larger step size than C2. What do you pass as `optimizer_params`, and what has to match?

<details><summary><b>Answers</b></summary>

1. **`Gap(0, 1)`, `Element(lens)`, `Gap(1, 2)`, `Element(aperture)`, `Gap(2, 3)`, `Element(plane)`, `Gap(3, 3)`.** The last gap runs to `z_end`, which defaults to the end of the last component. The first gap's length is `lens.z − rays.z`, so `lens.z` enters with +1 here and −1 in the next gap: moving the lens trades one drift against the other.
2. **To keep the gradient.** If a gap were skipped whenever it happened to be exactly zero long (a `Plane` placed at a lens, say), the graph would lose its dependence on those positions, and the gradient for moving the plane would come out zero though the true derivative isn't. It also keeps the structure uniform, which keeps every solver simple.
3. **B comes first in the next trace.** The order is decided from detached values at the start of each trace, so the gradient only describes the current order. For thin elements the loss is continuous across the swap, because coincident thin kicks commute (step 4, Q5). For extended elements, a step that makes them overlap raises the overlap error.
4. **Rotation mixes x into y, so M[x, x] = cos θ × (true A).** The real-only formula returns M·cos θ: it underestimates |M|, and at θ = 90° returns 0 (with d = 0/0). The complex version recovers both |M| and θ = arg M.
5. **The component named `C1`, whatever its position or insertion order.** Index-based names change meaning when components are added, removed or reordered.
6. **`{"C1": OptimizerParams.Adam(lr=1e-2), "C2": OptimizerParams.Adam(lr=1e-3)}`,** passed to `system.set_optimizer` or `tracer.optimize(optimizer_params=...)`. What must match:
   - the keys must be component names;
   - each listed component needs at least one learnable parameter;
   - all groups must use the same optimizer class;
   - LBFGS allows only one learning rate for all groups.

   A learnable component that isn't listed is left out of the optimization.

</details>

---

## Step 6: solvers

**Read:** [solvers/params.py](../src/em_ray_tracer/solvers/params.py) → [solvers/base.py](../src/em_ray_tracer/solvers/base.py) → [solvers/transfer_matrix.py](../src/em_ray_tracer/solvers/transfer_matrix.py)

**What to notice**

1. **Configs are dataclasses in two groups:**
   - `IntegratorParams`: `RK4`, `DormandPrince45`, `VelocityVerlet`, `Boris`;
   - `SolverParams`: `TransferMatrix(order)`, `Paraxial(integrator)`, `NonParaxial(expansion_order, integrator)`, `Lorentz(integrator)`.

   Each has a class-level `_name`. `parse_dict` looks the class up by `name` or `type`, case-insensitively. Configs nest: `Paraxial.__post_init__` turns an `integrator` given as a dict into its dataclass, so a whole solver config can come from YAML.
2. **`Trace`** holds everything a trace returns: `initial`, `final`, `planes` (a dict by name), `transfer_map`, `trajectory` (only with `record=True`), and a `transmission()` method.
3. **`SolverBase.from_params`** is the one place a config becomes a solver, via a `match` on the dataclass type. It imports the solver classes inside the function, because importing them at the top would be circular.
4. **`propagate(component, rays) → (rays, map | None)` is the only method a solver implements.**
5. **`trace()` is the shared loop:**
   - `Gap` → the exact drift;
   - `Element` → `propagate`;
   - records the rays at every `Plane`;
   - multiplies each element's map into the overall map, or drops it to `None`;
   - optionally saves snapshots.
6. **`TransferMatrixSolver.propagate`** gets the element's map (for the overall map) and calls `apply` (to move the rays, including masks and weights).

**Questions**

1. Write the plain-dict config for a non-paraxial solver with the field expanded to 5th order and an RK4 step of 1 µm. What does `SolverBase.from_params` return for it today, and when does it fail?
2. A new solver implements one method. Name three things the shared loop does for it.
3. Why are gaps handled in the shared loop, not by each solver? Is anything lost for the ODE solver?
4. Give three different reasons `trace.transfer_map` can be `None`.
5. Your system has a `Plane` named `"specimen"`, and you trace with `record=False`. Where are the rays at the specimen, and what is `trace.trajectory`?
6. Why does `SolverParams.Paraxial` need a `__post_init__`, while `SolverParams.TransferMatrix` doesn't?

<details><summary><b>Answers</b></summary>

1. **`{"name": "nonparaxial", "expansion_order": 5, "integrator": {"name": "rk4", "dz": 1e-6}}`.** `from_params` returns an `ODESolver` with `NonParaxialEquations(5)` and `RK4(dz=1e-6)`; constructing it succeeds. It fails only when a trace reaches a `FieldLens` and calls the stubbed `bind`. A system of thin elements traces fine. By contrast, `SolverParams.Lorentz()` fails immediately in `from_params`.
2. The shared loop:
   - walks the segments and applies the exact drift across each gap;
   - records the rays at each `Plane`;
   - builds up the overall transfer map, or gives up with `None`;
   - saves snapshots when `record=True`;
   - packages everything into the `Trace`.
3. **In slope coordinates a field-free drift is exact at every order and energy,** so no solver could do better, and integrating it would only add error and cost. Nothing is lost: the ODE solver integrates exactly where there is field, inside each `FieldLens`'s span (±10a by default). Field tails beyond that span are ignored, but that's a choice made by the field lens.
4. Three cases:
   - an element with no map (`Biprism`);
   - a field lens under the ODE solver, whose rays are integrated numerically with no map;
   - any map with higher-order terms, since composition isn't implemented beyond first order.

   The rays are still traced correctly; only the single system matrix is unavailable.
5. **`trace.planes["specimen"]`,** or equivalently `trace["specimen"]`. `trace.trajectory` is `None`. Snapshots are opt-in (for plotting), so optimization loops don't store every intermediate bundle.
6. **`Paraxial` has a nested config.** Built from a dict, its `integrator` arrives as a plain dict, and `__post_init__` turns it into `IntegratorParams.RK4(...)`; otherwise the solver's `case IntegratorParams.RK4(...)` would never match. `TransferMatrix` has only an `int`.

</details>

---

## Step 7: the stubs, where the next work goes

**Read**, in this order:
1. [components/field_lenses.py](../src/em_ray_tracer/components/field_lenses.py)
2. [fields/base.py](../src/em_ray_tracer/fields/base.py) → [analytic.py](../src/em_ray_tracer/fields/analytic.py) → [expansion.py](../src/em_ray_tracer/fields/expansion.py) → [sampled.py](../src/em_ray_tracer/fields/sampled.py)
3. [solvers/ode.py](../src/em_ray_tracer/solvers/ode.py) → [solvers/integrators.py](../src/em_ray_tracer/solvers/integrators.py)
4. [tests/test_stubs.py](../tests/test_stubs.py)

Everything here is fully connected to the rest of the package; only the physics raises `NotImplementedError`.

**What to notice**

1. **`FieldComponent` / `FieldLens`:**
   - The lens occupies `z ± field.half_extent` (10a by default), and the field's own coordinate is measured from the lens centre.
   - Entry points: `glaser`, `schiske`, `from_axial_samples`.
   - `local_transfer_map` is a stub. It will trace `Rays.principal()` through the field with the paraxial solver (step 1, Q6) to get the thick-lens matrix, which for a magnetic lens has the R(θ)·L structure.
2. **`AxialField.derivative(n, z)`,** not just `B(z)`: the off-axis expansion to order N needs derivatives up to 2N+1. `Field3D.evaluate(x, y, z) → (E, B)` is the interface for full 3D fields.
3. **`GlaserField` and `SchiskeField`:**
   - Their shape parameters (`B0`, `a`, `k`) are `nn.Parameter`s, so `lens.learn("field.B0")` will optimize an excitation.
   - The module docstring gives the exact nth derivative of 1/(a² + z²), which is what to implement.
4. **`LaplaceExpansion`** builds the off-axis field from the axial profile. Its docstring gives the series and a divergence identity to test it with. `SampledAxialField` and `FieldMap3D` are the FEM entry points.
5. **`ODESolver`:**
   - `propagate` sends field components to `integrate` and everything else to `apply`.
   - `integrate` requires `rays.voltage`, gets f(z, state) from `equations.bind(component, voltage)`, and runs fixed-step RK4.
   - `bind` is the stub. `ParaxialEquations` gives the lab-frame equations in its docstring.
6. **Integrators.** `rk4_step` and `velocity_verlet_step` are written and tested for their order of convergence; `dopri45_step` and `boris_push` are stubs.
7. **`test_stubs.py`** marks each test `xfail(strict=True, raises=NotImplementedError)`. The expected values are relativistic.

**Questions**

1. What span of z does `FieldLens.glaser(B0, a, z=0)` occupy? What happens if you also place a `ThinLens` at z = 5a?
2. Why does `AxialField` expose `derivative(n, z)` rather than just `B(z)`? Which derivatives does the paraxial equation need?
3. Predict what happens today when you trace a system containing a `FieldLens` with:
   - (a) `TransferMatrixSolver()`;
   - (b) `ODESolver.paraxial()`;
   - (c) `ODESolver.paraxial()`, but rays with `voltage=None`.
4. Once the paraxial equations exist, how would you build `FieldLens.local_transfer_map` in a few lines?
5. You implement `GlaserField.derivative` correctly. What does pytest report for its test, and what should you do? What if your implementation is wrong, so the assertion fails?
6. Which integrator would you choose for the paraxial Glaser trace, and why not velocity-Verlet?

<details><summary><b>Answers</b></summary>

1. **[−10a, +10a].** The span is `extent · a`, with `extent` = 10 by default, where the field has fallen to 1% of its peak. A `ThinLens` at 5a lies inside it, so `segments()` raises the overlap error; superposing overlapping fields isn't supported yet.
2. **The off-axis (Laplace) expansion to order N needs derivatives up to 2N+1,** e.g. the 7th at N = 3. Finite differences lose all their significant digits by then, hence the exact formula. The paraxial equation needs B and B′ for a magnetic lens (B′ enters the lab-frame x–y coupling), and φ, φ′, φ″ for an electrostatic one.
3. Today:
   - (a) `FieldLens.local_transfer_map` raises `NotImplementedError`.
   - (b) `ParaxialEquations.bind` raises `NotImplementedError`.
   - (c) `ValueError`: field components need the beam energy. This check runs before `bind`.
4. **Trace `Rays.principal(z=lens.z_start, voltage=...)` through the lens alone with `ODESolver.paraxial()`.** The final states give the columns of the 4×4 transverse block. Put them in a 6×6 identity; the δ entries stay zero at first order, because chromatic terms (x·δ) are second order.
5. **pytest reports `XPASS(strict)` as a failure; delete the `@stub` marker from that test.** A wrong implementation raises `AssertionError` rather than `NotImplementedError`, so pytest reports an ordinary failure. The marker never hides a real bug.
6. **RK4 (later, adaptive DOPRI45).** It's fourth order and handles velocity-dependent terms: the magnetic coupling s·B·y′ and the electrostatic damping term. Velocity-Verlet is second order and symplectic only when the force doesn't depend on velocity.

</details>

---

## Step 8: aberration fitting

**Read:** [aberrations/basis.py](../src/em_ray_tracer/aberrations/basis.py) (start with the module docstring) → [aberrations/fit.py](../src/em_ray_tracer/aberrations/fit.py) → [aberrations/coefficients.py](../src/em_ray_tracer/aberrations/coefficients.py)

**What to notice**

1. **The model.** Each term of the aberration function is χ_k = Re[c_k · norm_k · ω^a ω̄^b w^c w̄^d]:
   - ω = x′ + iy′ is the input slope, and w = x + iy the input position.
   - The ray deviation is Δw = 2 ∂χ/∂ω̄ = c ∂_ω̄ f + c̄ conj(∂_ω f), which is linear in (Re c, Im c). `AberrationTerm.columns` computes the two columns.
   - Terms whose monomial is real (a = b and c = d) have only Re c.
2. **Axial terms use polar notation.** `axial_term(n, m)` has powers ((n+1−m)/2, (n+1+m)/2), norm 1/(n+1), and c = C_nm e^{imφ_nm}. `AberrationBasis.axial(3)` gives `shift, C10, C12, C21, C23, C30, C32, C34`.
3. **`round_lens(3)`** adds the off-axial Seidel terms: `magnification`, `coma`, `field_astigmatism`, `field_curvature`, `distortion`. Imaginary parts are the anisotropic aberrations of magnetic lenses. **`full(order)`** has every monomial, for systems without symmetry.
4. **Every basis includes `shift`,** which absorbs deflections. `magnification` absorbs the Gaussian image, rotation included.
5. **The fit (`fit_coefficients`):**
   - stacks real and imaginary parts into one real least-squares problem;
   - normalizes columns;
   - solves by QR, which is differentiable;
   - names any term the rays don't constrain;
   - reports the RMS residual;
   - weights each ray by `weight × alive`.
6. **`AberrationCoefficients`:**
   - Entry points `fit`, `from_rays`, `from_trace`.
   - The reference is the weighted mean input position and slope.
   - **Convention:** positions at the reference plane against input slope and position, so a drift Δz past a focus gives C10 = Δz.
   - `rescaled(slope_scale)` refers the coefficients to image-side slopes; `angle` is arg(c)/m; `to_polar()` gives `{"C10", "C12", "phi12", ...}`.

**Questions**

1. Aberrations are non-linear in ω. Why is the fit still a *linear* least-squares problem?
2. What ray deviation does χ = ¼ C30 |ω|⁴ produce? For a point source and a detector a distance Δz past it (no lenses), what is C10?
3. Why are C10 and C30 real, while C12 is complex? What does the phase of C12 mean?
4. You fit `AberrationBasis.axial(1)` to `Rays.from_fan(...)`. What happens, and why?
5. Why normalize the columns before the QR solve?
6. How would you optimize a system to reach a target C30, and why does a gradient exist?
7. A deflector sits before the detector. Where does the deflection show up in the fit, and why does that matter?

<details><summary><b>Answers</b></summary>

1. **The model is linear in the *coefficients*,** even though each column is a non-linear function of ω and w evaluated at every ray.
2. **Δw = C30 |ω|² ω.** For the drift, Δw = Δz·ω, so **C10 = Δz** (`test_defocus_fits_as_C10`).
3. **Their monomials are real** (|ω|², |ω|⁴), so Re[c f] depends only on Re c. C12's monomial ω̄² has an orientation: c = C12 e^{2iφ12}, and the phase gives the astigmatism axis φ12 = arg(c)/2.
4. **`ValueError` naming C12.** A meridional fan has ω real, so ω̄ = ω and C12's real column duplicates C10's: the fit can't separate x- from y-astigmatism. Use `from_cone` or `from_disk`.
5. **Column scales differ enormously.** At 10 mrad a constant column is ~1 and an ω³ column ~1e-6. Unnormalized, the small columns would be poorly determined, and the rank check would flag them wrongly.
6. **Add `AberrationTarget("C30", target)` to `tracer.optimize`.** Ray positions depend differentiably on the parameters, and QR plus a triangular solve are differentiable, so gradients flow through the fit. The detector-to-focus test does exactly this with C10.
7. **In `shift`, the constant term.** Without it, a deflected spot would bias C10 and the other low-order terms (`test_deflection_goes_to_shift_not_C10`).

</details>

---

## Step 9: optimizers and objectives

**Read:** [optimize/params.py](../src/em_ray_tracer/optimize/params.py) → [optimize/mixin.py](../src/em_ray_tracer/optimize/mixin.py) → [optimize/objectives.py](../src/em_ray_tracer/optimize/objectives.py)

**What to notice**

1. **Configs, in `params.py`:**
   - `OptimizerParams` (`Adam`, `AdamW`, `SGD`, `LBFGS`, `NoneOptimizer`) and `SchedulerParams` (`Plateau`, `Exponential`, `CosineAnnealing`, `NoneScheduler`) are dataclasses; `params()` returns the torch keyword arguments, and `parse_dict` works as for solvers.
   - LBFGS's stopping tolerances are tightened because losses are squared SI lengths.
2. **`OptimizerMixin` (`mixin.py`)** matches parameter groups from `get_optimization_parameters()` with configs by key:
   - all groups must use one optimizer class;
   - `step_optimizer(closure)` returns the loss;
   - `step_scheduler(loss)` handles `ReduceLROnPlateau`.

   `OpticalSystem` supplies the groups (step 5).
3. **`Objective`** is a dataclass with a keyword-only `weight`; calling it returns `weight × loss(evaluation)`.
4. **`Evaluation`** computes each of these the first time an objective asks, then reuses it within one loss evaluation: `trace` (and `rays(plane)`), `gaussian_image`, `focal_length`, and `aberrations(basis, plane)`. A fresh one is made per evaluation, because the parameters change in between.
5. **The objectives:**

   | Objective | What it measures |
   |---|---|
   | `RayTarget` | Squared distance of the rays from a target point |
   | `SpotSize` | RMS spot radius about the centroid |
   | `Transmission` | Current fraction; `at_least` penalizes only a shortfall |
   | `ImagePlane` | Where the paraxial image lies |
   | `Magnification` | Complex magnification; `ignore_rotation` matches the size only |
   | `FocalLength` | Its modulus |
   | `AberrationTarget` | One fitted coefficient |

   The losses aren't normalized: `weight` sets the trade-off between them.

**Questions**

1. Why does `step_optimizer` take a closure?
2. You call `tracer.optimize` twice on the same system, passing `optimizer_params` only the first time. What runs the second time?
3. How many ray traces happen in one loss evaluation of `[ImagePlane(5.0), Magnification(-2.0)]`? Of `[SpotSize(plane="s"), AberrationTarget("C10"), AberrationTarget("C30")]`? How many aberration fits in the second?
4. With `Transmission(0.3)`, what is the loss when transmission is 0.5? When it's 0.2?
5. You combine `SpotSize` (a squared length) with `Transmission`. What sets the trade-off, and how exactly is the minimum current held?
6. Sketch a custom objective that keeps lens `C2`'s focal length above 1 mm. Is there a better way to enforce it?

<details><summary><b>Answers</b></summary>

1. **LBFGS evaluates the loss several times per step** (its line search), so it needs a function that recomputes the loss and gradients. The other optimizers call the closure once.
2. **The first optimizer continues,** with its state intact, because it lives on the system. Pass `optimizer_params` or `reset=True` to start fresh.
3. **Zero traces in the first case:** both objectives use the transfer map, and the image plane is computed once. **One trace in the second,** shared, and **one aberration fit**, since both targets use the same basis and plane (`test_loss_traces_only_when_needed_and_shares_work`).
4. **0 at 0.5** (no shortfall). **(0.3 − 0.2)² = 0.01 at 0.2,** times `weight`.
5. **The weights.** The minimum is a soft penalty: at the optimum, the spot-size gradient balances the penalty's, so the current falls slightly short by an amount that shrinks as `Transmission`'s weight grows.
6. A sketch:
   ```python
   @dataclass
   class MinFocalLength(Objective):
       minimum: float

       def loss(self, evaluation):
           f = 1 / evaluation.tracer.system["C2"].power
           return torch.relu(self.minimum - f) ** 2
   ```
   That's a soft penalty. An exact alternative is to change the parameterization so the constraint can't be violated, e.g. a power that is a softplus of a free parameter (step 4, Q2).

</details>

---

## Step 10: the top-level class, then the examples

**Read:** [tracer.py](../src/em_ray_tracer/tracer.py), then run and read [examples/sem_two_condenser.py](../examples/sem_two_condenser.py) and [examples/optimize_lenses.py](../examples/optimize_lenses.py):

```bash
uv run python examples/sem_two_condenser.py
uv run python examples/optimize_lenses.py
```

**What to notice**

1. **`RayTracer`** holds `system`, `rays`, `solver` and `z_end`. `from_models(system, rays, solver=...)` accepts a solver object, a `SolverParams` dataclass or a dict, and defaults to first-order transfer matrices.
2. **Forward methods:** `trace(record)`, `gaussian_image()`, `fit_aberrations(basis, plane)`.
3. **`loss(objectives)`** builds one `Evaluation` and sums the objectives.
4. **`optimize(objectives, num_iters, optimizer_params, scheduler_params, reset)`:**
   - checks there is something to learn;
   - creates an optimizer only when asked, or when none exists (default LBFGS);
   - runs the closure loop and steps the scheduler with the loss;
   - records `losses` and `lrs`.
5. **The examples:**
   - `sem_two_condenser.py`: a ray diagram with apertures.
   - `optimize_lenses.py`, part 1: the legacy focal-length problem, run with Adam + Plateau and with LBFGS.
   - Part 2: a two-lens design for a given magnification and image plane.
   - Part 3: the smallest spot subject to a minimum current.

**Questions**

1. How do you run `sem_two_condenser.py` with the paraxial ODE solver instead? What changes in the output, and why?
2. Why does part 1 put the lens at z = 42·12/99 ≈ 5.09 rather than 5?
3. Why does LBFGS recover f = 4.2 only to about 1e-7?
4. Part 2 starts from P1 = 0.4, P2 = 1. What goes wrong starting from P1 = P2 = 1?
5. In part 3, what do `Rays.from_disk` and `edge_width` each contribute? What radius should the optimizer find, and why?
6. Change part 3 to require 50% of the current. Predict the radius before running it.

<details><summary><b>Answers</b></summary>

1. **Pass `solver=SolverParams.Paraxial()` (or `ODESolver.paraxial()`) to `RayTracer.from_models`.** Nothing changes: the system has only thin elements, which both solvers handle identically (`test_ode_solver_matches_transfer_matrix_on_thin_elements`).
2. **The legacy code snapped the lens onto its z-grid** (100 points over 12 m), from 5 to grid point 42 at 5.0909 m. The target number −6.4502… encodes that position, so the regression test uses it. The new code has no grid.
3. **The legacy target was computed in float32.**
4. **With P1 = 1 the intermediate image lands exactly at C2's front focal plane,** so the final image is at infinity and the loss is NaN. Starting points must avoid such singular configurations; this one keeps the intermediate image virtual all the way to the solution.
5. **`from_disk` samples the cone uniformly,** so summed weight is current. **`edge_width` makes current a smooth function of the radius,** so the gradient is useful. The spot at the defocused specimen grows with the aperture, so the optimum is the smallest aperture that passes 25%: R* = L·tan α·√0.25 ≈ 0.500 mm.
6. **R* = L·tan α·√0.5 ≈ 0.707 mm.**

</details>

---

## Next

Pick the first stub in the roadmap at the end of [architecture.md](architecture.md): `GlaserField.derivative`. Implement it, watch its test in `tests/test_stubs.py` turn from `xfail` into `XPASS(strict)`, remove the marker, and move on to `ParaxialEquations.bind`.
