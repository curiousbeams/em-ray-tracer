# Progress

A checklist of what is implemented, to tick off as work lands. [architecture.md](architecture.md#what-is-stubbed-in-suggested-order) explains each stub, and [roadmap.md](roadmap.md) the milestones. A stub's test in `tests/test_stubs.py` is marked `xfail(strict=True)`; when it starts passing, delete the marker and tick the box here.

## M1: analytic round lenses and the z-domain ODE

- [x] `GlaserField.derivative` and `SchiskeField.derivative`: the closed-form pole derivative (`tests/test_fields.py`)
- [x] `ParaxialEquations.bind`: the lab-frame paraxial equations, magnetic and electrostatic (`tests/test_stubs.py`, `tests/test_paraxial.py`)
- [ ] `FieldLens.local_transfer_map`: the thick-lens matrix from the principal rays (`test_weak_glaser_lens_approaches_thin_lens_power`, `test_thick_lens_matrix_matches_ode_trace`)
- [ ] Second-order maps: `TransferMap.compose` for `higher`, and field-lens maps with chromatic aberration
- [ ] `LaplaceExpansion.evaluate` and `NonParaxialEquations.bind`, built as the ν = 0 case of `MultipoleField` (`test_laplace_expansion_divergence_identity`); fit C30 of a Glaser lens against its closed form
- [ ] `SampledAxialField` and `FieldMap3D`, with the time-domain Lorentz solver
- [ ] Integrators: `dopri45_step` and `boris_push`
- [ ] `config()` for field components, so systems with field lenses round-trip

## M2: `MultipoleField`

- [ ] On-axis multipole strengths a_ν(z), ν = 0…6, normal and skew, with exact derivatives
- [ ] Profiles: Glaser, Gaussian, Enge, and tabulated
- [ ] Tests: Glaser closed forms at ν = 0, divergence- and curl-free fields, a short quadrupole's C12, a hexapole's A2

## M3: aberration analysis

- [ ] Energy order on every term
- [ ] Rotational terms (columns × i)
- [ ] Statistics from the QR factors: covariance, errors, p-values, adjusted R²
- [ ] Conventions: `referred_to_object()`, frame rotation, a name map, angle–slope conversion
- [ ] Orders up to 7

## M4: external fields and full 3D placement

- [ ] Loaders for on-axis profiles, on-axis multipole exports, 2D (r, z) and 3D maps
- [ ] Smooth (C²) interpolation
- [ ] `Pose`: a full rigid transform for field components
- [ ] `FieldSuperposition`: overlapping fields

## M5: exact tracers

- [ ] `ZDomainLorentz` with adaptive DOPRI5(4) and exact screens
- [ ] `TimeDomainLorentz` with differentiable screen crossings
- [ ] Checkpointing for backpropagation

## M6: design and calibration

- [ ] Digital twin: fit excitations and misalignments to measurements
- [ ] Tolerancing: the Jacobian of aberrations with respect to misalignments
- [ ] Trade-offs: Pareto fronts from batched weight sweeps
