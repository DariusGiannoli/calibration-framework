# Triaxial T1 calibration-task decision worksheet

This worksheet is the human review companion to
`triaxial_t1_declarations.yaml`. It separates values that must be decided
before prospective trajectory comparison, values that must be preregistered
before final acquisition, and results that can only be computed after
acquisition. Blank scientific values in `triaxial_t1.yaml` must remain `null`
until the stated authority or evidence source supports them.

## Scope already declared

The following parts of the task do not need numerical invention:

- **Sensor input, \(s\):** `c1`, `c2`, and `c3`.
- **Reference output, \(q\):** `Fx`, `Fy`, and `Fz`.
- **Model, \(M\):** memoryless affine calibration.
- **Mounting condition:** rigid.
- **Generalization, \(G\):** new independent runs of the same physical sensor
  under the declared mounting, temperature, and loading-rate conditions.
- **Stage boundary:** the compensator coil, soft embedding, new sensor units,
  and daisy-chained sensors are outside T1.

The supervisor should explicitly confirm that the generalization statement is
the intended first claim. Changing it to new sensor units or new unrepresented
conditions would change the validation design, not merely fill a blank.

## Classification of the 37 live D0 declarations

| Primary classification | Count | Meaning for T1 |
| --- | ---: | --- |
| 1. Physical or application requirement | 18 | Comes from the intended use, safe domain, error budget, or regions requiring a guarantee. |
| 2. Hardware or reference specification | 0 | No item in the current 37-item D0 snapshot is a direct instrument fact; those facts are reported separately by D1. |
| 3. Experimental-design choice | 9 | Must be selected and recorded prospectively as part of the coverage, confounding, or analysis design. |
| 4. Statistical threshold requiring simulation or pilot justification | 7 | Must be justified through prospective convergence, precision, power, or sensitivity work. |
| 5. Value evaluated only after acquisition | 3 | The per-axis calibrated-force uncertainty estimates are results, not planning assumptions. |
| **Total** | **37** | Must exactly match the live `evaluate-d0` missing-evidence list. |

Resolution timing is also explicit: 19 D2/D3 declarations are needed before
candidate-trajectory simulation, 15 D6 declarations must be fixed before final
acquisition and test evaluation, and 3 uncertainty estimates are populated
only after acquisition.

## Meeting decisions: first practical group

Record the agreed value, unit, authority, rationale, approver, and date in the
versioned task/evidence files. Do not use observed minima, maxima, or achieved
performance from the evaluated dataset as substitutes.

### 1. Intended force domain \(\Omega_1\)

Decide:

- `task.d2.domain.{Fx,Fy,Fz}.{minimum,maximum}` from intended load cases and
  the safe application envelope;
- `task.d2.excluded_regions` from physically impossible, unsafe, or explicitly
  out-of-scope joint combinations; use an explicit empty list if no region is
  excluded.

After the domain is agreed, choose each D2 `grid_points` value using a
prospective grid-convergence study. Then justify
`task.d2.maximum_fill_distance` by relating simulated trajectory spacing to
acceptable interpolation risk. These numerical-analysis settings are not
application force limits.

### 2. Relevant conditions \(\mathcal{C}_1\)

Confirm rigid mounting and decide the claimed temperature and loading-rate
bounds from intended deployment, not from whatever range a pilot happened to
achieve. Select the two condition-grid resolutions prospectively.

The D3 condition decision must then be one of:

- `held_fixed`, if T1 is deliberately restricted to single declared values;
- `included_in_model`, only if the model is extended to use the conditions;
- `demonstrated_invariant`, if a preregistered experiment will support that
  claim.

The chosen strategy must agree with the task claim. A versioned confounding
review should document trajectory configuration, run order, reinitialization,
temperature, loading rate, drift, and other nuisance variables.

### 3. Performance requirements \(\mathcal{P}_1\)

For each force axis, decide:

- maximum acceptable held-out RMSE from the downstream application error
  budget;
- maximum acceptable calibrated-force uncertainty from the application
  uncertainty budget;
- maximum RMSE confidence-interval half-width, justified by a prospective
  run-level precision simulation.

Predeclare the performance regions that matter independently, such as domain
boundaries, force-magnitude bands, loading directions, octants, or condition
strata. The regions must reflect use or risk, not patterns discovered in final
test residuals.

The confidence level, number of bootstrap repetitions, and minimum number of
held-out bootstrap units require prospective statistical justification. The
bootstrap seed and versioned uncertainty method are reproducibility and
analysis-design choices. The three `calibrated_force_uncertainty` values remain
`null` until the locked method is applied to acquired evidence.

### 4. Generalization claim \(G_1\)

Confirm all four existing declarations together:

- target: new experimental runs;
- independent unit: complete run;
- sensor scope: the same physical sensor;
- conditions: only the declared mounting, temperature, and loading-rate
  domain.

This confirmation governs D4 replication and the D5 development/test split.
Forward/reverse or clockwise/counterclockwise recordings count as independent
runs only when separately acquired and reinitialized.

### 5. D1 measurement-system validity

D1 declarations are high priority even though they are not among the 37 D0
paths. Resolve them from the following authorities:

| Destination | Primary classification | Legitimate source |
| --- | --- | --- |
| `sensor_profile.channels.{c1,c2,c3}.{valid_min,valid_max}` | 2. Hardware specification | Coil electronics and DAQ validated range, including clipping/saturation limits. |
| `reference_profile.channels.{Fx,Fy,Fz}.{valid_min,valid_max}` | 2. Reference specification | Reference F/T sensor manual and active configuration. |
| `reference_profile.channels.{Fx,Fy,Fz}.expanded_uncertainty` | 2. Reference specification | Current calibration certificate and its coverage statement. |
| `reference_profile.calibration_certificate_id` | 2. Reference specification | Identifier of the certificate used for the experiment. |
| `task.d1.maximum_reference_uncertainty.{Fx,Fy,Fz}` | 1. Application requirement | Allocated reference contribution within the per-axis uncertainty budget. |
| `setup.maximum_time_offset_s` | 2/3. Verified system capability and design limit | Timing architecture, synchronization validation, and acceptable dynamic error. |
| `setup.{sensor,reference}_acquisition_bandwidth_hz` | 2. Hardware/setup specification | Configured acquisition chain and anti-alias filtering evidence. |
| `setup.sampling_process_id` | 3. Experimental-design choice | Versioned sampling, resampling, and synchronization procedure. |
| `setup.reference_to_sensor_rotation` | 2. Setup/metrology evidence | Fixture geometry, coordinate convention, and measured alignment. |
| `task.d1.invalid_observation_policy_id` | 3. Experimental-design choice | Predeclared policy for missing, corrupt, clipped, saturated, or unsynchronized observations. |
| `task.d1.exclusion_record_id` | 5. Post-acquisition evidence | Immutable record of observations excluded under the policy. |
| `task.d1.exclusions_reviewed` | 5. Post-acquisition evidence | Recorded review after exclusions exist. |

The reference sensor's actual uncertainty and the task's maximum acceptable
reference uncertainty are different quantities and must come from different
sources.

## Gate before comparing trajectories

Spiral, Lissajous, factorial, random, space-filling, optimized, and hybrid
protocols can be compared prospectively only after the 19 D2/D3 declarations
are resolved. The comparison should report planned D2 joint-domain fill
distance and D3 affine design-matrix rank/conditioning for every candidate. It
must not declare any trajectory universally sufficient or claim final D1,
D4--D7, or calibration-performance evidence before acquisition.
