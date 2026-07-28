# Software architecture

The software is an executable implementation of the criteria defined in the
research framework. Scientific requirements are defined in the paper first;
the software applies those requirements reproducibly.

## Layers

1. **Schema layer** — typed models describe task, instrument, setup, and result
   structures. The models can emit JSON Schema for a future GUI.
2. **Configuration layer** — YAML files store user-supplied instrument facts,
   setup facts, and application requirements.
3. **Criterion engine** — each universal criterion is implemented once and is
   parameterized by the declared task.
4. **Application layer** — a task YAML file maps a specific dataset and set of
   profiles to the universal criterion.
5. **Evidence layer** — evaluations return structured status, metrics, missing
   evidence, and violations.

```text
Future GUI
    |
    v
Typed schema -> validated YAML profiles -> criterion engine -> JSON evidence
```

The GUI will edit the same schema-backed configuration. It will not contain a
second copy of the criterion logic.

## Configuration ownership

| Configuration | Contains | Must not contain |
| --- | --- | --- |
| Sensor profile | channel units and valid hardware ranges | observed dataset minima/maxima |
| Reference profile | valid ranges, actual uncertainty, certificate evidence | application acceptance limits |
| Setup profile | synchronization tolerance and frame transformation | dataset observations |
| Task profile | calibration claim, applicability, column mappings, and criterion-specific requirements | inferred hardware specifications or observed dataset extrema |
| Declaration register | classification, resolution timing, decision question, and legitimate source for each unresolved D0 path | guessed scientific values or a second copy of resolved task values |
| Evidence manifest | file roles, SHA-256 records, versions, seeds, and recorded results | undocumented files or mutable identifiers |

Unknown scientific values are represented explicitly as `null`. Missing values
produce an `INDETERMINATE` result; the engine never invents defaults.

## Criterion result semantics

- `PASS`: all required evidence is present and no violation was detected.
- `FAIL`: at least one observed or configured value violates a declared rule.
- `INDETERMINATE`: no violation is known, but required evidence is missing.
- `NOT_APPLICABLE`: the task explicitly makes the underlying obligation
  irrelevant and records a justification.

A known failure takes precedence over missing evidence. Configuration syntax or
schema errors are execution errors rather than scientific criterion results.

## D0 implementation

D0 checks that the executable task explicitly and consistently declares the
sensor inputs, reference outputs, force domain, operating conditions, model
family, performance requirements, and generalization unit. It cross-checks the
claim against D2, D3, D4, D5, and D6 rather than accepting free-text
declarations. Missing claim elements produce `INDETERMINATE`; contradictory
declarations produce `FAIL`.

For the real triaxial T1 template, a typed declaration register classifies each
live D0 missing-evidence path. `audit-declarations` compares the register with
the evaluator output and reports paths that are unclassified or no longer
missing. The register intentionally stores resolution metadata rather than
scientific values: once supported, values belong in the task, instrument, or
setup profile that owns them.

The current 37-item snapshot contains D2, D3, and D6 paths. Direct hardware and
reference specifications are absent from that count because D1 evaluates them
from instrument and setup profiles. Per-axis calibrated-force uncertainty
estimates remain classified as post-acquisition results, so an aligned register
does not imply a complete claim or a D0 `PASS`.

## D1 implementation

D1 checks:

- presence and numeric validity of required sensor/reference channels;
- timestamp ordering and sensor/reference synchronization;
- sensor and reference valid ranges;
- existence and validity of the reference-to-sensor rotation;
- acquisition bandwidth and sampling-process declarations when required;
- an identified invalid-observation policy and exclusion record;
- confirmation that exclusions were reviewed;
- actual reference uncertainty against task-specific maximum uncertainty;
- reference calibration evidence.

The initial implementation treats any invalid retained observation as a D1
failure. Filtering or excluding observations must happen through a documented
preprocessing step, after which D1 is run again on the retained dataset. This
avoids introducing an arbitrary allowable-invalid-sample percentage.

## D2 implementation

D2 uses achieved reference measurements and recorded operating conditions from
a D1-valid dataset. It:

- requires D1 to return `PASS`;
- requires declared bounds and grid resolution for every continuous force or
  condition dimension;
- evaluates categorical and fixed conditions as joint strata;
- applies the configured reference-to-sensor rotation;
- normalizes the complete continuous \(\Omega\times\mathcal{C}\) domain;
- evaluates the nearest achieved point inside each claimed stratum;
- removes only predeclared, justified excluded regions from the claim;
- reports the estimated fill distance and worst-covered domain point; and
- compares the estimate with the declared maximum fill distance.

D2 never infers the intended domain, conditions, exclusions, grid resolution,
or acceptance threshold from observed data. Missing declarations produce
`INDETERMINATE`.
The report also includes the normalized grid covering radius so the grid
approximation can be assessed and refined explicitly.

## D3 implementation

D3 evaluates whether the supplied fitting dataset identifies the declared
parametric model with acceptable numerical conditioning. It:

- requires D1 to return `PASS`;
- requires an explicit model adapter and normalization bounds;
- constructs the model sensitivity/design matrix;
- computes singular values, numerical rank, and condition number;
- checks full parameter rank and the declared condition-number limit; and
- requires identifiers for the model-specific test and confounding review;
- records whether conditions were held fixed, modeled, or demonstrated
  invariant; and
- reports the weakest feature direction to help diagnose confounding.

The first model adapter implements the affine calibration defined in the paper.
Additional polynomial, nonlinear, or dynamic adapters can be added without
changing D3 result semantics. D3 does not require D2 to pass: domain coverage
and model identifiability are separate criteria.

## D4 implementation

D4 evaluates independent replication and within-run temporal dependence. It:

- requires D1 to return `PASS`;
- identifies runs and trajectory configurations using declared CSV columns;
- requires a declared dependence method and stationarity review for the
  initial autocorrelation-based effective-sample-size estimator;
- requires auditable evidence that every run has a unique acquisition start
  and completed initialization and zeroing procedures;
- checks the total independent-run and per-configuration repetition limits;
- estimates fixed-lag autocorrelations separately within each run;
- sums the resulting per-run effective sample sizes for each declared signal;
  and
- compares the limiting signal with the declared minimum effective sample size.

The lag-\(k\) estimator uses the centered lagged product divided by the total
centered sum of squares for that run. Constant signals, runs no longer than the
declared lag, and non-positive effective-sample-size denominators are reported
as failures because the declared D4 calculation cannot be supported.

Run labels alone are not evidence of independence. Splitting one continuous
recording into several labels cannot satisfy D4 because verified runs require
unique acquisition identifiers and explicit acquisition, initialization, and
zeroing evidence. D4 requires D1 but not D2 or D3.

## D5 implementation

D5 evaluates whether final performance evidence comes from a frozen,
leakage-resistant partition aligned with the declared generalization unit. The
initial application uses the complete experimental run as that unit. It:

- requires D1 to return `PASS`;
- requires an explicit, versioned development/test run manifest;
- verifies that the split is complete, disjoint, and contains the declared
  minimum number of held-out runs;
- checks provenance declarations for preprocessing, model selection, parameter
  estimation, performance-threshold selection, model locking, and final test
  evaluation;
- records the development-data selection or group-aware resampling method;
- fits the affine model declared by D3 using development-run samples only; and
- reports per-axis RMSE using held-out-run samples only.

The report includes a canonical SHA-256 digest of the split manifest. D5 does
not apply an RMSE acceptance threshold: it establishes whether the reported
performance is leakage-resistant, while performance adequacy is a separate
obligation. A software `PASS` verifies the supplied provenance declarations and
the evaluator's enforced fit/evaluate separation; it cannot retroactively prove
that undeclared external experiments never accessed the test data.

D5 requires D1 and the D3 model declaration, but it does not require D2, D3, or
D4 to pass. D4 independently determines whether labels represent genuinely
independent runs.

## D6 implementation

D6 evaluates whether the held-out runs support sufficiently precise performance
claims and then separately reports whether the calibration meets the declared
requirements. It:

- requires D5 to return `PASS`;
- reuses the frozen development/test run partition and affine model;
- fits on development runs once and never refits during bootstrap;
- resamples complete held-out runs with replacement using a declared seed;
- calculates percentile confidence intervals for per-axis held-out RMSE;
- repeats the same run-level bootstrap within predeclared force-condition
  regions to expose local failure hidden by global metrics;
- compares interval half-widths with the declared precision limits; and
- compares confidence-interval upper bounds and calibrated-force uncertainty
  estimates with their separate acceptance limits.

The top-level D6 criterion status represents **dataset adequacy for performance
inference**. Calibration acceptance is reported independently in
`metrics.calibration_acceptance_status`. Consequently, a narrow confidence
interval showing that RMSE exceeds its limit produces a D6 dataset-adequacy
`PASS` and a calibration-acceptance `FAIL`.

At least two independent held-out runs are required by the D6 schema, and the
application may declare a larger minimum. The bootstrap repetition count and
random seed are recorded so the evidence is reproducible.

The initial evaluator does not derive calibrated-force uncertainty from raw
measurements. It requires a declared uncertainty-method identifier and
per-axis uncertainty estimates, then checks those estimates against the
application limits. A future uncertainty-model adapter can calculate these
quantities without changing the separation between evidence precision and
calibration acceptance.

## D7 implementation

D7 reads the task-declared evidence manifest and:

- verifies every declared file against its SHA-256 record;
- binds the dataset, task, instrument/setup profiles, and recorded criterion
  results used by the evaluator to their exact manifest entries and roles;
- checks that the task-required evidence roles are present;
- checks the D6 random seed against the task;
- reruns the required D0--D6 criteria; and
- compares recorded outcomes and numerical metrics using declared absolute and
  relative tolerances.

Missing provenance is `INDETERMINATE`. A missing file, hash mismatch, seed
mismatch, or non-reproduced result is `FAIL`.

## Overall assessment

`evaluate-all` retains every criterion report and aggregates only applicable
criteria. D0 incompleteness forces an overall `INDETERMINATE`. Otherwise,
`FAIL` takes precedence over `INDETERMINATE`, and `PASS` requires every
applicable criterion to pass. `NOT_APPLICABLE` criteria are excluded only when
the task contains a reason. D6 calibration acceptance is carried separately
from the overall dataset-adequacy status.
