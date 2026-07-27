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
| Task profile | column mappings and criterion-specific application requirements | inferred hardware specifications or observed dataset extrema |

Unknown scientific values are represented explicitly as `null`. Missing values
produce an `INDETERMINATE` result; the engine never invents defaults.

## Criterion result semantics

- `PASS`: all required evidence is present and no violation was detected.
- `FAIL`: at least one observed or configured value violates a declared rule.
- `INDETERMINATE`: no violation is known, but required evidence is missing.

A known failure takes precedence over missing evidence. Configuration syntax or
schema errors are execution errors rather than scientific criterion results.

## D1 implementation

D1 checks:

- presence and numeric validity of required sensor/reference channels;
- timestamp ordering and sensor/reference synchronization;
- sensor and reference valid ranges;
- existence and validity of the reference-to-sensor rotation;
- actual reference uncertainty against task-specific maximum uncertainty;
- reference calibration evidence.

The initial implementation treats any invalid retained observation as a D1
failure. Filtering or excluding observations must happen through a documented
preprocessing step, after which D1 is run again on the retained dataset. This
avoids introducing an arbitrary allowable-invalid-sample percentage.

## D2 implementation

D2 uses the achieved reference measurements from a D1-valid dataset. It:

- requires D1 to return `PASS`;
- requires declared bounds and grid resolution for every domain axis;
- applies the configured reference-to-sensor rotation;
- normalizes each axis by its declared domain width;
- evaluates the nearest achieved point at every Cartesian grid point;
- reports the estimated fill distance and worst-covered domain point; and
- compares the estimate with the declared maximum fill distance.

D2 never infers the intended domain, grid resolution, or acceptance threshold
from the observed data. Missing declarations produce `INDETERMINATE`.
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
- reports the weakest feature direction to help diagnose confounding.

The first model adapter implements the affine calibration defined in the paper.
Additional polynomial, nonlinear, or dynamic adapters can be added without
changing D3 result semantics. D3 does not require D2 to pass: domain coverage
and model identifiability are separate criteria.

## D4 implementation

D4 evaluates independent replication and within-run temporal dependence. It:

- requires D1 to return `PASS`;
- identifies runs and trajectory configurations using declared CSV columns;
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
