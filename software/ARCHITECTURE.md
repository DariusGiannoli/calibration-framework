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
| Task profile | column mappings and maximum acceptable reference uncertainty | inferred hardware specifications |

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

