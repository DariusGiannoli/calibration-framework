# Calibration Dataset Adequacy Framework

This repository contains:

- `main.tex`: the working research definition of calibration adequacy;
- `software/`: an executable, configuration-driven implementation of the
  universal criteria;
- `software/configs/`: triaxial task, instrument, and setup templates;
- `software/examples/`: explicitly synthetic examples;
- `software/tests/`: automated criterion tests.

## Install the current software

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

## Run the tests

```bash
python -m unittest discover -s software/tests -v
```

## Evaluate the synthetic D1 example

```bash
calibration-adequacy evaluate-d1 \
  --task software/examples/d1/synthetic_demo/task.yaml \
  --dataset software/examples/d1/synthetic_demo/data.csv
```

Scientific `FAIL` and `INDETERMINATE` results use exit codes 1 and 2,
respectively. Configuration/schema errors use exit code 3.

## Generate schemas for a future GUI

```bash
calibration-adequacy write-schemas --output-dir reports/schemas
```

The generated JSON Schemas can drive GUI forms while YAML remains the
versioned, reproducible configuration format.

