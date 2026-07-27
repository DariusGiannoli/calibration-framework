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

## Evaluate the synthetic D2 example

```bash
calibration-adequacy evaluate-d2 \
  --task software/examples/d2/synthetic_demo/task.yaml \
  --dataset software/examples/d2/synthetic_demo/data.csv
```

## Evaluate the synthetic D3 example

```bash
calibration-adequacy evaluate-d3 \
  --task software/examples/d3/synthetic_demo/task.yaml \
  --dataset software/examples/d3/synthetic_demo/data.csv
```

## Evaluate the synthetic D4 example

```bash
calibration-adequacy evaluate-d4 \
  --task software/examples/d4/synthetic_demo/task.yaml \
  --dataset software/examples/d4/synthetic_demo/data.csv
```

## Evaluate the synthetic D5 example

```bash
calibration-adequacy evaluate-d5 \
  --task software/examples/d5/synthetic_demo/task.yaml \
  --dataset software/examples/d5/synthetic_demo/data.csv
```

## Evaluate the synthetic D6 example

```bash
calibration-adequacy evaluate-d6 \
  --task software/examples/d6/synthetic_demo/task.yaml \
  --dataset software/examples/d6/synthetic_demo/data.csv
```

D6 reports dataset adequacy for performance inference separately from
calibration acceptance. A precise held-out result may therefore pass D6 while
showing that the calibrated sensor fails a performance or uncertainty limit.

Scientific `FAIL` and `INDETERMINATE` results use exit codes 1 and 2,
respectively. Configuration/schema errors use exit code 3.

## Generate schemas for a future GUI

```bash
calibration-adequacy write-schemas --output-dir reports/schemas
```

The generated JSON Schemas can drive GUI forms while YAML remains the
versioned, reproducible configuration format.
