from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np

from ..models import TaskBundle


@dataclass(frozen=True)
class HeldoutAffineEvaluation:
    units: np.ndarray
    output_channels: List[str]
    development_inputs: np.ndarray
    development_references: np.ndarray
    test_inputs: np.ndarray
    test_references: np.ndarray
    development_design: np.ndarray
    coefficients: np.ndarray
    design_rank: int
    singular_values: np.ndarray
    test_predictions: np.ndarray
    test_errors: np.ndarray
    test_rmse: np.ndarray
    test_units_by_sample: np.ndarray


def load_model_arrays(
    path: Path,
    bundle: TaskBundle,
    run_id_column: str,
    input_channels: Sequence[str],
    output_channels: Sequence[str],
) -> Tuple[List[str], np.ndarray, np.ndarray]:
    input_columns = [
        bundle.task.dataset_mapping.sensor_channels[channel]
        for channel in input_channels
    ]
    output_columns = [
        bundle.task.dataset_mapping.reference_channels[channel]
        for channel in output_channels
    ]
    units: List[str] = []
    inputs: List[Tuple[float, ...]] = []
    references: List[Tuple[float, ...]] = []

    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        for row in reader:
            units.append((row[run_id_column] or "").strip())
            inputs.append(tuple(float(row[column]) for column in input_columns))
            references.append(
                tuple(float(row[column]) for column in output_columns)
            )

    rotation = np.asarray(
        bundle.setup.reference_to_sensor_rotation,
        dtype=float,
    )
    reference_array = np.asarray(references, dtype=float)
    rotated_references = reference_array @ rotation.T
    return (
        units,
        np.asarray(inputs, dtype=float),
        rotated_references,
    )


def evaluate_affine_holdout(
    path: Path,
    bundle: TaskBundle,
    development_units: Sequence[str],
    test_units: Sequence[str],
) -> HeldoutAffineEvaluation:
    d3_requirements = bundle.task.d3
    if d3_requirements is None:
        raise ValueError("D3 requirements are needed for held-out evaluation")
    run_id_column = bundle.task.dataset_mapping.run_id_column
    if run_id_column is None:
        raise ValueError("a run ID column is needed for held-out evaluation")

    input_channels = list(d3_requirements.input_channels)
    output_channels = list(bundle.task.dataset_mapping.reference_channels)
    units, sensor_inputs, reference_outputs = load_model_arrays(
        path,
        bundle,
        run_id_column,
        input_channels,
        output_channels,
    )
    unit_array = np.asarray(units, dtype=object)
    development_mask = np.isin(unit_array, list(development_units))
    test_mask = np.isin(unit_array, list(test_units))
    development_inputs = sensor_inputs[development_mask]
    development_references = reference_outputs[development_mask]
    test_inputs = sensor_inputs[test_mask]
    test_references = reference_outputs[test_mask]

    development_design = np.column_stack(
        (
            np.ones(development_inputs.shape[0], dtype=float),
            development_inputs,
        )
    )
    coefficients, _, design_rank, singular_values = np.linalg.lstsq(
        development_design,
        development_references,
        rcond=None,
    )
    test_design = np.column_stack(
        (
            np.ones(test_inputs.shape[0], dtype=float),
            test_inputs,
        )
    )
    test_predictions = test_design @ coefficients
    test_errors = test_predictions - test_references
    test_rmse = np.sqrt(np.mean(np.square(test_errors), axis=0))

    return HeldoutAffineEvaluation(
        units=unit_array,
        output_channels=output_channels,
        development_inputs=development_inputs,
        development_references=development_references,
        test_inputs=test_inputs,
        test_references=test_references,
        development_design=development_design,
        coefficients=coefficients,
        design_rank=int(design_rank),
        singular_values=singular_values,
        test_predictions=test_predictions,
        test_errors=test_errors,
        test_rmse=test_rmse,
        test_units_by_sample=unit_array[test_mask],
    )
