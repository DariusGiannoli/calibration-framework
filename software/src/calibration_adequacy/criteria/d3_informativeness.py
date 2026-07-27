from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from ..models import CriterionResult, CriterionStatus, TaskBundle, Violation
from .d1_measurement_reference import evaluate_d1


def _context(bundle: TaskBundle, dataset_path: Path, d1_status: str) -> Dict[str, str]:
    return {
        "task_id": bundle.task.task_id,
        "dataset_path": str(dataset_path),
        "sensor_profile_id": bundle.sensor.instrument_id,
        "reference_profile_id": bundle.reference.instrument_id,
        "setup_profile_id": bundle.setup.setup_id,
        "d1_status": d1_status,
    }


def _indeterminate_result(
    bundle: TaskBundle,
    dataset_path: Path,
    summary: str,
    missing_evidence: List[str],
    *,
    d1_status: str,
    metrics: Optional[Dict[str, object]] = None,
) -> CriterionResult:
    return CriterionResult(
        criterion_id="D3",
        criterion_name="Informativeness and Identifiability",
        status=CriterionStatus.INDETERMINATE,
        summary=summary,
        context=_context(bundle, dataset_path, d1_status),
        metrics=metrics or {},
        missing_evidence=list(dict.fromkeys(missing_evidence)),
        violations=[],
    )


def _missing_d3_evidence(bundle: TaskBundle) -> List[str]:
    requirements = bundle.task.d3
    if requirements is None:
        return ["task.d3"]

    missing: List[str] = []
    for channel in requirements.input_channels:
        if channel not in bundle.task.dataset_mapping.sensor_channels:
            missing.append(f"task.dataset_mapping.sensor_channels.{channel}")
        normalization = requirements.normalization.get(channel)
        if normalization is None:
            missing.append(f"task.d3.normalization.{channel}")
            continue
        if normalization.minimum is None:
            missing.append(f"task.d3.normalization.{channel}.minimum")
        if normalization.maximum is None:
            missing.append(f"task.d3.normalization.{channel}.maximum")
    if requirements.relative_rank_tolerance is None:
        missing.append("task.d3.relative_rank_tolerance")
    if requirements.maximum_condition_number is None:
        missing.append("task.d3.maximum_condition_number")
    if requirements.model_specific_test_id is None:
        missing.append("task.d3.model_specific_test_id")
    if requirements.confounding_review_id is None:
        missing.append("task.d3.confounding_review_id")
    if requirements.condition_handling is None:
        missing.append("task.d3.condition_handling")
    return missing


def _load_inputs(
    dataset_path: Path,
    bundle: TaskBundle,
    channels: Sequence[str],
) -> np.ndarray:
    columns = [
        bundle.task.dataset_mapping.sensor_channels[channel]
        for channel in channels
    ]
    rows: List[Tuple[float, ...]] = []
    with dataset_path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        for row in reader:
            rows.append(tuple(float(row[column]) for column in columns))
    return np.asarray(rows, dtype=float)


def evaluate_d3(
    dataset_path: Union[str, Path],
    bundle: TaskBundle,
) -> CriterionResult:
    """Evaluate D3 for the model and normalization declared by the task."""

    path = Path(dataset_path).expanduser().resolve()
    d1_result = evaluate_d1(path, bundle)
    if d1_result.status != CriterionStatus.PASS:
        return _indeterminate_result(
            bundle,
            path,
            "D3 was not evaluated because its D1 prerequisite did not pass.",
            ["prerequisite.D1_PASS"],
            d1_status=d1_result.status.value,
            metrics={
                "d1_status": d1_result.status.value,
                "d1_summary": d1_result.summary,
            },
        )

    requirements = bundle.task.d3
    if requirements is None:
        return _indeterminate_result(
            bundle,
            path,
            "D3 is indeterminate because no D3 task configuration was declared.",
            ["task.d3"],
            d1_status=d1_result.status.value,
        )

    missing_evidence = _missing_d3_evidence(bundle)
    if missing_evidence:
        return _indeterminate_result(
            bundle,
            path,
            f"D3 is indeterminate because {len(missing_evidence)} required "
            "declaration(s) are missing.",
            missing_evidence,
            d1_status=d1_result.status.value,
        )

    output_dimension = requirements.output_dimension
    declared_reference_dimension = len(
        bundle.task.dataset_mapping.reference_channels
    )
    if output_dimension != declared_reference_dimension:
        return CriterionResult(
            criterion_id="D3",
            criterion_name="Informativeness and Identifiability",
            status=CriterionStatus.FAIL,
            summary=(
                "D3 failed because the affine-model output dimension does not "
                "match the declared reference measurand."
            ),
            context=_context(bundle, path, d1_result.status.value),
            metrics={
                "model_output_dimension": output_dimension,
                "reference_dimension": declared_reference_dimension,
                "total_violations": 1,
                "violations_by_code": {"output_dimension_mismatch": 1},
            },
            missing_evidence=[],
            violations=[
                Violation(
                    code="output_dimension_mismatch",
                    message=(
                        "model output dimension must match the number of "
                        "reference channels"
                    ),
                    observed=output_dimension,
                    expected=str(declared_reference_dimension),
                )
            ],
        )

    channels = requirements.input_channels
    raw_inputs = _load_inputs(path, bundle, channels)
    lower_bounds = np.asarray(
        [
            float(requirements.normalization[channel].minimum)
            for channel in channels
        ],
        dtype=float,
    )
    upper_bounds = np.asarray(
        [
            float(requirements.normalization[channel].maximum)
            for channel in channels
        ],
        dtype=float,
    )
    midpoints = (lower_bounds + upper_bounds) / 2.0
    half_ranges = (upper_bounds - lower_bounds) / 2.0
    normalized_inputs = (raw_inputs - midpoints) / half_ranges

    sample_count = normalized_inputs.shape[0]
    design_matrix = np.column_stack(
        (np.ones(sample_count, dtype=float), normalized_inputs)
    )
    feature_names = ["intercept", *channels]
    feature_count = design_matrix.shape[1]

    _, raw_singular_values, right_singular_vectors = np.linalg.svd(
        design_matrix,
        full_matrices=False,
    )
    singular_values = np.zeros(feature_count, dtype=float)
    singular_values[: len(raw_singular_values)] = raw_singular_values
    largest_singular_value = float(singular_values[0])
    relative_tolerance = float(requirements.relative_rank_tolerance)
    absolute_rank_tolerance = relative_tolerance * largest_singular_value
    design_rank = int(np.count_nonzero(singular_values > absolute_rank_tolerance))
    full_rank = design_rank == feature_count

    if full_rank:
        smallest_singular_value = float(singular_values[-1])
        condition_number = largest_singular_value / smallest_singular_value
    else:
        smallest_singular_value = float(singular_values[-1])
        condition_number = math.inf

    weakest_direction: Dict[str, float] = {}
    if sample_count >= feature_count and right_singular_vectors.size:
        vector = right_singular_vectors[-1]
        weakest_direction = {
            feature: float(value)
            for feature, value in zip(feature_names, vector)
        }

    total_parameter_count = output_dimension * feature_count
    sensitivity_rank = output_dimension * design_rank
    maximum_condition_number = float(requirements.maximum_condition_number)
    normalized_minimum = {
        channel: float(np.min(normalized_inputs[:, index]))
        for index, channel in enumerate(channels)
    }
    normalized_maximum = {
        channel: float(np.max(normalized_inputs[:, index]))
        for index, channel in enumerate(channels)
    }
    outside_normalization_count = int(
        np.count_nonzero(np.any(np.abs(normalized_inputs) > 1.0, axis=1))
    )

    violations: List[Violation] = []
    if not full_rank:
        violations.append(
            Violation(
                code="rank_deficient",
                message="the affine design matrix does not have full column rank",
                observed=design_rank,
                expected=str(feature_count),
            )
        )
    if condition_number > maximum_condition_number:
        violations.append(
            Violation(
                code="condition_number_exceeded",
                message="design-matrix condition number exceeds kappa_max",
                observed=(
                    "Infinity"
                    if math.isinf(condition_number)
                    else condition_number
                ),
                expected=f"<= {maximum_condition_number}",
            )
        )

    violation_counts: Dict[str, int] = {}
    for violation in violations:
        violation_counts[violation.code] = (
            violation_counts.get(violation.code, 0) + 1
        )

    metrics = {
        "d1_status": d1_result.status.value,
        "model_type": requirements.model_type,
        "input_channels": channels,
        "output_dimension": output_dimension,
        "samples_used": sample_count,
        "feature_names": feature_names,
        "feature_count": feature_count,
        "total_parameter_count": total_parameter_count,
        "design_matrix_rank": design_rank,
        "required_design_matrix_rank": feature_count,
        "sensitivity_matrix_rank": sensitivity_rank,
        "required_sensitivity_matrix_rank": total_parameter_count,
        "singular_values": [float(value) for value in singular_values],
        "largest_singular_value": largest_singular_value,
        "smallest_singular_value": smallest_singular_value,
        "relative_rank_tolerance": relative_tolerance,
        "absolute_rank_tolerance": absolute_rank_tolerance,
        "condition_number": (
            None if math.isinf(condition_number) else condition_number
        ),
        "condition_number_is_infinite": math.isinf(condition_number),
        "maximum_condition_number": maximum_condition_number,
        "model_specific_test_id": requirements.model_specific_test_id,
        "confounding_review_id": requirements.confounding_review_id,
        "condition_handling": requirements.condition_handling,
        "weakest_feature_direction": weakest_direction,
        "normalized_achieved_minimum": normalized_minimum,
        "normalized_achieved_maximum": normalized_maximum,
        "samples_outside_normalization_range": outside_normalization_count,
        "total_violations": len(violations),
        "violations_by_code": violation_counts,
    }

    if violations:
        return CriterionResult(
            criterion_id="D3",
            criterion_name="Informativeness and Identifiability",
            status=CriterionStatus.FAIL,
            summary=(
                "D3 failed because the declared model is not fully identifiable "
                "or is insufficiently conditioned."
            ),
            context=_context(bundle, path, d1_result.status.value),
            metrics=metrics,
            missing_evidence=[],
            violations=violations,
        )

    return CriterionResult(
        criterion_id="D3",
        criterion_name="Informativeness and Identifiability",
        status=CriterionStatus.PASS,
        summary=(
            "D3 passed: the declared model has full numerical rank and its "
            "condition number is within the declared limit."
        ),
        context=_context(bundle, path, d1_result.status.value),
        metrics=metrics,
        missing_evidence=[],
        violations=[],
    )
