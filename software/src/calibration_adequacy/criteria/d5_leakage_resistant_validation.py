from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from ..models import (
    CriterionResult,
    CriterionStatus,
    D5Requirements,
    TaskBundle,
    Violation,
)
from ._heldout_affine import evaluate_affine_holdout, load_model_arrays
from .d1_measurement_reference import evaluate_d1

MAX_REPORTED_VIOLATIONS = 50


def _context(
    bundle: TaskBundle,
    dataset_path: Path,
    d1_status: str,
) -> Dict[str, str]:
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
    metrics: Optional[Dict[str, Any]] = None,
) -> CriterionResult:
    return CriterionResult(
        criterion_id="D5",
        criterion_name="Leakage-Resistant Validation",
        status=CriterionStatus.INDETERMINATE,
        summary=summary,
        context=_context(bundle, dataset_path, d1_status),
        metrics=metrics or {},
        missing_evidence=list(dict.fromkeys(missing_evidence)),
        violations=[],
    )


def _missing_d5_evidence(bundle: TaskBundle) -> List[str]:
    requirements = bundle.task.d5
    if requirements is None:
        return ["task.d5"]

    missing: List[str] = []
    if bundle.task.dataset_mapping.run_id_column is None:
        missing.append("task.dataset_mapping.run_id_column")
    if bundle.task.d3 is None:
        missing.append("task.d3")
    else:
        for channel in bundle.task.d3.input_channels:
            if channel not in bundle.task.dataset_mapping.sensor_channels:
                missing.append(
                    f"task.dataset_mapping.sensor_channels.{channel}"
                )

    if requirements.development_units is None:
        missing.append("task.d5.development_units")
    if requirements.test_units is None:
        missing.append("task.d5.test_units")
    if requirements.minimum_test_units is None:
        missing.append("task.d5.minimum_test_units")
    if requirements.split_manifest_id is None:
        missing.append("task.d5.split_manifest_id")
    if requirements.split_frozen_before_development is None:
        missing.append("task.d5.split_frozen_before_development")
    if requirements.development_selection_method_id is None:
        missing.append("task.d5.development_selection_method_id")

    if requirements.data_use is None:
        missing.append("task.d5.data_use")
    else:
        for field_name in (
            "data_dependent_preprocessing",
            "model_selection",
            "parameter_estimation",
            "performance_threshold_selection",
            "final_performance_evaluation",
            "model_locked_before_test_evaluation",
            "test_results_used_for_further_development",
        ):
            if getattr(requirements.data_use, field_name) is None:
                missing.append(f"task.d5.data_use.{field_name}")
    return missing


def _split_manifest_sha256(
    requirements: D5Requirements,
    development_units: List[str],
    test_units: List[str],
) -> str:
    canonical_manifest = {
        "validation_unit": requirements.validation_unit,
        "development_units": sorted(development_units),
        "test_units": sorted(test_units),
    }
    encoded = json.dumps(
        canonical_manifest,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def evaluate_d5(
    dataset_path: Union[str, Path],
    bundle: TaskBundle,
) -> CriterionResult:
    """Evaluate run-level leakage resistance and held-out affine performance."""

    path = Path(dataset_path).expanduser().resolve()
    d1_result = evaluate_d1(path, bundle)
    if d1_result.status != CriterionStatus.PASS:
        return _indeterminate_result(
            bundle,
            path,
            "D5 was not evaluated because its D1 prerequisite did not pass.",
            ["prerequisite.D1_PASS"],
            d1_status=d1_result.status.value,
            metrics={
                "d1_status": d1_result.status.value,
                "d1_summary": d1_result.summary,
            },
        )

    requirements = bundle.task.d5
    if requirements is None:
        return _indeterminate_result(
            bundle,
            path,
            "D5 is indeterminate because no D5 task configuration was declared.",
            ["task.d5"],
            d1_status=d1_result.status.value,
        )

    missing_evidence = _missing_d5_evidence(bundle)
    if missing_evidence:
        return _indeterminate_result(
            bundle,
            path,
            f"D5 is indeterminate because {len(missing_evidence)} required "
            "declaration(s) are missing.",
            missing_evidence,
            d1_status=d1_result.status.value,
        )

    d3_requirements = bundle.task.d3
    run_id_column = str(bundle.task.dataset_mapping.run_id_column)
    development_units = list(requirements.development_units or [])
    test_units = list(requirements.test_units or [])
    minimum_test_units = int(requirements.minimum_test_units)
    input_channels = list(d3_requirements.input_channels)
    output_channels = list(
        bundle.task.dataset_mapping.reference_channels
    )
    reference_dimension = len(output_channels)
    rotation_dimension = len(bundle.setup.reference_to_sensor_rotation or [])
    dimension_violations: List[Violation] = []
    dimension_violation_counts: Counter[str] = Counter()
    if d3_requirements.output_dimension != reference_dimension:
        dimension_violation_counts["model_output_dimension_mismatch"] += 1
        dimension_violations.append(
            Violation(
                code="model_output_dimension_mismatch",
                message=(
                    "the declared model output dimension does not match "
                    "the reference"
                ),
                observed=d3_requirements.output_dimension,
                expected=str(reference_dimension),
            )
        )
    if rotation_dimension != reference_dimension:
        dimension_violation_counts["reference_rotation_dimension_mismatch"] += 1
        dimension_violations.append(
            Violation(
                code="reference_rotation_dimension_mismatch",
                message=(
                    "reference channel count does not match the setup rotation"
                ),
                observed=reference_dimension,
                expected=str(rotation_dimension),
            )
        )
    if dimension_violations:
        return CriterionResult(
            criterion_id="D5",
            criterion_name="Leakage-Resistant Validation",
            status=CriterionStatus.FAIL,
            summary="D5 failed because the declared model dimensions disagree.",
            context=_context(bundle, path, d1_result.status.value),
            metrics={
                "model_output_dimension": d3_requirements.output_dimension,
                "reference_dimension": reference_dimension,
                "rotation_dimension": rotation_dimension,
                "total_violations": sum(dimension_violation_counts.values()),
                "violations_by_code": dict(dimension_violation_counts),
            },
            missing_evidence=[],
            violations=dimension_violations,
        )

    units, _, _ = load_model_arrays(
        path,
        bundle,
        run_id_column,
        input_channels,
        output_channels,
    )
    observed_units = sorted(set(units))
    observed_unit_set = set(observed_units)
    development_set = set(development_units)
    test_set = set(test_units)

    violations: List[Violation] = []
    violation_counts: Counter[str] = Counter()

    def add_violation(
        code: str,
        message: str,
        *,
        field: Optional[str] = None,
        observed: Optional[Any] = None,
        expected: Optional[str] = None,
    ) -> None:
        violation_counts[code] += 1
        if sum(violation_counts.values()) <= MAX_REPORTED_VIOLATIONS:
            violations.append(
                Violation(
                    code=code,
                    message=message,
                    field=field,
                    observed=observed,
                    expected=expected,
                )
            )

    if not development_units:
        add_violation(
            "empty_development_partition",
            "at least one development unit is required",
            field="development_units",
        )
    if not test_units:
        add_violation(
            "empty_test_partition",
            "at least one held-out test unit is required",
            field="test_units",
        )
    if len(development_units) != len(development_set):
        add_violation(
            "duplicate_development_unit",
            "development unit identifiers must be unique",
            field="development_units",
        )
    if len(test_units) != len(test_set):
        add_violation(
            "duplicate_test_unit",
            "test unit identifiers must be unique",
            field="test_units",
        )

    overlap = sorted(development_set & test_set)
    if overlap:
        add_violation(
            "development_test_unit_overlap",
            "development and test generalization units must be disjoint",
            observed=overlap,
            expected="empty intersection",
        )

    declared_units = development_set | test_set
    unassigned_units = sorted(observed_unit_set - declared_units)
    absent_declared_units = sorted(declared_units - observed_unit_set)
    if unassigned_units:
        add_violation(
            "observed_unit_not_partitioned",
            "every observed unit must be assigned to development or test",
            observed=unassigned_units,
            expected="complete partition",
        )
    if absent_declared_units:
        add_violation(
            "declared_unit_not_observed",
            "the split manifest contains units absent from the dataset",
            observed=absent_declared_units,
            expected="all declared units observed",
        )

    if len(test_set) < minimum_test_units:
        add_violation(
            "minimum_test_units_not_met",
            "held-out test-unit count is below the declared minimum",
            observed=len(test_set),
            expected=f">= {minimum_test_units}",
        )

    if not requirements.split_frozen_before_development:
        add_violation(
            "split_not_frozen_before_development",
            "the development/test split was not frozen before development",
            field="split_frozen_before_development",
            observed=requirements.split_frozen_before_development,
            expected="true",
        )

    data_use = requirements.data_use
    development_only_fields = (
        "data_dependent_preprocessing",
        "model_selection",
        "parameter_estimation",
        "performance_threshold_selection",
    )
    for field_name in development_only_fields:
        observed_scope = getattr(data_use, field_name)
        if observed_scope != "development_only":
            add_violation(
                "test_data_used_during_development",
                "a data-dependent development operation included test data",
                field=field_name,
                observed=observed_scope,
                expected="development_only",
            )
    if data_use.final_performance_evaluation != "test_only":
        add_violation(
            "final_evaluation_not_test_only",
            "final reported performance must be calculated on test data only",
            field="final_performance_evaluation",
            observed=data_use.final_performance_evaluation,
            expected="test_only",
        )
    if not data_use.model_locked_before_test_evaluation:
        add_violation(
            "model_not_locked_before_test",
            "the model was not locked before test-set evaluation",
            field="model_locked_before_test_evaluation",
            observed=data_use.model_locked_before_test_evaluation,
            expected="true",
        )
    if data_use.test_results_used_for_further_development:
        add_violation(
            "test_results_fed_back_into_development",
            "test results were used for further model development",
            field="test_results_used_for_further_development",
            observed=data_use.test_results_used_for_further_development,
            expected="false",
        )

    split_sha256 = _split_manifest_sha256(
        requirements,
        development_units,
        test_units,
    )
    metrics: Dict[str, Any] = {
        "d1_status": d1_result.status.value,
        "validation_unit": requirements.validation_unit,
        "split_manifest_id": requirements.split_manifest_id,
        "split_manifest_sha256": split_sha256,
        "observed_units": observed_units,
        "observed_unit_count": len(observed_units),
        "development_units": sorted(development_set),
        "development_unit_count": len(development_set),
        "test_units": sorted(test_set),
        "test_unit_count": len(test_set),
        "minimum_test_units": minimum_test_units,
        "overlapping_units": overlap,
        "unassigned_observed_units": unassigned_units,
        "absent_declared_units": absent_declared_units,
        "model_source": requirements.model_source,
        "development_selection_method_id": (
            requirements.development_selection_method_id
        ),
        "model_type": d3_requirements.model_type,
        "input_channels": input_channels,
        "output_channels": output_channels,
        "data_use": data_use.model_dump(mode="json"),
        "total_violations": sum(violation_counts.values()),
        "violations_by_code": dict(sorted(violation_counts.items())),
        "reported_violation_limit": MAX_REPORTED_VIOLATIONS,
    }

    if violations:
        return CriterionResult(
            criterion_id="D5",
            criterion_name="Leakage-Resistant Validation",
            status=CriterionStatus.FAIL,
            summary=(
                f"D5 failed with {sum(violation_counts.values())} known "
                "violation(s)."
            ),
            context=_context(bundle, path, d1_result.status.value),
            metrics=metrics,
            missing_evidence=[],
            violations=violations,
        )

    evaluation = evaluate_affine_holdout(
        path,
        bundle,
        development_units,
        test_units,
    )
    coefficient_matrix = evaluation.coefficients.T

    metrics.update(
        {
            "development_sample_count": int(
                evaluation.development_inputs.shape[0]
            ),
            "test_sample_count": int(evaluation.test_inputs.shape[0]),
            "development_design_rank": evaluation.design_rank,
            "development_design_required_rank": int(
                evaluation.development_design.shape[1]
            ),
            "development_design_singular_values": [
                float(value) for value in evaluation.singular_values
            ],
            "estimated_intercept": {
                channel: float(coefficient_matrix[index, 0])
                for index, channel in enumerate(output_channels)
            },
            "estimated_sensitivity_matrix": [
                [float(value) for value in row]
                for row in coefficient_matrix[:, 1:]
            ],
            "test_rmse": {
                channel: float(evaluation.test_rmse[index])
                for index, channel in enumerate(output_channels)
            },
            "performance_threshold_applied": False,
        }
    )

    return CriterionResult(
        criterion_id="D5",
        criterion_name="Leakage-Resistant Validation",
        status=CriterionStatus.PASS,
        summary=(
            "D5 passed: the frozen run-level split is disjoint, the test-use "
            "provenance is compliant, and held-out performance was evaluated "
            "without fitting on test runs."
        ),
        context=_context(bundle, path, d1_result.status.value),
        metrics=metrics,
        missing_evidence=[],
        violations=[],
    )
