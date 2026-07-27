from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

from ..models import (
    CriterionResult,
    CriterionStatus,
    TaskBundle,
    Violation,
)
from ._heldout_affine import evaluate_affine_holdout
from .d5_leakage_resistant_validation import evaluate_d5


def _context(
    bundle: TaskBundle,
    dataset_path: Path,
    d5_status: str,
) -> Dict[str, str]:
    return {
        "task_id": bundle.task.task_id,
        "dataset_path": str(dataset_path),
        "sensor_profile_id": bundle.sensor.instrument_id,
        "reference_profile_id": bundle.reference.instrument_id,
        "setup_profile_id": bundle.setup.setup_id,
        "d5_status": d5_status,
    }


def _indeterminate_result(
    bundle: TaskBundle,
    dataset_path: Path,
    summary: str,
    missing_evidence: List[str],
    *,
    d5_status: str,
    metrics: Optional[Dict[str, Any]] = None,
) -> CriterionResult:
    return CriterionResult(
        criterion_id="D6",
        criterion_name="Performance Evaluation and Uncertainty",
        status=CriterionStatus.INDETERMINATE,
        summary=summary,
        context=_context(bundle, dataset_path, d5_status),
        metrics=metrics or {},
        missing_evidence=list(dict.fromkeys(missing_evidence)),
        violations=[],
    )


def _missing_d6_evidence(bundle: TaskBundle) -> List[str]:
    requirements = bundle.task.d6
    if requirements is None:
        return ["task.d6"]

    missing: List[str] = []
    for field_name in (
        "confidence_level",
        "bootstrap_repetitions",
        "bootstrap_random_seed",
        "minimum_bootstrap_units",
        "uncertainty_method_id",
    ):
        if getattr(requirements, field_name) is None:
            missing.append(f"task.d6.{field_name}")

    output_channels = list(bundle.task.dataset_mapping.reference_channels)
    for channel in output_channels:
        axis = requirements.axes.get(channel)
        if axis is None:
            missing.append(f"task.d6.axes.{channel}")
            continue
        for field_name in (
            "maximum_interval_half_width",
            "maximum_rmse",
            "calibrated_force_uncertainty",
            "maximum_calibrated_force_uncertainty",
        ):
            if getattr(axis, field_name) is None:
                missing.append(f"task.d6.axes.{channel}.{field_name}")
    return missing


def evaluate_d6(
    dataset_path: Union[str, Path],
    bundle: TaskBundle,
) -> CriterionResult:
    """Evaluate run-bootstrap RMSE precision and calibration acceptance."""

    path = Path(dataset_path).expanduser().resolve()
    d5_result = evaluate_d5(path, bundle)
    if d5_result.status != CriterionStatus.PASS:
        return _indeterminate_result(
            bundle,
            path,
            "D6 was not evaluated because its D5 prerequisite did not pass.",
            ["prerequisite.D5_PASS"],
            d5_status=d5_result.status.value,
            metrics={
                "d5_status": d5_result.status.value,
                "d5_summary": d5_result.summary,
            },
        )

    requirements = bundle.task.d6
    if requirements is None:
        return _indeterminate_result(
            bundle,
            path,
            "D6 is indeterminate because no D6 task configuration was declared.",
            ["task.d6"],
            d5_status=d5_result.status.value,
        )

    missing_evidence = _missing_d6_evidence(bundle)
    if missing_evidence:
        return _indeterminate_result(
            bundle,
            path,
            f"D6 is indeterminate because {len(missing_evidence)} required "
            "declaration(s) are missing.",
            missing_evidence,
            d5_status=d5_result.status.value,
        )

    output_channels = list(bundle.task.dataset_mapping.reference_channels)
    configured_axes = set(requirements.axes)
    expected_axes = set(output_channels)
    if configured_axes != expected_axes:
        return CriterionResult(
            criterion_id="D6",
            criterion_name="Performance Evaluation and Uncertainty",
            status=CriterionStatus.FAIL,
            summary=(
                "D6 failed because the configured performance axes do not "
                "match the reference measurand."
            ),
            context=_context(bundle, path, d5_result.status.value),
            metrics={
                "expected_axes": output_channels,
                "configured_axes": sorted(configured_axes),
                "missing_axes": sorted(expected_axes - configured_axes),
                "unexpected_axes": sorted(configured_axes - expected_axes),
                "dataset_adequacy_status": CriterionStatus.FAIL.value,
                "calibration_acceptance_status": (
                    CriterionStatus.INDETERMINATE.value
                ),
                "total_violations": 1,
                "violations_by_code": {"performance_axis_mismatch": 1},
            },
            missing_evidence=[],
            violations=[
                Violation(
                    code="performance_axis_mismatch",
                    message=(
                        "D6 axes must exactly match the declared reference "
                        "channels"
                    ),
                    observed=sorted(configured_axes),
                    expected=str(output_channels),
                )
            ],
        )

    d5_requirements = bundle.task.d5
    development_units = list(d5_requirements.development_units or [])
    test_units = list(d5_requirements.test_units or [])
    minimum_bootstrap_units = int(requirements.minimum_bootstrap_units)
    test_unit_count = len(test_units)

    if test_unit_count < minimum_bootstrap_units:
        violation = Violation(
            code="insufficient_independent_test_units_for_bootstrap",
            message=(
                "the held-out partition contains fewer independent runs than "
                "the declared minimum for run-level bootstrap inference"
            ),
            observed=test_unit_count,
            expected=f">= {minimum_bootstrap_units}",
        )
        return CriterionResult(
            criterion_id="D6",
            criterion_name="Performance Evaluation and Uncertainty",
            status=CriterionStatus.FAIL,
            summary=(
                "D6 failed because there are too few independent held-out "
                "runs to support the declared bootstrap analysis."
            ),
            context=_context(bundle, path, d5_result.status.value),
            metrics={
                "metric": requirements.metric,
                "bootstrap_unit": requirements.bootstrap_unit,
                "test_unit_count": test_unit_count,
                "minimum_bootstrap_units": minimum_bootstrap_units,
                "dataset_adequacy_status": CriterionStatus.FAIL.value,
                "calibration_acceptance_status": (
                    CriterionStatus.INDETERMINATE.value
                ),
                "total_violations": 1,
                "violations_by_code": {violation.code: 1},
            },
            missing_evidence=[],
            violations=[violation],
        )

    evaluation = evaluate_affine_holdout(
        path,
        bundle,
        development_units,
        test_units,
    )
    errors_by_run = [
        evaluation.test_errors[
            evaluation.test_units_by_sample == test_unit
        ]
        for test_unit in test_units
    ]
    bootstrap_repetitions = int(requirements.bootstrap_repetitions)
    random_seed = int(requirements.bootstrap_random_seed)
    rng = np.random.default_rng(random_seed)
    bootstrap_rmse = np.empty(
        (bootstrap_repetitions, len(output_channels)),
        dtype=float,
    )
    for repetition in range(bootstrap_repetitions):
        sampled_run_indices = rng.integers(
            low=0,
            high=test_unit_count,
            size=test_unit_count,
        )
        sampled_errors = np.concatenate(
            [errors_by_run[index] for index in sampled_run_indices],
            axis=0,
        )
        bootstrap_rmse[repetition] = np.sqrt(
            np.mean(np.square(sampled_errors), axis=0)
        )

    confidence_level = float(requirements.confidence_level)
    alpha = 1.0 - confidence_level
    lower = np.quantile(bootstrap_rmse, alpha / 2.0, axis=0)
    upper = np.quantile(bootstrap_rmse, 1.0 - alpha / 2.0, axis=0)
    half_width = (upper - lower) / 2.0

    precision_violations: List[Violation] = []
    per_axis: Dict[str, Dict[str, Any]] = {}
    calibration_axes_accepted: List[bool] = []
    for index, channel in enumerate(output_channels):
        axis = requirements.axes[channel]
        maximum_half_width = float(axis.maximum_interval_half_width)
        maximum_rmse = float(axis.maximum_rmse)
        calibrated_uncertainty = float(axis.calibrated_force_uncertainty)
        maximum_calibrated_uncertainty = float(
            axis.maximum_calibrated_force_uncertainty
        )
        precision_pass = bool(half_width[index] <= maximum_half_width)
        rmse_acceptance_pass = bool(upper[index] <= maximum_rmse)
        uncertainty_acceptance_pass = bool(
            calibrated_uncertainty <= maximum_calibrated_uncertainty
        )
        calibration_accepted = (
            rmse_acceptance_pass and uncertainty_acceptance_pass
        )
        calibration_axes_accepted.append(calibration_accepted)
        per_axis[channel] = {
            "rmse_estimate": float(evaluation.test_rmse[index]),
            "confidence_interval": {
                "lower": float(lower[index]),
                "upper": float(upper[index]),
                "confidence_level": confidence_level,
            },
            "interval_half_width": float(half_width[index]),
            "maximum_interval_half_width": maximum_half_width,
            "performance_evidence_precise": precision_pass,
            "maximum_rmse": maximum_rmse,
            "rmse_requirement_met": rmse_acceptance_pass,
            "calibrated_force_uncertainty": calibrated_uncertainty,
            "maximum_calibrated_force_uncertainty": (
                maximum_calibrated_uncertainty
            ),
            "uncertainty_requirement_met": uncertainty_acceptance_pass,
            "calibration_axis_accepted": calibration_accepted,
        }
        if not precision_pass:
            precision_violations.append(
                Violation(
                    code="metric_interval_too_wide",
                    message=(
                        f"the {channel} RMSE confidence interval is wider "
                        "than the declared precision limit"
                    ),
                    field=f"axes.{channel}.maximum_interval_half_width",
                    observed=float(half_width[index]),
                    expected=f"<= {maximum_half_width}",
                )
            )

    dataset_status = (
        CriterionStatus.FAIL
        if precision_violations
        else CriterionStatus.PASS
    )
    calibration_status = (
        CriterionStatus.PASS
        if all(calibration_axes_accepted)
        else CriterionStatus.FAIL
    )
    violation_counts: Dict[str, int] = {}
    for violation in precision_violations:
        violation_counts[violation.code] = (
            violation_counts.get(violation.code, 0) + 1
        )

    metrics: Dict[str, Any] = {
        "d5_status": d5_result.status.value,
        "metric": requirements.metric,
        "bootstrap_unit": requirements.bootstrap_unit,
        "confidence_level": confidence_level,
        "bootstrap_repetitions": bootstrap_repetitions,
        "bootstrap_random_seed": random_seed,
        "bootstrap_quantile_method": "linear",
        "test_units": test_units,
        "test_unit_count": test_unit_count,
        "minimum_bootstrap_units": minimum_bootstrap_units,
        "test_samples_per_unit": {
            unit: int(len(errors))
            for unit, errors in zip(test_units, errors_by_run)
        },
        "test_sample_count": int(evaluation.test_errors.shape[0]),
        "uncertainty_method_id": requirements.uncertainty_method_id,
        "axes": per_axis,
        "dataset_adequacy_status": dataset_status.value,
        "calibration_acceptance_status": calibration_status.value,
        "total_violations": len(precision_violations),
        "violations_by_code": dict(sorted(violation_counts.items())),
    }

    if dataset_status == CriterionStatus.FAIL:
        summary = (
            "D6 failed: the held-out data do not estimate every required "
            "performance metric with the declared precision."
        )
    elif calibration_status == CriterionStatus.FAIL:
        summary = (
            "D6 passed for dataset adequacy: performance is estimated with "
            "the declared precision, but the calibration does not satisfy "
            "every performance and uncertainty requirement."
        )
    else:
        summary = (
            "D6 passed for dataset adequacy, and the calibration satisfies "
            "the declared performance and uncertainty requirements."
        )

    return CriterionResult(
        criterion_id="D6",
        criterion_name="Performance Evaluation and Uncertainty",
        status=dataset_status,
        summary=summary,
        context=_context(bundle, path, d5_result.status.value),
        metrics=metrics,
        missing_evidence=[],
        violations=precision_violations,
    )
