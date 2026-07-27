from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..models import (
    CriterionResult,
    CriterionStatus,
    TaskBundle,
    Violation,
)


def evaluate_d0(bundle: TaskBundle) -> CriterionResult:
    """Evaluate whether every component of the calibration claim is explicit."""

    task = bundle.task
    claim = task.claim
    missing: List[str] = []
    violations: List[Violation] = []

    if claim is None:
        missing.append("task.claim")
    else:
        for field_name in (
            "sensor_inputs",
            "reference_outputs",
            "operating_domain",
            "operating_conditions",
            "model_family",
            "performance_metrics",
            "generalization",
        ):
            if getattr(claim, field_name) is None:
                missing.append(f"task.claim.{field_name}")

    if claim is not None and claim.sensor_inputs is not None:
        _compare_declared_set(
            "sensor_input_mismatch",
            "claim sensor inputs",
            claim.sensor_inputs,
            list(task.dataset_mapping.sensor_channels),
            violations,
        )
    if claim is not None and claim.reference_outputs is not None:
        _compare_declared_set(
            "reference_output_mismatch",
            "claim reference outputs",
            claim.reference_outputs,
            list(task.dataset_mapping.reference_channels),
            violations,
        )

    if task.d2 is None:
        missing.append("task.d2")
    else:
        for axis in task.d2.axes:
            domain = task.d2.domain.get(axis)
            if domain is None:
                missing.append(f"task.d2.domain.{axis}")
                continue
            for field_name in ("minimum", "maximum", "grid_points"):
                if getattr(domain, field_name) is None:
                    missing.append(f"task.d2.domain.{axis}.{field_name}")
        if task.d2.maximum_fill_distance is None:
            missing.append("task.d2.maximum_fill_distance")
        if task.d2.conditions is None:
            missing.append("task.d2.conditions")
        else:
            for name, condition in task.d2.conditions.items():
                prefix = f"task.d2.conditions.{name}"
                if condition.kind == "continuous":
                    for field_name in (
                        "unit",
                        "minimum",
                        "maximum",
                        "grid_points",
                    ):
                        if getattr(condition, field_name) is None:
                            missing.append(f"{prefix}.{field_name}")
                elif (
                    condition.kind == "categorical"
                    and condition.categories is None
                ):
                    missing.append(f"{prefix}.categories")
                elif (
                    condition.kind == "fixed"
                    and condition.claimed_value is None
                ):
                    missing.append(f"{prefix}.claimed_value")
        if task.d2.excluded_regions is None:
            missing.append("task.d2.excluded_regions")
        if claim is not None and claim.operating_domain is not None:
            _compare_declared_set(
                "operating_domain_mismatch",
                "claim operating-domain axes",
                claim.operating_domain,
                task.d2.axes,
                violations,
            )
        if (
            claim is not None
            and claim.operating_conditions is not None
            and task.d2.conditions is not None
        ):
            _compare_declared_set(
                "operating_condition_mismatch",
                "claim operating conditions",
                claim.operating_conditions,
                list(task.d2.conditions),
                violations,
            )

    if task.d3 is None:
        missing.append("task.d3")
    else:
        for field_name in (
            "model_specific_test_id",
            "confounding_review_id",
            "condition_handling",
        ):
            if getattr(task.d3, field_name) is None:
                missing.append(f"task.d3.{field_name}")
        if (
            claim is not None
            and claim.model_family is not None
            and claim.model_family != task.d3.model_type
        ):
            violations.append(
                Violation(
                    code="model_family_mismatch",
                    message="claim model family disagrees with D3",
                    observed=claim.model_family,
                    expected=task.d3.model_type,
                )
            )

    if task.d6 is None:
        missing.append("task.d6")
    else:
        if claim is not None and claim.performance_metrics is not None:
            required_metrics = {
                task.d6.metric,
                "calibrated_force_uncertainty",
            }
            if not required_metrics.issubset(set(claim.performance_metrics)):
                violations.append(
                    Violation(
                        code="performance_requirement_mismatch",
                        message=(
                            "claim performance metrics do not contain every "
                            "D6 requirement"
                        ),
                        observed=claim.performance_metrics,
                        expected=str(sorted(required_metrics)),
                    )
                )
        for field_name in (
            "confidence_level",
            "bootstrap_repetitions",
            "bootstrap_random_seed",
            "minimum_bootstrap_units",
            "uncertainty_method_id",
            "regions",
        ):
            if getattr(task.d6, field_name) is None:
                missing.append(f"task.d6.{field_name}")
        for axis in task.dataset_mapping.reference_channels:
            requirement = task.d6.axes.get(axis)
            if requirement is None:
                missing.append(f"task.d6.axes.{axis}")
                continue
            for field_name in (
                "maximum_interval_half_width",
                "maximum_rmse",
                "calibrated_force_uncertainty",
                "maximum_calibrated_force_uncertainty",
            ):
                if getattr(requirement, field_name) is None:
                    missing.append(f"task.d6.axes.{axis}.{field_name}")

    generalization = claim.generalization if claim is not None else None
    if generalization is not None:
        for field_name in (
            "target_description",
            "independent_unit",
            "sensor_scope",
            "operating_conditions",
        ):
            if getattr(generalization, field_name) is None:
                missing.append(f"task.claim.generalization.{field_name}")
    if task.d4 is None:
        missing.append("task.d4")
    if task.d5 is None:
        missing.append("task.d5")
    if (
        generalization is not None
        and generalization.independent_unit is not None
    ):
        if (
            task.d4 is not None
            and generalization.independent_unit != task.d4.independent_unit
        ):
            violations.append(
                Violation(
                    code="d4_generalization_unit_mismatch",
                    message="D4 independent unit disagrees with the claim",
                    observed=task.d4.independent_unit,
                    expected=generalization.independent_unit,
                )
            )
        if (
            task.d5 is not None
            and generalization.independent_unit != task.d5.validation_unit
        ):
            violations.append(
                Violation(
                    code="d5_generalization_unit_mismatch",
                    message="D5 validation unit disagrees with the claim",
                    observed=task.d5.validation_unit,
                    expected=generalization.independent_unit,
                )
            )

    unique_missing = list(dict.fromkeys(missing))
    if violations:
        status = CriterionStatus.FAIL
        summary = (
            f"D0 failed with {len(violations)} contradictory claim "
            "declaration(s)."
        )
    elif unique_missing:
        status = CriterionStatus.INDETERMINATE
        summary = (
            f"D0 is indeterminate because {len(unique_missing)} calibration "
            "claim declaration(s) are missing."
        )
    else:
        status = CriterionStatus.PASS
        summary = (
            "D0 passed: every calibration-task component is explicitly "
            "declared and internally consistent."
        )

    counts: Dict[str, int] = {}
    for violation in violations:
        counts[violation.code] = counts.get(violation.code, 0) + 1
    return CriterionResult(
        criterion_id="D0",
        criterion_name="Calibration Claim Completeness",
        status=status,
        summary=summary,
        context={
            "task_id": task.task_id,
            "sensor_profile_id": bundle.sensor.instrument_id,
            "reference_profile_id": bundle.reference.instrument_id,
            "setup_profile_id": bundle.setup.setup_id,
        },
        metrics={
            "claim": claim.model_dump(mode="json") if claim else None,
            "total_violations": len(violations),
            "violations_by_code": counts,
        },
        missing_evidence=unique_missing,
        violations=violations,
    )


def _compare_declared_set(
    code: str,
    label: str,
    declared: List[str],
    configured: List[str],
    violations: List[Violation],
) -> None:
    if len(declared) != len(set(declared)) or set(declared) != set(configured):
        violations.append(
            Violation(
                code=code,
                message=f"{label} disagree with the executable task mapping",
                observed=declared,
                expected=str(configured),
            )
        )
