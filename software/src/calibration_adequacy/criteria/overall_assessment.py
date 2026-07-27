from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, Union

from ..models import (
    CriterionResult,
    CriterionStatus,
    OverallAssessmentResult,
    TaskBundle,
)
from .d0_claim_completeness import evaluate_d0
from .d1_measurement_reference import evaluate_d1
from .d2_domain_coverage import evaluate_d2
from .d3_informativeness import evaluate_d3
from .d4_replication_dependence import evaluate_d4
from .d5_leakage_resistant_validation import evaluate_d5
from .d6_performance_uncertainty import evaluate_d6
from .d7_reproducibility_provenance import evaluate_d7


def evaluate_all(
    dataset_path: Union[str, Path],
    bundle: TaskBundle,
) -> OverallAssessmentResult:
    """Evaluate D0-D7 and aggregate only task-applicable dataset criteria."""

    dataset = Path(dataset_path).expanduser().resolve()
    results: Dict[str, CriterionResult] = {"D0": evaluate_d0(bundle)}
    missing = []
    assessment = bundle.task.assessment
    if assessment is None:
        missing.append("task.assessment")

    evaluators: Dict[str, Callable[[], CriterionResult]] = {
        "D1": lambda: evaluate_d1(dataset, bundle),
        "D2": lambda: evaluate_d2(dataset, bundle),
        "D3": lambda: evaluate_d3(dataset, bundle),
        "D4": lambda: evaluate_d4(dataset, bundle),
        "D5": lambda: evaluate_d5(dataset, bundle),
        "D6": lambda: evaluate_d6(dataset, bundle),
        "D7": lambda: evaluate_d7(dataset, bundle),
    }
    applicable_statuses = []
    for criterion_id, evaluator in evaluators.items():
        applicability = (
            assessment.criteria.get(criterion_id)
            if assessment is not None
            else None
        )
        if applicability is None:
            missing.append(f"task.assessment.criteria.{criterion_id}")
            result = evaluator()
        elif applicability.applicable:
            result = evaluator()
            applicable_statuses.append(result.status)
        else:
            result = CriterionResult(
                criterion_id=criterion_id,
                criterion_name="Not Applicable",
                status=CriterionStatus.NOT_APPLICABLE,
                summary=(
                    f"{criterion_id} is not applicable to this task: "
                    f"{applicability.reason}"
                ),
                context={"task_id": bundle.task.task_id},
                metrics={"reason": applicability.reason},
                missing_evidence=[],
                violations=[],
            )
        results[criterion_id] = result

    d0_status = results["D0"].status
    if d0_status == CriterionStatus.FAIL:
        overall_status = CriterionStatus.FAIL
        summary = "Overall assessment failed because D0 is contradictory."
    elif d0_status != CriterionStatus.PASS:
        overall_status = CriterionStatus.INDETERMINATE
        summary = (
            "Overall assessment is indeterminate because the calibration "
            "claim is incomplete."
        )
    elif missing:
        overall_status = CriterionStatus.INDETERMINATE
        summary = (
            "Overall assessment is indeterminate because criterion "
            "applicability is not fully declared."
        )
    elif CriterionStatus.FAIL in applicable_statuses:
        overall_status = CriterionStatus.FAIL
        summary = "Overall dataset assessment failed at least one criterion."
    elif CriterionStatus.INDETERMINATE in applicable_statuses:
        overall_status = CriterionStatus.INDETERMINATE
        summary = (
            "Overall dataset assessment is indeterminate because required "
            "evidence is missing."
        )
    else:
        overall_status = CriterionStatus.PASS
        summary = "Overall dataset assessment passed every applicable criterion."

    calibration_acceptance = None
    d6_acceptance = results["D6"].metrics.get(
        "calibration_acceptance_status"
    )
    if d6_acceptance is not None:
        calibration_acceptance = CriterionStatus(d6_acceptance)

    return OverallAssessmentResult(
        status=overall_status,
        summary=summary,
        task_id=bundle.task.task_id,
        criteria=results,
        calibration_acceptance_status=calibration_acceptance,
        missing_evidence=list(dict.fromkeys(missing)),
    )
