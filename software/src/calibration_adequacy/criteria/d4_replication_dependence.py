from __future__ import annotations

import csv
import math
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from ..models import (
    CriterionResult,
    CriterionStatus,
    D4Requirements,
    TaskBundle,
    Violation,
)
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
        criterion_id="D4",
        criterion_name="Independent Replication and Dependence",
        status=CriterionStatus.INDETERMINATE,
        summary=summary,
        context=_context(bundle, dataset_path, d1_status),
        metrics=metrics or {},
        missing_evidence=list(dict.fromkeys(missing_evidence)),
        violations=[],
    )


def _signal_key(source: str, channel: str) -> str:
    return f"{source}.{channel}"


def _missing_d4_evidence(bundle: TaskBundle) -> List[str]:
    requirements = bundle.task.d4
    if requirements is None:
        return ["task.d4"]

    missing: List[str] = []
    mapping = bundle.task.dataset_mapping
    if mapping.run_id_column is None:
        missing.append("task.dataset_mapping.run_id_column")
    if requirements.minimum_runs_per_configuration is None:
        missing.append("task.d4.minimum_runs_per_configuration")
    elif (
        requirements.minimum_runs_per_configuration
        and mapping.configuration_id_column is None
    ):
        missing.append("task.dataset_mapping.configuration_id_column")

    if requirements.minimum_independent_runs is None:
        missing.append("task.d4.minimum_independent_runs")
    if requirements.minimum_effective_sample_size is None:
        missing.append("task.d4.minimum_effective_sample_size")
    if requirements.initialization_procedure_id is None:
        missing.append("task.d4.initialization_procedure_id")
    if requirements.zeroing_procedure_id is None:
        missing.append("task.d4.zeroing_procedure_id")

    for signal in requirements.signals:
        key = _signal_key(signal.source, signal.channel)
        channel_mapping = (
            mapping.sensor_channels
            if signal.source == "sensor"
            else mapping.reference_channels
        )
        if signal.channel not in channel_mapping:
            missing.append(
                f"task.dataset_mapping.{signal.source}_channels.{signal.channel}"
            )
        if signal.maximum_autocorrelation_lag is None:
            missing.append(
                f"task.d4.signals.{key}.maximum_autocorrelation_lag"
            )

    if not requirements.run_evidence:
        missing.append("task.d4.run_evidence")
    else:
        for evidence in requirements.run_evidence:
            prefix = f"task.d4.run_evidence.{evidence.run_id}"
            if evidence.acquisition_id is None:
                missing.append(f"{prefix}.acquisition_id")
            if evidence.separate_acquisition is None:
                missing.append(f"{prefix}.separate_acquisition")
            if evidence.initialization_completed is None:
                missing.append(f"{prefix}.initialization_completed")
            if evidence.zeroing_completed is None:
                missing.append(f"{prefix}.zeroing_completed")
    return missing


def _resolve_signals(
    bundle: TaskBundle,
    requirements: D4Requirements,
) -> List[Tuple[str, str, int]]:
    resolved: List[Tuple[str, str, int]] = []
    mapping = bundle.task.dataset_mapping
    for signal in requirements.signals:
        channel_mapping = (
            mapping.sensor_channels
            if signal.source == "sensor"
            else mapping.reference_channels
        )
        resolved.append(
            (
                _signal_key(signal.source, signal.channel),
                channel_mapping[signal.channel],
                int(signal.maximum_autocorrelation_lag),
            )
        )
    return resolved


def _autocorrelations(
    values: Sequence[float],
    maximum_lag: int,
) -> Tuple[List[float], float]:
    samples = np.asarray(values, dtype=float)
    centered = samples - np.mean(samples)
    variance_sum = float(np.dot(centered, centered))
    if variance_sum <= np.finfo(float).eps:
        raise ValueError("constant_signal")

    correlations = [
        float(np.dot(centered[:-lag], centered[lag:]) / variance_sum)
        for lag in range(1, maximum_lag + 1)
    ]
    denominator = 1.0 + 2.0 * sum(correlations)
    if not math.isfinite(denominator) or denominator <= 0:
        raise ValueError("nonpositive_denominator")
    return correlations, denominator


def evaluate_d4(
    dataset_path: Union[str, Path],
    bundle: TaskBundle,
) -> CriterionResult:
    """Evaluate D4 independent replication and fixed-lag effective sample size."""

    path = Path(dataset_path).expanduser().resolve()
    d1_result = evaluate_d1(path, bundle)
    if d1_result.status != CriterionStatus.PASS:
        return _indeterminate_result(
            bundle,
            path,
            "D4 was not evaluated because its D1 prerequisite did not pass.",
            ["prerequisite.D1_PASS"],
            d1_status=d1_result.status.value,
            metrics={
                "d1_status": d1_result.status.value,
                "d1_summary": d1_result.summary,
            },
        )

    requirements = bundle.task.d4
    if requirements is None:
        return _indeterminate_result(
            bundle,
            path,
            "D4 is indeterminate because no D4 task configuration was declared.",
            ["task.d4"],
            d1_status=d1_result.status.value,
        )

    missing_evidence = _missing_d4_evidence(bundle)
    if missing_evidence:
        return _indeterminate_result(
            bundle,
            path,
            f"D4 is indeterminate because {len(missing_evidence)} required "
            "declaration(s) are missing.",
            missing_evidence,
            d1_status=d1_result.status.value,
        )

    mapping = bundle.task.dataset_mapping
    run_id_column = str(mapping.run_id_column)
    configuration_requirements = dict(
        requirements.minimum_runs_per_configuration or {}
    )
    configuration_id_column = mapping.configuration_id_column
    resolved_signals = _resolve_signals(bundle, requirements)

    violations: List[Violation] = []
    violation_counts: Counter[str] = Counter()

    def add_violation(
        code: str,
        message: str,
        *,
        row_number: Optional[int] = None,
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
                    row_number=row_number,
                    field=field,
                    observed=observed,
                    expected=expected,
                )
            )

    required_columns = [
        run_id_column,
        *(column for _, column, _ in resolved_signals),
    ]
    if configuration_requirements and configuration_id_column is not None:
        required_columns.append(configuration_id_column)

    run_values: Dict[str, Dict[str, List[float]]] = {}
    run_configurations: Dict[str, str] = {}
    rows_evaluated = 0

    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        fieldnames = reader.fieldnames or []
        for column in dict.fromkeys(required_columns):
            if column not in fieldnames:
                add_violation(
                    "missing_required_column",
                    f"required D4 dataset column {column!r} is missing",
                    field=column,
                )

        if not violation_counts["missing_required_column"]:
            for row_number, row in enumerate(reader, start=2):
                rows_evaluated += 1
                run_id = (row.get(run_id_column) or "").strip()
                if not run_id:
                    add_violation(
                        "invalid_run_id",
                        "run identifier is missing or empty",
                        row_number=row_number,
                        field=run_id_column,
                        observed=row.get(run_id_column),
                    )
                    continue

                configuration_id = "__all__"
                if (
                    configuration_requirements
                    and configuration_id_column is not None
                ):
                    configuration_id = (
                        row.get(configuration_id_column) or ""
                    ).strip()
                    if not configuration_id:
                        add_violation(
                            "invalid_configuration_id",
                            "trajectory configuration identifier is missing "
                            "or empty",
                            row_number=row_number,
                            field=configuration_id_column,
                            observed=row.get(configuration_id_column),
                        )
                        continue

                previous_configuration = run_configurations.get(run_id)
                if (
                    previous_configuration is not None
                    and previous_configuration != configuration_id
                ):
                    add_violation(
                        "multiple_configurations_per_run",
                        "one run identifier is assigned to multiple configurations",
                        row_number=row_number,
                        field=configuration_id_column,
                        observed=configuration_id,
                        expected=previous_configuration,
                    )
                    continue
                run_configurations[run_id] = configuration_id

                values_by_signal = run_values.setdefault(
                    run_id,
                    {key: [] for key, _, _ in resolved_signals},
                )
                for key, column, _ in resolved_signals:
                    values_by_signal[key].append(float(row[column]))

    observed_run_ids = sorted(run_values)
    evidence_by_run = {
        evidence.run_id: evidence
        for evidence in requirements.run_evidence or []
    }

    for run_id in observed_run_ids:
        if run_id not in evidence_by_run:
            missing_evidence.append(f"task.d4.run_evidence.{run_id}")
    unknown_run_ids = [
        run_id for run_id in observed_run_ids if run_id not in evidence_by_run
    ]
    for run_id in sorted(set(evidence_by_run) - set(observed_run_ids)):
        add_violation(
            "declared_run_not_observed",
            "run evidence was declared for a run absent from the dataset",
            field=run_id_column,
            observed=run_id,
        )

    observed_acquisition_ids: Dict[str, str] = {}
    verified_run_ids: List[str] = []
    for run_id in observed_run_ids:
        evidence = evidence_by_run.get(run_id)
        if evidence is None:
            continue

        acquisition_id = str(evidence.acquisition_id)
        previous_run = observed_acquisition_ids.get(acquisition_id)
        if previous_run is not None:
            add_violation(
                "duplicate_acquisition_id",
                "multiple run identifiers refer to the same acquisition",
                field="acquisition_id",
                observed=acquisition_id,
                expected=f"unique; already used by {previous_run}",
            )
        else:
            observed_acquisition_ids[acquisition_id] = run_id

        if not evidence.separate_acquisition:
            add_violation(
                "separate_acquisition_not_established",
                "the run is not declared as a separately started acquisition",
                field=run_id,
                observed=evidence.separate_acquisition,
                expected="true",
            )
        if not evidence.initialization_completed:
            add_violation(
                "initialization_not_completed",
                "the declared initialization procedure was not completed",
                field=run_id,
                observed=evidence.initialization_completed,
                expected="true",
            )
        if not evidence.zeroing_completed:
            add_violation(
                "zeroing_not_completed",
                "the declared zeroing procedure was not completed",
                field=run_id,
                observed=evidence.zeroing_completed,
                expected="true",
            )

        if (
            evidence.separate_acquisition
            and evidence.initialization_completed
            and evidence.zeroing_completed
            and previous_run is None
        ):
            verified_run_ids.append(run_id)

    minimum_independent_runs = int(requirements.minimum_independent_runs)
    maximum_possible_independent_runs = (
        len(verified_run_ids) + len(unknown_run_ids)
    )
    if maximum_possible_independent_runs < minimum_independent_runs:
        add_violation(
            "minimum_independent_runs_not_met",
            "the maximum possible independent-run count is below the "
            "declared minimum",
            observed=maximum_possible_independent_runs,
            expected=f">= {minimum_independent_runs}",
        )

    configuration_run_counts: Counter[str] = Counter(
        run_configurations[run_id]
        for run_id in verified_run_ids
        if run_id in run_configurations
    )
    unknown_configuration_run_counts: Counter[str] = Counter(
        run_configurations[run_id]
        for run_id in unknown_run_ids
        if run_id in run_configurations
    )
    for configuration_id, required_count in configuration_requirements.items():
        maximum_possible_count = (
            configuration_run_counts[configuration_id]
            + unknown_configuration_run_counts[configuration_id]
        )
        if maximum_possible_count < required_count:
            add_violation(
                "minimum_configuration_runs_not_met",
                "independent repetitions for a configuration are below the "
                "declared minimum",
                field=configuration_id,
                observed=maximum_possible_count,
                expected=f">= {required_count}",
            )

    signal_metrics: Dict[str, Any] = {}
    effective_sample_sizes: Dict[str, float] = {}
    for signal_key, _, maximum_lag in resolved_signals:
        per_run: Dict[str, Any] = {}
        total_effective_sample_size = 0.0
        signal_computable = True

        for run_id in verified_run_ids:
            values = run_values[run_id][signal_key]
            sample_count = len(values)
            if sample_count <= maximum_lag:
                add_violation(
                    "insufficient_samples_for_autocorrelation_lag",
                    "run length must exceed the declared maximum lag",
                    field=f"{run_id}.{signal_key}",
                    observed=sample_count,
                    expected=f"> {maximum_lag}",
                )
                signal_computable = False
                continue

            try:
                correlations, denominator = _autocorrelations(
                    values,
                    maximum_lag,
                )
            except ValueError as exc:
                code = str(exc)
                if code == "constant_signal":
                    violation_code = "autocorrelation_undefined_constant_signal"
                    message = (
                        "autocorrelation is undefined for a constant run signal"
                    )
                else:
                    violation_code = "invalid_effective_sample_size_denominator"
                    message = (
                        "the fixed-lag autocorrelation sum gives a non-positive "
                        "effective-sample-size denominator"
                    )
                add_violation(
                    violation_code,
                    message,
                    field=f"{run_id}.{signal_key}",
                )
                signal_computable = False
                continue

            effective_sample_size = sample_count / denominator
            total_effective_sample_size += effective_sample_size
            per_run[run_id] = {
                "sample_count": sample_count,
                "maximum_autocorrelation_lag": maximum_lag,
                "autocorrelations": correlations,
                "effective_sample_size_denominator": denominator,
                "effective_sample_size": effective_sample_size,
            }

        signal_metrics[signal_key] = {
            "maximum_autocorrelation_lag": maximum_lag,
            "per_run": per_run,
            "effective_sample_size": (
                total_effective_sample_size
                if signal_computable and not unknown_run_ids
                else None
            ),
            "verified_runs_partial_effective_sample_size": (
                total_effective_sample_size if signal_computable else None
            ),
        }
        if signal_computable and not unknown_run_ids:
            effective_sample_sizes[signal_key] = total_effective_sample_size

    minimum_effective_sample_size = float(
        requirements.minimum_effective_sample_size
    )
    observed_minimum_effective_sample_size: Optional[float] = None
    limiting_signal: Optional[str] = None
    if len(effective_sample_sizes) == len(resolved_signals):
        limiting_signal = min(
            effective_sample_sizes,
            key=effective_sample_sizes.__getitem__,
        )
        observed_minimum_effective_sample_size = effective_sample_sizes[
            limiting_signal
        ]
        if observed_minimum_effective_sample_size < minimum_effective_sample_size:
            add_violation(
                "minimum_effective_sample_size_not_met",
                "the limiting signal effective sample size is below the "
                "declared minimum",
                field=limiting_signal,
                observed=observed_minimum_effective_sample_size,
                expected=f">= {minimum_effective_sample_size}",
            )

    metrics = {
        "d1_status": d1_result.status.value,
        "rows_evaluated": rows_evaluated,
        "observed_run_ids": observed_run_ids,
        "observed_run_count": len(observed_run_ids),
        "verified_independent_run_ids": verified_run_ids,
        "verified_independent_run_count": len(verified_run_ids),
        "runs_with_missing_independence_evidence": unknown_run_ids,
        "maximum_possible_independent_run_count": (
            maximum_possible_independent_runs
        ),
        "minimum_independent_runs": minimum_independent_runs,
        "configuration_run_counts": dict(sorted(configuration_run_counts.items())),
        "minimum_runs_per_configuration": configuration_requirements,
        "initialization_procedure_id": requirements.initialization_procedure_id,
        "zeroing_procedure_id": requirements.zeroing_procedure_id,
        "signals": signal_metrics,
        "effective_sample_sizes": effective_sample_sizes,
        "minimum_effective_sample_size_observed": (
            observed_minimum_effective_sample_size
        ),
        "limiting_signal": limiting_signal,
        "minimum_effective_sample_size_required": minimum_effective_sample_size,
        "total_violations": sum(violation_counts.values()),
        "violations_by_code": dict(sorted(violation_counts.items())),
        "reported_violation_limit": MAX_REPORTED_VIOLATIONS,
    }

    if violations:
        return CriterionResult(
            criterion_id="D4",
            criterion_name="Independent Replication and Dependence",
            status=CriterionStatus.FAIL,
            summary=(
                f"D4 failed with {sum(violation_counts.values())} known "
                "violation(s)."
            ),
            context=_context(bundle, path, d1_result.status.value),
            metrics=metrics,
            missing_evidence=list(dict.fromkeys(missing_evidence)),
            violations=violations,
        )

    if missing_evidence:
        return _indeterminate_result(
            bundle,
            path,
            f"D4 is indeterminate because {len(set(missing_evidence))} "
            "required evidence item(s) are missing.",
            missing_evidence,
            d1_status=d1_result.status.value,
            metrics=metrics,
        )

    return CriterionResult(
        criterion_id="D4",
        criterion_name="Independent Replication and Dependence",
        status=CriterionStatus.PASS,
        summary=(
            "D4 passed: independent-run, configuration-repetition, and "
            "effective-sample-size requirements are satisfied."
        ),
        context=_context(bundle, path, d1_result.status.value),
        metrics=metrics,
        missing_evidence=[],
        violations=[],
    )
