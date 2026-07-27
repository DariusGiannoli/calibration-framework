from __future__ import annotations

import csv
import math
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

from ..models import (
    ChannelProfile,
    CriterionResult,
    CriterionStatus,
    TaskBundle,
    Violation,
)

MAX_REPORTED_VIOLATIONS = 50
ROTATION_TOLERANCE = 1e-6


def _dot(left: Sequence[float], right: Sequence[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


def _determinant_3x3(matrix: Sequence[Sequence[float]]) -> float:
    return (
        matrix[0][0]
        * (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1])
        - matrix[0][1]
        * (matrix[1][0] * matrix[2][2] - matrix[1][2] * matrix[2][0])
        + matrix[0][2]
        * (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0])
    )


def _rotation_is_valid(matrix: Sequence[Sequence[float]]) -> bool:
    for row_index, row in enumerate(matrix):
        for other_index, other in enumerate(matrix):
            expected = 1.0 if row_index == other_index else 0.0
            if abs(_dot(row, other) - expected) > ROTATION_TOLERANCE:
                return False
    return abs(_determinant_3x3(matrix) - 1.0) <= ROTATION_TOLERANCE


def _parse_finite(value: Any) -> Optional[float]:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _missing_channel_evidence(
    profile_name: str,
    channel_name: str,
    profile: ChannelProfile,
    missing: List[str],
) -> None:
    if not profile.unit:
        missing.append(f"{profile_name}.{channel_name}.unit")
    if profile.valid_min is None:
        missing.append(f"{profile_name}.{channel_name}.valid_min")
    if profile.valid_max is None:
        missing.append(f"{profile_name}.{channel_name}.valid_max")


def evaluate_d1(
    dataset_path: Union[str, Path],
    bundle: TaskBundle,
) -> CriterionResult:
    """Evaluate D1 for a CSV dataset and a declared calibration task."""

    path = Path(dataset_path).expanduser().resolve()
    task = bundle.task
    context = {
        "task_id": task.task_id,
        "dataset_path": str(path),
        "sensor_profile_id": bundle.sensor.instrument_id,
        "reference_profile_id": bundle.reference.instrument_id,
        "setup_profile_id": bundle.setup.setup_id,
    }
    missing_evidence: List[str] = []
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

    sensor_mapping = task.dataset_mapping.sensor_channels
    reference_mapping = task.dataset_mapping.reference_channels

    for channel_name in sensor_mapping:
        profile = bundle.sensor.channels.get(channel_name)
        if profile is None:
            add_violation(
                "unknown_sensor_channel",
                f"sensor channel {channel_name!r} is not present in the profile",
                field=channel_name,
            )
            continue
        _missing_channel_evidence(
            "sensor_profile", channel_name, profile, missing_evidence
        )

    for channel_name in reference_mapping:
        profile = bundle.reference.channels.get(channel_name)
        if profile is None:
            add_violation(
                "unknown_reference_channel",
                f"reference channel {channel_name!r} is not present in the profile",
                field=channel_name,
            )
            continue
        _missing_channel_evidence(
            "reference_profile", channel_name, profile, missing_evidence
        )

        actual_uncertainty = profile.expanded_uncertainty
        allowed_uncertainty = task.d1.maximum_reference_uncertainty.get(channel_name)
        if actual_uncertainty is None:
            missing_evidence.append(
                f"reference_profile.{channel_name}.expanded_uncertainty"
            )
        if allowed_uncertainty is None:
            missing_evidence.append(
                f"task.d1.maximum_reference_uncertainty.{channel_name}"
            )
        if (
            actual_uncertainty is not None
            and allowed_uncertainty is not None
            and actual_uncertainty > allowed_uncertainty
        ):
            add_violation(
                "reference_uncertainty_exceeded",
                f"reference uncertainty for {channel_name} exceeds the task limit",
                field=channel_name,
                observed=actual_uncertainty,
                expected=f"<= {allowed_uncertainty}",
            )

    if (
        task.d1.require_reference_certificate
        and not bundle.reference.calibration_certificate_id
    ):
        missing_evidence.append("reference_profile.calibration_certificate_id")

    maximum_time_offset = bundle.setup.maximum_time_offset_s
    if maximum_time_offset is None:
        missing_evidence.append("setup.maximum_time_offset_s")

    rotation = bundle.setup.reference_to_sensor_rotation
    if rotation is None:
        missing_evidence.append("setup.reference_to_sensor_rotation")
    elif not _rotation_is_valid(rotation):
        add_violation(
            "invalid_coordinate_rotation",
            "reference-to-sensor rotation is not a proper orthonormal 3x3 rotation",
            field="reference_to_sensor_rotation",
        )

    required_columns = [
        task.dataset_mapping.sensor_timestamp_column,
        task.dataset_mapping.reference_timestamp_column,
        *sensor_mapping.values(),
        *reference_mapping.values(),
    ]

    rows_evaluated = 0
    invalid_rows = 0
    max_abs_time_offset: Optional[float] = None
    previous_sensor_time: Optional[float] = None
    previous_reference_time: Optional[float] = None

    try:
        stream = path.open("r", encoding="utf-8", newline="")
    except OSError as exc:
        add_violation(
            "dataset_unreadable",
            f"dataset cannot be read: {exc}",
            field=str(path),
        )
        return _build_result(
            missing_evidence,
            violations,
            violation_counts,
            rows_evaluated,
            invalid_rows,
            max_abs_time_offset,
            context,
        )

    with stream:
        reader = csv.DictReader(stream)
        fieldnames = reader.fieldnames or []
        if len(fieldnames) != len(set(fieldnames)):
            add_violation(
                "duplicate_columns",
                "dataset contains duplicate column names",
            )

        for column in required_columns:
            if column not in fieldnames:
                add_violation(
                    "missing_required_column",
                    f"required dataset column {column!r} is missing",
                    field=column,
                )

        if violation_counts["missing_required_column"]:
            return _build_result(
                missing_evidence,
                violations,
                violation_counts,
                rows_evaluated,
                invalid_rows,
                max_abs_time_offset,
                context,
            )

        for row_number, row in enumerate(reader, start=2):
            rows_evaluated += 1
            row_invalid = False

            sensor_time_column = task.dataset_mapping.sensor_timestamp_column
            reference_time_column = task.dataset_mapping.reference_timestamp_column
            sensor_time = _parse_finite(row.get(sensor_time_column))
            reference_time = _parse_finite(row.get(reference_time_column))

            if sensor_time is None:
                row_invalid = True
                add_violation(
                    "invalid_numeric_value",
                    "sensor timestamp is missing, non-numeric, or non-finite",
                    row_number=row_number,
                    field=sensor_time_column,
                    observed=row.get(sensor_time_column),
                )
            if reference_time is None:
                row_invalid = True
                add_violation(
                    "invalid_numeric_value",
                    "reference timestamp is missing, non-numeric, or non-finite",
                    row_number=row_number,
                    field=reference_time_column,
                    observed=row.get(reference_time_column),
                )

            if sensor_time is not None:
                if (
                    previous_sensor_time is not None
                    and sensor_time <= previous_sensor_time
                ):
                    row_invalid = True
                    add_violation(
                        "non_monotonic_timestamp",
                        "sensor timestamps must be strictly increasing",
                        row_number=row_number,
                        field=sensor_time_column,
                        observed=sensor_time,
                    )
                previous_sensor_time = sensor_time

            if reference_time is not None:
                if (
                    previous_reference_time is not None
                    and reference_time <= previous_reference_time
                ):
                    row_invalid = True
                    add_violation(
                        "non_monotonic_timestamp",
                        "reference timestamps must be strictly increasing",
                        row_number=row_number,
                        field=reference_time_column,
                        observed=reference_time,
                    )
                previous_reference_time = reference_time

            if sensor_time is not None and reference_time is not None:
                time_offset = abs(sensor_time - reference_time)
                max_abs_time_offset = (
                    time_offset
                    if max_abs_time_offset is None
                    else max(max_abs_time_offset, time_offset)
                )
                if (
                    maximum_time_offset is not None
                    and time_offset > maximum_time_offset
                ):
                    row_invalid = True
                    add_violation(
                        "synchronization_exceeded",
                        "sensor/reference time offset exceeds the declared maximum",
                        row_number=row_number,
                        observed=time_offset,
                        expected=f"<= {maximum_time_offset} s",
                    )

            row_invalid |= _check_channel_values(
                row,
                row_number,
                sensor_mapping.items(),
                bundle.sensor.channels,
                "sensor",
                add_violation,
            )
            row_invalid |= _check_channel_values(
                row,
                row_number,
                reference_mapping.items(),
                bundle.reference.channels,
                "reference",
                add_violation,
            )

            if row_invalid:
                invalid_rows += 1

    if rows_evaluated == 0:
        add_violation("empty_dataset", "dataset contains no observations")

    return _build_result(
        missing_evidence,
        violations,
        violation_counts,
        rows_evaluated,
        invalid_rows,
        max_abs_time_offset,
        context,
    )


def _check_channel_values(
    row: Dict[str, str],
    row_number: int,
    mappings: Iterable[Tuple[str, str]],
    profiles: Dict[str, ChannelProfile],
    profile_kind: str,
    add_violation: Any,
) -> bool:
    row_invalid = False
    for channel_name, column_name in mappings:
        parsed = _parse_finite(row.get(column_name))
        if parsed is None:
            row_invalid = True
            add_violation(
                "invalid_numeric_value",
                f"{profile_kind} value is missing, non-numeric, or non-finite",
                row_number=row_number,
                field=column_name,
                observed=row.get(column_name),
            )
            continue

        profile = profiles.get(channel_name)
        if profile is None:
            continue
        if profile.valid_min is not None and parsed < profile.valid_min:
            row_invalid = True
            add_violation(
                "measurement_out_of_range",
                f"{profile_kind} value is below its valid range",
                row_number=row_number,
                field=column_name,
                observed=parsed,
                expected=f">= {profile.valid_min}",
            )
        if profile.valid_max is not None and parsed > profile.valid_max:
            row_invalid = True
            add_violation(
                "measurement_out_of_range",
                f"{profile_kind} value is above its valid range",
                row_number=row_number,
                field=column_name,
                observed=parsed,
                expected=f"<= {profile.valid_max}",
            )
    return row_invalid


def _build_result(
    missing_evidence: List[str],
    violations: List[Violation],
    violation_counts: Counter[str],
    rows_evaluated: int,
    invalid_rows: int,
    max_abs_time_offset: Optional[float],
    context: Dict[str, Any],
) -> CriterionResult:
    unique_missing = list(dict.fromkeys(missing_evidence))
    total_violations = sum(violation_counts.values())

    if total_violations:
        status = CriterionStatus.FAIL
        summary = f"D1 failed with {total_violations} known violation(s)."
    elif unique_missing:
        status = CriterionStatus.INDETERMINATE
        summary = (
            f"D1 is indeterminate because {len(unique_missing)} required "
            "evidence item(s) are missing."
        )
    else:
        status = CriterionStatus.PASS
        summary = "D1 passed: all required evidence is present and no violation was detected."

    return CriterionResult(
        criterion_id="D1",
        criterion_name="Measurement and Reference Validity",
        status=status,
        summary=summary,
        context=context,
        metrics={
            "rows_evaluated": rows_evaluated,
            "invalid_rows": invalid_rows,
            "maximum_absolute_time_offset_s": max_abs_time_offset,
            "total_violations": total_violations,
            "violations_by_code": dict(sorted(violation_counts.items())),
            "reported_violation_limit": MAX_REPORTED_VIOLATIONS,
        },
        missing_evidence=unique_missing,
        violations=violations,
    )
