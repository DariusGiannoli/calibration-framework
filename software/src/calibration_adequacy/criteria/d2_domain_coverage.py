from __future__ import annotations

import csv
import itertools
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from ..models import (
    CriterionResult,
    CriterionStatus,
    D2ConditionDomain,
    D2ExcludedRegion,
    TaskBundle,
    Violation,
)
from .d1_measurement_reference import evaluate_d1

MAX_EVALUATION_GRID_POINTS = 2_000_000


@dataclass
class _KDNode:
    point: Tuple[float, ...]
    axis: int
    left: Optional["_KDNode"]
    right: Optional["_KDNode"]


@dataclass(frozen=True)
class _Observation:
    continuous: Tuple[float, ...]
    stratum: Tuple[str, ...]


def _build_kd_tree(
    points: List[Tuple[float, ...]],
    depth: int = 0,
) -> Optional[_KDNode]:
    if not points:
        return None
    axis = depth % len(points[0])
    points.sort(key=lambda point: point[axis])
    middle = len(points) // 2
    return _KDNode(
        point=points[middle],
        axis=axis,
        left=_build_kd_tree(points[:middle], depth + 1),
        right=_build_kd_tree(points[middle + 1 :], depth + 1),
    )


def _nearest_squared_distance(
    node: Optional[_KDNode],
    query: Sequence[float],
    best: float = math.inf,
) -> float:
    if node is None:
        return best
    best = min(
        best,
        sum((left - right) ** 2 for left, right in zip(node.point, query)),
    )
    difference = query[node.axis] - node.point[node.axis]
    near = node.left if difference <= 0 else node.right
    far = node.right if difference <= 0 else node.left
    best = _nearest_squared_distance(near, query, best)
    if difference * difference < best:
        best = _nearest_squared_distance(far, query, best)
    return best


def _matrix_vector_product(
    matrix: Sequence[Sequence[float]],
    vector: Sequence[float],
) -> Tuple[float, ...]:
    return tuple(
        sum(coefficient * value for coefficient, value in zip(row, vector))
        for row in matrix
    )


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
    metrics: Optional[Dict[str, object]] = None,
) -> CriterionResult:
    return CriterionResult(
        criterion_id="D2",
        criterion_name="Claimed-Domain Coverage",
        status=CriterionStatus.INDETERMINATE,
        summary=summary,
        context=_context(bundle, dataset_path, d1_status),
        metrics=metrics or {},
        missing_evidence=list(dict.fromkeys(missing_evidence)),
        violations=[],
    )


def _missing_condition_evidence(
    name: str,
    condition: D2ConditionDomain,
) -> List[str]:
    prefix = f"task.d2.conditions.{name}"
    missing: List[str] = []
    if condition.kind == "continuous":
        if condition.unit is None:
            missing.append(f"{prefix}.unit")
        if condition.minimum is None:
            missing.append(f"{prefix}.minimum")
        if condition.maximum is None:
            missing.append(f"{prefix}.maximum")
        if condition.grid_points is None:
            missing.append(f"{prefix}.grid_points")
    elif condition.kind == "categorical":
        if condition.categories is None:
            missing.append(f"{prefix}.categories")
    elif condition.claimed_value is None:
        missing.append(f"{prefix}.claimed_value")
    return missing


def _missing_d2_evidence(bundle: TaskBundle) -> List[str]:
    requirements = bundle.task.d2
    if requirements is None:
        return ["task.d2"]
    missing: List[str] = []
    for axis in requirements.axes:
        if axis not in bundle.task.dataset_mapping.reference_channels:
            missing.append(f"task.dataset_mapping.reference_channels.{axis}")
        axis_domain = requirements.domain.get(axis)
        if axis_domain is None:
            missing.append(f"task.d2.domain.{axis}")
            continue
        if axis_domain.minimum is None:
            missing.append(f"task.d2.domain.{axis}.minimum")
        if axis_domain.maximum is None:
            missing.append(f"task.d2.domain.{axis}.maximum")
        if axis_domain.grid_points is None:
            missing.append(f"task.d2.domain.{axis}.grid_points")
    if requirements.conditions is None:
        missing.append("task.d2.conditions")
    else:
        for name, condition in requirements.conditions.items():
            missing.extend(_missing_condition_evidence(name, condition))
    if requirements.excluded_regions is None:
        missing.append("task.d2.excluded_regions")
    if requirements.maximum_fill_distance is None:
        missing.append("task.d2.maximum_fill_distance")
    return missing


def _condition_value(
    row: Dict[str, str],
    condition: D2ConditionDomain,
) -> Any:
    if condition.source == "constant":
        return condition.constant
    return row.get(str(condition.column))


def _load_observations(
    path: Path,
    bundle: TaskBundle,
    axes: Sequence[str],
    rotation: Sequence[Sequence[float]],
    continuous_conditions: Sequence[Tuple[str, D2ConditionDomain]],
    stratum_conditions: Sequence[Tuple[str, D2ConditionDomain]],
) -> Tuple[List[_Observation], List[str]]:
    reference_columns = [
        bundle.task.dataset_mapping.reference_channels[axis] for axis in axes
    ]
    required_condition_columns = [
        str(condition.column)
        for _, condition in [*continuous_conditions, *stratum_conditions]
        if condition.source == "column"
    ]
    observations: List[_Observation] = []
    missing_columns: List[str] = []
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        fieldnames = set(reader.fieldnames or [])
        for column in required_condition_columns:
            if column not in fieldnames:
                missing_columns.append(column)
        if missing_columns:
            return [], missing_columns
        for row in reader:
            reference = tuple(float(row[column]) for column in reference_columns)
            rotated = _matrix_vector_product(rotation, reference)
            continuous = (
                *rotated,
                *(
                    float(_condition_value(row, condition))
                    for _, condition in continuous_conditions
                ),
            )
            stratum = tuple(
                str(_condition_value(row, condition))
                for _, condition in stratum_conditions
            )
            observations.append(
                _Observation(
                    continuous=tuple(continuous),
                    stratum=stratum,
                )
            )
    return observations, missing_columns


def _grid_point_excluded(
    physical_point: Dict[str, float],
    stratum: Dict[str, str],
    exclusions: Sequence[D2ExcludedRegion],
) -> Optional[str]:
    for region in exclusions:
        continuous_match = all(
            (
                interval.minimum is None
                or physical_point.get(name, -math.inf) >= interval.minimum
            )
            and (
                interval.maximum is None
                or physical_point.get(name, math.inf) <= interval.maximum
            )
            for name, interval in region.continuous_bounds.items()
        )
        categorical_match = all(
            stratum.get(name) in values
            for name, values in region.categorical_values.items()
        )
        if continuous_match and categorical_match:
            return region.region_id
    return None


def evaluate_d2(
    dataset_path: Union[str, Path],
    bundle: TaskBundle,
) -> CriterionResult:
    """Evaluate joint force-condition support over the declared claim domain."""

    path = Path(dataset_path).expanduser().resolve()
    d1_result = evaluate_d1(path, bundle)
    if d1_result.status != CriterionStatus.PASS:
        return _indeterminate_result(
            bundle,
            path,
            "D2 was not evaluated because its D1 prerequisite did not pass.",
            ["prerequisite.D1_PASS"],
            d1_status=d1_result.status.value,
            metrics={
                "d1_status": d1_result.status.value,
                "d1_summary": d1_result.summary,
            },
        )

    requirements = bundle.task.d2
    if requirements is None:
        return _indeterminate_result(
            bundle,
            path,
            "D2 is indeterminate because no D2 task configuration was declared.",
            ["task.d2"],
            d1_status=d1_result.status.value,
        )
    missing_evidence = _missing_d2_evidence(bundle)
    if missing_evidence:
        return _indeterminate_result(
            bundle,
            path,
            f"D2 is indeterminate because {len(missing_evidence)} required "
            "declaration(s) are missing.",
            missing_evidence,
            d1_status=d1_result.status.value,
        )

    axes = list(requirements.axes)
    rotation = bundle.setup.reference_to_sensor_rotation
    if rotation is None:
        return _indeterminate_result(
            bundle,
            path,
            "D2 is indeterminate because the coordinate transformation is missing.",
            ["setup.reference_to_sensor_rotation"],
            d1_status=d1_result.status.value,
        )
    if len(axes) != len(rotation):
        return _failure(
            bundle,
            path,
            d1_result.status.value,
            "axis_rotation_dimension_mismatch",
            "D2 axes must match the setup transformation dimension.",
            len(axes),
            str(len(rotation)),
        )

    conditions = requirements.conditions or {}
    continuous_conditions = [
        (name, condition)
        for name, condition in conditions.items()
        if condition.kind == "continuous"
    ]
    stratum_conditions = [
        (name, condition)
        for name, condition in conditions.items()
        if condition.kind in {"categorical", "fixed"}
    ]
    continuous_names = [*axes, *(name for name, _ in continuous_conditions)]
    lower_bounds = [
        *(float(requirements.domain[axis].minimum) for axis in axes),
        *(float(condition.minimum) for _, condition in continuous_conditions),
    ]
    upper_bounds = [
        *(float(requirements.domain[axis].maximum) for axis in axes),
        *(float(condition.maximum) for _, condition in continuous_conditions),
    ]
    grid_shape = [
        *(int(requirements.domain[axis].grid_points) for axis in axes),
        *(int(condition.grid_points) for _, condition in continuous_conditions),
    ]
    widths = [
        maximum - minimum
        for minimum, maximum in zip(lower_bounds, upper_bounds)
    ]

    claimed_stratum_values: List[List[str]] = []
    for _, condition in stratum_conditions:
        if condition.kind == "categorical":
            claimed_stratum_values.append(list(condition.categories or []))
        else:
            claimed_stratum_values.append([str(condition.claimed_value)])
    claimed_strata = list(itertools.product(*claimed_stratum_values))
    if not claimed_strata:
        claimed_strata = [tuple()]

    grid_points_per_stratum = math.prod(grid_shape)
    requested_grid_points = grid_points_per_stratum * len(claimed_strata)
    if requested_grid_points > MAX_EVALUATION_GRID_POINTS:
        return _indeterminate_result(
            bundle,
            path,
            "D2 was not evaluated because the joint grid exceeds the software limit.",
            ["task.d2.grid_within_software_limit"],
            d1_status=d1_result.status.value,
            metrics={
                "grid_shape": grid_shape,
                "claimed_stratum_count": len(claimed_strata),
                "requested_grid_point_count": requested_grid_points,
                "software_grid_point_limit": MAX_EVALUATION_GRID_POINTS,
            },
        )

    exclusions = requirements.excluded_regions or []
    known_continuous = set(continuous_names)
    known_strata = {name for name, _ in stratum_conditions}
    for exclusion in exclusions:
        unknown_continuous = set(exclusion.continuous_bounds) - known_continuous
        unknown_strata = set(exclusion.categorical_values) - known_strata
        if unknown_continuous or unknown_strata:
            return _failure(
                bundle,
                path,
                d1_result.status.value,
                "unknown_exclusion_dimension",
                "an excluded region refers to an undeclared domain dimension",
                sorted(unknown_continuous | unknown_strata),
                "declared D2 dimensions",
            )

    observations, missing_columns = _load_observations(
        path,
        bundle,
        axes,
        rotation,
        continuous_conditions,
        stratum_conditions,
    )
    if missing_columns:
        return _failure(
            bundle,
            path,
            d1_result.status.value,
            "missing_condition_column",
            "a declared operating-condition column is absent from the dataset",
            sorted(set(missing_columns)),
            "all D2 condition columns present",
        )
    if not observations:
        return _indeterminate_result(
            bundle,
            path,
            "D2 is indeterminate because no achieved observations are available.",
            ["dataset.D1_valid_observations"],
            d1_status=d1_result.status.value,
        )

    normalized_observations = [
        _Observation(
            continuous=tuple(
                (value - minimum) / width
                for value, minimum, width in zip(
                    observation.continuous,
                    lower_bounds,
                    widths,
                )
            ),
            stratum=observation.stratum,
        )
        for observation in observations
    ]
    normalized_grid_axes = [
        tuple(index / (count - 1) for index in range(count))
        for count in grid_shape
    ]
    stratum_names = [name for name, _ in stratum_conditions]
    maximum_fill_distance = -1.0
    worst_point: Dict[str, Any] = {}
    evaluated_grid_points = 0
    excluded_grid_points = 0
    stratum_metrics: Dict[str, Any] = {}

    for stratum_values in claimed_strata:
        stratum_key = "|".join(
            f"{name}={value}"
            for name, value in zip(stratum_names, stratum_values)
        ) or "__all__"
        points = [
            observation.continuous
            for observation in normalized_observations
            if observation.stratum == tuple(stratum_values)
        ]
        tree = _build_kd_tree(points.copy())
        stratum_maximum = -1.0
        stratum_worst: Dict[str, Any] = {}
        stratum_evaluated = 0
        stratum_excluded = 0
        stratum_dict = dict(zip(stratum_names, stratum_values))

        for normalized_grid_point in itertools.product(*normalized_grid_axes):
            physical = {
                name: lower + normalized * width
                for name, lower, width, normalized in zip(
                    continuous_names,
                    lower_bounds,
                    widths,
                    normalized_grid_point,
                )
            }
            exclusion_id = _grid_point_excluded(
                physical,
                stratum_dict,
                exclusions,
            )
            if exclusion_id is not None:
                excluded_grid_points += 1
                stratum_excluded += 1
                continue
            evaluated_grid_points += 1
            stratum_evaluated += 1
            nearest_squared = _nearest_squared_distance(tree, normalized_grid_point)
            distance = math.sqrt(nearest_squared)
            if distance > stratum_maximum:
                stratum_maximum = distance
                stratum_worst = {**physical, **stratum_dict}

        if stratum_evaluated == 0:
            stratum_maximum = 0.0
        if stratum_maximum > maximum_fill_distance:
            maximum_fill_distance = stratum_maximum
            worst_point = stratum_worst
        stratum_metrics[stratum_key] = {
            "achieved_samples": len(points),
            "evaluated_grid_points": stratum_evaluated,
            "excluded_grid_points": stratum_excluded,
            "estimated_fill_distance": stratum_maximum,
            "worst_covered_domain_point": stratum_worst,
        }

    inside_count = sum(
        observation.stratum in claimed_strata
        and all(0.0 <= value <= 1.0 for value in observation.continuous)
        for observation in normalized_observations
    )
    achieved_minimum = {
        name: min(observation.continuous[index] for observation in observations)
        for index, name in enumerate(continuous_names)
    }
    achieved_maximum = {
        name: max(observation.continuous[index] for observation in observations)
        for index, name in enumerate(continuous_names)
    }
    grid_covering_radius = 0.5 * math.sqrt(
        sum((1.0 / (count - 1)) ** 2 for count in grid_shape)
    )
    maximum_allowed = float(requirements.maximum_fill_distance)
    failed = maximum_fill_distance > maximum_allowed
    metrics = {
        "d1_status": d1_result.status.value,
        "samples_used": len(observations),
        "samples_inside_claim_domain": inside_count,
        "samples_outside_claim_domain": len(observations) - inside_count,
        "axes": axes,
        "condition_names": list(conditions),
        "continuous_dimensions": continuous_names,
        "categorical_dimensions": stratum_names,
        "domain_minimum": dict(zip(continuous_names, lower_bounds)),
        "domain_maximum": dict(zip(continuous_names, upper_bounds)),
        "achieved_minimum": achieved_minimum,
        "achieved_maximum": achieved_maximum,
        "grid_shape": grid_shape,
        "grid_point_count": evaluated_grid_points,
        "excluded_grid_point_count": excluded_grid_points,
        "declared_excluded_regions": [
            region.model_dump(mode="json") for region in exclusions
        ],
        "grid_covering_radius_normalized": grid_covering_radius,
        "estimated_fill_distance": maximum_fill_distance,
        "maximum_fill_distance": maximum_allowed,
        "worst_covered_domain_point": worst_point,
        "strata": stratum_metrics,
        "total_violations": 1 if failed else 0,
        "violations_by_code": {"fill_distance_exceeded": 1} if failed else {},
    }
    if not failed:
        return CriterionResult(
            criterion_id="D2",
            criterion_name="Claimed-Domain Coverage",
            status=CriterionStatus.PASS,
            summary=(
                "D2 passed: every non-excluded force-condition stratum is "
                "supported within the declared fill-distance limit."
            ),
            context=_context(bundle, path, d1_result.status.value),
            metrics=metrics,
            missing_evidence=[],
            violations=[],
        )
    return CriterionResult(
        criterion_id="D2",
        criterion_name="Claimed-Domain Coverage",
        status=CriterionStatus.FAIL,
        summary=(
            "D2 failed: at least one claimed force-condition region exceeds "
            "the declared fill-distance limit."
        ),
        context=_context(bundle, path, d1_result.status.value),
        metrics=metrics,
        missing_evidence=[],
        violations=[
            Violation(
                code="fill_distance_exceeded",
                message="joint force-condition fill distance exceeds h_max",
                observed=maximum_fill_distance,
                expected=f"<= {maximum_allowed}",
            )
        ],
    )


def _failure(
    bundle: TaskBundle,
    path: Path,
    d1_status: str,
    code: str,
    message: str,
    observed: Any,
    expected: str,
) -> CriterionResult:
    return CriterionResult(
        criterion_id="D2",
        criterion_name="Claimed-Domain Coverage",
        status=CriterionStatus.FAIL,
        summary=f"D2 failed: {message}",
        context=_context(bundle, path, d1_status),
        metrics={
            "total_violations": 1,
            "violations_by_code": {code: 1},
        },
        missing_evidence=[],
        violations=[
            Violation(
                code=code,
                message=message,
                observed=observed,
                expected=expected,
            )
        ],
    )
