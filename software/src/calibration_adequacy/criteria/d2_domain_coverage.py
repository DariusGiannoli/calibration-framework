from __future__ import annotations

import csv
import itertools
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

from ..models import (
    CriterionResult,
    CriterionStatus,
    D2AxisDomain,
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


def _build_kd_tree(
    points: List[Tuple[float, ...]],
    depth: int = 0,
) -> Optional[_KDNode]:
    if not points:
        return None
    dimensions = len(points[0])
    axis = depth % dimensions
    points.sort(key=lambda point: point[axis])
    middle = len(points) // 2
    return _KDNode(
        point=points[middle],
        axis=axis,
        left=_build_kd_tree(points[:middle], depth + 1),
        right=_build_kd_tree(points[middle + 1 :], depth + 1),
    )


def _squared_distance(
    first: Sequence[float],
    second: Sequence[float],
) -> float:
    return sum((left - right) ** 2 for left, right in zip(first, second))


def _nearest_squared_distance(
    node: Optional[_KDNode],
    query: Sequence[float],
    best: float = math.inf,
) -> float:
    if node is None:
        return best

    best = min(best, _squared_distance(node.point, query))
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
        criterion_id="D2",
        criterion_name="Domain Coverage",
        status=CriterionStatus.INDETERMINATE,
        summary=summary,
        context=_context(bundle, dataset_path, d1_status),
        metrics=metrics or {},
        missing_evidence=list(dict.fromkeys(missing_evidence)),
        violations=[],
    )


def _missing_d2_evidence(
    axes: Sequence[str],
    domain: Dict[str, D2AxisDomain],
    maximum_fill_distance: Optional[float],
    bundle: TaskBundle,
) -> List[str]:
    missing: List[str] = []
    for axis in axes:
        if axis not in bundle.task.dataset_mapping.reference_channels:
            missing.append(f"task.dataset_mapping.reference_channels.{axis}")
        axis_domain = domain.get(axis)
        if axis_domain is None:
            missing.append(f"task.d2.domain.{axis}")
            continue
        if axis_domain.minimum is None:
            missing.append(f"task.d2.domain.{axis}.minimum")
        if axis_domain.maximum is None:
            missing.append(f"task.d2.domain.{axis}.maximum")
        if axis_domain.grid_points is None:
            missing.append(f"task.d2.domain.{axis}.grid_points")
    if maximum_fill_distance is None:
        missing.append("task.d2.maximum_fill_distance")
    return missing


def evaluate_d2(
    dataset_path: Union[str, Path],
    bundle: TaskBundle,
) -> CriterionResult:
    """Evaluate the normalized grid fill-distance criterion declared as D2."""

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

    missing_evidence = _missing_d2_evidence(
        requirements.axes,
        requirements.domain,
        requirements.maximum_fill_distance,
        bundle,
    )
    if missing_evidence:
        return _indeterminate_result(
            bundle,
            path,
            f"D2 is indeterminate because {len(missing_evidence)} required "
            "declaration(s) are missing.",
            missing_evidence,
            d1_status=d1_result.status.value,
        )

    axes = requirements.axes
    rotation = bundle.setup.reference_to_sensor_rotation
    if rotation is None:
        # D1 cannot pass with a missing rotation, but keep this invariant explicit.
        return _indeterminate_result(
            bundle,
            path,
            "D2 is indeterminate because the coordinate transformation is missing.",
            ["setup.reference_to_sensor_rotation"],
            d1_status=d1_result.status.value,
        )

    if len(axes) != len(rotation):
        return CriterionResult(
            criterion_id="D2",
            criterion_name="Domain Coverage",
            status=CriterionStatus.FAIL,
            summary=(
                "D2 failed because the declared axes do not match the "
                "setup transformation."
            ),
            context=_context(bundle, path, d1_result.status.value),
            metrics={
                "declared_axis_count": len(axes),
                "transformation_dimension": len(rotation),
            },
            missing_evidence=[],
            violations=[
                Violation(
                    code="axis_rotation_dimension_mismatch",
                    message="D2 axis count must match the setup transformation",
                    observed=len(axes),
                    expected=str(len(rotation)),
                )
            ],
        )

    axis_domains = [requirements.domain[axis] for axis in axes]
    lower_bounds = [float(axis.minimum) for axis in axis_domains]
    upper_bounds = [float(axis.maximum) for axis in axis_domains]
    widths = [
        maximum - minimum
        for minimum, maximum in zip(lower_bounds, upper_bounds)
    ]
    grid_shape = [int(axis.grid_points) for axis in axis_domains]
    grid_point_count = math.prod(grid_shape)
    if grid_point_count > MAX_EVALUATION_GRID_POINTS:
        return _indeterminate_result(
            bundle,
            path,
            "D2 was not evaluated because the declared grid exceeds the current "
            "software safety limit.",
            ["task.d2.grid_within_software_limit"],
            d1_status=d1_result.status.value,
            metrics={
                "grid_shape": grid_shape,
                "grid_point_count": grid_point_count,
                "software_grid_point_limit": MAX_EVALUATION_GRID_POINTS,
            },
        )

    achieved_points = _load_achieved_points(path, bundle, axes, rotation)
    normalized_points = [
        tuple(
            (value - minimum) / width
            for value, minimum, width in zip(point, lower_bounds, widths)
        )
        for point in achieved_points
    ]
    tree = _build_kd_tree(normalized_points.copy())
    if tree is None:
        return _indeterminate_result(
            bundle,
            path,
            "D2 is indeterminate because no D1-valid reference points are available.",
            ["dataset.D1_valid_reference_points"],
            d1_status=d1_result.status.value,
        )

    normalized_grid_axes = [
        tuple(index / (count - 1) for index in range(count))
        for count in grid_shape
    ]
    maximum_squared_distance = -1.0
    worst_normalized_point: Optional[Tuple[float, ...]] = None
    for grid_point in itertools.product(*normalized_grid_axes):
        nearest_squared = _nearest_squared_distance(tree, grid_point)
        if nearest_squared > maximum_squared_distance:
            maximum_squared_distance = nearest_squared
            worst_normalized_point = tuple(grid_point)

    estimated_fill_distance = math.sqrt(maximum_squared_distance)
    worst_point = {
        axis: lower + normalized * width
        for axis, lower, width, normalized in zip(
            axes,
            lower_bounds,
            widths,
            worst_normalized_point or (),
        )
    }
    achieved_minimum = {
        axis: min(point[index] for point in achieved_points)
        for index, axis in enumerate(axes)
    }
    achieved_maximum = {
        axis: max(point[index] for point in achieved_points)
        for index, axis in enumerate(axes)
    }
    inside_domain_count = sum(
        all(0.0 <= coordinate <= 1.0 for coordinate in point)
        for point in normalized_points
    )
    grid_covering_radius = 0.5 * math.sqrt(
        sum((1.0 / (count - 1)) ** 2 for count in grid_shape)
    )
    maximum_allowed = float(requirements.maximum_fill_distance)

    metrics = {
        "d1_status": d1_result.status.value,
        "samples_used": len(achieved_points),
        "samples_inside_domain": inside_domain_count,
        "samples_outside_domain": len(achieved_points) - inside_domain_count,
        "axes": axes,
        "domain_minimum": dict(zip(axes, lower_bounds)),
        "domain_maximum": dict(zip(axes, upper_bounds)),
        "achieved_minimum": achieved_minimum,
        "achieved_maximum": achieved_maximum,
        "grid_shape": grid_shape,
        "grid_point_count": grid_point_count,
        "grid_covering_radius_normalized": grid_covering_radius,
        "estimated_fill_distance": estimated_fill_distance,
        "maximum_fill_distance": maximum_allowed,
        "worst_covered_domain_point": worst_point,
        "total_violations": 0,
        "violations_by_code": {},
    }

    if estimated_fill_distance <= maximum_allowed:
        return CriterionResult(
            criterion_id="D2",
            criterion_name="Domain Coverage",
            status=CriterionStatus.PASS,
            summary=(
                "D2 passed: the estimated fill distance is within the "
                "declared limit."
            ),
            context=_context(bundle, path, d1_result.status.value),
            metrics=metrics,
            missing_evidence=[],
            violations=[],
        )

    failure_metrics = {
        **metrics,
        "total_violations": 1,
        "violations_by_code": {"fill_distance_exceeded": 1},
    }
    return CriterionResult(
        criterion_id="D2",
        criterion_name="Domain Coverage",
        status=CriterionStatus.FAIL,
        summary="D2 failed: the estimated fill distance exceeds the declared limit.",
        context=_context(bundle, path, d1_result.status.value),
        metrics=failure_metrics,
        missing_evidence=[],
        violations=[
            Violation(
                code="fill_distance_exceeded",
                message="estimated normalized fill distance exceeds h_max",
                observed=estimated_fill_distance,
                expected=f"<= {maximum_allowed}",
            )
        ],
    )


def _load_achieved_points(
    path: Path,
    bundle: TaskBundle,
    axes: Sequence[str],
    rotation: Sequence[Sequence[float]],
) -> List[Tuple[float, ...]]:
    columns = [
        bundle.task.dataset_mapping.reference_channels[axis] for axis in axes
    ]
    points: List[Tuple[float, ...]] = []
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        for row in reader:
            reference_point = tuple(float(row[column]) for column in columns)
            points.append(_matrix_vector_product(rotation, reference_point))
    return points
