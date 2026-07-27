from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from ..config import load_evidence_manifest
from ..models import (
    CriterionResult,
    CriterionStatus,
    EvidenceFileRecord,
    TaskBundle,
    Violation,
)
from .d0_claim_completeness import evaluate_d0
from .d1_measurement_reference import evaluate_d1
from .d2_domain_coverage import evaluate_d2
from .d3_informativeness import evaluate_d3
from .d4_replication_dependence import evaluate_d4
from .d5_leakage_resistant_validation import evaluate_d5
from .d6_performance_uncertainty import evaluate_d6


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _indeterminate(
    bundle: TaskBundle,
    dataset_path: Path,
    missing: List[str],
    metrics: Optional[Dict[str, Any]] = None,
) -> CriterionResult:
    unique = list(dict.fromkeys(missing))
    return CriterionResult(
        criterion_id="D7",
        criterion_name="Reproducibility and Provenance",
        status=CriterionStatus.INDETERMINATE,
        summary=(
            f"D7 is indeterminate because {len(unique)} required provenance "
            "item(s) are missing."
        ),
        context={
            "task_id": bundle.task.task_id,
            "dataset_path": str(dataset_path),
        },
        metrics=metrics or {},
        missing_evidence=unique,
        violations=[],
    )


def evaluate_d7(
    dataset_path: Union[str, Path],
    bundle: TaskBundle,
) -> CriterionResult:
    """Verify evidence hashes and reproduce recorded D0-D6 outcomes."""

    dataset = Path(dataset_path).expanduser().resolve()
    requirements = bundle.task.d7
    if requirements is None:
        return _indeterminate(bundle, dataset, ["task.d7"])
    missing: List[str] = []
    for field_name in (
        "evidence_manifest_path",
        "evidence_package_id",
        "required_file_roles",
        "required_criterion_results",
        "numerical_absolute_tolerance",
        "numerical_relative_tolerance",
    ):
        if getattr(requirements, field_name) is None:
            missing.append(f"task.d7.{field_name}")
    if missing:
        return _indeterminate(bundle, dataset, missing)

    manifest_path = (
        bundle.task_path.parent / str(requirements.evidence_manifest_path)
    ).resolve()
    if not manifest_path.is_file():
        return _indeterminate(
            bundle,
            dataset,
            ["task.d7.evidence_manifest_path"],
            {"resolved_manifest_path": str(manifest_path)},
        )
    manifest = load_evidence_manifest(manifest_path)
    manifest_directory = manifest_path.parent
    violations: List[Violation] = []

    if manifest.package_id != requirements.evidence_package_id:
        violations.append(
            Violation(
                code="evidence_package_id_mismatch",
                message="manifest package ID disagrees with the task",
                observed=manifest.package_id,
                expected=str(requirements.evidence_package_id),
            )
        )

    file_paths = [record.path for record in manifest.files]
    if len(file_paths) != len(set(file_paths)):
        violations.append(
            Violation(
                code="duplicate_evidence_file",
                message="evidence manifest file paths must be unique",
            )
        )
    observed_roles = {record.role for record in manifest.files}
    for role in requirements.required_file_roles or []:
        if role not in observed_roles:
            missing.append(f"evidence.role.{role}")

    integrity_records: Dict[str, Any] = {}
    declared_files: Dict[Path, EvidenceFileRecord] = {}
    for record in manifest.files:
        evidence_path = (manifest_directory / record.path).resolve()
        if evidence_path in declared_files:
            violations.append(
                Violation(
                    code="duplicate_resolved_evidence_file",
                    message=(
                        "multiple manifest paths resolve to the same "
                        "evidence file"
                    ),
                    field=record.path,
                )
            )
        else:
            declared_files[evidence_path] = record
        item: Dict[str, Any] = {
            "role": record.role,
            "expected_sha256": record.sha256,
            "resolved_path": str(evidence_path),
        }
        if not evidence_path.is_file():
            item["status"] = "MISSING"
            violations.append(
                Violation(
                    code="evidence_file_missing",
                    message="a declared evidence file is missing",
                    field=record.path,
                )
            )
        else:
            actual_hash = _sha256(evidence_path)
            item["actual_sha256"] = actual_hash
            item["status"] = (
                "MATCH" if actual_hash == record.sha256 else "MISMATCH"
            )
            if actual_hash != record.sha256:
                violations.append(
                    Violation(
                        code="evidence_hash_mismatch",
                        message="an evidence file does not match its SHA-256 record",
                        field=record.path,
                        observed=actual_hash,
                        expected=record.sha256,
                    )
                )
        integrity_records[record.path] = item

    active_inputs = [
        (dataset, "raw_data", "dataset"),
        (bundle.task_path.resolve(), "task_config", "task configuration"),
        (
            (bundle.task_path.parent / bundle.task.profiles.sensor).resolve(),
            "sensor_profile",
            "sensor profile",
        ),
        (
            (bundle.task_path.parent / bundle.task.profiles.reference).resolve(),
            "reference_profile",
            "reference profile",
        ),
        (
            (bundle.task_path.parent / bundle.task.profiles.setup).resolve(),
            "setup_profile",
            "setup profile",
        ),
    ]
    for active_path, expected_role, label in active_inputs:
        _require_manifested_input(
            active_path,
            expected_role,
            label,
            declared_files,
            violations,
        )

    required_results = list(requirements.required_criterion_results or [])
    for criterion_id in required_results:
        if criterion_id not in manifest.criterion_results:
            missing.append(f"evidence.criterion_results.{criterion_id}")

    if bundle.task.d6 is not None:
        recorded_seed = manifest.random_seeds.get("d6.bootstrap")
        if recorded_seed is None:
            missing.append("evidence.random_seeds.d6.bootstrap")
        elif recorded_seed != bundle.task.d6.bootstrap_random_seed:
            violations.append(
                Violation(
                    code="random_seed_mismatch",
                    message="D6 bootstrap seed disagrees with the task",
                    observed=recorded_seed,
                    expected=str(bundle.task.d6.bootstrap_random_seed),
                )
            )

    live_results = {
        "D0": evaluate_d0(bundle),
        "D1": evaluate_d1(dataset, bundle),
        "D2": evaluate_d2(dataset, bundle),
        "D3": evaluate_d3(dataset, bundle),
        "D4": evaluate_d4(dataset, bundle),
        "D5": evaluate_d5(dataset, bundle),
        "D6": evaluate_d6(dataset, bundle),
    }
    reproduction: Dict[str, Any] = {}
    absolute_tolerance = float(requirements.numerical_absolute_tolerance)
    relative_tolerance = float(requirements.numerical_relative_tolerance)
    for criterion_id in required_results:
        result_path_text = manifest.criterion_results.get(criterion_id)
        if result_path_text is None:
            continue
        result_path = (manifest_directory / result_path_text).resolve()
        _require_manifested_input(
            result_path,
            "criterion_result",
            f"{criterion_id} recorded result",
            declared_files,
            violations,
        )
        if not result_path.is_file():
            continue
        try:
            recorded = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            violations.append(
                Violation(
                    code="criterion_result_unreadable",
                    message=f"recorded {criterion_id} result cannot be read: {exc}",
                    field=result_path_text,
                )
            )
            continue
        live = live_results[criterion_id].model_dump(mode="json")
        differences: List[str] = []
        _compare_values(
            recorded.get("status"),
            live["status"],
            f"{criterion_id}.status",
            absolute_tolerance,
            relative_tolerance,
            differences,
        )
        _compare_values(
            recorded.get("metrics"),
            live["metrics"],
            f"{criterion_id}.metrics",
            absolute_tolerance,
            relative_tolerance,
            differences,
        )
        reproduction[criterion_id] = {
            "recorded_result": result_path_text,
            "reproduced": not differences,
            "differences": differences[:50],
        }
        if differences:
            violations.append(
                Violation(
                    code="criterion_result_not_reproduced",
                    message=f"{criterion_id} result differs from recorded evidence",
                    field=result_path_text,
                    observed=differences[:10],
                    expected="matching outcome and numerical metrics",
                )
            )

    metrics = {
        "evidence_manifest_path": str(manifest_path),
        "evidence_package_id": manifest.package_id,
        "software_commit": manifest.software_commit,
        "dependency_snapshot_id": manifest.dependency_snapshot_id,
        "preprocessing_procedure_id": manifest.preprocessing_procedure_id,
        "exclusion_record_id": manifest.exclusion_record_id,
        "partition_manifest_id": manifest.partition_manifest_id,
        "random_seeds": manifest.random_seeds,
        "integrity_records": integrity_records,
        "reproduction": reproduction,
        "absolute_tolerance": absolute_tolerance,
        "relative_tolerance": relative_tolerance,
    }
    if violations:
        counts: Dict[str, int] = {}
        for violation in violations:
            counts[violation.code] = counts.get(violation.code, 0) + 1
        metrics.update(
            total_violations=len(violations),
            violations_by_code=counts,
        )
        return CriterionResult(
            criterion_id="D7",
            criterion_name="Reproducibility and Provenance",
            status=CriterionStatus.FAIL,
            summary=(
                f"D7 failed with {len(violations)} known integrity or "
                "reproduction violation(s)."
            ),
            context={
                "task_id": bundle.task.task_id,
                "dataset_path": str(dataset),
            },
            metrics=metrics,
            missing_evidence=list(dict.fromkeys(missing)),
            violations=violations,
        )
    if missing:
        return _indeterminate(bundle, dataset, missing, metrics)
    metrics.update(total_violations=0, violations_by_code={})
    return CriterionResult(
        criterion_id="D7",
        criterion_name="Reproducibility and Provenance",
        status=CriterionStatus.PASS,
        summary=(
            "D7 passed: evidence hashes are intact and recorded D0-D6 "
            "outcomes and numerical metrics were reproduced."
        ),
        context={
            "task_id": bundle.task.task_id,
            "dataset_path": str(dataset),
        },
        metrics=metrics,
        missing_evidence=[],
        violations=[],
    )


def _compare_values(
    recorded: Any,
    live: Any,
    path: str,
    absolute_tolerance: float,
    relative_tolerance: float,
    differences: List[str],
) -> None:
    if isinstance(recorded, bool) or isinstance(live, bool):
        if recorded != live:
            differences.append(path)
        return
    if isinstance(recorded, (int, float)) and isinstance(live, (int, float)):
        if not math.isclose(
            float(recorded),
            float(live),
            rel_tol=relative_tolerance,
            abs_tol=absolute_tolerance,
        ):
            differences.append(path)
        return
    if isinstance(recorded, dict) and isinstance(live, dict):
        if set(recorded) != set(live):
            differences.append(f"{path}.__keys__")
            return
        for key in recorded:
            _compare_values(
                recorded[key],
                live[key],
                f"{path}.{key}",
                absolute_tolerance,
                relative_tolerance,
                differences,
            )
        return
    if isinstance(recorded, list) and isinstance(live, list):
        if len(recorded) != len(live):
            differences.append(f"{path}.__length__")
            return
        for index, (recorded_item, live_item) in enumerate(zip(recorded, live)):
            _compare_values(
                recorded_item,
                live_item,
                f"{path}.{index}",
                absolute_tolerance,
                relative_tolerance,
                differences,
            )
        return
    if recorded != live:
        differences.append(path)


def _require_manifested_input(
    path: Path,
    expected_role: str,
    label: str,
    declared_files: Dict[Path, EvidenceFileRecord],
    violations: List[Violation],
) -> None:
    record = declared_files.get(path.resolve())
    if record is None:
        violations.append(
            Violation(
                code="active_input_not_manifested",
                message=f"the active {label} is not recorded in the manifest",
                field=str(path),
                expected=expected_role,
            )
        )
    elif record.role != expected_role:
        violations.append(
            Violation(
                code="active_input_role_mismatch",
                message=f"the active {label} has the wrong manifest role",
                field=record.path,
                observed=record.role,
                expected=expected_role,
            )
        )
