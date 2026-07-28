from __future__ import annotations

from collections import Counter
from pathlib import Path

from .criteria import evaluate_d0
from .models import (
    DeclarationClassification,
    DeclarationRegister,
    DeclarationRegisterAudit,
    DeclarationResolutionStage,
    TaskBundle,
)


def audit_declaration_register(
    bundle: TaskBundle,
    register: DeclarationRegister,
) -> DeclarationRegisterAudit:
    """Compare a declaration register with the task's live D0 result."""

    d0_result = evaluate_d0(bundle)
    actual_paths = d0_result.missing_evidence
    registered_paths = [entry.path for entry in register.entries]
    actual_set = set(actual_paths)
    registered_set = set(registered_paths)

    missing_from_register = [
        path for path in actual_paths if path not in registered_set
    ]
    no_longer_missing = [
        path for path in registered_paths if path not in actual_set
    ]
    task_id_matches = register.task_id == bundle.task.task_id
    task_config_matches = (
        Path(register.task_config).name == bundle.task_path.name
    )
    order_matches = registered_paths == actual_paths
    aligned = (
        task_id_matches
        and task_config_matches
        and register.expected_d0_status == d0_result.status
        and register.expected_missing_count == len(actual_paths)
        and len(registered_paths) == len(actual_paths)
        and not missing_from_register
        and not no_longer_missing
    )

    classification_counts = Counter(
        {
            classification.value: 0
            for classification in DeclarationClassification
        }
    )
    classification_counts.update(
        entry.classification.value for entry in register.entries
    )
    resolution_stage_counts = Counter(
        {
            stage.value: 0
            for stage in DeclarationResolutionStage
        }
    )
    resolution_stage_counts.update(
        entry.resolution_stage.value for entry in register.entries
    )
    return DeclarationRegisterAudit(
        aligned=aligned,
        register_id=register.register_id,
        task_id=bundle.task.task_id,
        task_id_matches=task_id_matches,
        task_config_matches=task_config_matches,
        d0_status=d0_result.status,
        expected_d0_status=register.expected_d0_status,
        expected_missing_count=register.expected_missing_count,
        actual_missing_count=len(actual_paths),
        registered_count=len(registered_paths),
        order_matches=order_matches,
        missing_from_register=missing_from_register,
        no_longer_missing=no_longer_missing,
        classification_counts=dict(sorted(classification_counts.items())),
        resolution_stage_counts=dict(sorted(resolution_stage_counts.items())),
    )
