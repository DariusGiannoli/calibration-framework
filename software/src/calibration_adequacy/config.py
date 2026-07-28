from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Type, TypeVar, Union

import yaml
from pydantic import BaseModel

from .models import (
    DeclarationRegister,
    EvidenceManifest,
    InstrumentProfile,
    SetupProfile,
    TaskBundle,
    TaskConfig,
)

ModelT = TypeVar("ModelT", bound=BaseModel)


class ConfigurationError(ValueError):
    """Raised when configuration files cannot be parsed or validated."""


def _load_yaml_model(path: Path, model_type: Type[ModelT]) -> ModelT:
    try:
        with path.open("r", encoding="utf-8") as stream:
            raw: Dict[str, Any] = yaml.safe_load(stream)
    except OSError as exc:
        raise ConfigurationError(f"cannot read configuration {path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise ConfigurationError(f"invalid YAML in {path}: {exc}") from exc

    if not isinstance(raw, dict):
        raise ConfigurationError(f"configuration {path} must contain a mapping")

    try:
        return model_type.model_validate(raw)
    except ValueError as exc:
        raise ConfigurationError(f"invalid configuration {path}: {exc}") from exc


def load_task_bundle(task_path: Union[str, Path]) -> TaskBundle:
    resolved_task_path = Path(task_path).expanduser().resolve()
    task = _load_yaml_model(resolved_task_path, TaskConfig)
    task_directory = resolved_task_path.parent

    sensor_path = (task_directory / task.profiles.sensor).resolve()
    reference_path = (task_directory / task.profiles.reference).resolve()
    setup_path = (task_directory / task.profiles.setup).resolve()

    sensor = _load_yaml_model(sensor_path, InstrumentProfile)
    reference = _load_yaml_model(reference_path, InstrumentProfile)
    setup = _load_yaml_model(setup_path, SetupProfile)

    if sensor.kind != "sensor":
        raise ConfigurationError(f"{sensor_path} is not a sensor profile")
    if reference.kind != "reference":
        raise ConfigurationError(f"{reference_path} is not a reference profile")

    return TaskBundle(
        task_path=resolved_task_path,
        task=task,
        sensor=sensor,
        reference=reference,
        setup=setup,
    )


def load_evidence_manifest(path: Union[str, Path]) -> EvidenceManifest:
    return _load_yaml_model(Path(path).expanduser().resolve(), EvidenceManifest)


def load_declaration_register(
    path: Union[str, Path],
) -> DeclarationRegister:
    return _load_yaml_model(
        Path(path).expanduser().resolve(),
        DeclarationRegister,
    )
