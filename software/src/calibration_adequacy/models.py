from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CriterionStatus(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    INDETERMINATE = "INDETERMINATE"


class Violation(StrictModel):
    code: str
    message: str
    row_number: Optional[int] = None
    field: Optional[str] = None
    observed: Optional[Any] = None
    expected: Optional[str] = None


class CriterionResult(StrictModel):
    criterion_id: str
    criterion_name: str
    status: CriterionStatus
    summary: str
    context: Dict[str, Any] = Field(default_factory=dict)
    metrics: Dict[str, Any] = Field(default_factory=dict)
    missing_evidence: List[str] = Field(default_factory=list)
    violations: List[Violation] = Field(default_factory=list)


class ChannelProfile(StrictModel):
    unit: Optional[str] = Field(
        default=None,
        description="Physical or raw unit used by the channel.",
    )
    valid_min: Optional[float] = Field(
        default=None,
        description="Minimum valid instrument reading, not a dataset minimum.",
    )
    valid_max: Optional[float] = Field(
        default=None,
        description="Maximum valid instrument reading, not a dataset maximum.",
    )
    expanded_uncertainty: Optional[float] = Field(
        default=None,
        ge=0,
        description="Expanded uncertainty of a reference channel.",
    )

    @model_validator(mode="after")
    def validate_range(self) -> "ChannelProfile":
        if (
            self.valid_min is not None
            and self.valid_max is not None
            and self.valid_min >= self.valid_max
        ):
            raise ValueError("valid_min must be smaller than valid_max")
        return self


class InstrumentProfile(StrictModel):
    schema_version: str = "0.1"
    instrument_id: str
    kind: Literal["sensor", "reference"]
    channels: Dict[str, ChannelProfile] = Field(min_length=1)
    calibration_certificate_id: Optional[str] = Field(
        default=None,
        description="Reference calibration certificate or equivalent evidence.",
    )


class SetupProfile(StrictModel):
    schema_version: str = "0.1"
    setup_id: str
    maximum_time_offset_s: Optional[float] = Field(default=None, ge=0)
    reference_to_sensor_rotation: Optional[List[List[float]]] = None

    @model_validator(mode="after")
    def validate_rotation_shape(self) -> "SetupProfile":
        rotation = self.reference_to_sensor_rotation
        if rotation is not None and (
            len(rotation) != 3 or any(len(row) != 3 for row in rotation)
        ):
            raise ValueError("reference_to_sensor_rotation must be a 3x3 matrix")
        return self


class ProfileReferences(StrictModel):
    sensor: str
    reference: str
    setup: str


class DatasetMapping(StrictModel):
    sensor_timestamp_column: str
    reference_timestamp_column: str
    sensor_channels: Dict[str, str] = Field(
        min_length=1,
        description="Map sensor profile channel names to CSV column names."
    )
    reference_channels: Dict[str, str] = Field(
        min_length=1,
        description="Map reference profile channel names to CSV column names."
    )


class D1Requirements(StrictModel):
    maximum_reference_uncertainty: Dict[str, Optional[float]]
    require_reference_certificate: bool = True

    @model_validator(mode="after")
    def validate_uncertainty_limits(self) -> "D1Requirements":
        negative = [
            channel
            for channel, value in self.maximum_reference_uncertainty.items()
            if value is not None and value < 0
        ]
        if negative:
            raise ValueError(
                "maximum reference uncertainty cannot be negative for: "
                + ", ".join(sorted(negative))
            )
        return self


class TaskConfig(StrictModel):
    schema_version: str = "0.1"
    task_id: str
    profiles: ProfileReferences
    dataset_mapping: DatasetMapping
    d1: D1Requirements


class TaskBundle(StrictModel):
    task_path: Path
    task: TaskConfig
    sensor: InstrumentProfile
    reference: InstrumentProfile
    setup: SetupProfile

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)
