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
            not rotation or any(len(row) != len(rotation) for row in rotation)
        ):
            raise ValueError("reference_to_sensor_rotation must be a square matrix")
        return self


class ProfileReferences(StrictModel):
    sensor: str
    reference: str
    setup: str


class DatasetMapping(StrictModel):
    sensor_timestamp_column: str
    reference_timestamp_column: str
    run_id_column: Optional[str] = Field(
        default=None,
        min_length=1,
        description="CSV column identifying independently acquired runs.",
    )
    configuration_id_column: Optional[str] = Field(
        default=None,
        min_length=1,
        description="CSV column identifying the trajectory configuration.",
    )
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


class D2AxisDomain(StrictModel):
    minimum: Optional[float] = Field(
        default=None,
        description="Lower bound of the intended domain for this axis.",
    )
    maximum: Optional[float] = Field(
        default=None,
        description="Upper bound of the intended domain for this axis.",
    )
    grid_points: Optional[int] = Field(
        default=None,
        ge=2,
        description="Number of evaluation-grid points including both boundaries.",
    )

    @model_validator(mode="after")
    def validate_bounds(self) -> "D2AxisDomain":
        if (
            self.minimum is not None
            and self.maximum is not None
            and self.minimum >= self.maximum
        ):
            raise ValueError("minimum must be smaller than maximum")
        return self


class D2Requirements(StrictModel):
    axes: List[str] = Field(
        min_length=1,
        description="Ordered reference axes used in the coverage calculation.",
    )
    domain: Dict[str, D2AxisDomain]
    maximum_fill_distance: Optional[float] = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_unique_axes(self) -> "D2Requirements":
        if len(self.axes) != len(set(self.axes)):
            raise ValueError("D2 axes must be unique")
        return self


class D3ChannelNormalization(StrictModel):
    minimum: Optional[float] = Field(
        default=None,
        description="Declared lower bound used to normalize this input.",
    )
    maximum: Optional[float] = Field(
        default=None,
        description="Declared upper bound used to normalize this input.",
    )

    @model_validator(mode="after")
    def validate_bounds(self) -> "D3ChannelNormalization":
        if (
            self.minimum is not None
            and self.maximum is not None
            and self.minimum >= self.maximum
        ):
            raise ValueError("minimum must be smaller than maximum")
        return self


class D3Requirements(StrictModel):
    model_type: Literal["affine"]
    input_channels: List[str] = Field(min_length=1)
    output_dimension: int = Field(ge=1)
    normalization: Dict[str, D3ChannelNormalization]
    relative_rank_tolerance: Optional[float] = Field(
        default=None,
        ge=0,
        lt=1,
    )
    maximum_condition_number: Optional[float] = Field(default=None, ge=1)

    @model_validator(mode="after")
    def validate_unique_inputs(self) -> "D3Requirements":
        if len(self.input_channels) != len(set(self.input_channels)):
            raise ValueError("D3 input channels must be unique")
        return self


class D4SignalRequirements(StrictModel):
    source: Literal["sensor", "reference"]
    channel: str = Field(min_length=1)
    maximum_autocorrelation_lag: Optional[int] = Field(default=None, ge=1)


class D4RunEvidence(StrictModel):
    run_id: str = Field(min_length=1)
    acquisition_id: Optional[str] = Field(default=None, min_length=1)
    separate_acquisition: Optional[bool] = None
    initialization_completed: Optional[bool] = None
    zeroing_completed: Optional[bool] = None


class D4Requirements(StrictModel):
    signals: List[D4SignalRequirements] = Field(min_length=1)
    minimum_independent_runs: Optional[int] = Field(default=None, ge=1)
    minimum_runs_per_configuration: Optional[Dict[str, int]] = None
    minimum_effective_sample_size: Optional[float] = Field(default=None, gt=0)
    initialization_procedure_id: Optional[str] = Field(
        default=None,
        min_length=1,
    )
    zeroing_procedure_id: Optional[str] = Field(default=None, min_length=1)
    run_evidence: Optional[List[D4RunEvidence]] = None

    @model_validator(mode="after")
    def validate_d4_declarations(self) -> "D4Requirements":
        signal_keys = [
            (signal.source, signal.channel)
            for signal in self.signals
        ]
        if len(signal_keys) != len(set(signal_keys)):
            raise ValueError("D4 signals must be unique")

        if self.minimum_runs_per_configuration is not None:
            configuration_counts = self.minimum_runs_per_configuration
            invalid = [
                configuration
                for configuration, count in configuration_counts.items()
                if not configuration or count < 1
            ]
            if invalid:
                raise ValueError(
                    "D4 configuration identifiers must be non-empty and "
                    "minimum run counts must be at least one"
                )

        if self.run_evidence is not None:
            run_ids = [evidence.run_id for evidence in self.run_evidence]
            if len(run_ids) != len(set(run_ids)):
                raise ValueError("D4 run_evidence run_id values must be unique")
        return self


class D5DataUseEvidence(StrictModel):
    data_dependent_preprocessing: Optional[
        Literal["development_only", "includes_test"]
    ] = None
    model_selection: Optional[
        Literal["development_only", "includes_test"]
    ] = None
    parameter_estimation: Optional[
        Literal["development_only", "includes_test"]
    ] = None
    performance_threshold_selection: Optional[
        Literal["development_only", "includes_test"]
    ] = None
    final_performance_evaluation: Optional[
        Literal["test_only", "includes_development"]
    ] = None
    model_locked_before_test_evaluation: Optional[bool] = None
    test_results_used_for_further_development: Optional[bool] = None


class D5Requirements(StrictModel):
    validation_unit: Literal["run"]
    model_source: Literal["task.d3"]
    development_units: Optional[List[str]] = None
    test_units: Optional[List[str]] = None
    minimum_test_units: Optional[int] = Field(default=None, ge=1)
    split_manifest_id: Optional[str] = Field(default=None, min_length=1)
    split_frozen_before_development: Optional[bool] = None
    data_use: Optional[D5DataUseEvidence] = None


class TaskConfig(StrictModel):
    schema_version: str = "0.1"
    task_id: str
    profiles: ProfileReferences
    dataset_mapping: DatasetMapping
    d1: D1Requirements
    d2: Optional[D2Requirements] = None
    d3: Optional[D3Requirements] = None
    d4: Optional[D4Requirements] = None
    d5: Optional[D5Requirements] = None


class TaskBundle(StrictModel):
    task_path: Path
    task: TaskConfig
    sensor: InstrumentProfile
    reference: InstrumentProfile
    setup: SetupProfile

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)
