from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CriterionStatus(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    INDETERMINATE = "INDETERMINATE"
    NOT_APPLICABLE = "NOT_APPLICABLE"


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


class OverallAssessmentResult(StrictModel):
    status: CriterionStatus
    summary: str
    task_id: str
    criteria: Dict[str, CriterionResult]
    calibration_acceptance_status: Optional[CriterionStatus] = None
    missing_evidence: List[str] = Field(default_factory=list)


class DeclarationClassification(str, Enum):
    PHYSICAL_OR_APPLICATION_REQUIREMENT = (
        "1_physical_or_application_requirement"
    )
    HARDWARE_OR_REFERENCE_SPECIFICATION = (
        "2_hardware_or_reference_specification"
    )
    EXPERIMENTAL_DESIGN_CHOICE = "3_experimental_design_choice"
    STATISTICAL_JUSTIFICATION = (
        "4_statistical_threshold_requiring_simulation_or_pilot_justification"
    )
    POST_ACQUISITION_VALUE = "5_value_evaluated_after_acquisition"


class DeclarationResolutionStage(str, Enum):
    BEFORE_TRAJECTORY_SIMULATION = "before_trajectory_simulation"
    BEFORE_FINAL_ACQUISITION = "before_final_acquisition"
    AFTER_ACQUISITION = "after_acquisition"


class DeclarationRegisterEntry(StrictModel):
    path: str = Field(min_length=1)
    criterion: Literal["D2", "D3", "D6"]
    classification: DeclarationClassification
    resolution_stage: DeclarationResolutionStage
    decision_question: str = Field(min_length=1)
    legitimate_source: str = Field(min_length=1)


class DeclarationRegister(StrictModel):
    schema_version: str = "0.1"
    register_id: str = Field(min_length=1)
    task_id: str = Field(min_length=1)
    task_config: str = Field(min_length=1)
    expected_d0_status: CriterionStatus
    expected_missing_count: int = Field(ge=1)
    scope_note: str = Field(min_length=1)
    entries: List[DeclarationRegisterEntry] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_unique_paths(self) -> "DeclarationRegister":
        paths = [entry.path for entry in self.entries]
        if len(paths) != len(set(paths)):
            raise ValueError("declaration register paths must be unique")
        return self


class DeclarationRegisterAudit(StrictModel):
    aligned: bool
    register_id: str
    task_id: str
    task_id_matches: bool
    task_config_matches: bool
    d0_status: CriterionStatus
    expected_d0_status: CriterionStatus
    expected_missing_count: int
    actual_missing_count: int
    registered_count: int
    order_matches: bool
    missing_from_register: List[str] = Field(default_factory=list)
    no_longer_missing: List[str] = Field(default_factory=list)
    classification_counts: Dict[str, int] = Field(default_factory=dict)
    resolution_stage_counts: Dict[str, int] = Field(default_factory=dict)


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
    sensor_acquisition_bandwidth_hz: Optional[float] = Field(
        default=None,
        gt=0,
    )
    reference_acquisition_bandwidth_hz: Optional[float] = Field(
        default=None,
        gt=0,
    )
    sampling_process_id: Optional[str] = Field(default=None, min_length=1)

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


class GeneralizationClaim(StrictModel):
    target_description: Optional[str] = Field(default=None, min_length=1)
    independent_unit: Optional[
        Literal["run", "condition", "sensor_unit"]
    ] = None
    sensor_scope: Optional[Literal["same_sensor", "new_sensors"]] = None
    operating_conditions: Optional[List[str]] = None


class CalibrationClaim(StrictModel):
    sensor_inputs: Optional[List[str]] = None
    reference_outputs: Optional[List[str]] = None
    operating_domain: Optional[List[str]] = None
    operating_conditions: Optional[List[str]] = None
    model_family: Optional[str] = Field(default=None, min_length=1)
    performance_metrics: Optional[List[str]] = None
    generalization: Optional[GeneralizationClaim] = None


class CriterionApplicability(StrictModel):
    applicable: bool
    reason: Optional[str] = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def require_reason_for_exclusion(self) -> "CriterionApplicability":
        if not self.applicable and not self.reason:
            raise ValueError("a non-applicable criterion requires a reason")
        return self


class AssessmentRequirements(StrictModel):
    criteria: Dict[
        Literal["D1", "D2", "D3", "D4", "D5", "D6", "D7"],
        CriterionApplicability,
    ]


class D1Requirements(StrictModel):
    maximum_reference_uncertainty: Dict[str, Optional[float]]
    require_reference_certificate: bool = True
    require_acquisition_bandwidth: bool = True
    require_sampling_process: bool = True
    invalid_observation_policy_id: Optional[str] = Field(
        default=None,
        min_length=1,
    )
    exclusion_record_id: Optional[str] = Field(default=None, min_length=1)
    exclusions_reviewed: Optional[bool] = None

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


class D2ConditionDomain(StrictModel):
    kind: Literal["continuous", "categorical", "fixed"]
    source: Literal["column", "constant"]
    column: Optional[str] = Field(default=None, min_length=1)
    constant: Optional[Union[str, float, int, bool]] = None
    unit: Optional[str] = Field(default=None, min_length=1)
    minimum: Optional[float] = None
    maximum: Optional[float] = None
    grid_points: Optional[int] = Field(default=None, ge=2)
    categories: Optional[List[str]] = None
    claimed_value: Optional[Union[str, float, int, bool]] = None

    @model_validator(mode="after")
    def validate_condition(self) -> "D2ConditionDomain":
        if self.source == "column" and self.column is None:
            raise ValueError("column source requires column")
        if self.source == "constant" and self.constant is None:
            raise ValueError("constant source requires constant")
        if self.kind == "continuous":
            if (
                self.minimum is not None
                and self.maximum is not None
                and self.minimum >= self.maximum
            ):
                raise ValueError("continuous minimum must be smaller than maximum")
        if self.kind == "categorical" and self.categories is not None:
            if (
                not self.categories
                or len(self.categories) != len(set(self.categories))
            ):
                raise ValueError("categorical values must be non-empty and unique")
        return self


class D2RegionInterval(StrictModel):
    minimum: Optional[float] = None
    maximum: Optional[float] = None

    @model_validator(mode="after")
    def validate_interval(self) -> "D2RegionInterval":
        if (
            self.minimum is not None
            and self.maximum is not None
            and self.minimum > self.maximum
        ):
            raise ValueError("excluded-region minimum cannot exceed maximum")
        return self


class D2ExcludedRegion(StrictModel):
    region_id: str = Field(min_length=1)
    reason: str = Field(min_length=1)
    continuous_bounds: Dict[str, D2RegionInterval] = Field(default_factory=dict)
    categorical_values: Dict[str, List[str]] = Field(default_factory=dict)


class D2Requirements(StrictModel):
    axes: List[str] = Field(
        min_length=1,
        description="Ordered reference axes used in the coverage calculation.",
    )
    domain: Dict[str, D2AxisDomain]
    conditions: Optional[Dict[str, D2ConditionDomain]] = None
    excluded_regions: Optional[List[D2ExcludedRegion]] = None
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
    model_specific_test_id: Optional[str] = Field(default=None, min_length=1)
    confounding_review_id: Optional[str] = Field(default=None, min_length=1)
    condition_handling: Optional[
        Literal["held_fixed", "included_in_model", "demonstrated_invariant"]
    ] = None

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
    independent_unit: Literal["run"] = "run"
    dependence_method_id: Optional[str] = Field(default=None, min_length=1)
    stationarity_reviewed: Optional[bool] = None
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
    development_selection_method_id: Optional[str] = Field(
        default=None,
        min_length=1,
    )
    data_use: Optional[D5DataUseEvidence] = None


class D6AxisRequirements(StrictModel):
    maximum_interval_half_width: Optional[float] = Field(default=None, ge=0)
    maximum_rmse: Optional[float] = Field(default=None, ge=0)
    calibrated_force_uncertainty: Optional[float] = Field(default=None, ge=0)
    maximum_calibrated_force_uncertainty: Optional[float] = Field(
        default=None,
        ge=0,
    )


class D6RegionDimension(StrictModel):
    minimum: Optional[float] = None
    maximum: Optional[float] = None
    values: Optional[List[str]] = None

    @model_validator(mode="after")
    def validate_selector(self) -> "D6RegionDimension":
        if (
            self.minimum is not None
            and self.maximum is not None
            and self.minimum > self.maximum
        ):
            raise ValueError("region minimum cannot exceed maximum")
        if self.values is not None and not self.values:
            raise ValueError("region values cannot be empty")
        if (
            self.minimum is None
            and self.maximum is None
            and self.values is None
        ):
            raise ValueError("a region dimension needs bounds or values")
        return self


class D6RegionalAxisRequirements(StrictModel):
    maximum_interval_half_width: Optional[float] = Field(default=None, ge=0)
    maximum_rmse: Optional[float] = Field(default=None, ge=0)


class D6RegionRequirements(StrictModel):
    region_id: str = Field(min_length=1)
    dimensions: Dict[str, D6RegionDimension] = Field(min_length=1)
    minimum_bootstrap_units: Optional[int] = Field(default=None, ge=2)
    axes: Dict[str, D6RegionalAxisRequirements] = Field(default_factory=dict)


class D6Requirements(StrictModel):
    metric: Literal["rmse"]
    confidence_level: Optional[float] = Field(default=None, gt=0, lt=1)
    bootstrap_repetitions: Optional[int] = Field(default=None, ge=100)
    bootstrap_random_seed: Optional[int] = Field(default=None, ge=0)
    bootstrap_unit: Literal["run"]
    minimum_bootstrap_units: Optional[int] = Field(default=None, ge=2)
    uncertainty_method_id: Optional[str] = Field(default=None, min_length=1)
    axes: Dict[str, D6AxisRequirements] = Field(default_factory=dict)
    regions: Optional[List[D6RegionRequirements]] = None

    @model_validator(mode="after")
    def validate_regions(self) -> "D6Requirements":
        if self.regions is not None:
            region_ids = [region.region_id for region in self.regions]
            if len(region_ids) != len(set(region_ids)):
                raise ValueError("D6 region identifiers must be unique")
        return self


EvidenceRole = Literal[
    "raw_data",
    "acquisition_metadata",
    "task_config",
    "sensor_profile",
    "reference_profile",
    "setup_profile",
    "preprocessing_config",
    "exclusion_record",
    "partition_manifest",
    "dependency_lock",
    "criterion_result",
    "diagnostic_figure",
]


class EvidenceFileRecord(StrictModel):
    path: str = Field(min_length=1)
    role: EvidenceRole
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class EvidenceManifest(StrictModel):
    schema_version: str = "0.1"
    package_id: str = Field(min_length=1)
    files: List[EvidenceFileRecord] = Field(min_length=1)
    software_commit: str = Field(min_length=7)
    dependency_snapshot_id: str = Field(min_length=1)
    preprocessing_procedure_id: str = Field(min_length=1)
    exclusion_record_id: str = Field(min_length=1)
    partition_manifest_id: str = Field(min_length=1)
    random_seeds: Dict[str, int] = Field(default_factory=dict)
    criterion_results: Dict[
        Literal["D0", "D1", "D2", "D3", "D4", "D5", "D6"],
        str,
    ]


class D7Requirements(StrictModel):
    evidence_manifest_path: Optional[str] = Field(default=None, min_length=1)
    evidence_package_id: Optional[str] = Field(default=None, min_length=1)
    required_file_roles: Optional[List[EvidenceRole]] = None
    required_criterion_results: Optional[
        List[Literal["D0", "D1", "D2", "D3", "D4", "D5", "D6"]]
    ] = None
    numerical_absolute_tolerance: Optional[float] = Field(default=None, ge=0)
    numerical_relative_tolerance: Optional[float] = Field(default=None, ge=0)


class TaskConfig(StrictModel):
    schema_version: str = "0.1"
    task_id: str
    profiles: ProfileReferences
    dataset_mapping: DatasetMapping
    claim: Optional[CalibrationClaim] = None
    assessment: Optional[AssessmentRequirements] = None
    d1: D1Requirements
    d2: Optional[D2Requirements] = None
    d3: Optional[D3Requirements] = None
    d4: Optional[D4Requirements] = None
    d5: Optional[D5Requirements] = None
    d6: Optional[D6Requirements] = None
    d7: Optional[D7Requirements] = None


class TaskBundle(StrictModel):
    task_path: Path
    task: TaskConfig
    sensor: InstrumentProfile
    reference: InstrumentProfile
    setup: SetupProfile

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)
