"""Pydantic schema for synthetic NDC experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class TimeConfig(BaseModel):
    """Configuration for simulation and observation time parameters.

    Attributes:
        t_start: Start time of the simulation.
        t_end: End time of the simulation.
        dt_sim: Simulation time step.
        dt_obs: Observation (sampling) time step.
        window: Analysis window size.
        step: Analysis step size.
        downsample_method: Method for downsampling ("interp", "nearest", or "decimate").
    """
    model_config = ConfigDict(extra="forbid")

    t_start: float = 0.0
    t_end: float = 10.0
    dt_sim: float = 0.001
    dt_obs: float = 0.01
    window: float = 0.5
    step: float = 0.25
    downsample_method: str = "interp"

    @model_validator(mode="after")
    def _validate_time(self) -> "TimeConfig":
        if self.t_end <= self.t_start:
            raise ValueError("t_end must be greater than t_start")
        if self.dt_sim <= 0 or self.dt_obs <= 0:
            raise ValueError("dt_sim and dt_obs must be positive")
        if self.window <= 0 or self.step <= 0:
            raise ValueError("window and step must be positive")
        if self.downsample_method not in {"interp", "nearest", "decimate"}:
            raise ValueError("downsample_method must be interp, nearest, or decimate")
        return self


class LandscapeConfig(BaseModel):
    """Configuration for the energy landscape.

    Attributes:
        name: Name of the landscape model.
        params: Parameters for the landscape model.
    """
    model_config = ConfigDict(extra="forbid")

    name: str = "quadratic_well"
    params: dict[str, Any] = Field(default_factory=dict)


class RVCConfig(BaseModel):
    """Configuration for Rhythmic Variance Control (RVC).

    Attributes:
        name: Name of the RVC gate model.
        params: Parameters for the RVC gate model.
    """
    model_config = ConfigDict(extra="forbid")

    name: str = "null_gate"
    params: dict[str, Any] = Field(default_factory=dict)


class DriverConfig(BaseModel):
    """Configuration for the external driver.

    Attributes:
        name: Name of the driver model.
        params: Parameters for the driver model.
    """
    model_config = ConfigDict(extra="forbid")

    name: str = "sine_rhythm"
    params: dict[str, Any] = Field(default_factory=dict)


class ObserverConfig(BaseModel):
    """Configuration for the observer model.

    Attributes:
        name: Name of the observer model.
        params: Parameters for the observer model.
    """
    model_config = ConfigDict(extra="forbid")

    name: str = "oracle"
    params: dict[str, Any] = Field(default_factory=dict)


class PlasticityConfig(BaseModel):
    """Configuration for latent space plasticity.

    Attributes:
        name: Name of the plasticity model.
        params: Parameters for the plasticity model.
    """
    model_config = ConfigDict(extra="forbid")

    name: str = "static"
    params: dict[str, Any] = Field(default_factory=dict)


class CirculationConfig(BaseModel):
    """Configuration for drift circulation.

    Attributes:
        name: Name of the circulation model ("none" or "rotation").
        params: Parameters for the circulation model.
    """
    model_config = ConfigDict(extra="forbid")

    name: str = "none"
    params: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_name(self) -> "CirculationConfig":
        if self.name not in {"none", "rotation"}:
            raise ValueError("circulation name must be none or rotation")
        return self


class BoundaryConfig(BaseModel):
    """Configuration for latent space boundaries.

    Attributes:
        name: Name of the boundary model ("none" or "reflecting_box").
        params: Parameters for the boundary model.
    """
    model_config = ConfigDict(extra="forbid")

    name: str = "none"
    params: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_name(self) -> "BoundaryConfig":
        if self.name not in {"none", "reflecting_box"}:
            raise ValueError("boundary name must be none or reflecting_box")
        return self


class RegimeSegment(BaseModel):
    """A segment of time with a specific regime label.

    Attributes:
        label: Label for the regime (e.g., "wake", "sleep").
        start: Start time of the segment.
        end: End time of the segment.
    """
    model_config = ConfigDict(extra="forbid")

    label: str
    start: float
    end: float

    @model_validator(mode="after")
    def _validate_segment(self) -> "RegimeSegment":
        if self.end <= self.start:
            raise ValueError("RegimeSegment end must be greater than start")
        return self


class SchedulerConfig(BaseModel):
    """Configuration for the regime scheduler.

    Attributes:
        default_label: Default regime label.
        regimes: List of regime segments.
    """
    model_config = ConfigDict(extra="forbid")

    default_label: str = "wake"
    regimes: list[RegimeSegment] = Field(default_factory=list)


class OutputConfig(BaseModel):
    """Configuration for experiment output.

    Attributes:
        output_dir: Directory to save results.
        file_prefix: Prefix for result filenames.
        save_oracle: Whether to save oracle (latent) data.
    """
    model_config = ConfigDict(extra="forbid")

    output_dir: str = "outputs"
    file_prefix: str = "run"
    save_oracle: bool = False


class ExperimentConfig(BaseModel):
    """Full configuration for a synthetic NDC experiment.

    Attributes:
        seed: Random seed for reproducibility.
        latent_dim: Dimension of the latent space.
        initial_state: Optional initial state vector.
        time: Timing configuration.
        landscape: Landscape configuration.
        rvc: Rhythmic Variance Control configuration.
        driver: External driver configuration.
        observer: Observer configuration.
        plasticity: Plasticity configuration.
        circulation: Circulation configuration.
        boundary: Boundary configuration.
        scheduler: Regime scheduler configuration.
        output: Output configuration.
        notes: Additional metadata for the experiment.
    """
    model_config = ConfigDict(extra="forbid")

    seed: int = 0
    latent_dim: int = 10
    initial_state: list[float] | None = None
    time: TimeConfig = Field(default_factory=TimeConfig)
    landscape: LandscapeConfig = Field(default_factory=LandscapeConfig)
    rvc: RVCConfig = Field(default_factory=RVCConfig)
    driver: DriverConfig = Field(default_factory=DriverConfig)
    observer: ObserverConfig = Field(default_factory=ObserverConfig)
    plasticity: PlasticityConfig = Field(default_factory=PlasticityConfig)
    circulation: CirculationConfig = Field(default_factory=CirculationConfig)
    boundary: BoundaryConfig = Field(default_factory=BoundaryConfig)
    scheduler: SchedulerConfig = Field(default_factory=SchedulerConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    notes: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_dim(self) -> "ExperimentConfig":
        if self.latent_dim <= 0:
            raise ValueError("latent_dim must be positive")
        if self.initial_state is not None and len(self.initial_state) != self.latent_dim:
            raise ValueError("initial_state length must match latent_dim")
        return self


def load_config(path: str | Path) -> ExperimentConfig:
    """Load an experiment configuration from a YAML file.

    Args:
        path: Path to the YAML configuration file.

    Returns:
        The validated ExperimentConfig object.
    """
    config_path = Path(path)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    return ExperimentConfig.model_validate(raw)

