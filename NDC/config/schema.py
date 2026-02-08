"""Pydantic schema for synthetic NDC experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class TimeConfig(BaseModel):
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
    model_config = ConfigDict(extra="forbid")

    name: str = "quadratic_well"
    params: dict[str, Any] = Field(default_factory=dict)


class RVCConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = "null_gate"
    params: dict[str, Any] = Field(default_factory=dict)


class DriverConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = "sine_rhythm"
    params: dict[str, Any] = Field(default_factory=dict)


class ObserverConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = "oracle"
    params: dict[str, Any] = Field(default_factory=dict)


class PlasticityConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = "static"
    params: dict[str, Any] = Field(default_factory=dict)


class CirculationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = "none"
    params: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_name(self) -> "CirculationConfig":
        if self.name not in {"none", "rotation"}:
            raise ValueError("circulation name must be none or rotation")
        return self


class BoundaryConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = "none"
    params: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_name(self) -> "BoundaryConfig":
        if self.name not in {"none", "reflecting_box"}:
            raise ValueError("boundary name must be none or reflecting_box")
        return self


class RegimeSegment(BaseModel):
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
    model_config = ConfigDict(extra="forbid")

    default_label: str = "wake"
    regimes: list[RegimeSegment] = Field(default_factory=list)


class OutputConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    output_dir: str = "outputs"
    file_prefix: str = "run"
    save_oracle: bool = False


class ExperimentConfig(BaseModel):
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
    """Load an experiment config from YAML."""
    config_path = Path(path)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    return ExperimentConfig.model_validate(raw)

