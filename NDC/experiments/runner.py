"""Experiment runner for synthetic NDC."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from NDC.config.schema import ExperimentConfig, load_config
from NDC.dynamics.circulation import build_circulation_matrix
from NDC.dynamics.geometry import IdentityGeometry
from NDC.dynamics.integrator import euler_maruyama
from NDC.dynamics.landscape import (
    FlatLandscape,
    GaussianMixture,
    PiecewiseQuadraticWell,
    QuadraticWell,
)
from NDC.dynamics.plasticity import StaticPlasticity
from NDC.dynamics.rvc import LinearGate, NullGate
from NDC.dynamics.scheduler import RegimeScheduler, RegimeSegment
from NDC.drivers.base import Driver
from NDC.drivers.composers import CompositeDriver
from NDC.drivers.file_timeseries import FileTimeseriesDriver
from NDC.drivers.inputs import StepInputDriver
from NDC.drivers.oscillators import SineRhythmDriver
from NDC.drivers.surrogates import PhaseRandomizedDriver, PhaseRandomizedRhythmWrapperDriver
from NDC.io.observation_export import ObservationExportMetadata, export_observations
from NDC.io.oracle_export import OracleExportMetadata, export_oracle
from NDC.io.results import save_results
from NDC.io.trajectory import save_trajectory
from NDC.observation.base import ObservationSeries
from NDC.observation.resampling import resample_series, windowed_mean
from NDC.observation.tiers import LinearObserver, NonlinearObserver, OracleObserver
from NDC.oracle.metrics import compute_oracle_metrics


LANDSCAPE_REGISTRY = {
    "quadratic_well": QuadraticWell,
    "gaussian_mixture": GaussianMixture,
    "flat": FlatLandscape,
    "piecewise_quadratic_well": PiecewiseQuadraticWell,
}

RVC_REGISTRY = {
    "null_gate": NullGate,
    "linear_gate": LinearGate,
}

DRIVER_REGISTRY: dict[str, type[Driver]] = {
    "sine_rhythm": SineRhythmDriver,
    "step_input": StepInputDriver,
    "phase_randomized": PhaseRandomizedDriver,
    "phase_randomized_rhythm": PhaseRandomizedRhythmWrapperDriver,
    "file_timeseries": FileTimeseriesDriver,
}

OBSERVER_REGISTRY = {
    "oracle": OracleObserver,
    "linear": LinearObserver,
    "nonlinear": NonlinearObserver,
}

PLASTICITY_REGISTRY = {
    "static": StaticPlasticity,
}


def _build_circulation(cfg: ExperimentConfig) -> np.ndarray | None:
    if cfg.circulation.name == "none":
        return None
    if cfg.circulation.name == "rotation":
        return build_circulation_matrix(cfg.latent_dim, cfg.circulation.params)
    raise ValueError(f"Unknown circulation {cfg.circulation.name}")


def _build_boundary(cfg: ExperimentConfig) -> dict[str, Any] | None:
    if cfg.boundary.name == "none":
        return None
    return {"name": cfg.boundary.name, "params": dict(cfg.boundary.params)}


def _hash_config_text(raw_text: str) -> str:
    return hashlib.sha256(raw_text.encode("utf-8")).hexdigest()


def _build_landscape(cfg: ExperimentConfig):
    cls = LANDSCAPE_REGISTRY[cfg.landscape.name]
    return cls.from_params(cfg.latent_dim, cfg.landscape.params)


def _build_rvc(cfg: ExperimentConfig):
    cls = RVC_REGISTRY[cfg.rvc.name]
    return cls(**cfg.rvc.params)


def _instantiate_driver(name: str, params: dict[str, Any], latent_dim: int) -> Driver:
    params = dict(params)
    if name == "step_input":
        direction = np.array(params.get("direction", np.ones(latent_dim)), dtype=float)
        params["direction"] = direction
    if name == "file_timeseries":
        params.setdefault("latent_dim", latent_dim)
    return DRIVER_REGISTRY[name](**params)


def _build_driver(cfg: ExperimentConfig, rng: np.random.Generator) -> Driver:
    name = cfg.driver.name
    params = dict(cfg.driver.params)
    if name in {"phase_randomized", "phase_randomized_rhythm"}:
        base_spec = params.get("base_driver")
        if not base_spec or "name" not in base_spec:
            raise ValueError("phase_randomized requires base_driver {name, params}")
        base_driver = _instantiate_driver(
            base_spec["name"], base_spec.get("params", {}), cfg.latent_dim
        )
        t_start = float(params.get("t_start", cfg.time.t_start))
        t_end = float(params.get("t_end", cfg.time.t_end))
        dt = float(params.get("dt", cfg.time.dt_sim))
        if name == "phase_randomized":
            return PhaseRandomizedDriver.from_driver(
                base_driver=base_driver,
                t_start=t_start,
                t_end=t_end,
                dt=dt,
                rng=rng,
            )
        return PhaseRandomizedRhythmWrapperDriver.from_driver(
            base_driver=base_driver,
            t_start=t_start,
            t_end=t_end,
            dt=dt,
            rng=rng,
        )
    return _instantiate_driver(name, params, cfg.latent_dim)


def _build_observer(cfg: ExperimentConfig, rng: np.random.Generator):
    name = cfg.observer.name
    params = dict(cfg.observer.params)
    input_dim = cfg.latent_dim
    output_dim = int(params.get("output_dim", input_dim))
    if name == "oracle":
        return OracleObserver()
    if name == "linear":
        return LinearObserver.from_params(input_dim, output_dim, rng, params)
    if name == "nonlinear":
        return NonlinearObserver.from_params(input_dim, output_dim, rng, params)
    raise ValueError(f"Unknown observer {name}")


def _build_plasticity(cfg: ExperimentConfig):
    cls = PLASTICITY_REGISTRY[cfg.plasticity.name]
    return cls(**cfg.plasticity.params)


def _build_scheduler(cfg: ExperimentConfig) -> RegimeScheduler:
    segments = [
        RegimeSegment(label=seg.label, start=seg.start, end=seg.end)
        for seg in cfg.scheduler.regimes
    ]
    return RegimeScheduler(default_label=cfg.scheduler.default_label, regimes=segments)


def _metadata_from_components(cfg: ExperimentConfig, config_hash: str, landscape, rvc, driver, observer) -> dict[str, Any]:
    observer_params: dict[str, Any] = dict(cfg.observer.params)
    if hasattr(observer, "matrix"):
        observer_params["matrix"] = observer.matrix.tolist()
    return {
        "config_hash": config_hash,
        "seed": cfg.seed,
        "latent_dim": cfg.latent_dim,
        "landscape_class": type(landscape).__name__,
        "landscape_params": landscape.get_params(),
        "rvc_class": type(rvc).__name__,
        "rvc_params": dict(cfg.rvc.params),
        "driver_class": type(driver).__name__,
        "driver_params": dict(cfg.driver.params),
        "observer_tier": observer.tier,
        "observer_params": observer_params,
        "circulation_name": cfg.circulation.name,
        "circulation_params": dict(cfg.circulation.params),
        "boundary_name": cfg.boundary.name,
        "boundary_params": dict(cfg.boundary.params),
        "dt_sim": cfg.time.dt_sim,
        "dt_obs": cfg.time.dt_obs,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def _run_from_cfg(
    cfg: ExperimentConfig,
    config_hash: str,
    output_dir: str | Path | None = None,
    *,
    save_outputs: bool = True,
) -> tuple[dict[str, Any], "Trajectory"]:
    seed_seq = np.random.SeedSequence(cfg.seed)
    rng_integrator, rng_observer, rng_driver = (
        np.random.default_rng(s) for s in seed_seq.spawn(3)
    )

    landscape = _build_landscape(cfg)
    rvc = _build_rvc(cfg)
    driver = _build_driver(cfg, rng_driver)
    observer = _build_observer(cfg, rng_observer)
    _ = _build_plasticity(cfg)
    circulation = _build_circulation(cfg)
    boundary = _build_boundary(cfg)
    scheduler = _build_scheduler(cfg)

    if cfg.initial_state is None:
        x0 = np.zeros(cfg.latent_dim, dtype=float)
    else:
        x0 = np.array(cfg.initial_state, dtype=float)
    traj = euler_maruyama(
        landscape=landscape,
        rvc=rvc,
        driver=driver,
        x0=x0,
        t_start=cfg.time.t_start,
        t_end=cfg.time.t_end,
        dt=cfg.time.dt_sim,
        rng=rng_integrator,
        geometry=IdentityGeometry(),
        circulation=circulation,
        boundary=boundary,
    )

    obs_times, obs_states = resample_series(
        traj.times, traj.states, cfg.time.dt_obs, method=cfg.time.downsample_method
    )
    obs_values = np.vstack(
        [observer.observe(x, t, rng_observer) for x, t in zip(obs_states, obs_times)]
    )
    obs_series = ObservationSeries(times=obs_times, values=obs_values)

    win_times, win_values = windowed_mean(
        obs_series.times, obs_series.values, cfg.time.window, cfg.time.step
    )

    oracle_metrics = compute_oracle_metrics(traj)
    metadata = _metadata_from_components(cfg, config_hash, landscape, rvc, driver, observer)
    if traj.metadata and "boundary" in traj.metadata:
        metadata["boundary_stats"] = traj.metadata["boundary"]
    traj.metadata = metadata

    out_dir = Path(output_dir or cfg.output.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = cfg.output.file_prefix
    traj_path = out_dir / f"{prefix}_trajectory.h5"
    results_path = out_dir / f"{prefix}_results.json"

    a_true = None
    if hasattr(landscape, "jacobian"):
        try:
            a_true = np.stack(
                [landscape.jacobian(x, t) for x, t in zip(traj.states, traj.times)]
            )
        except Exception:
            a_true = None

    if save_outputs:
        save_trajectory(
            traj_path, traj, metadata, windowed_times=win_times, windowed_values=win_values
        )
        obs_path = out_dir / f"{prefix}_observations.h5"
        export_observations(
            obs_path,
            times=obs_series.times,
            values=obs_series.values,
            metadata=ObservationExportMetadata(
                dt_sim=cfg.time.dt_sim,
                dt_obs=cfg.time.dt_obs,
                window=cfg.time.window,
                step=cfg.time.step,
                downsample_method=cfg.time.downsample_method,
                observer_tier=observer.tier,
                observer_params=metadata["observer_params"],
                seed=cfg.seed,
                config_hash=config_hash,
            ),
            windowed_times=win_times,
            windowed_values=win_values,
        )
        if cfg.output.save_oracle:
            oracle_path = out_dir / f"{prefix}_oracle.h5"
            regimes = np.array([scheduler.regime_at(t) for t in traj.times], dtype=object)
            export_oracle(
                oracle_path,
                times=traj.times,
                states=traj.states,
                metadata=OracleExportMetadata(
                    dt_sim=cfg.time.dt_sim,
                    t_start=cfg.time.t_start,
                    t_end=cfg.time.t_end,
                    latent_dim=cfg.latent_dim,
                    seed=cfg.seed,
                    config_hash=config_hash,
                    extra=metadata,
                ),
                rhythm=traj.rhythm,
                control=traj.control,
                regimes=regimes,
                boundary_hit_flags=traj.boundary_hit_flags,
                a_true=a_true,
            )

    results = {
        "metadata": metadata,
        "oracle_metrics": oracle_metrics,
        "observation_summary": {
            "obs_points": int(obs_series.times.shape[0]),
            "obs_dim": int(obs_series.values.shape[1]),
            "window_points": int(win_times.shape[0]),
        },
        "windowed_features": {
            "times": win_times.tolist(),
            "values": win_values.tolist(),
        },
    }
    if save_outputs:
        save_results(results_path, results)
    return results, traj


def run_experiment(config_path: str | Path, output_dir: str | Path | None = None) -> dict[str, Any]:
    """Run an NDC experiment from a configuration file.

    Args:
        config_path: Path to the YAML configuration file.
        output_dir: Optional override for the output directory.

    Returns:
        A dictionary containing experiment results and metadata.
    """
    config_path = Path(config_path)
    raw_text = config_path.read_text(encoding="utf-8")
    cfg = load_config(config_path)
    results, _ = _run_from_cfg(cfg, _hash_config_text(raw_text), output_dir=output_dir)
    return results


def run_experiment_from_config(config: ExperimentConfig, output_dir: str | Path | None = None) -> dict[str, Any]:
    """Run an NDC experiment from an ExperimentConfig object.

    Args:
        config: The experiment configuration object.
        output_dir: Optional override for the output directory.

    Returns:
        A dictionary containing experiment results and metadata.
    """
    config_hash = _hash_config_text(json.dumps(config.model_dump(), sort_keys=True))
    results, _ = _run_from_cfg(config, config_hash, output_dir=output_dir)
    return results


def run_experiment_with_traj(
    config: ExperimentConfig,
    output_dir: str | Path | None = None,
    *,
    save_outputs: bool = True,
) -> tuple[dict[str, Any], "Trajectory"]:
    config_hash = _hash_config_text(json.dumps(config.model_dump(), sort_keys=True))
    return _run_from_cfg(
        config, config_hash, output_dir=output_dir, save_outputs=save_outputs
    )

