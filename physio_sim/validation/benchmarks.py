from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from pydantic import BaseModel, Field

from physio_sim.config import CompoundConfig, configure_random_seed, load_subject
from physio_sim.pbpk.solver import simulate
from physio_sim.utils.metrics import auc_trapezoid, cmax_tmax
from physio_sim.validation.report import build_markdown_report


class BenchmarkConfig(BaseModel):
    name: str
    dataset: str
    subject_file: str
    dose_mg: float = Field(gt=0)
    route: Literal["oral", "iv"]
    t_end_h: float = Field(gt=0)
    dt_out_h: float = Field(gt=0)
    seed: int | None = None
    compound: CompoundConfig


@dataclass(frozen=True)
class MetricResult:
    auc_observed: float
    auc_simulated: float
    auc_relative_error: float
    cmax_observed: float
    cmax_simulated: float
    cmax_relative_error: float
    tmax_observed_h: float
    tmax_simulated_h: float
    tmax_abs_error_h: float
    rmse_mg_per_L: float


def _load_acceptance(path: Path) -> dict[str, float]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    default = payload["default"]
    return {
        "auc_relative_error_max": float(default["auc_relative_error_max"]),
        "cmax_relative_error_max": float(default["cmax_relative_error_max"]),
        "tmax_abs_error_h_max": float(default["tmax_abs_error_h_max"]),
    }


def _compute_metrics(observed: pd.DataFrame, simulated: pd.DataFrame) -> MetricResult:
    obs_t = observed["time_h"].to_numpy(dtype=float)
    obs_c = observed["C_plasma_mg_per_L"].to_numpy(dtype=float)
    sim_interp = np.interp(obs_t, simulated["time_h"], simulated["C_plasma_mg_per_L"])

    auc_obs = float(auc_trapezoid(obs_t, obs_c))
    auc_sim = float(auc_trapezoid(obs_t, sim_interp))
    cmax_obs, tmax_obs = cmax_tmax(obs_t, obs_c)
    cmax_sim, tmax_sim = cmax_tmax(obs_t, sim_interp)

    auc_rel = abs(auc_sim - auc_obs) / auc_obs if auc_obs > 0 else 0.0
    cmax_rel = abs(cmax_sim - cmax_obs) / cmax_obs if cmax_obs > 0 else 0.0
    tmax_abs = abs(tmax_sim - tmax_obs)
    rmse = float(np.sqrt(np.mean(np.square(sim_interp - obs_c))))

    return MetricResult(
        auc_observed=auc_obs,
        auc_simulated=auc_sim,
        auc_relative_error=float(auc_rel),
        cmax_observed=float(cmax_obs),
        cmax_simulated=float(cmax_sim),
        cmax_relative_error=float(cmax_rel),
        tmax_observed_h=float(tmax_obs),
        tmax_simulated_h=float(tmax_sim),
        tmax_abs_error_h=float(tmax_abs),
        rmse_mg_per_L=rmse,
    )


ROOT_DIR = Path(__file__).resolve().parents[2]


def _resolve_path(path_str: str, suite_root: Path) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    candidates = (suite_root / path, suite_root.parent / path, ROOT_DIR / path)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return ROOT_DIR / path


def _load_config(config_path: Path) -> BenchmarkConfig:
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        msg = f"Invalid benchmark config file: {config_path}"
        raise ValueError(msg)
    return BenchmarkConfig.model_validate(payload)


def run_benchmark_suite(suite_dir: Path, out_dir: Path) -> bool:
    suite_path = suite_dir.resolve()
    repo_root = suite_path.parent
    configs_dir = suite_path / "configs"
    acceptance_path = suite_path / "expected" / "acceptance.json"
    acceptance = _load_acceptance(acceptance_path)

    out_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []

    for config_path in sorted(configs_dir.glob("*.yaml")):
        config = _load_config(config_path)
        configure_random_seed(config.seed)
        subject = load_subject(_resolve_path(config.subject_file, repo_root))
        observed = pd.read_csv(_resolve_path(config.dataset, repo_root))
        simulated = simulate(
            subject,
            config.compound,
            dose_mg=config.dose_mg,
            route=config.route,
            t_end_h=config.t_end_h,
            dt_out_h=config.dt_out_h,
        ).timecourse
        metrics = _compute_metrics(observed, simulated)
        passes = {
            "auc": metrics.auc_relative_error <= acceptance["auc_relative_error_max"],
            "cmax": metrics.cmax_relative_error <= acceptance["cmax_relative_error_max"],
            "tmax": metrics.tmax_abs_error_h <= acceptance["tmax_abs_error_h_max"],
        }
        all_pass = all(passes.values())

        drug_out = out_dir / config.name
        drug_out.mkdir(parents=True, exist_ok=True)

        obs_t = observed["time_h"].to_numpy(dtype=float)
        sim_interp = np.interp(obs_t, simulated["time_h"], simulated["C_plasma_mg_per_L"])
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(
            simulated["time_h"], simulated["C_plasma_mg_per_L"], label="Simulated", color="tab:blue"
        )
        ax.scatter(obs_t, observed["C_plasma_mg_per_L"], label="Observed", color="black", s=16)
        ax.plot(obs_t, sim_interp, color="tab:orange", alpha=0.7, linestyle="--", label="Sim@obs")
        ax.set_xlabel("Time (h)")
        ax.set_ylabel("Plasma concentration (mg/L)")
        ax.set_title(f"{config.name} benchmark overlay")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(drug_out / "overlay.png", dpi=150)
        plt.close(fig)

        metrics_payload = {
            "drug": config.name,
            "seed": config.seed,
            "metrics": metrics.__dict__,
            "thresholds": acceptance,
            "passes": passes,
            "pass": all_pass,
        }
        (drug_out / "metrics.json").write_text(
            json.dumps(metrics_payload, indent=2), encoding="utf-8"
        )
        results.append(metrics_payload)

    summary = {
        "suite": str(suite_dir),
        "thresholds": acceptance,
        "overall_pass": all(item["pass"] for item in results),
        "results": results,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out_dir / "report.md").write_text(build_markdown_report(summary), encoding="utf-8")
    return bool(summary["overall_pass"])
