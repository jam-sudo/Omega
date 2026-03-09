"""Benchmark validation suite for multi-compound PBPK model verification.

Loads benchmark configurations (YAML), runs simulations against observed
clinical data (CSV), computes accuracy metrics (AUC/Cmax relative error,
Tmax absolute error, RMSE), and produces per-drug overlay plots and a
summary report.

Adapted for the v0.7 34-state WholeBodyPBPK engine.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from omega_pbpk._compat import np_trapz
from omega_pbpk.config import load_compound
from omega_pbpk.core.body import WholeBodyPBPK
from omega_pbpk.drugs.drug import Drug

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MetricResult:
    """Accuracy metrics for a single benchmark drug."""

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


def _load_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load benchmark CSV with columns: time_h, C_plasma_mg_per_L, [std_mg_per_L]."""
    data = np.genfromtxt(path, delimiter=",", skip_header=1)
    return data[:, 0].astype(np.float64), data[:, 1].astype(np.float64)


def _auc_trapezoid(time: np.ndarray, conc: np.ndarray) -> float:
    """Trapezoidal AUC."""
    return float(np_trapz(conc, time))


def _cmax_tmax(time: np.ndarray, conc: np.ndarray) -> tuple[float, float]:
    """Find Cmax and Tmax."""
    idx = int(np.argmax(conc))
    return float(conc[idx]), float(time[idx])


def _load_acceptance(path: Path) -> dict[str, float]:
    """Load acceptance criteria JSON (default thresholds)."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    default = payload["default"]
    return {
        "auc_relative_error_max": float(default["auc_relative_error_max"]),
        "cmax_relative_error_max": float(default["cmax_relative_error_max"]),
        "tmax_abs_error_h_max": float(default["tmax_abs_error_h_max"]),
    }


def _load_acceptance_for_drug(path: Path, drug_name: str) -> dict[str, float]:
    """Load acceptance criteria with per-drug overrides."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    default = payload["default"]
    result = {
        "auc_relative_error_max": float(default["auc_relative_error_max"]),
        "cmax_relative_error_max": float(default["cmax_relative_error_max"]),
        "tmax_abs_error_h_max": float(default["tmax_abs_error_h_max"]),
    }
    drug_overrides = payload.get("drugs", {}).get(drug_name, {})
    for key, value in drug_overrides.items():
        if key in result:
            result[key] = float(value)
    return result


def _compute_metrics(
    obs_time: np.ndarray,
    obs_conc: np.ndarray,
    sim_time: np.ndarray,
    sim_conc: np.ndarray,
) -> MetricResult:
    """Compute PK accuracy metrics (sim interpolated onto obs timepoints)."""
    sim_interp = np.interp(obs_time, sim_time, sim_conc)

    auc_obs = _auc_trapezoid(obs_time, obs_conc)
    auc_sim = _auc_trapezoid(obs_time, sim_interp)
    cmax_obs, tmax_obs = _cmax_tmax(obs_time, obs_conc)
    cmax_sim, tmax_sim = _cmax_tmax(obs_time, sim_interp)

    auc_rel = abs(auc_sim - auc_obs) / max(auc_obs, 1e-12)
    cmax_rel = abs(cmax_sim - cmax_obs) / max(cmax_obs, 1e-12)
    tmax_abs = abs(tmax_sim - tmax_obs)
    rmse = float(np.sqrt(np.mean(np.square(sim_interp - obs_conc))))

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


def _load_drug_from_benchmark(config: dict[str, Any], repo_root: Path) -> Drug:
    """Load Drug from benchmark config: either from compound_file or inline."""
    if "compound_file" in config:
        compound_path = repo_root / config["compound_file"]
        return load_compound(compound_path)

    # Inline compound definition
    clearance = config.get("clearance", {})
    absorption = config.get("absorption", {})
    distribution = config.get("distribution", {})

    return Drug(
        name=config.get("name", "unknown"),
        mw=float(config.get("mw", 300.0)),
        logP=float(config.get("logP", 2.0)),
        pka=config.get("pka", [7.0]),
        drug_type=config.get("drug_type", "neutral"),
        fup=float(config.get("fup", 0.5)),
        rbp=float(config.get("rbp", 1.0)),
        clint=clearance.get("clint_uL_min_pmol", {}),
        fm=clearance.get("fm", {}),
        clint_hepatic_L_per_h=float(clearance.get("clint_hepatic_L_per_h", 0.0)),
        clint_gut_L_per_h=float(clearance.get("clint_gut_L_per_h", 0.0)),
        peff=float(absorption.get("peff", 1.0)),
        solubility_mg_mL=float(absorption.get("solubility_mg_mL", 1.0)),
        kp=distribution.get("kp", {}),
        permeability_limited=distribution.get("permeability_limited", {}),
        gut_clint_multiplier=float(config.get("gut_clint_multiplier", 1.0)),
    )


def _generate_overlay_plot(
    sim_time: np.ndarray,
    sim_conc: np.ndarray,
    obs_time: np.ndarray,
    obs_conc: np.ndarray,
    drug_name: str,
    save_path: Path,
) -> None:
    """Generate overlay plot of simulated vs observed concentrations."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available; skipping overlay plot for %s", drug_name)
        return

    sim_interp = np.interp(obs_time, sim_time, sim_conc)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(sim_time, sim_conc, label="Simulated", color="tab:blue", linewidth=1.5)
    ax.scatter(obs_time, obs_conc, label="Observed", color="black", s=20, zorder=5)
    ax.plot(
        obs_time,
        sim_interp,
        color="tab:orange",
        alpha=0.7,
        linestyle="--",
        label="Sim@obs",
        linewidth=1.0,
    )
    ax.set_xlabel("Time (h)")
    ax.set_ylabel("Plasma concentration (mg/L)")
    ax.set_title(f"{drug_name} — benchmark overlay")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def _build_report(summary: dict[str, Any]) -> str:
    """Build a markdown report from benchmark results."""
    lines = [
        "# PBPK Benchmark Validation Report",
        "",
        f"**Overall**: {'PASS' if summary['overall_pass'] else 'FAIL'}",
        "",
        "| Drug | AUC RE | Cmax RE | Tmax AE (h) | RMSE | Pass |",
        "|------|--------|---------|-------------|------|------|",
    ]
    for r in summary["results"]:
        m = r["metrics"]
        p = "PASS" if r["pass"] else "FAIL"
        lines.append(
            f"| {r['drug']:12s} | {m['auc_relative_error']:.3f} "
            f"| {m['cmax_relative_error']:.3f} "
            f"| {m['tmax_abs_error_h']:.2f} "
            f"| {m['rmse_mg_per_L']:.4f} | {p} |"
        )
    lines.extend(["", f"Thresholds: {summary['thresholds']}"])
    return "\n".join(lines)


def run_benchmark_suite(
    suite_dir: Path | str,
    out_dir: Path | str,
    body_weight: float = 70.0,
) -> dict[str, Any]:
    """Run the full benchmark validation suite.

    Args:
        suite_dir: Path to benchmark directory (contains configs/, datasets/, expected/).
        out_dir: Output directory for results.
        body_weight: Subject body weight (kg).

    Returns:
        Summary dict with overall_pass, per-drug results, and metrics.
    """
    suite_dir = Path(suite_dir)
    out_dir = Path(out_dir)
    repo_root = Path.cwd()

    configs_dir = suite_dir / "configs"
    acceptance_path = suite_dir / "expected" / "acceptance.json"
    acceptance = _load_acceptance(acceptance_path)

    out_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []

    for config_path in sorted(configs_dir.glob("*.yaml")):
        config_raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if not isinstance(config_raw, dict):
            logger.warning("Skipping invalid config: %s", config_path)
            continue

        drug_name = config_raw.get("name", config_path.stem)
        dose_mg = float(config_raw["dose_mg"])
        route = config_raw.get("route", "oral")
        t_end_h = float(config_raw.get("t_end_h", 24.0))

        # Load observed data
        dataset_path = repo_root / config_raw["dataset"]
        obs_time, obs_conc = _load_csv(dataset_path)

        # Load drug
        drug = _load_drug_from_benchmark(config_raw, repo_root)

        # Run simulation
        model = WholeBodyPBPK(drug, body_weight=body_weight)
        if route == "iv":
            model.setup_iv(dose_mg)
        else:
            model.setup_oral(dose_mg)

        result = model.simulate(t_end_h=t_end_h)
        sim_time = result.time_h.astype(np.float64)
        sim_conc = result.plasma_concentration().astype(np.float64)

        # Compute metrics
        metrics = _compute_metrics(obs_time, obs_conc, sim_time, sim_conc)

        # Get per-drug acceptance thresholds (with overrides from acceptance.json)
        drug_acceptance = _load_acceptance_for_drug(acceptance_path, drug_name)

        passes = {
            "auc": metrics.auc_relative_error <= drug_acceptance["auc_relative_error_max"],
            "cmax": metrics.cmax_relative_error <= drug_acceptance["cmax_relative_error_max"],
            "tmax": metrics.tmax_abs_error_h <= drug_acceptance["tmax_abs_error_h_max"],
        }
        all_pass = all(passes.values())

        # Output per-drug results
        drug_out = out_dir / drug_name
        drug_out.mkdir(parents=True, exist_ok=True)

        _generate_overlay_plot(
            sim_time, sim_conc, obs_time, obs_conc, drug_name, drug_out / "overlay.png"
        )

        metrics_dict = {
            "auc_observed": metrics.auc_observed,
            "auc_simulated": metrics.auc_simulated,
            "auc_relative_error": metrics.auc_relative_error,
            "cmax_observed": metrics.cmax_observed,
            "cmax_simulated": metrics.cmax_simulated,
            "cmax_relative_error": metrics.cmax_relative_error,
            "tmax_observed_h": metrics.tmax_observed_h,
            "tmax_simulated_h": metrics.tmax_simulated_h,
            "tmax_abs_error_h": metrics.tmax_abs_error_h,
            "rmse_mg_per_L": metrics.rmse_mg_per_L,
        }

        entry = {
            "drug": drug_name,
            "metrics": metrics_dict,
            "thresholds": acceptance,
            "passes": passes,
            "pass": all_pass,
        }
        (drug_out / "metrics.json").write_text(json.dumps(entry, indent=2), encoding="utf-8")
        results.append(entry)

        status = "PASS" if all_pass else "FAIL"
        logger.info(
            "Benchmark %s: %s (AUC RE=%.3f, Cmax RE=%.3f, Tmax AE=%.2fh)",
            drug_name,
            status,
            metrics.auc_relative_error,
            metrics.cmax_relative_error,
            metrics.tmax_abs_error_h,
        )

    summary: dict[str, Any] = {
        "suite": str(suite_dir),
        "thresholds": acceptance,
        "overall_pass": all(item["pass"] for item in results) if results else False,
        "n_drugs": len(results),
        "n_pass": sum(1 for r in results if r["pass"]),
        "results": results,
    }

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out_dir / "report.md").write_text(_build_report(summary), encoding="utf-8")

    return summary


__all__ = ["MetricResult", "run_benchmark_suite"]
