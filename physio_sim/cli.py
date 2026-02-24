from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import typer
from pydantic import ValidationError

from physio_sim.calibration import run_mh_calibration
from physio_sim.config import (
    configure_random_seed,
    friendly_validation_error,
    load_compound,
    load_qsp,
    load_subject,
)
from physio_sim.pbpk.solver import simulate
from physio_sim.qsp.registry import list_qsp_models
from physio_sim.utils.io import ensure_dir, file_sha256, write_csv, write_json
from physio_sim.utils.metrics import auc_trapezoid, cmax_tmax, effect_summary
from physio_sim.utils.profile import run_with_profile
from physio_sim.validation import mass_balance_check, physiologic_sanity_check

app = typer.Typer(help="PBPK-like + PD simulation CLI")
LOGGER = logging.getLogger("physio_sim")


def _configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s %(name)s: %(message)s", force=True)


@app.callback()
def main(verbose: bool = typer.Option(False, "--verbose", help="Enable debug logging")) -> None:
    _configure_logging(verbose)


@app.command("simulate")
def simulate_cmd(
    compound: Path = typer.Option(..., exists=True, help="Compound YAML path"),
    subject: Path = typer.Option(
        Path("examples/subject_default.yaml"),
        exists=True,
        help="Subject YAML path",
    ),
    dose_mg: float = typer.Option(..., help="Dose in mg"),
    route: str = typer.Option("oral", help="oral|iv"),
    t_end_h: float = typer.Option(24.0, help="End time in hours"),
    dt_out_h: float = typer.Option(0.1, help="Output step in hours"),
    out: Path = typer.Option(Path("outputs/run"), help="Output directory"),
    validate: bool = typer.Option(
        False,
        "--validate",
        help="Run physiological and mass-balance checks",
    ),
    profile: bool = typer.Option(False, "--profile", help="Write cProfile stats to profile.txt"),
    seed: int | None = typer.Option(None, help="Global random seed"),
    qsp_model: str | None = typer.Option(None, help="Enable QSP model (e.g. turnover)"),
    qsp_config: Path | None = typer.Option(None, exists=True, help="QSP YAML config path"),
    qsp_mode: str = typer.Option("posthoc", help="posthoc|coupled"),
) -> None:
    configure_random_seed(seed)
    try:
        subject_cfg = load_subject(subject)
        compound_cfg = load_compound(compound)
    except ValidationError as exc:
        LOGGER.error("Validation error:\n%s", friendly_validation_error(exc))
        raise typer.Exit(code=1) from exc

    qsp_cfg = None
    if qsp_model is not None or qsp_config is not None:
        if qsp_model is None:
            msg = "--qsp-model is required when using QSP"
            raise typer.BadParameter(msg)
        if qsp_config is None:
            msg = "--qsp-config is required when using QSP"
            raise typer.BadParameter(msg)
        qsp_cfg = load_qsp(qsp_config)
        if qsp_cfg.model != qsp_model:
            msg = f"QSP config model ({qsp_cfg.model}) does not match --qsp-model ({qsp_model})"
            raise typer.BadParameter(msg)
        if qsp_mode not in {"posthoc", "coupled"}:
            msg = "--qsp-mode must be one of posthoc|coupled"
            raise typer.BadParameter(msg)
        if qsp_model not in list_qsp_models():
            msg = f"Unknown QSP model '{qsp_model}'"
            raise typer.BadParameter(msg)

    if validate:
        physio_warnings = physiologic_sanity_check(subject_cfg, compound_cfg)
        for warning_msg in physio_warnings:
            LOGGER.warning("[validation] %s", warning_msg)

    def _run() -> pd.DataFrame:
        result = simulate(
            subject_cfg,
            compound_cfg,
            dose_mg=dose_mg,
            route=route,
            t_end_h=t_end_h,
            dt_out_h=dt_out_h,
            qsp=qsp_cfg,
            qsp_mode=qsp_mode,
        )
        return result.timecourse

    df = run_with_profile(_run, out / "profile.txt") if profile else _run()
    ensure_dir(out)

    write_csv(df, out / "timecourse.csv")
    cmax, tmax = cmax_tmax(df["time_h"].to_numpy(), df["C_plasma_mg_per_L"].to_numpy())
    auc = auc_trapezoid(df["time_h"].to_numpy(), df["C_plasma_mg_per_L"].to_numpy())
    e_summary = effect_summary(df["time_h"].to_numpy(), df["Effect"].to_numpy())

    validation_warnings: list[str] = []
    if validate:
        validation_warnings.extend(mass_balance_check(df, dose_mg=dose_mg))

    summary: dict[str, object] = {
        "compound_file": str(compound),
        "subject_file": str(subject),
        "compound_sha256": file_sha256(compound),
        "subject_sha256": file_sha256(subject),
        "dose_mg": dose_mg,
        "route": route,
        "t_end_h": t_end_h,
        "seed": seed,
        "Cmax_mg_per_L": cmax,
        "Tmax_h": tmax,
        "AUC0_tend_mg_h_per_L": auc,
        **e_summary,
    }
    if qsp_cfg is not None and "B_biomarker" in df.columns:
        b_series = df["B_biomarker"]
        idx_max = int(b_series.idxmax())
        summary.update(
            {
                "qsp_model": qsp_cfg.model,
                "qsp_mode": qsp_mode,
                "Bmax": float(b_series.max()),
                "t_Bmax": float(df["time_h"].iloc[idx_max]),
                "Bend": float(b_series.iloc[-1]),
            }
        )
    if validate:
        summary["validation_warnings"] = validation_warnings
    write_json(summary, out / "summary.json")

    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    axes[0].plot(df["time_h"], df["C_plasma_mg_per_L"], color="tab:blue")
    axes[0].set_ylabel("Plasma Conc (mg/L)")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(df["time_h"], df["Effect"], color="tab:orange")
    axes[1].set_ylabel("Effect")
    axes[1].set_xlabel("Time (h)")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out / "plots.png", dpi=150)
    plt.close(fig)


@app.command("calibrate")
def calibrate_cmd(
    data: Path = typer.Option(..., exists=True, help="Observed CSV with time_h,C_plasma_mg_per_L"),
    compound: Path = typer.Option(..., exists=True, help="Compound YAML path"),
    subject: Path = typer.Option(
        Path("examples/subject_default.yaml"), exists=True, help="Subject YAML path"
    ),
    dose_mg: float = typer.Option(100.0, help="Dose in mg used in observed data"),
    route: str = typer.Option("oral", help="oral|iv"),
    t_end_h: float = typer.Option(24.0, help="End time in hours"),
    dt_out_h: float = typer.Option(0.1, help="Output step in hours"),
    n_samples: int = typer.Option(2000, help="MCMC samples"),
    burn_in: int = typer.Option(500, help="Burn-in iterations"),
    observation_sigma: float = typer.Option(0.1, help="Residual SD for Gaussian likelihood"),
    out: Path = typer.Option(Path("outputs/calibration"), help="Output directory"),
    profile: bool = typer.Option(False, "--profile", help="Write cProfile stats to profile.txt"),
    seed: int | None = typer.Option(None, help="Global random seed"),
) -> None:
    configure_random_seed(seed)
    try:
        subject_cfg = load_subject(subject)
        compound_cfg = load_compound(compound)
    except ValidationError as exc:
        LOGGER.error("Validation error:\n%s", friendly_validation_error(exc))
        raise typer.Exit(code=1) from exc

    observed = pd.read_csv(data)
    required_columns = {"time_h", "C_plasma_mg_per_L"}
    if not required_columns.issubset(observed.columns):
        msg = "Observed CSV must include columns: time_h,C_plasma_mg_per_L"
        raise typer.BadParameter(msg)

    ensure_dir(out)

    def _run_calibration() -> tuple[pd.DataFrame, float]:
        calibration = run_mh_calibration(
            observed=observed,
            subject=subject_cfg,
            compound=compound_cfg,
            dose_mg=dose_mg,
            route=route,
            t_end_h=t_end_h,
            dt_out_h=dt_out_h,
            n_samples=n_samples,
            burn_in=burn_in,
            observation_sigma=observation_sigma,
        )
        return calibration.posterior_samples, calibration.acceptance_rate

    posterior_samples, acceptance_rate = (
        run_with_profile(_run_calibration, out / "profile.txt") if profile else _run_calibration()
    )

    write_csv(posterior_samples, out / "posterior_samples.csv")

    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    axes[0].plot(posterior_samples["iter"], posterior_samples["CLint_L_per_h"])
    axes[0].set_ylabel("CLint (L/h)")
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(posterior_samples["iter"], posterior_samples["ka_per_h"])
    axes[1].set_ylabel("ka (1/h)")
    axes[1].set_xlabel("MCMC iteration")
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "trace_plots.png", dpi=150)
    plt.close(fig)

    posterior_draws = posterior_samples.sample(n=min(100, len(posterior_samples)), random_state=42)
    obs_time = observed["time_h"].to_numpy(dtype=float)
    predictions: list[np.ndarray] = []
    for _, draw in posterior_draws.iterrows():
        sampled_compound = compound_cfg.model_copy(
            update={
                "clint_L_per_h": float(draw["CLint_L_per_h"]),
                "ka_per_h": float(draw["ka_per_h"]),
            }
        )
        pred = simulate(
            subject_cfg,
            sampled_compound,
            dose_mg=dose_mg,
            route=route,
            t_end_h=t_end_h,
            dt_out_h=dt_out_h,
        ).timecourse
        predictions.append(np.interp(obs_time, pred["time_h"], pred["C_plasma_mg_per_L"]))

    pred_array = np.vstack(predictions)
    lower = np.percentile(pred_array, 5, axis=0)
    median = np.percentile(pred_array, 50, axis=0)
    upper = np.percentile(pred_array, 95, axis=0)

    fig2, ax = plt.subplots(figsize=(8, 4.5))
    ax.scatter(obs_time, observed["C_plasma_mg_per_L"], label="Observed", color="black", s=18)
    ax.plot(obs_time, median, label="Posterior median", color="tab:blue")
    ax.fill_between(obs_time, lower, upper, color="tab:blue", alpha=0.25, label="90% CI")
    ax.set_xlabel("Time (h)")
    ax.set_ylabel("Plasma Conc (mg/L)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig2.tight_layout()
    fig2.savefig(out / "posterior_predictive_overlay.png", dpi=150)
    plt.close(fig2)

    write_json(
        {
            "data_file": str(data),
            "compound_file": str(compound),
            "subject_file": str(subject),
            "data_sha256": file_sha256(data),
            "compound_sha256": file_sha256(compound),
            "subject_sha256": file_sha256(subject),
            "seed": seed,
            "acceptance_rate": acceptance_rate,
            "n_posterior_samples": int(len(posterior_samples)),
            "posterior_CLint_mean": float(posterior_samples["CLint_L_per_h"].mean()),
            "posterior_ka_mean": float(posterior_samples["ka_per_h"].mean()),
        },
        out / "calibration_summary.json",
    )


if __name__ == "__main__":
    app()
