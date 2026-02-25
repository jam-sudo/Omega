from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

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
    load_subject,
)
from physio_sim.core.sensitivity import local_sensitivity
from physio_sim.pbpk.solver import simulate
from physio_sim.pipeline.candidate_evaluator import evaluate_candidate
from physio_sim.risk.flags import compute_risk_flags
from physio_sim.utils.io import ensure_dir, file_sha256, write_csv, write_json
from physio_sim.utils.metrics import auc_trapezoid, cmax_tmax, effect_summary
from physio_sim.utils.profile import run_with_profile
from physio_sim.validation import mass_balance_check, physiologic_sanity_check
from physio_sim.validation.benchmarks import run_benchmark_suite

app = typer.Typer(help="PBPK-like + PD simulation CLI")
LOGGER = logging.getLogger("physio_sim")


def _configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s %(name)s: %(message)s", force=True)


def _population_run(
    subject: Any,
    compound: Any,
    dose_mg: float,
    route: str,
    t_end_h: float,
    dt_out_h: float,
    population: int,
    out: Path,
    deterministic: bool,
) -> dict[str, float]:
    rng = np.random.default_rng(42 if deterministic else None)
    curves: list[pd.DataFrame] = []
    summary_rows: list[dict[str, float]] = []
    for idx in range(population):
        sampled = compound.model_copy(
            update={
                "clint_L_per_h": float(
                    rng.lognormal(np.log(max(compound.clint_L_per_h, 1e-9)), 0.2)
                ),
                "fu_plasma": float(rng.lognormal(np.log(max(compound.fu_plasma, 1e-9)), 0.15)),
                "ka_per_h": float(rng.lognormal(np.log(max(compound.ka_per_h, 1e-9)), 0.2)),
            }
        )
        tc = simulate(
            subject,
            sampled,
            dose_mg,
            route,
            t_end_h,
            dt_out_h,
            deterministic=deterministic,
        ).timecourse
        curves.append(tc)
        cmax, _ = cmax_tmax(tc["time_h"].to_numpy(), tc["C_plasma_mg_per_L"].to_numpy())
        auc = auc_trapezoid(tc["time_h"].to_numpy(), tc["C_plasma_mg_per_L"].to_numpy())
        summary_rows.append({"subject": float(idx), "Cmax_mg_per_L": cmax, "AUC": auc})

    summary_df = pd.DataFrame(summary_rows)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for tc in curves[: min(40, len(curves))]:
        ax.plot(tc["time_h"], tc["C_plasma_mg_per_L"], color="tab:blue", alpha=0.2)
    ax.set_xlabel("Time (h)")
    ax.set_ylabel("Plasma Conc (mg/L)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "population_curves.png", dpi=150)
    plt.close(fig)

    payload = {
        "n": population,
        "Cmax_mean": float(summary_df["Cmax_mg_per_L"].mean()),
        "Cmax_cv": float(
            summary_df["Cmax_mg_per_L"].std(ddof=1) / summary_df["Cmax_mg_per_L"].mean()
        ),
        "AUC_mean": float(summary_df["AUC"].mean()),
        "AUC_cv": float(summary_df["AUC"].std(ddof=1) / summary_df["AUC"].mean()),
    }
    write_json(payload, out / "population_summary.json")
    return payload


def _write_academic_report(
    out: Path,
    summary: dict[str, object],
    sensitivity_path: Path | None,
    flags: dict[str, object] | None,
) -> None:
    lines = [
        "# Academic Report",
        "",
        "## Model description",
        "Perfusion-limited PBPK with liver dual inflow, renal elimination, "
        "and direct/linked Emax PD.",
        "",
        "## Parameter tables",
        "See input YAML files and summary.json metadata.",
        "",
        "## Validation summary",
        f"Warnings: {summary.get('validation_warnings', [])}",
        "",
        "## Uncertainty intervals",
        "If uncertainty or population analyses were run, see output JSON/CSV files.",
        "",
        "## Sensitivity ranking",
    ]
    if sensitivity_path is not None and sensitivity_path.exists():
        lines.append(f"Included in `{sensitivity_path.name}`.")
    else:
        lines.append("Not requested for this run.")
    lines.extend(["", "## Risk flags"])
    lines.append(
        json.dumps(flags, indent=2) if flags is not None else "Not requested for this run."
    )
    lines.extend(
        ["", "## Limitations", "Automated report; does not replace external clinical validation."]
    )
    report_path = out / "report_academic.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")


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
    deterministic: bool = typer.Option(
        False, "--deterministic", help="Deterministic reproducibility"
    ),
    sensitivity: bool = typer.Option(False, "--sensitivity", help="Run local sensitivity analysis"),
    population: int | None = typer.Option(None, "--population", help="Population simulation size"),
    academic_report: bool = typer.Option(
        False, "--academic-report", help="Write academic markdown report"
    ),
    candidate: Path | None = typer.Option(
        None, help="Candidate compound.yaml for evaluation pipeline"
    ),
) -> None:
    if deterministic and seed is None:
        seed = 0
    configure_random_seed(seed)
    try:
        subject_cfg = load_subject(subject)
        compound_cfg = load_compound(compound)
    except ValidationError as exc:
        LOGGER.error("Validation error:\n%s", friendly_validation_error(exc))
        raise typer.Exit(code=1) from exc

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
            deterministic=deterministic,
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
        "deterministic": deterministic,
        "solver_rtol": 1e-8 if deterministic else 1e-6,
        "solver_atol": 1e-10 if deterministic else 1e-9,
        "Cmax_mg_per_L": cmax,
        "Tmax_h": tmax,
        "AUC0_tend_mg_h_per_L": auc,
        **e_summary,
    }
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

    sensitivity_path: Path | None = None
    if sensitivity:
        sens = local_sensitivity(
            subject=subject_cfg,
            compound=compound_cfg,
            dose_mg=dose_mg,
            route=route,
            t_end_h=t_end_h,
            dt_out_h=dt_out_h,
            parameters=["clint_L_per_h", "clr_L_per_h", "fu_plasma", "ka_per_h", "f_gut"],
        )
        sensitivity_path = out / "sensitivity.csv"
        write_csv(sens.metrics, sensitivity_path)
        ranked = sens.metrics[["parameter", "influence_abs_sum"]].to_dict(orient="records")
        write_json({"ranked_parameter_influence": ranked}, out / "sensitivity_ranking.json")

    population_summary: dict[str, float] | None = None
    if population is not None and population > 0:
        population_summary = _population_run(
            subject_cfg,
            compound_cfg,
            dose_mg,
            route,
            t_end_h,
            dt_out_h,
            population,
            out,
            deterministic,
        )

    flags_payload: dict[str, object] | None = None
    if population_summary is not None:
        flags = compute_risk_flags(
            cmax_mg_per_l=float(cmax),
            half_life_h=float(t_end_h / 2.0),
            exposure_cv=float(population_summary["AUC_cv"]),
            ddi_risk_score=0.0,
        )
        flags_payload = flags.__dict__
        write_json(flags_payload, out / "risk_flags.json")

    if candidate is not None:
        evaluation = evaluate_candidate(
            candidate_path=candidate,
            default_subject_path=subject,
            dose_mg=dose_mg,
            route=route,
            t_end_h=t_end_h,
            dt_out_h=dt_out_h,
            seed=seed,
            deterministic=deterministic,
        )
        write_json(evaluation.scores, out / "candidate_scores.json")
        write_json(evaluation.risk_flags.__dict__, out / "risk_flags.json")
        flags_payload = evaluation.risk_flags.__dict__

    if academic_report:
        _write_academic_report(out, summary, sensitivity_path, flags_payload)


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


@app.command("benchmark")
def benchmark_cmd(
    suite: Path = typer.Option(Path("benchmarks"), exists=True, help="Benchmark suite directory"),
    out: Path = typer.Option(Path("outputs/benchmarks"), help="Benchmark output directory"),
) -> None:
    passed = run_benchmark_suite(suite, out)
    raise typer.Exit(code=0 if passed else 1)


if __name__ == "__main__":
    app()
