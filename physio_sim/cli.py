from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import typer
from pydantic import ValidationError

from physio_sim.config import friendly_validation_error, load_compound, load_subject
from physio_sim.pbpk.solver import simulate
from physio_sim.utils.io import ensure_dir, file_sha256, write_csv, write_json
from physio_sim.utils.metrics import auc_trapezoid, cmax_tmax, effect_summary
from physio_sim.validation import mass_balance_check, physiologic_sanity_check

app = typer.Typer(help="PBPK-like + PD simulation CLI")


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
) -> None:
    try:
        subject_cfg = load_subject(subject)
        compound_cfg = load_compound(compound)
    except ValidationError as exc:
        typer.echo(f"Validation error:\n{friendly_validation_error(exc)}")
        raise typer.Exit(code=1) from exc

    if validate:
        physio_warnings = physiologic_sanity_check(subject_cfg, compound_cfg)
        for warning_msg in physio_warnings:
            typer.echo(f"[validation] {warning_msg}")

    result = simulate(
        subject_cfg,
        compound_cfg,
        dose_mg=dose_mg,
        route=route,
        t_end_h=t_end_h,
        dt_out_h=dt_out_h,
    )
    df = result.timecourse
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


if __name__ == "__main__":
    app()
