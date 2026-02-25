"""Omega PBPK CLI — 7 commands for pharmacokinetic simulation and analysis.

Commands:
  simulate   — Run PBPK simulation (IV or oral)
  predict    — SMILES → ADME property prediction
  multidose  — Multi-dose steady-state simulation
  optimize   — Therapeutic window dose optimization
  safety     — Off-target safety panel
  pgx        — Pharmacogenomics analysis
  test       — Run test suite
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import typer

app = typer.Typer(
    name="omega",
    help="Omega PBPK — Whole-body pharmacokinetic simulation platform.",
    add_completion=False,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("omega_pbpk")


def _ensure_dir(path: Path) -> Path:
    """Create output directory if needed."""
    path.mkdir(parents=True, exist_ok=True)
    return path


@app.command()
def simulate(
    compound: str = typer.Option(..., help="Path to compound YAML file."),
    dose_mg: float = typer.Option(10.0, help="Dose in mg."),
    route: str = typer.Option("oral", help="Route: 'oral' or 'iv'."),
    t_end_h: float = typer.Option(24.0, help="Simulation end time (h)."),
    body_weight: float = typer.Option(70.0, help="Body weight (kg)."),
    smiles: str | None = typer.Option(None, help="SMILES string (alternative to compound YAML)."),
    out: str = typer.Option("outputs/run", help="Output directory."),
    subject: str | None = typer.Option(None, help="Path to subject YAML file."),
) -> None:
    """Run PBPK simulation for a compound."""

    from omega_pbpk.config import load_compound, load_subject
    from omega_pbpk.core.body import WholeBodyPBPK
    from omega_pbpk.drugs.drug import Drug
    from omega_pbpk.visualization.plots import PKPlotter

    # Load drug
    if smiles:
        from omega_pbpk.prediction.adme_predictor import ADMEPredictor

        predictor = ADMEPredictor()
        props = predictor.predict(smiles)
        drug = Drug(
            name="Predicted",
            mw=props.mw,
            logP=props.logP,
            fup=props.fup,
            rbp=props.rbp,
            peff=props.peff,
            clint={"CYP3A4": props.clint_3a4, "CYP2D6": props.clint_2d6},
        )
        typer.echo(f"Predicted ADME: MW={props.mw}, logP={props.logP}, fup={props.fup}")
    else:
        drug = load_compound(compound)

    # Load subject
    bw = body_weight
    if subject:
        subj = load_subject(subject)
        bw = subj["body_weight_kg"]

    # Setup and simulate
    model = WholeBodyPBPK(drug, body_weight=bw)
    if route == "iv":
        model.setup_iv(dose_mg)
    else:
        model.setup_oral(dose_mg)

    typer.echo(f"Simulating {drug.name}: {dose_mg} mg {route}, {t_end_h}h, {bw} kg...")
    result = model.simulate(t_end_h=t_end_h)

    # Output
    out_path = _ensure_dir(Path(out))

    # Timecourse CSV
    cp = result.plasma_concentration()
    csv_path = out_path / "timecourse.csv"
    with open(csv_path, "w") as f:
        f.write("time_h,Cp_mg_L\n")
        for t, c in zip(result.time_h, cp, strict=False):
            f.write(f"{t:.4f},{c:.8f}\n")

    # Summary JSON
    pk = result.pk_summary()
    mb = result.mass_balance()
    pk["mass_balance_pct"] = round(float(mb[-1] / dose_mg * 100), 4) if dose_mg > 0 else 0.0

    json_path = out_path / "summary.json"
    with open(json_path, "w") as f:
        json.dump(pk, f, indent=2, default=str)

    # Plot
    plotter = PKPlotter()
    plotter.plot_pk(
        result.time_h,
        cp,
        title=f"{drug.name} {dose_mg}mg {route}",
        save_path=out_path / "plots.png",
    )

    typer.echo(f"Results saved to {out_path}/")
    typer.echo(f"  Cmax = {pk['Cmax_mg_L']:.4f} mg/L at Tmax = {pk['Tmax_h']:.2f} h")
    typer.echo(f"  AUC  = {pk['AUC_mg_h_L']:.4f} mg·h/L")
    typer.echo(f"  t½   = {pk['half_life_h']} h")
    typer.echo(f"  Mass balance = {pk['mass_balance_pct']:.2f}%")


@app.command()
def predict(
    smiles: str = typer.Option(..., help="SMILES string."),
) -> None:
    """Predict ADME properties from SMILES."""
    from omega_pbpk.prediction.adme_predictor import ADMEPredictor

    predictor = ADMEPredictor()
    props = predictor.predict(smiles)

    typer.echo("ADME Prediction Results:")
    typer.echo(f"  MW:          {props.mw:.2f} g/mol")
    typer.echo(f"  logP:        {props.logP:.2f}")
    typer.echo(f"  logS:        {props.logS:.2f}")
    typer.echo(f"  Peff:        {props.peff:.3f} ×10⁻⁴ cm/s")
    typer.echo(f"  fup:         {props.fup:.4f}")
    typer.echo(f"  Rbp:         {props.rbp:.3f}")
    typer.echo(f"  CLint_3A4:   {props.clint_3a4:.3f} µL/min/pmol")
    typer.echo(f"  CLint_2D6:   {props.clint_2d6:.3f} µL/min/pmol")
    typer.echo(f"  hERG IC50:   {props.herg_ic50_uM:.2f} µM")
    typer.echo(f"  Confidence:  {props.confidence}")


@app.command()
def multidose(
    compound: str = typer.Option(..., help="Path to compound YAML file."),
    dose_mg: float = typer.Option(10.0, help="Dose per administration (mg)."),
    interval: float = typer.Option(12.0, "--interval", help="Dosing interval (h)."),
    days: int = typer.Option(7, help="Number of days."),
    out: str = typer.Option("outputs/multidose", help="Output directory."),
) -> None:
    """Simulate multi-dose steady-state PK."""
    from omega_pbpk.clinical.dose_optimization import MultiDoseSimulator
    from omega_pbpk.config import load_compound
    from omega_pbpk.visualization.plots import PKPlotter

    drug = load_compound(compound)
    sim = MultiDoseSimulator()
    result = sim.simulate(drug, dose_mg, interval, days)

    out_path = _ensure_dir(Path(out))

    # Save CSV
    with open(out_path / "multidose_timecourse.csv", "w") as f:
        f.write("time_h,Cp_mg_L\n")
        for t, c in zip(result.time_h, result.cp_mg_L, strict=False):
            f.write(f"{t:.4f},{c:.8f}\n")

    # Save summary
    summary = {
        "dose_mg": dose_mg,
        "interval_h": interval,
        "n_doses": result.n_doses,
        "Css_max": result.css_max,
        "Css_min": result.css_min,
        "Css_avg": result.css_avg,
        "accumulation_ratio": result.accumulation_ratio,
        "fluctuation_pct": result.fluctuation_percent,
    }
    with open(out_path / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Plot
    dose_times = [i * interval for i in range(result.n_doses)]
    plotter = PKPlotter()
    plotter.plot_multidose(
        result.time_h,
        result.cp_mg_L,
        dose_times,
        title=f"{drug.name} {dose_mg}mg q{interval}h",
        save_path=out_path / "plots.png",
    )

    typer.echo(f"Multi-dose simulation: {drug.name} {dose_mg}mg q{interval}h × {days} days")
    typer.echo(f"  Css,max = {result.css_max:.4f} mg/L")
    typer.echo(f"  Css,min = {result.css_min:.4f} mg/L")
    typer.echo(f"  Accumulation ratio = {result.accumulation_ratio:.3f}")
    typer.echo(f"  Results saved to {out_path}/")


@app.command()
def optimize(
    compound: str = typer.Option(..., help="Path to compound YAML file."),
    mec: float = typer.Option(..., help="Minimum effective concentration (mg/L)."),
    mtc: float = typer.Option(..., help="Maximum tolerated concentration (mg/L)."),
) -> None:
    """Optimize dose for therapeutic window."""
    from omega_pbpk.clinical.dose_optimization import DoseOptimizer
    from omega_pbpk.config import load_compound

    drug = load_compound(compound)
    optimizer = DoseOptimizer()
    result = optimizer.optimize_dose(drug, mec, mtc)

    if "error" in result:
        typer.echo(f"ERROR: {result['error']}", err=True)
        raise typer.Exit(1)

    typer.echo(f"Dose Optimization for {drug.name}:")
    typer.echo(f"  Therapeutic window: {mec} – {mtc} mg/L")
    typer.echo(f"  Optimal dose: {result['optimal_dose_mg']} mg")
    typer.echo(f"  Predicted Cmax: {result['Cmax_mg_L']:.4f} mg/L")
    typer.echo(f"  Time in window: {result['time_in_window_pct']}%")


@app.command()
def safety(
    smiles: str = typer.Option(..., help="SMILES string."),
    cmax_uM: float = typer.Option(1.0, "--cmax-uM", help="Expected Cmax (µM)."),
    fup: float = typer.Option(0.5, help="Fraction unbound in plasma."),
) -> None:
    """Run off-target safety panel."""
    from omega_pbpk.docking.off_target import SafetyPanel
    from omega_pbpk.prediction.adme_predictor import ADMEPredictor

    predictor = ADMEPredictor()
    props = predictor.predict(smiles)

    panel = SafetyPanel()
    report = panel.assess(
        compound_name=smiles[:30],
        logP=props.logP,
        mw=props.mw,
        cmax_uM=cmax_uM,
        fup=fup,
    )

    typer.echo(f"Safety Panel — Overall Risk: {report.overall_risk.upper()}")
    typer.echo(f"  Cmax (total): {report.cmax_total_uM} µM")
    typer.echo(f"  Cmax (unbound): {report.cmax_unbound_uM} µM")
    typer.echo()

    for r in report.target_results:
        icon = "✓" if r.flag == "safe" else ("!" if r.flag == "caution" else "✗")
        msg = f"  [{icon}] {r.target:12s}  IC50={r.predicted_ic50_uM:8.2f} µM"
        typer.echo(f"{msg}  Margin={r.margin:8.1f}×  {r.flag}")

    typer.echo()
    for r in report.cyp_results:
        typer.echo(f"  {r.enzyme:10s}  IC50={r.predicted_ic50_uM:8.2f} µM  DDI risk: {r.ddi_risk}")

    if report.flags:
        typer.echo()
        typer.echo("Flags:")
        for flag in report.flags:
            typer.echo(f"  ⚠ {flag}")


@app.command()
def pgx(
    gene: str = typer.Option("CYP2D6", help="CYP gene (CYP2D6, CYP2C19, CYP2C9, CYP3A5, CYP1A2)."),
    population: str = typer.Option(
        "Global", help="Population (Caucasian, East_Asian, African, Global)."
    ),
) -> None:
    """Pharmacogenomics analysis — CYP polymorphism impact."""
    from omega_pbpk.pharmacogenomics.cyp_polymorphism import PGxAnalyzer

    analyzer = PGxAnalyzer()
    summary = analyzer.population_summary(gene, population)
    results = analyzer.analyze_gene(gene, population)

    typer.echo(f"Pharmacogenomics: {gene} in {population}")
    typer.echo("  Phenotype distribution:")
    for pheno, freq in summary["phenotype_distribution"].items():
        bar = "█" * int(freq * 40)
        typer.echo(f"    {pheno:4s}: {freq:6.1%}  {bar}")

    typer.echo()
    typer.echo("  Top diplotypes:")
    for r in results[:8]:
        scale = r.clint_scaling_factor
        freq = r.population_frequency
        typer.echo(f"    {r.diplotype:15s}  {r.phenotype:3s}  CLint×{scale:.2f}  freq={freq:.4f}")


@app.command("test")
def run_tests() -> None:
    """Run the test suite."""
    import subprocess

    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "-v", "--tb=short"],
        cwd=str(Path(__file__).parent.parent.parent),
    )
    raise typer.Exit(result.returncode)


def main() -> None:
    """CLI entry point."""
    app()


if __name__ == "__main__":
    main()
