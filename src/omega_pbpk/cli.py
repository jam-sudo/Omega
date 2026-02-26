"""Omega PBPK CLI — 14 commands for pharmacokinetic simulation and analysis.

Commands:
  simulate    — Run PBPK simulation (IV or oral)
  predict     — SMILES → ADME property prediction
  multidose   — Multi-dose steady-state simulation
  optimize    — Therapeutic window dose optimization
  safety      — Off-target safety panel
  pgx         — Pharmacogenomics analysis
  calibrate   — Bayesian MCMC parameter calibration
  benchmark   — Multi-drug benchmark validation suite
  sensitivity — Local sensitivity analysis
  validate    — Mass balance and physiological sanity checks
  surrogate   — Train/use neural surrogate model
  uncertainty — Monte Carlo uncertainty propagation
  evaluate    — Integrated drug candidate evaluation
  test        — Run test suite
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


@app.command()
def calibrate(
    compound: str = typer.Option(..., help="Path to compound YAML file."),
    observed: str = typer.Option(
        ..., help="Path to observed data CSV (time_h, C_plasma_mg_per_L)."
    ),
    dose_mg: float = typer.Option(10.0, help="Dose (mg)."),
    route: str = typer.Option("oral", help="Route: 'oral' or 'iv'."),
    body_weight: float = typer.Option(70.0, help="Body weight (kg)."),
    t_end_h: float = typer.Option(24.0, help="Simulation end time (h)."),
    n_samples: int = typer.Option(2000, help="MCMC iterations."),
    burn_in: int = typer.Option(500, help="Burn-in samples to discard."),
    seed: int | None = typer.Option(None, help="Random seed."),
    out: str = typer.Option("outputs/calibration", help="Output directory."),
) -> None:
    """Run Bayesian MCMC calibration against observed clinical data."""
    import pandas as pd

    from omega_pbpk.calibration import run_mh_calibration
    from omega_pbpk.config import load_compound

    drug = load_compound(compound)
    obs_df = pd.read_csv(observed)

    typer.echo(f"Calibrating {drug.name}: {n_samples} MCMC samples, burn-in={burn_in}...")

    result = run_mh_calibration(
        observed=obs_df,
        drug=drug,
        dose_mg=dose_mg,
        route=route,
        body_weight=body_weight,
        t_end_h=t_end_h,
        n_samples=n_samples,
        burn_in=burn_in,
        seed=seed,
    )

    out_path = _ensure_dir(Path(out))

    # Save posterior samples
    result.posterior_samples.to_csv(out_path / "posterior_samples.csv", index=False)
    result.posterior_predictive.to_csv(out_path / "posterior_predictive.csv", index=False)

    # Save summary
    summary = {
        "acceptance_rate": round(result.acceptance_rate, 4),
        "map_estimate": result.map_estimate,
        "n_posterior_samples": len(result.posterior_samples),
    }
    with open(out_path / "calibration_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    typer.echo(f"Acceptance rate: {result.acceptance_rate:.2%}")
    typer.echo(f"MAP estimate: {result.map_estimate}")
    typer.echo(f"Results saved to {out_path}/")


@app.command()
def benchmark(
    suite_dir: str = typer.Option("benchmarks", help="Path to benchmark suite directory."),
    out: str = typer.Option("outputs/benchmark", help="Output directory."),
    body_weight: float = typer.Option(70.0, help="Body weight (kg)."),
) -> None:
    """Run multi-drug benchmark validation suite."""
    from omega_pbpk.validation.benchmarks import run_benchmark_suite

    typer.echo(f"Running benchmark suite from {suite_dir}...")
    summary = run_benchmark_suite(suite_dir, out, body_weight=body_weight)

    overall = "PASS" if summary["overall_pass"] else "FAIL"
    typer.echo(f"\nOverall: {overall} ({summary['n_pass']}/{summary['n_drugs']} drugs passed)")

    for r in summary["results"]:
        m = r["metrics"]
        status = "PASS" if r["pass"] else "FAIL"
        typer.echo(
            f"  {r['drug']:12s}  {status}  "
            f"AUC RE={m['auc_relative_error']:.3f}  "
            f"Cmax RE={m['cmax_relative_error']:.3f}  "
            f"Tmax AE={m['tmax_abs_error_h']:.2f}h"
        )

    typer.echo(f"\nDetailed results: {out}/")


@app.command("sensitivity")
def sensitivity_cmd(
    compound: str = typer.Option(..., help="Path to compound YAML file."),
    dose_mg: float = typer.Option(10.0, help="Dose (mg)."),
    route: str = typer.Option("oral", help="Route: 'oral' or 'iv'."),
    body_weight: float = typer.Option(70.0, help="Body weight (kg)."),
    t_end_h: float = typer.Option(24.0, help="Simulation end time (h)."),
    out: str = typer.Option("outputs/sensitivity", help="Output directory."),
) -> None:
    """Run local sensitivity analysis on compound parameters."""
    from omega_pbpk.config import load_compound
    from omega_pbpk.sensitivity import local_sensitivity

    drug = load_compound(compound)
    typer.echo(f"Sensitivity analysis: {drug.name} {dose_mg}mg {route}...")

    result = local_sensitivity(
        drug=drug,
        dose_mg=dose_mg,
        route=route,
        body_weight=body_weight,
        t_end_h=t_end_h,
    )

    out_path = _ensure_dir(Path(out))
    result.metrics.to_csv(out_path / "sensitivity.csv", index=False)

    typer.echo("\nParameter influence ranking (|dCmax| + |dAUC|):")
    for _, row in result.metrics.iterrows():
        typer.echo(
            f"  {row['parameter']:25s}  "
            f"base={row['base_value']:.4g}  "
            f"dCmax={row['dCmax_dparam']:+.4e}  "
            f"dAUC={row['dAUC_dparam']:+.4e}  "
            f"influence={row['influence_abs_sum']:.4e}"
        )

    typer.echo(f"\nResults saved to {out_path}/sensitivity.csv")


@app.command("validate")
def validate_cmd(
    compound: str = typer.Option(..., help="Path to compound YAML file."),
    dose_mg: float = typer.Option(10.0, help="Dose (mg)."),
    route: str = typer.Option("oral", help="Route: 'oral' or 'iv'."),
    body_weight: float = typer.Option(70.0, help="Body weight (kg)."),
    t_end_h: float = typer.Option(24.0, help="Simulation end time (h)."),
) -> None:
    """Run mass balance and physiological sanity checks."""
    from omega_pbpk.config import load_compound
    from omega_pbpk.core.body import WholeBodyPBPK
    from omega_pbpk.validation import mass_balance_check, physiologic_sanity_check

    drug = load_compound(compound)
    model = WholeBodyPBPK(drug, body_weight=body_weight)
    if route == "iv":
        model.setup_iv(dose_mg)
    else:
        model.setup_oral(dose_mg)

    result = model.simulate(t_end_h=t_end_h)
    typer.echo(f"Validation: {drug.name} {dose_mg}mg {route}\n")

    # Mass balance check
    mb_warnings = mass_balance_check(result.amounts, dose_mg, time_h=result.time_h)
    if mb_warnings:
        typer.echo("Mass balance: FAIL")
        for w in mb_warnings:
            typer.echo(f"  WARNING: {w}")
    else:
        mb_final = float(result.mass_balance()[-1])
        pct = mb_final / dose_mg * 100 if dose_mg > 0 else 0
        typer.echo(f"Mass balance: PASS ({pct:.4f}%)")

    # Physiological sanity
    sanity_warnings = physiologic_sanity_check(model.organs, model.cardiac_output, drug.fup)
    if sanity_warnings:
        typer.echo("\nPhysiological checks: FAIL")
        for w in sanity_warnings:
            typer.echo(f"  WARNING: {w}")
    else:
        typer.echo("Physiological checks: PASS")


@app.command("surrogate")
def surrogate_cmd(
    n_samples: int = typer.Option(500, help="Training samples to generate."),
    dose_mg: float = typer.Option(10.0, help="Dose (mg)."),
    route: str = typer.Option("oral", help="Route: 'oral' or 'iv'."),
    epochs: int = typer.Option(300, help="Training epochs."),
    seed: int = typer.Option(42, help="Random seed."),
    out: str = typer.Option("outputs/surrogate", help="Output directory."),
) -> None:
    """Train a neural surrogate model for fast PK prediction."""
    from omega_pbpk.surrogate import PKSurrogate
    from omega_pbpk.surrogate.data_generator import generate_training_data

    typer.echo(f"Generating {n_samples} training samples...")
    data = generate_training_data(n_samples=n_samples, dose_mg=dose_mg, route=route, seed=seed)
    typer.echo(f"Valid samples: {data.n_samples}")

    model = PKSurrogate(n_input=data.n_params, n_output=data.n_outputs, hidden_dim=64, n_layers=3)
    model.param_names = data.param_names
    model.output_names = data.output_names

    typer.echo(f"Training MLP ({model.n_layers} layers, dim={model.hidden_dim})...")
    history = model.train(data.X, data.y, epochs=epochs, seed=seed)

    out_path = _ensure_dir(Path(out))
    model.save(out_path / "model")

    final_loss = history["val_loss"][-1] if history["val_loss"] else 0
    typer.echo(f"Training complete. Final val loss: {final_loss:.6f}")
    typer.echo(f"Model saved to {out_path}/model/")


@app.command("uncertainty")
def uncertainty_cmd(
    compound: str = typer.Option(..., help="Path to compound YAML file."),
    dose_mg: float = typer.Option(10.0, help="Dose (mg)."),
    route: str = typer.Option("oral", help="Route: 'oral' or 'iv'."),
    n_samples: int = typer.Option(200, help="Monte Carlo samples."),
    seed: int = typer.Option(42, help="Random seed."),
) -> None:
    """Run Monte Carlo uncertainty propagation on compound parameters."""
    from omega_pbpk.config import load_compound
    from omega_pbpk.uncertainty import DistributionSpec, monte_carlo_propagation

    drug = load_compound(compound)
    base_params = {
        "clint_hepatic_L_per_h": drug.clint_scaled_L_per_h,
        "clint_gut_L_per_h": drug.gut_clint_scaled_L_per_h,
        "fup": drug.fup,
        "rbp": drug.rbp,
        "peff": drug.peff,
        "logP": drug.logP,
    }
    clint_val = base_params["clint_hepatic_L_per_h"]
    specs = [
        DistributionSpec("clint_hepatic_L_per_h", "lognormal", clint_val, 0.3),
        DistributionSpec("fup", "lognormal", max(base_params["fup"], 0.001), 0.2),
    ]

    typer.echo(f"MC uncertainty: {drug.name} {dose_mg}mg {route}, {n_samples} samples...")
    result = monte_carlo_propagation(
        drug_params=base_params,
        uncertainty_specs=specs,
        n_samples=n_samples,
        dose_mg=dose_mg,
        route=route,
        seed=seed,
    )

    typer.echo(f"\nResults ({result.n_samples} valid samples):")
    typer.echo(f"  Cmax CV = {result.cmax_cv:.3f}")
    typer.echo(f"  AUC  CV = {result.auc_cv:.3f}")
    typer.echo(f"  Cmax percentiles: {result.cmax_percentiles}")
    typer.echo(f"  AUC  percentiles: {result.auc_percentiles}")


@app.command("evaluate")
def evaluate_cmd(
    compound: str = typer.Option(..., help="Path to compound YAML file."),
    dose_mg: float = typer.Option(10.0, help="Dose (mg)."),
    route: str = typer.Option("oral", help="Route: 'oral' or 'iv'."),
    out: str = typer.Option("outputs/evaluation", help="Output directory."),
) -> None:
    """Run integrated drug candidate evaluation."""
    from omega_pbpk.config import load_compound
    from omega_pbpk.pipeline import evaluate_candidate

    drug = load_compound(compound)
    typer.echo(f"Evaluating candidate: {drug.name} {dose_mg}mg {route}...")

    report = evaluate_candidate(drug=drug, dose_mg=dose_mg, route=route)

    typer.echo(f"\n{'=' * 50}")
    typer.echo(f"  CANDIDATE EVALUATION: {report.drug_name}")
    typer.echo(f"{'=' * 50}")
    typer.echo(f"  Overall Score:      {report.overall_score}/100")
    typer.echo(f"  PK Stability:       {report.pk_stability_score:.3f}")
    typer.echo(f"  Exposure CV:        {report.exposure_cv:.3f}")
    typer.echo(f"  DDI Risk:           {report.ddi_risk_score:.3f}")
    typer.echo(f"  Clearance Risk:     {report.clearance_risk_score:.3f}")
    typer.echo(f"  Genotype AUC Ratio: {report.genotype_auc_ratio:.2f}x")
    typer.echo(f"  Risk Flags:         {report.risk_flags.risk_count}/4")
    for name, val in report.risk_flags.summary().items():
        if val:
            typer.echo(f"    !! {name}")
    typer.echo(
        f"\n  PK: Cmax={report.pk_summary['Cmax_mg_L']:.4f} mg/L, "
        f"AUC={report.pk_summary['AUC_mg_h_L']:.4f} mg*h/L, "
        f"t1/2={report.pk_summary['half_life_h']}h"
    )

    out_path = _ensure_dir(Path(out))
    summary = {
        "drug": report.drug_name,
        "overall_score": report.overall_score,
        "pk_summary": report.pk_summary,
        "risk_flags": report.risk_flags.summary(),
        "exposure_cv": report.exposure_cv,
        "pk_stability_score": report.pk_stability_score,
        "ddi_risk_score": report.ddi_risk_score,
        "genotype_auc_ratio": report.genotype_auc_ratio,
        "details": report.details,
    }
    with open(out_path / "report.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    typer.echo(f"\n  Report saved to {out_path}/report.json")


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
