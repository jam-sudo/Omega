"""Omega PBPK CLI — 21 commands for pharmacokinetic simulation and analysis.

Commands:
  simulate         — Run PBPK simulation (IV or oral)
  predict          — SMILES → full PK simulation (ADME + PBPK)
  smiles-report    — SMILES → complete formatted PK assessment report (all subsystems)
  multidose        — Multi-dose steady-state simulation
  optimize         — Therapeutic window dose optimization
  safety           — Off-target safety panel
  pgx              — Pharmacogenomics analysis
  pgx-sim          — PGx-stratified PBPK (PM/IM/NM/UM AUC ratios)
  calibrate        — Bayesian MCMC parameter calibration
  benchmark        — Multi-drug benchmark validation suite
  sensitivity      — Local sensitivity analysis
  validate         — Mass balance and physiological sanity checks
  surrogate        — Train neural surrogate model on real PBPK data
  uncertainty      — Monte Carlo uncertainty propagation
  evaluate         — Integrated drug candidate evaluation
  population       — Population PK simulation (virtual subjects)
  report           — Generate regulatory-grade HTML report
  invitro          — In vitro metabolic stability assay (microsomes/hepatocyte + Caco-2)
  invitro-pipeline — Full in vitro → IVIVE → Drug object pipeline from SMILES
  serve            — Start FastAPI REST API server
  test             — Run test suite
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import typer
from typer.core import TyperGroup

app = typer.Typer(
    name="omega",
    help="Omega PBPK — Whole-body pharmacokinetic simulation platform.",
    add_completion=False,
)

# ---------------------------------------------------------------------------
# 4-verb sub-apps (M3 CLI restructure)
# ---------------------------------------------------------------------------


class _SimulateGroup(TyperGroup):
    """Custom Typer Group that falls back to 'single' subcommand."""

    def parse_args(self, ctx: typer.Context, args: list[str]) -> list[str]:  # type: ignore[override]
        # If the first arg looks like a file path (not a known subcommand), prepend "single"
        if args and args[0] not in self.commands and not args[0].startswith("-"):
            args = ["single"] + args
        return super().parse_args(ctx, args)


simulate_app = typer.Typer(
    help="Run PBPK simulations.",
    cls=_SimulateGroup,
)
train_app = typer.Typer(help="Train ML surrogate or calibrate models.")
validate_app = typer.Typer(help="Validate, benchmark, and QA.")
serve_app = typer.Typer(help="Start API server.")

app.add_typer(simulate_app, name="simulate")
app.add_typer(train_app, name="train")
app.add_typer(validate_app, name="validate")
app.add_typer(serve_app, name="serve")

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("omega_pbpk")

# Audit logger — writes a separate record for each CLI invocation so that
# clinical simulations (dose, route, compound path) can be traced back to
# specific runs without polluting the main application log.
_audit_logger = logging.getLogger("omega_pbpk.audit")
if not _audit_logger.handlers:
    _audit_handler = logging.StreamHandler(sys.stderr)
    _audit_handler.setFormatter(
        logging.Formatter("AUDIT %(asctime)s %(message)s", datefmt="%Y-%m-%dT%H:%M:%S")
    )
    _audit_logger.addHandler(_audit_handler)
    _audit_logger.setLevel(logging.INFO)
    _audit_logger.propagate = False


def _audit(command: str, **kwargs: object) -> None:
    """Emit a structured audit record for a CLI invocation.

    Args:
        command: Name of the CLI command being executed.
        **kwargs: Key/value pairs of the command's principal inputs.
            Values are truncated to 200 chars to avoid log bloat.
    """
    parts = [f"cmd={command}"]
    for k, v in kwargs.items():
        truncated = str(v)[:200]
        parts.append(f"{k}={truncated!r}")
    _audit_logger.info(" ".join(parts))


def _ensure_dir(path: Path) -> Path:
    """Create output directory if needed."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def _output(data: dict, json_output: bool) -> None:
    """Output data as JSON or plain key:value pairs."""
    if json_output:
        typer.echo(json.dumps(data, indent=2))
    else:
        for k, v in data.items():
            typer.echo(f"{k}: {v}")


# Aliases: CLI/legacy parameter name → Drug field name for sensitivity analysis
_SENS_PARAM_ALIASES: dict[str, str] = {
    "clint_L_per_h": "clint_hepatic_L_per_h",
    "fu_plasma": "fup",
    "ka_per_h": "peff",
}


@app.command("simulate-legacy", hidden=True)
def simulate(
    compound: str = typer.Option(..., help="Path to compound YAML file."),
    dose_mg: float = typer.Option(10.0, help="Dose in mg."),
    route: str = typer.Option("oral", help="Route: 'oral' or 'iv'."),
    t_end_h: float = typer.Option(24.0, help="Simulation end time (h)."),
    body_weight: float = typer.Option(70.0, help="Body weight (kg)."),
    smiles: str | None = typer.Option(None, help="SMILES string (alternative to compound YAML)."),
    out: str = typer.Option("outputs/run", help="Output directory."),
    subject: str | None = typer.Option(None, help="Path to subject YAML file."),
    deterministic: bool = typer.Option(
        False,
        "--deterministic/--no-deterministic",
        help="Fixed seed + BDF solver for reproducibility.",
    ),
    sensitivity: bool = typer.Option(
        False,
        "--sensitivity/--no-sensitivity",
        help="Run local sensitivity analysis.",
    ),
    sensitivity_params: str | None = typer.Option(
        None,
        "--sensitivity-params",
        help="Comma-separated parameter names for sensitivity.",
    ),
    sensitivity_eps: float = typer.Option(
        0.01,
        "--sensitivity-eps",
        help="Relative finite-difference step for sensitivity.",
    ),
    population: int = typer.Option(
        0,
        "--population",
        help="Number of virtual subjects for population PK (0=skip).",
    ),
    academic_report: bool = typer.Option(
        False,
        "--academic-report/--no-academic-report",
        help="Generate academic-style Markdown report.",
    ),
    qsp_model: str | None = typer.Option(
        None, "--qsp-model", help="QSP model name (e.g. 'turnover')."
    ),
    qsp_config: str | None = typer.Option(None, "--qsp-config", help="Path to QSP config YAML."),
    qsp_mode: str = typer.Option(
        "posthoc", "--qsp-mode", help="QSP coupling mode: 'posthoc' or 'coupled'."
    ),
    uncertainty: bool = typer.Option(
        False,
        "--uncertainty/--no-uncertainty",
        help="Run conformal UQ propagation.",
    ),
    uncertainty_samples: int = typer.Option(
        200,
        "--uncertainty-samples",
        help="LHS samples for conformal UQ.",
    ),
) -> None:
    """Run PBPK simulation for a compound."""
    _audit(
        "simulate",
        compound=compound,
        smiles=smiles,
        dose_mg=dose_mg,
        route=route,
        t_end_h=t_end_h,
        body_weight=body_weight,
        out=out,
    )

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

    out_path = _ensure_dir(Path(out))
    cp = result.plasma_concentration()
    sim_time = result.time_h

    # --- QSP post-hoc integration -------------------------------------------
    b_biomarker: list[float] | None = None
    qsp_summary: dict[str, object] = {}
    if qsp_model:
        import numpy as _np
        import yaml as _yaml
        from scipy.integrate import solve_ivp
        from scipy.interpolate import interp1d

        from omega_pbpk.experimental.physio_sim.qsp.registry import get_qsp_model as _get_qsp

        _qsp_raw = (
            _yaml.safe_load(Path(qsp_config).read_text(encoding="utf-8")) if qsp_config else {}
        )
        _qsp_params = _qsp_raw.get("params", _qsp_raw)
        _qsp_inst = _get_qsp(qsp_model)
        _qsp_inst.validate_params(_qsp_params)
        _cp_fn = interp1d(sim_time, cp, kind="linear", bounds_error=False, fill_value=0.0)

        def _qsp_rhs(t: float, y: object) -> object:  # type: ignore[return]
            sigs = {"C_signal": max(0.0, float(_cp_fn(t)))}
            return _qsp_inst.rhs(t, y, sigs, _qsp_params)  # type: ignore[arg-type]

        _y0 = _qsp_inst.initial_state(_qsp_params)
        _sol = solve_ivp(_qsp_rhs, [0.0, t_end_h], _y0, t_eval=sim_time, method="RK45")
        b_biomarker = _sol.y[0].tolist()
        _bmax_idx = int(_np.argmax(b_biomarker))
        qsp_summary = {
            "qsp_model": qsp_model,
            "qsp_mode": qsp_mode,
            "Bmax": float(b_biomarker[_bmax_idx]),
            "t_Bmax": float(sim_time[_bmax_idx]),
            "Bend": float(b_biomarker[-1]),
        }

    # --- Timecourse CSV -------------------------------------------------------
    csv_path = out_path / "timecourse.csv"
    header = "time_h,Cp_mg_L" + (",B_biomarker" if b_biomarker is not None else "")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write(header + "\n")
        for i, (t, c) in enumerate(zip(sim_time, cp, strict=False)):
            row = f"{t:.4f},{c:.8f}"
            if b_biomarker is not None:
                row += f",{b_biomarker[i]:.8f}"
            f.write(row + "\n")

    # --- Summary JSON --------------------------------------------------------
    pk = result.pk_summary()
    mb = result.mass_balance()
    pk["mass_balance_pct"] = round(float(mb[-1] / dose_mg * 100), 4) if dose_mg > 0 else 0.0
    pk.update(qsp_summary)

    if deterministic:
        import importlib.metadata
        import subprocess as _sp
        from datetime import datetime, timezone

        try:
            _git = _sp.run(
                ["git", "rev-parse", "--short", "HEAD"],
                capture_output=True,
                text=True,
                check=False,
            ).stdout.strip()
        except Exception:
            _git = "unknown"
        try:
            _ver = importlib.metadata.version("omega-pbpk")
        except importlib.metadata.PackageNotFoundError:
            _ver = "dev"
        pk.update(
            {
                "seed": 0,
                "solver": {"method": "BDF"},
                "deterministic": True,
                "model_metadata": {
                    "package_version": _ver,
                    "git_commit": _git or "unknown",
                    "timestamp_utc": datetime.now(tz=timezone.utc).isoformat(),
                },
            }
        )

    json_path = out_path / "summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(pk, f, indent=2, default=str)

    # --- Plot ----------------------------------------------------------------
    plotter = PKPlotter()
    plotter.plot_pk(
        sim_time,
        cp,
        title=f"{drug.name} {dose_mg}mg {route}",
        save_path=out_path / "plots.png",
    )

    typer.echo(f"Results saved to {out_path}/")
    typer.echo(f"  Cmax = {pk['Cmax_mg_L']:.4f} mg/L at Tmax = {pk['Tmax_h']:.2f} h")
    typer.echo(f"  AUC  = {pk['AUC_mg_h_L']:.4f} mg*h/L")
    typer.echo(f"  t1/2 = {pk['half_life_h']} h")
    typer.echo(f"  Mass balance = {pk['mass_balance_pct']:.2f}%")

    # --- Sensitivity analysis ------------------------------------------------
    sens_df = None
    if sensitivity:
        from omega_pbpk.sensitivity import local_sensitivity

        params_display = (
            [p.strip() for p in sensitivity_params.split(",") if p.strip()]
            if sensitivity_params
            else list(_SENS_PARAM_ALIASES.keys())
        )
        params_drug = [_SENS_PARAM_ALIASES.get(p, p) for p in params_display]
        display_map = dict(zip(params_drug, params_display, strict=False))
        sens_result = local_sensitivity(
            drug,
            dose_mg=dose_mg,
            route=route,
            body_weight=bw,
            t_end_h=t_end_h,
            parameters=params_drug,
            rel_step=sensitivity_eps,
            n_workers=1,
        )
        sens_df = sens_result.metrics.copy()
        if len(sens_df) > 0:
            sens_df["parameter"] = sens_df["parameter"].map(lambda x: display_map.get(x, x))
        sens_df.to_csv(out_path / "sensitivity.csv", index=False)
        ranking = {"ranked_parameter_influence": sens_df["parameter"].tolist()}
        (out_path / "sensitivity_ranked.json").write_text(
            json.dumps(ranking, indent=2), encoding="utf-8"
        )
        typer.echo(f"  Sensitivity saved to {out_path}/sensitivity.csv")

    # --- Population PK -------------------------------------------------------
    pop_summary: dict[str, object] = {}
    if population > 0:
        from omega_pbpk.population.pop_simulator import PopulationSimulator

        pop_sim = PopulationSimulator(drug)
        pop_result = pop_sim.run(
            n_subjects=population,
            dose_mg=dose_mg,
            route=route,
            t_end_h=t_end_h,
            n_workers=1,
        )
        _auc_st = pop_result.auc_stats()
        pop_summary = {
            "n_subjects": population,
            "median_auc": _auc_st["median"],
            "cv_auc_pct": _auc_st["cv_pct"],
        }
        typer.echo(f"  Population PK: {population} subjects simulated")

    # --- Academic report -----------------------------------------------------
    if academic_report:
        import importlib.metadata
        from datetime import datetime, timezone

        try:
            _pkg_ver = importlib.metadata.version("omega-pbpk")
        except Exception:
            _pkg_ver = "dev"
        _ts = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        _det_str = "enabled (seed=0, BDF solver)" if deterministic else "disabled"
        if sens_df is not None and len(sens_df) > 0:
            _rows = [
                f"| {r['parameter']} | {r.get('influence_abs_sum', 0.0):.4f} |"
                for _, r in sens_df.iterrows()
            ]
            _sens_tbl = "| Parameter | Influence |\n|-----------|----------|\n" + "\n".join(_rows)
        else:
            _sens_tbl = "_Sensitivity analysis not run._"
        _pop_txt = (
            f"N={pop_summary.get('n_subjects', 0)} subjects simulated. "
            f"Median AUC={pop_summary.get('median_auc', 0.0):.2f} mg*h/L, "
            f"CV={pop_summary.get('cv_auc_pct', 0.0):.1f}%."
            if pop_summary
            else "_Population simulation not run._"
        )
        _rep = (
            f"## Model overview\n\n"
            f"Whole-body PBPK simulation for **{drug.name}** "
            f"({dose_mg} mg {route}) using Omega PBPK v{_pkg_ver}. "
            f"Simulation duration: {t_end_h} h, body weight: {bw} kg.\n\n"
            f"## Reproducibility\n\n"
            f"- Package version: {_pkg_ver}\n"
            f"- Timestamp: {_ts}\n"
            f"- Deterministic mode: {_det_str}\n\n"
            f"## Validation status\n\n"
            f"Key PK outputs:\n"
            f"- Cmax = {pk['Cmax_mg_L']:.4f} mg/L\n"
            f"- AUC = {pk['AUC_mg_h_L']:.4f} mg*h/L\n"
            f"- t1/2 = {pk['half_life_h']} h\n"
            f"- Mass balance = {pk['mass_balance_pct']:.2f}%\n\n"
            f"## Uncertainty results\n\n"
            f"{_pop_txt}\n\n"
            f"## Sensitivity ranking\n\n"
            f"{_sens_tbl}\n\n"
            f"## Limitations\n\n"
            f"- Heuristic tissue partition coefficients (Rodgers-Rowland).\n"
            f"- Linear PK assumed; saturable enzymes not modelled.\n"
            f"- Results are for research purposes only.\n"
        )
        (out_path / "report_academic.md").write_text(_rep, encoding="utf-8")
        typer.echo(f"  Academic report saved to {out_path}/report_academic.md")

    # --- Conformal UQ --------------------------------------------------------
    if uncertainty:
        from omega_pbpk.uncertainty.conformal_uq import (
            ParameterBounds,
            propagate_conformal_intervals,
        )

        bounds = ParameterBounds(
            fup_lo=drug.fup * 0.5,
            fup_hi=min(drug.fup * 2.0, 1.0),
            clint_lo=0.001,
            clint_hi=50.0,
            peff_lo=1e-6,
            peff_hi=5e-4,
            rbp_lo=0.55,
            rbp_hi=2.0,
        )
        uq = propagate_conformal_intervals(
            drug_name=drug.name,
            dose_mg=dose_mg,
            route=route,
            bounds=bounds,
            n_samples=uncertainty_samples,
        )
        typer.echo("\n=== Uncertainty Quantification (Conformal Intervals) ===")
        typer.echo(
            f"Cmax [mg/L]: p5={uq.cmax_p5:.4f}, p50={uq.cmax_p50:.4f}, p95={uq.cmax_p95:.4f}"
        )
        typer.echo(f"AUC  [mg·h/L]: p5={uq.auc_p5:.4f}, p50={uq.auc_p50:.4f}, p95={uq.auc_p95:.4f}")
        typer.echo(f"Tmax [h]:   p5={uq.tmax_p5:.4f}, p50={uq.tmax_p50:.4f}, p95={uq.tmax_p95:.4f}")
        typer.echo(
            f"t½   [h]:   p5={uq.t_half_p5:.4f}, p50={uq.t_half_p50:.4f}, p95={uq.t_half_p95:.4f}"
        )


@app.command()
def predict(
    smiles: str = typer.Option(..., "--smiles", "-s", help="SMILES string of the drug"),
    dose: float = typer.Option(100.0, "--dose", "-d", help="Dose in mg"),
    route: str = typer.Option("oral", "--route", "-r", help="Route: oral or iv"),
    duration: float = typer.Option(24.0, "--duration", help="Simulation duration (h)"),
    model: str = typer.Option(
        "ensemble",
        "--model",
        "-m",
        help="ADME model: 'ensemble' (default), 'admet-ai', or 'legacy'",
    ),
) -> None:
    """Predict PK profile for a new drug molecule from SMILES.

    Uses the ensemble ML predictor by default (ADMET-AI + XGBoost RBP + polynomial).
    Fall back to --model legacy if ML dependencies are not installed.
    """
    _audit("predict", smiles=smiles, dose_mg=dose, route=route, duration_h=duration, model=model)

    import math

    from omega_pbpk.pipeline import OmegaPipeline, SimulationRequest

    # Select ADME predictor based on --model flag
    adme_predictor = _create_adme_predictor(model)

    req = SimulationRequest(smiles=smiles, dose_mg=dose, route=route, duration_h=duration)
    pipeline = OmegaPipeline()

    # If we have an ML predictor, inject it into the pipeline
    if adme_predictor is not None:
        pipeline._adme_predictor = adme_predictor
        pipeline._initialized = True

    typer.echo(f"Running simulation (model={model})...")
    result = pipeline.simulate(req)

    typer.echo("\nOmega PBPK Prediction")
    typer.echo(f"{'=' * 40}")
    typer.echo(f"SMILES:     {smiles[:50]}{'...' if len(smiles) > 50 else ''}")
    typer.echo(f"Dose:       {dose} mg ({route})")
    typer.echo(f"Model:      {model}")
    typer.echo(f"Confidence: {result.confidence}")
    typer.echo("\nPK Summary:")
    typer.echo(f"  Cmax:    {result.cmax_mg_L:.4f} mg/L")
    typer.echo(f"  Tmax:    {result.tmax_h:.2f} h")
    typer.echo(f"  AUC0-t:  {result.auc0t_mg_h_L:.4f} mg·h/L")
    thalf = result.t_half_h
    typer.echo(f"  t½:      {'nan' if math.isnan(thalf) else f'{thalf:.2f} h'}")
    typer.echo("\nPredicted ADME:")
    for k, v in result.adme_properties.items():
        if k != "confidence" and isinstance(v, (int, float)):
            typer.echo(f"  {k:<12}: {v:.3g}")

    # Show uncertainty intervals if available
    interval_keys = [
        ("fup", "fup_lo", "fup_hi"),
        ("clint_3a4", "clint_3a4_lo", "clint_3a4_hi"),
        ("peff", "peff_lo", "peff_hi"),
        ("rbp", "rbp_lo", "rbp_hi"),
    ]
    has_intervals = False
    for _, lo_k, hi_k in interval_keys:
        lo_v = result.adme_properties.get(lo_k, 0.0)
        hi_v = result.adme_properties.get(hi_k, 0.0)
        if isinstance(lo_v, (int, float)) and isinstance(hi_v, (int, float)) and hi_v > lo_v > 0:
            has_intervals = True
            break
    if has_intervals:
        typer.echo("\nUncertainty Intervals (90% CI):")
        for prop, lo_k, hi_k in interval_keys:
            lo_v = result.adme_properties.get(lo_k, 0.0)
            hi_v = result.adme_properties.get(hi_k, 0.0)
            if isinstance(lo_v, (int, float)) and isinstance(hi_v, (int, float)) and hi_v > lo_v:
                typer.echo(f"  {prop:<12}: [{lo_v:.4g}, {hi_v:.4g}]")

    if result.warnings:
        typer.echo("\nWarnings:")
        for w in result.warnings:
            typer.echo(f"  ! {w}")


def _create_adme_predictor(model_name: str) -> object | None:
    """Create the requested ADME predictor.

    Args:
        model_name: One of 'ensemble', 'admet-ai', or 'legacy'.

    Returns:
        Predictor instance, or None to use pipeline default (legacy).
    """
    if model_name == "legacy":
        return None  # Pipeline uses its own ADMEPredictor

    if model_name == "admet-ai":
        try:
            from omega_pbpk.ml.models.adme.admet_ai_wrapper import ADMETAIPredictor

            return ADMETAIPredictor()
        except ImportError:
            logger.warning(
                "admet-ai not installed; falling back to legacy predictor. "
                "Install with: pip install admet-ai"
            )
            return None

    if model_name == "ensemble":
        try:
            from omega_pbpk.ml.models.adme.ensemble import EnsembleADMEPredictor

            return EnsembleADMEPredictor()
        except Exception as exc:
            logger.warning(
                "Ensemble predictor failed to initialize (%s); falling back to legacy.",
                exc,
            )
            return None

    logger.warning("Unknown model '%s'; using legacy predictor.", model_name)
    return None


@app.command("smiles-report")
def smiles_report(
    smiles: str = typer.Option(..., "--smiles", "-s", help="SMILES string of the drug"),
    name: str = typer.Option("Unknown", "--name", "-n", help="Drug name (optional label)"),
    dose: float = typer.Option(100.0, "--dose", "-d", help="Dose in mg"),
    route: str = typer.Option("oral", "--route", "-r", help="Route: oral or iv"),
    duration: float = typer.Option(24.0, "--duration", help="Simulation duration (h)"),
    output: str = typer.Option("", "--output", "-o", help="Save JSON report to this path"),
    no_uq: bool = typer.Option(False, "--no-uq", help="Skip uncertainty quantification"),
) -> None:
    """Generate a complete SMILES→PK assessment report (all subsystems).

    Runs the full pipeline: ADME prediction → PBPK simulation → uncertainty
    quantification → transporter profiling → BCS classification → risk flags.
    Outputs a structured text report; optionally saves JSON with --output.
    """
    import math
    from datetime import datetime, timezone

    _audit("smiles-report", smiles=smiles, name=name, dose_mg=dose, route=route)

    DIVIDER = "=" * 60
    SECTION = "-" * 60

    def _fmt(v: float, decimals: int = 4) -> str:
        return "nan" if (isinstance(v, float) and math.isnan(v)) else f"{v:.{decimals}f}"

    def _sparkline(values: list[float], width: int = 40) -> str:
        """ASCII sparkline using block chars."""
        blocks = " ▁▂▃▄▅▆▇█"
        if not values or max(values) == 0:
            return " " * width
        _, mx = 0.0, max(values)
        step = mx / (len(blocks) - 1)
        sampled = [values[int(i * (len(values) - 1) / (width - 1))] for i in range(width)]
        return "".join(blocks[min(int(v / step), len(blocks) - 1)] for v in sampled)

    typer.echo(DIVIDER)
    typer.echo("  Omega PBPK — Drug Assessment Report")
    typer.echo(DIVIDER)
    typer.echo(f"  Date:   {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    typer.echo(f"  Name:   {name}")
    typer.echo(f"  SMILES: {smiles[:55]}{'...' if len(smiles) > 55 else ''}")
    typer.echo(f"  Dose:   {dose} mg  |  Route: {route}  |  Duration: {duration} h")
    typer.echo(SECTION)

    # ------------------------------------------------------------------
    # 1. Full pipeline
    # ------------------------------------------------------------------
    typer.echo("  [1/4] Running ADME prediction + PBPK simulation...")
    from omega_pbpk.prediction.full_pipeline import run_full_prediction

    n_uq = 0 if no_uq else 50
    try:
        fp = run_full_prediction(
            smiles=smiles,
            dose_mg=dose,
            route=route,
            duration_h=duration,
            n_uq_samples=n_uq,
        )
    except Exception as exc:
        typer.echo(f"  ERROR: Full pipeline failed: {exc}", err=True)
        raise typer.Exit(1) from exc

    # ------------------------------------------------------------------
    # 2. BCS classification from ADME output
    # ------------------------------------------------------------------
    typer.echo("  [2/4] BCS classification...")
    bcs_class = "N/A"
    bcs_biowaiver = False
    try:
        from omega_pbpk.biopharmaceutics.bcs_classification import classify_bcs

        adme = fp.adme
        mw = float(adme.get("mw", 300.0))
        logs = float(adme.get("logS", -3.0))
        peff_pred = float(adme.get("peff", 1.0))
        # Convert logS (log mol/L) → mg/mL: S_mg_mL = 10^logS × MW / 1000
        sol_mg_mL = (10.0**logs) * mw / 1000.0
        # peff from ADME is in 10^-4 cm/s units; convert to cm/s
        peff_cm_s = peff_pred * 1e-4
        bcs_class = classify_bcs(
            solubility_mg_per_mL=max(sol_mg_mL, 1e-6),
            dose_mg=dose,
            permeability_cm_per_s=peff_cm_s,
        )
        bcs_biowaiver = bcs_class in ("I", "III")
    except Exception as exc:
        typer.echo(f"  Warning: BCS classification failed: {exc}")

    # ------------------------------------------------------------------
    # 3. Report assembly
    # ------------------------------------------------------------------
    typer.echo("  [3/4] Assembling report...")
    typer.echo("  [4/4] Done.\n")

    # --- PK Summary ---
    typer.echo(DIVIDER)
    typer.echo("  PK SUMMARY")
    typer.echo(SECTION)
    typer.echo(f"  Cmax      : {_fmt(fp.cmax_mg_L)} mg/L")
    typer.echo(f"  Tmax      : {_fmt(fp.tmax_h, 2)} h")
    typer.echo(f"  AUC(0-t)  : {_fmt(fp.auc0t_mg_h_L)} mg·h/L")
    typer.echo(f"  t½        : {_fmt(fp.t_half_h, 2)} h")

    # Concentration-time sparkline
    if fp.cp_mg_L:
        spark = _sparkline(list(fp.cp_mg_L))
        typer.echo(f"\n  C(t) profile (0 → {duration:.0f} h):")
        typer.echo(f"  |{spark}|")
        typer.echo(f"  0{' ' * 18}{duration / 2:.0f} h{' ' * 16}{duration:.0f} h")

    # --- Uncertainty Quantification ---
    if not no_uq and fp.cmax_p5 > 0:
        typer.echo(f"\n{SECTION}")
        typer.echo("  UNCERTAINTY QUANTIFICATION (p5 / p50 / p95)")
        typer.echo(SECTION)
        typer.echo(
            f"  Cmax [mg/L]   : {_fmt(fp.cmax_p5)} / {_fmt(fp.cmax_p50)} / {_fmt(fp.cmax_p95)}"
        )
        typer.echo(f"  AUC  [mg·h/L] : {_fmt(fp.auc_p5)} / {_fmt(fp.auc_p50)} / {_fmt(fp.auc_p95)}")

    # --- ADME Properties ---
    typer.echo(f"\n{SECTION}")
    typer.echo("  PREDICTED ADME PROPERTIES")
    typer.echo(SECTION)
    adme_labels = {
        "mw": ("MW", "Da"),
        "logP": ("logP", ""),
        "logS": ("logS", "log mol/L"),
        "peff": ("Peff", "×10⁻⁴ cm/s"),
        "fup": ("fup", ""),
        "rbp": ("RBP", ""),
        "clint_3a4": ("CLint CYP3A4", "µL/min/pmol"),
        "clint_2d6": ("CLint CYP2D6", "µL/min/pmol"),
        "herg_ic50_uM": ("hERG IC50", "µM"),
    }
    for key, (label, unit) in adme_labels.items():
        v = fp.adme.get(key)
        if v is not None:
            suffix = f" {unit}" if unit else ""
            typer.echo(f"  {label:<20}: {v:.3g}{suffix}")
    typer.echo(f"  {'ADME confidence':<20}: {fp.adme_confidence}")

    # --- BCS ---
    typer.echo(f"\n{SECTION}")
    typer.echo("  BCS CLASSIFICATION")
    typer.echo(SECTION)
    bcs_names = {
        "I": "Class I  (High sol / High perm)",
        "II": "Class II (Low sol / High perm)",
        "III": "Class III (High sol / Low perm)",
        "IV": "Class IV (Low sol / Low perm)",
    }
    typer.echo(f"  BCS Class    : {bcs_names.get(bcs_class, bcs_class)}")
    typer.echo(f"  Biowaiver    : {'Potentially eligible' if bcs_biowaiver else 'Not eligible'}")

    # --- Transporters ---
    typer.echo(f"\n{SECTION}")
    typer.echo("  TRANSPORTER PROFILE")
    typer.echo(SECTION)
    typer.echo(f"  Substrates   : {', '.join(fp.transporter_substrates) or 'None predicted'}")
    typer.echo(f"  Inhibitors   : {', '.join(fp.transporter_inhibitors) or 'None predicted'}")
    if fp.transporter_ddi_flags:
        typer.echo(f"  DDI flags    : {', '.join(fp.transporter_ddi_flags)}")
    typer.echo(f"  Confidence   : {fp.transporter_confidence}")

    # --- Risk Assessment ---
    typer.echo(f"\n{SECTION}")
    typer.echo("  RISK ASSESSMENT")
    typer.echo(SECTION)
    if fp.risk_flags:
        for flag, active in fp.risk_flags.items():
            marker = "[!]" if active else "[ ]"
            typer.echo(f"  {marker} {flag.replace('_', ' ').title()}")
    else:
        typer.echo("  No risk flags generated.")
    risk_color = {"low": "GREEN", "moderate": "YELLOW", "high": "RED"}.get(
        fp.overall_risk_level, ""
    )
    typer.echo(f"\n  Overall risk : {fp.overall_risk_level.upper()} [{risk_color}]")
    typer.echo(f"  Active flags : {fp.risk_count}")

    # --- Warnings ---
    # Filter UQ warnings when user explicitly skipped UQ
    visible_warnings = list(fp.warnings)
    if no_uq:
        visible_warnings = [w for w in visible_warnings if "UQ" not in w and "uq" not in w.lower()]
    if visible_warnings:
        typer.echo(f"\n{SECTION}")
        typer.echo("  WARNINGS")
        typer.echo(SECTION)
        for w in visible_warnings:
            typer.echo(f"  ! {w}")

    typer.echo(f"\n{DIVIDER}")
    typer.echo(
        f"  PubChem enriched: {fp.pubchem_enriched}  |  Pipeline confidence: {fp.confidence}"
    )
    typer.echo(DIVIDER)

    # ------------------------------------------------------------------
    # 4. Optional JSON export
    # ------------------------------------------------------------------
    if output:
        report_dict: dict[str, object] = {
            "name": name,
            "smiles": smiles,
            "dose_mg": dose,
            "route": route,
            "duration_h": duration,
            "pk_summary": {
                "cmax_mg_L": fp.cmax_mg_L,
                "tmax_h": fp.tmax_h,
                "auc0t_mg_h_L": fp.auc0t_mg_h_L,
                "t_half_h": fp.t_half_h,
            },
            "uq": {
                "cmax_p5": fp.cmax_p5,
                "cmax_p50": fp.cmax_p50,
                "cmax_p95": fp.cmax_p95,
                "auc_p5": fp.auc_p5,
                "auc_p50": fp.auc_p50,
                "auc_p95": fp.auc_p95,
            },
            "adme": fp.adme,
            "adme_confidence": fp.adme_confidence,
            "bcs_class": bcs_class,
            "bcs_biowaiver_eligible": bcs_biowaiver,
            "transporters": {
                "substrates": list(fp.transporter_substrates),
                "inhibitors": list(fp.transporter_inhibitors),
                "ddi_flags": list(fp.transporter_ddi_flags),
                "confidence": fp.transporter_confidence,
            },
            "risk": {
                "flags": fp.risk_flags,
                "risk_count": fp.risk_count,
                "overall_risk_level": fp.overall_risk_level,
            },
            "pubchem_enriched": fp.pubchem_enriched,
            "confidence": fp.confidence,
            "warnings": list(fp.warnings),
        }
        out_path = Path(output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report_dict, indent=2, default=str), encoding="utf-8")
        typer.echo(f"\n  Report saved: {out_path}")


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


@app.command("pgx-sim")
def pgx_sim(
    smiles: str = typer.Option(..., "--smiles", "-s", help="SMILES string of the drug."),
    dose: float = typer.Option(100.0, "--dose", "-d", help="Dose in mg."),
    gene: str = typer.Option(
        "CYP2D6", "--gene", "-g", help="CYP gene (CYP2D6, CYP2C19, CYP2C9, CYP3A5, CYP1A2)."
    ),
    route: str = typer.Option("oral", "--route", "-r", help="Route: oral or iv."),
    t_end_h: float = typer.Option(24.0, "--t-end", help="Simulation end time (h)."),
    body_weight: float = typer.Option(70.0, "--bw", help="Body weight (kg)."),
    out: str | None = typer.Option(
        None, "--out", "-o", help="Output HTML report path (e.g. pgx_report.html)."
    ),
) -> None:
    """PGx-stratified PBPK simulation — run PBPK for each CYP phenotype (PM/IM/NM/UM).

    Predicts ADME properties from SMILES, then runs four phenotype-stratified
    PBPK simulations scaling CLint by the CPIC activity-based factor for each
    phenotype of the specified gene.  Prints a summary table of AUC ratios and
    optionally saves an HTML report with an embedded forest plot.

    Example:

        omega pgx-sim --smiles "CCc1ccc(NC(=O)c2cc(OC)c(OC)c(OC)c2)cc1" --dose 100 \
            --gene CYP2D6 --out pgx_report.html
    """
    from omega_pbpk.clinical.pgx_pbpk import pgx_report_html, run_pgx_pbpk
    from omega_pbpk.drugs.drug import Drug
    from omega_pbpk.prediction.adme_predictor import ADMEPredictor

    # Predict ADME from SMILES
    typer.echo(f"Predicting ADME for SMILES: {smiles[:60]}{'...' if len(smiles) > 60 else ''}")
    predictor = ADMEPredictor()
    props = predictor.predict(smiles)
    drug = Drug(
        name="PGxCompound",
        mw=props.mw,
        logP=props.logP,
        fup=props.fup,
        rbp=props.rbp,
        peff=props.peff,
        clint={"CYP3A4": props.clint_3a4, "CYP2D6": props.clint_2d6},
        clint_hepatic_L_per_h=props.clint_3a4 * 40.0 * 45.0 * 1800.0 / 1e6 / 60.0,
    )
    typer.echo(
        f"  MW={props.mw:.1f}, logP={props.logP:.2f}, fup={props.fup:.3f}, rbp={props.rbp:.2f}"
    )

    # Run PGx-stratified PBPK
    typer.echo(f"Running PGx-PBPK ({gene}) — {dose} mg {route}, {t_end_h} h, {body_weight} kg...")
    result = run_pgx_pbpk(
        drug=drug,
        gene=gene,
        dose_mg=dose,
        route=route,
        t_end_h=t_end_h,
        body_weight=body_weight,
    )

    # Print summary table
    typer.echo(f"\nPGx-PBPK Results — {gene}")
    hdr = (
        f"  {'Phenotype':<10}  {'AUC (mg·h/L)':>14}"
        f"  {'Cmax (mg/L)':>12}  {'AUC Ratio':>10}  {'Pop. Freq.':>10}"
    )
    typer.echo(hdr)
    typer.echo("  " + "-" * 64)
    for rec in result.phenotype_results:
        typer.echo(
            f"  {rec['phenotype']:<10}"
            f"  {rec['auc_mg_h_L']:>14.4f}"
            f"  {rec['cmax_mg_L']:>12.4f}"
            f"  {rec['auc_ratio_vs_nm']:>10.3f}"
            f"  {rec['population_frequency'] * 100:>9.1f}%"
        )
    typer.echo(f"\n  NM AUC reference    : {result.nm_auc:.4f} mg·h/L")
    typer.echo(f"  Gene Sensitivity Index: {result.gene_sensitivity_index:.2f}x")

    # Optional HTML report
    if out:
        html = pgx_report_html(result, drug_name="PGxCompound")
        with open(out, "w") as fh:
            fh.write(html)
        typer.echo(f"\nHTML report saved to: {out}")


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
    _audit(
        "calibrate",
        compound=compound,
        observed=observed,
        dose_mg=dose_mg,
        route=route,
        body_weight=body_weight,
        n_samples=n_samples,
        seed=seed,
        out=out,
    )

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


@app.command("validate-legacy", hidden=True)
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
    train: bool = typer.Option(False, "--train", help="Train surrogate on real PBPK simulations"),
    n_samples: int = typer.Option(100, "--n-samples", help="Number of training simulations"),
    save: str = typer.Option("models/surrogate_real.npz", "--save", help="Output path for weights"),
    epochs: int = typer.Option(300, "--epochs", help="Training epochs"),
) -> None:
    """Train or use the neural surrogate model for fast PBPK prediction."""
    if train:
        import logging

        logging.basicConfig(level=logging.INFO)
        typer.echo(f"Training surrogate on {n_samples} real PBPK simulations...")
        import os

        from omega_pbpk.surrogate.train import train_surrogate

        os.makedirs(os.path.dirname(save) if os.path.dirname(save) else ".", exist_ok=True)
        model = train_surrogate(n_samples=n_samples, save_path=save, epochs=epochs)
        typer.echo(f"Surrogate trained and saved to {save}")
        typer.echo(f"  Input features: {model.n_input}, Output targets: {model.n_output}")
    else:
        typer.echo("Use --train to train a new surrogate model.")
        typer.echo("Example: omega surrogate --train --n-samples 100")


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
    _audit("evaluate", compound=compound, dose_mg=dose_mg, route=route, out=out)

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


@app.command()
def population(
    compound: str = typer.Option(..., "--compound", "-c", help="Path to compound YAML"),
    n_subjects: int = typer.Option(50, "--n-subjects", "-n", help="Number of virtual subjects"),
    dose: float = typer.Option(100.0, "--dose", "-d", help="Dose in mg"),
    route: str = typer.Option("oral", "--route", "-r", help="Route: oral or iv"),
    duration: float = typer.Option(24.0, "--duration", help="Simulation duration (h)"),
    seed: int = typer.Option(42, "--seed", help="Random seed"),
) -> None:
    """Run population PK simulation across N virtual subjects."""
    _audit(
        "population",
        compound=compound,
        n_subjects=n_subjects,
        dose_mg=dose,
        route=route,
        duration_h=duration,
        seed=seed,
    )

    from omega_pbpk.config import load_compound
    from omega_pbpk.population.pop_simulator import PopulationSimulator

    typer.echo(f"Loading compound from {compound}...")
    drug = load_compound(compound)
    typer.echo(f"Running PopPK for {n_subjects} subjects ({route}, {dose}mg)...")

    sim = PopulationSimulator(drug=drug)
    result = sim.run(n_subjects=n_subjects, dose_mg=dose, route=route, t_end_h=duration, seed=seed)

    cmax = result.cmax_stats()
    auc = result.auc_stats()

    typer.echo(f"\nPopulation PK Results (n={result.n_subjects})")
    typer.echo(f"{'=' * 45}")
    typer.echo(
        f"Cmax (mg/L):  median={cmax['median']:.3f}, p5={cmax['p5']:.3f},"
        f" p95={cmax['p95']:.3f}, CV={cmax['cv_pct']:.1f}%"
    )
    typer.echo(
        f"AUC  (mg·h/L): median={auc['median']:.3f}, p5={auc['p5']:.3f},"
        f" p95={auc['p95']:.3f}, CV={auc['cv_pct']:.1f}%"
    )
    if result.n_failed > 0:
        typer.echo(f"Warning: {result.n_failed} subjects failed simulation")


@app.command()
def report(
    smiles: str = typer.Option(..., "--smiles", "-s", help="SMILES string"),
    name: str = typer.Option("compound", "--name", help="Drug display name"),
    dose: float = typer.Option(100.0, "--dose", "-d", help="Dose in mg"),
    route: str = typer.Option("oral", "--route", "-r", help="Route: oral or iv"),
    out: str = typer.Option("omega_report.html", "--out", "-o", help="Output HTML path"),
    population: int = typer.Option(0, "--population", "-p", help="N subjects for PopPK (0=skip)"),
) -> None:
    """Generate a regulatory-grade HTML report (NCA + DDI + PopPK)."""
    _audit(
        "report",
        smiles=smiles,
        name=name,
        dose_mg=dose,
        route=route,
        out=out,
        population=population,
    )

    from omega_pbpk.clinical.report import quick_report

    typer.echo(f"Generating report for {name}...")
    path = quick_report(
        smiles=smiles,
        drug_name=name,
        dose_mg=dose,
        route=route,
        output_path=out,
        n_pop_subjects=population,
    )
    typer.echo(f"Report saved to: {path}")


@app.command("invitro")
def invitro_cmd(
    smiles: str = typer.Option(..., "--smiles", "-s", help="SMILES string of the drug."),
    assay_type: str = typer.Option(
        "hepatocyte",
        "--assay-type",
        "-a",
        help="Assay type: 'microsomes' (Phase 1 only) or 'hepatocyte' (Phase 1+2).",
    ),
    c0_uM: float = typer.Option(1.0, "--c0", help="Initial drug concentration in well (µM)."),
    protein_conc: float = typer.Option(
        0.5, "--protein-conc", help="Microsomal protein (mg/mL) or cell density (10⁶/mL)."
    ),
    caco2: bool = typer.Option(True, "--caco2/--no-caco2", help="Run Caco-2 permeability assay."),
    out: str | None = typer.Option(None, "--out", "-o", help="Output JSON path (optional)."),
) -> None:
    """Simulate in vitro metabolic stability assay from SMILES.

    Predicts ADME + Phase 2 CLint from SMILES, runs a simulated microsomal
    or hepatocyte substrate-depletion assay, and optionally runs a Caco-2
    bidirectional permeability simulation.

    Example::

        omega invitro --smiles "Cn1cnc2c1c(=O)n(C)c(=O)n2C" --assay-type hepatocyte --caco2
    """
    _audit("invitro", smiles=smiles, assay_type=assay_type, c0_uM=c0_uM, caco2=caco2)

    from omega_pbpk.in_vitro.pipeline import InVitroPipeline, InVitroRequest

    req = InVitroRequest(
        smiles=smiles,
        assay_type=assay_type,
        c0_uM=c0_uM,
        run_caco2=caco2,
        protein_conc_mg_mL=protein_conc if assay_type == "microsomes" else 0.5,
        cell_density_million_per_mL=protein_conc if assay_type == "hepatocyte" else 1.0,
    )

    typer.echo("Running in vitro assay simulation...")
    pipeline = InVitroPipeline()
    result = pipeline.run(req)

    typer.echo("\nIn Vitro Metabolic Stability")
    typer.echo(f"{'=' * 45}")
    typer.echo(f"SMILES:       {smiles[:55]}{'...' if len(smiles) > 55 else ''}")
    typer.echo(f"Assay type:   {assay_type}")
    typer.echo(f"Confidence:   {result.confidence}")
    typer.echo()
    typer.echo("ADME Properties:")
    typer.echo(f"  MW       = {result.adme.mw:.1f} Da")
    typer.echo(f"  logP     = {result.adme.logP:.2f}")
    typer.echo(f"  fup      = {result.adme.fup:.3f}")
    typer.echo(f"  hERG IC50= {result.adme.herg_ic50_uM:.2f} µM")
    typer.echo()
    typer.echo("Clearance:")
    typer.echo(f"  CLint Phase 1 = {result.clint_phase1_ul_min_mg:.3f} µL/min/mg")
    typer.echo(f"  CLint Phase 2 = {result.clint_phase2_ul_min_mg:.3f} µL/min/mg")
    typer.echo(f"  CLint total   = {result.clint_total_ul_min_mg:.3f} µL/min/mg")
    typer.echo(f"  t½ in assay   = {result.assay.t_half_min:.1f} min")
    typer.echo(f"  CLh (in vivo) = {result.clh_l_per_h:.3f} L/h")
    typer.echo()
    typer.echo("Phase 2 pathways:")
    typer.echo(
        f"  UGT  = {result.phase2.clint_ugt_ul_min_mg:.3f} µL/min/mg"
        f"  (prob={result.phase2.ugt_substrate_prob:.2f})"
    )
    typer.echo(
        f"  SULT = {result.phase2.clint_sult_ul_min_mg:.3f} µL/min/mg"
        f"  (prob={result.phase2.sult_substrate_prob:.2f})"
    )
    typer.echo(
        f"  MAO  = {result.phase2.clint_mao_ul_min_mg:.3f} µL/min/mg"
        f"  (prob={result.phase2.mao_substrate_prob:.2f})"
    )
    typer.echo(f"  Dominant pathway: {result.phase2.dominant_pathway}")

    if result.caco2 is not None:
        typer.echo()
        typer.echo("Caco-2 Permeability:")
        typer.echo(f"  Papp(A→B) = {result.caco2.papp_ab_cm_s:.2f} ×10⁻⁶ cm/s")
        typer.echo(f"  Papp(B→A) = {result.caco2.papp_ba_cm_s:.2f} ×10⁻⁶ cm/s")
        typer.echo(f"  ER        = {result.caco2.efflux_ratio:.2f}")
        typer.echo(f"  Flag      : {result.caco2.pgp_bcrp_flag}")

    if result.warnings:
        typer.echo()
        typer.echo("Warnings:")
        for w in result.warnings:
            typer.echo(f"  ! {w}")

    if out:
        summary = {
            "smiles": smiles,
            "assay_type": assay_type,
            "confidence": result.confidence,
            "adme": {
                "mw": result.adme.mw,
                "logP": result.adme.logP,
                "fup": result.adme.fup,
                "herg_ic50_uM": result.adme.herg_ic50_uM,
            },
            "phase1_clint_ul_min_mg": result.clint_phase1_ul_min_mg,
            "phase2_clint_ul_min_mg": result.clint_phase2_ul_min_mg,
            "total_clint_ul_min_mg": result.clint_total_ul_min_mg,
            "t_half_assay_min": result.assay.t_half_min,
            "clh_l_per_h": result.clh_l_per_h,
            "phase2": {
                "ugt": result.phase2.clint_ugt_ul_min_mg,
                "sult": result.phase2.clint_sult_ul_min_mg,
                "mao": result.phase2.clint_mao_ul_min_mg,
                "dominant": result.phase2.dominant_pathway,
            },
            "caco2": {
                "papp_ab": result.caco2.papp_ab_cm_s,
                "papp_ba": result.caco2.papp_ba_cm_s,
                "efflux_ratio": result.caco2.efflux_ratio,
                "flag": result.caco2.pgp_bcrp_flag,
            }
            if result.caco2
            else None,
            "warnings": result.warnings,
        }
        with open(out, "w") as f:
            json.dump(summary, f, indent=2)
        typer.echo(f"\nResults saved to {out}")


@app.command("invitro-pipeline")
def invitro_pipeline_cmd(
    smiles: str = typer.Option(..., "--smiles", "-s", help="SMILES string of the drug."),
    name: str = typer.Option("compound", "--name", "-n", help="Drug display name."),
    dose: float = typer.Option(100.0, "--dose", "-d", help="Dose for PBPK simulation (mg)."),
    assay_type: str = typer.Option(
        "hepatocyte",
        "--assay-type",
        "-a",
        help="Assay: 'microsomes' or 'hepatocyte'.",
    ),
    kd: float | None = typer.Option(
        None, "--kd", help="Receptor Kd (mg/L) for occupancy/EC50 estimation."
    ),
    tau: float = typer.Option(1.0, "--tau", help="Operational model transduction ratio."),
    caco2: bool = typer.Option(True, "--caco2/--no-caco2", help="Run Caco-2 assay."),
    out: str = typer.Option("outputs/invitro", "--out", "-o", help="Output directory."),
) -> None:
    """Full in vitro → IVIVE → Drug object pipeline from SMILES.

    Runs all 7 pipeline stages (ADME → Phase 2 → Assay → Caco-2 → Receptor
    → IVIVE → Drug) and saves the resulting Drug YAML and a JSON summary.
    The Drug object can be directly used for PBPK simulation with omega simulate.

    Example::

        omega invitro-pipeline --smiles "Cn1cnc2c1c(=O)n(C)c(=O)n2C" \
            --name caffeine --dose 200 --out outputs/caffeine_iv
    """
    _audit(
        "invitro-pipeline",
        smiles=smiles,
        name=name,
        dose_mg=dose,
        assay_type=assay_type,
        kd_mg_L=kd,
        tau=tau,
        caco2=caco2,
        out=out,
    )

    import yaml

    from omega_pbpk.in_vitro.pipeline import InVitroPipeline, InVitroRequest

    req = InVitroRequest(
        smiles=smiles,
        dose_mg=dose,
        assay_type=assay_type,
        run_caco2=caco2,
        kd_mg_L=kd,
        tau=tau,
        drug_name=name,
    )

    typer.echo(f"Running in vitro → IVIVE pipeline for {name}...")
    pipeline = InVitroPipeline()
    result = pipeline.run(req)

    typer.echo(f"\nIn Vitro → IVIVE Pipeline: {name}")
    typer.echo(f"{'=' * 55}")
    typer.echo(f"SMILES:        {smiles[:50]}{'...' if len(smiles) > 50 else ''}")
    typer.echo(f"Assay type:    {assay_type}")
    typer.echo(f"Confidence:    {result.confidence}")
    typer.echo()
    typer.echo("Predicted Properties:")
    typer.echo(f"  MW               = {result.adme.mw:.1f} Da")
    typer.echo(f"  logP             = {result.adme.logP:.2f}")
    typer.echo(f"  fup              = {result.adme.fup:.3f}")
    typer.echo(f"  rbp              = {result.adme.rbp:.3f}")
    typer.echo(f"  Peff             = {result.adme.peff:.4f} ×10⁻⁴ cm/s")
    typer.echo(f"  hERG IC50        = {result.adme.herg_ic50_uM:.2f} µM")
    typer.echo()
    typer.echo("Clearance (IVIVE):")
    typer.echo(f"  Phase 1 CLint    = {result.clint_phase1_ul_min_mg:.3f} µL/min/mg")
    typer.echo(f"  Phase 2 CLint    = {result.clint_phase2_ul_min_mg:.3f} µL/min/mg")
    typer.echo(f"    └ dominant P2  : {result.phase2.dominant_pathway}")
    typer.echo(f"  Total CLint      = {result.clint_total_ul_min_mg:.3f} µL/min/mg")
    typer.echo(f"  t½ (in vitro)    = {result.assay.t_half_min:.1f} min")
    typer.echo(f"  CLint (liver)    = {result.assay.clint_liver_L_per_h:.3f} L/h")
    typer.echo(f"  CLh (well-stirr) = {result.clh_l_per_h:.3f} L/h")

    if result.caco2:
        typer.echo()
        typer.echo("Caco-2:")
        typer.echo(f"  ER = {result.caco2.efflux_ratio:.2f}  ({result.caco2.pgp_bcrp_flag})")

    if result.receptor:
        typer.echo()
        typer.echo("Receptor / PD:")
        typer.echo(
            f"  EC50  = {result.receptor.ec50_mg_L:.4f} mg/L (R²={result.receptor.r_squared:.3f})"
        )
        typer.echo(f"  Emax  = {result.receptor.emax_effect:.1f}")

    if result.warnings:
        typer.echo()
        typer.echo("Warnings:")
        for w in result.warnings:
            typer.echo(f"  ! {w}")

    # --- Save outputs ---
    out_path = _ensure_dir(Path(out))

    # Drug YAML for use with omega simulate --compound
    drug = result.drug
    drug_dict = {
        "name": drug.name,
        "mw": drug.mw,
        "logP": drug.logP,
        "fup": drug.fup,
        "rbp": drug.rbp,
        "peff": drug.peff,
        "clint_hepatic_L_per_h": drug.clint_hepatic_L_per_h,
        "clint": drug.clint if hasattr(drug, "clint") else {},
    }
    drug_yaml_path = out_path / f"{name}.yaml"
    with open(drug_yaml_path, "w") as f:
        yaml.dump(drug_dict, f, default_flow_style=False)

    # Full JSON summary
    summary = {
        "drug_name": name,
        "smiles": smiles,
        "assay_type": assay_type,
        "confidence": result.confidence,
        "adme": {
            "mw": result.adme.mw,
            "logP": result.adme.logP,
            "fup": result.adme.fup,
            "rbp": result.adme.rbp,
            "peff": result.adme.peff,
            "clint_3a4": result.adme.clint_3a4,
            "clint_2d6": result.adme.clint_2d6,
            "herg_ic50_uM": result.adme.herg_ic50_uM,
            "confidence": result.adme.confidence,
        },
        "phase2": {
            "clint_ugt": result.phase2.clint_ugt_ul_min_mg,
            "clint_sult": result.phase2.clint_sult_ul_min_mg,
            "clint_mao": result.phase2.clint_mao_ul_min_mg,
            "total": result.phase2.clint_phase2_total_ul_min_mg,
            "dominant": result.phase2.dominant_pathway,
            "confidence": result.phase2.confidence,
        },
        "assay": {
            "type": assay_type,
            "k_dep_per_h": result.assay.k_dep_per_h,
            "t_half_min": result.assay.t_half_min,
            "clint_ul_min_per_system": result.assay.clint_ul_min_per_system,
            "clint_liver_L_per_h": result.assay.clint_liver_L_per_h,
        },
        "ivive": {
            "clint_phase1_ul_min_mg": result.clint_phase1_ul_min_mg,
            "clint_phase2_ul_min_mg": result.clint_phase2_ul_min_mg,
            "clint_total_ul_min_mg": result.clint_total_ul_min_mg,
            "clh_l_per_h": result.clh_l_per_h,
        },
        "caco2": {
            "papp_ab_cm_s": result.caco2.papp_ab_cm_s,
            "papp_ba_cm_s": result.caco2.papp_ba_cm_s,
            "efflux_ratio": result.caco2.efflux_ratio,
            "flag": result.caco2.pgp_bcrp_flag,
        }
        if result.caco2
        else None,
        "receptor": {
            "ec50_mg_L": result.receptor.ec50_mg_L,
            "emax_effect": result.receptor.emax_effect,
            "hill": result.receptor.hill,
            "r_squared": result.receptor.r_squared,
        }
        if result.receptor
        else None,
        "warnings": result.warnings,
    }
    json_path = out_path / "invitro_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    typer.echo(f"\nOutputs saved to {out_path}/")
    typer.echo(f"  Drug YAML : {drug_yaml_path.name}  (use with: omega simulate --compound)")
    typer.echo("  Summary   : invitro_summary.json")


@app.command("serve-legacy", hidden=True)
def serve(
    host: str = typer.Option("0.0.0.0", help="Host to bind the server to."),
    port: int = typer.Option(8000, help="Port to listen on."),
    reload: bool = typer.Option(False, help="Enable auto-reload on code changes."),
) -> None:
    """Start the FastAPI PBPK REST API server."""
    try:
        import uvicorn
    except ImportError:
        typer.echo("API server requires [api] extras. Install with: pip install omega-pbpk[api]")
        raise SystemExit(1) from None

    typer.echo(f"Starting Omega PBPK API server on http://{host}:{port}")
    uvicorn.run("omega_pbpk.api.app:app", host=host, port=port, reload=reload)


@app.command("test")
def run_tests() -> None:
    """Run the test suite."""
    import subprocess

    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "-v", "--tb=short"],
        cwd=str(Path(__file__).parent.parent.parent),
    )
    raise typer.Exit(result.returncode)


# ============================================================================
# M3: 4-verb sub-app commands
# ============================================================================

# --- simulate group ----------------------------------------------------------


def _run_single_sim(
    compound: str,
    dose: float,
    route: str,
    t_end: float,
    json_output: bool,
) -> None:
    """Shared logic for single-dose PBPK simulation."""
    _audit("simulate.single", compound=compound, dose_mg=dose, route=route, t_end_h=t_end)

    from omega_pbpk.config import load_compound
    from omega_pbpk.core.body import WholeBodyPBPK

    drug = load_compound(compound)
    model = WholeBodyPBPK(drug, body_weight=70.0)
    if route == "iv":
        model.setup_iv(dose)
    else:
        model.setup_oral(dose)

    result = model.simulate(t_end_h=t_end)
    pk = result.pk_summary()

    if json_output:
        _output(pk, json_output=True)
    else:
        # Delegate to full legacy simulate command for file outputs
        simulate(
            compound=compound,
            dose_mg=dose,
            route=route,
            t_end_h=t_end,
            body_weight=70.0,
            smiles=None,
            out="outputs/run",
            subject=None,
            deterministic=False,
            sensitivity=False,
            sensitivity_params=None,
            sensitivity_eps=0.01,
            population=0,
            academic_report=False,
            qsp_model=None,
            qsp_config=None,
            qsp_mode="posthoc",
        )


@simulate_app.command("single")
def simulate_single(
    compound: str = typer.Argument(..., help="Compound name or YAML path"),
    dose: float = typer.Option(100.0, help="Dose in mg"),
    route: str = typer.Option("oral", help="oral / iv / sc"),
    t_end: float = typer.Option(24.0, help="Simulation end time (h)"),
    json_output: bool = typer.Option(False, "--json", "-j", help="Output results as JSON"),
) -> None:
    """Single-dose PBPK simulation."""
    _run_single_sim(compound, dose, route, t_end, json_output)


@simulate_app.command("population")
def simulate_population(
    compound: str = typer.Argument(..., help="Path to compound YAML"),
    n_subjects: int = typer.Option(50, "--n-subjects", "-n", help="Number of virtual subjects"),
    dose: float = typer.Option(100.0, "--dose", "-d", help="Dose in mg"),
    route: str = typer.Option("oral", "--route", "-r", help="oral / iv"),
    duration: float = typer.Option(24.0, "--duration", help="Simulation duration (h)"),
    seed: int = typer.Option(42, "--seed", help="Random seed"),
) -> None:
    """Population PK simulation (virtual subjects)."""
    # TODO: delegate remaining population options
    population(
        compound=compound,
        n_subjects=n_subjects,
        dose=dose,
        route=route,
        duration=duration,
        seed=seed,
    )


@simulate_app.command("pgx")
def simulate_pgx(
    smiles: str = typer.Option(..., "--smiles", "-s", help="SMILES string of the drug."),
    dose: float = typer.Option(100.0, "--dose", "-d", help="Dose in mg."),
    gene: str = typer.Option("CYP2D6", "--gene", "-g", help="CYP gene."),
    route: str = typer.Option("oral", "--route", "-r", help="Route: oral or iv."),
    t_end_h: float = typer.Option(24.0, "--t-end", help="Simulation end time (h)."),
    body_weight: float = typer.Option(70.0, "--bw", help="Body weight (kg)."),
    out: str | None = typer.Option(None, "--out", "-o", help="Output HTML report path."),
) -> None:
    """PGx-stratified PBPK simulation."""
    pgx_sim(
        smiles=smiles,
        dose=dose,
        gene=gene,
        route=route,
        t_end_h=t_end_h,
        body_weight=body_weight,
        out=out,
    )


# --- train group -------------------------------------------------------------


@train_app.command("surrogate")
def train_surrogate_cmd(
    n_samples: int = typer.Option(500, help="Training samples"),
    output_dir: str = typer.Option("models/", help="Output directory"),
    epochs: int = typer.Option(300, help="Training epochs"),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
) -> None:
    """Train neural surrogate model."""
    import os

    save = os.path.join(output_dir, "surrogate_real.npz")
    surrogate_cmd(train=True, n_samples=n_samples, save=save, epochs=epochs)
    if json_output:
        _output({"status": "ok", "save_path": save, "n_samples": n_samples}, json_output=True)


@train_app.command("calibrate")
def train_calibrate_cmd(
    compound: str = typer.Option(..., help="Path to compound YAML file."),
    observed: str = typer.Option(..., help="Path to observed data CSV."),
    dose_mg: float = typer.Option(10.0, help="Dose (mg)."),
    route: str = typer.Option("oral", help="Route: 'oral' or 'iv'."),
    n_samples: int = typer.Option(2000, help="MCMC iterations."),
    out: str = typer.Option("outputs/calibration", help="Output directory."),
) -> None:
    """Bayesian MCMC parameter calibration."""
    # TODO: expose additional calibrate options (burn_in, seed, body_weight, t_end_h)
    calibrate(
        compound=compound,
        observed=observed,
        dose_mg=dose_mg,
        route=route,
        body_weight=70.0,
        t_end_h=24.0,
        n_samples=n_samples,
        burn_in=500,
        seed=None,
        out=out,
    )


# --- validate group ----------------------------------------------------------


@validate_app.command("benchmark")
def validate_benchmark_cmd(
    suite_dir: str = typer.Option("benchmarks", help="Path to benchmark suite directory."),
    out: str = typer.Option("outputs/benchmark", help="Output directory."),
    body_weight: float = typer.Option(70.0, help="Body weight (kg)."),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
) -> None:
    """Run multi-drug benchmark validation."""
    if json_output:
        from omega_pbpk.validation.benchmarks import run_benchmark_suite

        summary = run_benchmark_suite(suite_dir, out, body_weight=body_weight)
        _output(summary, json_output=True)
    else:
        benchmark(suite_dir=suite_dir, out=out, body_weight=body_weight)


@validate_app.command("mass-balance")
def validate_mass_balance_cmd(
    compound: str = typer.Option(..., help="Path to compound YAML file."),
    dose_mg: float = typer.Option(10.0, help="Dose (mg)."),
    route: str = typer.Option("oral", help="Route: 'oral' or 'iv'."),
    body_weight: float = typer.Option(70.0, help="Body weight (kg)."),
    t_end_h: float = typer.Option(24.0, help="Simulation end time (h)."),
) -> None:
    """Mass balance and physiological sanity checks."""
    validate_cmd(
        compound=compound,
        dose_mg=dose_mg,
        route=route,
        body_weight=body_weight,
        t_end_h=t_end_h,
    )


@validate_app.command("sensitivity")
def validate_sensitivity_cmd(
    compound: str = typer.Option(..., help="Path to compound YAML file."),
    dose_mg: float = typer.Option(10.0, help="Dose (mg)."),
    route: str = typer.Option("oral", help="Route: 'oral' or 'iv'."),
    body_weight: float = typer.Option(70.0, help="Body weight (kg)."),
    t_end_h: float = typer.Option(24.0, help="Simulation end time (h)."),
    out: str = typer.Option("outputs/sensitivity", help="Output directory."),
) -> None:
    """Local sensitivity analysis."""
    sensitivity_cmd(
        compound=compound,
        dose_mg=dose_mg,
        route=route,
        body_weight=body_weight,
        t_end_h=t_end_h,
        out=out,
    )


# --- serve group -------------------------------------------------------------


@serve_app.command("start")
def serve_start(
    host: str = typer.Option("0.0.0.0", help="Host to bind the server to."),
    port: int = typer.Option(8000, help="Port to listen on."),
    reload: bool = typer.Option(False, help="Enable auto-reload on code changes."),
) -> None:
    """Start FastAPI REST server."""
    try:
        import uvicorn
    except ImportError:
        typer.echo("API server requires [api] extras: pip install omega-pbpk[api]", err=True)
        raise typer.Exit(1) from None

    typer.echo(f"Starting Omega PBPK API server on http://{host}:{port}")
    uvicorn.run("omega_pbpk.api.app:app", host=host, port=port, reload=reload)


def main() -> None:
    """CLI entry point."""
    app()


if __name__ == "__main__":
    main()
