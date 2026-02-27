"""FDA-style regulatory HTML report generator."""

from __future__ import annotations

import base64
import io
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ReportInput:
    """All inputs needed to generate a regulatory report."""

    drug_name: str
    smiles: str = ""
    dose_mg: float = 100.0
    route: str = "oral"
    nca_result: Any = None  # NCAResult or None
    ddi_report: Any = None  # DDIRiskReport or None
    pop_pk_result: Any = None  # PopPKResult or None
    adme_properties: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


def _fig_to_base64(fig) -> str:
    """Convert matplotlib figure to base64 PNG string."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def _fmt(v: float, decimals: int = 3) -> str:
    if math.isnan(v) or math.isinf(v):
        return "—"
    return f"{v:.{decimals}f}"


def generate_report(inp: ReportInput, output_path: str) -> str:
    """Generate a standalone HTML regulatory report.

    Args:
        inp: ReportInput with all simulation results
        output_path: Path to write the .html file
    Returns:
        Path to the generated file
    """
    sections: list[str] = []

    # --- Drug summary ---
    _smiles_td = f'<td style="font-family:monospace;font-size:0.85em">{inp.smiles or "—"}</td>'
    sections.append(f"""
    <section>
      <h2>1. Drug Summary</h2>
      <table>
        <tr><th>Name</th><td>{inp.drug_name}</td></tr>
        <tr><th>SMILES</th>{_smiles_td}</tr>
        <tr><th>Dose</th><td>{inp.dose_mg} mg ({inp.route})</td></tr>
      </table>
    </section>""")

    # --- ADME properties ---
    if inp.adme_properties:
        rows = "".join(
            f"<tr><th>{k}</th><td>{_fmt(v) if isinstance(v, float) else v}</td></tr>"
            for k, v in inp.adme_properties.items()
            if k != "confidence"
        )
        conf = inp.adme_properties.get("confidence", "—")
        sections.append(f"""
    <section>
      <h2>2. Predicted ADME Properties</h2>
      <p>Prediction confidence: <strong>{conf}</strong></p>
      <table>{rows}</table>
    </section>""")

    # --- NCA ---
    if inp.nca_result is not None:
        r = inp.nca_result
        sections.append(f"""
    <section>
      <h2>3. Non-Compartmental Analysis (NCA)</h2>
      <table>
        <tr><th>Parameter</th><th>Value</th><th>Unit</th></tr>
        <tr><td>C<sub>max</sub></td><td>{_fmt(r.cmax_mg_L)}</td><td>mg/L</td></tr>
        <tr><td>T<sub>max</sub></td><td>{_fmt(r.tmax_h)}</td><td>h</td></tr>
        <tr><td>AUC<sub>0-t</sub></td><td>{_fmt(r.auc0t_mg_h_L)}</td><td>mg·h/L</td></tr>
        <tr><td>AUC<sub>0-∞</sub></td><td>{_fmt(r.auc0inf_mg_h_L)}</td><td>mg·h/L</td></tr>
        <tr><td>t<sub>½</sub></td><td>{_fmt(r.t_half_h)}</td><td>h</td></tr>
        <tr><td>CL</td><td>{_fmt(r.cl_L_per_h)}</td><td>L/h</td></tr>
        <tr><td>V<sub>z</sub></td><td>{_fmt(r.vz_L)}</td><td>L</td></tr>
        <tr><td>MRT</td><td>{_fmt(r.mrt_h)}</td><td>h</td></tr>
        <tr><td>R²</td><td>{_fmt(r.r_squared)}</td><td>—</td></tr>
      </table>
    </section>""")

    # --- DDI ---
    if inp.ddi_report is not None:
        d = inp.ddi_report
        if d.flags:
            flag_html = "".join(f"<li style='color:darkorange'>⚠ {f}</li>" for f in d.flags)
        else:
            flag_html = "<li style='color:green'>No risk flags</li>"
        _r3a4 = "⚠ Risk" if d.R1_3a4 >= 2 else "✓ OK"
        _r2d6 = "⚠ Risk" if d.R1_2d6 >= 2 else "✓ OK"
        _r2c9 = "⚠ Risk" if d.R1_2c9 >= 2 else "✓ OK"
        _r1a2 = "⚠ Risk" if d.R1_1a2 >= 2 else "✓ OK"
        _rgut = "⚠ Risk" if d.R1gut_3a4 >= 11 else "✓ OK"
        _rmbi = "⚠ Risk" if d.R2_3a4 >= 1.25 else "✓ OK"
        sections.append(f"""
    <section>
      <h2>4. DDI Risk Assessment (FDA 2020)</h2>
      <table>
        <tr><th>Enzyme</th><th>R1 (static)</th><th>Threshold</th><th>Status</th></tr>
        <tr><td>CYP3A4</td><td>{_fmt(d.R1_3a4)}</td><td>≥ 2.0</td><td>{_r3a4}</td></tr>
        <tr><td>CYP2D6</td><td>{_fmt(d.R1_2d6)}</td><td>≥ 2.0</td><td>{_r2d6}</td></tr>
        <tr><td>CYP2C9</td><td>{_fmt(d.R1_2c9)}</td><td>≥ 2.0</td><td>{_r2c9}</td></tr>
        <tr><td>CYP1A2</td><td>{_fmt(d.R1_1a2)}</td><td>≥ 2.0</td><td>{_r1a2}</td></tr>
        <tr><td>CYP3A4 (gut R1gut)</td><td>{_fmt(d.R1gut_3a4)}</td>
          <td>≥ 11.0</td><td>{_rgut}</td></tr>
        <tr><td>CYP3A4 (MBI R2)</td><td>{_fmt(d.R2_3a4)}</td><td>≥ 1.25</td><td>{_rmbi}</td></tr>
      </table>
      <p><strong>Flags:</strong></p>
      <ul>{flag_html}</ul>
      <p><strong>Recommendation:</strong> {d.recommendation}</p>
    </section>""")

    # --- Population PK ---
    if inp.pop_pk_result is not None:
        pr = inp.pop_pk_result
        cmax_s = pr.cmax_stats()
        auc_s = pr.auc_stats()

        # Try to embed VPC plot
        vpc_img = ""
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(8, 4))
            t = pr.time_h
            lo = pr.percentile_cp(5)
            med = pr.median_cp()
            hi = pr.percentile_cp(95)
            ax.fill_between(t, lo, hi, alpha=0.25, color="steelblue", label="5th–95th %ile")
            ax.plot(t, med, color="steelblue", linewidth=2, label="Median")
            ax.set_xlabel("Time (h)")
            ax.set_ylabel("Concentration (mg/L)")
            ax.set_title(f"Population PK — {pr.n_subjects} subjects")
            ax.legend()
            ax.set_xlim(left=0)
            ax.set_ylim(bottom=0)
            plt.tight_layout()
            img_b64 = _fig_to_base64(fig)
            plt.close(fig)
            vpc_img = f'<img src="data:image/png;base64,{img_b64}" style="max-width:700px"/>'
        except Exception:
            vpc_img = "<p><em>Plot unavailable (matplotlib not installed)</em></p>"

        sections.append(f"""
    <section>
      <h2>5. Population PK (n={pr.n_subjects})</h2>
      <table>
        <tr><th>Parameter</th><th>Median</th><th>5th %ile</th><th>95th %ile</th><th>CV%</th></tr>
        <tr><td>C<sub>max</sub> (mg/L)</td><td>{_fmt(cmax_s["median"])}</td>
          <td>{_fmt(cmax_s["p5"])}</td><td>{_fmt(cmax_s["p95"])}</td><td>{cmax_s["cv_pct"]:.1f}%</td></tr>
        <tr><td>AUC (mg·h/L)</td><td>{_fmt(auc_s["median"])}</td>
          <td>{_fmt(auc_s["p5"])}</td><td>{_fmt(auc_s["p95"])}</td><td>{auc_s["cv_pct"]:.1f}%</td></tr>
      </table>
      {vpc_img}
    </section>""")

    # --- Warnings ---
    if inp.warnings:
        w_html = "".join(f"<li>{w}</li>" for w in inp.warnings)
        sections.append(f"<section><h2>Warnings</h2><ul>{w_html}</ul></section>")

    css = """
    body{font-family:Arial,sans-serif;max-width:900px;margin:40px auto;color:#222;line-height:1.5}
    h1{color:#1a4fa0;border-bottom:2px solid #1a4fa0;padding-bottom:8px}
    h2{color:#2c6fad;margin-top:2em}
    table{border-collapse:collapse;width:100%;margin:1em 0}
    th,td{border:1px solid #ccc;padding:7px 12px;text-align:left}
    th{background:#f0f4fa;font-weight:600}
    tr:nth-child(even){background:#fafbff}
    section{margin-bottom:2em}
    """

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Omega PBPK — {inp.drug_name}</title>
  <style>{css}</style>
</head>
<body>
  <h1>Omega PBPK Regulatory Report</h1>
  <p><em>Generated by Omega PBPK v0.8.0</em></p>
  {"".join(sections)}
</body>
</html>"""

    Path(output_path).write_text(html, encoding="utf-8")
    return output_path


def quick_report(
    smiles: str,
    drug_name: str,
    dose_mg: float = 100.0,
    route: str = "oral",
    output_path: str = "omega_report.html",
    n_pop_subjects: int = 0,
) -> str:
    """Run full pipeline and generate HTML report in one call.

    Args:
        smiles: Drug SMILES string
        drug_name: Display name
        dose_mg: Dose in mg
        route: oral or iv
        output_path: Output HTML file path
        n_pop_subjects: If >0, run population PK with this many subjects
    Returns:
        Path to generated report
    """
    from omega_pbpk.clinical.nca import run_nca
    from omega_pbpk.pipeline import OmegaPipeline, SimulationRequest

    pipeline = OmegaPipeline()
    req = SimulationRequest(smiles=smiles, dose_mg=dose_mg, route=route)
    sim_result = pipeline.simulate(req)

    nca = run_nca(sim_result.time_h, sim_result.cp_mg_L, dose_mg)

    pop_result = None
    if n_pop_subjects > 0:
        try:
            from omega_pbpk.population.pop_simulator import PopulationSimulator

            drug = pipeline._build_drug(sim_result.adme_properties, [])
            pop_sim = PopulationSimulator(drug=drug)
            pop_result = pop_sim.run(n_subjects=n_pop_subjects, dose_mg=dose_mg, route=route)
        except Exception as e:
            sim_result.warnings.append(f"Population PK failed: {e}")

    inp = ReportInput(
        drug_name=drug_name,
        smiles=smiles,
        dose_mg=dose_mg,
        route=route,
        nca_result=nca,
        adme_properties=sim_result.adme_properties,
        warnings=sim_result.warnings,
        pop_pk_result=pop_result,
    )
    return generate_report(inp, output_path)
