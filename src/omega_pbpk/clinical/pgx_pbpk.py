"""PGx-stratified PBPK simulation.

Runs PBPK simulations for each CYP phenotype (PM/IM/NM/UM) by scaling
the drug's hepatic CLint by the CPIC activity-based scaling factor.
Produces AUC ratio forest data and summary metrics.
"""

from __future__ import annotations

import base64
import dataclasses
import io
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Canonical phenotype ordering for display (worst → best metaboliser)
_PHENOTYPE_ORDER = ["PM", "IM", "NM", "UM"]

# Representative canonical CLint scaling factors per phenotype
# (used to pick one representative result per phenotype category)
_PHENOTYPE_CANONICAL_SCALE: dict[str, float] = {
    "PM": 0.0,
    "IM": 0.25,
    "NM": 1.0,
    "UM": 2.0,
}


@dataclass(frozen=True)
class PGxPBPKResult:
    """Result of a PGx-stratified PBPK simulation.

    Attributes:
        gene: CYP gene that was analysed (e.g. ``"CYP2D6"``).
        phenotype_results: List of per-phenotype dicts, each containing
            ``phenotype``, ``auc_mg_h_L``, ``cmax_mg_L``,
            ``auc_ratio_vs_nm``, and ``population_frequency``.
        nm_auc: Reference NM AUC (mg·h/L).
        nm_cmax: Reference NM Cmax (mg/L).
        gene_sensitivity_index: Maximum AUC ratio across all phenotypes.
            Higher values indicate greater PGx sensitivity.
    """

    gene: str
    phenotype_results: list[dict]
    nm_auc: float
    nm_cmax: float
    gene_sensitivity_index: float


def _get_phenotype_scaling_factors(gene: str) -> dict[str, float]:
    """Return one canonical CLint scaling factor per phenotype for *gene*.

    Uses PGxAnalyzer to enumerate all diplotypes, then picks the scaling
    factor that best represents each phenotype category.  For PM and NM
    the exact canonical scales (0.0 and 1.0) are used when available;
    for IM and UM the median scaling factor across diplotypes in that
    phenotype class is returned.
    """
    from omega_pbpk.pharmacogenomics.cyp_polymorphism import PGxAnalyzer

    analyzer = PGxAnalyzer()
    all_results = analyzer.analyze_gene(gene, population="Global")

    # Collect scaling factors per phenotype
    per_phenotype: dict[str, list[float]] = {"PM": [], "IM": [], "NM": [], "UM": []}
    per_phenotype_freq: dict[str, float] = {"PM": 0.0, "IM": 0.0, "NM": 0.0, "UM": 0.0}
    for r in all_results:
        if r.phenotype in per_phenotype:
            per_phenotype[r.phenotype].append(r.clint_scaling_factor)
            per_phenotype_freq[r.phenotype] += r.population_frequency

    # Normalise frequencies so they sum to 1
    total_freq = sum(per_phenotype_freq.values())
    if total_freq > 0:
        per_phenotype_freq = {k: v / total_freq for k, v in per_phenotype_freq.items()}

    # Choose a canonical scaling factor per phenotype
    canonical_scales: dict[str, float] = {}
    canonical_freq: dict[str, float] = {}
    for pheno in _PHENOTYPE_ORDER:
        scales = per_phenotype[pheno]
        if not scales:
            # Phenotype not observed for this gene — use the standard default
            canonical_scales[pheno] = _PHENOTYPE_CANONICAL_SCALE[pheno]
            canonical_freq[pheno] = 0.0
        else:
            # Use the median scaling factor as representative
            sorted_scales = sorted(scales)
            mid = len(sorted_scales) // 2
            if len(sorted_scales) % 2 == 0:
                median_scale = (sorted_scales[mid - 1] + sorted_scales[mid]) / 2.0
            else:
                median_scale = sorted_scales[mid]
            canonical_scales[pheno] = median_scale
            canonical_freq[pheno] = per_phenotype_freq.get(pheno, 0.0)

    return canonical_scales, canonical_freq


def run_pgx_pbpk(
    drug,
    gene: str,
    dose_mg: float = 100.0,
    route: str = "oral",
    t_end_h: float = 24.0,
    body_weight: float = 70.0,
) -> PGxPBPKResult:
    """Run PBPK simulations stratified by CYP phenotype.

    For each CPIC phenotype (PM, IM, NM, UM) the drug's hepatic intrinsic
    clearance (``clint_hepatic_L_per_h``) is scaled by a representative
    phenotype-specific factor derived from the ALLELE_DB, then a full
    whole-body PBPK simulation is executed.

    Args:
        drug: A :class:`~omega_pbpk.drugs.drug.Drug` instance.  The field
            ``clint_hepatic_L_per_h`` is the clearance that will be scaled.
        gene: CYP gene to stratify by (e.g. ``"CYP2D6"``).
        dose_mg: Administered dose (mg).
        route: Administration route — ``"oral"`` or ``"iv"``.
        t_end_h: Simulation end time (h).
        body_weight: Subject body weight (kg).

    Returns:
        :class:`PGxPBPKResult` containing per-phenotype PK metrics and
        the gene sensitivity index.
    """
    from omega_pbpk.core.body import WholeBodyPBPK
    from omega_pbpk.drugs.drug import Drug

    base_clint = drug.clint_hepatic_L_per_h

    canonical_scales, canonical_freq = _get_phenotype_scaling_factors(gene)

    phenotype_results: list[dict] = []
    nm_auc: float = 0.0
    nm_cmax: float = 0.0

    for pheno in _PHENOTYPE_ORDER:
        scale = canonical_scales[pheno]
        pop_freq = canonical_freq[pheno]

        # Scale CLint — for PM (scale=0) clint becomes 0; NM stays at base value
        scaled_clint = base_clint * scale

        # Drug is frozen — use dataclasses.replace for immutable copy with new CLint
        scaled_drug: Drug = dataclasses.replace(
            drug,
            clint_hepatic_L_per_h=scaled_clint,
        )

        try:
            model = WholeBodyPBPK(drug=scaled_drug, body_weight=body_weight)
            if route == "iv":
                model.setup_iv(dose_mg=dose_mg)
            else:
                model.setup_oral(dose_mg=dose_mg)
            sim_result = model.simulate(t_end_h=t_end_h)
            pk = sim_result.pk_summary()
            auc = float(pk["AUC_mg_h_L"])
            cmax = float(pk["Cmax_mg_L"])
        except Exception as exc:
            logger.warning("PBPK simulation failed for %s %s: %s", gene, pheno, exc)
            auc = 0.0
            cmax = 0.0

        if pheno == "NM":
            nm_auc = auc
            nm_cmax = cmax

        phenotype_results.append(
            {
                "phenotype": pheno,
                "auc_mg_h_L": round(auc, 4),
                "cmax_mg_L": round(cmax, 6),
                "auc_ratio_vs_nm": None,  # filled in below
                "population_frequency": round(pop_freq, 4),
            }
        )

    # Compute AUC ratios vs NM
    for rec in phenotype_results:
        if nm_auc > 0:
            rec["auc_ratio_vs_nm"] = round(rec["auc_mg_h_L"] / nm_auc, 4)
        else:
            rec["auc_ratio_vs_nm"] = 1.0

    # Gene sensitivity index: max AUC ratio across all phenotypes
    ratios = [rec["auc_ratio_vs_nm"] for rec in phenotype_results]
    gene_sensitivity_index = float(max(ratios)) if ratios else 1.0

    return PGxPBPKResult(
        gene=gene,
        phenotype_results=phenotype_results,
        nm_auc=round(nm_auc, 4),
        nm_cmax=round(nm_cmax, 6),
        gene_sensitivity_index=round(gene_sensitivity_index, 4),
    )


def plot_pgx_forest(
    result: PGxPBPKResult,
    out_path: str | None = None,
) -> str:
    """Generate a forest plot of AUC ratios vs NM by phenotype.

    Uses :func:`~omega_pbpk.visualization.vpc.plot_forest` under the hood.
    The X-axis shows AUC ratio vs NM on a log scale; phenotypes appear on
    the Y-axis in PM → IM → NM → UM order.

    Args:
        result: A :class:`PGxPBPKResult` from :func:`run_pgx_pbpk`.
        out_path: If given, save the PNG to this path.  Otherwise the plot
            is returned as a base-64-encoded PNG string.

    Returns:
        The output path (if *out_path* was supplied) or a base-64 PNG
        string prefixed with ``"data:image/png;base64,"`` (for embedding
        in HTML).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker

    # Build ordered display lists (PM → IM → NM → UM)
    rec_map = {rec["phenotype"]: rec for rec in result.phenotype_results}
    labels: list[str] = []
    ratios: list[float] = []
    freqs: list[float] = []

    for pheno in _PHENOTYPE_ORDER:
        if pheno in rec_map:
            rec = rec_map[pheno]
            pop_freq_pct = rec["population_frequency"] * 100
            labels.append(f"{pheno} ({pop_freq_pct:.0f}%)")
            ratios.append(rec["auc_ratio_vs_nm"] if rec["auc_ratio_vs_nm"] else 1.0)
            freqs.append(rec["population_frequency"])

    n = len(labels)
    y_pos = list(range(n))

    fig, ax = plt.subplots(figsize=(8, max(3, n * 0.8 + 1.5)))

    colors = {"PM": "#d62728", "IM": "#ff7f0e", "NM": "#2ca02c", "UM": "#1f77b4"}
    pheno_labels = [p for p in _PHENOTYPE_ORDER if p in rec_map]

    for i, (_label, ratio, pheno) in enumerate(zip(labels, ratios, pheno_labels, strict=False)):
        color = colors.get(pheno, "steelblue")
        ax.scatter([ratio], [i], color=color, s=120, zorder=5, label=pheno)
        # Draw a thin horizontal line through the point as a visual guide
        ax.plot([ratio * 0.85, ratio * 1.15], [i, i], color=color, linewidth=1.5, alpha=0.6)

    # Reference line at ratio = 1.0 (NM)
    ax.axvline(1.0, color="gray", linestyle="--", alpha=0.7, linewidth=1.2, label="NM reference")

    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.2g"))
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=11)
    ax.set_xlabel("AUC ratio vs NM (log scale)", fontsize=11)
    ax.set_title(
        f"PGx-stratified AUC ratios — {result.gene}\n"
        f"Gene Sensitivity Index: {result.gene_sensitivity_index:.2f}",
        fontsize=12,
    )
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()

    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return out_path
    else:
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        b64 = base64.b64encode(buf.read()).decode("ascii")
        return f"data:image/png;base64,{b64}"


def pgx_report_html(result: PGxPBPKResult, drug_name: str = "Drug") -> str:
    """Generate a standalone HTML report with an embedded forest plot.

    Args:
        result: A :class:`PGxPBPKResult` from :func:`run_pgx_pbpk`.
        drug_name: Display name of the drug compound.

    Returns:
        HTML string with embedded base-64 forest plot and a summary table.
    """
    img_data = plot_pgx_forest(result, out_path=None)

    # Build table rows
    rows_html = ""
    for rec in result.phenotype_results:
        pheno = rec["phenotype"]
        auc = rec["auc_mg_h_L"]
        cmax = rec["cmax_mg_L"]
        ratio = rec["auc_ratio_vs_nm"]
        freq_pct = rec["population_frequency"] * 100
        rows_html += (
            f"<tr>"
            f"<td><b>{pheno}</b></td>"
            f"<td>{auc:.4f}</td>"
            f"<td>{cmax:.6f}</td>"
            f"<td>{ratio:.3f}</td>"
            f"<td>{freq_pct:.1f}%</td>"
            f"</tr>\n"
        )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>PGx-PBPK Report — {drug_name} ({result.gene})</title>
  <style>
    body {{ font-family: Arial, sans-serif; max-width: 900px; margin: 40px auto; color: #333; }}
    h1 {{ color: #2c3e50; }}
    h2 {{ color: #2980b9; border-bottom: 1px solid #aed6f1; padding-bottom: 4px; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th {{ background: #2980b9; color: white; padding: 8px 12px; text-align: left; }}
    td {{ border: 1px solid #ddd; padding: 7px 12px; }}
    tr:nth-child(even) {{ background: #f2f2f2; }}
    .metric {{ display: inline-block; margin: 8px 20px 8px 0; }}
    .metric-value {{ font-size: 1.4em; font-weight: bold; color: #2980b9; }}
    img {{ max-width: 100%; border: 1px solid #ddd; border-radius: 4px; margin-top: 12px; }}
  </style>
</head>
<body>
  <h1>PGx-PBPK Report</h1>
  <p><b>Drug:</b> {drug_name} &nbsp;&nbsp; <b>Gene:</b> {result.gene}</p>

  <div>
    <span class="metric">
      NM AUC: <span class="metric-value">{result.nm_auc:.4f}</span> mg·h/L
    </span>
    <span class="metric">
      NM Cmax: <span class="metric-value">{result.nm_cmax:.4f}</span> mg/L
    </span>
    <span class="metric">
      Gene Sensitivity Index: <span class="metric-value">{result.gene_sensitivity_index:.2f}</span>
    </span>
  </div>

  <h2>Forest Plot — AUC Ratio vs NM</h2>
  <img src="{img_data}" alt="Forest plot of AUC ratios by CYP phenotype">

  <h2>Summary Table</h2>
  <table>
    <thead>
      <tr>
        <th>Phenotype</th>
        <th>AUC (mg·h/L)</th>
        <th>Cmax (mg/L)</th>
        <th>AUC Ratio vs NM</th>
        <th>Population Frequency</th>
      </tr>
    </thead>
    <tbody>
{rows_html}    </tbody>
  </table>
</body>
</html>
"""
    return html
