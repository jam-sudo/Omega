from __future__ import annotations

from typing import Any


def build_markdown_report(summary: dict[str, Any]) -> str:
    lines = [
        "# Benchmark Validation Report",
        "",
        f"Overall pass: **{summary['overall_pass']}**",
        "",
        "Deterministic: "
        f"**{summary.get('deterministic', False)}**  | "
        f"Seed: **{summary.get('seed')}**",
        "",
        "## Thresholds",
        "",
        f"- AUC relative error <= {summary['thresholds']['auc_relative_error_max']}",
        f"- Cmax relative error <= {summary['thresholds']['cmax_relative_error_max']}",
        f"- Tmax absolute error <= {summary['thresholds']['tmax_abs_error_h_max']} h",
        "",
        "## Per-drug results",
        "",
        "| Drug | AUC rel err | Cmax rel err | Tmax abs err (h) | RMSE (mg/L) | Pass |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for item in summary["results"]:
        metrics = item["metrics"]
        lines.append(
            "| {drug} | {auc:.3f} | {cmax:.3f} | {tmax:.3f} | {rmse:.3f} | {passed} |".format(
                drug=item["drug"],
                auc=metrics["auc_relative_error"],
                cmax=metrics["cmax_relative_error"],
                tmax=metrics["tmax_abs_error_h"],
                rmse=metrics["rmse_mg_per_L"],
                passed="✅" if item["pass"] else "❌",
            )
        )
    lines.append("")
    lines.append("Artifacts are saved under `outputs/benchmarks/<drug>/`.")
    return "\n".join(lines)
