#!/usr/bin/env python
"""Build the combined Cmax training set from benchmark JSON files.

Reads expanded_benchmark_2026-03-16.json (primary) and the latest
benchmark_*.json, deduplicates, and writes a CSV with columns:
  drug, smiles, dose_mg, obs_cmax, pred_cmax_pbpk, log_obs_cmax_per_mg, tier
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS = REPO_ROOT / "outputs"
OUT_CSV = REPO_ROOT / "data" / "ml" / "clinical" / "cmax_training_set.csv"


def _extract_valid(per_drug: list[dict]) -> list[dict]:
    """Return drugs with valid pred_cmax AND obs_cmax > 0 and no error key."""
    result = []
    for d in per_drug:
        if "error" in d:
            continue
        pred = d.get("pred_cmax")
        obs = d.get("obs_cmax")
        dose = d.get("dose_mg")
        if pred is None or obs is None or dose is None:
            continue
        if obs <= 0 or dose <= 0:
            continue
        result.append(d)
    return result


def main() -> None:
    # 1. Read expanded benchmark (primary source)
    expanded_path = OUTPUTS / "expanded_benchmark_2026-03-16.json"
    with open(expanded_path) as f:
        expanded = json.load(f)

    expanded_valid = _extract_valid(expanded["per_drug"])
    seen_drugs: set[str] = {d["drug"] for d in expanded["per_drug"]}

    print(
        f"Expanded benchmark: {len(expanded_valid)} valid drugs "
        f"(from {len(expanded['per_drug'])} total)"
    )

    # 2. Read latest benchmark_*.json for drugs NOT in expanded set
    benchmark_files = sorted(OUTPUTS.glob("benchmark_*.json"), key=lambda p: p.stat().st_mtime)
    latest_bench = benchmark_files[-1] if benchmark_files else None

    bench_new: list[dict] = []
    if latest_bench:
        with open(latest_bench) as f:
            bench = json.load(f)
        bench_valid = _extract_valid(bench["per_drug"])
        bench_new = [d for d in bench_valid if d["drug"] not in seen_drugs]
        print(f"Latest benchmark ({latest_bench.name}): {len(bench_new)} new valid drugs")

    # 3. Combine
    all_drugs = expanded_valid + bench_new
    print(f"Combined: {len(all_drugs)} drugs")

    # 4. Write CSV
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for d in all_drugs:
        dose = float(d["dose_mg"])
        obs = float(d["obs_cmax"])
        log_obs_per_mg = math.log(obs / dose)
        rows.append(
            {
                "drug": d["drug"],
                "smiles": d["smiles"],
                "dose_mg": dose,
                "obs_cmax": obs,
                "pred_cmax_pbpk": float(d["pred_cmax"]),
                "log_obs_cmax_per_mg": log_obs_per_mg,
                "tier": d.get("tier", "gold"),
            }
        )

    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "drug",
                "smiles",
                "dose_mg",
                "obs_cmax",
                "pred_cmax_pbpk",
                "log_obs_cmax_per_mg",
                "tier",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved {len(rows)} drugs to {OUT_CSV}")


if __name__ == "__main__":
    main()
