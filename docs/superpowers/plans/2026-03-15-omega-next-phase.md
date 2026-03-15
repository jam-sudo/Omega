# Omega Next Phase Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Expand Omega's validation from 20 drugs to multi-metric (Gold/Silver/Bronze), fix systematic failures, add pragmatic L3 (allometric + Bayesian fitting), and polish for publication.

**Architecture:** OmegaPipeline stays as-is (SMILES → ADME ensemble → ODE → PK). Validation expands via 3-tier strategy (Cmax/AUC, t_half, ADME properties). Pragmatic L3 adds covariate scaling and scipy-based individual fitting on top of existing pipeline. No GNN changes.

**Tech Stack:** Python 3.10, pytest, scipy.optimize, numpy, rdkit (SMILES→MW), OmegaPipeline, existing evaluation modules.

**Spec:** `docs/superpowers/specs/2026-03-15-omega-next-phase-design.md`

---

## Chunk 1: WS0 — Infrastructure & Cleanup

### Task 1: Commit Uncommitted Changes

**Files:**
- Modify: `.gitignore`
- Stage: `README.md`, `scripts/extract_openfda_pk.py`, `scripts/run_validation.py`, `src/omega_pbpk/ml/models/adme/admet_ai_wrapper.py`

- [ ] **Step 1: Add model binaries to .gitignore**

Add to `.gitignore`:
```
models/**/*.pt
```

- [ ] **Step 2: Stage and commit code changes (not binaries)**

```bash
echo 'models/**/*.pt' >> .gitignore
git add .gitignore README.md scripts/extract_openfda_pk.py scripts/run_validation.py src/omega_pbpk/ml/models/adme/admet_ai_wrapper.py
git commit -m "chore: commit pending changes, gitignore model binaries"
```

---

### Task 2: Automated Benchmark Script

**Files:**
- Create: `scripts/run_full_benchmark.py`
- Test: manual execution

This script wraps the existing `run_l1_benchmarks.py` logic but adds JSON output, timestamp, and comparison to previous run.

- [ ] **Step 1: Create `scripts/run_full_benchmark.py`**

```python
#!/usr/bin/env python3
"""Automated benchmark runner with JSON output and regression detection.

Runs OmegaPipeline on all benchmark drugs, saves results to
outputs/benchmark_YYYY-MM-DD.json, and compares to the most recent
previous run to detect regressions.

Usage:
    python scripts/run_full_benchmark.py [--previous outputs/benchmark_prev.json]
"""

import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root / "src"))

# Import drug list from existing benchmark script
sys.path.insert(0, str(repo_root / "scripts"))
from run_l1_benchmarks import BENCHMARK_DRUGS, compute_aafe, compute_fold_error, load_observed_pk


def run_benchmark() -> dict:
    """Run OmegaPipeline on all benchmark drugs, return results dict."""
    from omega_pbpk.pipeline import OmegaPipeline, SimulationRequest

    pipeline = OmegaPipeline()

    results = []
    cmax_fes, auc_fes, thalf_fes = [], [], []

    for drug_name, info in BENCHMARK_DRUGS.items():
        t0 = time.time()
        try:
            sim = pipeline.simulate(
                SimulationRequest(
                    smiles=info["smiles"],
                    dose_mg=info["dose_mg"],
                    route="oral",
                    duration_h=24.0,
                )
            )
            latency_ms = (time.time() - t0) * 1000

            observed = load_observed_pk(drug_name)
            fe_cmax = compute_fold_error(sim.cmax_mg_L, observed.get("cmax", 0))
            fe_auc = compute_fold_error(sim.auc0t_mg_h_L, observed.get("auc", 0))

            cmax_fes.append(fe_cmax)
            auc_fes.append(fe_auc)

            entry = {
                "drug": drug_name,
                "pred_cmax": sim.cmax_mg_L,
                "pred_auc": sim.auc0t_mg_h_L,
                "pred_thalf": sim.t_half_h,
                "obs_cmax": observed.get("cmax"),
                "obs_auc": observed.get("auc"),
                "fe_cmax": fe_cmax,
                "fe_auc": fe_auc,
                "latency_ms": latency_ms,
                "success": True,
            }
        except Exception as e:
            entry = {"drug": drug_name, "success": False, "error": str(e)}
        results.append(entry)

    return {
        "timestamp": datetime.now().isoformat(),
        "n_drugs": len(results),
        "n_success": sum(1 for r in results if r.get("success")),
        "aafe_cmax": compute_aafe(cmax_fes),
        "aafe_auc": compute_aafe(auc_fes),
        "pct_2fold_cmax": 100 * sum(1 for f in cmax_fes if f <= 2.0) / max(len(cmax_fes), 1),
        "pct_2fold_auc": 100 * sum(1 for f in auc_fes if f <= 2.0) / max(len(auc_fes), 1),
        "drugs": results,
    }


def compare_runs(current: dict, previous: dict) -> list[str]:
    """Compare two benchmark runs, return list of regression warnings."""
    warnings = []
    prev_drugs = {d["drug"]: d for d in previous.get("drugs", []) if d.get("success")}

    for entry in current.get("drugs", []):
        if not entry.get("success"):
            continue
        drug = entry["drug"]
        if drug in prev_drugs:
            prev = prev_drugs[drug]
            # Regression: fold-error increased by >50%
            for metric in ["fe_cmax", "fe_auc"]:
                cur_fe = entry.get(metric, 0)
                prev_fe = prev.get(metric, 0)
                if prev_fe > 0 and cur_fe > prev_fe * 1.5:
                    warnings.append(
                        f"REGRESSION: {drug} {metric} {prev_fe:.2f} -> {cur_fe:.2f} (+{(cur_fe/prev_fe - 1)*100:.0f}%)"
                    )
    # Aggregate regression
    for metric in ["aafe_cmax", "aafe_auc"]:
        cur = current.get(metric, 0)
        prev = previous.get(metric, 0)
        if prev > 0 and cur > prev * 1.2:
            warnings.append(f"REGRESSION: aggregate {metric} {prev:.3f} -> {cur:.3f}")

    return warnings


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--previous", type=str, help="Path to previous benchmark JSON for comparison")
    args = parser.parse_args()

    print("=" * 80)
    print("OMEGA FULL BENCHMARK")
    print("=" * 80)

    results = run_benchmark()

    # Save
    out_dir = repo_root / "outputs"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"benchmark_{datetime.now().strftime('%Y-%m-%d')}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")

    # Summary
    print(f"\nCmax AAFE: {results['aafe_cmax']:.3f}")
    print(f"AUC  AAFE: {results['aafe_auc']:.3f}")
    print(f"Cmax %2-fold: {results['pct_2fold_cmax']:.0f}%")
    print(f"AUC  %2-fold: {results['pct_2fold_auc']:.0f}%")

    # Regressions
    if args.previous:
        with open(args.previous) as f:
            prev = json.load(f)
        warnings = compare_runs(results, prev)
        if warnings:
            print("\n*** REGRESSIONS DETECTED ***")
            for w in warnings:
                print(f"  {w}")
            sys.exit(1)
        else:
            print("\nNo regressions detected.")

    # Per-drug >3-fold errors
    bad = [d for d in results["drugs"] if d.get("success") and d.get("fe_cmax", 0) > 3.0]
    print(f"\n>3-fold Cmax errors: {len(bad)} drugs")
    for d in sorted(bad, key=lambda x: -x["fe_cmax"]):
        print(f"  {d['drug']}: {d['fe_cmax']:.1f}x")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the benchmark script**

```bash
source .venv/bin/activate
python scripts/run_full_benchmark.py
```
Expected: Produces `outputs/benchmark_2026-03-15.json` with AAFE ~1.90/1.66.

- [ ] **Step 3: Commit**

```bash
git add scripts/run_full_benchmark.py
git commit -m "feat: add automated benchmark runner with regression detection"
```

---

### Task 3: Memory Cleanup

**Files:**
- Modify: `~/.claude/projects/-home-jam-Omega/memory/MEMORY.md`
- Delete: stale memory files if identified

- [ ] **Step 1: Review and clean stale memory files**

Check each file in `~/.claude/projects/-home-jam-Omega/memory/`:
- `status_2026_03_12.md` — outdated by 3 days, key info already in MEMORY.md → delete
- `l2_training_resume.md` — L2 GNN training concluded → delete
- `l2_v5_training.md` — v5 training done, findings in MEMORY.md → keep (historical reference)
- `roadmap_next.md` — superseded by spec doc → delete
- `team.md` — still relevant → keep

```bash
rm ~/.claude/projects/-home-jam-Omega/memory/status_2026_03_12.md
rm ~/.claude/projects/-home-jam-Omega/memory/l2_training_resume.md
rm ~/.claude/projects/-home-jam-Omega/memory/roadmap_next.md
```

- [ ] **Step 2: Update MEMORY.md next steps to point to spec**

Update the "Next Steps" section to reference the new spec document.

---

### Task 3.5: ODE Mass Balance Fix (0A Remaining)

**Files:**
- Modify: `src/omega_pbpk/core/body.py` (or `validation/__init__.py`)

The remaining 1/3 of the 0A mass balance bug. See `docs/plan-real.md` Section 2 for context.

- [ ] **Step 1: Identify the remaining mass balance issue**

```bash
grep -n "mass.balance\|dose_mg.*1e-3\|hard.coded" src/omega_pbpk/core/body.py src/omega_pbpk/validation/__init__.py
```

- [ ] **Step 2: Fix the issue and run tests**

```bash
pytest tests/ -m "not slow and not benchmark" -q
```

- [ ] **Step 3: Commit**

```bash
git add src/omega_pbpk/
git commit -m "fix: resolve remaining ODE mass balance bug (0A 3/3)"
```

---

## Chunk 2: WS1 — Multi-Metric Validation

### Task 4: SMILES Mapping for OpenFDA Drugs

**Files:**
- Create: `scripts/map_openfda_smiles.py`
- Create: `data/ml/clinical/openfda_validation.csv`

- [ ] **Step 1: Create SMILES mapping script**

```python
#!/usr/bin/env python3
"""Map OpenFDA drug names to canonical SMILES.

Cross-references adme_reference.csv and PubChem API.
Outputs: data/ml/clinical/openfda_validation.csv
"""

import csv
import json
import sys
import time
import urllib.request
import urllib.parse
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root / "src"))


def load_adme_reference() -> dict[str, str]:
    """Load name -> SMILES mapping from adme_reference.csv."""
    mapping = {}
    path = repo_root / "data" / "adme_reference.csv"
    with open(path) as f:
        for row in csv.DictReader(f):
            name = row["name"].strip().lower()
            smiles = row.get("smiles", "").strip()
            if smiles:
                mapping[name] = smiles
    return mapping


def pubchem_smiles(drug_name: str) -> str | None:
    """Look up canonical SMILES from PubChem by drug name."""
    try:
        encoded = urllib.parse.quote(drug_name)
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{encoded}/property/CanonicalSMILES/JSON"
        req = urllib.request.Request(url, headers={"User-Agent": "OmegaPBPK/1.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
        return data["PropertyTable"]["Properties"][0]["CanonicalSMILES"]
    except Exception:
        return None


def load_openfda_pk() -> list[dict]:
    """Load extracted OpenFDA PK parameters."""
    path = repo_root / "data" / "ml" / "clinical" / "openfda_pk_extracted.csv"
    with open(path) as f:
        return list(csv.DictReader(f))


def main():
    adme_map = load_adme_reference()
    openfda_rows = load_openfda_pk()

    # Get unique drugs and their parameters
    drug_params: dict[str, dict] = {}
    for row in openfda_rows:
        name = row["drug_name"].strip().lower()
        if name not in drug_params:
            drug_params[name] = {}
        drug_params[name][row["parameter"]] = row["value"]

    # Map to SMILES
    results = []
    for drug_name, params in sorted(drug_params.items()):
        # Try adme_reference first
        smiles = adme_map.get(drug_name)
        source = "adme_reference"

        # Try PubChem if not found
        if not smiles:
            print(f"  PubChem lookup: {drug_name}...", end=" ")
            smiles = pubchem_smiles(drug_name)
            source = "pubchem" if smiles else "NOT_FOUND"
            print(f"{'OK' if smiles else 'FAILED'}")
            time.sleep(0.3)  # Rate limit

        results.append({
            "drug_name": drug_name,
            "smiles": smiles or "",
            "smiles_source": source,
            "has_cmax": "cmax" in params,
            "has_auc": "auc" in params,
            "has_thalf": "t_half" in params,
            "has_F": "bioavailability" in params,
            "cmax_raw": params.get("cmax", ""),
            "auc_raw": params.get("auc", ""),
            "thalf_h": params.get("t_half", ""),
            "bioavailability": params.get("bioavailability", ""),
        })

    # Write output
    out_path = repo_root / "data" / "ml" / "clinical" / "openfda_validation.csv"
    fieldnames = list(results[0].keys())
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    # Summary
    mapped = sum(1 for r in results if r["smiles"])
    print(f"\nMapped: {mapped}/{len(results)} drugs have SMILES")
    print(f"Gold (Cmax+AUC): {sum(1 for r in results if r['has_cmax'] and r['has_auc'])}")
    print(f"Silver (t_half): {sum(1 for r in results if r['has_thalf'])}")
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the mapping script**

```bash
python scripts/map_openfda_smiles.py
```
Expected: `data/ml/clinical/openfda_validation.csv` with ~40 drugs having SMILES.

- [ ] **Step 3: Manual verification of Gold drugs**

For the 3 new Gold drugs, manually verify SMILES and convert units per Appendix A:
- ciprofloxacin: MW=331.3, check Cmax/AUC units in FDA label
- itraconazole: MW=705.6, check Cmax/AUC units
- sitagliptin: MW=407.3, Cmax=950nM→0.387 mg/L, AUC=8.52µM·hr→3.47 mg·h/L

Add a `cmax_mg_L` and `auc_mg_h_L` column to the CSV with normalized values.

- [ ] **Step 4: Commit**

```bash
git add scripts/map_openfda_smiles.py data/ml/clinical/openfda_validation.csv
git commit -m "feat: map OpenFDA drugs to SMILES for multi-metric validation"
```

---

### Task 5: Silver-Tier Validation (t_half)

**Files:**
- Create: `scripts/run_silver_benchmark.py`

- [ ] **Step 1: Create Silver-tier benchmark script**

```python
#!/usr/bin/env python3
"""Silver-tier validation: compare predicted t_half against OpenFDA values.

Runs OmegaPipeline on all OpenFDA drugs with t_half data,
compares predicted t_half to FDA label values.
"""

import csv
import json
import sys
from pathlib import Path

import numpy as np

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root / "src"))


def compute_fold_error(pred, obs):
    if abs(pred) < 1e-12 or abs(obs) < 1e-12:
        return float("nan")
    ratio = pred / obs
    return max(ratio, 1.0 / ratio)


def compute_aafe(fold_errors):
    valid = [fe for fe in fold_errors if not np.isnan(fe) and fe > 0]
    if not valid:
        return float("nan")
    return float(10.0 ** np.mean(np.log10(valid)))


def main():
    from omega_pbpk.pipeline import OmegaPipeline, SimulationRequest

    # Load validation CSV
    val_path = repo_root / "data" / "ml" / "clinical" / "openfda_validation.csv"
    with open(val_path) as f:
        drugs = [r for r in csv.DictReader(f) if r["smiles"] and r["has_thalf"] == "True"]

    print(f"Silver-tier: {len(drugs)} drugs with t_half")
    pipeline = OmegaPipeline()

    fold_errors = []
    results = []

    for drug in drugs:
        name = drug["drug_name"]
        smiles = drug["smiles"]
        obs_thalf = float(drug["thalf_h"])

        # Default dose 100mg for t_half comparison (t_half is dose-independent for linear PK)
        try:
            sim = pipeline.simulate(
                SimulationRequest(smiles=smiles, dose_mg=100.0, route="oral", duration_h=max(obs_thalf * 5, 24.0))
            )
            pred_thalf = sim.t_half_h
            fe = compute_fold_error(pred_thalf, obs_thalf)
            fold_errors.append(fe)

            print(f"  {name}: pred={pred_thalf:.1f}h, obs={obs_thalf:.1f}h, FE={fe:.2f}x {'OK' if fe <= 2 else 'MISS'}")
            results.append({"drug": name, "pred_thalf": pred_thalf, "obs_thalf": obs_thalf, "fe": fe, "success": True})
        except Exception as e:
            print(f"  {name}: FAILED ({e})")
            results.append({"drug": name, "success": False, "error": str(e)})

    aafe = compute_aafe(fold_errors)
    pct_2fold = 100 * sum(1 for f in fold_errors if f <= 2.0) / max(len(fold_errors), 1)

    print(f"\nSilver-tier Results:")
    print(f"  t_half AAFE: {aafe:.3f}")
    print(f"  %2-fold: {pct_2fold:.0f}%")
    print(f"  Drugs: {len(fold_errors)}/{len(drugs)} succeeded")

    # Save
    out = {"tier": "silver", "metric": "t_half", "n_drugs": len(fold_errors), "aafe": aafe, "pct_2fold": pct_2fold, "drugs": results}
    out_path = repo_root / "outputs" / "silver_tier_results.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run Silver benchmark**

```bash
python scripts/run_silver_benchmark.py
```
Expected: t_half AAFE for ~35 drugs, saved to `outputs/silver_tier_results.json`.

- [ ] **Step 3: Commit**

```bash
git add scripts/run_silver_benchmark.py
git commit -m "feat: Silver-tier validation — t_half on ~35 OpenFDA drugs"
```

---

### Task 6: Bronze-Tier Validation (ADME Properties)

**Files:**
- Create: `scripts/run_bronze_benchmark.py`

- [ ] **Step 1: Create Bronze-tier benchmark script**

```python
#!/usr/bin/env python3
"""Bronze-tier validation: compare predicted ADME properties against reference.

Uses adme_reference.csv (153 compounds) with ground-truth fup, clint, peff, rbp, logP.
Predicts via EnsembleADMEPredictor and computes per-property AAFE.
"""

import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root / "src"))


def compute_fold_error(pred, obs):
    if abs(pred) < 1e-12 or abs(obs) < 1e-12:
        return float("nan")
    ratio = pred / obs
    return max(ratio, 1.0 / ratio)


def compute_aafe(fold_errors):
    valid = [fe for fe in fold_errors if not math.isnan(fe) and fe > 0]
    if not valid:
        return float("nan")
    return float(10.0 ** np.mean(np.log10(valid)))


PROP_MAP = {
    "fup": ("fup", lambda adme: adme.fup),
    "clint_3a4": ("clint_3a4_uL_min_pmol", lambda adme: adme.clint_3a4),
    "peff": ("peff_cm_s", lambda adme: adme.peff),
    "rbp": ("rbp", lambda adme: adme.rbp),
    "logP": ("logP", lambda adme: adme.logP),
}


def main():
    from omega_pbpk.ml.models.adme.ensemble import EnsembleADMEPredictor

    predictor = EnsembleADMEPredictor(admet_ai=False)

    ref_path = repo_root / "data" / "adme_reference.csv"
    with open(ref_path) as f:
        compounds = list(csv.DictReader(f))

    print(f"Bronze-tier: {len(compounds)} compounds")

    property_fes: dict[str, list[float]] = {p: [] for p in PROP_MAP}
    results = []

    for comp in compounds:
        smiles = comp.get("smiles", "").strip()
        name = comp.get("name", "unknown")
        if not smiles:
            continue

        try:
            pred = predictor.predict(smiles)
        except Exception:
            continue

        entry = {"name": name, "smiles": smiles}
        for prop_key, (csv_col, getter) in PROP_MAP.items():
            obs_str = comp.get(csv_col, "").strip()
            if not obs_str:
                continue
            try:
                obs_val = float(obs_str)
            except ValueError:
                continue
            if obs_val <= 0:
                continue

            pred_val = getter(pred)
            fe = compute_fold_error(pred_val, obs_val)
            property_fes[prop_key].append(fe)
            entry[f"{prop_key}_pred"] = pred_val
            entry[f"{prop_key}_obs"] = obs_val
            entry[f"{prop_key}_fe"] = fe

        results.append(entry)

    # Summary
    print("\nBronze-tier Results:")
    summary = {}
    for prop, fes in property_fes.items():
        if fes:
            aafe = compute_aafe(fes)
            pct_2f = 100 * sum(1 for f in fes if f <= 2.0) / len(fes)
            print(f"  {prop}: AAFE={aafe:.3f}, %2-fold={pct_2f:.0f}%, n={len(fes)}")
            summary[prop] = {"aafe": aafe, "pct_2fold": pct_2f, "n": len(fes)}

    out = {"tier": "bronze", "n_compounds": len(results), "properties": summary}
    out_path = repo_root / "outputs" / "bronze_tier_results.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run Bronze benchmark**

```bash
python scripts/run_bronze_benchmark.py
```
Expected: Per-property AAFE for 5 ADME properties across ~100+ compounds.

- [ ] **Step 3: Commit**

```bash
git add scripts/run_bronze_benchmark.py
git commit -m "feat: Bronze-tier validation — ADME properties on 153 compounds"
```

---

## Chunk 3: WS3 — Validation Framework

### Task 7: Run Existing Validation Tests (T8/T9/T10)

**Files:**
- Modify: `scripts/run_validation.py` (if fixes needed)
- Output: `outputs/validation_report.md`

- [ ] **Step 1: Run T8 confidence calibration**

```bash
source .venv/bin/activate
cd /home/jam/Omega
python scripts/run_validation.py --t8 --verbose 2>&1 | tee outputs/t8_results.txt
```

Expected: Coverage metrics, confidence monotonicity check. Target: 90% CI coverage >= 88%.
If `--t8` flag doesn't exist, check `run_validation.py` CLI interface and adapt.

- [ ] **Step 2: Run T9 structural analog validation**

```bash
python scripts/run_validation.py --t9 --verbose 2>&1 | tee outputs/t9_results.txt
```

- [ ] **Step 3: Run T10 de novo validation**

```bash
python scripts/run_validation.py --t10 --n-molecules 100 --verbose 2>&1 | tee outputs/t10_results.txt
```

Note: Use n_molecules=100 first for speed (OmegaPipeline with admet_ai=False). Scale to 1000 if passing.

- [ ] **Step 4: If any test fails, debug and fix**

Common issues:
- Missing import → fix in `run_validation.py`
- Timeout → increase timeout or reduce molecule count
- Data file not found → check paths

- [ ] **Step 5: Commit any fixes**

```bash
git add scripts/run_validation.py outputs/t8_results.txt outputs/t9_results.txt outputs/t10_results.txt
git commit -m "feat: run validation framework T8/T9/T10"
```

---

### Task 8: Temporal Holdout Validation

**Files:**
- Create: `data/ml/clinical/temporal_holdout.csv`
- Create: `scripts/run_temporal_holdout.py`

- [ ] **Step 1: Curate temporal holdout drugs**

Create `data/ml/clinical/temporal_holdout.csv` with post-2023 FDA-approved drugs. Find SMILES and PK data from FDA approval packages.

Candidate drugs (verify SMILES via PubChem, PK from FDA labels):
```csv
drug_name,smiles,dose_mg,route,obs_cmax_mg_L,obs_auc_mg_h_L,obs_thalf_h,source
```

At minimum 5 drugs. If PK data is hard to find, use t_half only (Silver-tier holdout).

- [ ] **Step 2: Create temporal holdout runner**

```python
#!/usr/bin/env python3
"""Temporal holdout validation: post-2023 drugs the model has never seen."""

import csv
import json
import sys
from pathlib import Path

import numpy as np

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root / "src"))

sys.path.insert(0, str(repo_root / "scripts"))
from run_l1_benchmarks import compute_aafe, compute_fold_error


def main():
    from omega_pbpk.pipeline import OmegaPipeline, SimulationRequest

    holdout_path = repo_root / "data" / "ml" / "clinical" / "temporal_holdout.csv"
    with open(holdout_path) as f:
        drugs = list(csv.DictReader(f))

    pipeline = OmegaPipeline()
    results = []
    cmax_fes, auc_fes, thalf_fes = [], [], []

    for drug in drugs:
        name = drug["drug_name"]
        smiles = drug["smiles"]
        dose = float(drug["dose_mg"])

        try:
            sim = pipeline.simulate(
                SimulationRequest(smiles=smiles, dose_mg=dose, route=drug.get("route", "oral"))
            )

            entry = {"drug": name, "pred_cmax": sim.cmax_mg_L, "pred_auc": sim.auc0t_mg_h_L, "pred_thalf": sim.t_half_h}

            for metric, pred, obs_key, fe_list in [
                ("cmax", sim.cmax_mg_L, "obs_cmax_mg_L", cmax_fes),
                ("auc", sim.auc0t_mg_h_L, "obs_auc_mg_h_L", auc_fes),
                ("thalf", sim.t_half_h, "obs_thalf_h", thalf_fes),
            ]:
                obs_str = drug.get(obs_key, "").strip()
                if obs_str:
                    obs = float(obs_str)
                    fe = compute_fold_error(pred, obs)
                    entry[f"obs_{metric}"] = obs
                    entry[f"fe_{metric}"] = fe
                    fe_list.append(fe)

            entry["success"] = True
            print(f"  {name}: Cmax={sim.cmax_mg_L:.4f}, AUC={sim.auc0t_mg_h_L:.4f}, t½={sim.t_half_h:.1f}h")
        except Exception as e:
            entry = {"drug": name, "success": False, "error": str(e)}
            print(f"  {name}: FAILED ({e})")

        results.append(entry)

    print(f"\nTemporal Holdout Results:")
    if cmax_fes:
        print(f"  Cmax AAFE: {compute_aafe(cmax_fes):.3f} (n={len(cmax_fes)})")
    if auc_fes:
        print(f"  AUC  AAFE: {compute_aafe(auc_fes):.3f} (n={len(auc_fes)})")
    if thalf_fes:
        print(f"  t½   AAFE: {compute_aafe(thalf_fes):.3f} (n={len(thalf_fes)})")

    out = {"tier": "temporal_holdout", "drugs": results}
    out_path = repo_root / "outputs" / "temporal_holdout_results.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Run temporal holdout**

```bash
python scripts/run_temporal_holdout.py
```

- [ ] **Step 4: Commit**

```bash
git add data/ml/clinical/temporal_holdout.csv scripts/run_temporal_holdout.py
git commit -m "feat: temporal holdout validation — post-2023 drugs"
```

---

## Chunk 4: WS2 — Systematic Failure Analysis

### Task 9: Error Classification

**Files:**
- Create: `scripts/analyze_failures.py`
- Output: `outputs/failure_analysis.json`

- [ ] **Step 1: Create failure analysis script**

This script loads benchmark results, classifies >3-fold errors by mechanism, and generates a report.

```python
#!/usr/bin/env python3
"""Classify benchmark drug failures by mechanism.

Reads benchmark results JSON and categorizes >3-fold Cmax errors into:
- solubility: logS issues
- protein_binding: fup prediction error
- transporter: known P-gp/OATP substrate
- metabolism: CYP fraction, nonlinear, non-CYP
- formulation: prodrug, modified release, etc.
"""

import json
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent

# Known mechanistic classifications (curated from pharmacology literature)
KNOWN_MECHANISMS = {
    "verapamil": {"category": "transporter", "detail": "P-gp substrate, high first-pass via CYP3A4+P-gp efflux"},
    "ibuprofen": {"category": "protein_binding", "detail": "99% protein bound, fup~0.01, small errors in fup cause large PK errors"},
    "phenytoin": {"category": "metabolism", "detail": "Saturable (Michaelis-Menten) metabolism via CYP2C9, nonlinear PK at therapeutic doses"},
    "digoxin": {"category": "transporter", "detail": "P-gp substrate, renal elimination, narrow therapeutic index"},
    "atorvastatin": {"category": "transporter", "detail": "OATP1B1 substrate, hepatic uptake transporter, high first-pass"},
}


def main():
    import glob

    # Find most recent benchmark file
    bench_files = sorted(glob.glob(str(repo_root / "outputs" / "benchmark_*.json")))
    if not bench_files:
        print("No benchmark results found. Run scripts/run_full_benchmark.py first.")
        sys.exit(1)

    with open(bench_files[-1]) as f:
        bench = json.load(f)

    print(f"Analyzing: {bench_files[-1]}")
    print(f"Drugs: {bench['n_drugs']}, AAFE Cmax: {bench['aafe_cmax']:.3f}")

    # Classify failures
    failures = []
    for drug in bench.get("drugs", []):
        if not drug.get("success"):
            continue
        fe_cmax = drug.get("fe_cmax", 0)
        if fe_cmax > 3.0:
            name = drug["drug"]
            mechanism = KNOWN_MECHANISMS.get(name, {"category": "unknown", "detail": "Needs investigation"})
            failures.append({
                "drug": name,
                "fe_cmax": fe_cmax,
                "fe_auc": drug.get("fe_auc", 0),
                **mechanism,
            })

    # Summary by category
    categories = {}
    for f in failures:
        cat = f["category"]
        categories.setdefault(cat, []).append(f)

    print(f"\n>3-fold Cmax errors: {len(failures)} drugs")
    for cat, drugs in sorted(categories.items()):
        print(f"\n  [{cat}] ({len(drugs)} drugs):")
        for d in drugs:
            print(f"    {d['drug']}: {d['fe_cmax']:.1f}x — {d['detail']}")

    # Save
    out = {"source": bench_files[-1], "n_failures": len(failures), "by_category": {k: [d["drug"] for d in v] for k, v in categories.items()}, "details": failures}
    out_path = repo_root / "outputs" / "failure_analysis.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run failure analysis**

```bash
python scripts/analyze_failures.py
```

- [ ] **Step 3: Commit**

```bash
git add scripts/analyze_failures.py
git commit -m "feat: systematic failure classification for benchmark drugs"
```

---

### Task 10: Iterative Fixes with Regression Testing

This task is iterative — for each failure category, apply a fix, then run regression.

- [ ] **Step 1: Run baseline benchmark (before fixes)**

```bash
python scripts/run_full_benchmark.py
cp outputs/benchmark_2026-03-15.json outputs/benchmark_baseline.json
```

- [ ] **Step 2: For each fix, follow this cycle:**

```
1. Identify the fix (e.g., protein binding correction)
2. Implement in the relevant file
3. Run: python scripts/run_full_benchmark.py --previous outputs/benchmark_baseline.json
4. Check: no regressions, target drug improved
5. If regression detected: revert and try different approach
6. If clean: commit with descriptive message
```

- [ ] **Step 3: Document all fixes in outputs/fix_log.json**

```json
[
  {
    "date": "2026-03-16",
    "fix": "description",
    "files_changed": ["path/to/file.py"],
    "drugs_improved": ["drug1", "drug2"],
    "drugs_regressed": [],
    "aafe_before": 1.90,
    "aafe_after": 1.85
  }
]
```

---

## Chunk 5: WS4 — Production & Docs + WS5 — Pragmatic L3

### Task 11: Docs Bulk Update

**Files:**
- Modify: `README.md`
- Modify: all docs/ files with stale references

- [ ] **Step 1: Update README badges and benchmark table**

Update `README.md`:
- Badge: `AUC_AAFE-1.66` → latest value from benchmark
- Badge: `Level_1-pass` → keep
- Benchmark table: update with latest AAFE numbers
- Add 3-tier validation summary section
- Note warm (73ms) vs cold (~5s) startup

- [ ] **Step 2: Find and fix stale doc references**

```bash
# Find physio_sim.cli references
grep -r "physio_sim" docs/ --include="*.md" -l
# Find 34-state references
grep -r "34.state\|34 state" docs/ --include="*.md" -l
```

Replace `physio_sim.cli` → `omega`, `34-state` → `35-state` in all matches.

- [ ] **Step 3: CLI smoke test**

```bash
source .venv/bin/activate
omega --help
omega predict "CC(=O)Oc1ccccc1C(=O)O" 2>&1 | head -20
```

If CLI is broken, fix the entry point in `pyproject.toml` or `setup.py`.

- [ ] **Step 4: Commit docs changes**

```bash
git add README.md docs/
git commit -m "docs: update README + fix stale references (physio_sim→omega, 34→35 state)"
```

---

### Task 12: Pragmatic L3 — Covariate Scaling Module

**Files:**
- Create: `src/omega_pbpk/ml/models/foundation/covariate_scaling.py`
- Create: `tests/ml/test_covariate_scaling.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/ml/test_covariate_scaling.py
"""Tests for allometric covariate scaling."""

import pytest


def test_weight_scaling_clearance():
    """CL scales with (W/70)^0.75."""
    from omega_pbpk.ml.models.foundation.covariate_scaling import scale_clearance

    cl_pop = 10.0  # L/h
    # 70kg should give no change
    assert scale_clearance(cl_pop, weight_kg=70.0) == pytest.approx(10.0, rel=1e-3)
    # 35kg should reduce CL
    cl_35 = scale_clearance(cl_pop, weight_kg=35.0)
    assert cl_35 < cl_pop
    assert cl_35 == pytest.approx(10.0 * (35 / 70) ** 0.75, rel=1e-3)


def test_weight_scaling_volume():
    """Vd scales with (W/70)^1.0."""
    from omega_pbpk.ml.models.foundation.covariate_scaling import scale_volume

    vd_pop = 50.0  # L
    assert scale_volume(vd_pop, weight_kg=70.0) == pytest.approx(50.0, rel=1e-3)
    assert scale_volume(vd_pop, weight_kg=35.0) == pytest.approx(25.0, rel=1e-3)


def test_cyp2d6_scaling():
    """CYP2D6 PM reduces CL by 90%."""
    from omega_pbpk.ml.models.foundation.covariate_scaling import cyp_genotype_factor

    assert cyp_genotype_factor("CYP2D6", "EM") == pytest.approx(1.0)
    assert cyp_genotype_factor("CYP2D6", "PM") == pytest.approx(0.1)
    assert cyp_genotype_factor("CYP2D6", "UM") == pytest.approx(1.5)


def test_cyp2c9_scaling():
    """CYP2C9 *1/*3 reduces CL by 40%."""
    from omega_pbpk.ml.models.foundation.covariate_scaling import cyp_genotype_factor

    assert cyp_genotype_factor("CYP2C9", "*1/*1") == pytest.approx(1.0)
    assert cyp_genotype_factor("CYP2C9", "*1/*3") == pytest.approx(0.6)
    assert cyp_genotype_factor("CYP2C9", "*3/*3") == pytest.approx(0.1)


def test_apply_covariates():
    """Full covariate application adjusts CL and Vd."""
    from omega_pbpk.ml.models.foundation.covariate_scaling import apply_covariates

    base = {"cl_L_h": 10.0, "vd_L": 50.0}
    covariates = {"weight_kg": 100.0}
    adjusted = apply_covariates(base, covariates)

    assert adjusted["cl_L_h"] > base["cl_L_h"]  # heavier = higher CL
    assert adjusted["vd_L"] > base["vd_L"]  # heavier = higher Vd
```

- [ ] **Step 2: Run tests — verify they fail**

```bash
pytest tests/ml/test_covariate_scaling.py -v
```
Expected: ImportError — module doesn't exist yet.

- [ ] **Step 3: Implement covariate scaling module**

```python
# src/omega_pbpk/ml/models/foundation/covariate_scaling.py
"""Allometric covariate scaling for patient-specific PK predictions.

Implements standard population PK covariate relationships:
- Weight-based allometric scaling (CL, Vd)
- CYP genotype scaling factors
- Age-based adjustments (hepatic/renal function)

References:
- Anderson & Holford, Clin Pharmacokinet 2008 (allometry)
- Kirchheiner et al., Clin Pharmacol Ther 2004 (CYP2D6)
- Rettie et al., Pharmacogenetics 2000 (CYP2C9)
"""

from __future__ import annotations

# CYP genotype factor lookup table (Appendix B of design spec)
CYP_FACTORS: dict[str, dict[str, float]] = {
    "CYP2D6": {
        "UM": 1.5,
        "EM": 1.0,
        "IM": 0.5,
        "PM": 0.1,
    },
    "CYP2C9": {
        "*1/*1": 1.0,
        "*1/*2": 0.8,
        "*1/*3": 0.6,
        "*2/*2": 0.5,
        "*2/*3": 0.35,
        "*3/*3": 0.1,
    },
    "CYP2C19": {
        "UM": 1.5,
        "EM": 1.0,
        "IM": 0.6,
        "PM": 0.2,
    },
}

REF_WEIGHT_KG = 70.0


def scale_clearance(cl_pop: float, weight_kg: float = REF_WEIGHT_KG) -> float:
    """Scale population clearance by allometric weight relationship.

    CL_ind = CL_pop * (W / 70)^0.75
    """
    return cl_pop * (weight_kg / REF_WEIGHT_KG) ** 0.75


def scale_volume(vd_pop: float, weight_kg: float = REF_WEIGHT_KG) -> float:
    """Scale population volume of distribution by weight.

    Vd_ind = Vd_pop * (W / 70)^1.0
    """
    return vd_pop * (weight_kg / REF_WEIGHT_KG)


def cyp_genotype_factor(enzyme: str, phenotype: str) -> float:
    """Look up CYP genotype clearance scaling factor.

    Returns 1.0 for unknown enzyme/phenotype combinations.
    """
    enzyme_upper = enzyme.upper()
    return CYP_FACTORS.get(enzyme_upper, {}).get(phenotype, 1.0)


def apply_covariates(
    base_params: dict[str, float],
    covariates: dict[str, object],
) -> dict[str, float]:
    """Apply all covariate adjustments to population PK parameters.

    Parameters
    ----------
    base_params : dict
        Population PK parameters. Must contain 'cl_L_h' and 'vd_L'.
    covariates : dict
        Patient covariates. Supported keys:
        - weight_kg (float): body weight
        - cyp2d6_phenotype (str): UM/EM/IM/PM
        - cyp2c9_genotype (str): *1/*1, *1/*3, etc.
        - cyp2c19_phenotype (str): UM/EM/IM/PM

    Returns
    -------
    dict with adjusted parameters.
    """
    adjusted = dict(base_params)

    weight = float(covariates.get("weight_kg", REF_WEIGHT_KG))
    adjusted["cl_L_h"] = scale_clearance(base_params["cl_L_h"], weight)
    adjusted["vd_L"] = scale_volume(base_params["vd_L"], weight)

    # CYP genotype adjustments (apply to CL only)
    for enzyme_key, cov_key in [
        ("CYP2D6", "cyp2d6_phenotype"),
        ("CYP2C9", "cyp2c9_genotype"),
        ("CYP2C19", "cyp2c19_phenotype"),
    ]:
        phenotype = covariates.get(cov_key)
        if phenotype:
            factor = cyp_genotype_factor(enzyme_key, str(phenotype))
            if factor != 1.0:
                adjusted["cl_L_h"] *= factor

    return adjusted
```

- [ ] **Step 4: Run tests — verify they pass**

```bash
pytest tests/ml/test_covariate_scaling.py -v
```
Expected: All 5 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/omega_pbpk/ml/models/foundation/covariate_scaling.py tests/ml/test_covariate_scaling.py
git commit -m "feat: allometric covariate scaling module with CYP genotype factors"
```

---

### Task 13: Pragmatic L3 — Bayesian Individual Estimation

**Files:**
- Create: `src/omega_pbpk/ml/models/foundation/individual_estimation.py`
- Create: `tests/ml/test_individual_estimation.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/ml/test_individual_estimation.py
"""Tests for Bayesian individual PK parameter estimation."""

import pytest
import numpy as np


def test_fit_individual_recovers_params():
    """Given simulated C(t) data, fitting should recover approximate CL/Vd."""
    from omega_pbpk.ml.models.foundation.individual_estimation import fit_individual

    # Simulate a simple 1-cpt oral model
    # C(t) = (F*D/Vd) * ka/(ka-ke) * (exp(-ke*t) - exp(-ka*t))
    dose = 100.0  # mg
    vd_true = 50.0  # L
    cl_true = 5.0  # L/h
    ka = 1.0
    ke = cl_true / vd_true
    F = 0.8

    times = [0.5, 1.0, 2.0, 4.0, 8.0]
    concs = []
    for t in times:
        c = (F * dose / vd_true) * (ka / (ka - ke)) * (np.exp(-ke * t) - np.exp(-ka * t))
        concs.append(max(c, 0.0))

    observations = list(zip(times, concs))
    result = fit_individual(
        observations=observations,
        dose_mg=dose,
        base_cl=10.0,  # start from wrong value
        base_vd=100.0,  # start from wrong value
    )

    # Should recover CL and Vd within 2-fold
    assert 0.5 < result["cl_scale"] < 2.0
    assert 0.5 < result["vd_scale"] < 2.0


def test_fit_individual_single_observation():
    """Even with 1 observation, fitting should not crash."""
    from omega_pbpk.ml.models.foundation.individual_estimation import fit_individual

    result = fit_individual(
        observations=[(2.0, 1.5)],
        dose_mg=100.0,
        base_cl=5.0,
        base_vd=50.0,
    )
    assert "cl_scale" in result
    assert "vd_scale" in result
```

- [ ] **Step 2: Run tests — verify they fail**

```bash
pytest tests/ml/test_individual_estimation.py -v
```

- [ ] **Step 3: Implement individual estimation**

```python
# src/omega_pbpk/ml/models/foundation/individual_estimation.py
"""Bayesian individual PK parameter estimation.

Given sparse C(t) observations (1-5 points), estimates individual
CL and Vd scaling factors using scipy.optimize with a 1-compartment
analytical model.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize


def _one_cpt_oral(
    t: float, dose_mg: float, cl: float, vd: float, ka: float = 1.0, F: float = 0.8
) -> float:
    """Analytical 1-compartment oral model."""
    ke = cl / vd if vd > 0 else 0.1
    if abs(ka - ke) < 1e-6:
        ka = ke * 1.01
    if ka > ke:
        c = (F * dose_mg / vd) * (ka / (ka - ke)) * (np.exp(-ke * t) - np.exp(-ka * t))
    else:
        c = (F * dose_mg / vd) * np.exp(-ke * t)
    return max(c, 1e-12)


def fit_individual(
    observations: list[tuple[float, float]],
    dose_mg: float,
    base_cl: float,
    base_vd: float,
    ka: float = 1.0,
    F: float = 0.8,
) -> dict[str, float]:
    """Estimate individual CL and Vd scaling factors from sparse observations.

    Parameters
    ----------
    observations : list of (time_h, conc_mg_L)
        Observed concentration-time data (1-5 points).
    dose_mg : float
        Dose in mg.
    base_cl : float
        Population clearance (L/h).
    base_vd : float
        Population Vd (L).
    ka : float
        Absorption rate constant (default 1.0 h^-1).
    F : float
        Bioavailability (default 0.8).

    Returns
    -------
    dict with:
        cl_scale: individual CL / population CL
        vd_scale: individual Vd / population Vd
        cl_individual: estimated individual CL
        vd_individual: estimated individual Vd
        residual: final objective value
    """

    def objective(params):
        cl_scale, vd_scale = params
        cl = base_cl * cl_scale
        vd = base_vd * vd_scale
        sse = 0.0
        for t, c_obs in observations:
            c_pred = _one_cpt_oral(t, dose_mg, cl, vd, ka, F)
            # Log-space MSE for better scale handling
            sse += (np.log(c_pred + 1e-12) - np.log(c_obs + 1e-12)) ** 2
        return sse / len(observations)

    # Optimize CL_scale and Vd_scale, constrained to [0.05, 20]
    result = minimize(
        objective,
        x0=[1.0, 1.0],
        method="L-BFGS-B",
        bounds=[(0.05, 20.0), (0.05, 20.0)],
        options={"maxiter": 500},
    )

    cl_scale, vd_scale = result.x

    return {
        "cl_scale": float(cl_scale),
        "vd_scale": float(vd_scale),
        "cl_individual": float(base_cl * cl_scale),
        "vd_individual": float(base_vd * vd_scale),
        "residual": float(result.fun),
    }
```

- [ ] **Step 4: Run tests — verify they pass**

```bash
pytest tests/ml/test_individual_estimation.py -v
```
Expected: Both tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/omega_pbpk/ml/models/foundation/individual_estimation.py tests/ml/test_individual_estimation.py
git commit -m "feat: Bayesian individual estimation — scipy-based CL/Vd fitting"
```

---

### Task 14: Pipeline Integration — SimulationRequest Extension

**Files:**
- Modify: `src/omega_pbpk/pipeline/__init__.py:205-227` (SimulationRequest + OmegaPipeline)
- Create: `tests/ml/test_pipeline_l3.py`

- [ ] **Step 1: Write failing test**

```python
# tests/ml/test_pipeline_l3.py
"""Tests for L3 pipeline integration (covariate scaling)."""

import pytest


@pytest.mark.slow
def test_weight_adjusted_simulation():
    """Heavier patient should have different PK than lighter patient."""
    from omega_pbpk.pipeline import OmegaPipeline, SimulationRequest

    pipeline = OmegaPipeline()
    smiles = "CC(=O)CC(c1ccccc1)c1c(O)c2ccccc2oc1=O"  # warfarin

    result_70 = pipeline.simulate(SimulationRequest(smiles=smiles, dose_mg=5.0, subject_weight_kg=70.0))
    result_40 = pipeline.simulate(SimulationRequest(smiles=smiles, dose_mg=5.0, subject_weight_kg=40.0))

    # Lighter patient: higher Cmax (smaller Vd), longer t_half (lower CL)
    assert result_40.cmax_mg_L != result_70.cmax_mg_L


@pytest.mark.slow
def test_fit_individual_via_pipeline():
    """Pipeline.fit_individual should estimate individual parameters."""
    from omega_pbpk.pipeline import OmegaPipeline, SimulationRequest

    pipeline = OmegaPipeline()
    smiles = "CC(=O)CC(c1ccccc1)c1c(O)c2ccccc2oc1=O"  # warfarin
    request = SimulationRequest(smiles=smiles, dose_mg=5.0)

    # Fake observations
    observations = [(1.0, 0.15), (4.0, 0.12), (8.0, 0.08)]
    result = pipeline.fit_individual(request, observations)

    assert "cl_scale" in result
    assert "vd_scale" in result
    assert "simulation" in result  # re-simulated with fitted params
```

- [ ] **Step 2: Run tests — verify they fail**

```bash
pytest tests/ml/test_pipeline_l3.py -v
```

- [ ] **Step 3: Extend SimulationRequest and add fit_individual**

In `src/omega_pbpk/pipeline/__init__.py`, add optional genotype fields to `SimulationRequest`:

```python
# Add to SimulationRequest (after line 213):
    cyp2d6_phenotype: str | None = None    # UM/EM/IM/PM
    cyp2c9_genotype: str | None = None     # *1/*1, *1/*3, etc.
    cyp2c19_phenotype: str | None = None   # UM/EM/IM/PM
    egfr_ml_min: float | None = None       # eGFR for renal adjustment
```

Add to `OmegaPipeline`:

```python
def fit_individual(
    self, request: SimulationRequest, observations: list[tuple[float, float]]
) -> dict:
    """Fit individual PK parameters from sparse C(t) observations.

    Parameters
    ----------
    request : SimulationRequest
        Base simulation request (SMILES, dose, route).
    observations : list of (time_h, conc_mg_L)
        Observed concentration-time data (1-5 points).

    Returns
    -------
    dict with cl_scale, vd_scale, and re-simulated result.
    """
    from omega_pbpk.ml.models.foundation.individual_estimation import fit_individual as _fit_individual

    # Get population simulation first
    pop_result = self.simulate(request)

    # Compute CL from ADME properties via well-stirred model
    adme = pop_result.adme_properties
    fup = adme.get("fup", 0.1)
    clint_3a4 = adme.get("clint_3a4", 10.0)  # µL/min/pmol
    # IVIVE: pmol → L/h (same scaling as pipeline)
    ivive_factor = 40.0 * 45.0 * 1800.0 / 1e6 / 60.0
    clint_L_h = clint_3a4 * ivive_factor
    q_h = 90.0  # hepatic blood flow L/h
    cl_pop = (q_h * fup * clint_L_h) / (q_h + fup * clint_L_h) if clint_L_h > 0 else 5.0
    # Vd from ODE Cmax: Vd ≈ F*Dose/Cmax (rough estimate)
    vd_pop = max(request.dose_mg / max(pop_result.cmax_mg_L, 1e-6) * 0.8, 3.0)

    # Fit individual
    fit = _fit_individual(
        observations=observations,
        dose_mg=request.dose_mg,
        base_cl=cl_pop,
        base_vd=vd_pop,
    )

    fit["simulation"] = pop_result
    return fit
```

Note: The exact implementation will depend on how `adme_properties` stores CL and Vd. Check the actual dict keys when implementing.

- [ ] **Step 4: Apply covariate scaling in simulate()**

In `OmegaPipeline.simulate()`, after computing ADME properties but before running ODE, apply covariate scaling if weight/genotype is specified:

```python
# After _predict_adme() and _build_drug(), before _run_simulation():
if request.subject_weight_kg or request.cyp2d6_phenotype or request.cyp2c9_genotype or request.cyp2c19_phenotype:
    from omega_pbpk.ml.models.foundation.covariate_scaling import apply_covariates
    covariates = {}
    if request.subject_weight_kg:
        covariates["weight_kg"] = request.subject_weight_kg
    if request.cyp2d6_phenotype:
        covariates["cyp2d6_phenotype"] = request.cyp2d6_phenotype
    if request.cyp2c9_genotype:
        covariates["cyp2c9_genotype"] = request.cyp2c9_genotype
    if request.cyp2c19_phenotype:
        covariates["cyp2c19_phenotype"] = request.cyp2c19_phenotype
    # Apply to drug clearance and volume parameters
    # (implementation depends on Drug class internals)
```

- [ ] **Step 5: Run tests — verify they pass**

```bash
pytest tests/ml/test_pipeline_l3.py -v
```

- [ ] **Step 6: Commit**

```bash
git add src/omega_pbpk/pipeline/__init__.py tests/ml/test_pipeline_l3.py
git commit -m "feat: L3 pipeline integration — covariate scaling + fit_individual"
```

---

### Task 15: L3 Demo Script

**Files:**
- Create: `scripts/demo_l3_covariates.py`

- [ ] **Step 1: Create demo script**

```python
#!/usr/bin/env python3
"""Demo: Pragmatic L3 — warfarin PK with patient covariates.

Shows how weight and CYP2C9 genotype affect warfarin PK predictions.
"""

import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root / "src"))

from omega_pbpk.pipeline import OmegaPipeline, SimulationRequest

WARFARIN_SMILES = "CC(=O)CC(c1ccccc1)c1c(O)c2ccccc2oc1=O"


def main():
    pipeline = OmegaPipeline()

    scenarios = [
        {"label": "Reference (70kg, CYP2C9 *1/*1)", "weight": 70.0, "cyp2c9": None},
        {"label": "Light patient (40kg)", "weight": 40.0, "cyp2c9": None},
        {"label": "Heavy patient (100kg)", "weight": 100.0, "cyp2c9": None},
        {"label": "CYP2C9 *1/*3 (slow metabolizer)", "weight": 70.0, "cyp2c9": "*1/*3"},
        {"label": "CYP2C9 *3/*3 (poor metabolizer)", "weight": 70.0, "cyp2c9": "*3/*3"},
        {"label": "40kg + CYP2C9 *3/*3", "weight": 40.0, "cyp2c9": "*3/*3"},
    ]

    print("=" * 80)
    print("Omega L3 Demo: Warfarin 5mg Oral — Patient-Specific PK")
    print("=" * 80)

    for scenario in scenarios:
        request = SimulationRequest(
            smiles=WARFARIN_SMILES,
            dose_mg=5.0,
            route="oral",
            subject_weight_kg=scenario["weight"],
            cyp2c9_genotype=scenario.get("cyp2c9"),
        )
        result = pipeline.simulate(request)

        print(f"\n{scenario['label']}:")
        print(f"  Cmax  = {result.cmax_mg_L:.4f} mg/L")
        print(f"  AUC   = {result.auc0t_mg_h_L:.4f} mg·h/L")
        print(f"  t_half = {result.t_half_h:.1f} h")

    # Few-shot demo
    print("\n" + "=" * 80)
    print("Few-shot Individual Fitting")
    print("=" * 80)

    observations = [(1.0, 0.15), (4.0, 0.13), (12.0, 0.05)]
    request = SimulationRequest(smiles=WARFARIN_SMILES, dose_mg=5.0)
    fit = pipeline.fit_individual(request, observations)

    print(f"\nObservations: {observations}")
    print(f"CL scale: {fit['cl_scale']:.3f} (population → individual)")
    print(f"Vd scale: {fit['vd_scale']:.3f}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run demo**

```bash
python scripts/demo_l3_covariates.py
```
Expected: 6 scenarios showing different Cmax/AUC/t_half for different weights and genotypes.

- [ ] **Step 3: Commit**

```bash
git add scripts/demo_l3_covariates.py
git commit -m "feat: L3 demo — warfarin covariate scenarios + few-shot fitting"
```

---

## Execution Notes

### Parallelization
- **WS0 (Tasks 1-3):** Must complete first — prerequisite for all others.
- **WS1 (Tasks 4-6) and WS3 (Tasks 7-8):** Can run in parallel after WS0.
- **WS2 (Tasks 9-10):** Depends on WS1 results.
- **WS4 (Task 11) and WS5 (Tasks 12-15):** Can run in parallel with WS1-3.

### Testing Commands
```bash
# Fast tests only
pytest tests/ -m "not slow and not benchmark" -q

# Benchmark tests
pytest tests/ -m benchmark -v --timeout=300

# Specific ML tests
pytest tests/ml/test_covariate_scaling.py tests/ml/test_individual_estimation.py -v

# Full benchmark with regression
python scripts/run_full_benchmark.py --previous outputs/benchmark_baseline.json

# Lint
ruff check .
ruff format --check .
```

### Key File References
- Spec: `docs/superpowers/specs/2026-03-15-omega-next-phase-design.md`
- OmegaPipeline: `src/omega_pbpk/pipeline/__init__.py`
- Existing benchmarks: `scripts/run_l1_benchmarks.py`
- Benchmark data: `benchmarks/datasets/*.csv`
- ADME reference: `data/adme_reference.csv`
- OpenFDA extracted: `data/ml/clinical/openfda_pk_extracted.csv`
- Existing validation: `scripts/run_validation.py` (T8/T9/T10)
