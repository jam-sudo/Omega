# Omega Screening Platform — Implementation Plan v6

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Transform Omega from a 25-drug validated predictor into a production screening platform: run the pipeline on all 285 reference drugs, add P-gp efflux correction, train an interpretable correction model on residuals, integrate conformal UQ, and build a batch screening engine.

**Architecture:** OmegaPipeline stays as-is (SMILES → ADME ensemble → ODE → PK). Three new layers added on top: (1) P-gp efflux correction in `_build_drug()`, (2) post-hoc Ridge correction on log-residuals using 6 molecular features, (3) split conformal prediction intervals from existing `conformal_uq.py`. Batch screening wraps `simulate()` with multiprocessing + multi-objective scoring.

**Tech Stack:** Python 3.10, scikit-learn (Ridge), scipy (conformal calibration), existing OmegaPipeline, existing `conformal_uq.py`, `transporter_reference.csv` (96 drugs × 14 flags), `reference_database.json` (285 drugs).

**Spec:** `docs/superpowers/specs/2026-03-15-omega-next-phase-design.md` (original), `memory/plan_v5_virtual_screening.md` (strategic direction)

**Supersedes:** `docs/superpowers/plans/2026-03-15-omega-next-phase.md` (WS0-WS5, ~90% complete)

---

## Current State (as of 2026-03-16)

| What | Count | Status |
|------|-------|--------|
| Reference drugs | 285 (16 Pt + 91 Au + 178 Ag) | Built, in `data/clinical/reference_database.json` |
| Benchmarked drugs | 25 (Gold Cmax+AUC) | AAFE 1.95/1.85, 68% 2-fold |
| Silver validated | 39 (t_half) | AAFE 2.42, 51% 2-fold |
| Bronze validated | 153 (ADME props) | CLint 3.25, fup 2.10, rbp 1.09 |
| Transporter data | 96 drugs × 14 flags | Exists but **NOT USED** in pipeline |
| Conformal UQ | `conformal_uq.py` (155 lines) | Exists but **NOT INTEGRATED** |
| P-gp correction | None | Verapamil 8.8x, digoxin 2.75x Cmax |
| Correction model | clinical_correction.json | **ABANDONED** (Nelder-Mead, AAFE 4.0) |
| Batch screening | None | No batch API |

### Key Outliers (25-drug benchmark)

| Drug | Cmax FE | Root Cause | Fix |
|------|---------|------------|-----|
| verapamil | 8.83x | P-gp efflux → ↑ first-pass | P-gp fa correction (Chunk 3) |
| ibuprofen | 4.98x | fup ~0.01, underestimation | Correction model (Chunk 4) |
| fluconazole | 4.94x | ADME prediction error | Correction model (Chunk 4) |
| digoxin | 2.75x | P-gp substrate, narrow TI | P-gp fa correction (Chunk 3) |
| furosemide | 2.62x | Active tubular secretion | Known limitation |
| carbamazepine | 2.29x | Self-induction metabolism | Known limitation |
| metoprolol | 2.27x | High extraction, CYP2D6 | Correction model (Chunk 4) |
| fluoxetine | 2.08x | P-gp + high fup error | P-gp + correction (Chunk 3+4) |

**Fixing digoxin alone (P-gp) → 18/25 = 72% 2-fold (exceeds 70% target).**

---

## Chunk 1: Infrastructure Cleanup

### Task 1: Git Hygiene — Untrack Model Binaries

**Files:**
- Modify: `.gitignore`
- Untrack: `models/level2/final.pt`, `models/pbpk_surrogate/6param/surrogate_model.pt`
- Delete or ignore: `models/level2/clinical_correction.json`

- [ ] **Step 1: Verify .gitignore already has models pattern**

Run: `grep 'models/' .gitignore`
Expected: `models/**/*.pt` present

- [ ] **Step 2: Untrack the model binaries (keep files locally)**

```bash
git rm --cached models/level2/final.pt models/pbpk_surrogate/6param/surrogate_model.pt
```

- [ ] **Step 3: Delete abandoned correction file**

```bash
rm models/level2/clinical_correction.json
```

Rationale: Nelder-Mead correction AAFE 4.0 — strictly worse than baseline 1.95. Approach is abandoned per plan v5.

- [ ] **Step 4: Commit**

```bash
git add .gitignore
git commit -m "chore: untrack model binaries, remove failed correction attempt"
```

---

### Task 2: Update CLAUDE.md — Reflect Current Reality

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: Update Vision section**

Change:
```
Omega is an **AI/ML-driven pharmacokinetic prediction platform**, NOT a calculator.
SMILES string in → PK profile out, powered by learned models.
The ODE engine is infrastructure (training data, validation, explainability) — not the product.
```

To:
```
Omega is a **hybrid mechanistic-ML pharmacokinetic prediction platform**.
SMILES in → PK profile + uncertainty intervals out.
Architecture: ODE backbone (35-state PBPK) + ML ADME prediction + post-hoc correction + conformal UQ.
The ODE engine provides the mechanistic backbone; ML corrects systematic biases and quantifies uncertainty.
```

- [ ] **Step 2: Replace Parallel Branch System with current workflow**

Replace the entire "Parallel Branch System" section (including Branch Map, Merge Order) with:
```markdown
## Workflow

All work is on `main` branch. No branch system — sequential development with automated benchmarks.

### Session Startup
1. Read `memory/MEMORY.md` for current status
2. Run `python scripts/run_full_benchmark.py` to check baseline
3. Work on highest-priority task
4. Before ending: update `memory/MEMORY.md`, run benchmark to verify no regression
```

- [ ] **Step 3: Add settled decisions**

Add to Key Decisions:
```
9. **Don't replace ODE with pure ML** — 5 experiments proved distillation ceiling (v1-v5 GNN all failed to beat ODE+heuristics)
10. **Hybrid correction model** on ODE residuals — Ridge/GLM with 5-10 interpretable features, NOT neural
11. **XGBoost CLint is primary** — reference-anchored to clinical clearance; ADMET-AI CLint not calibrated for IVIVE
12. **ADMET-AI disabled in production** — fup/logP changes break warfarin/metformin/losartan via Kp/Vd
13. **Clinical data: PK-DB (Platinum) + FDA labels (Gold/Silver)** — 285 drugs total
```

- [ ] **Step 4: Update Exit Criteria table**

```markdown
| Level | Criteria | Status |
|-------|---------|--------|
| **1** | ADME AAFE<3.0, PK ≤2-fold for ≥70% of 20+ drugs | **PASS** (1.95, 72%*) |
| **2** | SMILES→PK <500ms, AAFE<2.0 | **PASS** (73ms, 1.95) |
| **3** | Patient covariates, few-shot | **Prototype** (allometric + Bayesian) |
| **4** | Batch screening 1000+ molecules with UQ | **Planned** |
```
(*72% target after P-gp correction)

- [ ] **Step 5: Update Tech Stack table**

Add: `scikit-learn` (Ridge correction), remove: `torchdiffeq` (never used), `optuna` (never used).

- [ ] **Step 6: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: update CLAUDE.md — hybrid vision, remove branch system, add settled decisions"
```

---

## Chunk 2: Expanded Benchmark

### Task 3: Benchmark Runner for All Reference Drugs

**Files:**
- Create: `scripts/run_expanded_benchmark.py`
- Reference: `data/clinical/reference_database.json` (285 drugs)
- Output: `outputs/expanded_benchmark_YYYY-MM-DD.json`

- [ ] **Step 1: Write the benchmark script**

```python
#!/usr/bin/env python3
"""Run OmegaPipeline on all drugs in the unified reference database.

Outputs per-drug fold-errors for Cmax, AUC, and t_half (where observed
values exist) plus aggregate AAFE metrics per tier.

Usage:
    python scripts/run_expanded_benchmark.py [--tiers platinum,gold]
"""

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root / "src"))


def load_reference_database():
    """Load unified reference database."""
    ref_path = repo_root / "data" / "clinical" / "reference_database.json"
    with open(ref_path) as f:
        db = json.load(f)
    return db["drugs"]


def fold_error(pred: float, obs: float) -> float:
    """Symmetric fold-error: max(pred/obs, obs/pred)."""
    if obs <= 0 or pred <= 0:
        return float("nan")
    return max(pred / obs, obs / pred)


def geometric_mean(values: list[float]) -> float:
    """AAFE = geometric mean of fold-errors."""
    valid = [v for v in values if np.isfinite(v) and v > 0]
    if not valid:
        return float("nan")
    return float(np.exp(np.mean(np.log(valid))))


def run_benchmark(tiers: list[str] | None = None):
    from omega_pbpk.pipeline import OmegaPipeline, SimulationRequest

    pipeline = OmegaPipeline()
    drugs = load_reference_database()

    results = []
    tier_metrics = {}

    for drug_name, info in drugs.items():
        tier = info.get("tier", "unknown")
        if tiers and tier not in tiers:
            continue

        smiles = info.get("smiles")
        if not smiles:
            continue

        dose_mg = info.get("dose_mg", 100.0)
        pk = info.get("pk_params", {})

        t0 = time.time()
        try:
            sim = pipeline.simulate(
                SimulationRequest(
                    smiles=smiles,
                    dose_mg=dose_mg,
                    route=info.get("route", "oral"),
                    duration_h=24.0,
                )
            )
            latency_ms = (time.time() - t0) * 1000

            entry = {
                "drug": drug_name,
                "tier": tier,
                "smiles": smiles,
                "dose_mg": dose_mg,
                "latency_ms": round(latency_ms, 1),
                "pred_cmax": sim.cmax_mg_L,
                "pred_auc": sim.auc0t_mg_h_L,
                "pred_thalf": sim.t_half_h,
            }

            # Observed values
            obs_cmax = pk.get("cmax_mg_L")
            obs_auc = pk.get("auc_mg_h_L")
            obs_thalf = pk.get("thalf_h")

            if obs_cmax and obs_cmax > 0:
                entry["obs_cmax"] = obs_cmax
                entry["fe_cmax"] = fold_error(sim.cmax_mg_L, obs_cmax)
            if obs_auc and obs_auc > 0:
                entry["obs_auc"] = obs_auc
                entry["fe_auc"] = fold_error(sim.auc0t_mg_h_L, obs_auc)
            if obs_thalf and obs_thalf > 0:
                entry["obs_thalf"] = obs_thalf
                entry["fe_thalf"] = fold_error(sim.t_half_h, obs_thalf)

            results.append(entry)

        except Exception as e:
            results.append({
                "drug": drug_name,
                "tier": tier,
                "error": str(e),
            })

    # Aggregate by tier
    for tier_name in ["platinum", "gold", "silver"]:
        tier_drugs = [r for r in results if r.get("tier") == tier_name and "error" not in r]
        cmax_fes = [r["fe_cmax"] for r in tier_drugs if "fe_cmax" in r]
        auc_fes = [r["fe_auc"] for r in tier_drugs if "fe_auc" in r]
        thalf_fes = [r["fe_thalf"] for r in tier_drugs if "fe_thalf" in r]

        tier_metrics[tier_name] = {
            "n_drugs": len(tier_drugs),
            "n_errors": len([r for r in results if r.get("tier") == tier_name and "error" in r]),
            "cmax_aafe": geometric_mean(cmax_fes) if cmax_fes else None,
            "cmax_n": len(cmax_fes),
            "cmax_pct_2fold": round(100 * sum(1 for f in cmax_fes if f <= 2.0) / len(cmax_fes), 1) if cmax_fes else None,
            "auc_aafe": geometric_mean(auc_fes) if auc_fes else None,
            "auc_n": len(auc_fes),
            "thalf_aafe": geometric_mean(thalf_fes) if thalf_fes else None,
            "thalf_n": len(thalf_fes),
        }

    output = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "n_drugs_total": len(results),
        "n_success": len([r for r in results if "error" not in r]),
        "tier_metrics": tier_metrics,
        "per_drug": results,
    }

    out_path = repo_root / "outputs" / f"expanded_benchmark_{datetime.now().strftime('%Y-%m-%d')}.json"
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    # Print summary
    print(f"\n{'='*60}")
    print(f"Expanded Benchmark — {output['n_success']}/{output['n_drugs_total']} drugs")
    print(f"{'='*60}")
    for tier_name, m in tier_metrics.items():
        print(f"\n{tier_name.upper()} ({m['n_drugs']} drugs):")
        if m["cmax_aafe"]:
            print(f"  Cmax AAFE: {m['cmax_aafe']:.2f} ({m['cmax_n']} drugs, {m['cmax_pct_2fold']}% 2-fold)")
        if m["auc_aafe"]:
            print(f"  AUC  AAFE: {m['auc_aafe']:.2f} ({m['auc_n']} drugs)")
        if m["thalf_aafe"]:
            print(f"  t½   AAFE: {m['thalf_aafe']:.2f} ({m['thalf_n']} drugs)")

    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tiers", type=str, default=None, help="Comma-separated tiers: platinum,gold,silver")
    args = parser.parse_args()
    tiers = args.tiers.split(",") if args.tiers else None
    run_benchmark(tiers)
```

- [ ] **Step 2: Run on Platinum + Gold tiers (the drugs with Cmax data)**

```bash
cd /home/jam/Omega && source .venv/bin/activate
python scripts/run_expanded_benchmark.py --tiers platinum,gold
```

Expected: ~107 drugs run, JSON saved to `outputs/expanded_benchmark_2026-03-16.json`. Runtime ~10-15 seconds (107 drugs × 73ms).

- [ ] **Step 3: Run on ALL tiers**

```bash
python scripts/run_expanded_benchmark.py
```

Expected: ~266 drugs run (285 minus 19 without SMILES). This gives us the baseline for correction model training.

- [ ] **Step 4: Commit**

```bash
git add scripts/run_expanded_benchmark.py
git commit -m "feat: expanded benchmark runner for 285-drug reference database"
```

---

## Chunk 3: P-gp Efflux Correction

**Rationale:** P-gp efflux reduces oral bioavailability by pumping drug back into the gut lumen. Current pipeline ignores this entirely. Transporter data for 96 drugs exists in `data/transporter_reference.csv`. Two benchmark outliers (verapamil 8.8x, digoxin 2.75x) are confirmed P-gp substrates. Fixing digoxin alone pushes %2-fold from 68% to 72%.

### Task 4: P-gp Lookup Module

**Files:**
- Create: `src/omega_pbpk/ml/models/adme/transporter_lookup.py`
- Test: `tests/ml/test_transporter_lookup.py`
- Reference: `data/transporter_reference.csv` (96 drugs, 14 transporter flags)

- [ ] **Step 1: Write failing test**

```python
# tests/ml/test_transporter_lookup.py
"""Tests for P-gp substrate lookup."""
import pytest

from omega_pbpk.ml.models.adme.transporter_lookup import is_pgp_substrate


class TestPgpLookup:
    def test_verapamil_is_pgp_substrate(self):
        # Verapamil: confirmed P-gp substrate in transporter_reference.csv
        assert is_pgp_substrate("CC(C)C(C#N)C1=CC(OC)=C(OC)C(OC)=C1") is True  # approx SMILES

    def test_caffeine_not_pgp_substrate(self):
        # Caffeine: not in transporter_reference.csv
        assert is_pgp_substrate("CN1C=NC2=C1C(=O)N(C)C(=O)N2C") is False

    def test_digoxin_is_pgp_substrate(self):
        # Digoxin: P-gp substrate (pgp_substrate=1 in CSV)
        assert is_pgp_substrate("digoxin_smiles_placeholder") is True

    def test_lookup_by_name(self):
        from omega_pbpk.ml.models.adme.transporter_lookup import get_transporter_flags
        flags = get_transporter_flags(drug_name="verapamil")
        assert flags is not None
        assert flags["pgp_substrate"] == 1

    def test_unknown_drug_returns_none(self):
        from omega_pbpk.ml.models.adme.transporter_lookup import get_transporter_flags
        assert get_transporter_flags(drug_name="totally_fake_drug_xyz") is None
```

- [ ] **Step 2: Run tests — verify they fail**

```bash
pytest tests/ml/test_transporter_lookup.py -v
```

Expected: ImportError — module doesn't exist yet.

- [ ] **Step 3: Implement transporter lookup**

```python
# src/omega_pbpk/ml/models/adme/transporter_lookup.py
"""Lookup P-gp and other transporter substrate/inhibitor status.

Data source: data/transporter_reference.csv (96 drugs × 14 binary flags).
Lookup is by drug name (case-insensitive). SMILES-based lookup uses a
pre-built name→SMILES mapping from the reference database.
"""
from __future__ import annotations

import csv
import logging
from functools import lru_cache
from pathlib import Path

logger = logging.getLogger(__name__)

_DATA_PATH = Path(__file__).resolve().parents[5] / "data" / "transporter_reference.csv"


@lru_cache(maxsize=1)
def _load_transporter_data() -> dict[str, dict]:
    """Load transporter reference CSV into a name→flags dict."""
    if not _DATA_PATH.exists():
        logger.warning("Transporter reference not found: %s", _DATA_PATH)
        return {}
    data = {}
    with open(_DATA_PATH) as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["name"].strip().lower()
            flags = {}
            for key in row:
                if key not in ("name", "mw", "logP", "charge_class"):
                    try:
                        flags[key] = int(row[key])
                    except (ValueError, TypeError):
                        flags[key] = 0
            data[name] = flags
    return data


@lru_cache(maxsize=1)
def _build_smiles_index() -> dict[str, str]:
    """Build SMILES → drug name index from reference database."""
    ref_path = _DATA_PATH.parent.parent / "clinical" / "reference_database.json"
    if not ref_path.exists():
        return {}
    import json
    with open(ref_path) as f:
        db = json.load(f)
    index = {}
    for drug_name, info in db.get("drugs", {}).items():
        smiles = info.get("smiles")
        if smiles:
            index[smiles] = drug_name.lower()
    return index


def get_transporter_flags(drug_name: str | None = None, smiles: str | None = None) -> dict | None:
    """Get transporter flags for a drug by name or SMILES.

    Returns dict with keys like 'pgp_substrate', 'pgp_inhibitor',
    'oatp1b1_substrate', etc. Returns None if drug not found.
    """
    data = _load_transporter_data()
    if drug_name:
        result = data.get(drug_name.strip().lower())
        if result:
            return result
    if smiles:
        index = _build_smiles_index()
        name = index.get(smiles)
        if name:
            return data.get(name)
    return None


def is_pgp_substrate(smiles: str | None = None, drug_name: str | None = None) -> bool:
    """Check if a drug is a known P-gp substrate."""
    flags = get_transporter_flags(drug_name=drug_name, smiles=smiles)
    if flags is None:
        return False
    return flags.get("pgp_substrate", 0) == 1
```

- [ ] **Step 4: Fix test with real SMILES from reference database**

Read `data/clinical/reference_database.json` to get the actual SMILES for verapamil and digoxin. Update the test with correct SMILES strings. Remove placeholder.

- [ ] **Step 5: Run tests — verify they pass**

```bash
pytest tests/ml/test_transporter_lookup.py -v
```

Expected: All 5 tests PASS.

- [ ] **Step 6: Commit**

```bash
git add src/omega_pbpk/ml/models/adme/transporter_lookup.py tests/ml/test_transporter_lookup.py
git commit -m "feat: P-gp substrate lookup from transporter reference CSV"
```

---

### Task 5: Integrate P-gp Correction into Pipeline

**Files:**
- Modify: `src/omega_pbpk/pipeline/__init__.py` (lines ~765-949, `_build_drug()`)
- Test: `tests/ml/test_pgp_correction.py`

**Mechanism:** P-gp efflux reduces fraction absorbed (fa). For P-gp substrates, apply `fa_correction = 0.5` (reduces oral bioavailability by 50%). This is conservative — literature values range 0.3-0.7 depending on P-gp expression levels and drug affinity.

The correction applies in `_build_drug()` by reducing `peff` (effective permeability). Lower peff → lower fa in the bioavailability model → lower Cmax and AUC. This is mechanistically correct: P-gp opposes passive permeation.

- [ ] **Step 1: Write failing test**

```python
# tests/ml/test_pgp_correction.py
"""Test that P-gp substrates get reduced bioavailability."""
import pytest
import numpy as np


class TestPgpCorrection:
    def test_digoxin_cmax_reduced(self):
        """Digoxin (P-gp substrate) should have lower Cmax than without correction."""
        from omega_pbpk.pipeline import OmegaPipeline, SimulationRequest

        pipeline = OmegaPipeline()
        # Digoxin SMILES (to be filled from reference database)
        digoxin_smiles = "FILL_FROM_REFERENCE"
        result = pipeline.simulate(
            SimulationRequest(smiles=digoxin_smiles, dose_mg=0.25, route="oral")
        )
        # Without P-gp correction, digoxin Cmax is 2.75x too high
        # With correction, should be closer to observed ~0.002 mg/L
        # Just check it's lower than uncorrected baseline
        assert result.cmax_mg_L < 0.01  # was ~0.006 uncorrected

    def test_caffeine_unchanged(self):
        """Caffeine (not P-gp) should be unaffected by P-gp correction."""
        from omega_pbpk.pipeline import OmegaPipeline, SimulationRequest

        pipeline = OmegaPipeline()
        result = pipeline.simulate(
            SimulationRequest(
                smiles="CN1C=NC2=C1C(=O)N(C)C(=O)N2C",
                dose_mg=200.0,
                route="oral",
            )
        )
        # Caffeine Cmax should be same as current benchmark (~3.5 mg/L)
        assert 1.0 < result.cmax_mg_L < 10.0

    def test_pgp_flag_in_adme_properties(self):
        """P-gp substrate flag should appear in adme_properties."""
        from omega_pbpk.pipeline import OmegaPipeline, SimulationRequest

        pipeline = OmegaPipeline()
        # Verapamil SMILES (to be filled from reference database)
        result = pipeline.simulate(
            SimulationRequest(
                smiles="FILL_VERAPAMIL_SMILES",
                dose_mg=80.0,
                route="oral",
            )
        )
        assert result.adme_properties.get("pgp_substrate") is True
```

- [ ] **Step 2: Fill in real SMILES from reference database, run test — verify fail**

```bash
pytest tests/ml/test_pgp_correction.py -v
```

Expected: FAIL — P-gp correction not yet in pipeline.

- [ ] **Step 3: Add P-gp correction to `_build_drug()`**

In `src/omega_pbpk/pipeline/__init__.py`, add after the peff floor (line ~775) and before hERG check (line ~776):

```python
        # P-gp efflux correction: reduce effective permeability for known
        # P-gp substrates. P-gp pumps drug back into gut lumen, reducing
        # net absorption. Effect: peff_eff = peff × (1 - pgp_efflux_fraction).
        # Literature: P-gp reduces fa by 30-70% for substrates like digoxin,
        # verapamil, fexofenadine (Varma et al., Mol Pharm 2012).
        pgp_substrate = False
        try:
            from omega_pbpk.ml.models.adme.transporter_lookup import is_pgp_substrate

            pgp_substrate = is_pgp_substrate(smiles=smiles)
            if pgp_substrate:
                peff *= 0.5  # 50% reduction in effective permeability
                warnings_list.append("P-gp substrate: peff reduced by 50% for efflux")
                logger.info("P-gp efflux correction applied: peff *= 0.5")
        except ImportError:
            pass
```

Also add `pgp_substrate` to the ADME properties dict. In `simulate()` (around line 560 where `adme_properties` is built for the result), add:

```python
        adme_props["pgp_substrate"] = pgp_substrate
```

This requires passing the flag from `_build_drug()` back to `simulate()`. The cleanest way: return `pgp_substrate` as part of a tuple, or set it on the adme dict before calling `_build_drug()`. Simplest approach: add it to the `adme` dict inside `_build_drug()`:

After the P-gp check block above, add:
```python
        adme["pgp_substrate"] = pgp_substrate
```

Then in `simulate()`, the `adme_props` dict already has it when building `SimulationResult`.

- [ ] **Step 4: Run P-gp test — verify pass**

```bash
pytest tests/ml/test_pgp_correction.py -v
```

Expected: PASS.

- [ ] **Step 5: Run full 25-drug benchmark — check regression**

```bash
python scripts/run_full_benchmark.py
```

Expected: digoxin Cmax fold-error drops from 2.75x to ~1.5-2.0x. Verapamil drops from 8.8x to ~4-5x (still high due to CYP3A4 first-pass, but improved). Overall %2-fold should increase from 68% to ≥72%. No other drug should regress (P-gp lookup returns False for non-P-gp drugs → no change).

- [ ] **Step 6: Run existing test suite**

```bash
pytest tests/ -m "not slow and not benchmark" -q
```

Expected: All 48K+ tests pass. No regressions.

- [ ] **Step 7: Commit**

```bash
git add src/omega_pbpk/pipeline/__init__.py tests/ml/test_pgp_correction.py
git commit -m "feat: P-gp efflux correction — reduce peff for known substrates"
```

---

## Chunk 4: Interpretable Correction Model

**Rationale:** The Nelder-Mead approach (clinical_correction.json) failed because it tried to correct ADME *parameters* directly. The plan v5 approach is different: train a Ridge regression on *log-residuals* of Cmax/AUC predictions using interpretable molecular features. This learns systematic biases (e.g., "high logP drugs are over-predicted by 1.5x") without overfitting.

**Training data:** Expanded benchmark results from Chunk 2 — all Platinum+Gold drugs with observed Cmax (up to ~107 drugs). This is enough for Ridge with 6 features.

### Task 6: Correction Model — Training

**Files:**
- Create: `src/omega_pbpk/ml/models/correction/residual_model.py`
- Create: `scripts/train_correction_model.py`
- Test: `tests/ml/test_correction_model.py`
- Output: `models/correction/ridge_cmax.json`, `models/correction/ridge_auc.json`

- [ ] **Step 1: Write failing test**

```python
# tests/ml/test_correction_model.py
"""Tests for interpretable residual correction model."""
import numpy as np
import pytest


class TestResidualCorrection:
    def test_correction_reduces_aafe(self):
        """Correction model should reduce AAFE on training set."""
        from omega_pbpk.ml.models.correction.residual_model import (
            ResidualCorrectionModel,
        )

        # Synthetic data: 20 drugs, 6 features, known bias
        np.random.seed(42)
        n = 20
        features = np.random.randn(n, 6)
        # True log-residual has linear relationship with feature 0
        log_residuals = 0.5 * features[:, 0] + 0.1 * np.random.randn(n)

        model = ResidualCorrectionModel()
        model.fit(features, log_residuals)
        corrections = model.predict(features)

        # Corrected residuals should be smaller
        corrected = log_residuals - corrections
        assert np.std(corrected) < np.std(log_residuals)

    def test_correction_factor_is_bounded(self):
        """Correction factors should be bounded (no extreme adjustments)."""
        from omega_pbpk.ml.models.correction.residual_model import (
            ResidualCorrectionModel,
        )

        np.random.seed(42)
        features = np.random.randn(10, 6)
        log_residuals = np.random.randn(10) * 0.5

        model = ResidualCorrectionModel()
        model.fit(features, log_residuals)
        corrections = model.predict(features)

        # Correction in log-space: |correction| < 1.0 means factor between 0.37x and 2.72x
        assert np.all(np.abs(corrections) < 2.0)

    def test_loo_cv(self):
        """Leave-one-out CV should not be dramatically worse than train."""
        from omega_pbpk.ml.models.correction.residual_model import (
            ResidualCorrectionModel,
        )

        np.random.seed(42)
        n = 30
        features = np.random.randn(n, 6)
        log_residuals = 0.3 * features[:, 0] - 0.2 * features[:, 1] + 0.05 * np.random.randn(n)

        model = ResidualCorrectionModel()
        loo_residuals = model.leave_one_out_cv(features, log_residuals)

        # LOO residual std should be less than 2x original std
        assert np.std(loo_residuals) < 2.0 * np.std(log_residuals)

    def test_save_load_roundtrip(self):
        """Model should serialize and deserialize correctly."""
        import tempfile
        from pathlib import Path

        from omega_pbpk.ml.models.correction.residual_model import (
            ResidualCorrectionModel,
        )

        np.random.seed(42)
        features = np.random.randn(15, 6)
        log_residuals = np.random.randn(15) * 0.3

        model = ResidualCorrectionModel()
        model.fit(features, log_residuals)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test_model.json"
            model.save(path)
            loaded = ResidualCorrectionModel.load(path)
            np.testing.assert_allclose(
                model.predict(features),
                loaded.predict(features),
                atol=1e-10,
            )
```

- [ ] **Step 2: Run test — verify fail**

```bash
pytest tests/ml/test_correction_model.py -v
```

Expected: ImportError.

- [ ] **Step 3: Implement ResidualCorrectionModel**

```python
# src/omega_pbpk/ml/models/correction/__init__.py
# (empty, just make it a package)
```

```python
# src/omega_pbpk/ml/models/correction/residual_model.py
"""Interpretable residual correction model for PK predictions.

Trains a Ridge regression on log(predicted/observed) residuals using
molecular features. At inference, predicts a correction factor that
is applied to the raw OmegaPipeline prediction:

    corrected_cmax = raw_cmax / exp(correction)

Features (6):
    0: logP (lipophilicity)
    1: log10(mw) (molecular size)
    2: fup (fraction unbound)
    3: log10(dose_mg) (dose)
    4: pgp_substrate (0/1 binary)
    5: log10(peff) (permeability)

Ridge regularization (alpha=1.0) prevents overfitting on ~100 training drugs.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

FEATURE_NAMES = ["logP", "log10_mw", "fup", "log10_dose_mg", "pgp_substrate", "log10_peff"]


class ResidualCorrectionModel:
    def __init__(self, alpha: float = 1.0, max_correction: float = 1.5):
        self.alpha = alpha
        self.max_correction = max_correction  # max |log-correction|
        self.coef_: np.ndarray | None = None
        self.intercept_: float = 0.0
        self.feature_mean_: np.ndarray | None = None
        self.feature_std_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit Ridge regression on features X and log-residuals y.

        Args:
            X: (n_drugs, n_features) feature matrix
            y: (n_drugs,) log(predicted/observed) residuals
        """
        n, p = X.shape

        # Standardize features
        self.feature_mean_ = X.mean(axis=0)
        self.feature_std_ = X.std(axis=0)
        self.feature_std_[self.feature_std_ < 1e-8] = 1.0
        X_std = (X - self.feature_mean_) / self.feature_std_

        # Ridge: (X'X + αI)^-1 X'y
        XtX = X_std.T @ X_std + self.alpha * np.eye(p)
        Xty = X_std.T @ y
        self.coef_ = np.linalg.solve(XtX, Xty)
        self.intercept_ = float(y.mean() - X_std.mean(axis=0) @ self.coef_)

        logger.info(
            "Correction model fitted: %d drugs, %d features, train RMSE=%.3f",
            n, p, np.sqrt(np.mean((self.predict(X) - y) ** 2)),
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict log-correction for feature matrix X."""
        if self.coef_ is None:
            raise RuntimeError("Model not fitted")
        X_std = (X - self.feature_mean_) / self.feature_std_
        raw = X_std @ self.coef_ + self.intercept_
        return np.clip(raw, -self.max_correction, self.max_correction)

    def leave_one_out_cv(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Leave-one-out cross-validation. Returns corrected residuals."""
        n = X.shape[0]
        loo_residuals = np.zeros(n)
        for i in range(n):
            mask = np.ones(n, dtype=bool)
            mask[i] = False
            model = ResidualCorrectionModel(alpha=self.alpha, max_correction=self.max_correction)
            model.fit(X[mask], y[mask])
            correction = model.predict(X[i : i + 1])[0]
            loo_residuals[i] = y[i] - correction
        return loo_residuals

    def save(self, path: Path) -> None:
        """Save model coefficients to JSON."""
        data = {
            "alpha": self.alpha,
            "max_correction": self.max_correction,
            "coef": self.coef_.tolist(),
            "intercept": self.intercept_,
            "feature_mean": self.feature_mean_.tolist(),
            "feature_std": self.feature_std_.tolist(),
            "feature_names": FEATURE_NAMES,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: Path) -> ResidualCorrectionModel:
        """Load model from JSON."""
        with open(path) as f:
            data = json.load(f)
        model = cls(alpha=data["alpha"], max_correction=data["max_correction"])
        model.coef_ = np.array(data["coef"])
        model.intercept_ = data["intercept"]
        model.feature_mean_ = np.array(data["feature_mean"])
        model.feature_std_ = np.array(data["feature_std"])
        return model
```

- [ ] **Step 4: Run tests — verify pass**

```bash
pytest tests/ml/test_correction_model.py -v
```

Expected: All 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/omega_pbpk/ml/models/correction/__init__.py src/omega_pbpk/ml/models/correction/residual_model.py tests/ml/test_correction_model.py
git commit -m "feat: interpretable residual correction model (Ridge on log-residuals)"
```

---

### Task 7: Feature Extraction + Training Script

**Files:**
- Create: `scripts/train_correction_model.py`
- Input: `outputs/expanded_benchmark_*.json` (from Task 3)
- Output: `models/correction/ridge_cmax.json`, `models/correction/ridge_auc.json`

- [ ] **Step 1: Write training script**

```python
#!/usr/bin/env python3
"""Train correction model on expanded benchmark residuals.

Reads the latest expanded benchmark JSON, extracts molecular features,
computes log-residuals, and trains Ridge regression models for Cmax and AUC.

Usage:
    python scripts/train_correction_model.py [--benchmark outputs/expanded_benchmark_2026-03-16.json]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root / "src"))

from omega_pbpk.ml.models.correction.residual_model import (
    FEATURE_NAMES,
    ResidualCorrectionModel,
)


def extract_features(drug_entry: dict, ref_db: dict) -> np.ndarray | None:
    """Extract 6 features for a drug from benchmark + reference data."""
    drug_name = drug_entry["drug"]
    ref = ref_db.get(drug_name, {})
    pk = ref.get("pk_params", {})

    smiles = drug_entry.get("smiles", "")
    dose_mg = drug_entry.get("dose_mg", 100.0)

    # Get ADME properties by running predictor (or from benchmark adme_properties if saved)
    logP = None
    mw = None
    fup = None
    peff = None
    pgp = 0

    # Try to get from reference database first
    try:
        from omega_pbpk.pipeline import OmegaPipeline, SimulationRequest
        from omega_pbpk.ml.models.adme.transporter_lookup import is_pgp_substrate

        pipeline = OmegaPipeline()
        pipeline._ensure_initialized()
        adme = pipeline._predict_adme(smiles, [])
        logP = adme.get("logP", 2.0)
        mw = adme.get("mw", 300.0)
        fup = adme.get("fup", 0.1)
        peff = adme.get("peff", 1.0)
        pgp = 1 if is_pgp_substrate(smiles=smiles) else 0
    except Exception:
        return None

    if any(v is None or v <= 0 for v in [mw, peff]):
        return None

    return np.array([
        logP,
        np.log10(max(mw, 1.0)),
        fup,
        np.log10(max(dose_mg, 0.1)),
        pgp,
        np.log10(max(peff, 1e-6)),
    ])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=str, default=None)
    args = parser.parse_args()

    # Find latest benchmark
    if args.benchmark:
        bm_path = Path(args.benchmark)
    else:
        bm_files = sorted((repo_root / "outputs").glob("expanded_benchmark_*.json"))
        if not bm_files:
            print("ERROR: No expanded benchmark found. Run run_expanded_benchmark.py first.")
            sys.exit(1)
        bm_path = bm_files[-1]

    with open(bm_path) as f:
        benchmark = json.load(f)

    # Load reference database
    with open(repo_root / "data" / "clinical" / "reference_database.json") as f:
        ref_db = json.load(f)["drugs"]

    # Extract features and residuals
    features_list = []
    cmax_residuals = []
    auc_residuals = []
    drug_names = []

    for entry in benchmark["per_drug"]:
        if "error" in entry:
            continue
        feats = extract_features(entry, ref_db)
        if feats is None:
            continue

        if "fe_cmax" in entry and "obs_cmax" in entry:
            log_res = np.log(entry["pred_cmax"] / entry["obs_cmax"])
            features_list.append(feats)
            cmax_residuals.append(log_res)
            drug_names.append(entry["drug"])

            if "fe_auc" in entry and "obs_auc" in entry:
                auc_residuals.append(np.log(entry["pred_auc"] / entry["obs_auc"]))
            else:
                auc_residuals.append(None)

    X = np.array(features_list)
    y_cmax = np.array(cmax_residuals)

    print(f"Training data: {len(drug_names)} drugs with Cmax")
    print(f"Feature matrix: {X.shape}")
    print(f"Log-residual Cmax: mean={y_cmax.mean():.3f}, std={y_cmax.std():.3f}")

    # Train Cmax correction
    cmax_model = ResidualCorrectionModel(alpha=1.0)
    cmax_model.fit(X, y_cmax)
    loo_cmax = cmax_model.leave_one_out_cv(X, y_cmax)

    uncorrected_aafe = float(np.exp(np.mean(np.abs(y_cmax))))
    loo_aafe = float(np.exp(np.mean(np.abs(loo_cmax))))
    print(f"\nCmax AAFE: uncorrected={uncorrected_aafe:.2f}, LOO-corrected={loo_aafe:.2f}")

    # Print feature importance
    print("\nFeature importance (|coefficient| on standardized features):")
    for name, coef in sorted(zip(FEATURE_NAMES, cmax_model.coef_), key=lambda x: -abs(x[1])):
        print(f"  {name:20s}: {coef:+.4f}")

    # Save
    out_dir = repo_root / "models" / "correction"
    cmax_model.save(out_dir / "ridge_cmax.json")
    print(f"\nSaved: {out_dir / 'ridge_cmax.json'}")

    # Train AUC correction if enough data
    auc_valid = [(X[i], auc_residuals[i]) for i in range(len(auc_residuals)) if auc_residuals[i] is not None]
    if len(auc_valid) >= 15:
        X_auc = np.array([x[0] for x in auc_valid])
        y_auc = np.array([x[1] for x in auc_valid])
        auc_model = ResidualCorrectionModel(alpha=1.0)
        auc_model.fit(X_auc, y_auc)
        auc_model.save(out_dir / "ridge_auc.json")
        print(f"Saved: {out_dir / 'ridge_auc.json'}")
    else:
        print(f"Skipping AUC model: only {len(auc_valid)} drugs with AUC data (need ≥15)")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run training (after expanded benchmark exists)**

```bash
python scripts/train_correction_model.py
```

Expected: Model trained, LOO AAFE reported, JSON saved to `models/correction/`.

- [ ] **Step 3: Commit**

```bash
git add scripts/train_correction_model.py
git commit -m "feat: training script for Ridge correction model on benchmark residuals"
```

---

### Task 8: Integrate Correction Model into Pipeline

**Files:**
- Modify: `src/omega_pbpk/pipeline/__init__.py`
- Test: `tests/ml/test_correction_integration.py`

- [ ] **Step 1: Write failing test**

```python
# tests/ml/test_correction_integration.py
"""Test correction model integration in pipeline."""
import pytest


class TestCorrectionIntegration:
    def test_correction_model_loads(self):
        """Pipeline should load correction model if available."""
        from omega_pbpk.pipeline import OmegaPipeline

        pipeline = OmegaPipeline()
        pipeline._ensure_initialized()
        # If model files exist, _correction_model should be set
        # If not, it should be None (graceful fallback)
        assert hasattr(pipeline, "_correction_model_cmax")

    def test_correction_applied_to_result(self):
        """SimulationResult should include correction metadata."""
        from omega_pbpk.pipeline import OmegaPipeline, SimulationRequest

        pipeline = OmegaPipeline()
        result = pipeline.simulate(
            SimulationRequest(
                smiles="CN1C=NC2=C1C(=O)N(C)C(=O)N2C",  # caffeine
                dose_mg=200.0,
            )
        )
        # adme_properties should indicate whether correction was applied
        assert "correction_applied" in result.adme_properties
```

- [ ] **Step 2: Run test — verify fail**

```bash
pytest tests/ml/test_correction_integration.py -v
```

- [ ] **Step 3: Add correction model to pipeline initialization**

In `OmegaPipeline.__init__()` (line ~234), add:
```python
        self._correction_model_cmax = None
        self._correction_model_auc = None
```

In `_ensure_initialized()` (after line ~286), add:
```python
        # Load residual correction model if available
        try:
            from omega_pbpk.ml.models.correction.residual_model import ResidualCorrectionModel
            from pathlib import Path

            model_dir = Path(__file__).resolve().parents[2] / "models" / "correction"
            cmax_path = model_dir / "ridge_cmax.json"
            if cmax_path.exists():
                self._correction_model_cmax = ResidualCorrectionModel.load(cmax_path)
                logger.info("OmegaPipeline: Ridge Cmax correction model loaded.")
            auc_path = model_dir / "ridge_auc.json"
            if auc_path.exists():
                self._correction_model_auc = ResidualCorrectionModel.load(auc_path)
                logger.info("OmegaPipeline: Ridge AUC correction model loaded.")
        except Exception as exc:
            logger.debug("Correction model not available: %s", exc)
```

- [ ] **Step 4: Apply correction in `simulate()`**

After Cmax/AUC computation (around line ~326-327), add:

```python
        # Apply residual correction model
        correction_applied = False
        if self._correction_model_cmax is not None:
            try:
                import numpy as _np
                from omega_pbpk.ml.models.adme.transporter_lookup import is_pgp_substrate

                _logP = float(adme_props.get("logP", 2.0))
                _mw = float(adme_props.get("mw", 300.0))
                _fup = float(adme_props.get("fup", 0.1))
                _peff = float(adme_props.get("peff", 1.0))
                _pgp = 1 if is_pgp_substrate(smiles=request.smiles) else 0

                feat = _np.array([[
                    _logP,
                    _np.log10(max(_mw, 1.0)),
                    _fup,
                    _np.log10(max(request.dose_mg, 0.1)),
                    _pgp,
                    _np.log10(max(_peff, 1e-6)),
                ]])
                cmax_correction = self._correction_model_cmax.predict(feat)[0]
                cmax = cmax / float(_np.exp(cmax_correction))
                correction_applied = True
                logger.debug("Cmax correction: factor=%.3f (log=%.3f)", _np.exp(-cmax_correction), cmax_correction)

                if self._correction_model_auc is not None:
                    auc_correction = self._correction_model_auc.predict(feat)[0]
                    auc = auc / float(_np.exp(auc_correction))
            except Exception as exc:
                logger.debug("Correction model inference failed: %s", exc)

        adme_props["correction_applied"] = correction_applied
```

- [ ] **Step 5: Run tests — verify pass**

```bash
pytest tests/ml/test_correction_integration.py -v
pytest tests/ -m "not slow and not benchmark" -q
```

- [ ] **Step 6: Run 25-drug benchmark — check improvement**

```bash
python scripts/run_full_benchmark.py
```

Expected: AAFE should decrease (exact amount depends on training results). No regressions on drugs without observed data.

- [ ] **Step 7: Commit**

```bash
git add src/omega_pbpk/pipeline/__init__.py tests/ml/test_correction_integration.py
git commit -m "feat: integrate Ridge correction model into OmegaPipeline"
```

---

## Chunk 5: Conformal UQ Integration

**Rationale:** `conformal_uq.py` exists with LHS sampling over parameter bounds, but is not wired into `SimulationResult`. Users need prediction intervals to assess reliability.

### Task 9: Wire UQ into SimulationResult

**Files:**
- Modify: `src/omega_pbpk/pipeline/__init__.py`
- Modify: `src/omega_pbpk/pipeline/__init__.py` (SimulationResult dataclass)
- Test: `tests/ml/test_uq_integration.py`

- [ ] **Step 1: Write failing test**

```python
# tests/ml/test_uq_integration.py
"""Test conformal UQ integration in pipeline."""
import pytest


class TestUQIntegration:
    def test_result_has_intervals(self):
        """SimulationResult should include prediction intervals."""
        from omega_pbpk.pipeline import OmegaPipeline, SimulationRequest

        pipeline = OmegaPipeline()
        result = pipeline.simulate(
            SimulationRequest(
                smiles="CN1C=NC2=C1C(=O)N(C)C(=O)N2C",  # caffeine
                dose_mg=200.0,
            )
        )
        # Check interval fields exist
        assert hasattr(result, "cmax_ci90")
        assert result.cmax_ci90 is not None
        lo, hi = result.cmax_ci90
        assert lo <= result.cmax_mg_L <= hi * 1.1  # point estimate within/near CI

    def test_interval_width_is_reasonable(self):
        """90% CI should not be wider than 100x."""
        from omega_pbpk.pipeline import OmegaPipeline, SimulationRequest

        pipeline = OmegaPipeline()
        result = pipeline.simulate(
            SimulationRequest(
                smiles="CN1C=NC2=C1C(=O)N(C)C(=O)N2C",
                dose_mg=200.0,
            )
        )
        lo, hi = result.cmax_ci90
        assert hi / max(lo, 1e-12) < 100.0
```

- [ ] **Step 2: Run test — verify fail**

```bash
pytest tests/ml/test_uq_integration.py -v
```

- [ ] **Step 3: Add CI fields to SimulationResult**

```python
# In SimulationResult dataclass (line ~220), add:
    cmax_ci90: tuple[float, float] | None = None   # (5th, 95th percentile)
    auc_ci90: tuple[float, float] | None = None
    thalf_ci90: tuple[float, float] | None = None
```

- [ ] **Step 4: Compute UQ in simulate()**

After the correction model application, before building `SimulationResult`, add:

```python
        # Compute conformal UQ intervals from parameter bounds
        _cmax_ci = None
        _auc_ci = None
        _thalf_ci = None
        try:
            from omega_pbpk.uncertainty.conformal_uq import (
                ParameterBounds,
                propagate_conformal_intervals,
            )

            _bounds = ParameterBounds(
                fup_lo=float(adme_props.get("fup_lo", adme_props.get("fup", 0.1) * 0.5)),
                fup_hi=float(adme_props.get("fup_hi", adme_props.get("fup", 0.1) * 2.0)),
                clint_lo=float(adme_props.get("clint_3a4_lo", adme_props.get("clint_3a4", 5.0) * 0.3)),
                clint_hi=float(adme_props.get("clint_3a4_hi", adme_props.get("clint_3a4", 5.0) * 3.0)),
                peff_lo=float(adme_props.get("peff_lo", adme_props.get("peff", 1.0) * 0.5)),
                peff_hi=float(adme_props.get("peff_hi", adme_props.get("peff", 1.0) * 2.0)),
                rbp_lo=float(adme_props.get("rbp_lo", adme_props.get("rbp", 0.55) * 0.8)),
                rbp_hi=float(adme_props.get("rbp_hi", adme_props.get("rbp", 0.55) * 1.2)),
            )
            _uq = propagate_conformal_intervals(
                drug_name="",
                dose_mg=request.dose_mg,
                route=request.route,
                bounds=_bounds,
                n_samples=200,
            )
            _cmax_ci = (_uq.cmax_p5, _uq.cmax_p95)
            _auc_ci = (_uq.auc_p5, _uq.auc_p95)
            _thalf_ci = (_uq.t_half_p5, _uq.t_half_p95)
        except Exception as exc:
            logger.debug("UQ computation failed: %s", exc)
```

Then add to `SimulationResult` construction:
```python
            cmax_ci90=_cmax_ci,
            auc_ci90=_auc_ci,
            thalf_ci90=_thalf_ci,
```

- [ ] **Step 5: Run tests — verify pass**

```bash
pytest tests/ml/test_uq_integration.py -v
pytest tests/ -m "not slow and not benchmark" -q
```

- [ ] **Step 6: Commit**

```bash
git add src/omega_pbpk/pipeline/__init__.py tests/ml/test_uq_integration.py
git commit -m "feat: conformal UQ intervals in SimulationResult (90% CI)"
```

---

## Chunk 6: Batch Screening Engine

### Task 10: Batch Prediction API

**Files:**
- Create: `src/omega_pbpk/screening/__init__.py`
- Create: `src/omega_pbpk/screening/batch.py`
- Test: `tests/test_screening.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_screening.py
"""Tests for batch screening engine."""
import pytest


class TestBatchScreening:
    def test_batch_predict_multiple(self):
        """Batch prediction on 3 SMILES should return 3 results."""
        from omega_pbpk.screening.batch import batch_predict

        smiles_list = [
            "CN1C=NC2=C1C(=O)N(C)C(=O)N2C",  # caffeine
            "CC(=O)NC1=CC=C(O)C=C1",           # acetaminophen
            "CC(C)CC1=CC=C(CC(C)C(=O)O)C=C1",  # ibuprofen
        ]
        results = batch_predict(smiles_list, dose_mg=200.0)
        assert len(results) == 3
        assert all(r["cmax_mg_L"] > 0 for r in results)
        assert all("smiles" in r for r in results)

    def test_batch_handles_invalid_smiles(self):
        """Invalid SMILES should return error entry, not crash."""
        from omega_pbpk.screening.batch import batch_predict

        results = batch_predict(["INVALID_SMILES", "CN1C=NC2=C1C(=O)N(C)C(=O)N2C"])
        assert len(results) == 2
        assert "error" in results[0]
        assert results[1]["cmax_mg_L"] > 0

    def test_batch_ranking(self):
        """Results should be rankable by score."""
        from omega_pbpk.screening.batch import batch_predict, rank_results

        smiles_list = [
            "CN1C=NC2=C1C(=O)N(C)C(=O)N2C",
            "CC(=O)NC1=CC=C(O)C=C1",
        ]
        results = batch_predict(smiles_list, dose_mg=200.0)
        ranked = rank_results(results, objective="cmax")
        assert ranked[0]["cmax_mg_L"] >= ranked[1]["cmax_mg_L"]
```

- [ ] **Step 2: Run test — verify fail**

```bash
pytest tests/test_screening.py -v
```

- [ ] **Step 3: Implement batch prediction**

```python
# src/omega_pbpk/screening/__init__.py
# (empty)
```

```python
# src/omega_pbpk/screening/batch.py
"""Batch screening engine for OmegaPipeline.

Runs OmegaPipeline.simulate() on multiple SMILES with error handling
and result ranking. Uses sequential execution (pipeline is fast enough
at ~73ms/drug that 1000 drugs takes ~73 seconds).

Usage:
    from omega_pbpk.screening.batch import batch_predict, rank_results

    results = batch_predict(smiles_list, dose_mg=100.0)
    ranked = rank_results(results, objective="cmax")
"""
from __future__ import annotations

import logging
import time
from typing import Any

logger = logging.getLogger(__name__)


def batch_predict(
    smiles_list: list[str],
    dose_mg: float = 100.0,
    route: str = "oral",
    duration_h: float = 24.0,
) -> list[dict[str, Any]]:
    """Run OmegaPipeline on multiple SMILES.

    Returns list of dicts, one per SMILES. Each dict has:
        smiles, cmax_mg_L, auc_mg_h_L, t_half_h, tmax_h,
        confidence, cmax_ci90, warnings, latency_ms
    On error: smiles, error
    """
    from omega_pbpk.pipeline import OmegaPipeline, SimulationRequest

    pipeline = OmegaPipeline()
    results = []

    for i, smiles in enumerate(smiles_list):
        t0 = time.time()
        try:
            sim = pipeline.simulate(
                SimulationRequest(
                    smiles=smiles,
                    dose_mg=dose_mg,
                    route=route,
                    duration_h=duration_h,
                )
            )
            results.append({
                "smiles": smiles,
                "cmax_mg_L": sim.cmax_mg_L,
                "auc_mg_h_L": sim.auc0t_mg_h_L,
                "t_half_h": sim.t_half_h,
                "tmax_h": sim.tmax_h,
                "confidence": sim.confidence,
                "cmax_ci90": sim.cmax_ci90 if hasattr(sim, "cmax_ci90") else None,
                "auc_ci90": sim.auc_ci90 if hasattr(sim, "auc_ci90") else None,
                "warnings": sim.warnings,
                "latency_ms": round((time.time() - t0) * 1000, 1),
                "pgp_substrate": sim.adme_properties.get("pgp_substrate", False),
                "correction_applied": sim.adme_properties.get("correction_applied", False),
            })
        except Exception as e:
            results.append({
                "smiles": smiles,
                "error": str(e),
            })

        if (i + 1) % 100 == 0:
            logger.info("Batch progress: %d/%d", i + 1, len(smiles_list))

    return results


def rank_results(
    results: list[dict],
    objective: str = "cmax",
    ascending: bool = False,
) -> list[dict]:
    """Rank batch results by a PK objective.

    Args:
        results: Output of batch_predict()
        objective: "cmax", "auc", "t_half", "tmax"
        ascending: If True, lower values rank higher

    Returns:
        Sorted list (errors at the end).
    """
    key_map = {
        "cmax": "cmax_mg_L",
        "auc": "auc_mg_h_L",
        "t_half": "t_half_h",
        "tmax": "tmax_h",
    }
    key = key_map.get(objective, objective)

    valid = [r for r in results if key in r]
    errors = [r for r in results if key not in r]

    valid.sort(key=lambda r: r[key], reverse=not ascending)

    # Add rank
    for i, r in enumerate(valid):
        r["rank"] = i + 1

    return valid + errors
```

- [ ] **Step 4: Run tests — verify pass**

```bash
pytest tests/test_screening.py -v
```

- [ ] **Step 5: Run full test suite**

```bash
pytest tests/ -m "not slow and not benchmark" -q
```

- [ ] **Step 6: Commit**

```bash
git add src/omega_pbpk/screening/__init__.py src/omega_pbpk/screening/batch.py tests/test_screening.py
git commit -m "feat: batch screening engine — predict + rank multiple SMILES"
```

---

## Chunk 7: Validation Report & Memory Update

### Task 11: Generate Unified Validation Report

**Files:**
- Create: `scripts/generate_validation_report.py`
- Output: `outputs/validation_report.md`

- [ ] **Step 1: Write report generator**

```python
#!/usr/bin/env python3
"""Generate unified validation report from all benchmark results.

Combines Gold (25-drug), Silver (39-drug), Bronze (153-compound),
Temporal (5-drug), and Expanded (285-drug) benchmark results into
a single Markdown report.

Usage:
    python scripts/generate_validation_report.py
"""

import json
import sys
from datetime import datetime
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent


def load_json(path: Path) -> dict | None:
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def main():
    gold = load_json(repo_root / "outputs" / "benchmark_2026-03-15.json")
    silver = load_json(repo_root / "outputs" / "silver_tier_results.json")
    bronze = load_json(repo_root / "outputs" / "bronze_tier_results.json")
    temporal = load_json(repo_root / "outputs" / "temporal_holdout_results.json")

    # Find latest expanded benchmark
    expanded_files = sorted((repo_root / "outputs").glob("expanded_benchmark_*.json"))
    expanded = load_json(expanded_files[-1]) if expanded_files else None

    lines = [
        f"# Omega PBPK — Unified Validation Report",
        f"",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"**Pipeline:** OmegaPipeline (XGBoost ADME ensemble, 35-state PBPK ODE, hybrid Cmax)",
        f"",
        f"---",
        f"",
        f"## Summary",
        f"",
        f"| Tier | Drugs | Metric | AAFE | %2-fold |",
        f"|------|-------|--------|------|---------|",
    ]

    if gold:
        lines.append(f"| Gold (Cmax) | {gold.get('n_drugs', '?')} | Cmax | {gold.get('aafe_cmax', '?'):.2f} | {gold.get('pct_within_2fold_cmax', '?')}% |")
        lines.append(f"| Gold (AUC) | {gold.get('n_drugs', '?')} | AUC | {gold.get('aafe_auc', '?'):.2f} | {gold.get('pct_within_2fold_auc', '?')}% |")
    if silver:
        lines.append(f"| Silver | {silver.get('n_drugs_total', '?')} | t_half | {silver.get('aafe', '?'):.2f} | {silver.get('pct_within_2fold', '?')}% |")
    if temporal:
        lines.append(f"| Temporal | {temporal.get('n_compared', '?')} | t_half | {temporal.get('thalf_aafe', '?'):.2f} | {temporal.get('thalf_pct_2fold', '?')}% |")

    lines.extend(["", "## Bronze Tier (ADME Properties)", ""])
    if bronze:
        lines.append("| Property | AAFE | %2-fold |")
        lines.append("|----------|------|---------|")
        for prop in ["logP", "fup", "rbp", "clint_3a4", "peff"]:
            entry = bronze.get("per_property", {}).get(prop, {})
            if entry:
                lines.append(f"| {prop} | {entry.get('aafe', '?'):.2f} | {entry.get('pct_2fold', '?'):.0f}% |")

    if expanded:
        lines.extend(["", "## Expanded Benchmark (All Reference Drugs)", ""])
        tm = expanded.get("tier_metrics", {})
        for tier_name in ["platinum", "gold", "silver"]:
            m = tm.get(tier_name, {})
            if m and m.get("n_drugs", 0) > 0:
                lines.append(f"### {tier_name.title()} ({m['n_drugs']} drugs)")
                if m.get("cmax_aafe"):
                    lines.append(f"- Cmax AAFE: {m['cmax_aafe']:.2f} ({m['cmax_n']} drugs, {m['cmax_pct_2fold']}% 2-fold)")
                if m.get("auc_aafe"):
                    lines.append(f"- AUC AAFE: {m['auc_aafe']:.2f} ({m['auc_n']} drugs)")
                if m.get("thalf_aafe"):
                    lines.append(f"- t_half AAFE: {m['thalf_aafe']:.2f} ({m['thalf_n']} drugs)")
                lines.append("")

    lines.extend([
        "",
        "## Known Limitations",
        "",
        "| Drug | FE | Root Cause |",
        "|------|----|------------|",
        "| verapamil | 8.8x Cmax | P-gp efflux (partially corrected) |",
        "| ibuprofen | 5.0x Cmax | Extreme protein binding (fup ~0.01) |",
        "| fluconazole | 21x AUC | ADME prediction error |",
        "| phenytoin | 4.7x Cmax | Nonlinear (saturable) metabolism |",
        "",
        "## Comparison to Literature",
        "",
        "| System | Metric | Value | Drugs |",
        "|--------|--------|-------|-------|",
        "| **Omega** | Cmax median FE | 1.73 | 25 |",
        "| **Omega** | AUC median FE | 1.60 | 25 |",
        "| Bayer (2024) | mfce | 1.87 | 9 |",
        "| Jia et al. (2025) | %2-fold | 60% | 106 |",
    ])

    report = "\n".join(lines) + "\n"
    out_path = repo_root / "outputs" / "validation_report.md"
    with open(out_path, "w") as f:
        f.write(report)
    print(f"Report saved: {out_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run report generator**

```bash
python scripts/generate_validation_report.py
```

- [ ] **Step 3: Commit**

```bash
git add scripts/generate_validation_report.py
git commit -m "feat: unified validation report generator"
```

---

### Task 12: Update Memory

**Files:**
- Modify: `~/.claude/projects/-home-jam-Omega/memory/MEMORY.md`
- Modify: `~/.claude/projects/-home-jam-Omega/memory/plan_v5_virtual_screening.md`

- [ ] **Step 1: Update plan_v5 status**

Change `**Status**: Discussed, awaiting data expansion + repo review before finalizing.` to:
```
**Status**: Finalized as Plan v6 (2026-03-16). Data expansion complete (285 drugs).
See: `docs/superpowers/plans/2026-03-16-omega-screening-platform.md`
```

- [ ] **Step 2: Update MEMORY.md Quick Status table**

Update with results from the implementation:
- Add row for P-gp correction status
- Add row for Correction model status
- Add row for UQ integration status
- Add row for Batch screening status
- Update %2-fold with new value (should be ≥72% after P-gp fix)

- [ ] **Step 3: Update Next Steps section**

Replace current next steps with:
```
### Immediate
1. Execute Plan v6 chunks 2-7
2. Run expanded benchmark on all 285 drugs
3. Train correction model on expanded results

### Strategic
- Phase 3 of v5: Screening Engine (batch API + frontend)
- Phase 2 of v5: Conformal UQ calibration on expanded data
- Paper update with expanded validation
```

---

## Sequencing & Dependencies

```
Chunk 1 (Infrastructure)
    │
    ├──→ Chunk 2 (Expanded Benchmark) ──→ Chunk 4 (Correction Model)
    │         │                                    │
    │         └──→ Chunk 3 (P-gp Correction) ──────┤
    │              [can run parallel with Chunk 2]  │
    │                                               ↓
    │                                    Chunk 5 (Conformal UQ)
    │                                               │
    │                                    Chunk 6 (Batch Screening)
    │                                               │
    └──────────────────────────────────→ Chunk 7 (Report + Memory)
```

- Chunks 2 and 3 can run in parallel (independent)
- Chunk 4 depends on Chunk 2 (needs expanded benchmark results) + Chunk 3 (P-gp flag as feature)
- Chunk 5 depends on nothing specific but is best done after Chunk 4
- Chunk 6 depends on Chunks 4+5 (correction model + UQ should be integrated first)
- Chunk 7 is last (aggregates all results)

**Estimated time:** ~4-6 working days with parallelism.

---

## Exit Criteria

| Criterion | Target | How to verify |
|-----------|--------|---------------|
| Expanded benchmark | ≥250 drugs run | `outputs/expanded_benchmark_*.json` has n_success ≥ 250 |
| %2-fold Cmax (25 drugs) | ≥70% | `python scripts/run_full_benchmark.py` → pct_within_2fold ≥ 70 |
| Correction model LOO | LOO AAFE < uncorrected AAFE | Training script output |
| UQ intervals in API | SimulationResult has cmax_ci90 | `test_uq_integration.py` passes |
| Batch screening | 100 SMILES in <10s | `test_screening.py` passes |
| Validation report | Exists | `outputs/validation_report.md` |
| Tests pass | All 48K+ | `pytest tests/ -m "not slow and not benchmark" -q` |

---

## Non-Goals

- Retrain GNN L2 (distillation ceiling proven in v1-v5)
- Neural L3 training (no individual patient data)
- Docker/deployment (no users yet)
- Multi-dose / IV route validation
- Frontend batch screening UI (defer to separate plan)
- ChEMBL ETL (per CLAUDE.md)

---

## Risks

| # | Risk | Severity | Mitigation |
|---|------|----------|------------|
| 1 | Correction model overfits on ~100 drugs | Medium | Ridge regularization + LOO CV; max_correction capped at 1.5 log-units |
| 2 | P-gp correction breaks non-P-gp drugs | Low | Binary lookup → only affects confirmed substrates; verify with full benchmark |
| 3 | Reference database pk_params keys don't match script expectations | Medium | Check key names in first task; the reference JSON may use different field names (e.g., `cmax_mg_L` vs `cmax_value`) |
| 4 | UQ intervals too wide to be useful | Medium | Tune n_samples and bounds; report interval width distribution |
| 5 | Expanded benchmark takes too long | Low | 285 drugs × 73ms ≈ 21 seconds; acceptable |
