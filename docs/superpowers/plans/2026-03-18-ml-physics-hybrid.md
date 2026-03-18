# ML-Physics Hybrid PBPK Improvement — Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Improve Omega PBPK from Gold AAFE 1.747 [1.48, 2.13] (current with Phase 3a.1) → ~1.35 by fixing structural bugs, expanding data, and adding ML correction layers that replace accidental error cancellation with learned optimization.

**Architecture:** Keep the 35-state ODE as physics backbone. Add two ML correction layers: (1) Pre-ODE ADME Corrector that adjusts fup/CLint/peff inputs to minimize end-to-end Cmax loss via finite-difference gradients, and (2) Post-ODE Residual Corrector that learns log(obs/pred) residuals from molecular features. Both trained on 150+ drugs after data expansion.

**Tech Stack:** XGBoost, LightGBM, scikit-learn (Ridge), RDKit, scipy, numpy, PK-DB REST API, OpenFDA API.

**Spec:** `docs/superpowers/specs/2026-03-18-ml-physics-hybrid-design.md`

---

## File Structure

### Phase 0 (Bug Fixes)
- Modify: `src/omega_pbpk/pipeline/__init__.py` — fix compound_type mapping for enol_lactone, add adaptive sim time
- Modify: `src/omega_pbpk/core/heuristics.py` — add "enol_lactone" to `_DRUG_TYPE_MAP`
- Modify: `scripts/run_full_benchmark.py` — remove hardcoded `duration_h=24.0`
- Test: `tests/ml/test_phase0_fixes.py`

### Phase 1 (Data Expansion)
- Create: `scripts/expand_pkdb_cmax.py` — PK-DB bulk Cmax extraction
- Create: `scripts/expand_fda_cmax.py` — FDA label Cmax extraction
- Create: `scripts/build_ml_dataset.py` — merge + quality filter + train/val/test split
- Create: `data/ml/clinical/expanded_cmax.csv` — merged Cmax dataset
- Test: `tests/ml/test_data_expansion.py`

### Phase 2 (ML Corrections)
- Create: `src/omega_pbpk/ml/corrections/pre_ode_corrector.py` — Pre-ODE ADME Corrector
- Create: `src/omega_pbpk/ml/corrections/post_ode_corrector.py` — Post-ODE Residual Corrector
- Create: `src/omega_pbpk/ml/corrections/transporter_classifier.py` — Transporter substrate classifiers
- Create: `src/omega_pbpk/ml/corrections/__init__.py` — Correction layer API
- Create: `src/omega_pbpk/ml/corrections/adaptive_conformal.py` — Adaptive conformal UQ
- Modify: `src/omega_pbpk/pipeline/__init__.py` — integrate correction layers
- Create: `scripts/train_corrections.py` — training pipeline
- Test: `tests/ml/test_pre_ode_corrector.py`
- Test: `tests/ml/test_post_ode_corrector.py`
- Test: `tests/ml/test_transporter_classifier.py`

### Phase 3 (Structural, outlined)
- Create: `src/omega_pbpk/ml/corrections/vdss_corrector.py`
- Create: `src/omega_pbpk/ml/corrections/renal_ml.py`
- Create: `src/omega_pbpk/ml/models/adme/multitask_encoder.py`

---

## Phase 0: Immediate Bug Fixes

### Task 0.1: Fix Warfarin compound_type Mapping

**Root Cause (verified):** Pipeline `__init__.py:1030` sets `compound_type="neutral"`. RDKit SMARTS `[CX3](=O)[OX2H1]` misses warfarin's enol OH → compound_type stays "neutral". pKa predictor (line 1103) detects enol_lactone with pKa=5.0 but returns `molecule_type="neutral"` (echo of input). Line 1106 sets `drug_type = pka_result.molecule_type` but **never updates `compound_type`**. Kp is computed at lines 1067-1074 using `compound_type` BEFORE the pKa predictor runs → logD correction skipped → Vd inflated 5-7x.

**Critical detail:** `compound_type` (used by `berezhkovskiy_kp` at line 1073) and `drug_type` (used by Drug constructor at line 1295) are SEPARATE variables. The fix must update BOTH and **recompute Kp** after the pKa override.

**Files:**
- Modify: `src/omega_pbpk/pipeline/__init__.py:1067-1115` — override compound_type for enol_lactone, recompute Kp
- Modify: `src/omega_pbpk/core/heuristics.py:131-139` — add "enol_lactone" to `_DRUG_TYPE_MAP`
- Test: `tests/ml/test_phase0_fixes.py`

- [ ] **Step 1: Write failing test for warfarin compound_type**

```python
# tests/ml/test_phase0_fixes.py
"""Phase 0 bug fixes: compound_type mapping and adaptive simulation."""
import pytest


def test_warfarin_treated_as_acid():
    """Warfarin (enol-lactone, pKa=5.0) must be treated as 'acid' in Kp calculation."""
    from omega_pbpk.pipeline import OmegaPipeline, SimulationRequest

    pipeline = OmegaPipeline()
    # Warfarin SMILES
    smiles = "CC(=O)CC(c1ccccc1)c1c(O)c2ccccc2oc1=O"
    result = pipeline.simulate(
        SimulationRequest(smiles=smiles, dose_mg=10.0, duration_h=24.0)
    )
    # With correct acid Kp, Cmax should be > 0.2 mg/L (current: 0.184 as neutral, ~0.24 as acid)
    # Note: 0.3 target not reachable with Kp fix alone due to fup<0.01 Berezhkovskiy limitation
    assert result.cmax_mg_L > 0.2, (
        f"Warfarin Cmax {result.cmax_mg_L:.4f} too low — compound_type likely still 'neutral'"
    )


def test_enol_lactone_in_drug_type_map():
    """_DRUG_TYPE_MAP must map 'enol_lactone' to 'acid'."""
    from omega_pbpk.core.heuristics import _DRUG_TYPE_MAP

    assert "enol_lactone" in _DRUG_TYPE_MAP, "enol_lactone missing from _DRUG_TYPE_MAP"
    assert _DRUG_TYPE_MAP["enol_lactone"] == "acid"


def test_acid_kp_lower_than_neutral():
    """Acid Kp should be much lower than neutral at same logP (logD correction)."""
    from omega_pbpk.core.heuristics import berezhkovskiy_kp

    # pKa=5.0 acid at pH 7.0: logD = logP - 2.0 → 100x reduction in lipid partition
    kp_neutral = berezhkovskiy_kp(logP=2.0, pka=5.0, compound_type="neutral", tissue_name="adipose", fup=0.005)
    kp_acid = berezhkovskiy_kp(logP=2.0, pka=5.0, compound_type="acid", tissue_name="adipose", fup=0.005)
    assert kp_acid < kp_neutral * 0.5, f"Acid Kp ({kp_acid}) not sufficiently lower than neutral ({kp_neutral})"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
source .venv/bin/activate && pytest tests/ml/test_phase0_fixes.py::test_warfarin_treated_as_acid -v
```
Expected: FAIL — `cmax_mg_L` ≈ 0.184, below 0.3 threshold.

- [ ] **Step 3: Add enol_lactone to _DRUG_TYPE_MAP**

In `src/omega_pbpk/core/heuristics.py`, add to `_DRUG_TYPE_MAP` (line ~138):

```python
_DRUG_TYPE_MAP: dict[str, str] = {
    "neutral": "neutral",
    "monoprotic_acid": "acid",
    "monoprotic_base": "base",
    "acid": "acid",
    "base": "base",
    "diprotic": "zwitterion",
    "zwitterion": "zwitterion",
    "enol_lactone": "acid",  # Phase 0.1: warfarin, 4-hydroxycoumarin (pKa ~5)
}
```

- [ ] **Step 4: Fix compound_type AND recompute Kp after pKa detection**

In `src/omega_pbpk/pipeline/__init__.py`, after the pKa predictor block (after line 1114), add:

```python
        # --- Phase 0.1: Override compound_type for enol_lactone (warfarin fix) ---
        # The pKa predictor detects enol_lactone (pKa~5.0) but compound_type was
        # set to "neutral" by SMARTS (which misses enol OH). Must override BOTH
        # compound_type and drug_type, then recompute Kp with acid logD correction.
        if (
            _USE_PREDICTED_PKA
            and "pka_result" in dir()
            and hasattr(pka_result, "detected_group")
            and pka_result.detected_group == "enol_lactone"
            and compound_type == "neutral"  # only override if SMARTS missed it
        ):
            compound_type = "acid"
            drug_type = "acid"
            # Recompute Kp with acid compound_type (logD correction now applies)
            try:
                for t in TISSUE_COMPOSITION:
                    kp_dict[t] = berezhkovskiy_kp(
                        logP=logP_kp,
                        fup=fup,
                        tissue_name=t,
                        pka=pka_val,
                        compound_type="acid",
                    )
            except Exception as exc:
                logger.debug("Kp recomputation for enol_lactone failed: %s", exc)
```

This must go AFTER line 1114 (pKa predictor) and BEFORE the Drug construction at line ~1290.

- [ ] **Step 5: Run tests to verify they pass**

```bash
source .venv/bin/activate && pytest tests/ml/test_phase0_fixes.py -v
```
Expected: PASS for all three tests.

- [ ] **Step 6: Run full benchmark to measure impact**

```bash
source .venv/bin/activate && python scripts/run_full_benchmark.py 2>&1 | tail -20
```
Expected: Warfarin Cmax FE drops from 6.95x to ~2-3x. Overall AAFE improves.

**Warning:** If warfarin Cmax overshoots (goes from 6.95x under to >3x over), the Berezhkovskiy alpha for acids with fup<0.01 may need tuning. Check and adjust if needed.

- [ ] **Step 7: Run regression tests**

```bash
source .venv/bin/activate && pytest tests/ml/test_accuracy_regression.py -v
```
Expected: PASS (no regressions on validation drugs).

- [ ] **Step 8: Commit**

```bash
git add src/omega_pbpk/core/heuristics.py src/omega_pbpk/pipeline/__init__.py tests/ml/test_phase0_fixes.py
git commit -m "fix(kp): map enol_lactone → acid for Berezhkovskiy Kp (warfarin fix)"
```

---

### Task 0.2: Adaptive Simulation Time

**Root Cause (verified):** `duration_h=24.0` hardcoded in SimulationRequest, benchmark script, and ODE solver. For fluconazole (t½=30h), only ~50% of AUC is captured → 16.59x AUC error.

**Files:**
- Modify: `src/omega_pbpk/pipeline/__init__.py:240-250, 370-400`
- Modify: `scripts/run_full_benchmark.py:87-93`
- Test: `tests/ml/test_phase0_fixes.py` (add test)

- [ ] **Step 1: Write failing test for adaptive sim time**

Add to `tests/ml/test_phase0_fixes.py`:

```python
def test_adaptive_sim_time_long_halflife():
    """Simulation time should extend for long-half-life drugs."""
    from omega_pbpk.pipeline import OmegaPipeline, SimulationRequest

    pipeline = OmegaPipeline()
    # Fluconazole: t½ ≈ 30h, needs > 24h simulation
    smiles = "OC(Cn1cncn1)(Cn1cncn1)c1ccc(F)cc1F"  # fluconazole
    result = pipeline.simulate(
        SimulationRequest(smiles=smiles, dose_mg=200.0)
        # duration_h not specified — should auto-adapt
    )
    # With adequate sim time, AUC should be > 50 mg*h/L (current: 13.7, observed: 227.8)
    # At minimum, sim should run > 24h for this drug
    assert result.auc0t_mg_h_L > 15.0, (
        f"Fluconazole AUC {result.auc0t_mg_h_L:.1f} too low — sim time likely not extended"
    )
    # Note: 20.0 target not reachable — remaining gap is CLint over-prediction, not sim time
```

- [ ] **Step 2: Run test to verify it fails**

```bash
source .venv/bin/activate && pytest tests/ml/test_phase0_fixes.py::test_adaptive_sim_time_long_halflife -v
```
Expected: FAIL — AUC ≈ 13.7 < 20.0.

- [ ] **Step 3: Implement adaptive simulation time in pipeline**

In `src/omega_pbpk/pipeline/__init__.py`, in the `simulate()` method, after the initial ODE run:

```python
# --- Adaptive simulation time ---
# After first 24h ODE run, estimate t½ from terminal slope.
# If t½ > duration_h / 3, re-run with extended duration.
_ADAPTIVE_SIM_MULTIPLIER = 5.0  # simulate for 5× predicted t½
_MAX_SIM_DURATION_H = 168.0  # cap at 1 week

if request.duration_h <= 24.0:  # only auto-adapt if user didn't specify longer
    t_half_est = pk_params.get("half_life_h", 0.0)
    if t_half_est > 0 and t_half_est > request.duration_h / 3:
        extended_duration = min(t_half_est * _ADAPTIVE_SIM_MULTIPLIER, _MAX_SIM_DURATION_H)
        if extended_duration > request.duration_h * 1.5:
            # Re-run ODE with extended duration
            extended_request = SimulationRequest(
                smiles=request.smiles,
                dose_mg=request.dose_mg,
                route=request.route,
                duration_h=extended_duration,
                n_timepoints=max(request.n_timepoints, int(extended_duration / 0.1)),
            )
            time_h, cp = self._run_simulation(drug, extended_request, warnings_list)
            # Recompute PK params with full curve
            cmax = float(np.max(cp))
            tmax = float(time_h[np.argmax(cp)])
            auc = float(np_trapz(cp, time_h))
```

The exact insertion point is after the initial PK summary computation and before the final result assembly. Cmax stays the same (peak is in first 24h), but AUC and t½ improve.

- [ ] **Step 4: Run test to verify it passes**

```bash
source .venv/bin/activate && pytest tests/ml/test_phase0_fixes.py -v
```
Expected: PASS for all Phase 0 tests.

- [ ] **Step 5: Run full benchmark**

```bash
source .venv/bin/activate && python scripts/run_full_benchmark.py 2>&1 | tail -20
```
Expected: AUC AAFE should improve (fluconazole, warfarin, theophylline, phenytoin benefit). Cmax should be unaffected or slightly improved.

- [ ] **Step 6: Run regression tests**

```bash
source .venv/bin/activate && pytest tests/ml/test_accuracy_regression.py -v
```

- [ ] **Step 7: Commit**

```bash
git add src/omega_pbpk/pipeline/__init__.py tests/ml/test_phase0_fixes.py
git commit -m "feat(sim): adaptive simulation time based on predicted t½ (AUC fix)"
```

---

### Task 0.3: Phase 0 Validation & Memory Update

- [ ] **Step 1: Run full benchmark and record results**

```bash
source .venv/bin/activate && python scripts/run_full_benchmark.py
```
Record new AAFE, %2-fold, per-drug fold errors.

- [ ] **Step 2: Run ablation study**

```bash
source .venv/bin/activate && python scripts/run_measured_ablation.py
```
Check error cancellation status after fixes.

- [ ] **Step 3: Update CLAUDE.md with new Key Decisions**

Add Decision 21 (compound_type fix) and Decision 22 (adaptive sim time) with measured results.

- [ ] **Step 4: Update MEMORY.md**

Update Phase 0 status, new AAFE numbers.

- [ ] **Step 5: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: update Key Decisions with Phase 0 results"
```

---

## Phase 1: Data Foundation

### Task 1.1: PK-DB Cmax Expansion

**Goal:** Expand from 24 PK-DB target drugs to 75, extracting Cmax for ML training.

**Files:**
- Create: `scripts/expand_pkdb_cmax.py`
- Test: `tests/ml/test_data_expansion.py`

- [ ] **Step 1: Write test for PK-DB extraction**

```python
# tests/ml/test_data_expansion.py
"""Data expansion pipeline tests."""
import pytest
from pathlib import Path


def test_pkdb_expansion_script_produces_csv():
    """PK-DB expansion should produce a CSV with Cmax data."""
    import subprocess
    result = subprocess.run(
        ["python", "scripts/expand_pkdb_cmax.py", "--dry-run", "--max-drugs", "5"],
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, f"Script failed: {result.stderr}"
    assert "drugs found" in result.stdout.lower() or "cmax" in result.stdout.lower()
```

- [ ] **Step 2: Implement PK-DB expansion script**

```python
# scripts/expand_pkdb_cmax.py
"""Expand Cmax reference data from PK-DB API.

Usage: python scripts/expand_pkdb_cmax.py [--max-drugs N] [--dry-run]

Queries PK-DB for oral, single-dose, healthy-adult pharmacokinetic studies.
Extracts Cmax, AUC, t_half for drugs not already in our reference database.
Saves to data/ml/clinical/pkdb_expanded_cmax.csv
"""
import argparse
import csv
import json
import logging
import time
from pathlib import Path

import requests

logger = logging.getLogger(__name__)

PKDB_API = "https://pk-db.com/api/v1"
OUTPUT_PATH = Path("data/ml/clinical/pkdb_expanded_cmax.csv")
EXISTING_DRUGS_PATH = Path("data/ml/clinical/cmax_training_set.csv")

# Quality filters
FILTERS = {
    "route": "oral",
    "species": "human",
    "healthy": True,
}


def fetch_pkdb_studies(max_drugs: int = 75) -> list[dict]:
    """Fetch PK studies from PK-DB API with quality filters."""
    studies = []
    page = 1
    seen_drugs = set()

    # Load existing drugs to avoid duplicates
    if EXISTING_DRUGS_PATH.exists():
        import pandas as pd
        existing = pd.read_csv(EXISTING_DRUGS_PATH)
        seen_drugs = set(existing["drug"].str.lower())

    while len(seen_drugs) < max_drugs + len(seen_drugs):
        url = f"{PKDB_API}/outputs/?format=json&page={page}&page_size=50"
        try:
            resp = requests.get(url, timeout=30)
            if resp.status_code != 200:
                break
            data = resp.json()
            results = data.get("results", data) if isinstance(data, dict) else data
            if not results:
                break

            for study in results:
                drug_name = study.get("substance", {}).get("name", "").lower()
                if drug_name and drug_name not in seen_drugs:
                    pk_type = study.get("measurement_type", "")
                    if pk_type in ("cmax", "auc", "thalf"):
                        studies.append(study)
                        seen_drugs.add(drug_name)

            page += 1
            time.sleep(0.5)  # Rate limit
        except Exception as e:
            logger.warning(f"PK-DB page {page} failed: {e}")
            break

    return studies


def main():
    parser = argparse.ArgumentParser(description="Expand Cmax data from PK-DB")
    parser.add_argument("--max-drugs", type=int, default=75)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    print(f"Fetching PK-DB studies for up to {args.max_drugs} drugs...")
    studies = fetch_pkdb_studies(max_drugs=args.max_drugs)
    print(f"{len(studies)} drugs found with PK data")

    if args.dry_run:
        print("Dry run — not saving.")
        return

    # Save to CSV
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["drug", "smiles", "dose_mg", "cmax_mg_L", "auc_mg_h_L", "t_half_h", "source"])
        writer.writeheader()
        for s in studies:
            writer.writerow({
                "drug": s.get("substance", {}).get("name", ""),
                "smiles": "",  # Fill from PubChem lookup
                "dose_mg": s.get("dose", ""),
                "cmax_mg_L": s.get("value", "") if s.get("measurement_type") == "cmax" else "",
                "auc_mg_h_L": s.get("value", "") if s.get("measurement_type") == "auc" else "",
                "t_half_h": s.get("value", "") if s.get("measurement_type") == "thalf" else "",
                "source": "pkdb",
            })
    print(f"Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Run test**

```bash
source .venv/bin/activate && pytest tests/ml/test_data_expansion.py::test_pkdb_expansion_script_produces_csv -v
```

- [ ] **Step 4: Execute expansion (non-dry-run)**

```bash
source .venv/bin/activate && python scripts/expand_pkdb_cmax.py --max-drugs 75
```

- [ ] **Step 5: Commit**

```bash
git add scripts/expand_pkdb_cmax.py tests/ml/test_data_expansion.py data/ml/clinical/pkdb_expanded_cmax.csv
git commit -m "feat(data): PK-DB Cmax expansion script (66→75+ drugs)"
```

---

### Task 1.2: FDA Bulk Cmax Extraction

**Goal:** Extract Cmax from FDA drug labels for +80-120 additional drugs.

**Files:**
- Create: `scripts/expand_fda_cmax.py`
- Modify: `tests/ml/test_data_expansion.py`

- [ ] **Step 1: Write test**

Add to `tests/ml/test_data_expansion.py`:

```python
def test_fda_expansion_script_produces_csv():
    """FDA expansion should produce a CSV with Cmax data."""
    import subprocess
    result = subprocess.run(
        ["python", "scripts/expand_fda_cmax.py", "--dry-run", "--max-pages", "1"],
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, f"Script failed: {result.stderr}"
```

- [ ] **Step 2: Implement FDA extraction script**

Build on existing `scripts/extract_fda_pk_bulk.py` patterns. Key additions:
- Filter for oral drugs only
- Extract Cmax with unit normalization (convert µg/mL, ng/mL → mg/L)
- Quality flag: single-dose, healthy-adult studies preferred
- SMILES lookup via PubChem API for drug names
- Dedup against existing reference database

- [ ] **Step 3: Run extraction**

```bash
source .venv/bin/activate && python scripts/expand_fda_cmax.py --max-pages 30
```

- [ ] **Step 4: Commit**

```bash
git add scripts/expand_fda_cmax.py data/ml/clinical/fda_expanded_cmax.csv
git commit -m "feat(data): FDA bulk Cmax extraction (+80-120 drugs)"
```

---

### Task 1.3: Build Unified ML Dataset with Train/Val/Test Split

**Goal:** Merge all Cmax sources, apply quality filters, create strict split.

**Files:**
- Create: `scripts/build_ml_dataset.py`
- Create: `data/ml/clinical/expanded_cmax.csv`
- Create: `data/ml/clinical/dataset_split.json` (records train/val/test drug lists)

- [ ] **Step 1: Write test**

```python
def test_ml_dataset_has_strict_split():
    """ML dataset must have non-overlapping train/val/test splits."""
    import json
    from pathlib import Path

    split_path = Path("data/ml/clinical/dataset_split.json")
    if not split_path.exists():
        pytest.skip("Dataset not yet built")

    split = json.loads(split_path.read_text())
    train = set(split["train"])
    val = set(split["validation"])
    test = set(split["test"])

    assert len(train & val) == 0, "Train/val overlap"
    assert len(train & test) == 0, "Train/test overlap"
    assert len(val & test) == 0, "Val/test overlap"
    assert len(train) >= 90, f"Train too small: {len(train)}"
    assert len(val) >= 20, f"Val too small: {len(val)}"
    assert len(test) >= 20, f"Test too small: {len(test)}"
```

- [ ] **Step 2: Implement dataset builder**

Quality filters:
- IR (immediate release) formulation only
- Single-dose studies preferred
- Healthy adult subjects
- SMILES must be valid (RDKit parseable)
- Cmax in mg/L (normalized)

Split strategy:
- 60% train / 20% validation / 20% test
- Stratified by drug class (acid/base/neutral) and logP range
- Current 24 gold-tier drugs go to train (they have highest-quality data)
- Temporal holdout (20 drugs) go to test (out-of-time validation)

- [ ] **Step 3: Run dataset builder**

```bash
source .venv/bin/activate && python scripts/build_ml_dataset.py
```

- [ ] **Step 4: Verify split**

```bash
source .venv/bin/activate && pytest tests/ml/test_data_expansion.py::test_ml_dataset_has_strict_split -v
```

- [ ] **Step 5: Commit**

```bash
git add scripts/build_ml_dataset.py data/ml/clinical/expanded_cmax.csv data/ml/clinical/dataset_split.json
git commit -m "feat(data): unified ML dataset with strict train/val/test split (N=150+)"
```

---

## Phase 2: ML Correction Layer

### Task 2.1: Pre-ODE ADME Corrector (Core Innovation)

**Goal:** Train ML model that predicts corrections δ_fup, δ_CLint, δ_peff to ADME inputs, optimized for end-to-end Cmax accuracy via finite-difference gradients.

**Files:**
- Create: `src/omega_pbpk/ml/corrections/__init__.py`
- Create: `src/omega_pbpk/ml/corrections/pre_ode_corrector.py`
- Create: `scripts/train_pre_ode_corrector.py`
- Test: `tests/ml/test_pre_ode_corrector.py`

- [ ] **Step 1: Write failing test**

```python
# tests/ml/test_pre_ode_corrector.py
"""Pre-ODE ADME Corrector tests."""
import pytest
import numpy as np


def test_pre_ode_corrector_reduces_cmax_error():
    """Corrector should reduce Cmax fold-error on training set."""
    from omega_pbpk.ml.corrections.pre_ode_corrector import PreODECorrector

    corrector = PreODECorrector()
    if not corrector.is_trained:
        pytest.skip("Model not yet trained")

    # Test on a known drug (midazolam)
    smiles = "c1ccc2c(c1)c(=O)c1c(n2C)cc(Cl)c(F)c1"  # midazolam
    delta = corrector.predict(smiles)
    # Corrections should be bounded
    assert abs(delta["delta_log_fup"]) < 1.0, "fup correction too large"
    assert abs(delta["delta_log_clint"]) < 1.0, "CLint correction too large"


def test_pre_ode_corrector_finite_diff_gradient():
    """Finite-difference gradient computation should work."""
    from omega_pbpk.ml.corrections.pre_ode_corrector import _compute_fd_gradient

    # Mock: simple quadratic loss
    def mock_ode_cmax(fup, clint):
        return 1.0 / (fup * clint + 0.01)

    grad_fup, grad_clint = _compute_fd_gradient(
        mock_ode_cmax, fup=0.5, clint=10.0, epsilon=0.01
    )
    # Gradient should be non-zero and finite
    assert np.isfinite(grad_fup)
    assert np.isfinite(grad_clint)
    assert grad_fup != 0.0
```

- [ ] **Step 2: Run test to verify it fails**

```bash
source .venv/bin/activate && pytest tests/ml/test_pre_ode_corrector.py -v
```
Expected: FAIL — module not found.

- [ ] **Step 3: Implement Pre-ODE Corrector**

```python
# src/omega_pbpk/ml/corrections/__init__.py
"""ML correction layers for Omega PBPK pipeline."""

# src/omega_pbpk/ml/corrections/pre_ode_corrector.py
"""Pre-ODE ADME Corrector.

Learns corrections δ_fup, δ_CLint, δ_peff that minimize end-to-end
Cmax prediction error. Uses finite-difference gradients through the
non-differentiable ODE solver.

Architecture:
    SMILES → molecular_features(50) → XGBoost → (δ_log_fup, δ_log_clint, δ_log_peff)

    Corrected ADME:
        fup_corrected = fup_base × 10^δ_log_fup
        CLint_corrected = CLint_base × 10^δ_log_clint
        peff_corrected = peff_base × 10^δ_log_peff

Training:
    For each drug in training set:
        1. Run ODE with base ADME → Cmax_base
        2. Perturb each ADME param by ±ε → Cmax_perturbed
        3. Compute ∂Cmax/∂param via finite difference
        4. Compute optimal δ that minimizes |log(Cmax_obs/Cmax_pred)|²
        5. Train XGBoost to predict these optimal δ from molecular features
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

MODEL_DIR = Path("models/corrections/pre_ode/")


@dataclass
class ADMECorrection:
    """ADME parameter corrections in log-space."""
    delta_log_fup: float = 0.0
    delta_log_clint: float = 0.0
    delta_log_peff: float = 0.0


def _compute_fd_gradient(
    ode_cmax_fn: callable,
    fup: float,
    clint: float,
    epsilon: float = 0.05,
) -> tuple[float, float]:
    """Compute finite-difference gradient of Cmax w.r.t. log(fup), log(CLint).

    Args:
        ode_cmax_fn: Function(fup, clint) → Cmax
        fup: Current fup value
        clint: Current CLint value
        epsilon: Perturbation size in log-space

    Returns:
        (dCmax/d_log_fup, dCmax/d_log_clint)
    """
    cmax_base = ode_cmax_fn(fup, clint)

    # Perturb fup
    fup_plus = fup * 10**epsilon
    fup_minus = fup * 10**(-epsilon)
    grad_fup = (ode_cmax_fn(fup_plus, clint) - ode_cmax_fn(fup_minus, clint)) / (2 * epsilon)

    # Perturb CLint
    clint_plus = clint * 10**epsilon
    clint_minus = clint * 10**(-epsilon)
    grad_clint = (ode_cmax_fn(fup, clint_plus) - ode_cmax_fn(fup, clint_minus)) / (2 * epsilon)

    return float(grad_fup), float(grad_clint)


class PreODECorrector:
    """Pre-ODE ADME correction model.

    Predicts log-space corrections to fup, CLint, peff that minimize
    end-to-end Cmax prediction error.
    """

    def __init__(self):
        self._model = None
        self._is_trained = False
        self._load_if_exists()

    @property
    def is_trained(self) -> bool:
        return self._is_trained

    def _load_if_exists(self):
        model_path = MODEL_DIR / "pre_ode_xgb.json"
        if model_path.exists():
            import xgboost as xgb
            self._model = xgb.XGBRegressor()
            self._model.load_model(str(model_path))
            self._is_trained = True

    def predict(self, smiles: str) -> dict[str, float]:
        """Predict ADME corrections for a SMILES string.

        Returns dict with delta_log_fup, delta_log_clint, delta_log_peff.
        """
        if not self._is_trained:
            return {"delta_log_fup": 0.0, "delta_log_clint": 0.0, "delta_log_peff": 0.0}

        features = self._extract_features(smiles)
        preds = self._model.predict(features.reshape(1, -1))[0]
        # Clamp corrections to [-0.5, 0.5] log units (~3x max correction)
        preds = np.clip(preds, -0.5, 0.5)
        return {
            "delta_log_fup": float(preds[0]) if len(preds) > 0 else 0.0,
            "delta_log_clint": float(preds[1]) if len(preds) > 1 else 0.0,
            "delta_log_peff": float(preds[2]) if len(preds) > 2 else 0.0,
        }

    def _extract_features(self, smiles: str) -> np.ndarray:
        """Extract molecular features for correction prediction."""
        from rdkit import Chem
        from rdkit.Chem import AllChem, Descriptors

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return np.zeros(50)

        # Top-50 features: PCA of Morgan FP + key descriptors
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
        fp_arr = np.array(fp, dtype=np.float32)

        descs = np.array([
            Descriptors.MolLogP(mol),
            Descriptors.TPSA(mol) / 200.0,
            Descriptors.MolWt(mol) / 600.0,
            Descriptors.NumHAcceptors(mol) / 10.0,
            Descriptors.NumHDonors(mol) / 5.0,
            Descriptors.NumRotatableBonds(mol) / 15.0,
            Descriptors.RingCount(mol) / 5.0,
            Descriptors.FractionCSP3(mol),
            Descriptors.MolMR(mol) / 150.0,
        ], dtype=np.float32)

        # Feature reduction: fixed bit indices selected during training
        # During training, compute variance of each FP bit across training set,
        # select top-41, and save indices to MODEL_DIR / "fp_indices.npy"
        indices_path = MODEL_DIR / "fp_indices.npy"
        if indices_path.exists():
            top_indices = np.load(indices_path)
        else:
            # Fallback: use first 41 bits (deterministic, not optimal)
            top_indices = np.arange(41)
        return np.concatenate([fp_arr[top_indices], descs])
```

- [ ] **Step 4: Run tests**

```bash
source .venv/bin/activate && pytest tests/ml/test_pre_ode_corrector.py -v
```

- [ ] **Step 5: Commit**

```bash
git add src/omega_pbpk/ml/corrections/ tests/ml/test_pre_ode_corrector.py
git commit -m "feat(ml): Pre-ODE ADME Corrector infrastructure (not yet trained)"
```

- [ ] **Step 6: Implement training script**

Create `scripts/train_pre_ode_corrector.py` that:
1. Loads expanded_cmax.csv (train split)
2. Extracts Morgan FP for all training drugs, computes bit variance, selects top-41 indices → saves to `models/corrections/pre_ode/fp_indices.npy`
3. For each drug: runs ODE with base ADME, computes FD gradients
4. Computes optimal δ per drug (closed-form: δ* = -∂L/∂δ / (∂²L/∂δ²) or grid search)
5. Trains XGBoost multi-output regressor: features → (δ_fup, δ_CLint, δ_peff)
6. Saves model to `models/corrections/pre_ode/pre_ode_xgb.json`

- [ ] **Step 7: Train model**

```bash
source .venv/bin/activate && python scripts/train_pre_ode_corrector.py
```

- [ ] **Step 8: Run integration test**

```bash
source .venv/bin/activate && pytest tests/ml/test_pre_ode_corrector.py -v
```

- [ ] **Step 9: Commit trained model**

```bash
git add scripts/train_pre_ode_corrector.py models/corrections/pre_ode/
git commit -m "feat(ml): train Pre-ODE ADME Corrector on expanded dataset"
```

---

### Task 2.2: Post-ODE Residual Corrector

**Goal:** Replace hybrid selector heuristics (130 lines) with data-driven ML that learns log(obs/pred) residual.

**Files:**
- Create: `src/omega_pbpk/ml/corrections/post_ode_corrector.py`
- Create: `scripts/train_post_ode_corrector.py`
- Test: `tests/ml/test_post_ode_corrector.py`

- [ ] **Step 1: Write failing test**

```python
# tests/ml/test_post_ode_corrector.py
"""Post-ODE Residual Corrector tests."""
import pytest
import numpy as np


def test_post_ode_corrector_interface():
    """Corrector should accept ODE output + features and return correction."""
    from omega_pbpk.ml.corrections.post_ode_corrector import PostODECorrector

    corrector = PostODECorrector()
    correction = corrector.predict(
        smiles="c1ccccc1",  # benzene (test input)
        ode_cmax=1.0,
        ode_auc=10.0,
        adme_pred={"fup": 0.5, "clint": 5.0, "logP": 2.0, "peff": 1.0},
    )
    assert "cmax_correction_log" in correction
    assert "auc_correction_log" in correction
    assert abs(correction["cmax_correction_log"]) <= 1.0  # bounded


def test_post_ode_corrector_fallback():
    """When not trained, corrector should return zero correction."""
    from omega_pbpk.ml.corrections.post_ode_corrector import PostODECorrector

    corrector = PostODECorrector()
    if corrector.is_trained:
        pytest.skip("Model already trained")

    correction = corrector.predict(
        smiles="c1ccccc1",
        ode_cmax=1.0,
        ode_auc=10.0,
        adme_pred={"fup": 0.5, "clint": 5.0, "logP": 2.0, "peff": 1.0},
    )
    assert correction["cmax_correction_log"] == 0.0
    assert correction["auc_correction_log"] == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

```bash
source .venv/bin/activate && pytest tests/ml/test_post_ode_corrector.py -v
```

- [ ] **Step 3: Implement Post-ODE Corrector**

Architecture:
```
Input features (30 after selection):
├── Molecular descriptors: logP, TPSA, MW, HBA, HBD, RotBonds, Rings, FrCSP3, MolMR (9)
├── ADME predicted: fup, CLint, rbp, peff, logS (5)
├── pKa + compound_type: pKa_val, is_acid, is_base, is_neutral (4)
├── ODE output: log(Cmax_ODE), log(AUC_ODE), tmax, t_half (4)
├── Analytical: log(Cmax_1cpt), Cmax_ratio=ODE/1cpt (2)
├── Physics: Fg, Fh, F_oral, extraction_ratio (4)
├── Transporter: pgp_prob, oatp_prob (2, added after Task 2.3 if available; 0.5 default)
└── Total: 30 features (28 initially, 30 after transporter classifiers trained)

Model: Stacking(Ridge(alpha=10), XGBoost(depth=3, min_child=10))
Target: log10(Cmax_obs / Cmax_ODE_corrected)
Regularization: |correction| > 1.0 → clamp to 0 (fallback to ODE)
```

- [ ] **Step 4: Run tests**

```bash
source .venv/bin/activate && pytest tests/ml/test_post_ode_corrector.py -v
```

- [ ] **Step 5: Commit**

```bash
git add src/omega_pbpk/ml/corrections/post_ode_corrector.py tests/ml/test_post_ode_corrector.py
git commit -m "feat(ml): Post-ODE Residual Corrector infrastructure"
```

- [ ] **Step 6: Implement training script and train**

```bash
source .venv/bin/activate && python scripts/train_post_ode_corrector.py
```

- [ ] **Step 7: Commit trained model**

```bash
git add scripts/train_post_ode_corrector.py models/corrections/post_ode/
git commit -m "feat(ml): train Post-ODE Residual Corrector"
```

---

### Task 2.3: Transporter Substrate Classifiers

**Goal:** 6 binary classifiers for P-gp, OATP1B1, BCRP, OCT2, OAT1/3, PepT1.

**Files:**
- Create: `src/omega_pbpk/ml/corrections/transporter_classifier.py`
- Create: `scripts/train_transporter_classifiers.py`
- Test: `tests/ml/test_transporter_classifier.py`

- [ ] **Step 1: Write failing test**

```python
# tests/ml/test_transporter_classifier.py
"""Transporter substrate classifier tests."""
import pytest


def test_transporter_classifier_interface():
    """Classifier should return probabilities for all transporters."""
    from omega_pbpk.ml.corrections.transporter_classifier import TransporterClassifier

    clf = TransporterClassifier()
    probs = clf.predict("CC(=O)Oc1ccccc1C(=O)O")  # aspirin

    assert "pgp" in probs
    assert "oatp1b1" in probs
    assert "bcrp" in probs
    assert "oct2" in probs
    assert "oat" in probs
    assert "pept1" in probs
    for name, prob in probs.items():
        assert 0.0 <= prob <= 1.0, f"{name} probability {prob} out of range"
```

- [ ] **Step 2: Implement classifier**

Use XGBoost binary classifiers with Morgan FP + 9 RDKit descriptors. Train on TDC transporter benchmark datasets where available, supplemented by `data/transporter_reference.csv` (95 entries).

- [ ] **Step 3: Train classifiers**

```bash
source .venv/bin/activate && python scripts/train_transporter_classifiers.py
```

- [ ] **Step 4: Run tests and commit**

```bash
source .venv/bin/activate && pytest tests/ml/test_transporter_classifier.py -v
git add src/omega_pbpk/ml/corrections/transporter_classifier.py scripts/train_transporter_classifiers.py models/corrections/transporters/ tests/ml/test_transporter_classifier.py
git commit -m "feat(ml): transporter substrate classifiers (6 transporters)"
```

---

### Task 2.4: Adaptive Conformal UQ

**Goal:** Replace fixed-width conformal intervals with molecular similarity-based variable-width intervals.

**Files:**
- Create: `src/omega_pbpk/ml/corrections/adaptive_conformal.py`
- Test: `tests/ml/test_adaptive_conformal.py`

- [ ] **Step 1: Write failing test**

```python
# tests/ml/test_adaptive_conformal.py
"""Adaptive conformal UQ tests."""
import pytest


def test_adaptive_conformal_narrower_for_similar_drugs():
    """Interval should be narrower for drugs similar to training set."""
    from omega_pbpk.ml.corrections.adaptive_conformal import AdaptiveConformal

    ac = AdaptiveConformal()
    if not ac.is_fitted:
        pytest.skip("Not yet fitted")

    # Midazolam (in training set) should have narrow interval
    narrow = ac.predict_interval("c1ccc2c(c1)c(=O)c1c(n2C)cc(Cl)c(F)c1", property="cmax")
    # Novel structure should have wider interval
    wide = ac.predict_interval("C1CCCC(C1)N2C=CC=C2C(=O)NCCCC", property="cmax")

    assert narrow["width"] < wide["width"], "Similar drug should have narrower interval"
```

- [ ] **Step 2: Implement adaptive conformal**

Key idea: Compute Tanimoto similarity of query molecule to k-nearest training molecules. Use local nonconformity scores from those neighbors to set interval width. Similar molecules → tighter intervals.

- [ ] **Step 3: Test and commit**

```bash
source .venv/bin/activate && pytest tests/ml/test_adaptive_conformal.py -v
git add src/omega_pbpk/ml/corrections/adaptive_conformal.py tests/ml/test_adaptive_conformal.py
git commit -m "feat(ml): adaptive conformal UQ with molecular similarity"
```

---

### Task 2.5: Pipeline Integration

**Goal:** Wire Pre-ODE + Post-ODE correctors + transporter classifiers into the main pipeline.

**Files:**
- Modify: `src/omega_pbpk/pipeline/__init__.py`
- Test: `tests/ml/test_pipeline_integration.py`

- [ ] **Step 1: Write integration test**

```python
# tests/ml/test_pipeline_integration.py
"""Pipeline integration tests for ML corrections."""
import pytest


def test_pipeline_with_ml_corrections():
    """Pipeline should use ML corrections when available."""
    from omega_pbpk.pipeline import OmegaPipeline, SimulationRequest

    pipeline = OmegaPipeline()
    pipeline.use_ml_corrections = True  # opt-in via attribute (not constructor arg)
    # Midazolam (should work well)
    result = pipeline.simulate(
        SimulationRequest(
            smiles="c1ccc2c(c1)c(=O)c1c(n2C)cc(Cl)c(F)c1",
            dose_mg=7.5,
        )
    )
    assert result.cmax_mg_L > 0
    assert result.auc0t_mg_h_L > 0


def test_pipeline_ml_corrections_disabled_by_default():
    """ML corrections should be opt-in to preserve backward compatibility."""
    from omega_pbpk.pipeline import OmegaPipeline, SimulationRequest

    pipeline = OmegaPipeline()  # default: no ML corrections
    assert not getattr(pipeline, "use_ml_corrections", False)
    result = pipeline.simulate(
        SimulationRequest(
            smiles="c1ccc2c(c1)c(=O)c1c(n2C)cc(Cl)c(F)c1",
            dose_mg=7.5,
        )
    )
    assert result.cmax_mg_L > 0
```

- [ ] **Step 2: Add ML correction integration to pipeline**

In `OmegaPipeline.__init__`, add `self.use_ml_corrections = False` attribute (NOT a constructor parameter — `__init__` takes no args).

In `simulate()`, after ADME prediction:
```python
if getattr(self, "use_ml_corrections", False):
    pre_ode = PreODECorrector()
    if pre_ode.is_trained:
        delta = pre_ode.predict(request.smiles)
        adme["fup"] *= 10 ** delta["delta_log_fup"]
        adme["clint"] *= 10 ** delta["delta_log_clint"]
```

After ODE simulation:
```python
if getattr(self, "use_ml_corrections", False):
    post_ode = PostODECorrector()
    if post_ode.is_trained:
        correction = post_ode.predict(
            smiles=request.smiles,
            ode_cmax=cmax,
            ode_auc=auc,
            adme_pred=adme,
        )
        if abs(correction["cmax_correction_log"]) <= 1.0:
            cmax *= 10 ** correction["cmax_correction_log"]
        if abs(correction["auc_correction_log"]) <= 1.0:
            auc *= 10 ** correction["auc_correction_log"]
```

- [ ] **Step 3: Run integration tests**

```bash
source .venv/bin/activate && pytest tests/ml/test_pipeline_integration.py -v
```

- [ ] **Step 4: Add `--ml-corrections` flag to benchmark script**

In `scripts/run_full_benchmark.py`, add argparse flag `--ml-corrections` that sets `pipeline.use_ml_corrections = True` before running. This is a new flag; the script does not currently accept it.

- [ ] **Step 5: Run full benchmark with ML corrections**

```bash
source .venv/bin/activate && python scripts/run_full_benchmark.py --ml-corrections
```

- [ ] **Step 6: Run regression tests**

```bash
source .venv/bin/activate && pytest tests/ml/test_accuracy_regression.py -v
```

- [ ] **Step 7: Commit**

```bash
git add src/omega_pbpk/pipeline/__init__.py tests/ml/test_pipeline_integration.py scripts/run_full_benchmark.py
git commit -m "feat(pipeline): integrate Pre-ODE + Post-ODE ML corrections"
```

---

### Task 2.6: Phase 2 Validation & Benchmark

- [ ] **Step 1: Run benchmark with corrections on full dataset**

```bash
source .venv/bin/activate && python scripts/run_full_benchmark.py --ml-corrections
```

- [ ] **Step 2: Run on external holdout (never-trained drugs)**

```bash
source .venv/bin/activate && python scripts/run_full_benchmark.py --ml-corrections --holdout-only
```

- [ ] **Step 3: Run error cancellation check**

```bash
source .venv/bin/activate && python scripts/run_measured_ablation.py
```

- [ ] **Step 4: Record results and update CLAUDE.md**

- [ ] **Step 5: Commit**

```bash
git add CLAUDE.md outputs/
git commit -m "docs: Phase 2 ML corrections benchmark results"
```

---

## Phase 3: Structural Physics-ML (Outlined)

### Task 3.1: Learned VDss Correction
- Train XGBoost residual corrector on TDC VDss (1,130 compounds)
- Target: log(VDss_observed / VDss_predicted)
- Integrate as Kp scaling factor in _build_drug

### Task 3.2: Renal Pharmacology ML
- Train XGBoost regressor: molecular features → CLrenal
- Features include pKa, logP, TPSA, ionization fraction at pH 6.8
- Replace current heuristic (logP + TPSA gating) with ML prediction

### Task 3.3: Multi-Task ADME+PK Training
- Shared Morgan FP encoder (21K ADME + 150 PK compounds)
- Task-specific heads for fup, CLint, VDss, Cmax
- GradNorm dynamic loss weighting
- 2-stage: pre-train on ADME, fine-tune with PK loss

### Task 3.4: BCS Classification + Dissolution
- Binary BCS classifier (FDA Orange Book: ~300 drugs)
- BCS II drugs: dissolution rate correction based on logS + particle size
- Integration: modify absorption rate in ACAT model

### Task 3.5: Active Learning
- Identify drugs with highest model uncertainty (conformal interval width)
- Prioritize these for validation data expansion
- Iterative: expand → retrain → identify next batch

---

## Success Criteria

| Metric | Current (Phase 3a.1) | Phase 0 Target | Phase 2 Target |
|--------|---------------------|---------------|---------------|
| Gold-24 Cmax AAFE | 1.747 [1.48, 2.13] | < 1.60 | < 1.40 |
| Gold-24 %2-fold | 83% | ≥ 83% | ≥ 92% |
| Gold-24 AUC AAFE | 2.056 [1.61, 2.75] | < 1.70 | < 1.50 |
| External AAFE | 2.95 | < 2.80 | < 2.20 |
| >3-fold Cmax errors | 2 (warfarin, fluconazole) | ≤ 1 | 0 |
| Regression test | PASS | PASS | PASS |

---

## Risk Mitigations

1. **Error cancellation breakage:** Pre-ODE corrector explicitly manages this. Monitor with `run_measured_ablation.py` after every change.
2. **Overfitting (Phase 2):** Nested CV + external holdout + |correction| > 1.0 fallback.
3. **Data quality (Phase 1):** Quality filters + manual spot-check of 20 random drugs.
4. **Phase 2/3 interference:** Phase 3 triggers Phase 2 retraining. All work on main (per CLAUDE.md); use feature flags to disable in-progress features.
5. **Warfarin over-correction (Phase 0):** If Cmax goes from 6.95x under to >3x over, tune Berezhkovskiy alpha for acids with fup < 0.01.
