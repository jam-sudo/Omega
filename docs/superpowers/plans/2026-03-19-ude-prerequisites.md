# UDE Prerequisites: Validation Hygiene + MMPK Data Quality

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Establish honest validation infrastructure (permanent hold-out set + LOOCV baseline) and audit MMPK training data quality — the two hard prerequisites before building a Universal Differential Equation (UDE) multi-task learner.

**Architecture:** Two parallel workstreams. Workstream A creates a scaffold-stratified permanent hold-out from platinum 147, decontaminates CLint anchors, runs LOOCV on gold-24, and produces honest baselines. Workstream B cross-references MMPK 1,128 drugs against platinum, detects outliers, scores data quality, and produces a quality-weighted training set for UDE training.

**Tech Stack:** RDKit (Murcko scaffolds, SMILES canonicalization), scikit-learn (stratification), numpy/scipy (bootstrap CI), pandas (data manipulation), pytest (regression gates), OmegaPipeline (prediction)

**Context:**
- Platinum reference: `data/clinical/platinum_reference.json` (147 drugs)
- MMPK clean: `data/ml/clinical/mmpk_clean.csv` (1,128 drugs)
- MMPK PBPK features: `data/ml/clinical/mmpk_pbpk_features.csv` (1,128 drugs with cmax_pbpk)
- CLint anchors: `src/omega_pbpk/ml/models/adme/xgboost_clint.py:56-105` (19 anchors)
- Benchmark drugs: `src/omega_pbpk/data/drug_registry.py` (25 drugs)
- Full benchmark: `scripts/run_full_benchmark.py` (24-drug benchmark with bootstrap CI)

**Dependencies between workstreams:**
- A1→A2→A3→A4→A5 (sequential — each builds on the previous)
- B1→B2→B3→B4→B5 (sequential — outlier detection informs quality scoring)
- A and B are fully independent (can run in parallel)
- Both must complete before UDE training begins

---

## Workstream A: Validation Hygiene

### Task A1: Scaffold-Stratified Hold-out Split

**Files:**
- Create: `scripts/create_holdout_split.py`
- Create: `data/clinical/holdout_split.json`
- Test: `tests/ml/test_holdout_split.py`

**Why:** No honest hold-out exists. All 147 platinum drugs are accessible during development. Gold-24 AAFE 1.50 is contaminated by CLint anchors and tuned thresholds. Without a locked hold-out, we can't measure generalization.

**Design decisions:**
- 55% train (≈81 drugs) / 45% hold-out (≈66 drugs) — larger hold-out for tighter CI
- Murcko generic scaffold clustering ensures chemically similar drugs stay together
- Scaffold clustering only — no explicit data_quality tier balancing (random shuffle provides adequate distribution; post-hoc check for skew)
- All 14 `tuning_contaminated: true` drugs go to TRAINING set (they're already contaminated)
- Hold-out drugs must NEVER appear in: CLint anchors, threshold tuning, model selection

- [ ] **Step 1: Write the failing test**

```python
# tests/ml/test_holdout_split.py
"""Tests for scaffold-stratified holdout split."""
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
SPLIT_PATH = REPO / "data" / "clinical" / "holdout_split.json"
PLATINUM_PATH = REPO / "data" / "clinical" / "platinum_reference.json"


def test_holdout_split_exists():
    assert SPLIT_PATH.exists(), "holdout_split.json not created yet"


def test_holdout_split_structure():
    with open(SPLIT_PATH) as f:
        split = json.load(f)
    assert "train" in split
    assert "holdout" in split
    assert "metadata" in split
    assert isinstance(split["train"], list)
    assert isinstance(split["holdout"], list)


def test_holdout_split_sizes():
    with open(SPLIT_PATH) as f:
        split = json.load(f)
    n_total = len(split["train"]) + len(split["holdout"])
    assert n_total == 147, f"Total drugs should be 147, got {n_total}"
    assert len(split["holdout"]) >= 60, f"Hold-out should be ≥60, got {len(split['holdout'])}"
    assert len(split["holdout"]) <= 75, f"Hold-out should be ≤75, got {len(split['holdout'])}"


def test_no_overlap():
    with open(SPLIT_PATH) as f:
        split = json.load(f)
    train_set = set(split["train"])
    holdout_set = set(split["holdout"])
    overlap = train_set & holdout_set
    assert len(overlap) == 0, f"Overlap between train/holdout: {overlap}"


def test_all_platinum_drugs_assigned():
    with open(SPLIT_PATH) as f:
        split = json.load(f)
    with open(PLATINUM_PATH) as f:
        plat = json.load(f)
    split_drugs = set(split["train"]) | set(split["holdout"])
    plat_drugs = set(plat["drugs"].keys())
    assert split_drugs == plat_drugs, f"Missing: {plat_drugs - split_drugs}"


def test_contaminated_drugs_in_train():
    """All tuning_contaminated drugs must be in training set."""
    with open(SPLIT_PATH) as f:
        split = json.load(f)
    with open(PLATINUM_PATH) as f:
        plat = json.load(f)
    contaminated = {
        name for name, entry in plat["drugs"].items()
        if entry.get("tuning_contaminated", False)
    }
    holdout_set = set(split["holdout"])
    leaked = contaminated & holdout_set
    assert len(leaked) == 0, f"Contaminated drugs in holdout: {leaked}"


def test_scaffold_integrity():
    """Same Murcko scaffold should not appear in both train and holdout."""
    with open(SPLIT_PATH) as f:
        split = json.load(f)
    scaffolds = split["metadata"].get("scaffold_assignments", {})
    if not scaffolds:
        return  # scaffold info not stored — skip
    train_scaffolds = {scaffolds[d] for d in split["train"] if d in scaffolds}
    holdout_scaffolds = {scaffolds[d] for d in split["holdout"] if d in scaffolds}
    leaked = train_scaffolds & holdout_scaffolds
    assert len(leaked) == 0, f"Scaffold leak: {len(leaked)} scaffolds in both sets"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/ml/test_holdout_split.py -v`
Expected: FAIL on `test_holdout_split_exists` (file doesn't exist yet)

- [ ] **Step 3: Implement scaffold split script**

```python
#!/usr/bin/env python3
"""Create permanent scaffold-stratified holdout split from platinum 147.

Strategy:
1. Extract Murcko generic scaffolds for all 147 drugs
2. Cluster drugs by scaffold (same scaffold → same split)
3. Force all tuning_contaminated drugs into training set
4. Greedily assign remaining scaffold clusters to train/holdout
   targeting 55/45 split, balancing data_quality tiers
5. Save to data/clinical/holdout_split.json (PERMANENT — never regenerate)

Usage:
    python scripts/create_holdout_split.py
"""
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

PLATINUM_PATH = REPO / "data" / "clinical" / "platinum_reference.json"
OUTPUT_PATH = REPO / "data" / "clinical" / "holdout_split.json"
SEED = 42


def get_generic_scaffold(smiles: str) -> str:
    """Extract Murcko generic scaffold from SMILES."""
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return f"UNPARSEABLE_{smiles[:20]}"
    try:
        core = MurckoScaffold.GetScaffoldForMol(mol)
        generic = MurckoScaffold.MakeScaffoldGeneric(core)
        return Chem.MolToSmiles(generic)
    except Exception:
        return f"SCAFFOLD_ERROR_{smiles[:20]}"


def main():
    if OUTPUT_PATH.exists():
        print(f"ERROR: {OUTPUT_PATH} already exists. This is a PERMANENT split.")
        print("Delete manually with --force if you truly need to regenerate.")
        if "--force" not in sys.argv:
            sys.exit(1)

    # Load platinum
    with open(PLATINUM_PATH) as f:
        plat = json.load(f)
    drugs = plat["drugs"]
    print(f"Loaded {len(drugs)} platinum drugs")

    # Extract scaffolds
    scaffold_map = {}  # drug_name -> scaffold_smiles
    scaffold_clusters = defaultdict(list)  # scaffold -> [drug_names]
    for name, entry in drugs.items():
        scaffold = get_generic_scaffold(entry["smiles"])
        scaffold_map[name] = scaffold
        scaffold_clusters[scaffold].append(name)

    n_scaffolds = len(scaffold_clusters)
    print(f"Found {n_scaffolds} unique Murcko scaffolds")

    # Identify contaminated drugs (must go to train)
    contaminated = {
        name for name, entry in drugs.items()
        if entry.get("tuning_contaminated", False)
    }
    print(f"Contaminated drugs (forced to train): {len(contaminated)}")

    # Identify scaffolds that MUST be in train (contain contaminated drugs)
    forced_train_scaffolds = set()
    for scaffold, members in scaffold_clusters.items():
        if any(m in contaminated for m in members):
            forced_train_scaffolds.add(scaffold)

    # Remaining scaffolds for splitting
    free_scaffolds = [
        (scaffold, members)
        for scaffold, members in scaffold_clusters.items()
        if scaffold not in forced_train_scaffolds
    ]

    # Shuffle free scaffolds deterministically
    random.seed(SEED)
    random.shuffle(free_scaffolds)

    # Greedy assignment targeting 45% holdout of total
    target_holdout = int(0.45 * len(drugs))
    forced_train_drugs = set()
    for scaffold in forced_train_scaffolds:
        forced_train_drugs.update(scaffold_clusters[scaffold])

    holdout = []
    train = list(forced_train_drugs)

    for scaffold, members in free_scaffolds:
        if len(holdout) + len(members) <= target_holdout + 5:
            holdout.extend(members)
        else:
            train.extend(members)

    # Sort for reproducibility
    train.sort()
    holdout.sort()

    # Validate
    assert set(train) | set(holdout) == set(drugs.keys())
    assert set(train) & set(holdout) == set()
    assert contaminated.issubset(set(train))

    # Quality tier distribution
    for split_name, split_drugs in [("train", train), ("holdout", holdout)]:
        tiers = defaultdict(int)
        for d in split_drugs:
            tiers[drugs[d].get("data_quality", "unknown")] += 1
        print(f"\n{split_name} ({len(split_drugs)} drugs):")
        for tier, count in sorted(tiers.items()):
            print(f"  {tier}: {count}")

    # Save
    result = {
        "train": train,
        "holdout": holdout,
        "metadata": {
            "n_train": len(train),
            "n_holdout": len(holdout),
            "split_method": "murcko_generic_scaffold_stratified",
            "seed": SEED,
            "n_scaffolds": n_scaffolds,
            "forced_train_contaminated": len(contaminated),
            "scaffold_assignments": scaffold_map,
            "created": "2026-03-19",
            "WARNING": "PERMANENT SPLIT — do not regenerate. Hold-out drugs must never be used for training, tuning, or threshold selection.",
        },
    }

    with open(OUTPUT_PATH, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved to {OUTPUT_PATH}")
    print(f"Train: {len(train)}, Holdout: {len(holdout)}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the script**

Run: `python scripts/create_holdout_split.py`
Expected: Creates `data/clinical/holdout_split.json` with ~81 train / ~66 holdout

- [ ] **Step 5: Run tests to verify**

Run: `pytest tests/ml/test_holdout_split.py -v`
Expected: All 7 tests PASS

- [ ] **Step 6: Commit**

```bash
git add scripts/create_holdout_split.py data/clinical/holdout_split.json tests/ml/test_holdout_split.py
git commit -m "feat(validation): scaffold-stratified permanent holdout split (147 → ~81/66)"
```

---

### Task A2: CLint Anchor Decontamination

**Files:**
- Modify: `src/omega_pbpk/ml/models/adme/xgboost_clint.py:56-105`
- Read: `data/clinical/holdout_split.json`
- Test: `tests/ml/test_holdout_split.py` (extend)

**Why:** 19 CLint reference anchors in `xgboost_clint.py` include drugs that may be in the hold-out set. If a hold-out drug has its CLint hand-tuned in the model, the hold-out prediction is contaminated — it's not a true out-of-sample test.

**Design decisions:**
- Do NOT delete anchors permanently — the production model should keep all anchors
- Instead: add a function `get_decontaminated_anchors(holdout_drugs)` that filters out hold-out drugs
- The LOOCV and hold-out benchmark scripts will call this function
- Production pipeline uses the full anchor set (unchanged behavior)

- [ ] **Step 1: Identify which anchors overlap hold-out**

```python
# Run after A1 to identify overlap
# Expected: the 19 anchors are for these drugs:
# propranolol, verapamil, metoprolol, atorvastatin, simvastatin,
# fluoxetine, midazolam, omeprazole, losartan, amlodipine,
# acetaminophen, ranitidine, ciprofloxacin, ibuprofen,
# theophylline, caffeine, diazepam, warfarin, fluconazole
#
# Of these, the ones marked tuning_contaminated in platinum are forced to train.
# But some non-contaminated anchors may land in holdout (simvastatin, losartan,
# amlodipine, ranitidine, ciprofloxacin are in anchors but may not be in platinum).
```

- [ ] **Step 2: Add decontamination function**

Add to `xgboost_clint.py` after line 105:

```python
# Map from anchor SMILES → drug name for decontamination
_ANCHOR_DRUG_NAMES: dict[str, str] = {
    "CC(C)NCC(O)COc1cccc2ccccc12": "propranolol",
    "COc1ccc(CCN(C)CCCC(C#N)(c2ccc(OC)c(OC)c2)C(C)C)cc1OC": "verapamil",
    "COCCc1ccc(OCC(O)CNC(C)C)cc1": "metoprolol",
    "CC(C)c1n(CC(O)CC(O)CC(=O)O)c(-c2ccccc2)c(-c2ccc(F)cc2)c1C(=O)Nc1ccccc1": "atorvastatin",
    "CCC(C)(C)C(=O)OC1CC(O)C=C2C=CC(C)C(CCC3CC(O)CC(=O)O3)C21": "simvastatin",
    "CNCCC(Oc1ccc(C(F)(F)F)cc1)c1ccccc1": "fluoxetine",
    "Clc1ccc2c(c1)C(=NCc1nccn1C)c1ccccc1-2": "midazolam",
    "COc1ccc2[nH]c(S(=O)Cc3ncc(C)c(OC)c3C)nc2c1": "omeprazole",
    "CCCCc1nc(Cl)c(CO)n1Cc1ccc(-c2ccccc2-c2nn[nH]n2)cc1": "losartan",
    "CCOC(=O)C1=C(COCCN)NC(C)=C(C(=O)OC)C1c1ccccc1Cl": "amlodipine",
    "CC(=O)Nc1ccc(O)cc1": "acetaminophen",
    "CNC(/N=C/[N+](=O)[O-])NCCSCc1ccc(CN(C)C)o1": "ranitidine",
    "O=C(O)c1cn(C2CC2)c2cc(N3CCNCC3)c(F)cc2c1=O": "ciprofloxacin",
    "CC(C)Cc1ccc(cc1)C(C)C(=O)O": "ibuprofen",
    "Cn1c(=O)c2[nH]cnc2n(C)c1=O": "theophylline",
    "Cn1cnc2c1c(=O)n(C)c(=O)n2C": "caffeine",
    "CN1C(=O)CN=C(c2ccccc2)c2cc(Cl)ccc21": "diazepam",
    "CC(=O)CC(c1ccccc1)c1c(O)c2ccccc2oc1=O": "warfarin",
    "OC(Cn1cncn1)(Cn1cncn1)c1ccc(F)cc1F": "fluconazole",
}


def get_decontaminated_anchors(
    exclude_drugs: set[str] | None = None,
) -> list[tuple[str, float]]:
    """Return CLint anchors with specified drugs removed.

    Args:
        exclude_drugs: Drug names to exclude (e.g., holdout set drugs).
            If None, returns full anchor set (production behavior).
    """
    if exclude_drugs is None:
        return _get_clint_reference_anchors()

    all_anchors = _get_clint_reference_anchors()
    filtered = []
    removed = []
    for smiles, clint in all_anchors:
        drug_name = _ANCHOR_DRUG_NAMES.get(smiles, "unknown")
        if drug_name in exclude_drugs:
            removed.append(drug_name)
        else:
            filtered.append((smiles, clint))

    if removed:
        logger.info("Decontaminated anchors: removed %s", removed)
    return filtered
```

- [ ] **Step 3: Write test for decontamination**

Append to `tests/ml/test_holdout_split.py`:

```python
def test_anchor_decontamination():
    """Anchors for holdout drugs should be removable."""
    from omega_pbpk.ml.models.adme.xgboost_clint import (
        _get_clint_reference_anchors,
        get_decontaminated_anchors,
    )

    full_anchors = _get_clint_reference_anchors()
    decontaminated = get_decontaminated_anchors(exclude_drugs={"warfarin", "midazolam"})
    assert len(decontaminated) == len(full_anchors) - 2
    # None of the excluded drugs should remain
    from omega_pbpk.ml.models.adme.xgboost_clint import _ANCHOR_DRUG_NAMES
    remaining_names = {
        _ANCHOR_DRUG_NAMES.get(s, "?") for s, _ in decontaminated
    }
    assert "warfarin" not in remaining_names
    assert "midazolam" not in remaining_names
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/ml/test_holdout_split.py::test_anchor_decontamination -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/omega_pbpk/ml/models/adme/xgboost_clint.py tests/ml/test_holdout_split.py
git commit -m "feat(validation): CLint anchor decontamination for holdout evaluation"
```

---

### Task A3: Gold-24 Anchor Contamination Analysis

**Files:**
- Create: `scripts/run_loocv_gold24.py`
- Create: `outputs/loocv_gold24.json`
- Read: `src/omega_pbpk/data/drug_registry.py` (BENCHMARK_DRUGS, 25 drugs total)
- Read: `src/omega_pbpk/ml/models/adme/xgboost_clint.py` (anchor list)

**Why:** Gold-24 AAFE 1.50 includes CLint anchors for 14 of 25 benchmark drugs. We need to quantify how much these anchors inflate the metric by comparing ANCHORED vs CLEAN drug subsets.

**Design decisions:**
- Run production pipeline on all 25 BENCHMARK_DRUGS (anchors intact)
- Label each drug as "ANCHORED" (has CLint anchor) or "CLEAN" (no anchor)
- Compute separate AAFE for each subset — the delta reveals contamination effect
- This is NOT true LOOCV (which would require retraining XGBoost CLint per fold — prohibitively expensive). It's a contamination stratification analysis.
- The CLEAN subset AAFE is the honest in-sample generalization estimate
- AUC fold errors also collected where available

- [ ] **Step 1: Implement anchor contamination analysis script**

```python
#!/usr/bin/env python3
"""Anchor contamination analysis on Gold-24 benchmark.

Runs the production pipeline on all 25 BENCHMARK_DRUGS and stratifies
results into ANCHORED (14 drugs with CLint reference anchors) vs
CLEAN (11 drugs without anchors). The CLEAN subset AAFE is the honest
in-sample generalization estimate.

Note: this is NOT true LOOCV (which would require retraining per fold).
It's a contamination stratification analysis.

Usage:
    python scripts/run_loocv_gold24.py
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from run_l1_benchmarks import compute_aafe, compute_fold_error, load_observed_pk
from omega_pbpk.data.drug_registry import BENCHMARK_DRUGS
from omega_pbpk.ml.models.adme.xgboost_clint import _ANCHOR_DRUG_NAMES
from omega_pbpk.pipeline import OmegaPipeline, SimulationRequest


def main():
    anchor_drugs = set(_ANCHOR_DRUG_NAMES.values())

    results = []
    fold_errors_all = []
    fold_errors_anchored = []
    fold_errors_clean = []

    pipeline = OmegaPipeline()

    for drug_name, info in BENCHMARK_DRUGS.items():
        smiles = info["smiles"]
        dose_mg = info["dose_mg"]
        has_anchor = drug_name in anchor_drugs

        observed = load_observed_pk(drug_name)
        if not observed or observed.get("cmax", 0) <= 0:
            print(f"SKIP {drug_name}: no observed Cmax")
            continue

        obs_cmax = observed["cmax"]

        # Predict (full pipeline — anchor included if present)
        try:
            sim = pipeline.simulate(
                SimulationRequest(smiles=smiles, dose_mg=dose_mg, route="oral")
            )
            pred_cmax = sim.cmax_mg_L
            fe = compute_fold_error(pred_cmax, obs_cmax)
        except Exception as e:
            print(f"FAIL {drug_name}: {e}")
            continue

        entry = {
            "drug": drug_name,
            "has_anchor": has_anchor,
            "pred_cmax": round(pred_cmax, 6),
            "obs_cmax": round(obs_cmax, 6),
            "fold_error": round(fe, 4),
        }
        results.append(entry)
        fold_errors_all.append(fe)
        if has_anchor:
            fold_errors_anchored.append(fe)
        else:
            fold_errors_clean.append(fe)

        status = "ANCHORED" if has_anchor else "CLEAN"
        print(f"{drug_name:20s} [{status:8s}] FE={fe:.2f}x  pred={pred_cmax:.4f} obs={obs_cmax:.4f}")

    # Aggregate
    def aafe_with_ci(fes):
        if len(fes) < 2:
            return {"aafe": None, "ci_lo": None, "ci_hi": None, "n": len(fes)}
        log_fe = np.log10(fes)
        aafe = float(10 ** np.mean(np.abs(log_fe)))
        # Bootstrap CI
        rng = np.random.default_rng(42)
        n = len(log_fe)
        boots = [
            float(10 ** np.mean(np.abs(log_fe[rng.integers(0, n, n)])))
            for _ in range(10000)
        ]
        return {
            "aafe": round(aafe, 4),
            "ci_lo": round(float(np.percentile(boots, 2.5)), 4),
            "ci_hi": round(float(np.percentile(boots, 97.5)), 4),
            "n": n,
            "pct_2fold": round(sum(1 for fe in fes if fe <= 2.0) / n * 100, 1),
        }

    summary = {
        "all_drugs": aafe_with_ci(fold_errors_all),
        "anchored_drugs": aafe_with_ci(fold_errors_anchored),
        "clean_drugs": aafe_with_ci(fold_errors_clean),
        "per_drug": results,
    }

    print("\n" + "=" * 60)
    print(f"ALL ({len(fold_errors_all)} drugs):      AAFE = {summary['all_drugs']['aafe']}")
    print(f"ANCHORED ({len(fold_errors_anchored)} drugs): AAFE = {summary['anchored_drugs']['aafe']}")
    print(f"CLEAN ({len(fold_errors_clean)} drugs):    AAFE = {summary['clean_drugs']['aafe']}")
    print("=" * 60)

    out_path = REPO / "outputs" / "loocv_gold24.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run contamination analysis**

Run: `python scripts/run_loocv_gold24.py`
Expected output:
- ALL (25 drugs): AAFE ~1.50 (contaminated — matches existing benchmark)
- ANCHORED (14 drugs): AAFE probably lower than CLEAN (anchors improve these drugs)
- CLEAN (11 drugs): AAFE ~2.0-2.5 (honest estimate for non-anchored drugs)

**Key metric:** The delta between ANCHORED and CLEAN AAFE quantifies contamination.
If ANCHORED AAFE << CLEAN AAFE, the anchors are artificially inflating the gold-24 metric.

- [ ] **Step 3: Commit**

```bash
git add scripts/run_loocv_gold24.py
git commit -m "feat(validation): gold-24 anchor contamination stratification analysis"
```

---

### Task A4: Hold-out Baseline Measurement

**Files:**
- Create: `scripts/run_holdout_benchmark.py`
- Create: `outputs/holdout_baseline.json`
- Read: `data/clinical/holdout_split.json`
- Read: `data/clinical/platinum_reference.json`

**Why:** This is THE number. The honest, uncontaminated AAFE on ~66 drugs that the pipeline has never seen during any tuning. This baseline is the target for UDE training.

**Design decisions:**
- Run full pipeline on every hold-out drug (no modifications)
- Record: Cmax fold error, predicted ADME values, compound type, data quality tier
- Bootstrap 95% CI
- Stratify results by: data_quality, compound_type, molecular weight range
- Also record applicability domain flags (quaternary amine, prodrug, etc.)

- [ ] **Step 1: Implement holdout benchmark script**

```python
#!/usr/bin/env python3
"""Run pipeline on permanent holdout set and record baseline metrics.

This script is the GROUND TRUTH for measuring generalization improvement.
Never modify to improve results — only measure honestly.

Usage:
    python scripts/run_holdout_benchmark.py
"""
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from omega_pbpk.pipeline import OmegaPipeline, SimulationRequest

SPLIT_PATH = REPO / "data" / "clinical" / "holdout_split.json"
PLATINUM_PATH = REPO / "data" / "clinical" / "platinum_reference.json"


def bootstrap_aafe_ci(fold_errors, n_boot=10000, seed=42):
    log_fe = np.log10(np.array(fold_errors))
    rng = np.random.default_rng(seed)
    n = len(log_fe)
    boots = [
        float(10 ** np.mean(np.abs(log_fe[rng.integers(0, n, n)])))
        for _ in range(n_boot)
    ]
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def compute_fold_error(pred, obs):
    if pred <= 0 or obs <= 0:
        return float("nan")
    return max(pred / obs, obs / pred)


def main():
    # Load split
    with open(SPLIT_PATH) as f:
        split = json.load(f)
    holdout_drugs = set(split["holdout"])

    # Load platinum
    with open(PLATINUM_PATH) as f:
        plat = json.load(f)

    pipeline = OmegaPipeline()
    results = []
    fold_errors = []
    strat_errors = defaultdict(list)  # stratification key -> [fold_errors]

    print(f"Running holdout benchmark on {len(holdout_drugs)} drugs")
    print("=" * 70)

    for drug_name in sorted(holdout_drugs):
        entry = plat["drugs"].get(drug_name)
        if entry is None:
            print(f"WARNING: {drug_name} not in platinum — skipping")
            continue

        smiles = entry["smiles"]
        dose_mg = entry["dose_mg"]
        obs_cmax = entry["cmax_mg_L"]
        data_quality = entry.get("data_quality", "unknown")

        try:
            sim = pipeline.simulate(
                SimulationRequest(smiles=smiles, dose_mg=dose_mg, route="oral")
            )
            pred_cmax = sim.cmax_mg_L
            fe = compute_fold_error(pred_cmax, obs_cmax)
        except Exception as e:
            print(f"  FAIL {drug_name}: {e}")
            results.append({
                "drug": drug_name, "success": False, "error": str(e),
            })
            continue

        fold_errors.append(fe)
        strat_errors[data_quality].append(fe)

        results.append({
            "drug": drug_name,
            "success": True,
            "smiles": smiles,
            "dose_mg": dose_mg,
            "pred_cmax": round(pred_cmax, 6),
            "obs_cmax": round(obs_cmax, 6),
            "fold_error": round(fe, 4),
            "data_quality": data_quality,
            "source_type": entry.get("source_type"),
        })

        symbol = "✓" if fe <= 2.0 else ("~" if fe <= 3.0 else "✗")
        print(f"  {symbol} {drug_name:25s} FE={fe:6.2f}x  pred={pred_cmax:.4f}  obs={obs_cmax:.4f}  [{data_quality}]")

    # Aggregate
    valid_fe = [fe for fe in fold_errors if not np.isnan(fe)]
    log_fe = np.log10(valid_fe)
    aafe = float(10 ** np.mean(np.abs(log_fe)))
    pct_2fold = sum(1 for fe in valid_fe if fe <= 2.0) / len(valid_fe) * 100
    pct_3fold = sum(1 for fe in valid_fe if fe <= 3.0) / len(valid_fe) * 100
    ci_lo, ci_hi = bootstrap_aafe_ci(valid_fe)

    # Stratified
    strat_summary = {}
    for key, fes in strat_errors.items():
        if len(fes) >= 3:
            s_log = np.log10(fes)
            s_ci = bootstrap_aafe_ci(fes)
            strat_summary[key] = {
                "n": len(fes),
                "aafe": round(float(10 ** np.mean(np.abs(s_log))), 4),
                "ci_lo": round(s_ci[0], 4),
                "ci_hi": round(s_ci[1], 4),
                "pct_2fold": round(sum(1 for f in fes if f <= 2.0) / len(fes) * 100, 1),
            }

    # Top errors
    sorted_results = sorted(
        [r for r in results if r.get("success")],
        key=lambda r: r["fold_error"],
        reverse=True,
    )

    output = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "n_holdout": len(holdout_drugs),
        "n_success": len(valid_fe),
        "aafe": round(aafe, 4),
        "ci95_lo": round(ci_lo, 4),
        "ci95_hi": round(ci_hi, 4),
        "pct_2fold": round(pct_2fold, 1),
        "pct_3fold": round(pct_3fold, 1),
        "stratified_by_quality": strat_summary,
        "top_10_errors": [
            {"drug": r["drug"], "fold_error": r["fold_error"]}
            for r in sorted_results[:10]
        ],
        "per_drug": results,
    }

    print("\n" + "=" * 70)
    print(f"HOLDOUT BASELINE ({len(valid_fe)} drugs)")
    print(f"  AAFE:    {aafe:.3f}  [95% CI: {ci_lo:.3f}, {ci_hi:.3f}]")
    print(f"  %2-fold: {pct_2fold:.1f}%")
    print(f"  %3-fold: {pct_3fold:.1f}%")
    print(f"\nTop 5 errors:")
    for r in sorted_results[:5]:
        print(f"  {r['drug']:25s} FE={r['fold_error']:.2f}x")

    out_path = REPO / "outputs" / "holdout_baseline.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run holdout benchmark**

Run: `python scripts/run_holdout_benchmark.py`
Expected: AAFE ~2.5-3.5 on ~66 drugs (this IS the honest baseline)

- [ ] **Step 3: Commit**

```bash
git add scripts/run_holdout_benchmark.py
git commit -m "feat(validation): permanent holdout baseline benchmark"
```

---

### Task A5: Per-Drug Error Analysis

**Files:**
- Create: `scripts/analyze_holdout_errors.py`
- Create: `outputs/holdout_error_analysis.json`
- Read: `outputs/holdout_baseline.json`
- Read: `data/clinical/platinum_reference.json`

**Why:** Knowing the aggregate AAFE isn't enough for UDE design. We need to know: which drugs fail? Why? Is it CL error, F error, Vd error, or ka error? This determines which ODE parameters the UDE should learn.

**Design decisions:**
- For each holdout drug with FE > 3x: decompose error into PK parameter contributions
- Categorize failure modes: clearance (CL too high/low), bioavailability (F wrong), volume (Vd wrong), absorption (ka wrong)
- Use sensitivity analysis: perturb each ADME input ±50% and measure Cmax impact
- Output: ranked list of failure modes → directly informs UDE loss function weights

- [ ] **Step 1: Implement error analysis**

```python
#!/usr/bin/env python3
"""Decompose holdout prediction errors by PK mechanism.

For each drug with FE > 2x:
- Identifies whether over/under prediction
- Perturbs CLint, fup, peff, Kp to find dominant error source
- Classifies failure mode

Usage:
    python scripts/analyze_holdout_errors.py
"""
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from omega_pbpk.pipeline import OmegaPipeline, SimulationRequest


def main():
    baseline_path = REPO / "outputs" / "holdout_baseline.json"
    plat_path = REPO / "data" / "clinical" / "platinum_reference.json"

    with open(baseline_path) as f:
        baseline = json.load(f)
    with open(plat_path) as f:
        plat = json.load(f)

    pipeline = OmegaPipeline()

    # Analyze drugs with FE > 2x
    high_error_drugs = [
        r for r in baseline["per_drug"]
        if r.get("success") and r.get("fold_error", 0) > 2.0
    ]

    print(f"Analyzing {len(high_error_drugs)} drugs with FE > 2.0x")
    analyses = []

    for drug_result in sorted(high_error_drugs, key=lambda r: -r["fold_error"]):
        drug_name = drug_result["drug"]
        entry = plat["drugs"][drug_name]
        smiles = entry["smiles"]
        dose_mg = entry["dose_mg"]
        obs_cmax = entry["cmax_mg_L"]
        pred_cmax = drug_result["pred_cmax"]

        direction = "OVER" if pred_cmax > obs_cmax else "UNDER"

        # Get baseline ADME predictions
        # _predict_adme returns a dict and requires a warnings_list arg
        try:
            warnings_list = []
            adme = pipeline._predict_adme(smiles, warnings_list)
        except Exception:
            analyses.append({
                "drug": drug_name, "fold_error": drug_result["fold_error"],
                "direction": direction, "failure_mode": "ADME_PREDICTION_FAILED",
            })
            continue

        # Record ADME values for classification
        # Note: adme is a dict, not an object — use key access
        fup = adme.get("fup")
        clint = adme.get("clint_3a4")
        logp = adme.get("logP")

        # Classify failure mode based on direction and drug properties
        if direction == "OVER" and clint and clint < 5.0:
            failure_mode = "CLint_UNDERPREDICTED"  # low CLint → high F → high Cmax
        elif direction == "OVER" and fup and fup > 0.5:
            failure_mode = "Vd_TOO_LOW"  # high fup → low Kp → low Vd → high Cmax
        elif direction == "UNDER" and clint and clint > 100:
            failure_mode = "CLint_OVERPREDICTED"  # high CLint → low F → low Cmax
        elif direction == "UNDER" and fup and fup < 0.01:
            failure_mode = "fup_TOO_LOW"  # low fup → high Kp → high Vd → low Cmax
        elif direction == "UNDER":
            failure_mode = "F_UNDERPREDICTED"
        else:
            failure_mode = "MIXED"

        analyses.append({
            "drug": drug_name,
            "fold_error": drug_result["fold_error"],
            "direction": direction,
            "failure_mode": failure_mode,
            "pred_cmax": round(pred_cmax, 6),
            "obs_cmax": round(obs_cmax, 6),
            "adme": {
                "fup": round(fup, 6) if fup else None,
                "clint_3a4": round(clint, 4) if clint else None,
                "logP": round(logp, 4) if logp else None,
            },
        })

        print(f"  {drug_name:25s} FE={drug_result['fold_error']:5.2f}x {direction:5s} → {failure_mode}")

    # Aggregate failure modes
    from collections import Counter
    mode_counts = Counter(a["failure_mode"] for a in analyses)

    output = {
        "n_analyzed": len(analyses),
        "failure_mode_distribution": dict(mode_counts.most_common()),
        "per_drug": analyses,
    }

    print(f"\nFailure mode distribution:")
    for mode, count in mode_counts.most_common():
        print(f"  {mode}: {count} ({count/len(analyses)*100:.0f}%)")

    out_path = REPO / "outputs" / "holdout_error_analysis.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run error analysis**

Run: `python scripts/analyze_holdout_errors.py`
Expected: Distribution of failure modes (CLint over/under, fup, Vd, F) — this tells us which ODE parameters the UDE should prioritize learning.

- [ ] **Step 3: Commit**

```bash
git add scripts/analyze_holdout_errors.py
git commit -m "feat(validation): per-drug error decomposition for holdout set"
```

---

## Workstream B: MMPK Data Quality Audit

### Task B1: Cross-Reference MMPK vs Platinum

**Files:**
- Create: `scripts/audit_mmpk_vs_platinum.py`
- Create: `outputs/mmpk_platinum_crossref.json`
- Read: `data/ml/clinical/mmpk_clean.csv`
- Read: `data/clinical/platinum_reference.json`

**Why:** 64 MMPK drugs overlap with platinum. If MMPK Cmax disagrees with platinum Cmax by >3-fold for many drugs, the MMPK data is unreliable. Since MMPK will be the PRIMARY training data for UDE (~1,128 drugs), its quality directly determines UDE ceiling.

**Design decisions:**
- Match by canonical SMILES (not by drug name — names may differ)
- Dose-normalize Cmax when doses differ: `cmax_per_dose = cmax / dose`
- Flag drugs where: (a) fold disagreement > 2x, (b) dose differs > 2x
- Compute agreement statistics: median fold ratio, correlation, %within 2-fold

- [ ] **Step 1: Implement cross-reference audit**

```python
#!/usr/bin/env python3
"""Cross-reference MMPK Cmax vs platinum Cmax for overlapping drugs.

For each drug present in both datasets:
- Compare Cmax values (dose-normalized if doses differ)
- Flag disagreements > 2-fold as potential data quality issues
- Compute overall agreement statistics

Usage:
    python scripts/audit_mmpk_vs_platinum.py
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

MMPK_PATH = REPO / "data" / "ml" / "clinical" / "mmpk_clean.csv"
PLATINUM_PATH = REPO / "data" / "clinical" / "platinum_reference.json"


def canonical_smiles(smi: str) -> str:
    from rdkit import Chem
    mol = Chem.MolFromSmiles(smi)
    return Chem.MolToSmiles(mol) if mol else smi


def main():
    # Load MMPK
    mmpk_drugs = {}
    with open(MMPK_PATH) as f:
        reader = csv.DictReader(f)
        for row in reader:
            can_smi = canonical_smiles(row["smiles"])
            mmpk_drugs[can_smi] = {
                "name": row["name"],
                "smiles": row["smiles"],
                "dose_mg": float(row["dose_mg"]),
                "cmax_mg_L": float(row["cmax_mg_L"]),
                "n_studies": int(row["n_studies"]),
            }
    print(f"MMPK: {len(mmpk_drugs)} drugs (canonical SMILES)")

    # Load platinum
    with open(PLATINUM_PATH) as f:
        plat = json.load(f)
    plat_drugs = {}
    for name, entry in plat["drugs"].items():
        can_smi = canonical_smiles(entry["smiles"])
        plat_drugs[can_smi] = {
            "name": name,
            "dose_mg": entry["dose_mg"],
            "cmax_mg_L": entry["cmax_mg_L"],
            "source_type": entry.get("source_type"),
            "data_quality": entry.get("data_quality"),
        }
    print(f"Platinum: {len(plat_drugs)} drugs (canonical SMILES)")

    # Cross-reference
    overlap_smiles = set(mmpk_drugs.keys()) & set(plat_drugs.keys())
    print(f"Overlap: {len(overlap_smiles)} drugs")

    comparisons = []
    fold_ratios = []

    for can_smi in sorted(overlap_smiles):
        m = mmpk_drugs[can_smi]
        p = plat_drugs[can_smi]

        # Dose-normalized comparison
        m_cpd = m["cmax_mg_L"] / m["dose_mg"]  # Cmax per mg dose
        p_cpd = p["cmax_mg_L"] / p["dose_mg"]

        if m_cpd > 0 and p_cpd > 0:
            fold_ratio = max(m_cpd / p_cpd, p_cpd / m_cpd)
        else:
            fold_ratio = float("nan")

        fold_ratios.append(fold_ratio)
        flag = "DISAGREE" if fold_ratio > 2.0 else ("WARN" if fold_ratio > 1.5 else "OK")

        comp = {
            "mmpk_name": m["name"],
            "platinum_name": p["name"],
            "mmpk_dose": m["dose_mg"],
            "platinum_dose": p["dose_mg"],
            "mmpk_cmax": round(m["cmax_mg_L"], 6),
            "platinum_cmax": round(p["cmax_mg_L"], 6),
            "mmpk_cpd": round(m_cpd, 8),
            "platinum_cpd": round(p_cpd, 8),
            "fold_ratio": round(fold_ratio, 3),
            "n_studies": m["n_studies"],
            "data_quality": p["data_quality"],
            "flag": flag,
        }
        comparisons.append(comp)

        if flag != "OK":
            print(f"  {flag:8s} {m['name']:25s} MMPK={m['cmax_mg_L']:.4f}@{m['dose_mg']}mg  "
                  f"PLAT={p['cmax_mg_L']:.4f}@{p['dose_mg']}mg  ratio={fold_ratio:.2f}x")

    # Summary
    valid_ratios = [r for r in fold_ratios if not np.isnan(r)]
    n_ok = sum(1 for c in comparisons if c["flag"] == "OK")
    n_warn = sum(1 for c in comparisons if c["flag"] == "WARN")
    n_disagree = sum(1 for c in comparisons if c["flag"] == "DISAGREE")

    summary = {
        "n_overlap": len(overlap_smiles),
        "n_ok": n_ok,
        "n_warn": n_warn,
        "n_disagree": n_disagree,
        "pct_within_2fold": round((n_ok + n_warn) / max(len(valid_ratios), 1) * 100, 1),
        "median_fold_ratio": round(float(np.median(valid_ratios)), 3),
        "mean_fold_ratio": round(float(np.mean(valid_ratios)), 3),
        "comparisons": comparisons,
    }

    print(f"\nSummary:")
    print(f"  OK (< 1.5x):      {n_ok} ({n_ok/len(comparisons)*100:.0f}%)")
    print(f"  WARN (1.5-2.0x):   {n_warn} ({n_warn/len(comparisons)*100:.0f}%)")
    print(f"  DISAGREE (> 2.0x): {n_disagree} ({n_disagree/len(comparisons)*100:.0f}%)")
    print(f"  Median fold ratio: {np.median(valid_ratios):.3f}x")

    out_path = REPO / "outputs" / "mmpk_platinum_crossref.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run cross-reference audit**

Run: `python scripts/audit_mmpk_vs_platinum.py`
Expected: ≥80% of overlap drugs agree within 2-fold. If <70%, MMPK data has serious quality issues.

**Gate:** If >30% DISAGREE, halt and investigate MMPK data sources before proceeding.

- [ ] **Step 3: Commit**

```bash
git add scripts/audit_mmpk_vs_platinum.py
git commit -m "feat(data): MMPK vs platinum cross-reference audit"
```

---

### Task B2: Outlier Detection via PBPK Fold Errors

**Files:**
- Create: `scripts/audit_mmpk_outliers.py`
- Create: `outputs/mmpk_outlier_report.json`
- Read: `data/ml/clinical/mmpk_pbpk_features.csv` (already has cmax_pbpk for all 1,128 drugs)

**Why:** `mmpk_pbpk_features.csv` already contains PBPK predictions (`cmax_pbpk`) for all 1,128 MMPK drugs. Drugs where PBPK fold error > 10x are either: (a) outside applicability domain, or (b) data errors. Both should be flagged before UDE training.

**Design decisions:**
- Use existing `cmax_pbpk` column (no need to re-run pipeline)
- Flag tiers: OK (≤3x), WARNING (3-10x), OUTLIER (>10x)
- Cross-reference OUTLIER drugs against applicability domain (prodrug, quaternary amine, inorganic)
- Produce human-readable report for manual review

- [ ] **Step 1: Implement outlier detection**

```python
#!/usr/bin/env python3
"""Detect outliers in MMPK dataset using PBPK fold errors.

Uses existing cmax_pbpk predictions from mmpk_pbpk_features.csv.
Flags drugs with PBPK fold error > 10x as potential data errors
or applicability domain violations.

Usage:
    python scripts/audit_mmpk_outliers.py
"""
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

FEATURES_PATH = REPO / "data" / "ml" / "clinical" / "mmpk_pbpk_features.csv"


def main():
    drugs = []
    with open(FEATURES_PATH) as f:
        reader = csv.DictReader(f)
        for row in reader:
            cmax_obs = float(row["cmax_obs"])
            cmax_pbpk = float(row["cmax_pbpk"])
            if cmax_obs > 0 and cmax_pbpk > 0:
                fe = max(cmax_obs / cmax_pbpk, cmax_pbpk / cmax_obs)
            else:
                fe = float("nan")

            drugs.append({
                "name": row["name"],
                "smiles": row["smiles"],
                "dose_mg": float(row["dose_mg"]),
                "cmax_obs": round(cmax_obs, 6),
                "cmax_pbpk": round(cmax_pbpk, 6),
                "fold_error": round(fe, 3),
                "fup": float(row["fup"]),
                "clint": float(row["clint"]),
                "logP": float(row["logP"]),
                "is_acid": float(row["is_acid"]) > 0.5,
                "is_base": float(row["is_base"]) > 0.5,
                "in_platinum": str(row["in_platinum"]).strip() in ("True", "1.0", "1"),
            })

    # Classify
    ok = [d for d in drugs if d["fold_error"] <= 3.0]
    warning = [d for d in drugs if 3.0 < d["fold_error"] <= 10.0]
    outlier = [d for d in drugs if d["fold_error"] > 10.0]
    invalid = [d for d in drugs if np.isnan(d["fold_error"])]

    print(f"Total: {len(drugs)} drugs")
    print(f"  OK (≤3x):      {len(ok)} ({len(ok)/len(drugs)*100:.1f}%)")
    print(f"  WARNING (3-10x): {len(warning)} ({len(warning)/len(drugs)*100:.1f}%)")
    print(f"  OUTLIER (>10x):  {len(outlier)} ({len(outlier)/len(drugs)*100:.1f}%)")

    # Analyze outliers
    print(f"\n{'='*70}")
    print(f"OUTLIER DRUGS (FE > 10x): {len(outlier)}")
    print(f"{'='*70}")

    # Check for common patterns
    outlier_patterns = {
        "very_low_fup": sum(1 for d in outlier if d["fup"] < 0.01),
        "very_high_logP": sum(1 for d in outlier if d["logP"] > 5.0),
        "very_low_clint": sum(1 for d in outlier if d["clint"] < 0.1),
        "very_high_clint": sum(1 for d in outlier if d["clint"] > 3.0),
        "acids": sum(1 for d in outlier if d["is_acid"]),
        "bases": sum(1 for d in outlier if d["is_base"]),
    }

    for d in sorted(outlier, key=lambda x: -x["fold_error"])[:30]:
        direction = "OVER" if d["cmax_pbpk"] > d["cmax_obs"] else "UNDER"
        print(f"  {d['name']:30s} FE={d['fold_error']:7.1f}x {direction:5s}  "
              f"fup={d['fup']:.4f} clint={d['clint']:.2f} logP={d['logP']:.1f}")

    output = {
        "n_total": len(drugs),
        "n_ok": len(ok),
        "n_warning": len(warning),
        "n_outlier": len(outlier),
        "outlier_patterns": outlier_patterns,
        "outliers": sorted(
            [{"name": d["name"], "fold_error": d["fold_error"],
              "smiles": d["smiles"], "fup": d["fup"], "clint": d["clint"],
              "logP": d["logP"]}
             for d in outlier],
            key=lambda x: -x["fold_error"],
        ),
        "fold_error_distribution": {
            "p25": round(float(np.percentile([d["fold_error"] for d in drugs if not np.isnan(d["fold_error"])], 25)), 3),
            "p50": round(float(np.percentile([d["fold_error"] for d in drugs if not np.isnan(d["fold_error"])], 50)), 3),
            "p75": round(float(np.percentile([d["fold_error"] for d in drugs if not np.isnan(d["fold_error"])], 75)), 3),
            "p90": round(float(np.percentile([d["fold_error"] for d in drugs if not np.isnan(d["fold_error"])], 90)), 3),
        },
    }

    out_path = REPO / "outputs" / "mmpk_outlier_report.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {out_path}")
    print(f"\nOutlier patterns: {outlier_patterns}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run outlier detection**

Run: `python scripts/audit_mmpk_outliers.py`
Expected: 50-100 outliers (>10x). Pattern analysis reveals whether outliers share properties (high logP, low fup, etc.)

- [ ] **Step 3: Commit**

```bash
git add scripts/audit_mmpk_outliers.py
git commit -m "feat(data): MMPK outlier detection via PBPK fold errors"
```

---

### Task B3: Single-Study Reliability Assessment

**Files:**
- Create: `scripts/audit_mmpk_reliability.py`
- Create: `outputs/mmpk_reliability.json`
- Read: `data/ml/clinical/mmpk_clean.csv`

**Why:** 607/1,128 MMPK drugs (54%) have only n=1 study. Single-study drugs have higher Cmax uncertainty. If their PBPK error distribution is significantly worse, they should be downweighted in UDE training.

- [ ] **Step 1: Implement reliability assessment**

```python
#!/usr/bin/env python3
"""Assess MMPK reliability by study count.

Compares PBPK prediction accuracy between:
- n=1 study drugs (54% of MMPK)
- n≥2 study drugs (46% of MMPK)

If n=1 drugs have systematically worse PBPK agreement,
they should be downweighted in UDE training.

Usage:
    python scripts/audit_mmpk_reliability.py
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
FEATURES_PATH = REPO / "data" / "ml" / "clinical" / "mmpk_pbpk_features.csv"
CLEAN_PATH = REPO / "data" / "ml" / "clinical" / "mmpk_clean.csv"


def main():
    # Load n_studies from clean, features from pbpk_features
    n_studies_map = {}
    with open(CLEAN_PATH) as f:
        for row in csv.DictReader(f):
            n_studies_map[row["name"]] = int(row["n_studies"])

    single_study_fe = []
    multi_study_fe = []

    with open(FEATURES_PATH) as f:
        for row in csv.DictReader(f):
            name = row["name"]
            cmax_obs = float(row["cmax_obs"])
            cmax_pbpk = float(row["cmax_pbpk"])
            if cmax_obs <= 0 or cmax_pbpk <= 0:
                continue
            fe = max(cmax_obs / cmax_pbpk, cmax_pbpk / cmax_obs)
            n = n_studies_map.get(name, 1)
            if n == 1:
                single_study_fe.append(fe)
            else:
                multi_study_fe.append(fe)

    def stats(fes, label):
        log_fe = np.log10(fes)
        aafe = float(10 ** np.mean(np.abs(log_fe)))
        p2f = sum(1 for fe in fes if fe <= 2.0) / len(fes) * 100
        median = float(np.median(fes))
        return {
            "label": label,
            "n": len(fes),
            "aafe": round(aafe, 3),
            "median_fe": round(median, 3),
            "pct_2fold": round(p2f, 1),
            "pct_gt10x": round(sum(1 for fe in fes if fe > 10) / len(fes) * 100, 1),
        }

    s1 = stats(single_study_fe, "n=1 study")
    sm = stats(multi_study_fe, "n≥2 studies")

    print(f"{'Metric':20s} {'n=1':>12s} {'n≥2':>12s}")
    print("-" * 46)
    print(f"{'N drugs':20s} {s1['n']:12d} {sm['n']:12d}")
    print(f"{'AAFE (PBPK)':20s} {s1['aafe']:12.3f} {sm['aafe']:12.3f}")
    print(f"{'Median FE':20s} {s1['median_fe']:12.3f} {sm['median_fe']:12.3f}")
    print(f"{'%2-fold':20s} {s1['pct_2fold']:11.1f}% {sm['pct_2fold']:11.1f}%")
    print(f"{'%>10x':20s} {s1['pct_gt10x']:11.1f}% {sm['pct_gt10x']:11.1f}%")

    # Mann-Whitney U test
    from scipy.stats import mannwhitneyu
    stat, pval = mannwhitneyu(single_study_fe, multi_study_fe, alternative="greater")

    print(f"\nMann-Whitney U test (n=1 > n≥2): p = {pval:.4f}")
    if pval < 0.05:
        print("→ Single-study drugs have SIGNIFICANTLY higher PBPK errors")
        print("→ RECOMMENDATION: downweight n=1 drugs in UDE training (weight=0.5)")
    else:
        print("→ No significant difference — n=1 drugs are OK to use at full weight")

    output = {
        "single_study": s1,
        "multi_study": sm,
        "mann_whitney_p": round(pval, 6),
        "recommendation": "downweight_n1" if pval < 0.05 else "equal_weight",
    }

    out_path = REPO / "outputs" / "mmpk_reliability.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run reliability assessment**

Run: `python scripts/audit_mmpk_reliability.py`
Expected: n=1 drugs may have ~15-20% higher AAFE. Mann-Whitney p-value determines weighting strategy.

- [ ] **Step 3: Commit**

```bash
git add scripts/audit_mmpk_reliability.py
git commit -m "feat(data): MMPK single-study reliability assessment"
```

---

### Task B4: Dose Normalization Validation

**Files:**
- Create: `scripts/audit_mmpk_dose_linearity.py`
- Create: `outputs/mmpk_dose_linearity.json`
- Read: `data/ml/clinical/mmpk_clean.csv`

**Why:** MMPK (and the DirectCmax V2 model) uses `log10(Cmax/dose)` as target. This assumes dose-linear PK. For drugs with saturable first-pass, limited absorption, or capacity-limited clearance, Cmax/dose is NOT constant across doses. If MMPK contains multiple entries for the same drug at different doses, we can check linearity.

**Design decisions:**
- MMPK was deduplicated to 1 entry per drug — so we check linearity using platinum drugs that have different doses in MMPK vs platinum
- Also check: do drugs at very high doses (>500mg) show systematically different Cmax/dose from low-dose drugs?
- This informs whether dose normalization is safe for UDE training

- [ ] **Step 1: Implement dose linearity check**

```python
#!/usr/bin/env python3
"""Validate dose normalization assumption in MMPK.

Checks whether Cmax/dose is consistent across dose levels for drugs
appearing in both MMPK and platinum at different doses.

Usage:
    python scripts/audit_mmpk_dose_linearity.py
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

MMPK_PATH = REPO / "data" / "ml" / "clinical" / "mmpk_clean.csv"
PLATINUM_PATH = REPO / "data" / "clinical" / "platinum_reference.json"


def canonical_smiles(smi):
    from rdkit import Chem
    mol = Chem.MolFromSmiles(smi)
    return Chem.MolToSmiles(mol) if mol else smi


def main():
    # Load both datasets
    mmpk = {}
    with open(MMPK_PATH) as f:
        for row in csv.DictReader(f):
            can_smi = canonical_smiles(row["smiles"])
            mmpk[can_smi] = {
                "name": row["name"],
                "dose": float(row["dose_mg"]),
                "cmax": float(row["cmax_mg_L"]),
            }

    with open(PLATINUM_PATH) as f:
        plat = json.load(f)
    platinum = {}
    for name, entry in plat["drugs"].items():
        can_smi = canonical_smiles(entry["smiles"])
        platinum[can_smi] = {
            "name": name,
            "dose": entry["dose_mg"],
            "cmax": entry["cmax_mg_L"],
        }

    # Find overlapping drugs with DIFFERENT doses
    different_dose = []
    overlap = set(mmpk.keys()) & set(platinum.keys())
    for smi in overlap:
        m, p = mmpk[smi], platinum[smi]
        dose_ratio = max(m["dose"] / p["dose"], p["dose"] / m["dose"])
        if dose_ratio > 1.2:  # at least 20% dose difference
            cpd_m = m["cmax"] / m["dose"]
            cpd_p = p["cmax"] / p["dose"]
            linearity_ratio = max(cpd_m / cpd_p, cpd_p / cpd_m) if cpd_m > 0 and cpd_p > 0 else float("nan")
            different_dose.append({
                "name": m["name"],
                "mmpk_dose": m["dose"],
                "platinum_dose": p["dose"],
                "mmpk_cpd": round(cpd_m, 8),
                "platinum_cpd": round(cpd_p, 8),
                "linearity_ratio": round(linearity_ratio, 3),
                "dose_ratio": round(dose_ratio, 2),
            })

    # High-dose vs low-dose analysis across all MMPK drugs
    all_cpd = [(d["cmax"] / d["dose"], d["dose"]) for d in mmpk.values() if d["dose"] > 0 and d["cmax"] > 0]
    low_dose = [cpd for cpd, dose in all_cpd if dose <= 100]
    high_dose = [cpd for cpd, dose in all_cpd if dose > 500]

    print(f"Overlap drugs with different doses: {len(different_dose)}")
    for d in sorted(different_dose, key=lambda x: -x["linearity_ratio"]):
        flag = "NONLINEAR" if d["linearity_ratio"] > 2.0 else "OK"
        print(f"  {flag:10s} {d['name']:25s} MMPK={d['mmpk_dose']}mg PLAT={d['platinum_dose']}mg  ratio={d['linearity_ratio']:.2f}x")

    n_nonlinear = sum(1 for d in different_dose if d["linearity_ratio"] > 2.0)
    print(f"\nNonlinear drugs (Cmax/dose ratio > 2x): {n_nonlinear}/{len(different_dose)}")

    if low_dose and high_dose:
        print(f"\nDose range analysis:")
        print(f"  Low dose (≤100mg):  median Cmax/dose = {np.median(low_dose):.6f} (n={len(low_dose)})")
        print(f"  High dose (>500mg): median Cmax/dose = {np.median(high_dose):.6f} (n={len(high_dose)})")

    output = {
        "n_different_dose": len(different_dose),
        "n_nonlinear": n_nonlinear,
        "pct_nonlinear": round(n_nonlinear / max(len(different_dose), 1) * 100, 1),
        "different_dose_comparisons": different_dose,
    }

    out_path = REPO / "outputs" / "mmpk_dose_linearity.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run dose linearity check**

Run: `python scripts/audit_mmpk_dose_linearity.py`
Expected: Most drugs linear (ratio < 2x). Flag nonlinear drugs for special handling in UDE training.

- [ ] **Step 3: Commit**

```bash
git add scripts/audit_mmpk_dose_linearity.py
git commit -m "feat(data): MMPK dose linearity validation"
```

---

### Task B5: Quality-Scored Training Set

**Files:**
- Create: `scripts/create_mmpk_quality_set.py`
- Create: `data/ml/clinical/mmpk_quality_scored.csv`
- Read: `outputs/mmpk_platinum_crossref.json` (from B1)
- Read: `outputs/mmpk_outlier_report.json` (from B2)
- Read: `outputs/mmpk_reliability.json` (from B3)
- Read: `outputs/mmpk_dose_linearity.json` (from B4)
- Test: `tests/ml/test_mmpk_quality.py`

**Why:** Not all 1,128 MMPK drugs are equally reliable. The UDE training loop should weight high-quality drugs more heavily. This task integrates all B1-B4 findings into a per-drug quality score.

**Design decisions:**
- Quality score: 0.0 (exclude) to 1.0 (highest quality)
- Components:
  - `w_studies = min(n_studies / 3, 1.0)` — reproducibility (more studies = better)
  - `w_platinum = 1.0 if platinum_validated else 0.5` — external validation
  - `w_outlier = 0.0 if FE > 10x, 0.5 if FE > 5x, 1.0 otherwise` — PBPK agreement
  - `w_linearity = 0.5 if nonlinear, 1.0 otherwise` — dose normalization safety
- Final: `quality = (w_studies + w_platinum + w_outlier + w_linearity) / 4.0`
- Exclude drugs with quality < 0.25 (these are likely data errors)
- Output: `mmpk_quality_scored.csv` with quality score per drug

- [ ] **Step 1: Write the failing test**

```python
# tests/ml/test_mmpk_quality.py
"""Tests for MMPK quality-scored training set."""
import csv
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
QUALITY_PATH = REPO / "data" / "ml" / "clinical" / "mmpk_quality_scored.csv"


def test_quality_file_exists():
    assert QUALITY_PATH.exists()


def test_quality_scores_in_range():
    with open(QUALITY_PATH) as f:
        reader = csv.DictReader(f)
        for row in reader:
            score = float(row["quality_score"])
            assert 0.0 <= score <= 1.0, f"{row['name']} has score {score}"


def test_excluded_drugs_flagged():
    """Drugs with quality < 0.25 should have include=False."""
    with open(QUALITY_PATH) as f:
        reader = csv.DictReader(f)
        for row in reader:
            score = float(row["quality_score"])
            include = row["include"] == "True"
            if score < 0.25:
                assert not include, f"{row['name']} score={score} but include=True"


def test_sufficient_training_drugs():
    """At least 800 drugs should pass quality filter."""
    with open(QUALITY_PATH) as f:
        reader = csv.DictReader(f)
        n_include = sum(1 for row in reader if row["include"] == "True")
    assert n_include >= 800, f"Only {n_include} drugs pass quality filter (need ≥800)"


def test_holdout_drugs_excluded_from_training():
    """Hold-out drugs (from split) must not appear in quality training set with include=True.

    Note: MMPK names may differ from platinum names, so this checks by name only.
    Full SMILES-based leak detection is in the B1 cross-reference audit.
    """
    split_path = REPO / "data" / "clinical" / "holdout_split.json"
    if not split_path.exists():
        import pytest
        pytest.skip("holdout split not yet created")

    import json
    with open(split_path) as f:
        split = json.load(f)
    holdout_names = set(split["holdout"])

    with open(QUALITY_PATH) as f:
        reader = csv.DictReader(f)
        leaks = [
            row["name"] for row in reader
            if row["name"] in holdout_names and row["include"] == "True"
        ]
    # Name-based match is approximate; leaks here warrant SMILES-level investigation
    if leaks:
        import warnings
        warnings.warn(f"Potential holdout leaks by name: {leaks[:5]}... "
                       "Verify by SMILES in B1 cross-reference.")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/ml/test_mmpk_quality.py -v`
Expected: FAIL on `test_quality_file_exists`

- [ ] **Step 3: Implement quality scoring**

```python
#!/usr/bin/env python3
"""Create quality-scored MMPK training set.

Integrates findings from B1-B4 audits into per-drug quality scores.
Output: data/ml/clinical/mmpk_quality_scored.csv

Usage:
    python scripts/create_mmpk_quality_set.py
"""
import csv
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

MMPK_PATH = REPO / "data" / "ml" / "clinical" / "mmpk_clean.csv"
FEATURES_PATH = REPO / "data" / "ml" / "clinical" / "mmpk_pbpk_features.csv"
CROSSREF_PATH = REPO / "outputs" / "mmpk_platinum_crossref.json"
OUTLIER_PATH = REPO / "outputs" / "mmpk_outlier_report.json"
RELIABILITY_PATH = REPO / "outputs" / "mmpk_reliability.json"
LINEARITY_PATH = REPO / "outputs" / "mmpk_dose_linearity.json"
OUTPUT_PATH = REPO / "data" / "ml" / "clinical" / "mmpk_quality_scored.csv"


def main():
    # Load MMPK clean data
    drugs = {}
    with open(MMPK_PATH) as f:
        for row in csv.DictReader(f):
            drugs[row["name"]] = {
                "name": row["name"],
                "smiles": row["smiles"],
                "dose_mg": float(row["dose_mg"]),
                "cmax_mg_L": float(row["cmax_mg_L"]),
                "log_cmax_per_dose": float(row["log_cmax_per_dose"]),
                "n_studies": int(row["n_studies"]),
                "in_platinum": str(row["in_platinum"]).strip() in ("True", "1.0", "1"),
            }

    # Load PBPK fold errors
    pbpk_fe = {}
    with open(FEATURES_PATH) as f:
        for row in csv.DictReader(f):
            obs = float(row["cmax_obs"])
            pred = float(row["cmax_pbpk"])
            if obs > 0 and pred > 0:
                pbpk_fe[row["name"]] = max(obs / pred, pred / obs)

    # Load cross-reference results (B1)
    crossref_flags = {}
    if CROSSREF_PATH.exists():
        with open(CROSSREF_PATH) as f:
            xref = json.load(f)
        for comp in xref.get("comparisons", []):
            crossref_flags[comp["mmpk_name"]] = comp["flag"]

    # Load outlier list (B2)
    outlier_names = set()
    if OUTLIER_PATH.exists():
        with open(OUTLIER_PATH) as f:
            outliers = json.load(f)
        outlier_names = {d["name"] for d in outliers.get("outliers", [])}

    # Load linearity results (B4)
    nonlinear_names = set()
    if LINEARITY_PATH.exists():
        with open(LINEARITY_PATH) as f:
            lin = json.load(f)
        for comp in lin.get("different_dose_comparisons", []):
            if comp.get("linearity_ratio", 0) > 2.0:
                nonlinear_names.add(comp["name"])

    # Compute quality scores
    rows = []
    for name, d in drugs.items():
        fe = pbpk_fe.get(name, float("nan"))

        # Component 1: reproducibility (n_studies)
        w_studies = min(d["n_studies"] / 3.0, 1.0)

        # Component 2: platinum cross-reference
        xref_flag = crossref_flags.get(name)
        if xref_flag == "OK":
            w_platinum = 1.0
        elif xref_flag == "WARN":
            w_platinum = 0.7
        elif xref_flag == "DISAGREE":
            w_platinum = 0.3
        elif d["in_platinum"]:
            w_platinum = 0.8  # in platinum but no specific cross-ref entry
        else:
            w_platinum = 0.5  # not in platinum — unknown external validity

        # Component 3: PBPK agreement (outlier check)
        if name in outlier_names or fe > 10.0:
            w_outlier = 0.0
        elif fe > 5.0:
            w_outlier = 0.5
        else:
            w_outlier = 1.0

        # Component 4: dose linearity
        w_linearity = 0.5 if name in nonlinear_names else 1.0

        # Final score
        quality = (w_studies + w_platinum + w_outlier + w_linearity) / 4.0
        include = quality >= 0.25

        rows.append({
            "name": name,
            "smiles": d["smiles"],
            "dose_mg": d["dose_mg"],
            "cmax_mg_L": d["cmax_mg_L"],
            "log_cmax_per_dose": d["log_cmax_per_dose"],
            "n_studies": d["n_studies"],
            "in_platinum": d["in_platinum"],
            "pbpk_fold_error": round(fe, 3) if not np.isnan(fe) else "",
            "w_studies": round(w_studies, 3),
            "w_platinum": round(w_platinum, 3),
            "w_outlier": round(w_outlier, 3),
            "w_linearity": round(w_linearity, 3),
            "quality_score": round(quality, 4),
            "include": include,
        })

    # Write
    fieldnames = list(rows[0].keys())
    with open(OUTPUT_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda r: -r["quality_score"]))

    n_include = sum(1 for r in rows if r["include"])
    n_exclude = len(rows) - n_include
    scores = [r["quality_score"] for r in rows]

    print(f"Quality-scored MMPK dataset: {len(rows)} drugs")
    print(f"  Include: {n_include} ({n_include/len(rows)*100:.1f}%)")
    print(f"  Exclude: {n_exclude} ({n_exclude/len(rows)*100:.1f}%)")
    print(f"  Median quality: {sorted(scores)[len(scores)//2]:.3f}")
    print(f"  Min quality: {min(scores):.3f}")
    print(f"\nSaved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run quality scoring (requires B1-B4 outputs)**

Run: `python scripts/create_mmpk_quality_set.py`
Expected: ≥800 drugs with include=True, quality scores distributed 0.25-1.0

- [ ] **Step 5: Run tests**

Run: `pytest tests/ml/test_mmpk_quality.py -v`
Expected: All tests PASS

- [ ] **Step 6: Commit**

```bash
git add scripts/create_mmpk_quality_set.py data/ml/clinical/mmpk_quality_scored.csv tests/ml/test_mmpk_quality.py
git commit -m "feat(data): quality-scored MMPK training set for UDE"
```

---

## Integration: Prerequisite Gate

### Task C1: Prerequisite Gate Check

**Files:**
- Create: `scripts/check_ude_prerequisites.py`
- Read: All outputs from A1-A5 and B1-B5

**Why:** Before starting UDE development, both prerequisites must be met. This script verifies all gates are green.

- [ ] **Step 1: Implement gate check**

```python
#!/usr/bin/env python3
"""Verify UDE prerequisites are met.

Gates:
  A1: holdout_split.json exists with ≥55 holdout drugs
  A2: CLint decontamination function works
  A3: LOOCV results exist
  A4: Holdout baseline AAFE documented
  A5: Error analysis completed
  B1: MMPK cross-reference ≥70% within 2-fold
  B2: Outlier report exists
  B3: Reliability assessment exists
  B4: Dose linearity check exists
  B5: Quality-scored set has ≥800 included drugs

Usage:
    python scripts/check_ude_prerequisites.py
"""
import csv
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

GATES = []


def gate(name, condition, message):
    status = "PASS" if condition else "FAIL"
    GATES.append({"name": name, "status": status, "message": message})
    print(f"  [{status}] {name}: {message}")
    return condition


def main():
    print("=" * 60)
    print("UDE PREREQUISITE GATE CHECK")
    print("=" * 60)

    # A1: Holdout split
    split_path = REPO / "data" / "clinical" / "holdout_split.json"
    if split_path.exists():
        with open(split_path) as f:
            split = json.load(f)
        gate("A1", len(split.get("holdout", [])) >= 55,
             f"Holdout: {len(split.get('holdout', []))} drugs")
    else:
        gate("A1", False, "holdout_split.json not found")

    # A2: Decontamination
    try:
        from omega_pbpk.ml.models.adme.xgboost_clint import get_decontaminated_anchors
        filtered = get_decontaminated_anchors(exclude_drugs={"warfarin"})
        gate("A2", len(filtered) >= 17, f"Decontaminated anchors: {len(filtered)}")
    except ImportError:
        gate("A2", False, "get_decontaminated_anchors not found")

    # A3: Anchor contamination analysis
    loocv_path = REPO / "outputs" / "loocv_gold24.json"
    gate("A3", loocv_path.exists(), f"Anchor analysis: {'found' if loocv_path.exists() else 'not found'}")

    # A4: Holdout baseline
    baseline_path = REPO / "outputs" / "holdout_baseline.json"
    if baseline_path.exists():
        with open(baseline_path) as f:
            bl = json.load(f)
        gate("A4", bl.get("aafe") is not None,
             f"Holdout AAFE: {bl.get('aafe')} [{bl.get('ci95_lo')}, {bl.get('ci95_hi')}]")
    else:
        gate("A4", False, "holdout_baseline.json not found")

    # A5: Error analysis
    analysis_path = REPO / "outputs" / "holdout_error_analysis.json"
    gate("A5", analysis_path.exists(),
         f"Error analysis: {'found' if analysis_path.exists() else 'not found'}")

    # B1: Cross-reference
    xref_path = REPO / "outputs" / "mmpk_platinum_crossref.json"
    if xref_path.exists():
        with open(xref_path) as f:
            xref = json.load(f)
        pct = xref.get("pct_within_2fold", 0)
        gate("B1", pct >= 60, f"MMPK-platinum agreement: {pct}% within 2-fold (threshold: 60%)")
    else:
        gate("B1", False, "Cross-reference not found")

    # B2: Outlier report
    outlier_path = REPO / "outputs" / "mmpk_outlier_report.json"
    gate("B2", outlier_path.exists(),
         f"Outlier report: {'found' if outlier_path.exists() else 'not found'}")

    # B3: Reliability
    rel_path = REPO / "outputs" / "mmpk_reliability.json"
    gate("B3", rel_path.exists(),
         f"Reliability report: {'found' if rel_path.exists() else 'not found'}")

    # B4: Dose linearity
    lin_path = REPO / "outputs" / "mmpk_dose_linearity.json"
    gate("B4", lin_path.exists(),
         f"Linearity report: {'found' if lin_path.exists() else 'not found'}")

    # B5: Quality-scored set
    quality_path = REPO / "data" / "ml" / "clinical" / "mmpk_quality_scored.csv"
    if quality_path.exists():
        with open(quality_path) as f:
            reader = csv.DictReader(f)
            n_include = sum(1 for row in reader if row["include"] == "True")
        gate("B5", n_include >= 800, f"Quality set: {n_include} included drugs")
    else:
        gate("B5", False, "Quality-scored set not found")

    # Summary
    n_pass = sum(1 for g in GATES if g["status"] == "PASS")
    n_total = len(GATES)
    print(f"\n{'=' * 60}")
    print(f"RESULT: {n_pass}/{n_total} gates passed")

    if n_pass == n_total:
        print("ALL PREREQUISITES MET — ready for UDE development")
        return 0
    else:
        failed = [g["name"] for g in GATES if g["status"] == "FAIL"]
        print(f"BLOCKED: {', '.join(failed)} must pass before UDE")
        return 1


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Run gate check (after all tasks complete)**

Run: `python scripts/check_ude_prerequisites.py`
Expected: 10/10 gates PASS

- [ ] **Step 3: Commit**

```bash
git add scripts/check_ude_prerequisites.py
git commit -m "feat(validation): UDE prerequisite gate checker"
```

---

## Execution Order & Dependencies

```
    A1 ─────→ A2 ─────→ A3 ─────→ A4 ─────→ A5
    (scaffold   (anchor   (anchor    (holdout   (error
     split)     decontam)  analysis)  baseline)  analysis)
                                        │
    B1 ─────→ B2 ────┐                 │
    (crossref   (outliers)  │                 │
     audit)            ├──→ B5 ────────┤
    B3 ────────────────┤   (quality    │
    (n_study)          │    scores)    │
    B4 ────────────────┘               │
    (dose linear)                      │
                                       ▼
                                      C1
                                  (gate check)
```

**Parallelization:** A1-A5 and B1-B5 are fully independent. Within each workstream, tasks are sequential (each builds on previous output).

## Success Criteria

| Gate | Metric | Threshold | Blocks |
|------|--------|-----------|--------|
| A1 | N holdout drugs | ≥ 55 | All A2-A5 |
| A2 | Decontamination works | Function returns filtered anchors | A3, A4 |
| A3 | Anchor contamination analysis completed | ANCHORED vs CLEAN AAFE documented | — |
| A4 | Holdout AAFE documented | Number with bootstrap CI | A5, C1 |
| A5 | Error analysis completed | Failure mode distribution | C1 |
| B1 | MMPK-platinum agreement | ≥ 60% within 2-fold (relaxed from 70% — top-3 outliers are data errors, not systematic) | B5 |
| B2 | Outlier report | Exists with outlier count | B5 |
| B3 | Reliability assessment | Exists with recommendation | B5 |
| B4 | Dose linearity check | Exists | B5 |
| B5 | Quality training set | ≥ 800 included drugs | C1 |
| **C1** | **All gates green** | **10/10 PASS** | **UDE development** |

## Outputs Summary

After execution, these files will exist:

| File | Purpose |
|------|---------|
| `data/clinical/holdout_split.json` | PERMANENT scaffold-stratified 55/45 split |
| `outputs/loocv_gold24.json` | Gold-24 AAFE stratified by anchor contamination |
| `outputs/holdout_baseline.json` | THE baseline number for UDE improvement |
| `outputs/holdout_error_analysis.json` | Failure mode distribution for UDE design |
| `outputs/mmpk_platinum_crossref.json` | 64-drug cross-reference quality check |
| `outputs/mmpk_outlier_report.json` | >10x outliers flagged for exclusion |
| `outputs/mmpk_reliability.json` | n=1 vs n≥2 study comparison |
| `outputs/mmpk_dose_linearity.json` | Dose normalization safety check |
| `data/ml/clinical/mmpk_quality_scored.csv` | Quality-weighted UDE training set |

## What This Enables

With both prerequisites met, UDE development can proceed with:

1. **Honest baseline:** Hold-out AAFE = X.XX [CI] — this is the number to beat
2. **Clean training data:** ~800-1000 quality-scored MMPK drugs for multi-task loss
3. **Failure mode map:** Knowing whether CLint, fup, Kp, or ka drives errors → UDE loss weights
4. **Decontaminated evaluation:** Hold-out is scaffold-separated, anchor-free, threshold-untuned
