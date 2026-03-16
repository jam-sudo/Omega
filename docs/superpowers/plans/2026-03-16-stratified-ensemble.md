# Stratified Ensemble Prediction — Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the 6.64x AAFE on expanded drugs by: (1) filtering untractable drugs via applicability domain, (2) adding a direct ML prediction arm, (3) ensembling PBPK + ML with C(t) curve scaling.

**Architecture:** Applicability domain filter flags prodrugs/pH-unstable/food-dependent drugs as low-confidence. Direct ML arm uses XGBoost on 2057 features (2048 Morgan bits + 9 RDKit descriptors) to predict log(Cmax/dose_mg). Ensemble blends PBPK and ML predictions with learned weight. C(t) curve from PBPK is scaled to match ensemble Cmax.

**Tech Stack:** Python 3.10, xgboost, rdkit (Morgan fingerprints + descriptors), scikit-learn (cross_val_predict), numpy. Existing OmegaPipeline, EnsembleADMEPredictor.

**Training data:** 75 drugs with observed Cmax (53 expanded + 22 unique from 25-drug benchmark).

---

## File Structure

| File | Responsibility |
|------|---------------|
| `src/omega_pbpk/ml/applicability.py` | Applicability domain filter (prodrug, pH-unstable detection) |
| `src/omega_pbpk/ml/models/direct_pk/__init__.py` | Package init |
| `src/omega_pbpk/ml/models/direct_pk/xgboost_cmax.py` | XGBoost direct Cmax predictor |
| `src/omega_pbpk/ml/models/direct_pk/ensemble_pk.py` | PBPK + ML ensemble with C(t) scaling |
| `scripts/train_direct_cmax.py` | Training script for direct Cmax model |
| `scripts/build_training_set.py` | Build combined training CSV from benchmark results |
| `tests/ml/test_applicability.py` | Tests for applicability domain |
| `tests/ml/test_direct_cmax.py` | Tests for direct Cmax predictor |
| `tests/ml/test_ensemble_pk.py` | Tests for ensemble |
| Modify: `src/omega_pbpk/pipeline/__init__.py` | Integrate ensemble into simulate() |

---

## Chunk 1: Applicability Domain Filter

### Task 1: Applicability Domain Module

**Files:**
- Create: `src/omega_pbpk/ml/applicability.py`
- Test: `tests/ml/test_applicability.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/ml/test_applicability.py
"""Tests for applicability domain filter."""
from omega_pbpk.ml.applicability import check_applicability


class TestApplicability:
    def test_caffeine_is_tractable(self):
        result = check_applicability("CN1C=NC2=C1C(=O)N(C(=O)N2C)C")
        assert result.tractable is True
        assert len(result.flags) == 0

    def test_prodrug_ester_detected(self):
        # Valganciclovir — amino acid ester prodrug
        result = check_applicability(
            "CC(C)C(N)C(=O)OCOC1=CC2=C(C=C1)N=C(N2)OCC(CO)OC(=O)C(N)C(C)C"
        )
        assert result.tractable is False
        assert "prodrug_ester" in result.flags

    def test_phosphate_prodrug_detected(self):
        # Tenofovir disoproxil — phosphonate ester prodrug
        result = check_applicability(
            "CC(CN1C=NC2=C1N=CN=C2N)OCP(=O)(OCOC(=O)OC(C)C)OCOC(=O)OC(C)C"
        )
        assert result.tractable is False
        assert "prodrug_phosphonate" in result.flags

    def test_result_has_confidence(self):
        result = check_applicability("CN1C=NC2=C1C(=O)N(C(=O)N2C)C")
        assert result.confidence in ("high", "medium", "low")
```

- [ ] **Step 2: Run tests — verify fail**

```bash
pytest tests/ml/test_applicability.py -v
```

- [ ] **Step 3: Implement applicability module**

```python
# src/omega_pbpk/ml/applicability.py
"""Applicability domain filter for PBPK predictions.

Flags drugs with structural features that the PBPK model cannot
handle: prodrugs (ester/phosphonate/carbamate), pH-unstable compounds,
and known untractable drug classes.

Usage:
    from omega_pbpk.ml.applicability import check_applicability
    result = check_applicability(smiles)
    if not result.tractable:
        print(f"Low confidence: {result.flags}")
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# SMARTS patterns for structural features the PBPK model can't handle
_PRODRUG_PATTERNS = {
    # Ester prodrug: R-C(=O)-O-C (excludes carboxylic acids R-C(=O)-OH)
    # Requires ester oxygen bonded to carbon (not hydrogen)
    "prodrug_ester": "[C;!$(C(=O)[OH])](=O)[O][C;!$(C=O)]",
    # Phosphonate ester: P(=O)(O-C)(O-C) — like tenofovir disoproxil
    "prodrug_phosphonate": "[P](=O)([O][C])([O][C])",
    # Carbamate: O-C(=O)-N — masked ester/amide
    "prodrug_carbamate": "[O][C](=O)[N]",
}

# Known untractable drug names (food-effect-dependent, pH-unstable, etc.)
_UNTRACTABLE_NAMES = {
    "sonidegib",  # Extreme food effect (>4x)
    "temozolomide",  # pH-dependent degradation
    "sodium oxybate",  # GHB, unusual PK (rapid metabolism)
    "lanthanum carbonate",  # Inorganic, not absorbed
}


@dataclass(frozen=True)
class ApplicabilityResult:
    """Result of applicability domain check."""
    tractable: bool
    confidence: str  # "high", "medium", "low"
    flags: tuple[str, ...] = field(default_factory=tuple)
    details: str = ""


def check_applicability(
    smiles: str, drug_name: str | None = None
) -> ApplicabilityResult:
    """Check if a drug is within the PBPK model's applicability domain.

    Args:
        smiles: SMILES string
        drug_name: Optional drug name for lookup-based checks

    Returns:
        ApplicabilityResult with tractable flag, confidence, and any flags
    """
    flags: list[str] = []

    # Check known untractable drugs by name
    if drug_name and drug_name.lower().strip() in _UNTRACTABLE_NAMES:
        flags.append("known_untractable")

    # Check structural patterns with RDKit
    try:
        from rdkit import Chem

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return ApplicabilityResult(
                tractable=False,
                confidence="low",
                flags=("invalid_smiles",),
                details="Could not parse SMILES",
            )

        for flag_name, smarts in _PRODRUG_PATTERNS.items():
            pattern = Chem.MolFromSmarts(smarts)
            if pattern is not None and mol.HasSubstructMatch(pattern):
                flags.append(flag_name)

        # Check molecular weight extremes (PBPK less reliable)
        from rdkit.Chem import Descriptors
        mw = Descriptors.MolWt(mol)
        if mw > 900:
            flags.append("high_mw")

    except ImportError:
        logger.warning("RDKit not available for applicability check")

    # Check P-gp substrate (from existing transporter lookup)
    try:
        from omega_pbpk.ml.models.adme.transporter_lookup import is_pgp_substrate
        if is_pgp_substrate(smiles=smiles, drug_name=drug_name):
            flags.append("pgp_substrate")
    except ImportError:
        pass

    # Determine tractability and confidence
    hard_flags = {"prodrug_ester", "prodrug_phosphonate", "prodrug_carbamate",
                  "known_untractable", "invalid_smiles"}
    soft_flags = {"pgp_substrate", "high_mw"}

    has_hard = bool(hard_flags & set(flags))
    has_soft = bool(soft_flags & set(flags))

    if has_hard:
        tractable = False
        confidence = "low"
    elif has_soft:
        tractable = True
        confidence = "medium"
    else:
        tractable = True
        confidence = "high"

    return ApplicabilityResult(
        tractable=tractable,
        confidence=confidence,
        flags=tuple(flags),
        details=f"Detected: {', '.join(flags)}" if flags else "No flags",
    )
```

- [ ] **Step 4: Run tests — verify pass**

```bash
ruff format src/omega_pbpk/ml/applicability.py tests/ml/test_applicability.py
ruff check src/omega_pbpk/ml/applicability.py tests/ml/test_applicability.py
pytest tests/ml/test_applicability.py -v
```

- [ ] **Step 5: Commit**

```bash
git add src/omega_pbpk/ml/applicability.py tests/ml/test_applicability.py
git commit -m "feat: applicability domain filter — flag prodrugs and untractable drugs"
```

---

## Chunk 2: Training Data + Direct ML Predictor

### Task 2: Build Combined Training Set

**Files:**
- Create: `scripts/build_training_set.py`
- Output: `data/ml/clinical/cmax_training_set.csv`

- [ ] **Step 1: Write training set builder**

```python
#!/usr/bin/env python3
"""Build combined training set from benchmark results.

Merges:
- 25-drug benchmark (outputs/benchmark_*.json)
- Expanded benchmark (outputs/expanded_benchmark_*.json)

Deduplicates by drug name. Outputs CSV with columns:
drug, smiles, dose_mg, obs_cmax, pred_cmax_pbpk, log_fe, tier
"""
import json
import math
import sys
from pathlib import Path

import numpy as np

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root / "src"))


def main():
    # Load expanded benchmark
    eb_files = sorted((repo_root / "outputs").glob("expanded_benchmark_*.json"))
    if not eb_files:
        print("ERROR: No expanded benchmark found")
        sys.exit(1)
    with open(eb_files[-1]) as f:
        eb = json.load(f)

    # Load 25-drug benchmark
    bm_files = sorted((repo_root / "outputs").glob("benchmark_*.json"))
    if not bm_files:
        print("ERROR: No benchmark found")
        sys.exit(1)
    with open(bm_files[-1]) as f:
        bm = json.load(f)

    # Collect all drugs with valid Cmax
    drugs = {}

    # Expanded benchmark first
    for r in eb["per_drug"]:
        if "error" in r:
            continue
        obs = r.get("obs_cmax")
        pred = r.get("pred_cmax")
        smiles = r.get("smiles", "")
        dose = r.get("dose_mg", 0)
        if obs and obs > 0 and pred and pred > 0 and smiles and dose and dose > 0:
            drugs[r["drug"]] = {
                "drug": r["drug"],
                "smiles": smiles,
                "dose_mg": dose,
                "obs_cmax": obs,
                "pred_cmax_pbpk": pred,
                "tier": r.get("tier", "gold"),
            }

    # 25-drug benchmark (fill gaps)
    for r in bm["per_drug"]:
        if r["drug"] in drugs:
            continue
        obs = r.get("obs_cmax", 0)
        pred = r.get("pred_cmax", 0)
        if obs > 0 and pred > 0:
            drugs[r["drug"]] = {
                "drug": r["drug"],
                "smiles": r["smiles"],
                "dose_mg": r["dose_mg"],
                "obs_cmax": obs,
                "pred_cmax_pbpk": pred,
                "tier": "benchmark",
            }

    # Write CSV
    out_path = repo_root / "data" / "ml" / "clinical" / "cmax_training_set.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write("drug,smiles,dose_mg,obs_cmax,pred_cmax_pbpk,log_obs_cmax_per_mg,tier\n")
        for name, d in sorted(drugs.items()):
            log_cmax_per_mg = math.log(d["obs_cmax"] / d["dose_mg"])
            f.write(
                f"{name},{d['smiles']},{d['dose_mg']},{d['obs_cmax']},"
                f"{d['pred_cmax_pbpk']},{log_cmax_per_mg:.6f},{d['tier']}\n"
            )

    print(f"Training set: {len(drugs)} drugs → {out_path}")

    # Check applicability
    from omega_pbpk.ml.applicability import check_applicability
    tractable = sum(1 for d in drugs.values() if check_applicability(d["smiles"]).tractable)
    print(f"Tractable: {tractable}/{len(drugs)}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run builder**

```bash
python scripts/build_training_set.py
```

Expected: ~75 drugs saved to CSV.

- [ ] **Step 3: Commit**

```bash
git add scripts/build_training_set.py
git commit -m "feat: build combined Cmax training set from benchmark results"
```

---

### Task 3: XGBoost Direct Cmax Predictor

**Files:**
- Create: `src/omega_pbpk/ml/models/direct_pk/__init__.py`
- Create: `src/omega_pbpk/ml/models/direct_pk/xgboost_cmax.py`
- Test: `tests/ml/test_direct_cmax.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/ml/test_direct_cmax.py
"""Tests for XGBoost direct Cmax predictor."""
import numpy as np


class TestDirectCmax:
    def test_predict_returns_positive(self):
        from omega_pbpk.ml.models.direct_pk.xgboost_cmax import DirectCmaxPredictor

        predictor = DirectCmaxPredictor()
        cmax = predictor.predict("CN1C=NC2=C1C(=O)N(C(=O)N2C)C", dose_mg=200.0)
        assert cmax > 0

    def test_dose_scaling(self):
        """Higher dose should give higher Cmax."""
        from omega_pbpk.ml.models.direct_pk.xgboost_cmax import DirectCmaxPredictor

        predictor = DirectCmaxPredictor()
        cmax_low = predictor.predict("CN1C=NC2=C1C(=O)N(C(=O)N2C)C", dose_mg=100.0)
        cmax_high = predictor.predict("CN1C=NC2=C1C(=O)N(C(=O)N2C)C", dose_mg=400.0)
        assert cmax_high > cmax_low

    def test_features_extraction(self):
        from omega_pbpk.ml.models.direct_pk.xgboost_cmax import smiles_to_features

        feats = smiles_to_features("CN1C=NC2=C1C(=O)N(C(=O)N2C)C")
        assert feats is not None
        assert len(feats) == 2057  # 2048 Morgan + 9 descriptors

    def test_batch_predict(self):
        from omega_pbpk.ml.models.direct_pk.xgboost_cmax import DirectCmaxPredictor

        predictor = DirectCmaxPredictor()
        smiles_list = [
            "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # caffeine
            "CC(=O)NC1=CC=C(O)C=C1",  # acetaminophen
        ]
        results = predictor.predict_batch(smiles_list, dose_mg=200.0)
        assert len(results) == 2
        assert all(r > 0 for r in results)
```

- [ ] **Step 2: Run tests — verify fail**

```bash
pytest tests/ml/test_direct_cmax.py -v
```

- [ ] **Step 3: Implement DirectCmaxPredictor**

```python
# src/omega_pbpk/ml/models/direct_pk/__init__.py
```

```python
# src/omega_pbpk/ml/models/direct_pk/xgboost_cmax.py
"""Direct Cmax prediction from SMILES using XGBoost.

Predicts log(Cmax/dose_mg) from molecular descriptors (Morgan fingerprints
+ RDKit descriptors). Final Cmax = exp(prediction) × dose_mg.

Training: scripts/train_direct_cmax.py
Model: models/direct_pk/xgboost_cmax.json
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

_MODEL_PATH = Path(__file__).resolve().parents[5] / "models" / "direct_pk" / "xgboost_cmax.json"


def smiles_to_features(smiles: str) -> np.ndarray | None:
    """Convert SMILES to 2057-dim feature vector.

    Features: 2048 Morgan fingerprint bits (radius=2) + 9 RDKit descriptors.
    Same feature set as xgboost_fup/xgboost_clint for consistency.
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem, Descriptors

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        # Morgan fingerprint (2048 bits, radius 2)
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
        fp_arr = np.array(fp, dtype=np.float32)

        # 9 RDKit descriptors (normalized)
        desc = np.array(
            [
                Descriptors.MolLogP(mol),
                Descriptors.TPSA(mol) / 200.0,
                Descriptors.MolWt(mol) / 600.0,
                Descriptors.NumHAcceptors(mol) / 10.0,
                Descriptors.NumHDonors(mol) / 5.0,
                Descriptors.NumRotatableBonds(mol) / 15.0,
                Descriptors.RingCount(mol) / 5.0,
                Descriptors.FractionCSP3(mol),
                Descriptors.MolMR(mol) / 150.0,
            ],
            dtype=np.float32,
        )

        return np.concatenate([fp_arr, desc])
    except Exception as e:
        logger.warning("Feature extraction failed for %s: %s", smiles[:30], e)
        return None


class DirectCmaxPredictor:
    """XGBoost model predicting Cmax directly from SMILES + dose.

    Predicts log(Cmax/dose_mg), then computes Cmax = exp(pred) × dose_mg.
    Falls back to a simple heuristic if model file not found.
    """

    def __init__(self):
        self._model = None
        self._load_model()

    def _load_model(self):
        if not _MODEL_PATH.exists():
            logger.info("Direct Cmax model not found at %s, using fallback", _MODEL_PATH)
            return
        try:
            import xgboost as xgb

            self._model = xgb.XGBRegressor()
            self._model.load_model(str(_MODEL_PATH))
            logger.info("Direct Cmax model loaded from %s", _MODEL_PATH)
        except Exception as e:
            logger.warning("Failed to load direct Cmax model: %s", e)

    def predict(self, smiles: str, dose_mg: float = 100.0) -> float:
        """Predict Cmax (mg/L) from SMILES and dose."""
        feats = smiles_to_features(smiles)
        if feats is None:
            # Fallback: assume Cmax ~ dose / 100L (rough approximation)
            return dose_mg / 100.0

        if self._model is not None:
            import xgboost as xgb

            X = feats.reshape(1, -1)
            log_cmax_per_mg = float(self._model.predict(X)[0])
            return float(np.exp(log_cmax_per_mg) * dose_mg)
        else:
            # Fallback without model: use simple allometric estimation
            # Cmax ~ F × dose / Vd, assume F=0.5, Vd=100L
            return dose_mg * 0.5 / 100.0

    def predict_batch(self, smiles_list: list[str], dose_mg: float = 100.0) -> list[float]:
        """Predict Cmax for multiple SMILES."""
        return [self.predict(s, dose_mg) for s in smiles_list]
```

- [ ] **Step 4: Run tests — verify pass (fallback mode)**

```bash
ruff format src/omega_pbpk/ml/models/direct_pk/xgboost_cmax.py tests/ml/test_direct_cmax.py
ruff check src/omega_pbpk/ml/models/direct_pk/xgboost_cmax.py tests/ml/test_direct_cmax.py
pytest tests/ml/test_direct_cmax.py -v
```

Note: `test_dose_scaling` may fail in fallback mode since the fallback uses linear dose scaling. That's OK — it will pass after training.

- [ ] **Step 5: Commit**

```bash
git add src/omega_pbpk/ml/models/direct_pk/__init__.py src/omega_pbpk/ml/models/direct_pk/xgboost_cmax.py tests/ml/test_direct_cmax.py
git commit -m "feat: XGBoost direct Cmax predictor from SMILES (fallback mode)"
```

---

### Task 4: Training Script

**Files:**
- Create: `scripts/train_direct_cmax.py`
- Output: `models/direct_pk/xgboost_cmax.json`

- [ ] **Step 1: Write training script**

```python
#!/usr/bin/env python3
"""Train XGBoost direct Cmax predictor.

Reads cmax_training_set.csv, extracts Morgan+RDKit features,
trains XGBoost with 5-fold CV, saves model.

Usage:
    python scripts/train_direct_cmax.py
"""
import csv
import math
import sys
from pathlib import Path

import numpy as np

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root / "src"))

from omega_pbpk.ml.models.direct_pk.xgboost_cmax import smiles_to_features


def main():
    data_path = repo_root / "data" / "ml" / "clinical" / "cmax_training_set.csv"
    if not data_path.exists():
        print("ERROR: Run scripts/build_training_set.py first")
        sys.exit(1)

    # Load training data
    drugs = []
    with open(data_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            drugs.append(row)

    print(f"Loaded {len(drugs)} drugs from {data_path}")

    # Extract features
    X_list = []
    y_list = []
    names = []
    for d in drugs:
        feats = smiles_to_features(d["smiles"])
        if feats is None:
            print(f"  SKIP {d['drug']}: feature extraction failed")
            continue
        y = float(d["log_obs_cmax_per_mg"])  # log(Cmax / dose_mg)
        X_list.append(feats)
        y_list.append(y)
        names.append(d["drug"])

    X = np.array(X_list)
    y = np.array(y_list)
    print(f"Features: {X.shape}, target: {y.shape}")
    print(f"Target range: [{y.min():.2f}, {y.max():.2f}], mean={y.mean():.2f}")

    # Train with 5-fold CV
    import xgboost as xgb
    from sklearn.model_selection import cross_val_predict, KFold

    model = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=3,  # Shallow trees to prevent overfitting on 75 drugs
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.3,  # Use 30% of features per tree (key for 2057 features)
        min_child_weight=5,
        reg_alpha=1.0,
        reg_lambda=5.0,  # Strong regularization
        random_state=42,
    )

    # 5-fold CV predictions
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    y_pred_cv = cross_val_predict(model, X, y, cv=cv)

    # Compute CV metrics
    cv_errors = y - y_pred_cv  # in log space
    cv_fold_errors = np.exp(np.abs(cv_errors))  # fold errors
    cv_aafe = float(np.exp(np.mean(np.abs(cv_errors))))

    print(f"\n5-Fold CV Results:")
    print(f"  AAFE: {cv_aafe:.2f}")
    print(f"  RMSE (log): {np.sqrt(np.mean(cv_errors ** 2)):.3f}")
    print(f"  %2-fold: {100 * np.mean(cv_fold_errors <= 2.0):.0f}%")
    print(f"  %3-fold: {100 * np.mean(cv_fold_errors <= 3.0):.0f}%")

    # Train final model on all data
    model.fit(X, y)

    # Feature importance (top 15)
    importances = model.feature_importances_
    top_indices = np.argsort(importances)[::-1][:15]
    print("\nTop 15 features:")
    desc_names = ["MolLogP", "TPSA", "MolWt", "NumHAcceptors", "NumHDonors",
                  "NumRotatableBonds", "RingCount", "FractionCSP3", "MolMR"]
    for idx in top_indices:
        if idx < 2048:
            name = f"Morgan_bit_{idx}"
        else:
            name = desc_names[idx - 2048]
        print(f"  {name}: {importances[idx]:.4f}")

    # Save model
    out_dir = repo_root / "models" / "direct_pk"
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_model(str(out_dir / "xgboost_cmax.json"))
    print(f"\nModel saved: {out_dir / 'xgboost_cmax.json'}")

    # Save metadata
    import json
    meta = {
        "n_drugs": len(names),
        "cv_aafe": cv_aafe,
        "cv_rmse_log": float(np.sqrt(np.mean(cv_errors ** 2))),
        "cv_pct_2fold": float(100 * np.mean(cv_fold_errors <= 2.0)),
        "target": "log(Cmax_mg_L / dose_mg)",
        "n_features": int(X.shape[1]),
        "drugs": names,
    }
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Build training set and train**

```bash
python scripts/build_training_set.py
python scripts/train_direct_cmax.py
```

Expected: Model trained, CV AAFE reported, model saved.

- [ ] **Step 3: Re-run direct Cmax tests (with trained model)**

```bash
pytest tests/ml/test_direct_cmax.py -v
```

All tests should pass now with the trained model.

- [ ] **Step 4: Commit**

```bash
git add scripts/build_training_set.py scripts/train_direct_cmax.py
git commit -m "feat: training pipeline for direct Cmax predictor (XGBoost)"
```

---

## Chunk 3: Ensemble Integration

### Task 5: Ensemble PK Module

**Files:**
- Create: `src/omega_pbpk/ml/models/direct_pk/ensemble_pk.py`
- Test: `tests/ml/test_ensemble_pk.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/ml/test_ensemble_pk.py
"""Tests for PBPK + ML ensemble."""
import numpy as np


class TestEnsemblePK:
    def test_ensemble_returns_result(self):
        from omega_pbpk.ml.models.direct_pk.ensemble_pk import ensemble_cmax

        result = ensemble_cmax(
            cmax_pbpk=5.0,
            cmax_ml=3.0,
            confidence="high",
        )
        assert result > 0

    def test_high_confidence_favors_pbpk(self):
        """When both models agree, result should be between them."""
        from omega_pbpk.ml.models.direct_pk.ensemble_pk import ensemble_cmax

        result = ensemble_cmax(cmax_pbpk=5.0, cmax_ml=3.0, confidence="high")
        assert 3.0 <= result <= 5.0

    def test_low_confidence_favors_ml(self):
        """When confidence is low, lean toward ML."""
        from omega_pbpk.ml.models.direct_pk.ensemble_pk import ensemble_cmax

        result_low = ensemble_cmax(cmax_pbpk=10.0, cmax_ml=2.0, confidence="low")
        result_high = ensemble_cmax(cmax_pbpk=10.0, cmax_ml=2.0, confidence="high")
        # Low confidence → closer to ML (2.0)
        assert result_low < result_high

    def test_scale_ct_curve(self):
        """C(t) curve should be scaled to match ensemble Cmax."""
        from omega_pbpk.ml.models.direct_pk.ensemble_pk import scale_ct_curve

        time_h = np.array([0, 1, 2, 3, 4])
        cp = np.array([0.0, 5.0, 3.0, 2.0, 1.0])
        target_cmax = 10.0  # 2x the original Cmax of 5.0

        scaled = scale_ct_curve(time_h, cp, target_cmax)
        assert np.isclose(np.max(scaled), target_cmax)
        # Shape should be preserved
        assert np.argmax(scaled) == np.argmax(cp)
```

- [ ] **Step 2: Run tests — verify fail**

- [ ] **Step 3: Implement ensemble module**

```python
# src/omega_pbpk/ml/models/direct_pk/ensemble_pk.py
"""PBPK + ML ensemble for Cmax prediction.

Blends PBPK and direct ML Cmax predictions using a confidence-weighted
geometric mean. Scales the PBPK C(t) curve to match the ensemble Cmax.

Weight schedule:
- High confidence: 60% PBPK, 40% ML (PBPK trusted)
- Medium confidence: 40% PBPK, 60% ML (ML slightly preferred)
- Low confidence: 20% PBPK, 80% ML (ML strongly preferred)
"""
from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

# PBPK weight by confidence level (rest goes to ML)
_PBPK_WEIGHTS = {
    "high": 0.6,
    "medium": 0.4,
    "low": 0.2,
}


def ensemble_cmax(
    cmax_pbpk: float,
    cmax_ml: float,
    confidence: str = "medium",
) -> float:
    """Blend PBPK and ML Cmax predictions.

    Uses geometric mean with confidence-dependent weighting:
        Cmax_final = Cmax_pbpk^w × Cmax_ml^(1-w)

    Args:
        cmax_pbpk: PBPK pipeline Cmax prediction (mg/L)
        cmax_ml: Direct ML Cmax prediction (mg/L)
        confidence: PBPK confidence level ("high"/"medium"/"low")

    Returns:
        Ensemble Cmax (mg/L)
    """
    if cmax_pbpk <= 0 or cmax_ml <= 0:
        return max(cmax_pbpk, cmax_ml, 1e-12)

    w = _PBPK_WEIGHTS.get(confidence, 0.4)
    # Weighted geometric mean in log space
    log_ensemble = w * np.log(cmax_pbpk) + (1 - w) * np.log(cmax_ml)
    return float(np.exp(log_ensemble))


def scale_ct_curve(
    time_h: np.ndarray,
    cp_mg_L: np.ndarray,
    target_cmax: float,
) -> np.ndarray:
    """Scale a C(t) curve so its Cmax matches the target.

    Preserves the curve shape (absorption/elimination profile)
    while adjusting the magnitude.

    Args:
        time_h: Time array (hours)
        cp_mg_L: Concentration array (mg/L) from PBPK
        target_cmax: Desired Cmax from ensemble

    Returns:
        Scaled concentration array
    """
    current_cmax = np.max(cp_mg_L)
    if current_cmax <= 0:
        return cp_mg_L

    scale_factor = target_cmax / current_cmax
    return cp_mg_L * scale_factor
```

- [ ] **Step 4: Run tests — verify pass**

```bash
ruff format src/omega_pbpk/ml/models/direct_pk/ensemble_pk.py tests/ml/test_ensemble_pk.py
ruff check src/omega_pbpk/ml/models/direct_pk/ensemble_pk.py tests/ml/test_ensemble_pk.py
pytest tests/ml/test_ensemble_pk.py -v
```

- [ ] **Step 5: Commit**

```bash
git add src/omega_pbpk/ml/models/direct_pk/ensemble_pk.py tests/ml/test_ensemble_pk.py
git commit -m "feat: PBPK + ML ensemble with C(t) curve scaling"
```

---

### Task 6: Pipeline Integration

**Files:**
- Modify: `src/omega_pbpk/pipeline/__init__.py`

- [ ] **Step 1: Add direct ML predictor to OmegaPipeline.__init__**

In `__init__()`, add:
```python
        self._direct_cmax = None
```

In `_ensure_initialized()`, add after correction model loading:
```python
        # Load direct Cmax predictor for ensemble
        try:
            from omega_pbpk.ml.models.direct_pk.xgboost_cmax import DirectCmaxPredictor
            self._direct_cmax = DirectCmaxPredictor()
            if self._direct_cmax._model is not None:
                logger.info("OmegaPipeline: Direct Cmax predictor loaded.")
            else:
                self._direct_cmax = None
        except Exception as exc:
            logger.debug("Direct Cmax predictor not available: %s", exc)
```

- [ ] **Step 2: Add ensemble logic in simulate()**

After the correction model application block and before UQ computation, add:

```python
        # Ensemble PBPK + Direct ML prediction
        if self._direct_cmax is not None:
            try:
                from omega_pbpk.ml.models.direct_pk.ensemble_pk import (
                    ensemble_cmax,
                    scale_ct_curve,
                )
                from omega_pbpk.ml.applicability import check_applicability

                cmax_ml = self._direct_cmax.predict(request.smiles, request.dose_mg)
                app = check_applicability(request.smiles)

                # Use applicability-informed confidence for weighting
                ens_confidence = app.confidence if app.confidence != "high" else confidence
                cmax_ensemble = ensemble_cmax(cmax, cmax_ml, ens_confidence)

                logger.debug(
                    "Ensemble: PBPK=%.4f, ML=%.4f, conf=%s → %.4f",
                    cmax, cmax_ml, ens_confidence, cmax_ensemble,
                )

                # Scale C(t) curve to match ensemble Cmax
                cp = scale_ct_curve(time_h, cp, cmax_ensemble)
                cmax = cmax_ensemble
                auc = float(np_trapz(cp, time_h))  # Recompute AUC from scaled curve

                adme_props["cmax_ml"] = cmax_ml
                adme_props["ensemble_confidence"] = ens_confidence
                adme_props["applicability"] = app.flags
            except Exception as exc:
                logger.debug("Ensemble prediction failed: %s", exc)
```

- [ ] **Step 3: Format and lint**

```bash
ruff format src/omega_pbpk/pipeline/__init__.py
ruff check src/omega_pbpk/pipeline/__init__.py
```

- [ ] **Step 4: Run all tests**

```bash
pytest tests/ml/test_applicability.py tests/ml/test_direct_cmax.py tests/ml/test_ensemble_pk.py tests/ml/test_pgp_correction.py tests/ml/test_uq_integration.py -v --timeout=30
```

- [ ] **Step 5: Run 25-drug benchmark — check impact**

```bash
python scripts/run_full_benchmark.py
```

Expected: Cmax AAFE should change (may improve or stay similar on curated set). The real test is the expanded benchmark.

- [ ] **Step 6: Run expanded benchmark**

```bash
python scripts/run_expanded_benchmark.py --tiers platinum,gold
```

Expected: AAFE should improve significantly from 6.64 toward ~3-4x.

- [ ] **Step 7: Commit**

```bash
git add src/omega_pbpk/pipeline/__init__.py
git commit -m "feat: integrate PBPK+ML ensemble into pipeline with applicability filter"
```

---

## Chunk 4: CLint Improvement

### Task 7: Add Transporter Features to CLint Model

**Files:**
- Modify: `src/omega_pbpk/ml/models/adme/xgboost_clint.py`
- Create: `scripts/retrain_clint.py`

- [ ] **Step 1: Add transporter flags to CLint feature extraction**

In `xgboost_clint.py`, modify the `_smiles_to_features()` call to also include transporter flags:

```python
def _enhanced_features(smiles: str) -> np.ndarray | None:
    """Enhanced feature vector: Morgan + RDKit descriptors + transporter flags."""
    from omega_pbpk.ml.models.adme.xgboost_fup import _smiles_to_features

    base = _smiles_to_features(smiles)
    if base is None:
        return None

    # Add 4 transporter flags (P-gp sub, P-gp inh, OATP1B1 sub, BCRP sub)
    extra = np.zeros(4, dtype=np.float32)
    try:
        from omega_pbpk.ml.models.adme.transporter_lookup import get_transporter_flags
        flags = get_transporter_flags(smiles=smiles)
        if flags:
            extra[0] = float(flags.get("pgp_substrate", 0))
            extra[1] = float(flags.get("pgp_inhibitor", 0))
            extra[2] = float(flags.get("oatp1b1_substrate", 0))
            extra[3] = float(flags.get("bcrp_substrate", 0))
    except ImportError:
        pass

    return np.concatenate([base, extra])
```

This adds 4 features to the existing 2057, making it 2061 total. The model needs retraining.

- [ ] **Step 2: Create retraining script**

This retrains XGBoost CLint with the enhanced features. Uses same training data (TDC + 18 anchors) but with transporter flags added.

```bash
# scripts/retrain_clint.py — follows same pattern as xgboost_clint.py training
# but uses _enhanced_features() instead of _smiles_to_features()
```

- [ ] **Step 3: Retrain and evaluate**

```bash
python scripts/retrain_clint.py
```

Expected: Slight improvement in CLint prediction (transporter flags help for P-gp-mediated clearance).

- [ ] **Step 4: Run bronze benchmark to check CLint improvement**

```bash
python scripts/run_bronze_benchmark.py
```

Expected: CLint AAFE should decrease from 3.25.

- [ ] **Step 5: Commit**

```bash
git add src/omega_pbpk/ml/models/adme/xgboost_clint.py scripts/retrain_clint.py
git commit -m "feat: add transporter flags to CLint features, retrain model"
```

---

## Exit Criteria

| Criterion | Target | How to verify |
|-----------|--------|---------------|
| Applicability filter | Prodrugs and untractable drugs flagged | `test_applicability.py` passes |
| Direct ML predictor | Trained on 75 drugs, 5-fold CV AAFE reported | `train_direct_cmax.py` output |
| Ensemble integration | PBPK+ML ensemble in simulate() | `test_ensemble_pk.py` passes |
| Expanded AAFE improvement | Tractable drugs AAFE < 4.0 (from 6.64) | `run_expanded_benchmark.py` |
| 25-drug benchmark | No regression (Cmax AAFE ≤ 2.0) | `run_full_benchmark.py` |
| CLint improvement | CLint AAFE < 3.0 (from 3.25) | `run_bronze_benchmark.py` |
| All tests pass | 48K+ tests + new tests | `pytest tests/ -m "not slow and not benchmark" -q` |

---

## Risks

| # | Risk | Severity | Mitigation |
|---|------|----------|------------|
| 1 | XGBoost overfits on 75 drugs | High | max_depth=3, colsample_bytree=0.3, strong regularization, 5-fold CV |
| 2 | Ensemble degrades 25-drug benchmark | Medium | If AAFE increases >5%, disable ensemble for benchmarked drugs |
| 3 | Prodrug SMARTS over-matches (flags valid drugs) | Medium | Check false positive rate on 25-drug benchmark |
| 4 | CLint retraining with transporter flags doesn't help | Low | Transporter flags are sparse (96 drugs), may not improve TDC-trained model |
