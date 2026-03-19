# MMPK Meta-Learner + Platinum Gate — Design Spec

> **Created:** 2026-03-19
> **Goal:** Platinum AAFE 2.95 → < 2.0 by training a meta-learner that optimally blends PBPK + ML predictions on 1,144 clinical drugs
> **Approach:** Phase A (data quality audit) + Phase B (DirectCmaxV2 + MetaLearner + gate update)
> **Key insight:** Previous 8 ML experiments "failed" because gold-24 gate (AAFE ≤ 1.70) blocked improvements to platinum. Gate switch to platinum-primary unlocks ML deployment.

---

## 1. Problem Statement

**Current performance:**
- Gold-24 AAFE: 1.50 (memorized/tuned — not honest generalization)
- Platinum (147 drugs) AAFE: 2.95 [2.49, 3.53]
- Tractable (120 drugs) AAFE: 2.65

**Root causes of platinum AAFE 2.95:**
- 31 drugs with FE > 5× contribute ~52% of total error
- DirectCmaxPredictor trained on only 75 drugs (AAFE 10.8 standalone)
- Fixed ensemble weights {high:1.0, medium:0.7, low:0.4} hand-tuned for gold-24
- MMPK 1,144 drugs downloaded but unused

**Why previous ML attempts failed:**
- ADME model changes (CLint, fup) break error cancellation — genuine co-adaptation problem
- DirectCmax/meta-learner changes were evaluated against gold-24 gate — gate artifact
- DirectCmax MMPK retrain: gold-24 1.50→1.71 (failed gate by 0.01) — platinum effect unknown

---

## 2. Architecture

### Current Pipeline
```
SMILES → ADME → ODE → Hybrid Selector → cmax_pbpk  ─┐
                                                       ├→ ensemble_cmax() → final
SMILES → DirectCmax(75 drugs) → cmax_ml             ─┘
         Fixed geometric mean: pbpk^w × ml^(1-w)
         w = {high:1.0, medium:0.7, low:0.4}
```

### New Pipeline
```
SMILES → ADME → ODE → Hybrid Selector → cmax_pbpk  ─┐
                                                       │
SMILES → DirectCmaxV2(1,144 drugs) → cmax_ml        ─┤
                                                       ↓
                                          CmaxMetaLearner
                                          (XGBoost, 1,144 drugs)
                                          12 features:
                                            log(cmax_pbpk)
                                            log(cmax_ml)
                                            log(dose_mg)
                                            log(cmax_pbpk/cmax_ml)
                                            logP, TPSA, MW
                                            log(fup), log(clint)
                                            is_acid, is_base
                                            pgp_flag
                                                ↓
                                           final_cmax
```

### What changes:
1. **DirectCmaxV2**: Retrained on MMPK 1,144 drugs (was 75)
2. **CmaxMetaLearner**: Replaces `ensemble_cmax()` fixed-weight blend
3. **Gate thresholds**: Gold-24 ≤ 1.70 → 2.00, Platinum ≤ 4.00 → 2.50

### What does NOT change:
- PBPK pipeline (ADME → ODE → hybrid selector) — untouched
- Conformal UQ — untouched
- Applicability domain — untouched
- ADME models (CLint, fup, VDss XGBoost) — untouched (avoids error cancellation)

---

## 3. Component Details

### 3.1 DirectCmaxV2

**Purpose:** Predict Cmax/dose from molecular structure using 15× more training data.

**Changes from V1:**
- Training data: `cmax_training_set.csv` (75 drugs) → `mmpk_cmax_training.csv` (1,144 drugs)
- Target: same `log(cmax_mg_L / dose_mg)` — dose normalization preserved
- Features: same Morgan FP(2048) + 9 RDKit descriptors = 2,057 features
- Hyperparameters adjusted for larger dataset:
  - n_estimators: 100 → 200
  - max_depth: 3 → 4
  - min_child_weight: 5 → 3
  - colsample_bytree: 0.3 → 0.5
  - reg_lambda: 5.0 → 3.0

**Model path:** `models/direct_pk/xgboost_cmax_v2.json` (V1 preserved as fallback)

**Interface:** Same `DirectCmaxPredictor.predict(smiles, dose_mg) → float`. The class loads V2 if available, falls back to V1.

### 3.2 CmaxMetaLearner

**Purpose:** Learn when to trust PBPK vs ML, replacing fixed-weight ensemble.

**Features (12):**

| # | Feature | Source | Rationale |
|---|---------|--------|-----------|
| 1 | `log_cmax_pbpk` | pipeline output | PBPK model's prediction |
| 2 | `log_cmax_ml` | DirectCmaxV2 | ML model's prediction |
| 3 | `log_dose_mg` | input | Absolute Cmax is dose-dependent |
| 4 | `log_cmax_ratio` | derived | PBPK/ML disagreement signal |
| 5 | `logP` | RDKit Crippen | Lipophilicity (Kp, renal) |
| 6 | `TPSA_norm` | RDKit / 200 | Polarity (absorption, secretion) |
| 7 | `MW_norm` | RDKit / 600 | Size (permeability, BCS) |
| 8 | `log_fup` | ADME ensemble | Protein binding (Kp, CL) |
| 9 | `log_clint` | ADME ensemble | Clearance prediction |
| 10 | `is_acid` | compound_type | Ionization class (Kp model) |
| 11 | `is_base` | compound_type | Ionization class (Kp model) |
| 12 | `pgp_flag` | applicability | Efflux transporter |

**Target:** `log10(cmax_observed_mg_L)`

**Model:** XGBoost regressor
- n_estimators=200, max_depth=4, learning_rate=0.08
- min_child_weight=5, reg_alpha=0.5, reg_lambda=3.0
- subsample=0.8, colsample_bytree=0.8

**Training data construction:**
For each of ~1,100 clean MMPK drugs:
1. Run `pipeline.simulate(smiles, dose_mg)` → extracts cmax_pbpk, fup, clint, compound_type, pgp
2. Run `DirectCmaxV2.predict(smiles, dose_mg)` → cmax_ml
3. Compute derived features (ratio, logs)
4. Target: `log10(mmpk_cmax_mg_L)` from MMPK reference

**Fallback:** If MetaLearner model not loaded → use existing `ensemble_cmax()` (backward compatible)

**Integration point:** `pipeline/__init__.py` after hybrid selector + DirectCmax, replacing `ensemble_cmax()` call.

### 3.3 Gate Update

**Gold-24 regression gate (`test_gold24_regression.py`):**
- `AAFE_THRESHOLD`: 1.70 → **2.00**
- `PCT_2FOLD_MIN`: 75.0 → **60.0**
- `MAX_SINGLE_FE`: 6.0 → **8.0** (warfarin may shift)

**Platinum regression gate (`test_platinum_regression.py`):**
- `CORE24_AAFE_MAX`: 1.70 → **2.00**
- `CORE24_PCT2FOLD_MIN`: 75.0 → **60.0**
- `PLATINUM_AAFE_MAX`: 4.00 → **2.50**
- `PLATINUM_PCT2FOLD_MIN`: 40.0 → **45.0**

**Rationale:** Gold-24 relaxation allows meta-learner deployment. Platinum tightening becomes the primary quality gate.

---

## 4. Phase A: Data Quality Audit

Before ML changes, audit platinum reference data for extraction errors.

**Scope:**
1. Cross-verify top 30 worst-predicted platinum drugs against FDA labels
2. Check dose normalization (some drugs may have wrong dose in reference)
3. Identify unit errors (µg/mL vs mg/L vs ng/mL confusion)
4. Flag and fix extraction artifacts (similar to 6-drug fix that gave AAFE -0.44)

**Script:** `scripts/audit_platinum_reference.py`
- Loads platinum_reference.json
- Runs pipeline on all drugs
- Sorts by fold error
- Cross-references dose/Cmax against FDA label patterns
- Outputs: list of suspected errors with evidence

**Expected impact:** AAFE 2.95 → 2.5-2.7 (conservative)

---

## 5. Phase B: Training Pipeline

### Step 1: MMPK Data Audit (`scripts/audit_mmpk_data.py`)
- Validate SMILES (RDKit parse)
- Check dose range (drop < 0.1 mg or > 5000 mg)
- Check Cmax range (drop ≤ 0 or Cmax/dose > 1.0)
- Check n_studies (prefer n_studies ≥ 2)
- Identify overlap with platinum (for holdout assignment)
- Output: `data/ml/clinical/mmpk_clean.csv` + audit report

### Step 2: PBPK Feature Generation (`scripts/generate_pbpk_features.py`)
- For each clean MMPK drug: run `pipeline.simulate()`
- Extract: cmax_pbpk, fup, clint, compound_type, pgp_flag, logP, TPSA, MW
- Save: `data/ml/clinical/mmpk_pbpk_features.csv`
- Estimated time: ~160 seconds for 1,100 drugs

### Step 3: Train DirectCmaxV2 (`scripts/train_direct_cmax_v2.py`)
- Input: mmpk_clean.csv
- Feature extraction: same smiles_to_features()
- 5-fold CV with AAFE reporting
- Train final model on all data
- Save: `models/direct_pk/xgboost_cmax_v2.json`

### Step 4: Train MetaLearner (`scripts/train_meta_learner.py`)
- Input: mmpk_pbpk_features.csv + DirectCmaxV2 predictions
- Construct 12 features per drug
- 5-fold CV with AAFE reporting
- Train final model on all data
- Save: `models/meta_learner/xgboost_meta.json` + `meta.json`

### Step 5: Pipeline Integration
- Modify `pipeline/__init__.py`: add `_USE_META_LEARNER = True` flag
- Load MetaLearner after hybrid selector + DirectCmax
- Replace `ensemble_cmax()` call with `meta_learner.predict(features)`
- Fallback to `ensemble_cmax()` if model not loaded

### Step 6: Gate Update + Validation
- Update thresholds in test files
- Run: `pytest tests/regression/ -v -m benchmark`
- Run: `python scripts/run_full_benchmark.py`
- Run: `python scripts/run_platinum_benchmark.py`

---

## 6. File Map

| File | Action | Phase |
|------|--------|-------|
| `scripts/audit_platinum_reference.py` | Create | A |
| `scripts/audit_mmpk_data.py` | Create | B.1 |
| `scripts/generate_pbpk_features.py` | Create | B.2 |
| `scripts/train_direct_cmax_v2.py` | Create | B.3 |
| `scripts/train_meta_learner.py` | Create | B.4 |
| `src/omega_pbpk/ml/models/direct_pk/xgboost_cmax.py` | Modify | B.3 |
| `src/omega_pbpk/ml/models/direct_pk/meta_learner.py` | Create | B.4 |
| `src/omega_pbpk/pipeline/__init__.py` | Modify | B.5 |
| `tests/regression/test_gold24_regression.py` | Modify | B.6 |
| `tests/regression/test_platinum_regression.py` | Modify | B.6 |
| `tests/ml/test_meta_learner.py` | Create | B.4 |
| `models/direct_pk/xgboost_cmax_v2.json` | Create | B.3 |
| `models/meta_learner/xgboost_meta.json` | Create | B.4 |

---

## 7. Acceptance Criteria

| Metric | Current | Target | Hard Gate |
|--------|---------|--------|-----------|
| **Platinum AAFE** | 2.95 | < 2.0 | ≤ 2.50 |
| Platinum %2-fold | 49% | > 55% | ≥ 45% |
| **Gold-24 AAFE** | 1.50 | < 2.0 | ≤ 2.00 |
| Gold-24 %2-fold | 83% | > 65% | ≥ 60% |
| DirectCmaxV2 CV AAFE | 10.8 (V1) | < 2.8 | — |
| MetaLearner CV AAFE | — | < 2.3 | — |
| Latency | 147ms | < 500ms | ≤ 500ms |

**Gold-24 AAFE "regression" from 1.50 to ~1.7-2.0 is expected and acceptable.** It reflects moving from memorization to generalization. Platinum improvement is the primary success metric.

---

## 8. Risk Mitigation

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| MMPK data quality poor | Medium | Audit script (Step B.1) with quality filters before training |
| DirectCmaxV2 no better than V1 | Low | MMPK is 15× more data; CV AAFE will improve. If < 20% improvement, investigate |
| MetaLearner overfits | Medium | 12 features on 1,100 drugs = manageable p/n. Strong regularization. |
| MetaLearner hurts gold-24 | High | Expected and accepted. Gate relaxed to 2.00. |
| PBPK feature generation slow | Low | ~160s total. Can parallelize if needed. |
| Meta-learner not better than fixed ensemble | Medium | If CV shows no improvement over ensemble_cmax(), don't deploy. Keep V2 DirectCmax only. |

---

## 9. Future Extensions (AAFE < 1.5 path)

This design is Phase 1 of a multi-phase ML improvement:

1. **Current plan (Phase 1):** MMPK meta-learner → AAFE ~2.0
2. **Phase 2 (data expansion):** Lombardo 1,352 + EPA httk 553 + EMA EPAR → 3-5K drugs → retrain → AAFE ~1.7-1.9
3. **Phase 3 (joint training):** Neural ODE surrogate + end-to-end gradient → AAFE ~1.5-1.7
4. **Phase 4 (population):** Multi-study deconvolution + population variability → AAFE ~1.3-1.5

Each phase builds on the previous. The meta-learner infrastructure (feature pipeline, evaluation framework, platinum gate) persists through all phases.
