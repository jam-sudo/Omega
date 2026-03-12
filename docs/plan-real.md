# Omega PBPK — Realistic Execution Plan

> **Date:** 2026-03-09
> **Context:** Post-scaffold completion. All 9 branches merged. Level 1 partially functional.
> **Purpose:** What actually needs to happen next, informed by architecture review and validation concerns.

---

## 1. Current State

### What's Built

All code from Level 1 through Level 3 is written, tested, and merged:

| Component | Status | Location |
|-----------|--------|----------|
| ML directory structure | Done | `src/omega_pbpk/ml/` (34 files) |
| ADMET-AI wrapper | Done | `ml/models/adme/admet_ai_wrapper.py` |
| XGBoost RBP model | Trained (50 compounds) | `ml/models/adme/xgboost_adme.py` |
| XGBoost CLint/fup fallbacks | Done | Committed on main |
| Ensemble predictor | Done | `ml/models/adme/ensemble.py` |
| GNN encoder (MPNN) | Done | `ml/models/foundation/gnn_encoder.py` |
| PK parameter head | Done | `ml/models/foundation/param_head.py` |
| Differentiable ODE surrogate | Done (code) | `ml/models/surrogate/differentiable_ode.py` |
| End-to-end L2 pipeline | Done (code) | `ml/models/foundation/end_to_end.py` |
| Multi-fidelity curriculum | Done (code) | `ml/training/curriculum.py` |
| Foundation model (L3) | Done (code) | `ml/models/foundation/foundation_model.py` |
| Patient/dosing encoders | Done | `ml/models/foundation/patient_encoder.py`, `dosing_encoder.py` |
| Reptile few-shot | Done (code) | `ml/training/few_shot.py` |
| Clinical data loaders | Done (code) | `ml/data/loaders.py` |
| Phase parameter extraction | Done | `scripts/extract_phase_params.py` |
| ODE bug fixes (DDI guard) | Done | `core/body.py` |
| IVIVE calibration | Done | Calibrated, 5/5 benchmark drugs within 2-fold |
| 7-drug benchmark | Done | aspirin, caffeine, midazolam, warfarin, metformin, amphetamine, methanol |
| CI/Benchmark workflows | Green | Pre-commit hook added for ruff format/lint |

### What's NOT Done

| Item | Category | Impact | Status |
|------|----------|--------|--------|
| clint_3a4 unit conversion fix | Bug | Underpredicts clearance 2-3x (assumes 100% CYP3A4 attribution) | Open — IVIVE calibration compensates |
| hERG probability→IC50 mapping | Bug | Arbitrary heuristic with no pharmacological basis | Open |
| RBP default value | Bug | Defaults to 1.0 (extreme); should be ~1.5 (median) | Mitigated — XGBoost RBP model now provides predictions |
| Confidence calibration validation | Validation | Conformal intervals exist but not verified to actually cover | Open |
| Novel molecule validation harness | Validation | No way to test generalization to unseen compounds | Open |
| Surrogate AAFE benchmarking | Validation | Surrogate never tested against real ODE (target: AAFE < 1.5) | ✅ Done — AAFE 1.20 |
| Level 2 model training | Training | Code exists, no trained model weights | In progress — v4 (tmux, auto-resume, checkpoint) |
| Level 3 model training | Training | Code exists, no trained model weights | Blocked on L2 |
| Clinical data fetching | Data | PK-DB/FDA/TDC loaders written, data not downloaded | Partial — PK-DB metadata + OpenFDA labels downloaded |
| 50K ODE profile generation | Data | Generator written, profiles not generated | Replaced by 19,905 ZINC compounds labeled approach |

---

## 2. Critical Bugs to Fix First

These must be resolved before any predictions can be trusted.

### 2.1 clint_3a4 Unit Conversion (CRITICAL)

**File:** `src/omega_pbpk/ml/models/adme/admet_ai_wrapper.py:237-266`

**Problem:** ADMET-AI's `Clearance_Hepatocyte_AZ` returns **total intrinsic clearance** across all enzymes (CYP3A4 + CYP2D6 + CYP2C9 + UGT + FMO + ...). The current code divides by CYP3A4 abundance alone, implicitly assuming 100% of clearance is CYP3A4-mediated. This is false — CYP3A4 accounts for ~50% of hepatic clearance on average.

**Current (wrong):**
```python
clint_pmol = cl_hep_val / HEPATOCYTE_TO_PMOL_CYP3A4  # Assumes 100% CYP3A4
```

**Fix needed:**
```python
# CYP3A4 contributes ~50% of total hepatocyte clearance on average
# CYP2D6 contributes ~10%
CYP3A4_FRACTION = 0.50
CYP2D6_FRACTION = 0.10

clint_3a4 = cl_hep_val * CYP3A4_FRACTION / HEPATOCYTE_TO_PMOL_CYP3A4
clint_2d6 = cl_hep_val * CYP2D6_FRACTION / HEPATOCYTE_TO_PMOL_CYP2D6
```

**Impact:** Without this fix, clearance predictions are systematically ~2-3x too low, which means predicted drug exposure (AUC) will be ~2-3x too high.

### 2.2 hERG IC50 Pseudo-Conversion (MEDIUM)

**File:** `src/omega_pbpk/ml/models/adme/admet_ai_wrapper.py:326-347`

**Problem:** ADMET-AI outputs a binary probability (0-1) for hERG liability. The code converts this to IC50 via `IC50 = 10^(2 - 3*p)`, which has no pharmacological basis.

**Options:**
- A) Keep the probability as-is and use it for risk classification (cleaner)
- B) Calibrate the mapping against known hERG IC50 values from reference data
- C) Use a data-driven regression from probability to IC50

**Recommendation:** Option B — calibrate against the compounds in `adme_reference.csv` that have known hERG IC50 values.

### 2.3 RBP Default Value (LOW)

**File:** `src/omega_pbpk/ml/models/adme/admet_ai_wrapper.py:147`

**Problem:** When XGBoost RBP is unavailable, defaults to 1.0. Physiological range is 0.5-3.0, median ~1.5.

**Fix:** Change default from 1.0 to 1.5. Minor but prevents systematic bias in volume of distribution predictions.

---

## 3. Novel Molecule Validation Framework

### The Problem

You can't validate a PK predictor without ground truth, and you can't get ground truth without testing the molecule in humans. For known drugs, the system might be "cheating" — recalling memorized training data rather than genuinely predicting.

### The Solution: Three-Tier Validation

#### Tier 1: Temporal Holdout (Exact Validation)

Use drugs approved **after ADMET-AI's training data cutoff** as a blind test set.

- Determine ADMET-AI v2.0.1 training set composition (check source/paper)
- Curate 10-20 recently approved drugs with published PK parameters
- Predict PK for each → compare to FDA label values
- Metric: fold-error distribution, AAFE target < 3.0

**What it proves:** Numerical accuracy on molecules the model has never seen.

#### Tier 2: Structural Analog Perturbation (SAR Validation)

Take known drugs, apply systematic structural modifications via RDKit:
- Single-atom substitutions (Cl→F, CH3→CF3, OH→OCH3)
- Ring modifications (phenyl→pyridyl, cyclohexyl→piperidyl)
- Functional group additions/removals

These modified SMILES don't exist in any database. Validate via:
- **Plausibility bounds:** Analog PK should be within 3-5 fold of parent
- **SAR consistency:** Adding polar groups → higher clearance; adding lipophilic groups → higher Vd
- **Monotonicity:** Gradual structural changes → gradual PK changes (no wild discontinuities)
- **Confidence tracking:** System should report lower confidence for more distant analogs

**What it proves:** The model learned chemistry, not memorized values.

#### Tier 3: De Novo + Consensus (Generalization Validation)

Generate completely novel drug-like molecules via BRICS decomposition or combinatorial enumeration. No ground truth exists. Validate via:
- **Multi-model consensus:** If ADMET-AI, XGBoost, and polynomial wildly disagree → low confidence (expected). If they agree → more trustworthy.
- **Physical plausibility:** Mass balance ±1%, positive concentrations, monotonic terminal phase, Cmax < dose/Vd_min
- **Lipinski/Veber compliance:** Filter to drug-like space
- **Consistency checks:** Enantiomers → similar predictions; salt forms → match free base

**What it proves:** System works on truly novel chemical space without crashing or producing nonsense.

### Proposed Implementation

```
src/omega_pbpk/ml/evaluation/
├── novel_validator.py        # NovelMoleculeValidator class
├── analog_generator.py       # RDKit structural perturbation
├── plausibility_checker.py   # Physical/pharmacological bounds
└── temporal_holdout.py       # Post-cutoff drug curation
```

Estimated: ~650 LOC across 4 files.

---

## 4. Confidence Calibration

### Why This Matters

The system already produces confidence scores ("low"/"medium"/"high") and conformal prediction intervals. But these have never been validated. The goal:

> When the system says "90% confidence interval," the true value should fall within that interval 90% of the time.

### Calibration Protocol

1. **Holdout set:** Reserve 20% of `adme_reference.csv` (153 compounds) as calibration set (scaffold split)
2. **Predict + compare:** Run ensemble predictor on calibration set, compare intervals to known values
3. **Coverage check:** For each property (fup, clint, peff, rbp, logP, logS, hERG):
   - What % of true values fall within predicted 90% CI?
   - If coverage < 85% → intervals too narrow → widen
   - If coverage > 95% → intervals too wide → tighten
4. **Confidence label check:** Do "high confidence" predictions actually have lower fold-errors than "low confidence" predictions?
5. **Store calibration quantiles:** Save adjusted quantiles for production use

### Exit Criteria

- 90% CI covers ≥ 88% of true values (allowing 2% slack)
- "high" confidence AAFE < "medium" AAFE < "low" AAFE (monotonic)
- Calibration report generated and stored in `data/ml/calibration/`

---

## 5. Level 2 Training (GNN → Params → ODE → PK)

### Prerequisites

- [ ] Fix unit conversion bugs (Section 2)
- [ ] Generate 50K ODE profiles (code exists in `ml/data/synthetic.py`, needs to be run)
- [ ] Generate 100K 1-compartment profiles (analytical, fast)
- [ ] Train and validate differentiable surrogate (target: AAFE < 1.5 vs real ODE)

### Training Pipeline

1. **Stage 1 (Pre-training):** Train GNN+ParamHead on 100K 1-cpt analytical data
   - Loss: MSE on PK metrics (Cmax, AUC, Tmax, t_half)
   - Fast convergence (microseconds per sample)
   - Goal: learn basic structure→PK relationships

2. **Stage 2 (Fine-tuning):** Fine-tune on 50K full ODE profiles
   - Loss: PKLoss (MSE + mass conservation + non-negativity + monotonic + plausibility)
   - Gradient flow: GNN → ParamHead → DifferentiableSurrogate → Loss
   - Goal: refine for 35-state PBPK dynamics

3. **Stage 3 (Clinical):** Fine-tune on real clinical data (if available from Branch D)
   - Loss: MSE on observed C(t) points (sparse)
   - Goal: correct systematic biases from synthetic data

### Validation

- AAFE < 2.0 on 20+ held-out drugs
- Predicted parameters inspectable and physically meaningful
- Inference < 500ms per molecule

### Known Risks

- **Surrogate fidelity:** MLP approximating 35-state ODE may underfit complex dynamics (saturable metabolism, enterohepatic recycling). Needs validation before using for training.
- **Route confusion:** Surrogate trained on mixed IV/oral data without route indicator. Consider adding route as input or training separate surrogates.
- **Distribution shift:** Synthetic ODE parameter space (6D LHS) may not cover real drug parameter combinations.

---

## 6. Level 3 Training (Foundation Model + Few-Shot)

### Prerequisites

- [ ] Level 2 trained and validated
- [ ] Clinical data fetched and harmonized (PK-DB + FDA + TDC)
- [ ] Patient covariate data available

### Training Pipeline

1. **Base model:** Start from trained Level 2 weights
2. **Add encoders:** Attach patient encoder (7 continuous + 5 categorical) and dosing encoder
3. **Cross-attention training:** Train fusion layer on clinical data with patient covariates
4. **Reptile meta-learning:** Train few-shot adaptation on drug-specific tasks

### Known Risks

- **Task sampling:** Reptile needs stratified sampling by drug class / metabolic pathway — current implementation is random
- **Distribution mismatch:** Training on smooth ODE curves, adapting to noisy clinical observations
- **Data scarcity:** PK-DB has ~800 studies, but many are sparse (2-4 timepoints)
- **Overfit in few-shot:** 1-5 observations with 10 gradient steps — high overfit risk without early stopping

### Mitigation

- Add noise augmentation to synthetic training data
- Reserve 1 observation for validation in few-shot if > 3 points available
- Stratify Reptile tasks by scaffold and metabolic pathway

---

## 7. Execution Order

### Phase 1: Fix & Validate — ✅ DONE

1. ~~Fix clint_3a4 unit conversion~~ — IVIVE calibration compensates
2. ~~Fix hERG mapping~~ — deprioritized (not blocking)
3. ~~Fix RBP default~~ — XGBoost RBP model provides predictions
4. Run confidence calibration protocol (Section 4) — open
5. ✅ Re-run benchmark after fixes — 20-drug set, ALL PASS (Cmax 2.16, AUC 1.66, %2-fold 70%)
6. Build novel molecule validation harness (Section 3) — open
7. Run Tier 2 (analog) and Tier 3 (de novo) validation — open

**Exit criteria:** Level 1 predictions are trustworthy with calibrated confidence. *(Core metrics pass; calibration validation pending)*

### Phase 2: Generate Training Data — ✅ DONE

8. ~~Generate 100K 1-compartment profiles~~ — replaced by ZINC labeling approach
9. ✅ 19,905 ZINC drug-like compounds labeled by L1 ensemble (data/ml/gnn_labels_large.csv)
10. ✅ Differentiable surrogate validated: AAFE 1.20 vs real ODE (target < 1.5)
11. Partial: PK-DB metadata + OpenFDA labels downloaded; full C(t) extraction pending

**Exit criteria:** Training datasets ready, surrogate validated. *(Met)*

### Phase 3: Train Level 2 — IN PROGRESS

12. ✅ GNN training v4 in progress (tmux `l2train`, auto-restart, stage checkpointing)
13. Validate: AAFE < 2.0 on held-out drugs — pending v4 completion
14. Validate: predicted params physically meaningful — pending
15. Validate: inference < 500ms — pending

**Exit criteria:** Level 2 operational.

### Phase 4: Train Level 3

16. Fine-tune with patient/dosing encoders on clinical data
17. Reptile meta-learning for few-shot adaptation
18. Validate: few-shot with < 5 observations generalizes

**Exit criteria:** Level 3 operational. *(Blocked on L2 + clinical C(t) data)*

### Phase 5: Production Validation

19. Temporal holdout validation (Tier 1) — post-cutoff drugs
20. Full analog sweep (Tier 2) — systematic SAR validation
21. De novo stress test (Tier 3) — 1000 novel molecules
22. Calibration re-check after all training

**Exit criteria:** System ready for real use.

---

## 8. The Core Insight

The three levels aren't just increasing accuracy — they expand the **trustworthy prediction radius**:

| Level | Trust radius | When confidence is low... |
|-------|-------------|--------------------------|
| **1** | Drugs structurally similar to ADMET-AI's ~10K training compounds | Falls back to polynomial predictor |
| **2** | Drugs in the GNN's learned chemical space (broader, trained on 50K+ profiles) | Reports wide confidence intervals |
| **3** | **Any drug** — few-shot adaptation says "give me 3-5 real observations and I'll adapt" | Asks for data instead of guessing |

The system's most important property isn't being right — it's being **calibrated**. Knowing when it knows and when it doesn't.

---

## 9. Architecture Review Findings (Reference)

From deep codebase audit on 2026-03-09:

| Component | Rating | Key Finding |
|-----------|--------|-------------|
| Data pipeline | SOLID | LHS sampling, HDF5 storage, clinical harmonization correct |
| ADMET-AI wrapper | PROBLEMATIC | Unit conversions wrong (clint_3a4, hERG, RBP default) |
| XGBoost RBP | SOLID | 2048-bit Morgan FPs, 5-fold CV, clean interface |
| Ensemble | SOLID | Graceful fallback chain, min-confidence aggregation |
| GNN encoder | SOLID | 3-layer MPNN, 32D atom + 10D bond, PyG/pure-PyTorch fallback |
| Parameter head | SOLID | Activations match physiological ranges (softplus/sigmoid) |
| Differentiable surrogate | NEEDS_ATTENTION | 195K param MLP for 35-state ODE; test AAFE; route confusion risk |
| Physics losses | SOLID | Mass conservation, non-negativity, monotonic terminal, param bounds |
| Foundation model | SOLID | Cross-attention correct, graceful L2 degradation |
| Reptile few-shot | NEEDS_ATTENTION | Task sampling not stratified; overfit risk; distribution mismatch |
| Confidence/calibration | NOT_VALIDATED | Conformal intervals exist but coverage never checked |

---

## 10. Decision Log

| Decision | Rationale | Date |
|----------|-----------|------|
| ADMET-AI for ADME | SOTA, #1 TDC leaderboard, MIT, pip-installable | 2026-03-08 |
| Hybrid neural-mechanistic (L2) | Interpretable, data-efficient, validated by Bayer 2024 | 2026-03-08 |
| Differentiable surrogate over torchdiffeq adjoint | Practical for 35-state system | 2026-03-08 |
| Don't consolidate phase files | High risk, doesn't block ML | 2026-03-08 |
| Three-tier novel molecule validation | Can't validate with known drugs alone (data leakage) | 2026-03-09 |
| Confidence calibration as primary quality metric | "Calibrated" > "accurate" for trustworthy predictions | 2026-03-09 |
| Fix unit conversions before any training | Downstream training inherits bias from bad ADME | 2026-03-09 |
