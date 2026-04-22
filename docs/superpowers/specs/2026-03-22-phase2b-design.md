# Phase 2B: Coverage Expansion + Metric Infrastructure

> **Date:** 2026-03-22
> **Status:** Spec — awaiting approval
> **Baseline:** Holdout in-domain AAFE 1.847 (stale, pre-E2E constants). Needs re-measurement.
> **Goal:** Expand pipeline scope (enzyme coverage, endpoint validation, dissolution) + build statistically meaningful benchmark infrastructure.

---

## Motivation

Phase 2A (CLAUDE.md Tasks 1-6) is largely exhausted:
- Tasks 1, 3: already deployed (meta-learner ON, E2E constants committed)
- Tasks 4, 6: blocked by error cancellation (8/8 individual component fixes worsened AAFE)
- Task 2: diminishing returns (batch 1: -1.5, batch 2: -0.4, batch 3: -0.08)
- Task 5: marginal (AD filter = reporting, not model improvement)

The 53-drug holdout cannot distinguish AAFE 1.70 from 1.85 (CI [1.65, 2.09]). Squeezing precision within the current architecture's ceiling is statistically meaningless.

**Phase 2B shifts strategy from precision to coverage:**
- More enzymes → more drugs predictable
- More endpoints → more complete validation
- Larger benchmark → statistically meaningful measurements
- Dissolution model → BCS II drugs enter scope

---

## Architecture Principle: Minimize Error Cancellation Risk

```
CLint → IVIVE → Fg → Fh → Kp → Vd → hybrid selector
```

Every component in this chain has compensating errors. 8/8 individual fixes worsened AAFE. Phase 2B items are categorized by their risk to this chain:

- **Items 1, 3: Zero risk.** Meta-learner feature injection operates AFTER the hybrid selector. AUC validation is data/metric infrastructure only. No model change.
- **Item 2: Zero risk.** Benchmark infrastructure. No model change.
- **Item 4: Medium risk.** Dissolution modifies the absorption rate feeding into gut_wall, which is the input to the Fg calculation. This IS within the error cancellation chain. Mitigation: pilot-gated deployment with explicit abort criteria. Only activates for BCS II/IV drugs (BCS I/III untouched).

---

## Item 1: Meta-Learner Value Verification + Feature Expansion

### 1a. Verify Meta-Learner Adds Value

**Problem:** Meta-learner (12-feature XGBoost, CV AAFE 1.974) vs ML-only (AAFE 2.00) shows only 0.026 improvement. Is this real or training leakage?

The meta-learner was trained on V2's in-sample predictions for `log_cmax_ml` (V2 trained on same 1,128 drugs). This makes `log_cmax_ml` optimistically accurate during training, inflating its 50% feature importance.

**Experiments (diagnostic script, ~150 lines):**

```
A. Simple geometric mean:  sqrt(cmax_pbpk × cmax_ml)
B. Optimal fixed weight:   cmax_pbpk^w × cmax_ml^(1-w), grid search w∈[0.2..0.8] on train
C. Current meta-learner:   12-feature XGBoost (production)
D. Nested-CV meta-learner: retrain with V2 out-of-fold predictions as log_cmax_ml
```

**Experiment D implementation detail:** Run V2 (DirectCmaxPredictor) in 5-fold scaffold-split CV on the 1,098 MMPK training drugs. For each fold, V2 trains on 4 folds (~878 drugs) and predicts on the held-out fold (~220 drugs). Collect all out-of-fold predictions. Use these as `log_cmax_ml` when retraining the meta-learner. Runtime: 5 × V2 training (~10s each) + meta-learner retrain (~5s) = ~1 minute.

Evaluate all 4 on holdout. Paired bootstrap test (same drugs, different methods) for statistical comparison.

**Decision rule:**
- If C ≈ A (paired p > 0.1): meta-learner is overfitting → replace with simple blend, remove complexity
- If C > A significantly: meta-learner is valid → proceed to 1b
- If D > C: nested CV training is better → retrain production model

### 1b. Feature Expansion (CYP1A2 Substrate Flag + UGT Susceptibility Flag)

**Only proceed if meta-learner is validated in 1a.**

Add 2 features to meta-learner, operating entirely outside the error cancellation chain:

**Feature 1: `is_cyp1a2_substrate` (binary CYP1A2 substrate flag)**

CYP1A2-specific CLint prediction is infeasible: TDC clearance data is total microsomal CLint (not CYP-specific), CYP1A2 inhibition IC50 ≠ substrate CLint, and N~50-100 known substrates is too small for meaningful XGBoost training. The project's own experience confirms MLP/XGBoost cannot learn at this scale (Decision 26).

Instead, use a **curated substrate lookup + SMARTS fallback:**
- Curated list: ~30 known CYP1A2 substrates from FDA drug labels (caffeine, theophylline, tizanidine, clozapine, olanzapine, fluvoxamine, melatonin, etc.)
- SMARTS fallback for unlisted drugs: planar aromatic amines `[nH]1cccc1` and polycyclic aromatics (CYP1A2 preferentially metabolizes flat, lipophilic molecules)
- Output: binary (0/1)
- Integration: meta-learner feature only. Does NOT enter IVIVE/Fg/Fh chain.
- **Abort criterion:** if feature importance < 0.5% after meta-learner retrain, drop it.

**Feature 2: `has_ugt_susceptible_group` (Phase II metabolism susceptibility)**

Renamed from `is_ugt_substrate` — these SMARTS will match many drugs where UGT metabolism is minor. The feature indicates structural susceptibility, not confirmed UGT substrate status.

- Detection: SMARTS patterns for functional groups susceptible to glucuronidation
  ```
  Phenol:              [OH1]c1ccccc1        → UGT1A1/1A9
  Carboxylic acid:     [CX3](=O)[OH1]       → UGT2B7 (acyl glucuronidation)
  Aliphatic hydroxyl:  [CH1,CH2][OH1]        → UGT2B7
  Aromatic amine:      [NH2]c1ccccc1         → UGT1A4 (N-glucuronidation)
  ```
- Output: binary (0/1). Expected high false-positive rate (ibuprofen, acetaminophen will match). This is acceptable — the meta-learner learns whether the signal is useful.
- Integration: meta-learner feature only. Does NOT affect CLint/IVIVE.
- Separately: AD filter flag `PHASE2_PRIMARY` when `has_ugt_susceptible_group=1 AND clint_3a4 < 1.0 AND clint_2d6 < 1.0` (drug likely cleared by Phase II, pipeline CYP-only CLint unreliable).
- **Abort criterion:** if feature importance < 0.5% after meta-learner retrain, drop it.

**Meta-learner retrain:**
- Features: 12 existing + `is_cyp1a2_substrate` + `has_ugt_susceptible_group` = 14
- Training set: 1,098 MMPK drugs (minus 30 holdout exclusions)
- Validation: scaffold-split 5-fold CV, then holdout evaluation
- If new features have importance < 0.5%: drop them, they don't help

**Risk:** Zero for pipeline. XGBoost regularization ignores useless features. Meta-learner operates after hybrid selector.

### Deliverables
- `scripts/verify_meta_learner.py` — Experiments A-D
- CYP1A2 substrate list + SMARTS in `src/omega_pbpk/ml/models/direct_pk/meta_learner.py`
- UGT susceptibility SMARTS in `src/omega_pbpk/ml/models/direct_pk/meta_learner.py`
- `PHASE2_PRIMARY` AD flag in `pipeline/__init__.py` `_check_applicability_domain()`
- Retrained `models/meta_learner/xgboost_meta_v2.json` with 14 features (only if 1a validates)

---

## Item 2: Weighted MMPK Primary Metric

### Problem

N=53 holdout cannot distinguish AAFE 1.70 from 1.85. The 95% CI width is 0.44. Any improvement within this range is noise.

### Design

Three-tier benchmark hierarchy:

```
Tier 1 (primary, statistical power):
  Multi-study MMPK subset (n_studies ≥ 2)
  ~250 in-domain, non-prodrug drugs (estimated from 521 multi-study
  minus holdout exclusions, minus prodrugs, minus AD-flagged)
  Unweighted AAFE (all drugs equal weight within tier)
  Threshold: AAFE ≤ 2.0
  Bootstrap CI (10,000 resamples)

Tier 2 (breadth):
  Full in-domain MMPK set
  ~700 drugs (multi-study + single-study, after AD filtering)
  Unweighted AAFE
  Threshold: AAFE ≤ 2.3

Tier 3 (regression gate, unchanged):
  Core-24 gold benchmark
  AAFE ≤ 2.0, %2-fold ≥ 75%
```

**Rationale for unweighted AAFE within tiers:** The original design proposed `w_i = min(n_studies_i, 5) / 5` but this adds complexity without demonstrated value. The tier structure already separates multi-study (reliable) from single-study (noisy). Within Tier 1, all drugs have n≥2 studies — further weighting by study count adds marginal information. Keep it simple.

### Implementation

**`scripts/run_mmpk_benchmark.py` (~250 lines):**

1. Load `data/ml/clinical/mmpk_pbpk_features.csv` (1,128 drugs)
2. Exclude holdout drugs (30 from `mmpk_holdout_exclusions.json`)
3. Run fresh `OmegaPipeline.simulate()` on each drug (stale pre-computed cmax_pbpk not used)
4. Apply AD filter → in-domain subset
5. Remove prodrug/extreme F drugs (using existing `_check_applicability_domain()`)
6. Stratify by study count: n≥2 (Tier 1) vs all (Tier 2)
7. Compute AAFE, bootstrap CI, %2-fold, %3-fold, stratified results (by MW/logP/enzyme class)
8. Output: `outputs/mmpk_benchmark_YYYY-MM-DD.json`

**Study count source:** `mmpk_expanded_full.csv` has multiple entries per drug when multi-dose data exists. Count unique dose values per drug name as proxy for study count. Verify by checking 10 drugs manually against MMPK documentation. Note: 971/1,260 unique drugs have n_studies=1 (from `mmpk_reliability.json`), so Tier 1 is the smaller, cleaner subset.

**Fresh predictions:** ~700 drugs × ~73ms = ~51 seconds. Acceptable.

**`tests/ml/test_mmpk_regression.py`:**
```python
def test_mmpk_tier1_aafe():
    """Multi-study MMPK AAFE must stay ≤ 2.0."""
    ...

def test_mmpk_tier2_aafe():
    """Full in-domain MMPK AAFE must stay ≤ 2.3."""
    ...
```

### Risk
None — no model change. Pure measurement infrastructure.

### Deliverables
- `scripts/run_mmpk_benchmark.py`
- `tests/ml/test_mmpk_regression.py`
- `outputs/mmpk_benchmark_YYYY-MM-DD.json` (baseline measurement)

---

## Item 3: AUC Reference Collection + Validation

### Problem

98.9% of platinum drugs have null AUC. Pipeline predicts AUC for every drug but it's essentially unvalidated. AUC is clinically more important than Cmax for narrow-TI drugs and DDI assessment.

### Design

**Phase 1: Data collection**

Primary source: **MMPK raw data** (`mmpk_expanded_full.csv`). MMPK has AUC values for ~1,223 of 1,260 drugs (in `auc_ng_h_ml` column). Unit conversion: ng·h/mL → mg·h/L = ×1e-3. This alone exceeds the 50-drug target.

Secondary sources (for platinum drugs not in MMPK):
1. FDA labels via DailyMed — AUC commonly reported alongside Cmax
2. PK-DB REST API — curated clinical AUC values

Target: ≥50 drugs in `platinum_reference.json` with validated AUC. Achievable from MMPK cross-reference alone for the ~64 platinum-MMPK overlap drugs.

Schema addition to `platinum_reference.json`:
```json
{
  "name": "midazolam",
  "auc_mg_h_L": 0.105,
  "auc_source": "MMPK (converted from ng·h/mL)",
  "auc_dose_mg": 5.0,
  "auc_single_dose": true
}
```

**Phase 2: AUC benchmark**

Add AUC evaluation to existing benchmark scripts:
```python
# In run_holdout_benchmark.py and run_mmpk_benchmark.py:
if drug.get("auc_mg_h_L") is not None:
    auc_fe = max(pred_auc / obs_auc, obs_auc / pred_auc)
    auc_errors.append(auc_fe)

auc_aafe = geometric_mean(auc_errors)
```

**Phase 3: AUC regression gate**

```python
def test_auc_aafe():
    """AUC AAFE on validated subset. Threshold set after measuring baseline."""
    ...
```

Initial threshold TBD — measure baseline first, then set threshold at 1.2× baseline.

### Why AUC matters for the pipeline

- AUC ∝ F × Dose / CL — directly tests clearance prediction accuracy
- Cmax is dominated by absorption rate + Vd, partially orthogonal to clearance
- Error cancellation may present differently for AUC vs Cmax
- AUC validation reveals whether CLint improvements (blocked for Cmax) might work for AUC

### Risk
None — data collection + metric addition. No model change.

### Deliverables
- Updated `data/clinical/platinum_reference.json` with AUC values (≥50 drugs)
- AUC metrics in benchmark scripts
- `tests/ml/test_auc_regression.py`

---

## Item 4: BCS II Dissolution Model → ODE Integration

### Problem

Current ODE assumes instant dissolution: `absorption_rate = ka × C_total`. For BCS II drugs (low solubility, high permeability — ~40% of market), dissolution is rate-limiting. This causes systematic Cmax over-prediction (drug dissolves slower than modeled → real Cmax is lower and later).

### Current State

Dissolution code exists but is disconnected from main ODE:
- `src/omega_pbpk/biopharmaceutics/bcs_classification.py` — BCS I-IV classification + Noyes-Whitney formula
- `src/omega_pbpk/core/dissolution_absorption_coupling.py` — dissolution + absorption + transit ODE
- `src/omega_pbpk/biopharmaceutics/dissolution.py` — general dissolution models

Main ODE (`body.py`) uses fixed ka calibrated by Optuna on 1,020 MMPK drugs.

### Risk Profile

**Medium risk. Dissolution IS within the error cancellation chain.** Slowing dissolution changes the absorption rate into gut_wall, which is the direct input to the Fg (gut extraction) calculation. BCS II drugs may currently have compensating errors: instant-dissolution assumption over-predicts absorption rate, but other errors (Kp, Vd) under-predict Cmax, resulting in accidentally correct final values.

Mitigation: pilot-gated deployment. Only deploy if pilot shows improvement on ≥3 of 5 BCS II drugs.

### Design

**BCS classification at drug build time:**

```python
# In _build_drug():
from omega_pbpk.biopharmaceutics.bcs_classification import classify_bcs

# classify_bcs requires: solubility (mg/mL), dose (mg), permeability (cm/s), volume (mL)
# Note: pipeline peff is in units of 1e-4 cm/s → convert to cm/s
bcs_class = classify_bcs(
    solubility_mg_per_mL=drug.solubility_mg_mL,
    dose_mg=request.dose_mg,
    permeability_cm_per_s=drug.peff * 1e-4,  # convert from 1e-4 cm/s to cm/s
    volume_mL=250.0,  # standard GI volume
)
drug.bcs_class = bcs_class  # "I", "II", "III", or "IV"
```

**Dissolution activation guard (beyond BCS classification):**

BCS classification alone may misclassify due to predicted solubility errors. Apply dissolution model only when `dose_number > 2.0` (strong evidence of solubility limitation):
```python
dose_number = request.dose_mg / (drug.solubility_mg_mL * 250.0)
use_dissolution = (bcs_class in ("II", "IV")) and (dose_number > 2.0)
```

**Dissolution-limited absorption for qualifying drugs:**

```python
# In body.py gut absorption:
if self.drug.use_dissolution:
    # Noyes-Whitney dissolution rate
    k_diss = 3 * D_eff / (rho * r0**2)  # 1/s
    c_sat = self.drug.solubility_mg_mL

    # Each GI segment tracks solid + dissolved fractions
    diss_rate = k_diss * max(0, c_sat - c_dissolved) * mass_solid
    abs_rate = ka * c_dissolved * V_segment

    dydt[solid_idx] = -diss_rate + transit_in_solid - transit_out_solid
    dydt[dissolved_idx] = diss_rate - abs_rate + transit_in_dissolved - transit_out_dissolved
else:
    # Current behavior (instant dissolution, unchanged)
    abs_rate = ka * c_total * V_segment
```

**Default particle parameters (when formulation unknown):**

```python
DEFAULT_PARTICLE_RADIUS_UM = 25.0   # standard milled (Simcyp default)
DEFAULT_DIFFUSION_CM2_S = 5e-6      # aqueous diffusion, MW ~300
DEFAULT_DENSITY_G_CM3 = 1.2         # crystalline organic solid
```

**ODE state variable handling:**

The simplest approach: always allocate the 8 extra GI solid-phase states (total: 43 states), but initialize solid=0 and dissolved=full_dose for BCS I/III drugs. This avoids conditional array sizes while maintaining backward-compatible behavior for non-dissolution drugs (solid stays at 0, dissolved behaves identically to current total).

Impact on existing code:
- `N_STATES` increases from 35 to 43
- All array allocations, index constants, and mass-balance checks must be updated
- `pk_summary()` and downstream consumers handle new states
- 48K+ tests: BCS I/III behavior is identical (solid=0 throughout), so existing test assertions should pass if the dissolved state maps to the old total state

**Sensitivity analysis (part of pilot):**

Run each pilot drug at particle radius = {10, 25, 50, 100} µm. If Cmax varies >3x across this range, particle size uncertainty dominates and the model is unreliable without formulation-specific data. In that case, document the sensitivity and only deploy for drugs where particle size has small impact.

**Pilot drugs (BCS II, must have reference data in platinum/MMPK):**

Before committing to this list, verify each drug exists in platinum_reference.json or MMPK with Cmax reference data. Candidate drugs (subject to verification):

| Drug | BCS | Notes |
|------|-----|-------|
| Nifedipine | II | Common BCS II reference, gold-24 drug |
| Carbamazepine | II | In MMPK, well-characterized |
| Ibuprofen | II | In gold-24, BCS II borderline |
| Phenytoin | II | Nonlinear PK — may need exclusion |
| Griseofulvin | II | Classic BCS II but may not be in reference sets |

**Pilot protocol:**
1. Verify reference data availability for each pilot drug
2. Run pipeline with dissolution OFF → record Cmax fold errors
3. Run pipeline with dissolution ON → record Cmax fold errors
4. Run particle size sensitivity sweep {10, 25, 50, 100} µm
5. Decision: deploy if ≥3/5 drugs improve AND sensitivity sweep shows <3x Cmax variation

### Deliverables
- Modified `src/omega_pbpk/core/body.py` — 43-state ODE with solid/dissolved GI tracking
- Modified `src/omega_pbpk/pipeline/__init__.py` — BCS classification + dose_number guard in `_build_drug()`
- `tests/ml/test_dissolution_integration.py` — BCS I behavior unchanged, BCS II dissolution active
- Pilot results document with before/after + sensitivity analysis

---

## Execution Order + Dependencies

```
Item 2 (MMPK metric)          ← first: establishes measurement infrastructure
  │
  ├── Item 1a (meta-learner verification)    ← diagnostic, uses new metric
  │     │
  │     └── Item 1b (feature expansion)      ← only if 1a validates meta-learner
  │
  ├── Item 3 (AUC collection)               ← independent, parallel with 1a
  │
  └── Item 4 (dissolution pilot)            ← independent, parallel with 1a/3
        │
        └── Item 4 full deployment           ← only if pilot shows ≥3/5 improvement
```

**Item 2 is prerequisite for all others** — without a statistically meaningful benchmark, we can't evaluate whether Items 1/3/4 help.

---

## Success Criteria

| Metric | Current | Target | How |
|--------|---------|--------|-----|
| MMPK Tier 1 AAFE (multi-study) | Unknown | ≤ 2.0 | Item 2 measures, Items 1/4 improve |
| MMPK Tier 2 AAFE (full) | ~2.22 | ≤ 2.3 | Same |
| AUC validated drugs | ~2 | ≥ 50 | Item 3 (primarily from MMPK cross-ref) |
| AUC AAFE | Unknown | Baseline × 1.2 | Item 3 measures baseline, sets threshold |
| Meta-learner validated | No | Yes/No + evidence | Item 1a |
| Dissolution pilot positive | N/A | ≥ 3/5 drugs improved | Item 4 pilot |

**Removed: "In-domain drug count 53→70."** Neither Item 1b (adds AD flags, which REMOVES drugs from in-domain) nor Item 4 (changes absorption model for existing drugs, does not add new drugs to reference sets) actually expands the in-domain count. Honest success criteria measure what the items actually deliver.

---

## What This Plan Does NOT Do

- Does not attempt AAFE 1.847 → 1.70 on N=53 (statistically meaningless)
- Does not add drug-specific if/else heuristics
- Does not recalibrate K, IVIVE_ALPHA, IVIVE_BETA (error cancellation wall)
- Does not retrain on holdout drugs
- Does not replace ODE with ML (settled decision)
- Does not claim dissolution model is "outside" error cancellation chain (it is inside, gated by pilot)
