# Phase 2B: Final Authoritative Plan

> **Date:** 2026-03-22
> **Status:** Approved — supersedes `specs/2026-03-22-phase2b-design.md` and `specs/2026-03-22-phase2b-consensus.md`
> **Debate participants:** systems architect (team-lead) + senior TPM (critic), 4 rounds
> **Scope:** Measurement infrastructure only. Zero model changes. Zero ODE changes.
> **Duration:** ~2.5 days (Day 0 = 1 hour, Days 1–3 = 2 days net)

---

## What This Plan Resolves

The original spec (4 items, ~2 weeks) and consensus doc (Phase 0-3 structure) were both over-scoped. This plan strips to what actually delivers information:

1. **Stale baseline is the #1 problem.** Holdout AAFE 1.847 is pre-E2E constants. All spec/consensus work was debating improvements to a number nobody has measured since March 19.
2. **Phase 0 "cheap measurements" are not cheap.** ka confound test requires Optuna re-run gated on BCS classification gated on unreliable predicted logS — circular, not 30 min.
3. **5-arm ablation requires pipeline engineering that doesn't exist.** Hybrid selector is hardwired in 130 lines of `simulate()`. Building a toggle is 0.5–1 day before a single ablation run.
4. **Item 4 (dissolution) has 7 open blockers and ~0.03 AAFE expected gain.** Phase 3.
5. **Meta-learner diagnostic is 3 lines, not a multi-experiment battery.** Extract intermediates from existing output, compare three blends on the same holdout run.

---

## Day 0 — Establish Ground Truth (~1 hour, do this first)

### Step 0a: Resolve selector state conflict
```bash
grep -rn "use_hybrid_selector\|_USE_HYBRID_SELECTOR" src/omega_pbpk/pipeline/
```
MEMORY.md says `use_hybrid_selector = False (already set)`. Architect says selector is hardwired with no flag. One is wrong. This grep takes 5 seconds and determines what "production" means for every downstream experiment.

Document finding in MEMORY.md: either "selector is hardwired ON" or "selector is controlled by `<flag_name>`, currently `<value>`."

### Step 0b: Run current benchmarks
```bash
source .venv/bin/activate
python scripts/run_holdout_benchmark.py    # 71 drugs, ~5s
python scripts/run_full_benchmark.py       # gold-24, ~2s
```

Update MEMORY.md with actual numbers. Until these run, every conversation about Phase 2B improvements is hypothetical.

### Step 0c: Inspect top-10 holdout errors
From the holdout output, identify the 10 drugs with highest fold error. Check if any have `clint_3a4 < 0.5 AND clint_2d6 < 0.5` — these are candidates for UGT-primary clearance where the pipeline's CYP-only CLint is unreliable. If ≥ 3 such drugs appear in the top-10, add the `PHASE2_PRIMARY` AD flag in Phase 1 (5 lines in `pipeline/_check_applicability_domain()`). Otherwise, skip — the flag solves a problem that isn't visible in the error distribution.

---

## Day 1 — Meta-Learner Diagnostic (half-day)

### Context
The meta-learner (12-feature XGBoost) shows only 0.026 AAFE improvement in CV over the baseline blend. Its top features are `log_cmax_ml` (25.6%) and `log_cmax_pbpk` (60.2%) — both blend inputs, not domain knowledge. The question is whether the 12-feature apparatus adds value over simpler blends, or whether it's capturing in-sample artifacts.

Intermediate values are already exposed: `cmax_pbpk` (post-selector ODE prediction) and `cmax_ml` (DirectCmaxPredictor output) are both available in `SimulationResult.adme_properties` at pipeline lines 902–903 and 945.

### Three experiments, same 71 holdout drugs

**Experiment A — Geometric mean (simplest possible baseline):**
```python
cmax_blend_A = np.sqrt(cmax_pbpk * cmax_ml)
```
No existing code. ~30-line script extracting from holdout benchmark output JSON.

**Experiment B — Ensemble fallback (`_USE_META_LEARNER = False`):**
Set the existing `_USE_META_LEARNER` flag to `False`, re-run `run_holdout_benchmark.py`. This activates the `ensemble_cmax()` fallback: confidence-weighted blend with `_PBPK_WEIGHTS = {"high": 1.0, "medium": 0.7, "low": 0.4}`. This is a fixed-weight blend, NOT equal-weight geometric mean — the two experiments are distinct.

**Experiment C — Current production (meta-learner ON, control):**
Already measured in Day 0. No additional work.

### Decision rule (three buckets)

| Gap `|C − A|` on holdout | Interpretation | Action |
|---|---|---|
| < 0.02 AAFE | Meta-learner is noise | Schedule simplification to fixed blend in Phase 3 |
| 0.02–0.05 AAFE | Undecided at N=71 | Report without action; revisit in Phase 3 with larger N |
| ≥ 0.05 AAFE | Meta-learner validated | Keep as-is; Experiment D (nested CV) becomes Phase 3 diagnostic |

Also compare B vs A: if B ≈ A, confidence weighting adds nothing over equal weights. If B > A, existing fallback is useful even without meta-learner.

### Deliverable
`scripts/verify_meta_learner_blend.py` (~80 lines): loads holdout benchmark output, extracts `cmax_pbpk`/`cmax_ml`, computes A and B, reports AAFE + %2-fold for all three experiments with bootstrap CI.

---

## Day 2 — AUC Validation (1 day)

### Problem
98.9% of platinum drugs have null AUC. Pipeline predicts AUC for every drug but it's unvalidated. AUC ∝ F × Dose / CL — directly tests clearance prediction orthogonal to Cmax.

### Data source
MMPK `mmpk_expanded_full.csv` has `auc_ng_h_ml` for ~1,223 of 1,260 drugs. Unit conversion: × 1e-3 → mg·h/L. Dose-normalize: compare AUC/dose_mg (predicted vs observed) to handle multi-dose entries.

Cross-reference with the 64 platinum-MMPK overlap drugs → ≥50 platinum entries with validated AUC, meeting the target.

### Schema addition to `platinum_reference.json`
```json
{
  "name": "midazolam",
  "auc_mg_h_L": 0.105,
  "auc_dose_mg": 5.0,
  "auc_source": "MMPK (ng·h/mL → mg·h/L)",
  "auc_single_dose": true
}
```

### Benchmark integration
Add to `run_holdout_benchmark.py` and `run_full_benchmark.py`:
```python
if drug.get("auc_mg_h_L") is not None:
    pred_auc_per_dose = result.auc / request.dose_mg
    obs_auc_per_dose = drug["auc_mg_h_L"] / drug["auc_dose_mg"]
    fe = max(pred_auc_per_dose / obs_auc_per_dose, obs_auc_per_dose / pred_auc_per_dose)
    auc_errors.append(fe)
auc_aafe = geometric_mean(auc_errors)
```

### Regression gate threshold
Measure AUC AAFE baseline first. Set gate at `min(3.0, baseline × 1.3)` — pre-specified absolute cap prevents post-hoc tuning to look good.

### Deliverable
- Updated `data/clinical/platinum_reference.json` with AUC for ≥50 drugs
- AUC metrics in benchmark scripts
- `tests/ml/test_auc_regression.py` — threshold set after baseline measurement

---

## Day 3 — MMPK Benchmark Script (1 day)

### Problem
N=71 holdout cannot distinguish AAFE 1.70 from 1.85 (CI width 0.44). Statistical power requires larger N. MMPK provides ~700 in-domain non-prodrug drugs, but 75% are in the V2/meta-learner training set — this set measures in-training-distribution behavior, not generalization. Must be labeled accordingly.

### Implementation: `scripts/run_mmpk_benchmark.py` (~250 lines)

1. Load `data/ml/clinical/mmpk_pbpk_features.csv` (1,128 drugs)
2. Exclude 30 holdout drugs (`mmpk_holdout_exclusions.json`)
3. Run fresh `OmegaPipeline.simulate()` on each drug (~700 × 73ms ≈ 51 seconds)
4. Apply AD filter → in-domain subset
5. Stratify by study count: Tier 1 (n≥2) vs Tier 2 (all)
6. Compute:
   - AAFE + bootstrap CI (10,000 resamples)
   - %2-fold, %3-fold
   - Spearman ρ: `scipy.stats.spearmanr(log_pred, log_obs)` ← 3 lines, closes MEMORY.md gap
   - Stratified results by MW / logP / study count
7. Output: `outputs/mmpk_benchmark_YYYY-MM-DD.json`

### Threshold
```python
threshold = max(2.0, baseline_aafe * 1.10)  # regression floor, not quality standard
```
Threshold is set from THIS run's output. Subsequent runs must stay within it.

### Mandatory contamination warning
All output files and test docstrings must include:
```
WARNING: ~75% of MMPK drugs are in the V2/meta-learner training set.
This benchmark measures in-training-distribution behavior.
Generalization estimate: holdout N=71 only.
```

### Regression tests
`tests/ml/test_mmpk_regression.py`:
```python
def test_mmpk_tier1_aafe():
    """Multi-study MMPK AAFE stays within regression threshold."""
    ...

def test_mmpk_tier2_aafe():
    """Full in-domain MMPK AAFE stays within regression threshold."""
    ...
```

---

## What Is NOT In Phase 2B

| Item | Reason | Destination |
|------|--------|-------------|
| Phase 0 ka confound test | Circular: uses predicted logS to classify BCS II drugs | Kill |
| BCS classification metadata in `_build_drug()` | Wrong predictions for the drugs that matter; Phase 3 should use measured solubility | Kill |
| Item 4 dissolution (43-state ODE) | 7 open blockers, ~0.03 AAFE expected gain | Phase 3 |
| Experiment D (nested CV) | Warranted only if meta-learner validated in Day 1; ~1 day ML work | Phase 3 (conditional) |
| CYP1A2/UGT feature expansion (Item 1b) | Requires stable meta-learner; backward to retrain before deciding if it should exist | Phase 3 |
| CYP1A2 spot-check | Requires data curation (SMILES + reference Cmax for non-benchmark drugs) | Phase 3 |
| 5-arm ablation | Requires selector toggle engineering (~0.5 day); contaminated MMPK set gives in-sample results only | Phase 3 (if warranted) |
| Selector toggle + refactor | Hybrid selector is 130 hardwired lines; flagging it is a real engineering task | Phase 3 |

---

## Phase 3 Entry Criteria

Phase 3 begins when Phase 2B is complete AND at least one condition is met:

| Condition | Phase 3 Action |
|-----------|----------------|
| Day 1: `|C − A|` ≥ 0.05 on holdout | Experiment D (nested CV V2 OOF → meta-learner retrain) |
| Day 3: MMPK Tier 1 AAFE > 2.0 | Investigate specific drug classes from stratified output |
| Day 2: AUC AAFE > 3.5 | CLint investigation (fuinc correction, Phase II enzymes) |
| Day 0 top-10 errors: ≥ 3 UGT-primary drugs | Item 1b UGT AD flag + meta-learner feature |
| Any trigger: mechanistic hypothesis formulated | Dissolution pilot, pKa integration, or transporter modeling |

If no condition is met after Phase 2B, the pipeline is at its practical ceiling for the current architecture. Next step is new data sources (PK-DB access, DrugBank), not more tuning.

---

## Success Criteria

| Metric | Current | Target | Source |
|--------|---------|--------|--------|
| Holdout AAFE (current) | 1.847 (stale) | Measured | Day 0 |
| Gold-24 AAFE | Unknown (stale) | Measured, ≤ 2.0 | Day 0 |
| AUC validated drugs | ~2 | ≥ 50 | Day 2 |
| AUC AAFE baseline | Unknown | Measured | Day 2 |
| MMPK Tier 1 AAFE | Unknown | Measured | Day 3 |
| Spearman ρ (ranking) | Never measured | Measured | Day 3 |
| Meta-learner verdict | Unknown | A/B/C comparison | Day 1 |

All targets are "measured" — this phase establishes baselines, it does not claim improvements.
