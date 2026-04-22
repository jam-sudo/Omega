# Phase 2B: Final Authoritative Plan

> **Date:** 2026-03-22
> **Status:** Final — supersedes `specs/2026-03-22-phase2b-design.md` and `specs/2026-03-22-phase2b-consensus.md`
> **Process:** Original spec → 2-round expert debate (6 participants) → devil's advocate stress-test → synthesis debate → self-review convergence
> **Scope:** Measurement infrastructure only. Zero permanent model changes. Diagnostic flag toggles only.
> **Duration:** ~2.5 days (Day 0 = 1 hour, Days 1–3 = 2 days net)

---

## What This Plan Resolves

The original spec (4 items, ~2 weeks) and consensus doc (Phase 0-3 structure) were both over-scoped. This plan strips to what actually delivers information:

1. **Stale baseline is the #1 problem.** Holdout AAFE 1.847 is pre-E2E constants. All spec/consensus work was debating improvements to a number nobody has measured since March 19.
2. **Phase 0 "cheap measurements" are not cheap.** ka confound test requires Optuna re-run gated on BCS classification gated on unreliable predicted logS — circular, not 30 min.
3. **5-arm ablation requires pipeline engineering that doesn't exist.** Hybrid selector is hardwired in 130 lines of `simulate()`. Building a toggle is 0.5–1 day before a single ablation run.
4. **Item 4 (dissolution) has 7 open blockers and ~0.03 AAFE expected gain.** Phase 3.
5. **Meta-learner diagnostic is ~80 lines, not a multi-experiment battery.** Extract intermediates from existing output, compare three blends on the same holdout drugs.

---

## Structural Findings (from debate, not in original spec)

### 1. Hybrid selector confounds component-level diagnosis
The selector's ratio-dependent weighting (ODE_Cmax / analytical_Cmax → blend weight) transforms any upstream change into an unpredictable downstream effect. This may explain why 8/8 individual component fixes worsened AAFE — not error cancellation per se, but the selector amplifying perturbations.

**Status:** Candidate hypothesis, NOT confirmed. Decision #3/#14 (selector is essential) is NOT revoked. Any future revocation must cite mechanistic justification, not underpowered holdout statistics.

### 2. ka_scale = 0.0004 creates a hidden dependency chain
Optuna calibrated ka on 1,020 MMPK drugs under instant-dissolution assumption. BCS II drugs in the training set (~20%) had their dissolution delay absorbed into ka. Adding dissolution without ka recalibration would double-correct BCS II drugs.

**Status:** Unmeasured. Measuring it requires Optuna re-run with BCS-conditional objective, which depends on reliable BCS classification (circular with predicted logS). Deferred to Phase 3 when dissolution is scoped.

### 3. In-sample metrics cannot justify out-of-sample decisions
CYP1A2 was dropped based on MMPK AAFE 1.67 — an in-sample metric. This contradicts the consensus position that MMPK metrics are contaminated. Any coverage feature decision must use holdout evaluation or non-benchmark spot-check.

### 4. Pipeline contamination stack
- **MMPK-fitted:** ka (Optuna), V2 DirectCmax (1,098 drugs), meta-learner (1,128 drugs)
- **Independent:** CLint/fup/VDss XGBoost (TDC data), Berezhkovskiy Kp (physics), well-stirred CLh (IVIVE formula)
- **Only clean generalization estimate:** 71-drug scaffold holdout

---

## Day 0 — Establish Ground Truth (~1 hour)

### Step 0a: Resolve selector state conflict
```bash
grep -rn "use_hybrid_selector\|_USE_HYBRID_SELECTOR" src/omega_pbpk/pipeline/
```
MEMORY.md says `use_hybrid_selector = False (already set)`. Code exploration shows selector is hardwired in lines 489-627 with no flag. Resolve this in 5 seconds. Document finding in MEMORY.md.

### Step 0b: Run current benchmarks
```bash
source .venv/bin/activate
python scripts/run_holdout_benchmark.py    # 71 drugs, ~5s
python scripts/run_full_benchmark.py       # gold-24, ~2s
```
Update MEMORY.md with actual numbers. Until these run, everything is hypothetical.

### Step 0c: Inspect top-10 holdout errors
From holdout output, identify 10 drugs with highest fold error. Check if any have `clint_3a4 < 0.5 AND clint_2d6 < 0.5` (UGT-primary candidates). If ≥ 3 such drugs in top-10: schedule `PHASE2_PRIMARY` AD flag addition for Day 3 (alongside MMPK script work). Otherwise skip.

---

## Day 1 — Meta-Learner Diagnostic (half-day)

### Context
The meta-learner (12-feature XGBoost) shows only 0.026 AAFE improvement in CV over the baseline blend. Feature importance from `meta.json`: `log_cmax_ml` = **49.9%**, `log_cmax_pbpk` = **25.6%**, `log_cmax_ratio` = **11.4%** — 86.8% of the model is blending two in-sample predictions. The 0.026 gap compares in-sample vs in-sample on MMPK; holdout gap is unknown.

Intermediate values are already exposed: `adme_properties["cmax_ml"]` (line 945) and `cmax_pbpk` (post-selector ODE Cmax, available at line 902 as `MetaFeatures.cmax_pbpk`).

### Three experiments, same 71 holdout drugs

**Experiment A — Geometric mean (simplest possible baseline):**
```python
cmax_blend_A = np.sqrt(cmax_pbpk * cmax_ml)
```
~30-line script extracting intermediates from holdout benchmark output.

**Experiment B — Ensemble fallback (`_USE_META_LEARNER = False`):**
Set existing `_USE_META_LEARNER` flag to `False` (temporary diagnostic toggle, revert after), re-run `run_holdout_benchmark.py`. Activates `ensemble_cmax()` fallback: confidence-weighted blend with `_PBPK_WEIGHTS = {"high": 1.0, "medium": 0.7, "low": 0.4}`. Distinct from equal-weight geometric mean.

**Experiment C — Current production (meta-learner ON):**
Already measured in Step 0b. No additional work.

### Decision rule (three buckets)

| Gap `|C − A|` on holdout | Interpretation | Action |
|---|---|---|
| < 0.02 AAFE | Meta-learner is noise | Schedule simplification to fixed blend in Phase 3 |
| 0.02–0.05 AAFE | Undecided at N=71 | Report without action; revisit in Phase 3 with larger N |
| ≥ 0.05 AAFE | Meta-learner validated | Keep as-is; Experiment D (nested CV) becomes Phase 3 diagnostic |

Also compare B vs A: if B ≈ A, confidence weighting adds nothing. If B > A, existing fallback is useful without meta-learner.

### Deliverable
`scripts/verify_meta_learner_blend.py` (~80 lines): extracts `cmax_pbpk`/`cmax_ml` from holdout results, computes A blend, reports AAFE + %2-fold for A/B/C with bootstrap CI.

---

## Day 2 — AUC Validation (1 day)

### Problem
98.9% of platinum drugs have null AUC. Pipeline predicts AUC for every drug but it's unvalidated. AUC ∝ F × Dose / CL — directly tests clearance prediction, orthogonal to Cmax.

### Data source
MMPK `mmpk_expanded_full.csv` has `auc_ng_h_ml` for ~1,223 of 1,260 drugs. Unit conversion: × 1e-3 → mg·h/L. Dose-normalize: compare AUC/dose_mg (predicted vs observed).

Cross-reference with 64 platinum-MMPK overlap drugs → ≥50 platinum entries with AUC.

### AUC comparison methodology
**Important:** Pipeline produces `auc0t_mg_h_L` (area from 0 to simulation end time). MMPK reference may report AUC0-inf or AUC0-t with different observation windows. For honest comparison:
- Use dose-normalized AUC: `AUC/dose` for both predicted and observed
- Flag drugs where pipeline simulation duration < 3× predicted t½ (AUC0t significantly underestimates AUC0inf)
- Document which AUC type (0-t vs 0-inf) each MMPK reference represents, if available

### Schema addition to `platinum_reference.json`
```json
{
  "name": "midazolam",
  "auc_mg_h_L": 0.105,
  "auc_dose_mg": 5.0,
  "auc_source": "MMPK (ng·h/mL → mg·h/L)",
  "auc_type": "AUC0-inf",
  "auc_single_dose": true
}
```

### Regression gate threshold
Measure AUC AAFE baseline first. Regression gate: `baseline_auc_aafe × 1.3` — same direction as MMPK gate (don't get worse). Absolute cap at 3.0 as a separate quality flag, NOT conflated with regression detection:
```python
auc_regression_threshold = baseline_auc_aafe * 1.3   # regression gate
auc_quality_flag = auc_aafe > 3.0                     # quality concern (separate)
```

### Deliverables
- Updated `data/clinical/platinum_reference.json` with AUC for ≥50 drugs (dose + AUC type annotated)
- AUC metrics in benchmark scripts (dose-normalized, AUC type flagged)
- `tests/ml/test_auc_regression.py`

---

## Day 3 — MMPK Benchmark Script (1 day)

### Problem
N=71 holdout has CI width 0.44 — insufficient for detecting 0.15 AAFE changes. MMPK N=700 provides statistical power, but ~75% are in V2/meta-learner training set. This measures training-distribution behavior, not generalization.

### Implementation: `scripts/run_mmpk_benchmark.py` (~250 lines)

1. Load `data/ml/clinical/mmpk_pbpk_features.csv` (1,128 drugs)
2. Exclude 30 holdout drugs (`mmpk_holdout_exclusions.json`)
3. Run fresh `OmegaPipeline.simulate()` on each drug (~700 × 73ms ≈ 51s)
4. Apply AD filter → in-domain subset
5. Stratify by study count: Tier 1 (n≥2) vs Tier 2 (all)
   - **Study count source:** `mmpk_quality_scored.csv:n_studies` field directly (NOT dose-uniqueness proxy)
6. Compute:
   - AAFE + bootstrap CI (10,000 resamples)
   - %2-fold, %3-fold
   - Spearman ρ: `scipy.stats.spearmanr(np.log10(pred_cmax), np.log10(obs_cmax))` — rank correlation of log10 Cmax predictions vs observations
   - Stratified results by MW / logP / study count
7. Output: `outputs/mmpk_benchmark_YYYY-MM-DD.json`

### Threshold
```python
threshold = max(2.0, baseline_aafe * 1.10)  # regression floor, not quality standard
```

### Mandatory contamination warning
All output files and test docstrings:
```
WARNING: ~75% of MMPK drugs are in the V2/meta-learner training set.
ka was Optuna-calibrated on 1,020 MMPK drugs.
This benchmark measures in-training-distribution behavior.
Generalization estimate: holdout N=71 only.
```

### Conditional: UGT AD flag
If Step 0c identified ≥ 3 UGT-primary candidates in holdout top-10 errors, add `PHASE2_PRIMARY` flag to `_check_applicability_domain()` as part of Day 3 work (5 lines, no model change):
```python
# Conjunction: structural UGT susceptibility AND near-zero CYP clearance
if has_ugt_susceptible_group and clint_3a4 < 1.0 and clint_2d6 < 1.0:
    flags.append("PHASE2_PRIMARY")
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
| Phase 0 ka confound test | Circular: predicted logS unreliable for BCS II → measurement target | Deferred to Phase 3 (dissolution context) |
| BCS classification in `_build_drug()` | Wrong predictions baked as metadata debt | Deferred to Phase 3 (with measured solubility) |
| Item 4 dissolution (43-state ODE) | 7 open blockers, ~0.03 AAFE gain, ka confound unmeasured | Phase 3 |
| Experiment D (nested CV) | Warranted only if meta-learner validated in Day 1 | Phase 3 (conditional on |C−A| ≥ 0.05) |
| CYP1A2/UGT feature expansion (1b) | Requires stable meta-learner decision first | Phase 3 |
| CYP1A2 spot-check | Data curation work (SMILES + reference for non-benchmark drugs) | Phase 3 |
| 5-arm ablation | Selector toggle doesn't exist; contaminated MMPK gives in-sample results | Phase 3 (if warranted by MMPK results) |
| Selector toggle + refactor | 130 hardwired lines; real engineering task | Phase 3 |

---

## Phase 3 Entry Criteria

Phase 3 begins when Phase 2B is complete AND at least one condition is met:

| Condition | Phase 3 Action |
|-----------|----------------|
| Day 1: \|C − A\| ≥ 0.05 on holdout | Experiment D (nested CV V2 OOF → meta-learner retrain) |
| Day 3: MMPK Tier 1 AAFE > 2.0 | Investigate specific drug classes from stratified output |
| Day 2: AUC AAFE > 3.5 | CLint investigation — re-evaluate fuinc correction for AUC specifically (note: CLAUDE.md Decision 18 found fuinc worsens Cmax AAFE; AUC may respond differently since AUC ∝ 1/CL directly) |
| Day 0c: ≥ 3 UGT-primary drugs in top-10 | Item 1b UGT AD flag + meta-learner feature |
| Any trigger: mechanistic hypothesis formulated | Dissolution pilot, pKa integration, or transporter modeling |

If no condition is met, the pipeline is at its practical ceiling for the current architecture. Next step is new data sources (PK-DB access, DrugBank), not more tuning.

---

## Success Criteria

| Metric | Current | Target | Source |
|--------|---------|--------|--------|
| Holdout AAFE (current) | 1.847 (stale) | Measured | Day 0 |
| Gold-24 AAFE | Unknown (stale) | Measured, ≤ 2.0 | Day 0 |
| AUC validated drugs | ~2 | ≥ 50 | Day 2 |
| AUC AAFE baseline | Unknown | Measured | Day 2 |
| MMPK Tier 1 AAFE | Unknown | Measured | Day 3 |
| Spearman ρ (log10 Cmax ranking) | Never measured | Measured | Day 3 |
| Meta-learner verdict | Unknown | A/B/C comparison | Day 1 |

All targets are "measured" — this phase establishes baselines, it does not claim improvements.

---

## Self-Review Corrections Applied

Issues found during convergence self-review and corrected in this document:

1. **Feature importance numbers were swapped** in previous version (said 25.6% / 60.2%; actual: 49.9% / 25.6%). Fixed.
2. **AUC0t vs AUC0inf gap** — added AUC comparison methodology section specifying dose normalization, AUC type annotation, and sim duration flag.
3. **Phase 3 fuinc contradiction** — CLAUDE.md Decision 18 says fuinc worsens Cmax AAFE. Phase 3 criteria now notes this and specifies "re-evaluate for AUC specifically" since AUC ∝ 1/CL may respond differently.
4. **Threshold formula inconsistency** — AUC and MMPK thresholds now both use `baseline × factor` for regression detection. AUC quality concern (> 3.0) is a separate flag, not conflated with regression gate.
5. **n_studies source** — explicitly specified as `mmpk_quality_scored.csv:n_studies` (not dose-uniqueness proxy).
6. **Spearman ρ target** — specified as `log10(predicted Cmax) vs log10(observed Cmax)`.
7. **UGT flag timing** — moved from "Day 1" to "Day 3" (alongside MMPK script work).
8. **"Zero model changes" framing** — corrected to "zero permanent model changes; diagnostic flag toggles only."
9. **"Kill" vs "Deferred"** — items needed for Phase 3 changed from "Kill" to "Deferred to Phase 3."
