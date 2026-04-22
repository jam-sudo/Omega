# Phase 2B Consensus — 2-Round Expert Debate Results

> **Date:** 2026-03-22
> **Participants:** Systems architect, computational biologist, chemical engineer, ML engineer (+ 3 devil's advocates)
> **Rounds:** 2 (constructive debate → devil's advocate stress-test)
> **Input spec:** `docs/superpowers/specs/2026-03-22-phase2b-design.md`
> **Status:** Consensus document. Original spec needs revision to incorporate these findings.

---

## Structural Findings (Debate-Originated)

### 1. Hybrid selector confounds component-level diagnosis

The hybrid selector's ratio-dependent weighting (ODE_Cmax / analytical_Cmax → blend weight) transforms any upstream change — dissolution rate, Fg, Kp — into an unpredictable downstream effect. When an individual component fix worsens AAFE, it is impossible to determine whether the component itself is wrong or whether the selector's ratio-dependent weighting amplified the change unexpectedly.

This reframes the "8/8 individual fixes worsened AAFE" finding: the failures may not be due to error cancellation in the pharmacokinetic components themselves, but to the selector acting as a nonlinear amplifier of upstream perturbations.

**Important:** The selector is NOT confirmed as the primary barrier — it is a **candidate**. The gold-24 ablation (+0.278 AAFE without selector) proves it adds value in-sample. Whether it adds value on holdout is unknown. The MMPK N=700 ablation (see Phase 1 below) will resolve this.

**Decision #3/#14 (selector is essential) is NOT revoked.** Any future revocation must cite mechanistic justification (Optuna artifact on synthetic CSV), not underpowered holdout statistics.

### 2. ka_scale = 0.0004 creates a hidden dependency chain

Optuna E2E calibrated ka_scale on 1,020 MMPK drugs under instant-dissolution assumption. For BCS II drugs in the training set (~20%), ka implicitly absorbed dissolution delays. This creates three downstream problems:

- **Dissolution double-correction:** Adding Noyes-Whitney without recalibrating ka double-slows BCS II drugs
- **Meta-learner contamination:** `log_cmax_pbpk` (25.6% feature importance) carries ka-biased ODE predictions for MMPK training drugs. Nested CV (Experiment D) fixes V2 leakage but leaves ODE-feature contamination intact
- **Execution order dependency:** Meta-learner validation on BCS II drugs is unreliable until ka is recalibrated post-dissolution

**This is measurable in 30 minutes:** Re-run Optuna with BCS II drugs excluded from the objective. If Δka < 10%, the confound is within noise. If Δka ≥ 10%, recalibration is required before dissolution deployment.

### 3. In-sample metrics cannot justify out-of-sample decisions

The CYP1A2 "non-bottleneck" conclusion (MMPK AAFE 1.67) was based on contaminated in-sample performance — the same contamination the consensus criticizes for MMPK evaluation. Using in-sample metrics to drop coverage features while simultaneously arguing that in-sample metrics are unreliable is contradictory.

**Rule:** Any decision to drop a coverage feature must be based on holdout evaluation or a spot-check on non-benchmark drugs, not MMPK in-sample AAFE.

### 4. Reduced dose approach cooperates with hybrid selector; full ODE fights it

The hybrid selector blends ODE Cmax and analytical Cmax based on their ratio. Full ODE dissolution changes ODE Cmax but not analytical Cmax → ratio shifts → blend weights change unpredictably. The pre-ODE reduced dose approach (effective_dose = dose × fraction_dissolved) feeds both paths the same reduced dose → ratio stays stable → selector interference minimized.

However: reduced dose produces physically wrong tmax (drug appears fully available at t=0 despite dissolution taking hours). For Cmax-only evaluation this is acceptable; for PK/PD modeling with temporal profiles it is not.

**Decision depends on ka test result:** If ka confound < 10%, full 43-state ODE is viable (43-state implementation difficulty was overstated in Round 1 — existing `dissolution_absorption_coupling.py` has the coupled ODE, and body.py's index management is cleanly appendable). If ka confound ≥ 10%, reduced dose is the safer Phase 2B fallback while ka recalibration proceeds in Phase 3.

---

## Original Spec vs Final Consensus

| Spec Element | Original Spec | Round 1 Consensus | Round 2 Stress-Test |
|---|---|---|---|
| **Top priority** | Item 2 (MMPK benchmark) | Item 0 (holdout ablation) | **30-min cheap measurements** (ka confound + BCS audit + CYP1A2 spot-check) |
| **Selector judgment** | Not mentioned | "candidate barrier to improvements" | **"confounds component-level diagnosis"** — diagnostic purpose, not removal. Decision #3/#14 retained |
| **Ablation location** | N/A | Holdout N=53 (standalone) | **MMPK N=700** (merged into Item 2, statistical power) |
| **Ablation metric** | N/A | AAFE comparison | **%2-fold (McNemar's test)** primary, AAFE secondary. CI overlap on holdout = always keep selector |
| **Item 4 dissolution** | 43-state ODE, Phase 2B | Phase 3 full deferral | **ka test → data-driven:** ka<10% → full ODE Phase 2B; ka≥10% → Phase 3 |
| **Dissolution approach** | Full ODE only | Reduced dose only | **ka<10% → full 43-state ODE** preferred (reduced dose as fallback for ka≥10%) |
| **Dissolution formula** | `k_diss = 3D/(ρr²)` (wrong units) | Use `dissolution.py` | **Use `dissolution.py` shrinking sphere** (bcs_classification.py formula has dimensional error: units are cm³/(g·s), not 1/s) |
| **Dissolution guard** | Binary: dose_number > 2.0 | Same | **Soft flag:** dose_number ∈ [0.5, 2.0] → `DISSOLUTION_UNCERTAIN` AD flag |
| **Dissolution scope** | All BCS II/IV | Non-ionizable only | **Non-ionizable only** until pKa predictor is integrated (pH-dependent Cs without pKa is garbage) |
| **Pilot drugs** | Nifedipine, carbamazepine, griseofulvin, danazol, phenytoin | Phenytoin/ibuprofen removed | **Griseofulvin, danazol, felodipine** (all require IR formulation + reference data verification pre-pilot) |
| **Optuna recalibration** | Not mentioned | Not mentioned | **Mandatory after dissolution deployment** (ka re-optimization on BCS I/III drugs) |
| **Item 1a decision rule** | p > 0.1 = overfitting | ΔAAFE < 0.05 | **D vs A on holdout** (primary gate). C quantifies leakage only. Report full distribution (improved/worsened/unchanged per drug) |
| **Item 1a evaluation set** | MMPK CV | Holdout | **Holdout (primary)**. MMPK for directional reference only |
| **CYP1A2** | SMARTS `[nH]1cccc1` + curated list | Completely dropped | **Curated list (30 drugs) retained** as meta-learner feature + 5 non-benchmark spot-check before deciding on coverage expansion |
| **CYP1A2 SMARTS** | `[nH]1cccc1` (pyrrole) | Dropped | **Dropped** (pyrrole ≠ CYP1A2 pharmacophore; matches beta-lactams, indoles, porphyrins) |
| **UGT feature** | `is_ugt_substrate` meta-learner feature | AD flag primary | **AD flag (`PHASE2_PRIMARY`) primary;** meta-learner feature secondary with 0.5% abort. Test whether CLint threshold alone catches same drugs as SMARTS+CLint conjunction |
| **Item 1b ordering** | Before Item 4 | Before Item 4 | **After Item 4 + Optuna recalibration** (dissolution changes cmax_pbpk distribution → meta-learner needs stable ODE output) |
| **MMPK benchmark label** | "primary metric" | "training distribution gate" | **"training distribution gate"** with full contamination stack documented (ka/V2/meta-learner all MMPK-fitted) |
| **MMPK threshold** | AAFE ≤ 2.0 (fixed) | ≤ baseline × 1.05 | **max(2.0, baseline × 1.10)** — regression floor, not quality standard |
| **MMPK contamination note** | Not mentioned | Required | **Mandatory in all outputs:** "~75% of drugs in V2/meta-learner training set. Generalization estimate: holdout only." |
| **AUC validation** | Multi-source collection | MMPK primary source | **MMPK primary** (1,223 drugs with AUC), dose-normalized comparison required. AUC is cleaner than Cmax for contamination (CLint/Kp/CLh are physics-based, not MMPK-fitted) |
| **AUC threshold** | ≤ 3.0 | Baseline × 1.2 | **Measure baseline first.** If baseline < 2.0: threshold = baseline × 1.2. If baseline ≥ 2.0: threshold = 2.5 absolute |
| **tmax validation** | Not mentioned | Not mentioned | **Explicitly excluded for BCS II drugs** under reduced-dose approach. Document as known limitation |
| **In-domain count target** | 53 → 70 | Removed (no mechanism) | **Removed** — confirmed no Item achieves this |

---

## Final Execution Order

```
Phase 0: Cheap measurements (half-day, no code changes)
├── ka confound test (30 min)
│   Re-run Optuna excluding BCS II drugs from objective
│   Measure Δka_scale vs current 0.0004
│   Decision: < 10% = noise, ≥ 10% = real confound
│
├── BCS II holdout audit (2 hours)
│   Classify all 71 holdout drugs by BCS class
│   Count BCS II/IV with dose_number > 2.0
│   Check directional error: is Cmax over-predicted?
│   Abort: < 3 qualifying drugs → Item 4 dropped entirely
│
└── CYP1A2 spot-check (2 hours)
    Run pipeline on 5 known CYP1A2 substrates NOT in any benchmark
    If AAFE > 3.0 → coverage problem, add CYP1A2 AD flag
    If AAFE < 2.5 → non-issue, close the question

Phase 1: Infrastructure + Diagnostics (1-2 days, parallel)
├── Item 2: MMPK benchmark script
│   ├── N=700 fresh predictions
│   ├── 5-arm ablation (ODE-only / +selector / +meta-learner / +geomean / full)
│   ├── Tier 1 (multi-study n≥2) + Tier 2 (all) + bootstrap CI
│   ├── Contamination labels in all outputs
│   └── Threshold: max(2.0, baseline × 1.10)
│
├── Item 1a: Meta-learner verification
│   ├── Experiments A (geomean), B (fixed-weight), D (nested-CV) on holdout
│   ├── Primary gate: D vs A on holdout ΔAAFE > 0.05
│   ├── Stratify by in-domain vs OOD subsets
│   ├── Report per-drug improved/worsened/unchanged counts
│   └── C (production) used for leakage quantification only (C − D gap)
│
└── Item 3: AUC validation (fully independent)
    ├── MMPK AUC cross-reference (1,223 drugs, unit conversion ng·h/mL → mg·h/L)
    ├── Dose-normalized comparison (AUC/dose)
    ├── Platinum reference enrichment (≥50 drugs)
    └── Threshold: measure baseline, then set per formula

Phase 2: Dissolution (gated on Phase 0 results)
├── Gate check:
│   ├── ka confound < 10% AND ≥ 3 BCS II holdout drugs? → proceed
│   ├── ka confound ≥ 10% AND ≥ 3 drugs? → Phase 3 (ka recalibration first)
│   └── < 3 qualifying drugs? → Item 4 dropped
│
├── Implementation (if gated):
│   ├── ka < 10%: full 43-state ODE with dissolution.py shrinking sphere
│   ├── ka ≥ 10%: reduced dose fallback (pre-ODE fraction_dissolved)
│   ├── Non-ionizable drugs only (pH-dependent Cs needs working pKa)
│   ├── Pilot: griseofulvin, danazol, felodipine (IR formulation verified)
│   ├── Pilot gate: ≥ 3/5 drugs improved (pre-screen for reference data availability)
│   └── tmax validation explicitly excluded for BCS II (known limitation)
│
└── Post-deployment:
    └── Optuna ka recalibration on BCS I/III drugs (mandatory)

Phase 3: Meta-learner retrain (after stable ODE output)
├── Item 1b: retrain on post-dissolution, post-recalibration ODE output
├── Features: 12 existing + CYP1A2 curated list + UGT AD flag
├── UGT: test CLint-only vs SMARTS+CLint conjunction (if same recall → drop SMARTS)
├── 0.5% importance abort criterion for new features
└── Final holdout evaluation
```

---

## Blocking Issues Identified (must fix in spec before implementation)

| # | Issue | Source | Fix |
|---|-------|--------|-----|
| 1 | Noyes-Whitney formula dimensional error (cm³/(g·s) ≠ 1/s) | comp-bio + chem-eng | Use `dissolution.py` shrinking sphere, not bcs_classification.py |
| 2 | GI segment volumes undefined in ODE | chem-eng | Define V_segment constants or use fixed 31 mL/segment |
| 3 | Item 1a decision rule logically inverted (p>0.1 ≠ equivalence) | ml-eng | Primary: D vs A on holdout. C for leakage quantification only |
| 4 | Item 1b ordered before Item 4 (trains on wrong ODE output) | chem-eng-devil + ml-eng-devil | Reorder: 4 → Optuna recal → 1b |
| 5 | ka confound unmeasured but used to descope Item 4 | skeptic-bio | Measure first (30 min Optuna re-run), then decide |
| 6 | Dissolution deployment missing Optuna recalibration | chem-eng-devil | Add to Item 4 deliverables |
| 7 | MMPK threshold set before baseline measured | skeptic-bio + ml-eng-devil | Formula: max(2.0, baseline × 1.10) |
| 8 | CYP1A2 dropped based on contaminated MMPK metric | ml-eng-devil + skeptic-bio | Spot-check 5 non-benchmark drugs before deciding |
| 9 | Pilot drug list not pre-screened for reference data | chem-eng-devil | Verify IR formulation + reference Cmax before engineering work |
| 10 | Ibuprofen is BCS I at intestinal pH (not BCS II) | comp-bio | Remove from any pilot list |
| 11 | Phenytoin has nonlinear PK (Michaelis-Menten) | comp-bio | Hard exclude from all dissolution evaluation |
| 12 | tmax scope gap for BCS II drugs | chem-eng-devil | Explicitly exclude from validation, document as limitation |

---

## What the Debate Proved About the Pipeline

1. **The pipeline is more contaminated than documented.** ka, V2, meta-learner are all MMPK-fitted. Only ADME XGBoost models (CLint/fup/VDss from TDC) and Berezhkovskiy Kp (physics) are genuinely independent. The 71-drug scaffold holdout is the ONLY clean generalization estimate.

2. **Error cancellation operates THROUGH the hybrid selector**, not just through Fg/Fh coupling. The selector's ratio-dependent weighting is a nonlinear amplifier that makes upstream perturbation analysis unreliable. This reframes the "8/8 fixes failed" narrative.

3. **Cheap measurements before expensive implementations.** Three 30-minute tests (ka confound, BCS audit, CYP1A2 spot-check) would have prevented weeks of wasted engineering on Item 4 in the original Phase 2 plan. The lesson: measure the assumption before building on it.

4. **The meta-learner's apparent value (0.026 AAFE) is an in-sample artifact** until proven otherwise on holdout. 86.8% of its feature importance comes from blending two in-sample predictions. Experiment D (nested CV) on holdout is the only honest test.

5. **Coverage expansion requires out-of-sample evaluation by definition.** You cannot assess whether CYP1A2/UGT/BCS II coverage matters using drugs already in the benchmark. Spot-checks on non-benchmark drugs are the minimum honest evaluation.
