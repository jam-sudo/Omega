# Omega PBPK — Next Phase Design Spec

> **Date:** 2026-03-15
> **Status:** Approved
> **Strategy:** Multi-metric validation → Systematic fixes → Pragmatic L3 → Publish

---

## 1. Context

### Current State
- **L1/L2 PASS**: Cmax AAFE 1.90, AUC AAFE 1.66, %2-fold 70%, 73ms (OmegaPipeline + GSE fix)
- **GNN L2 failed**: distillation ceiling — can't beat the teacher (OmegaPipeline)
- **L3 architecture**: complete code (1500 LOC), no trained weights, no patient data
- **Benchmark**: 25 drugs with C(t) curves, 20 used for validation
- **Clinical data**: OpenFDA 118 params from 43 drugs (but only 5 have Cmax+AUC)
- **ADME reference**: 153 compounds with SMILES + 7 ADME properties
- **CI**: 48,671 tests, 0 failures

### Key Data Reality
OpenFDA extraction yields only **3 new drugs** with both Cmax+AUC. The "expand to 50+ drugs" assumption was wrong. However:
- **39 drugs** have t_half → Silver-tier validation
- **20 drugs** have bioavailability → ADME validation
- **153 compounds** in adme_reference.csv → Bronze-tier property validation

### Known Outliers
| Drug | Fold-error | Root cause |
|------|-----------|------------|
| verapamil | 8.8x Cmax | P-gp efflux not modeled |
| ibuprofen | 5.0x Cmax | High protein binding (fup prediction) |
| phenytoin | 4.7x Cmax | Nonlinear (saturable) metabolism |

---

## 2. Architecture

### 3-Tier Validation Strategy

```
Gold Tier (28 drugs)        → Cmax + AUC fold-error
  └─ 25 existing benchmark C(t) curves
  └─ 3 new OpenFDA drugs (ciprofloxacin, itraconazole, sitagliptin)

Silver Tier (~40 drugs)     → t_half fold-error
  └─ 39 OpenFDA drugs with extracted half-life
  └─ Deduplicated with Gold tier

Bronze Tier (153 compounds) → ADME property accuracy
  └─ adme_reference.csv (fup, clint, peff, rbp, logP, logS)
  └─ Per-property AAFE + coverage metrics
```

### Pragmatic L3 Architecture

Instead of Neural L3 (GNN + CrossAttention + Reptile), implement clinically standard approach:

```
OmegaPipeline (population PK)
    │
    ├─→ Allometric scaling (deterministic)
    │     CL_ind = CL_pop × (W/70)^0.75 × genotype_factor
    │     Vd_ind = Vd_pop × (W/70)^1.0
    │
    └─→ Bayesian individual estimation (1-5 observations)
          scipy.optimize → (CL_scale, Vd_scale) to fit C(t) points
          └─→ Individual C(t) prediction
```

Existing Neural L3 code preserved for future activation when real patient data is available.

---

## 3. Workstreams

### WS0: Infrastructure & Cleanup (1 day)

**Purpose:** Clean repo state + build automated benchmark runner (prerequisite for all other WS).

| Task | Details |
|------|---------|
| 0.1 Commit uncommitted | README, scripts, admet_ai_wrapper.py |
| 0.2 Gitignore models | `models/**/*.pt` → .gitignore (binary blobs) |
| 0.3 Auto benchmark script | `scripts/run_full_benchmark.py` — runs 25 drugs, outputs JSON with per-drug fold-errors + aggregate AAFE, compares to previous run |
| 0.4 ODE mass balance | Fix remaining 1/3 of 0A bug |
| 0.5 Memory cleanup | Remove stale memory files, update MEMORY.md |

**Exit criteria:** `git status` clean, `run_full_benchmark.py` executes successfully and produces `outputs/benchmark_YYYY-MM-DD.json`.

### WS1: Multi-Metric Validation (2-3 days)

**Purpose:** Expand validation beyond 20 drugs using available data at multiple fidelity tiers.

| Task | Details |
|------|---------|
| 1.1 SMILES mapping | Map 43 OpenFDA drug names → canonical SMILES via PubChem + adme_reference.csv cross-reference |
| 1.2 Unit normalization | Convert OpenFDA values → mg/L and mg·h/L using MW. **Manual curation required** for Gold drugs — see Appendix A. For Silver (t_half in hours), no conversion needed. Value selection policy: prefer fasted, single-dose, healthy volunteer, IR formulation. When multiple values exist, use geometric mean |
| 1.3 Pipeline execution | Run OmegaPipeline on all mapped drugs |
| 1.4 Gold report | 28 drugs: Cmax AAFE, AUC AAFE, %2-fold. Dev set (20) vs Validation set (8) separately |
| 1.5 Silver report | ~40 drugs: t_half AAFE, %2-fold |
| 1.6 Bronze report | 153 compounds: per-property AAFE (fup, clint, peff, rbp, logP, logS) |
| 1.7 Validation CSV | `data/ml/clinical/validation_set.csv` — drug, SMILES, dose_mg, route, observed metrics, units, source |

**Exit criteria:** 3-tier validation report saved to `outputs/multi_metric_validation.json`. Gold/Silver/Bronze AAFE all reported.

### WS2: Systematic Failure Analysis (3-5 days, after WS1)

**Purpose:** Categorize and fix systematic prediction failures.

| Task | Details |
|------|---------|
| 2.1 Error classification | All >3-fold errors → categorize: solubility / protein binding / transporter / metabolism / formulation |
| 2.2 Priority fixes | Fix by frequency. Drug-agnostic corrections only (no drug-specific hacks) |
| 2.3 ADMET-AI exploration | Per-drug ADMET-AI on/off analysis — find which drugs benefit from ADMET-AI predictions |
| 2.4 Regression testing | Every fix → `run_full_benchmark.py` → before/after diff |
| 2.5 Transporter flagging | Flag known P-gp/OATP substrates using `data/transporter_reference.csv` + ADMET-AI `Pgp_Broccatelli` predictions. For flagged drugs, apply fa correction factor (×0.3-0.5 for strong P-gp substrates). Integration: add `transporter_flag` field to Drug, apply correction in `_predict_adme()`. **Mini-design required before implementation** |

**Constraints:**
- No global parameter changes without full regression pass
- Drug-agnostic corrections only (must improve ≥2 drugs without regressing any)
- Track all changes in `outputs/fix_log.json`

**Fallback:** If drug-agnostic corrections cannot reduce >3-fold errors to ≤5, reclassify mechanistic outliers (P-gp substrates, saturable metabolism) as "expected limitations" with documented root causes. Report both "all drugs" and "excluding known-mechanism outliers" AAFE.

**Exit criteria:** ≤5 drugs with >3-fold Cmax error in 25-drug benchmark, OR all >3-fold errors have documented mechanistic root causes. Regression test green.

### WS3: Validation Framework (2-3 days, parallel with WS1)

**Purpose:** Complete the validation infrastructure and run all tiers.

| Task | Details |
|------|---------|
| 3.1 T8 Confidence calibration | Run `run_validation.py` T8 on scaffold holdout. Target: 90% CI coverage ≥ 88%, confidence monotonic |
| 3.2 T9 Structural analogs | SAR consistency, monotonicity, plausibility bounds |
| 3.3 T10 De novo | 1000 novel SMILES, physical plausibility checks (mass balance, positive concentrations, monotonic terminal). Note: ~22min at 1.3s/drug ADMET-AI latency; run with `admet_ai=False` for speed, or batch |
| 3.4 Temporal holdout | Curate 5-10 post-2023 FDA-approved small molecules with published PK. Candidates: lenacapavir (2022), futibatinib (2022), adagrasib (2022), elacestrant (2023), capivasertib (2023). Run OmegaPipeline blind. Report fold-errors |
| 3.5 Comprehensive report | `outputs/validation_report.md` — all 4 sub-tiers |

**Exit criteria:** T8/T9/T10/temporal holdout all executed and reported. Calibration coverage ≥ 88%.

### WS4: Production & Docs (parallel with WS1-3)

**Purpose:** Make Omega usable and presentable.

| Task | Details |
|------|---------|
| 4.1 README update | Latest benchmarks (AAFE 1.90), 3-tier validation summary |
| 4.2 Docs bulk update | physio_sim.cli → omega, 34-state → 35-state across all docs |
| 4.3 CLI verification | `omega predict <SMILES>` smoke test, fix if broken |
| 4.4 CI benchmark | Add benchmark as `pytest -m benchmark` in CI (with timeout) |
| 4.5 Cold start note | Document warm (73ms) vs cold (~5s) startup in README |

**Exit criteria:** README accurate, docs consistent, CLI works, CI includes benchmarks.

### WS5: Pragmatic L3 (5 days, after WS1-3)

**Purpose:** Add patient-specific PK prediction using clinically standard methods.

| Task | Details |
|------|---------|
| 5.1 Design doc | `docs/l3_pragmatic_design.md` — architecture, equations, validation plan |
| 5.2 Allometric module | `src/omega_pbpk/ml/models/foundation/covariate_scaling.py` — weight/age/sex/genotype → parameter corrections. See Appendix B for genotype factor table |
| 5.3 Bayesian fitting | `src/omega_pbpk/ml/models/foundation/individual_estimation.py` — scipy.optimize on sparse C(t) observations |
| 5.4 Pipeline integration | Extend `SimulationRequest` dataclass with optional fields: `cyp2d6_phenotype`, `cyp2c9_genotype`, `cyp2c19_phenotype`, `egfr_ml_min`. Existing `subject_weight_kg` and `age_years` fields are already present. Add new method `OmegaPipeline.fit_individual(request, observations)` for Bayesian fitting (keeps `simulate()` API unchanged) |
| 5.5 Demo | warfarin: 70kg→40kg→100kg, CYP2C9 *1/*3 → show PK change |
| 5.6 Neural L3 preservation | Keep existing code, add note in module docstring about activation path |

**Exit criteria:** `SimulationRequest(smiles=..., subject_weight_kg=40)` produces weight-adjusted PK. `pipeline.fit_individual(request, observations=[(1.0, 0.5), (4.0, 0.3)])` returns fitted PK. 1-drug demo with ≥3 covariate scenarios.

**Note:** Gold-tier validation uses 25 drugs with time-resolved C(t) curves + 3 drugs with summary PK values (Cmax/AUC from FDA labels). These are different fidelity levels — report separately if results diverge.

---

## 4. Sequencing

```
WS0 (infra, 1d) ──────────────────────────────────
        │
        ├──→ WS1 (multi-metric, 2-3d) ──→ WS2 (failure analysis, 3-5d)
        │                                         │
        ├──→ WS3 (validation, 2-3d) ──────────────┤
        │         [parallel with WS1]              │
        ├──→ WS4 (docs/prod) ─────────────────────┤
        │         [parallel with WS1-3]            │
        └─────────────────────────────────→ WS5 (L3, 5d, last)
```

Total: ~14-17 working days if serialized. ~10-12 days with parallelism.

---

## 5. Risks

| # | Risk | Severity | Mitigation |
|---|------|----------|------------|
| 1 | OpenFDA unit inconsistency | Medium | MW-based conversion + manual verification per drug |
| 2 | Fixes cause regressions | High | Automated benchmark runner (WS0.3), before/after diff |
| 3 | Temporal holdout drug scarcity | Medium | 5 drugs sufficient; FDA approval packages are public |
| 4 | Gold-tier limited to 28 drugs | Medium | Silver (39) + Bronze (153) provide breadth |
| 5 | Pragmatic L3 seen as trivial | Low | Frame as "clinically validated approach"; Neural L3 on roadmap |
| 6 | Scope creep in WS2 | High | Strict exit criteria; no drug-specific hacks |

---

## 6. Success Metrics

| Metric | Current | Target |
|--------|---------|--------|
| Gold-tier drugs validated | 20 | 28 |
| Silver-tier drugs validated | 0 | ~40 |
| Bronze-tier compounds validated | 0 | 153 |
| >3-fold Cmax errors (25 drugs) | ~5-6 | ≤5 |
| Confidence calibration coverage | unknown | ≥88% |
| Temporal holdout drugs | 0 | 5-10 |
| L3 covariate support | none | weight, age, CYP genotype |
| L3 few-shot (Bayesian) | none | 1-5 observations → individual PK |

---

## 7. Non-Goals

- Retrain GNN L2 (distillation ceiling proven)
- Full Neural L3 training (no individual patient data available)
- Docker/deployment (no users yet)
- TDC leaderboard submission (premature without broader validation)
- ChEMBL ETL (in vitro, low ROI — per CLAUDE.md)
- Consolidate/refactor legacy phase files (per CLAUDE.md)

---

## 8. Future Work (Post This Phase)

- **Neural L3 activation** — when real individual patient C(t) data is obtained (PK-DB studies, academic collaboration)
- **Multi-dose support** — steady-state predictions, accumulation
- **IV route** — currently oral-only validation
- **TDC leaderboard** — after validation is comprehensive
- **Publication** — after multi-metric validation + comparison to Bayer 2024

---

## Appendix A: Gold Drug Unit Normalization

Manual curation for the 3 new Gold-tier drugs from OpenFDA:

| Drug | OpenFDA Cmax | Unit | MW | → mg/L | OpenFDA AUC | Unit | → mg·h/L | Dose | Source context |
|------|-------------|------|----|--------|-------------|------|----------|------|----------------|
| ciprofloxacin | TBD | TBD | 331.3 | TBD | TBD | TBD | TBD | 500mg oral | FDA label |
| itraconazole | TBD | TBD | 705.6 | TBD | TBD | TBD | TBD | 200mg oral | FDA label |
| sitagliptin | 950 | nM | 407.3 | 0.387 | 8.52 | µM·hr | 3.47 | 100mg oral | FDA label |

**Conversion formulas:**
- nM → mg/L: `value × MW / 1e6`
- µM·hr → mg·h/L: `value × MW / 1e3`
- ng/mL → mg/L: `value / 1e3`
- µg/mL → mg/L: `value` (same unit)

---

## Appendix B: CYP Genotype Scaling Factors

Genotype-to-CL scaling factors for allometric module (WS5.2). Based on FDA pharmacogenomic guidance and published meta-analyses.

| Enzyme | Phenotype | Factor (×CL) | Reference |
|--------|-----------|-------------|-----------|
| CYP2D6 | Ultra-rapid (UM) | 1.5 | Kirchheiner 2004 |
| CYP2D6 | Extensive (EM) | 1.0 | (reference) |
| CYP2D6 | Intermediate (IM) | 0.5 | Kirchheiner 2004 |
| CYP2D6 | Poor (PM) | 0.1 | Kirchheiner 2004 |
| CYP2C9 | *1/*1 | 1.0 | (reference) |
| CYP2C9 | *1/*2 | 0.8 | Rettie 2000 |
| CYP2C9 | *1/*3 | 0.6 | Rettie 2000 |
| CYP2C9 | *2/*2 | 0.5 | Rettie 2000 |
| CYP2C9 | *2/*3 | 0.35 | Rettie 2000 |
| CYP2C9 | *3/*3 | 0.1 | Rettie 2000 |
| CYP2C19 | UM | 1.5 | Sim 2006 |
| CYP2C19 | EM | 1.0 | (reference) |
| CYP2C19 | IM | 0.6 | Sim 2006 |
| CYP2C19 | PM | 0.2 | Sim 2006 |

**Application:** `CL_ind = CL_pop × (W/70)^0.75 × CYP_factor`. The CYP factor applies only to the fraction of clearance mediated by that enzyme. For drugs with mixed CYP metabolism: `CL_ind = CL_pop × (fm_3A4 × factor_3A4 + fm_2D6 × factor_2D6 + (1-fm_3A4-fm_2D6))`.

**fm data source:** ADMET-AI does not predict fm per isoform. Initial approach: use ADMET-AI CYP substrate probability as a proxy (e.g., `CYP2D6_Substrate_CarbonMangels > 0.5` → fm_2D6 = 0.3, else 0.0). CYP3A4 is assumed dominant for remaining hepatic clearance (fm_3A4 = 1.0 - fm_2D6). Simplification acceptable for v1; refine in WS5.1 design doc if needed.

---

## Appendix C: WS3↔WS2 Dependency Note

WS3 T8 (confidence calibration) may reveal that confidence labels are not monotonic, which could inform WS2 parameter fixes. The sequencing diagram shows WS3 parallel with WS1, but T8 results should be shared with WS2 as they become available. Specifically:
- If T8 reveals fup predictions have poor coverage → WS2 should prioritize protein binding fixes
- If T8 reveals clint predictions are well-calibrated → WS2 can deprioritize metabolism fixes

This is an information dependency, not a blocking dependency. WS2 can start without T8 results but should incorporate them when available.
