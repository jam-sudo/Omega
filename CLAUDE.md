# Omega — Project Instructions

> **Active plan:** `docs/superpowers/plans/2026-03-17-omega-v7-scientific-rigor.md`
> **Progress tracker:** `memory/MEMORY.md`
> **Plans archive:** `docs/superpowers/plans/`
> **Memory path:** `~/.claude/projects/-home-jam-Omega/memory/`
> **Auto-loaded every conversation. Source of truth for all sessions.**

---

## Vision

Omega's ultimate goal is a **full digital general human** — a computational model of human physiology that can simulate any molecule's journey through the body, predict therapeutic outcomes, and personalize treatment.

**Current stage:** PBPK (pharmacokinetic) prediction from molecular structure.
**Roadmap:** PK → PK/PD → Systems Pharmacology → Digital Twin → Digital General Human.

The current PBPK module is the foundation layer: SMILES in → PK profile + uncertainty intervals out.

### Pipeline Architecture
```
SMILES → EnsembleADMEPredictor (XGBoost CLint/fup/rbp/VDss + polynomial)
       → _build_drug (IVIVE scaling + Berezhkovskiy Kp + renal CL + P-gp correction)
       → 35-state ODE simulation → raw C(t) curve
       → PBPK/ML ensemble (DirectCmaxPredictor, confidence-weighted)
       → Conformal UQ (90% prediction intervals)
       → SimulationResult
```

### Honest Performance (2026-03-22, scaffold-stratified holdout)
- **Core-24 (clinical ref, selector OFF):** Cmax AAFE **1.977** [1.65, 2.43], 58% 2-fold
- **Holdout ALL (100 drugs):** Cmax AAFE **2.520** [2.14, 3.00], 49% 2-fold, 72% 3-fold
- **Holdout IN-DOMAIN (79 drugs):** Cmax AAFE **1.987** [1.79, 2.22], 57% 2-fold, **82% 3-fold**
- **Out-of-domain (21 drugs):** prodrugs, extreme lipophilic, P-gp efflux, DDI-boosted, HIGH_MW
- **Spearman ρ = 0.9379** (in-domain ranking correlation) — excellent for screening
- **CLint anchors NOT inflating:** ANCHORED 1.813 vs CLEAN 1.736 (delta +0.078)
- **Data leakage:** 36/107 (34%) gold tier drugs overlap with ADME training set
- **Error cancellation confirmed:** predicted ADME beats measured ADME — structural, not CLint-specific
- **Data quality >> model improvements:** 19 reference fixes = AAFE 3.520→1.987 (-44%), zero model changes
- **Benchmark CSVs are synthetic** (1-cpt generated, not real clinical C(t) data)
- **UQ intervals over-wide:** Cmax CI 97% coverage but median width 4880x; AUC/t½ CI broken

## Workflow

All work is on `main` branch. Sequential development with automated benchmarks.

### Session Startup
1. Read `memory/MEMORY.md` for current status
2. Run `python scripts/run_full_benchmark.py` to check baseline
3. Work on highest-priority task
4. Before ending: run benchmark + `pytest tests/ml/test_accuracy_regression.py -v`, update `memory/MEMORY.md`

### Team Structure
`/team` creates an Agent Team (via `TeamCreate`) — task에 맞는 역할만 동적 선택 (1-4명).
Teammates are independent sessions that communicate via `SendMessage` and share a task list.
Available roles: ml-engineer, infra-engineer, data-engineer, ci-auditor, domain-scientist, ode-engineer.
Details + cross-review protocol: `.claude/commands/team.md`

## Key Decisions (SETTLED — do not revisit)

1. **ADMET-AI disabled in production** — fup/logP changes break warfarin/metformin/losartan via Kp/Vd
2. **XGBoost CLint is primary** — 18 reference anchors at 50x weight (semi-supervised, partially circular)
3. **~~REVOKED~~ Hybrid Cmax selector is DISABLED** — ablation on synthetic CSV showed +0.278 AAFE, but holdout 100 drugs shows Δ-0.284 (HARMFUL). Selector was overfitted to N=24 synthetic benchmark. `use_hybrid_selector=False` since 2026-03-22.
4. **Don't replace ODE with pure ML** — v1-v5 GNN all failed (distillation ceiling)
5. **Don't touch phase files** — build ML alongside, deprecate later
6. **PK-DB + FDA labels** for clinical data; ChEMBL deprioritized
7. **~~REVISED~~ ODE >> Analytical 1-cpt for Cmax** — Holdout verification (2026-03-22): ODE AAFE 2.41 vs Analytical 11.64, ODE wins 81% of drugs. Original "analytical > ODE" (67% vs 38% 2-fold) was on synthetic CSV = overfitting. Pure ODE is default and correct.
8. **All benchmark CSVs are synthetic** — generated from 1-cpt model, not real clinical data. Warfarin replaced with PK-DB data (2026-03-17)
9. **External validation AAFE ~3.0** — in-sample 1.72 reflects tuning, not generalization
10. **Error cancellation is real** — predicted ADME beats measured ADME (2.46 vs 2.69). Fix ODE structure BEFORE improving ADME.
11. **Ridge correction is dead code** — not loaded in pipeline, zero contribution confirmed by ablation
12. **Gut CLint drives Cmax, not hepatic CLint** — Sobol: gut CLint ST=0.470, hepatic CLint ST=0.000 for Cmax. Hepatic CLint only affects AUC.
13. **Ridge correction is confirmed dead code** — ablation study (Phase 0.1) shows NO_RIDGE = FULL with Δ=0.000 AAFE. The ridge model file exists in models/correction/ but is never loaded at inference. Keep for reproducibility only.
14. **~~REVOKED~~ Hybrid Cmax selector was overfitted** — The Δ+0.278 ablation was measured on synthetic CSV benchmark (N=24). On holdout 100 drugs: selector WORSENS AAFE by 0.284. Selector LOO-CV tuned on N=24 synthetic data = classic overfitting. See #3.
15. **Error cancellation is systematic** — 79% of drugs (CI < 0.5), mean CI = 0.303. Fixing individual ADME params without joint balance will worsen aggregate AAFE.
16. **Phase 3a blocker is fm_CYP3A4 false positives, NOT Fh** — Polynomial clint_3a4 assigns fm_CYP3A4=0.887 to propranolol, 0.939 to ibuprofen. Fix: threshold guard `clint_3a4 > 2.0 µL/min/pmol` → AAFE 1.747, 83% 2-fold (diagnostic: 2026-03-18). Combined with acid-Kp D-fix → AAFE 1.665, 88% 2-fold.
17. **CLint_gut formula uses pre-inverted CLint** — `clint_L_per_h` is 22-223x larger than `CLh_target`; the 1.7× factor was calibrated for CLh_target. This is a known architectural bug, not a Phase 3a blocker. Fix in Phase 3b.
18. **fup calibration for low-fup drugs WORSENS AAFE** — isotonic regression improved individual fup accuracy but degraded AAFE +0.088 (fluconazole/atenolol rely on XGBoost fup under-prediction via error cancellation). Do not apply global low-fup calibration.
19. **CLint_gut K=1.7 is currently optimal — do NOT recalibrate to K=3.1** — Fitting K to Fg_lit data gives K_optimal=3.1 but AAFE worsens +0.094 (83%→79%). Root cause: pipeline fup_pred > fup_lit for CYP3A4 drugs (midazolam: 0.037 vs 0.024), so higher K is required to hit Fg_lit, but this over-corrects and breaks error cancellation. K recalibration requires fup fix first.
21. **Berezhkovskiy acid-Kp D-fix is correct and safe** — Using D (distribution coefficient at tissue/plasma pH) instead of P (neutral partition coefficient) for the neutral-lipid term in Berezhkovskiy Kp for acids. Fix applies to `compound_type="acid"` only (NOT bases/zwitterions). Dramatically corrects strong acids (ibuprofen pKa=4: Kp_adipose 8.4→0.67, fe_cmax 5.39→1.55x). AAFE 1.747→1.665, %2-fold 83%→88%, confirmed no regression on bases/neutrals. Implementation: `heuristics.py` `berezhkovskiy_kp` + `rodgers_rowland_kp`.
22. **Enol_lactone compound_type override works but limited by fup<0.01** — Warfarin enol_lactone→acid Kp fix: 6.95x→5.24x. Kp adipose dropped from 6.27→1.97 (correct), but overall Vd still inflated due to Berezhkovskiy overestimation at fup=0.009. AAFE 1.665→1.646.
23. **Adaptive sim time improves AUC but CLint remains bottleneck** — Fluconazole AUC 13.7→17.7 (extended to 68h), but observed=227.8. Remaining 12.9x gap is CLint over-prediction, not sim time. Triggers when t½ > duration/3.
24. **ML corrections (Pre-ODE + Post-ODE) worsen gold-24 AAFE** — Trained on 127-drug expanded set (AAFE ~2.9), corrections over-correct gold-24 drugs (already at 1.646). Gold-24 with ML: AAFE 2.69. Root cause: training data mismatch — model learns corrections for high-error drugs and applies them universally. Infrastructure is correct (integration validated); models need retraining with gold-tier leave-one-out CV. `pipeline.use_ml_corrections = True` is opt-in, default OFF.
20. **OATP correction disabled — wrong direction for atorvastatin** — Atorvastatin AUC is UNDER-predicted (fe=3.64×; pred=0.048 vs obs=0.176 mg*h/L), meaning CLint is already over-predicted. OATP adds more clearance → makes AUC worse. CLint already >>QH (near-complete extraction), so any CLint addition has minimal but harmful effect. Root cause: CLint over-prediction, not missing uptake transporter. Code archived in pipeline with `_ENABLE_OATP_CORRECTION = False`.
25. **CLint anchors do NOT inflate gold-24 metrics** — Anchor contamination analysis: ANCHORED AAFE 1.813 vs CLEAN 1.736 (delta +0.078). Error cancellation is structural (pipeline architecture), not CLint-specific.
26. **MLP cannot beat XGBoost at 1K-4K drug scale** — UDE Phase 1/2 (134K params MLP) achieved holdout AAFE 3.46-3.50 vs pipeline 3.52. Early stopping at epoch 5-8 = underfitting. Multi-dose data expansion (3.3x) WORSENED results due to noise. XGBoost remains superior.
27. **Data quality >> model improvements** — 19 platinum reference fixes + AD filter achieved AAFE 3.520→1.987 (-44%) on holdout in-domain with ZERO model changes. Single highest-ROI intervention.
28. **Applicability domain filter in pipeline** — `SimulationResult.in_applicability_domain` + `ad_flags`. SMARTS: val-ester, thienopyridine, pivoxil, nucleoside 5'-ester, quaternary amine, inorganic. Thresholds: logP>5.5, MW>700, P-gp efflux risk (MW>500+logP>3.5+TPSA>100). DDI-boosted flag in platinum reference.
29. **Permanent scaffold-stratified holdout** — 76 train / 100 holdout from platinum 176 (Murcko generic, seed=42). `data/clinical/holdout_split.json`. 30 MMPK drugs SMILES-match holdout → excluded from UDE training.
30. **torchdiffeq ODE is impractical for training** — 13-state PBPK ODE: 11s/drug forward+backward. 60 epochs = 160 hours. Need surrogate ODE approach for differentiable training.
31. **Pipeline structural gaps** — flutamide (CYP1A2 172x), buspirone (F=4% 44x), pantoprazole (enteric coating 5x). These are genuine mechanistic limitations, not data errors.
32. **All benchmarks must use clinical reference only** — Synthetic 1-cpt CSV benchmark deprecated (2026-03-22). Core-24 AAFE with clinical ref = 1.977 (was 1.502 on synthetic). Synthetic CSV inflated accuracy by ~0.5 AAFE.
33. **Optuna E2E constants do not generalize** — 5 Optuna-tuned constants (gut_threshold=0.97, peff_min=0.76, pgp=0.34, gse=1.11, ka_scale) hurt holdout by +0.091 AAFE. MMPK optimization doesn't transfer to platinum holdout.
34. **Overfitting has 3 layers** — (1) synthetic CSV benchmark, (2) hybrid selector LOO-CV on N=24, (3) Optuna E2E on MMPK. Removing all 3: holdout 3.064→2.690→2.520.
35. **Spearman ρ = 0.94 (in-domain)** — Pipeline ranking is excellent despite AAFE ~2.0. Screening applications viable. Kendall τ = 0.80. Binary high/low accuracy = 88%.
36. **UQ recalibrated (2026-03-22)** — Cmax: 93.7% coverage (in-domain), median width 20.6x (was 4880x). AUC/t½: heuristic scaling from Cmax q-value (q×1.35 for AUC, q×1.0 for t½). AdaptiveConformal recalibrated on 68 clean drugs, k=30.
38. **CYP3A4 ML classifier deferred** — TDC 670 compounds, test AUROC 0.634 → too low. Multi-CYP normalization partially works but unreliable. Zero holdout drugs trigger gut wall fix → no holdout improvement possible. Model saved for future use.
39. **ODE >> Analytical for Cmax on clinical data** — Holdout: ODE AAFE 2.41 vs Analytical 11.64, ODE wins 81%. KD#7 "analytical > ODE" was synthetic CSV artifact.
37. **AD filter catches prodrugs + DDI-boosted** — SMARTS: val-ester, thienopyridine, pivoxil phosphonate, nucleoside 5'-ester. Flags: PRODRUG, DDI_BOOSTED, EXTREME_LIPOPHILIC, HIGH_MW, PGP_EFFLUX_RISK, QUATERNARY_AMINE, INORGANIC. 21/100 holdout excluded.
40. **AUC AAFE 3.2 on holdout (32 drugs)** — 2x worse than Cmax (1.7). AUC Spearman ρ=0.77 (vs Cmax 0.94). Root cause: VDss over-prediction + CLint error compounding through ODE. AUC improvement requires better CL prediction, not Cmax tuning.
41. **VDss systematically over-predicted** — Lombardo cross-validation (17 drugs): VDss AAFE 3.71, Spearman ρ=0.27. XGBoost VDss AAFE 1.31 (94% 2-fold) vs Berezhkovskiy 4.11. Fix: weighted geometric mean (XGB^0.7 × Berez^0.3) for t½ → Core-24 AUC improved 2.344→2.142 (-8.6%). Cmax unchanged (ODE Kp preserved).

## Codebase Rules

- New ML code → `src/omega_pbpk/ml/`
- New ML tests → `tests/ml/`
- Screening code → `src/omega_pbpk/screening/`
- Analytical PK engine → `src/omega_pbpk/pipeline/pk_engine.py`
- Don't modify phase files (549 in core/, 481 in prediction/, 94 in clinical/)
- Preserve `ADMEProperties` contract exactly:
  - Props: mw, logP, logS, peff, fup, rbp, clint_3a4, clint_2d6, herg_ic50_uM
  - Units: clint=µL/min/pmol, peff=1e-4 cm/s, fup=0-1, rbp=0.5-3.0
  - Required: confidence ("low"/"medium"/"high"), conformal intervals
- Don't break existing 48K+ tests
- ODE engine (core/body.py) = training data oracle — keep accurate

## Known Limitations (from 50-iteration deep review, 2026-03-17)

### Structural Issues (Plan v7 targets)
- **pKa hardcoded to 7.0**: `pka_predictor.py` exists but pipeline doesn't use it. All drugs treated as neutral → ionization-dependent Kp (R&R) is effectively disabled. Basic drugs (propranolol pKa 9.5) get wrong Kp for lung/kidney.
- **Gut wall Fg ≈ 1.0**: `gut_clint_multiplier=1.0` + 18g gut mass → Fg=0.997 for midazolam (measured 0.44). CYP3A4 substrates have systematically overpredicted F.
- **Error cancellation**: Fg↑ × Fh↓ partially cancel for CYP3A4 drugs. Gold tier 1.70 is partly fortuitous; temporal holdout 3.12 is where cancellation breaks.
- **Salt form ignored**: `salt_form_prediction.py` exists but unconnected. BCS II drugs get free-base logS → 10-15x Cmax error for fluoxetine/verapamil.

### IVIVE & CLint Issues
- **CLint partially circular**: 12/24 benchmark drugs have 50x-weighted anchors back-calculated from clinical CL using same IVIVE formula
- **No fuinc correction**: Microsomal non-specific binding not accounted for → lipophilic CLint systematically underpredicted
- **CLint conformal coverage 89.4%** — improved with 11.24× calibrated multiplier (local conformal; Phase 3b.4 done)

### Statistical Issues (Phase 0 resolved, 2026-03-17)
- **Bootstrap CI now in benchmark**: AAFE 1.72 [1.49, 2.04] — Bayer 1.87 inside CI, comparison NOT significant ✓
- **Ridge correction confirmed dead**: not loaded in pipeline, zero ablation effect, dead code ✓
- **ER-stratified done**: No high-ER drugs in benchmark (all ER < 0.52). Low ER (1.72) ≈ Mid ER (1.74) — no artificial inflation ✓
- **34% data leakage**: 36/107 gold tier drugs in ADME training set — must report separately

### Other Known Issues
- **All benchmark CSVs are synthetic**: generated from 1-cpt model with 20% constant SD, not real clinical data
- **Confidence is constant**: with ADMET-AI disabled, all drugs get "medium" confidence
- **Hybrid selector is critical but hacky**: 130 lines of non-standard heuristics, no industry precedent
- **Vd fails for fup < 0.01**: Berezhkovskiy Kp overestimates, XGBoost VDss also 2-4x off (warfarin 6.69x)
- **No transporter modeling in ODE**: P-gp is binary peff correction, OATP/OCT2/OAT not modeled
- **No Phase II metabolism**: UGT, NAT2, SULT enzymes not represented
- **No dissolution model**: BCS II drugs assume pre-dissolved drug

## Exit Criteria

| Level | Criteria | Status |
|-------|---------|--------|
| **1** | `omega predict <SMILES>` → PK profile. ADME AAFE<3.0. PK ≤2-fold for ≥70% of 20+ drugs | **PASS** (core-24: 1.977) |
| **2** | SMILES→PK <500ms. AAFE<2.0 | **PASS** (73ms, core-24 1.977 in-sample) |
| **3** | Patient covariates. Few-shot (<5 obs) | **Prototype** (allometric + Bayesian) |
| **4** | Batch screening 1000+ molecules with UQ | **Done** (batch_predict + conformal CI, but UQ intervals over-wide) |
| **Ext** | External validation AAFE<2.5 on unseen drugs | **PASS** (holdout in-domain 79 drugs: AAFE 1.987) |
| **v7** | Bootstrap CI, ER-stratified, N=50+ holdout, scaffold-split | **PASS** (100-drug holdout, scaffold-stratified, bootstrap CI) |
| **v8** | Holdout in-domain AAFE<1.7, %2-fold>70% | **NOT MET** (1.987, 57.0%) |
| **v9** | Spearman ρ>0.90, UQ coverage 85-95% | **PARTIAL** (ρ=0.94 PASS, UQ 97% over-covered) |

## Tech Stack

| Purpose | Tool |
|---------|------|
| ADME prediction | xgboost (CLint, fup, rbp, VDss), polynomial (logP, logS) |
| Direct Cmax ML | xgboost (Morgan FP + RDKit descriptors) |
| Molecular features | rdkit (fingerprints, descriptors, SMARTS) |
| UQ | scipy (conformal prediction, LHS sampling) |
| Applicability domain | rdkit SMARTS (prodrug detection) |
| Benchmarks | PyTDC (CLint, fup training data) |
| Clinical data | PK-DB REST API, OpenFDA API |
| Screening | omega_pbpk.screening.batch (batch_predict + rank_results) |

## What NOT To Do

- Don't consolidate/refactor legacy phase files
- Don't build ChEMBL ETL (in vitro, low ROI)
- Don't replace ODE pipeline with pure ML (distillation ceiling proven)
- Don't add drug-specific heuristics to pipeline (already 30+ tunable parameters for 24 drugs)
- Don't tune pipeline on benchmark without external validation — in-sample metrics are misleading
- Don't report in-sample AAFE as generalization performance
- Don't merge without running `pytest tests/ml/test_accuracy_regression.py`
- Don't break ADMEProperties contract
- Don't change structural parameters (Kp, Fg, IVIVE) without running error cancellation monitor
- Don't improve ADME prediction without first checking if it breaks error cancellation (Phase 0.2 ablation)
- Don't report AAFE without bootstrap CI

## Build / Test / Lint Commands

```bash
source .venv/bin/activate
pytest tests/ -m "not slow and not benchmark" -q          # fast tests (~48K)
pytest tests/ml/test_accuracy_regression.py -v             # accuracy regression (5 validation drugs)
pytest tests/ -m benchmark -v --timeout=300               # full benchmarks
python scripts/run_full_benchmark.py                      # 24-drug Cmax/AUC benchmark (now includes bootstrap CI)
python scripts/run_holdout_benchmark.py                   # 71-drug permanent holdout benchmark
python scripts/check_ude_prerequisites.py                 # UDE prerequisite gate check (11 gates)
python scripts/run_measured_ablation.py                   # error cancellation check (measured vs predicted ADME)
python scripts/run_stratified_validation.py               # ER-stratified validation
python scripts/audit_reference_data.py                    # data leakage + pKa + salt form audit
ruff check .                                              # lint
ruff format --check .                                     # format check
omega --help                                              # CLI smoke test
```

## Claude ↔ Codex Collaboration

For scoped implementation tasks (bug fixes, narrow refactors, bounded features):
- Use `/codex-loop <task>` to run the Claude-as-architect / Codex-as-implementer workflow
- Claude writes `ai/plan.md`, Codex implements, Claude reviews
- Keep diffs small, avoid scope creep, verify after edits
- Bootstrap/automation files live under `ai/`, `.claude/`, `.agents/`, `scripts/`
- Do not use `/codex-loop` for large architectural rewrites
