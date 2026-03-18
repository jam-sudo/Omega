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
       → Hybrid Cmax selector (adaptive-weight blend of ODE + analytical 1-cpt)
       → PBPK/ML ensemble (DirectCmaxPredictor, confidence-weighted)
       → Conformal UQ (90% prediction intervals)
       → SimulationResult
```

### Honest Performance (2026-03-18, with bootstrap CI)
- **In-sample (24 drugs):** Cmax AAFE 1.665 **[95% CI: 1.44, 1.98]**, 88% 2-fold — Bayer 1.87 falls inside CI (not significant)
- **External (8 drugs):** Cmax AAFE 2.95, 62% 2-fold — true out-of-sample performance
- **Data leakage:** 36/107 (34%) gold tier drugs overlap with ADME training set
- **Error cancellation confirmed:** predicted ADME (AAFE 2.46) beats measured ADME (2.69) — ML errors compensate ODE structural biases
- **Benchmark CSVs are synthetic** (1-cpt generated, not real clinical C(t) data)

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
3. **Hybrid Cmax selector is essential** — ablation: +0.278 AAFE, 88% → 58% 2-fold (biggest single component)
4. **Don't replace ODE with pure ML** — v1-v5 GNN all failed (distillation ceiling)
5. **Don't touch phase files** — build ML alongside, deprecate later
6. **PK-DB + FDA labels** for clinical data; ChEMBL deprioritized
7. **Analytical 1-cpt > raw ODE for Cmax** (67% vs 38% 2-fold) — but hybrid of both is best (79%)
8. **All benchmark CSVs are synthetic** — generated from 1-cpt model, not real clinical data. Warfarin replaced with PK-DB data (2026-03-17)
9. **External validation AAFE ~3.0** — in-sample 1.72 reflects tuning, not generalization
10. **Error cancellation is real** — predicted ADME beats measured ADME (2.46 vs 2.69). Fix ODE structure BEFORE improving ADME.
11. **Ridge correction is dead code** — not loaded in pipeline, zero contribution confirmed by ablation
12. **Gut CLint drives Cmax, not hepatic CLint** — Sobol: gut CLint ST=0.470, hepatic CLint ST=0.000 for Cmax. Hepatic CLint only affects AUC.
13. **Ridge correction is confirmed dead code** — ablation study (Phase 0.1) shows NO_RIDGE = FULL with Δ=0.000 AAFE. The ridge model file exists in models/correction/ but is never loaded at inference. Keep for reproducibility only.
14. **Hybrid Cmax selector is the dominant component** — ablation Δ+0.278 AAFE without it (88%→58% 2-fold) with Phase 3a.1 + acid-Kp pipeline. Don't remove or simplify it.
15. **Error cancellation is systematic** — 79% of drugs (CI < 0.5), mean CI = 0.303. Fixing individual ADME params without joint balance will worsen aggregate AAFE.
16. **Phase 3a blocker is fm_CYP3A4 false positives, NOT Fh** — Polynomial clint_3a4 assigns fm_CYP3A4=0.887 to propranolol, 0.939 to ibuprofen. Fix: threshold guard `clint_3a4 > 2.0 µL/min/pmol` → AAFE 1.747, 83% 2-fold (diagnostic: 2026-03-18). Combined with acid-Kp D-fix → AAFE 1.665, 88% 2-fold.
17. **CLint_gut formula uses pre-inverted CLint** — `clint_L_per_h` is 22-223x larger than `CLh_target`; the 1.7× factor was calibrated for CLh_target. This is a known architectural bug, not a Phase 3a blocker. Fix in Phase 3b.
18. **fup calibration for low-fup drugs WORSENS AAFE** — isotonic regression improved individual fup accuracy but degraded AAFE +0.088 (fluconazole/atenolol rely on XGBoost fup under-prediction via error cancellation). Do not apply global low-fup calibration.
19. **CLint_gut K=1.7 is currently optimal — do NOT recalibrate to K=3.1** — Fitting K to Fg_lit data gives K_optimal=3.1 but AAFE worsens +0.094 (83%→79%). Root cause: pipeline fup_pred > fup_lit for CYP3A4 drugs (midazolam: 0.037 vs 0.024), so higher K is required to hit Fg_lit, but this over-corrects and breaks error cancellation. K recalibration requires fup fix first.
21. **Berezhkovskiy acid-Kp D-fix is correct and safe** — Using D (distribution coefficient at tissue/plasma pH) instead of P (neutral partition coefficient) for the neutral-lipid term in Berezhkovskiy Kp for acids. Fix applies to `compound_type="acid"` only (NOT bases/zwitterions). Dramatically corrects strong acids (ibuprofen pKa=4: Kp_adipose 8.4→0.67, fe_cmax 5.39→1.55x). AAFE 1.747→1.665, %2-fold 83%→88%, confirmed no regression on bases/neutrals. Implementation: `heuristics.py` `berezhkovskiy_kp` + `rodgers_rowland_kp`.
20. **OATP correction disabled — wrong direction for atorvastatin** — Atorvastatin AUC is UNDER-predicted (fe=3.64×; pred=0.048 vs obs=0.176 mg*h/L), meaning CLint is already over-predicted. OATP adds more clearance → makes AUC worse. CLint already >>QH (near-complete extraction), so any CLint addition has minimal but harmful effect. Root cause: CLint over-prediction, not missing uptake transporter. Code archived in pipeline with `_ENABLE_OATP_CORRECTION = False`.

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
| **1** | `omega predict <SMILES>` → PK profile. ADME AAFE<3.0. PK ≤2-fold for ≥70% of 20+ drugs | **PASS** (in-sample 1.665, 88%) |
| **2** | SMILES→PK <500ms. AAFE<2.0 | **PASS** (73ms, 1.665 in-sample) |
| **3** | Patient covariates. Few-shot (<5 obs) | **Prototype** (allometric + Bayesian) |
| **4** | Batch screening 1000+ molecules with UQ | **Done** (batch_predict + conformal CI) |
| **Ext** | External validation AAFE<2.5 on unseen drugs | **NOT MET** (2.95 on 8 drugs) |
| **v7** | Bootstrap CI on all metrics, ER-stratified, N=50+ gold tier, temporal AAFE<2.5 | **Plan v7 in progress** |

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
