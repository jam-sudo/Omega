# Omega — Project Instructions

> **Execution plan:** `docs/superpowers/plans/2026-03-16-omega-screening-platform.md`
> **Progress tracker:** `memory/MEMORY.md`
> **Memory path:** `~/.claude/projects/-home-jam-Omega/memory/`
> **Auto-loaded every conversation. Source of truth for all sessions.**

---

## Vision

Omega's ultimate goal is a **full digital general human** — a computational model of human physiology that can simulate any molecule's journey through the body, predict therapeutic outcomes, and personalize treatment.

**Current stage:** PBPK (pharmacokinetic) prediction from molecular structure.
**Roadmap:** PK → PK/PD → Systems Pharmacology → Digital Twin → Digital General Human.

The current PBPK module is the foundation layer: SMILES in → PK profile + uncertainty intervals out.
Architecture: ODE backbone (35-state PBPK) + ML ADME prediction + hybrid Cmax selector + PBPK/ML ensemble + conformal UQ.

### Honest Performance (2026-03-17)
- **In-sample (24 drugs):** Cmax AAFE 1.72, 79% 2-fold — 12/24 have CLint anchors (semi-supervised)
- **External (8 drugs):** Cmax AAFE 2.95, 62% 2-fold — true out-of-sample performance
- **Benchmark CSVs are synthetic** (1-cpt generated, not real clinical C(t) data)

## Workflow

All work is on `main` branch. Sequential development with automated benchmarks.

### Session Startup
1. Read `memory/MEMORY.md` for current status
2. Run `python scripts/run_full_benchmark.py` to check baseline
3. Work on highest-priority task
4. Before ending: update `memory/MEMORY.md`, run benchmark to verify no regression

### Team Structure
`/team` creates an Agent Team (via `TeamCreate`) — task에 맞는 역할만 동적 선택 (1-4명).
Teammates are independent sessions that communicate via `SendMessage` and share a task list.
Available roles: ml-engineer, infra-engineer, data-engineer, ci-auditor, domain-scientist, ode-engineer.
Details + cross-review protocol: `.claude/commands/team.md`

## Key Decisions (SETTLED — do not revisit)

1. **ADMET-AI disabled in production** — fup/logP changes break warfarin/metformin/losartan via Kp/Vd
2. **XGBoost CLint is primary** — 18 reference anchors at 50x weight (semi-supervised, partially circular)
3. **Hybrid Cmax selector is essential** — ablation proved it's the most valuable component (without: 44% 2-fold; with: 79%)
4. **Don't replace ODE with pure ML** — v1-v5 GNN all failed (distillation ceiling)
5. **Don't touch phase files** — build ML alongside, deprecate later
6. **PK-DB + FDA labels** for clinical data; ChEMBL deprioritized
7. **Analytical 1-cpt > raw ODE for Cmax** (67% vs 38% 2-fold) — but hybrid of both is best (79%)
8. **All benchmark CSVs are synthetic** — generated from 1-cpt model, not real clinical data. Warfarin replaced with PK-DB data (2026-03-17)
9. **External validation AAFE ~3.0** — in-sample 1.72 reflects tuning, not generalization

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

## Known Limitations (from 60-round self-critique, 2026-03-17)

- **CLint prediction is partially circular**: 12/24 benchmark drugs have 50x-weighted anchors back-calculated from clinical CL using the same IVIVE formula
- **All benchmark CSVs are synthetic**: generated from 1-cpt model with 20% constant SD, not real clinical data
- **fg = 1.0 for ALL drugs**: gut wall CYP3A4 metabolism not modeled (midazolam, nifedipine F overestimated)
- **Confidence is constant**: with ADMET-AI disabled, all drugs get "medium" confidence
- **Hybrid selector is critical but hacky**: 130 lines of non-standard heuristics, no industry precedent
- **Vd fails for fup < 0.01**: Berezhkovskiy Kp overestimates, XGBoost VDss also 2-4x off (warfarin 6.69x)
- **Warfarin and fluconazole are persistent outliers**: Vd and CLint prediction failures respectively

## Exit Criteria

| Level | Criteria | Status |
|-------|---------|--------|
| **1** | `omega predict <SMILES>` → PK profile. ADME AAFE<3.0. PK ≤2-fold for ≥70% of 20+ drugs | **PASS** (in-sample 1.72, 79%) |
| **2** | SMILES→PK <500ms. AAFE<2.0 | **PASS** (73ms, 1.72 in-sample) |
| **3** | Patient covariates. Few-shot (<5 obs) | **Prototype** (allometric + Bayesian) |
| **4** | Batch screening 1000+ molecules with UQ | **Done** (batch_predict + conformal CI) |
| **Ext** | External validation AAFE<2.5 on unseen drugs | **NOT MET** (2.95 on 8 drugs) |

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

## Build / Test / Lint Commands

```bash
source .venv/bin/activate
pytest tests/ -m "not slow and not benchmark" -q          # fast tests
pytest tests/ -m benchmark -v --timeout=300               # benchmarks
ruff check .                                              # lint
ruff format --check .                                     # format check
mypy src/omega_pbpk/core src/omega_pbpk/drugs src/omega_pbpk/config.py  # type check
omega --help                                              # CLI smoke test
```

## Claude ↔ Codex Collaboration

For scoped implementation tasks (bug fixes, narrow refactors, bounded features):
- Use `/codex-loop <task>` to run the Claude-as-architect / Codex-as-implementer workflow
- Claude writes `ai/plan.md`, Codex implements, Claude reviews
- Keep diffs small, avoid scope creep, verify after edits
- Bootstrap/automation files live under `ai/`, `.claude/`, `.agents/`, `scripts/`
- Do not use `/codex-loop` for large architectural rewrites
