# Omega PBPK — Project Instructions

> **Execution plan:** `docs/superpowers/plans/2026-03-16-omega-screening-platform.md`
> **Progress tracker:** `memory/MEMORY.md`
> **Memory path:** `~/.claude/projects/-home-jam-Omega/memory/`
> **Auto-loaded every conversation. Source of truth for all sessions.**

---

## Vision

Omega is a **hybrid mechanistic-ML pharmacokinetic prediction platform**.
SMILES in → PK profile + uncertainty intervals out.
Architecture: ODE backbone (35-state PBPK) + ML ADME prediction + post-hoc correction + conformal UQ.
The ODE engine provides the mechanistic backbone; ML corrects systematic biases and quantifies uncertainty.

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

1. **ADMET-AI** (`pip install admet-ai`) is available ADME predictor (disabled in production)
2. **RBP** needs custom model (no public model exists)
3. **Hybrid neural-mechanistic** for Level 2: GNN → Param Head → existing ODE
4. **Differentiable surrogate** of ODE for backprop; real ODE at inference
5. **Multi-fidelity training:** 1-cpt → 35-state ODE → real clinical data
6. **Don't touch phase files** — build ML alongside, deprecate later
7. **PyTorch + PyG** for ML; ADMET-AI standalone
8. **PK-DB + FDA labels** for clinical data; ChEMBL deprioritized
9. **Don't replace ODE with pure ML** — 5 experiments proved distillation ceiling (v1-v5 GNN all failed to beat ODE+heuristics)
10. **Hybrid correction model** on ODE residuals — Ridge/GLM with interpretable features, NOT neural
11. **XGBoost CLint is primary** — reference-anchored to clinical clearance; ADMET-AI CLint not calibrated for IVIVE
12. **ADMET-AI disabled in production** — fup/logP changes break warfarin/metformin/losartan via Kp/Vd
13. **Clinical data: PK-DB (Platinum) + FDA labels (Gold/Silver)** — 285 drugs total

## Codebase Rules

- New ML code → `src/omega_pbpk/ml/`
- New ML tests → `tests/ml/`
- Don't modify phase files (549 in core/, 481 in prediction/, 94 in clinical/)
- Preserve `ADMEProperties` contract exactly:
  - Props: mw, logP, logS, peff, fup, rbp, clint_3a4, clint_2d6, herg_ic50_uM
  - Units: clint=µL/min/pmol, peff=1e-4 cm/s, fup=0-1, rbp=0.5-3.0
  - Required: confidence ("low"/"medium"/"high"), conformal intervals
- Don't break existing 48K+ tests
- ODE engine (core/body.py) = training data oracle — keep accurate

## Exit Criteria

| Level | Criteria | Status |
|-------|---------|--------|
| **1** | `omega predict <SMILES>` → PK profile. ADME AAFE<3.0. PK ≤2-fold for ≥70% of 20+ drugs | **PASS** (1.95, 68%) |
| **2** | SMILES→PK <500ms. AAFE<2.0. Predicted params physically meaningful | **PASS** (73ms, 1.95) |
| **3** | Patient covariates. Few-shot (<5 obs). Generalizes to novel compounds | **Prototype** (allometric + Bayesian) |
| **4** | Batch screening 1000+ molecules with UQ | **In progress** |

## Tech Stack

| Purpose | Tool |
|---------|------|
| ADME (Level 1) | admet-ai (disabled), xgboost (primary) |
| Features | rdkit, torch-geometric |
| GNN (Level 2+) | torch-geometric, chemprop |
| Correction model | scikit-learn (Ridge) |
| UQ | scipy (conformal prediction) |
| Benchmarks | PyTDC |
| Clinical data | PK-DB REST API, OpenFDA API |
| Tracking | wandb or mlflow |

## What NOT To Do

- Don't consolidate/refactor legacy phase files
- Don't build ChEMBL ETL (in vitro, low ROI)
- Don't train ADME from scratch (ADMET-AI exists)
- Don't use NumPy for new ML (use PyTorch)
- Don't break ADMEProperties contract
- Don't replace ODE pipeline with pure ML (distillation ceiling proven)
- Don't merge without running full test suite

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
