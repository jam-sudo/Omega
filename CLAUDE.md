# Omega PBPK — Project Instructions

> **Execution plan:** `docs/plan-real.md` | **Progress tracker:** `memory/MEMORY.md`
> **Memory path:** `~/.claude/projects/-home-jam-Omega/memory/`
> **Auto-loaded every conversation. Source of truth for all sessions.**

---

## Vision

Omega is an **AI/ML-driven pharmacokinetic prediction platform**, NOT a calculator.
SMILES string in → PK profile out, powered by learned models.
The ODE engine is infrastructure (training data, validation, explainability) — not the product.

## Parallel Branch System

> **NOTE:** In practice, all work has been done on `main`. The branch system below
> was planned but never executed. Kept for future reference only.

Work is organized into **5 parallel git branches**. Each session, check
`memory/MEMORY.md` for current status and pick the highest-priority unfinished task.

### Branch Map

| Branch | Git Branch | Purpose | Depends On | Level |
|--------|-----------|---------|-----------|-------|
| **0A** | `fix/ode-critical-bugs` | Fix 3 ODE bugs | Nothing | Prereq |
| **0B** | `feat/ml-infrastructure` | Create ml/ dir, deps, interface | Nothing | Prereq |
| **B** | `feat/adme-ml-level1` | ADMET-AI + RBP + ensemble + validation | 0B merged | L1 |
| **A** | `feat/ode-training-data` | 50K ODE profiles + differentiable surrogate | 0A merged | L2 |
| **C** | `feat/gnn-architecture` | GNN encoder + parameter head | 0B merged | L2 |
| **D** | `feat/clinical-data-pipeline` | PK-DB + FDA label extraction | Nothing | L3 |
| **E** | `feat/phase-param-extraction` | Extract phase params → YAML tables | Nothing | Parallel |

### Merge Order
```
0A ──merge──→ main ──→ unlocks Branch A
0B ──merge──→ main ──→ unlocks Branch B, Branch C
B  ──merge──→ main ──→ Level 1 complete
A + C merge → main ──→ then feat/level2-training (2.5-2.7) → Level 2 complete
D  ──merge──→ main ──→ data ready for Level 3
Level 2 + D → feat/foundation-model (3.3-3.7) → Level 3 complete
```

### Session Startup Checklist
1. Read `memory/MEMORY.md` for current status
2. Ask user what to work on (or pick highest-priority TODO from MEMORY.md)
3. Before ending: update status in `memory/MEMORY.md`

### Team Structure
`/team` creates an Agent Team (via `TeamCreate`) — task에 맞는 역할만 동적 선택 (1-4명).
Teammates are independent sessions that communicate via `SendMessage` and share a task list.
Available roles: ml-engineer, infra-engineer, data-engineer, ci-auditor, domain-scientist, ode-engineer.
Details + cross-review protocol: `.claude/commands/team.md`

## Key Decisions (SETTLED — do not revisit)

1. **ADMET-AI** (`pip install admet-ai`) is primary ADME predictor
2. **RBP** needs custom model (no public model exists)
3. **Hybrid neural-mechanistic** for Level 2: GNN → Param Head → existing ODE
4. **Differentiable surrogate** of ODE for backprop; real ODE at inference
5. **Multi-fidelity training:** 1-cpt → 35-state ODE → real clinical data
6. **Don't touch phase files** — build ML alongside, deprecate later
7. **PyTorch + PyG** for ML; ADMET-AI standalone; torchdiffeq for Neural ODE
8. **PK-DB + FDA labels** for clinical data; ChEMBL deprioritized

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

| Level | Criteria |
|-------|---------|
| **1** | `omega predict <SMILES>` → PK profile. ADME AAFE<3.0. PK ≤2-fold for ≥70% of 20+ drugs |
| **2** | SMILES→PK <500ms. AAFE<2.0. Predicted params are physically meaningful |
| **3** | Patient covariates. Few-shot (<5 obs). Generalizes to novel compounds |

## Tech Stack

| Purpose | Tool |
|---------|------|
| ADME (Level 1) | admet-ai, xgboost |
| Features | rdkit, torch-geometric |
| GNN (Level 2+) | torch-geometric, chemprop |
| Differentiable ODE | torchdiffeq |
| Hyperparams | optuna |
| Benchmarks | PyTDC |
| Clinical data | PK-DB REST API |
| Tracking | wandb or mlflow |

## What NOT To Do

- Don't consolidate/refactor legacy phase files
- Don't build ChEMBL ETL (in vitro, low ROI)
- Don't train ADME from scratch (ADMET-AI exists)
- Don't use NumPy for new ML (use PyTorch)
- Don't break ADMEProperties contract
- Don't work across branches in one session without user approval
- Don't merge branches without running full test suite

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
