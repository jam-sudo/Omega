# Omega PBPK Team — Dynamic Agent Team

Analyze the current task, pick the right roles, and spawn an Agent Team using `TeamCreate`.

## How It Works

1. **Analyze the task** — determine which expertise is needed
2. **Pick 1-4 roles** from the pool below (most tasks need 2-3)
3. **Use `TeamCreate`** to spawn the team — teammates are independent sessions that communicate via `SendMessage` and share a task list
4. **Monitor & steer** — check progress, resolve blockers, synthesize findings

> This uses **Agent Teams** (experimental), NOT subagents.
> Teammates can message each other directly and self-claim tasks.

## Available Roles

| Role | Expertise | Key Files | When to Spawn |
|------|-----------|-----------|---------------|
| ml-engineer | ML module, ADMET-AI, GNN, L2 training, benchmarks | `src/omega_pbpk/ml/`, `scripts/train_l2_*.py`, `models/` | ML code changes, model training, evaluation |
| infra-engineer | Dev env, CI/CD, deps, pipeline integration, test fixes | `pyproject.toml`, `src/omega_pbpk/pipeline/`, `tests/` | Build issues, dep conflicts, pipeline wiring |
| data-engineer | PK-DB, FDA labels, TDC, benchmark data, data harmonization | `src/omega_pbpk/ml/data/`, `data/`, `benchmarks/datasets/` | Data ingestion, format changes, new data sources |
| ci-auditor | CI failure monitoring, ruff/mypy fixes, test assertion fixes | `tests/`, `scripts/`, `pyproject.toml` | After code changes to verify CI health |
| domain-scientist | PK validation, unit checks, drug selection, clinical benchmarks | `benchmarks/`, `compounds/`, `docs/` | Validating PK correctness, reviewing results |
| ode-engineer | ODE bugs, training data generation, differentiable surrogate | `src/omega_pbpk/core/body.py`, `models/pbpk_surrogate/` | ODE engine fixes, surrogate training |

## Teammate Preamble Template

Each teammate should receive:
```
You are [ROLE] on the Omega PBPK team.
Project: /home/jam/Omega
Task: [SPECIFIC TASK]
Findings → docs/team/findings.md (## [Role] section)
Blockers → docs/team/blockers.md
Do NOT modify code outside your scope.
Do NOT commit — team-lead handles git.
```

## Cross-Review Protocol

After teammates complete work, assign cross-reviews based on domain overlap:

| If work touches... | Reviewer should be... | Focus |
|---------------------|----------------------|-------|
| ML + ODE integration | ode-engineer or domain-scientist | Physiological correctness of ML outputs |
| Data formats/schemas | ml-engineer | Downstream compatibility |
| Drug data completeness | domain-scientist | Clinical validity |
| Any code change | ci-auditor | Lint, types, test pass |

## Communication

- Teammates message each other directly via `SendMessage`
- Findings → `docs/team/findings.md` (under their ## section)
- Blockers → `docs/team/blockers.md`
- Team-lead reads findings, resolves blockers, commits/pushes
- Teammates do NOT commit — team-lead handles all git operations
- Use `Shift+Down` to cycle through teammates (in-process mode)

## File Conflict Rule

Two teammates must NOT edit the same file. Break work so each teammate owns different files.

## Current Priorities (update as needed)

**Active: Plan v7 — Scientific Rigor & Structural Fixes**
See: `docs/superpowers/plans/2026-03-17-omega-v7-scientific-rigor.md`

1. **Phase 0: Diagnostic Sprint** — 4 teammates (data-engineer, domain-scientist, ml-engineer, ci-auditor)
   - Ablation study, measured-ADME ablation, bootstrap CI, Sobol GSA, ER-stratified, data audit
   - NO CODE CHANGES — scripts only
2. **Phase 1-2: Stats + Integration** — connect pKa predictor, salt form, error cancellation monitor
3. **Phase 3-4: Structural fixes + mechanistic extensions** — gut wall Fg, fuinc, P-gp ACAT, OATP
4. **Phase 5: Validation expansion + paper** — N=50+ gold tier, paper rewrite with CI/ablation/stratification
