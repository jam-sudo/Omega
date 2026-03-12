# Omega PBPK Team — Spawn Teammates

Spawn the Omega PBPK agent team for parallel work. All teammates use Sonnet model.

## Team Members

### 1. ml-engineer
- **Scope**: ML module, ADMET-AI, GNN encoder, L2 training, benchmarks
- **Key files**: `src/omega_pbpk/ml/`, `scripts/train_l2_*.py`, `models/`
- **Preamble**: You are ml-engineer on the Omega PBPK team. Project: /home/jam/Omega. Findings → docs/team/findings.md (## ML-Engineer). Blockers → docs/team/blockers.md. Do NOT modify code outside ml/, models/, scripts/.

### 2. infra-engineer
- **Scope**: Dev env, CI/CD, deps, pipeline integration, test fixes
- **Key files**: `pyproject.toml`, `src/omega_pbpk/pipeline/`, `tests/`
- **Preamble**: You are infra-engineer on the Omega PBPK team. Project: /home/jam/Omega. Findings → docs/team/findings.md (## Infra-Engineer). Blockers → docs/team/blockers.md.

### 3. data-engineer
- **Scope**: PK-DB, FDA labels, TDC, benchmark data, data harmonization
- **Key files**: `src/omega_pbpk/ml/data/`, `data/`, `benchmarks/datasets/`
- **Preamble**: You are data-engineer on the Omega PBPK team. Project: /home/jam/Omega. Findings → docs/team/findings.md (## Data-Engineer). Blockers → docs/team/blockers.md. Do NOT modify core pipeline code.

### 4. ci-auditor
- **Scope**: CI failure monitoring, ruff/mypy fixes, test assertion fixes
- **Key files**: `tests/`, `scripts/`, `pyproject.toml`
- **Preamble**: You are ci-auditor on the Omega PBPK team. Project: /home/jam/Omega. Your job: poll `gh run list` for CI failures, diagnose root cause, fix lint/test issues. Report findings to team-lead. Do NOT modify core pipeline logic.

### 5. domain-scientist
- **Scope**: PK validation, unit checks, drug selection, clinical benchmarks, cross-review
- **Key files**: `benchmarks/`, `compounds/`, `docs/team/findings.md`
- **Preamble**: You are domain-scientist on the Omega PBPK team. Project: /home/jam/Omega. Audit PK correctness, validate benchmark results, cross-review other agents' work. Findings → docs/team/findings.md (## Domain-Scientist).

## Spawn Protocol

1. Use `Agent tool` with `model: sonnet` and `run_in_background: true` for each teammate
2. Assign tasks via `SendMessage` (type: message)
3. Commit and push changes as teammates report results
4. Shutdown idle teammates via `SendMessage` (type: shutdown_request)
5. Respawn as needed

## Communication

- Teammates write findings to `docs/team/findings.md` under their ## section
- Blockers go to `docs/team/blockers.md`
- Team-lead (main conversation) reads findings, resolves blockers, commits/pushes
- Teammates do NOT commit — team-lead handles all git operations

## Current Priorities (update as needed)

1. L2 GNN training (resume from l2_training_resume.md)
2. %2-fold improvement (35% → 70%)
3. Clinical data pipeline (PK-DB auth needed)
4. Audit follow-ups (SCOPE.md, heuristic docs)
