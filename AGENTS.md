# Omega PBPK — Agent Instructions

## Project

Omega PBPK v0.7 — Whole-body PBPK simulation platform.

## Structure

- `src/omega_pbpk/` — Main package (34-state ODE engine)
- `compounds/` — YAML compound definitions (midazolam, caffeine)
- `tests/test_all.py` — 47 tests
- `pyproject.toml` — Package config, entry point: `omega = omega_pbpk.cli:main`

## Development

```bash
pip install -e ".[dev]"
pytest tests/ -v
ruff check src/
ruff format src/
```

## Technical Rules

- ODE solver: `solve_ivp`, method='LSODA', rtol=1e-8, atol=1e-10
- Mass balance: IV dose = sum(all states) ± 0.5%
- Units: mg/L (concentration), L (volume), L/h (flow/clearance), h (time), mg (dose)
- Drug parameters: always use Drug dataclass, never raw dicts
- YAML loading: yaml.safe_load only, through config.py
- All new features require tests

## Coding Conventions

- Python 3.10+ with full type hints on public functions.
- Keep functions small, side effects explicit, module boundaries clear.
- Run formatting/lint with `ruff` and type checks with `mypy`.

## Commit Convention

`type(scope): description`
- type: feat, fix, refactor, test, docs, chore
- scope: core, drugs, population, prediction, docking, qsp, pgx, clinical, api, cli
