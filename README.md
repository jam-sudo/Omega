# Omega PBPK — Whole-Body Pharmacokinetic Simulation Platform

Research software for **whole-body PBPK** simulation with ADME prediction,
DDI assessment, pharmacodynamics, pharmacogenomics, and clinical dose optimization.

## Safety Scope

- For computational research and model prototyping only.
- Not validated for clinical decision making.
- No real-world dosing recommendations are provided.

## Features

- **34-state ODE engine** — 15 organs (perfusion + permeability-limited), 8-segment ACAT absorption
- **SMILES → ADME** — 9 QSPR property predictions (logP, solubility, permeability, fup, CLint, hERG)
- **DDI** — competitive inhibition, mechanism-based inactivation (MBI), enzyme induction
- **Population PK** — Monte Carlo virtual population (N=50) from ICRP reference physiology
- **QSP/PD** — Emax, indirect response (4 types), tumor growth (Simeoni), biomarker turnover
- **Safety** — 11 FDA off-target panel + 5 CYP inhibition prediction
- **Pharmacogenomics** — 5 CYP gene allele database (CPIC standard)
- **Clinical** — FIH dose calculation, multi-dose steady state, therapeutic window optimization
- **GNN scaffold** — MPNN architecture (~200K params) for ADME prediction (requires training data)
- **REST API** — 10 FastAPI endpoints (optional)
- **CLI** — 7 commands

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Quick Start

```bash
# Oral simulation
omega simulate --compound compounds/midazolam.yaml --dose-mg 7.5 --route oral --out outputs/run1

# IV simulation
omega simulate --compound compounds/midazolam.yaml --dose-mg 5 --route iv --out outputs/run_iv

# ADME prediction from SMILES
omega predict --smiles "Cn1c(=O)c2c(ncn2C)n(C)c1=O"

# Multi-dose
omega multidose --compound compounds/midazolam.yaml --dose-mg 7.5 --interval 12 --days 7

# Dose optimization
omega optimize --compound compounds/midazolam.yaml --mec 0.01 --mtc 1.0

# Safety panel
omega safety --smiles "Clc1ccc2c(c1)C(=NCc3nccn3C)c1cc(F)ccc1N2" --cmax-uM 0.15 --fup 0.032

# Pharmacogenomics
omega pgx --gene CYP2D6 --population East_Asian
```

## Outputs

```
outputs/run1/
├── timecourse.csv    # time_h, Cp_mg_L
├── summary.json      # {Cmax, Tmax, AUC, half_life, CL, Vss, ...}
└── plots.png         # Linear + semilog PK curves
```

## Validation (Midazolam, FDA PBPK Probe Drug)

| Metric | Value | Target |
|--------|-------|--------|
| IV AAFE | 1.62 | < 2.0 |
| Oral AAFE | 2.55 | < 2.0 (WIP) |
| DDI AUC ratio | 9.4× | obs 10-16× |
| PopPK CV (Cmax) | 27% | 20-40% |
| Mass balance (IV) | ~100% | ± 0.5% |
| Test suite | 47/47 | all pass |

## Architecture

```
src/omega_pbpk/
├── core/           # 34-state ODE engine (body.py, organ.py)
├── drugs/          # Drug dataclass + reference compounds
├── population/     # ICRP physiology + virtual population
├── prediction/     # SMILES → ADME QSPR
├── docking/        # Off-target safety + CYP inhibition
├── qsp/            # PD models (Emax, IDR, tumor, biomarker)
├── pharmacogenomics/  # CYP polymorphism (CPIC)
├── clinical/       # FIH dose, multi-dose, optimization
├── ml_models/      # GNN MPNN scaffold
├── api/            # FastAPI REST endpoints
├── visualization/  # Publication-quality PK plots
├── config.py       # YAML loader
└── cli.py          # 7-command CLI
```

## Tests

```bash
pytest tests/ -v
```

## Formatting

```bash
ruff format src/
ruff check src/
```

## License

MIT
