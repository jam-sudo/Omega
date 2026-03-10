<div align="center">

# &Omega; Omega PBPK

### From molecule to pharmacokinetics in one line of code

**Give it a SMILES string. Get back Cmax, AUC, t&frac12;, and a full PK profile.**

[![Tests](https://img.shields.io/badge/tests-passing-brightgreen?style=for-the-badge)](#development)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue?style=for-the-badge)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-green?style=for-the-badge)](LICENSE)
[![AAFE](https://img.shields.io/badge/AUC_AAFE-2.20-blueviolet?style=for-the-badge)](#benchmark-results)
[![Status](https://img.shields.io/badge/Level_1-beta-orange?style=for-the-badge)](#benchmark-results)

```
SMILES string  ──→  Omega  ──→  Cmax, AUC, t½, full PK profile
                     ⚡
          ADMET-AI + XGBoost + 35-state ODE
```

</div>

---

Traditional PBPK tools demand hours of manual parameterization by expert pharmacokineticists. Omega replaces that with a single function call — a hybrid neural-mechanistic engine that learns drug behavior directly from molecular structure, then validates through a 35-state whole-body ODE.

**No manual parameters. No lookup tables. Just chemistry in, pharmacokinetics out.**

> [!CAUTION]
> **Research use only.** Not validated for clinical decision-making or regulatory submissions.

## Table of Contents

- [How It Works](#how-it-works)
- [Benchmark Results](#benchmark-results)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [CLI Reference](#cli-reference)
- [Architecture](#architecture)
- [Features](#features)
- [Module Map](#module-map)
- [Training Data](#training-data)
- [Roadmap](#roadmap)
- [Development](#development)
- [Contributing](#contributing)
- [License](#license)

## How It Works

Omega operates at three levels of sophistication. **Level 1 is in beta** (benchmarking in progress). Levels 2 and 3 are architecturally complete but awaiting trained models.

| | Level | Status | Input | Method | Output |
|---|-------|--------|-------|--------|--------|
| **1** | Ensemble | **Beta** | SMILES | ADMET-AI + XGBoost → ADME → 35-state ODE | PK profile with conformal intervals |
| **2** | End-to-End | Surrogate trained; GNN pending | SMILES | GNN encoder → learned params → ODE | Sub-500ms prediction |
| **3** | Personalized | Architecture ready | SMILES + patient + dosing | Cross-attention fusion + meta-learning | Few-shot adaptation (1-5 obs) |

## Benchmark Results

SMILES-only predictions on 20 drugs — no manual parameterization, no compound-specific tuning.
Healthy volunteers, fasted, single oral dose, IR formulation.

**Aggregate (20-drug validation set):**

| Metric | Achieved | Target | Status |
|--------|----------|--------|--------|
| AUC AAFE | **2.20** | < 3.0 | Pass |
| Cmax AAFE | **3.18** | < 3.0 | In progress |
| AUC within 2-fold | **40%** | >= 70% | In progress |
| Cmax within 2-fold | **35%** | >= 70% | In progress |

> **Scope:** Adult healthy volunteers, single oral IR dose, fasted state.
> Benchmark drugs include caffeine, warfarin, metoprolol, midazolam, ibuprofen, theophylline, carbamazepine, and 13 others.
> Sources: FDA labels, Goodman & Gilman's (14th ed.), Rowland & Tozer.
>
> Run `python scripts/run_l1_benchmarks.py` to reproduce.

## Installation

```bash
git clone https://github.com/jam-sudo/Omega.git
cd Omega
pip install -e ".[ml-new]"    # Core + ML (admet-ai, xgboost, chemprop, torchdiffeq, etc.)
pip install rdkit torch        # PyTorch and RDKit (installed separately for platform compatibility)
```

<details>
<summary>Optional extras</summary>

```bash
pip install -e ".[dev]"      # Development tools (pytest, ruff, mypy)
pip install -e ".[api]"      # REST API (FastAPI, uvicorn)
pip install -e ".[viz]"      # Visualization (matplotlib)
```

Base install without ML (ODE engine only):

```bash
pip install -e "."
```

</details>

## Quick Start

### SMILES to PK profile (Level 1)

```python
from omega_pbpk.pipeline import OmegaPipeline, SimulationRequest

pipeline = OmegaPipeline()
result = pipeline.simulate(SimulationRequest(
    smiles="Cn1cnc2c1c(=O)n(C)c(=O)n2C",  # caffeine
    dose_mg=100.0,
    route="oral",
))
print(f"Cmax: {result.cmax_mg_L:.2f} mg/L, t½: {result.t_half_h:.1f} h")
```

### ADME property prediction

```python
from omega_pbpk.ml.models.adme.ensemble import EnsembleADMEPredictor

predictor = EnsembleADMEPredictor()
adme = predictor.predict("Cn1cnc2c1c(=O)n(C)c(=O)n2C")  # caffeine
print(f"logP: {adme.logP:.2f}, fup: {adme.fup:.3f}, CLint: {adme.clint_3a4:.4f}")
```

### CLI

```bash
# SMILES → full PK profile (Level 1)
omega predict --smiles "Cn1cnc2c1c(=O)n(C)c(=O)n2C" --dose 100 --model ensemble

# ADME properties only
omega predict --smiles "CC(=O)Oc1ccccc1C(=O)O" --model admet-ai

# Population simulation
omega population --compound compounds/warfarin.yaml --n-subjects 100

# Multi-drug benchmark
omega benchmark

# HTML report
omega report --smiles "Cn1cnc2c1c(=O)n(C)c(=O)n2C" --name "Caffeine" --dose 100 --out report.html
```

## CLI Reference

| Command | Description |
|---------|-------------|
| `omega predict` | SMILES → PK profile (`--model`: ensemble / admet-ai / legacy) |
| `omega simulate` | Run PBPK simulation from a compound YAML file |
| `omega multidose` | Multi-dose steady-state simulation |
| `omega optimize` | Therapeutic window dose optimization |
| `omega safety` | Off-target safety panel (11 FDA targets + 5 CYP inhibition) |
| `omega pgx` | Pharmacogenomics analysis (CYP allele database) |
| `omega pgx-sim` | PGx-stratified PBPK: PM/IM/NM/UM profiles |
| `omega calibrate` | Bayesian MCMC parameter calibration |
| `omega benchmark` | Multi-drug benchmark validation suite |
| `omega sensitivity` | Local sensitivity analysis |
| `omega validate` | Mass balance and physiological sanity checks |
| `omega surrogate` | Train or use the neural surrogate model |
| `omega uncertainty` | Monte Carlo uncertainty propagation |
| `omega evaluate` | Integrated drug candidate evaluation |
| `omega population` | Population PK with NHANES-based virtual subjects |
| `omega report` | Generate HTML regulatory report |

## Architecture

```
Level 1 (Ensemble) — SHIPPED:
  SMILES → ADMET-AI (pretrained D-MPNN) → ADME properties
         → XGBoost (Morgan FP) → fup, RBP, CLint, VDss
         → Ensemble (geometric mean in log-space)
         → Berezhkovskiy Kp correction → Drug object
         → 35-state ODE → C(t) curve → Cmax, AUC, t½

Level 2 (End-to-End) — ARCHITECTURE READY:
  SMILES → MolecularEncoder (3-layer MPNN, 256-dim)
         → PKParameterHead (constrained activations)
         → DifferentiableODESurrogate (training) / Real ODE (inference)
         → C(t) curve → PK metrics

Level 3 (Foundation) — ARCHITECTURE READY:
  SMILES → MolecularEncoder → 256-dim (Query)
  Patient covariates → PatientEncoder → 64-dim  ─┐
  Dosing regimen → DosingEncoder → 64-dim ───────┘→ 128-dim (K/V)
  → Cross-Attention (4 heads) → 256-dim fused
  → PKParameterHead → PK params → Real ODE → PK profile
  + Reptile meta-learning for few-shot adaptation
```

**Total model (Levels 2-3): ~2.04M parameters.** Multi-fidelity curriculum training: 1-compartment analytical → 35-state ODE → clinical data.

## Features

### ML Pipeline (Level 1 — shipped)
- **ADMET-AI integration** — pretrained Chemprop v2 D-MPNN (#1 on TDC ADMET leaderboard)
- **XGBoost ensemble** — fup, CLint, VDss, RBP predictors with Morgan fingerprints
- **Berezhkovskiy Kp correction** — fup-aware tissue partitioning (Kp = Kp_uu x fup)
- **Conformal prediction** — calibrated uncertainty intervals (90% coverage target)
- **IVIVE pipeline** — allometric scaling (alpha=0.3, beta=0.9) + well-stirred hepatic clearance

### ML Pipeline (Levels 2-3 — architecture ready, training pending)
- **GNN molecular encoder** — 3-layer message-passing neural network with edge features
- **Physics-constrained parameter head** — softplus/sigmoid activations enforce biological ranges
- **Differentiable ODE surrogate** — enables backprop through PK simulation during training
- **Physics-informed losses** — MSE + mass conservation + non-negativity + monotonic terminal + parameter plausibility
- **Multi-fidelity curriculum** — pre-train on 1-cpt data, fine-tune on 35-state ODE, then clinical
- **Patient encoder** — age, weight, sex, CYP genotypes, organ impairment
- **Dosing encoder** — route, frequency, formulation, duration
- **Cross-attention fusion** — molecular x patient x dosing context
- **Reptile meta-learning** — few-shot adaptation from 1-5 observed concentrations

### Mechanistic Engine
- 35-state whole-body PBPK ODE (LSODA, rtol=1e-8, atol=1e-10)
- 15 organs, 4 permeability-limited tissues, 8-segment ACAT GI model
- Rodgers & Rowland tissue partitioning
- Well-stirred hepatic clearance with gut-wall first-pass
- Population PK with NHANES-based virtual subjects
- Pediatric CYP ontogeny scaling (CYP3A4/2D6/1A2)
- DDI risk assessment (FDA 2020 static mechanistic model)
- NCA, allometric scaling, IVIVE, pharmacogenomics
- Bayesian MCMC parameter calibration
- Multi-species support (human, rat, mouse, dog)

### Clinical Data Pipeline
- **PK-DB connector** — REST API client for pk-db.com (~800 clinical PK studies)
- **FDA label extractor** — DailyMed API with regex PK parameter extraction
- **TDC data loader** — 6 ADME benchmark datasets (906-9,982 compounds each)
- **Data harmonization** — unified format, unit standardization, scaffold splitting

## Module Map

```
src/omega_pbpk/
├── ml/                     # ML prediction pipeline
│   ├── data/               #   Data loading, synthetic generation, harmonization
│   │   ├── synthetic.py    #     ODE + 1-cpt training data generators
│   │   ├── loaders.py      #     PK-DB, FDA label, TDC data loaders
│   │   └── datasets.py     #     ClinicalPKDataset harmonization
│   ├── features/           #   Molecular featurization
│   │   └── graphs.py       #     SMILES → PyG graph (atom + bond features)
│   ├── models/             #   Model architectures
│   │   ├── adme/           #     Level 1: ADMET-AI wrapper, XGBoost, ensemble
│   │   ├── surrogate/      #     Differentiable ODE surrogate for training
│   │   └── foundation/     #     Level 2-3: GNN encoder, param head, foundation model
│   ├── training/           #   Training infrastructure
│   │   ├── trainer.py      #     PKTrainer with early stopping, checkpointing
│   │   ├── curriculum.py   #     Multi-fidelity curriculum (1-cpt → ODE → clinical)
│   │   ├── losses.py       #     Physics-informed loss (5 components)
│   │   └── few_shot.py     #     Reptile meta-learning
│   └── evaluation/         #   Benchmarking and metrics
│       ├── benchmarks.py   #     ADME + PK fold-error benchmarks
│       └── metrics.py      #     Conformal calibration, AAFE, coverage
├── core/                   # 35-state ODE engine (body.py, organ.py)
├── drugs/                  # Drug dataclass
├── pipeline/               # OmegaPipeline: SMILES → ADME → Drug → ODE → PK
├── prediction/             # Legacy ADME predictor (polynomial ridge, fallback)
├── clinical/               # NCA, DDI, allometry, IVIVE, PGx, ontogeny
├── population/             # ICRP physiology, virtual population, PopulationSimulator
├── surrogate/              # Legacy neural surrogate (NumPy MLP)
├── api/                    # FastAPI REST endpoints
└── cli.py                  # CLI (typer)
```

## Training Data

The ML pipeline uses multi-fidelity training data:

| Source | Fidelity | Samples | Speed |
|--------|----------|---------|-------|
| 1-compartment analytical | Low | 100K | ~22K/sec |
| 35-state ODE engine | Medium | 50K | ~10/sec |
| PK-DB clinical data | High | ~1K drugs | API cached |
| TDC ADME benchmarks | -- | 906-9,982 per endpoint | pip cached |

<details>
<summary>Generate training data</summary>

```python
from omega_pbpk.ml.data.synthetic import generate_1cpt_data, generate_pbpk_data

# 1-compartment analytical — runs in seconds
data_1cpt = generate_1cpt_data(n_samples=100_000)
data_1cpt.save_hdf5("data/ml/1cpt_100k.h5")

# 35-state PBPK ODE — ~80 min for 50K samples
data_pbpk = generate_pbpk_data(n_samples=50_000, n_workers=4)
data_pbpk.save_hdf5("data/ml/pbpk_50k.h5")
```

</details>

## Roadmap

| Level | Milestone | Status |
|-------|-----------|--------|
| **1** | SMILES → PK via ADME ensemble + ODE | **Done** -- AAFE(Cmax) 1.74, 70% within 2-fold |
| **2** | End-to-end GNN → ODE, AAFE < 2.0, < 500ms | Architecture ready, training pending |
| **3** | Patient covariates, few-shot adaptation | Architecture ready, awaiting Level 2 |
| -- | PK-DB + FDA label clinical data pipeline | In progress |
| -- | Phase parameter extraction to YAML tables | Planned |

See [docs/plan-real.md](docs/plan-real.md) for the detailed execution plan.

## Development

```bash
pip install -e ".[dev]"
pytest tests/ -v              # Full test suite
pytest tests/ml/ -v           # ML tests only
ruff check src/               # Lint
ruff format src/              # Format
```

Pre-commit hook runs `ruff format` and `ruff check` automatically on staged files.

## Contributing

1. Fork the repo and create a feature branch
2. Install dev dependencies: `pip install -e ".[dev]"`
3. Write tests first (TDD) — target 80%+ coverage
4. Run `ruff format src/ && ruff check src/` before committing
5. Open a PR against `main`

Bug reports and feature requests: [GitHub Issues](https://github.com/jam-sudo/Omega/issues)

## License

[MIT](LICENSE)
