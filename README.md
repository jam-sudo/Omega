<div align="center">

# &Omega; Omega PBPK

### From molecule to pharmacokinetics in one line of code

**Give it a SMILES string. Get back Cmax, AUC, t&frac12;, and a full PK profile.**

[![Tests](https://img.shields.io/badge/tests-passing-brightgreen?style=for-the-badge)](#development)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue?style=for-the-badge)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-green?style=for-the-badge)](LICENSE)
[![AAFE](https://img.shields.io/badge/Cmax_AAFE-1.90-blueviolet?style=for-the-badge)](#benchmark-results)
[![Status](https://img.shields.io/badge/Level_2-pass-brightgreen?style=for-the-badge)](#benchmark-results)
[![Drugs](https://img.shields.io/badge/validated-39_drugs-blue?style=for-the-badge)](#multi-tier-validation)
[![Speed](https://img.shields.io/badge/speed-73ms-orange?style=for-the-badge)](#benchmark-results)

```
SMILES string  ──→  Omega  ──→  Cmax, AUC, t½, full PK profile
                     ⚡
          XGBoost + polynomial ADME + 35-state ODE
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

Omega operates at three levels of sophistication:

| | Level | Status | Input | Method | Output |
|---|-------|--------|-------|--------|--------|
| **1-2** | Ensemble Pipeline | **All exit criteria pass** | SMILES | XGBoost + polynomial + GSE + hybrid selectors → 35-state ODE | PK profile, 73ms, AAFE 1.90 |
| **3** | Personalized | **Prototype working** | SMILES + patient | Allometric scaling + Bayesian individual fitting | Weight/genotype-adjusted PK |

## Benchmark Results

SMILES-only predictions — no manual parameterization, no compound-specific tuning, no lookup tables.

**Gold Tier: PK Accuracy (20 drugs, Cmax + AUC)**

| Metric | Achieved | Target | Status |
|--------|----------|--------|--------|
| Cmax AAFE | **1.90** | < 2.0 | Pass |
| AUC AAFE | **1.66** | < 2.0 | Pass |
| Cmax within 2-fold | **70%** | >= 70% | Pass |
| AUC within 2-fold | **70%** | >= 70% | Pass |
| Speed (warm) | **73ms** | < 500ms | Pass |

> Healthy volunteers, single oral IR dose, fasted state. 20 drugs including caffeine, warfarin, metoprolol, midazolam, ibuprofen, theophylline, carbamazepine, and 13 others.
>
> Run `python scripts/run_full_benchmark.py` to reproduce.

### Multi-Tier Validation

Omega is validated at multiple levels of fidelity across different data sources:

| Tier | Scope | Key Metric | Result |
|------|-------|------------|--------|
| **Gold** | 20 drugs (Cmax + AUC from C(t) curves) | Cmax AAFE | **1.90** |
| **Silver** | 39 drugs (t_half from FDA labels) | t_half AAFE | **2.42** |
| **Bronze** | 151 compounds (ADME properties) | logP AAFE / fup AAFE / peff AAFE | **1.54 / 2.10 / 1.46** |
| **Temporal** | 5 post-2022 drugs (truly unseen) | t_half AAFE, 3/5 within 2-fold | **3.12** |

<details>
<summary>Bronze-tier ADME property breakdown</summary>

| Property | AAFE | % within 2-fold | n |
|----------|------|-----------------|---|
| logP | 1.54 | 82% | 131 |
| fup | 2.10 | 58% | 151 |
| rbp | 1.09 | 98% | 151 |
| peff | 1.46 | 86% | 151 |
| clint | 3.25 | 34% | 151 |

</details>

<details>
<summary>Known limitations</summary>

- **P-gp substrates** (verapamil, digoxin): efflux transport not yet modeled, Cmax under-predicted
- **Highly protein-bound drugs** (ibuprofen, fup < 0.01): small fup prediction errors amplify PK errors
- **Nonlinear metabolism** (phenytoin): saturable CYP2C9 kinetics not captured by linear model
- **clint prediction** (AAFE 3.25): structure-based clearance prediction is an active research challenge

</details>

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

predictor = EnsembleADMEPredictor(admet_ai=False)  # production config
adme = predictor.predict("Cn1cnc2c1c(=O)n(C)c(=O)n2C")  # caffeine
print(f"logP: {adme.logP:.2f}, fup: {adme.fup:.3f}, CLint: {adme.clint_3a4:.4f}")
```

### Patient-Specific Prediction (Level 3)

```python
from omega_pbpk.pipeline import OmegaPipeline, SimulationRequest

pipeline = OmegaPipeline()

# Weight-adjusted simulation
result = pipeline.simulate(SimulationRequest(
    smiles="CC(=O)CC(c1ccccc1)c1c(O)c2ccccc2oc1=O",  # warfarin
    dose_mg=5.0,
    subject_weight_kg=40.0,  # lighter patient → higher Cmax
))

# Few-shot individual fitting (1-5 observed concentrations)
fit = pipeline.fit_individual(
    SimulationRequest(smiles="CC(=O)CC(c1ccccc1)c1c(O)c2ccccc2oc1=O", dose_mg=5.0),
    observations=[(1.0, 0.15), (4.0, 0.13), (12.0, 0.05)],  # (time_h, conc_mg_L)
)
print(f"Individual CL scale: {fit['cl_scale']:.2f}, Vd scale: {fit['vd_scale']:.2f}")
```

### CLI

```bash
# SMILES → full PK profile (Level 1)
omega predict --smiles "Cn1cnc2c1c(=O)n(C)c(=O)n2C" --dose 100 --model ensemble

# ADME properties only (XGBoost ensemble)
omega predict --smiles "CC(=O)Oc1ccccc1C(=O)O" --model ensemble

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
Level 1-2 (Ensemble Pipeline) — SHIPPED, ALL EXIT CRITERIA PASS:
  SMILES → XGBoost (Morgan FP) → fup, RBP, CLint, VDss
         → Polynomial + GSE solubility floor → logP, logS
         → Berezhkovskiy Kp + XGBoost VDss correction → Drug
         → 35-state ODE → hybrid Cmax/t½ selector → PK profile
         Speed: 73ms/drug (warm), AAFE 1.90 Cmax / 1.66 AUC

Level 3 (Personalized) — PROTOTYPE WORKING:
  Level 1-2 output → population PK parameters
  Patient covariates → allometric scaling (CL, Vd)
                     → CYP genotype factors (2D6, 2C9, 2C19)
  Sparse observations → Bayesian fitting (scipy L-BFGS-B)
                       → individual CL/Vd → patient-specific PK
```

**Level 1-2:** Hybrid ensemble (XGBoost + polynomial + heuristic corrections). **Level 3:** Allometric + Bayesian individual estimation.

## Features

### ML Pipeline (Level 1-2 — shipped)
- **XGBoost ensemble** — fup, CLint, VDss, RBP predictors with 2048-bit Morgan fingerprints (production primary)
- **Polynomial ridge fallback** — logP, logS with GSE solubility floor
- **ADMET-AI integration** — pretrained Chemprop D-MPNN available but disabled in production (causes tissue partitioning instabilities for some drugs)
- **Berezhkovskiy Kp correction** — fup-aware tissue partitioning with XGBoost VDss calibration
- **Conformal prediction** — uncertainty intervals (fup/rbp/peff well-calibrated; clint under-calibrated)
- **IVIVE pipeline** — allometric scaling (alpha=0.3, beta=0.9) + well-stirred hepatic clearance

### Level 3 — Patient-Specific (prototype working)
- **Allometric covariate scaling** — CL x (W/70)^0.75, Vd x (W/70)^1.0
- **CYP genotype factors** — CYP2D6 (UM/EM/IM/PM), CYP2C9 (6 diplotypes), CYP2C19
- **Bayesian individual estimation** — scipy L-BFGS-B fitting from 1-5 C(t) observations
- **SimulationRequest integration** — `subject_weight_kg`, `cyp2c9_genotype`, etc.

### Neural Architecture (research, not in production pipeline)
- **GNN molecular encoder** — 3-layer MPNN, 256-dim, with edge features
- **Differentiable ODE surrogate** — MLP approximation of 35-state ODE (AAFE 1.20)
- **Cross-attention fusion** — molecular x patient x dosing context
- **Reptile meta-learning** — few-shot adaptation (code ready, awaiting clinical data)

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

The production pipeline uses pre-trained XGBoost models and polynomial regression. The following data sources were used:

| Source | Purpose | Samples |
|--------|---------|---------|
| TDC PPBR_AZ | XGBoost fup training | 1,614 compounds |
| TDC Clearance_Hepatocyte_AZ | XGBoost CLint training | 1,213 compounds |
| TDC VDss_Lombardo | XGBoost VDss training | 1,130 compounds |
| adme_reference.csv | XGBoost RBP training + calibration | 153 compounds |
| ZINC drug-like | GNN label generation (research) | 21,811 compounds |
| OpenFDA labels | Silver-tier validation (t_half) | 43 drugs |

<details>
<summary>Generate synthetic ODE training data (for research)</summary>

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
| **1** | SMILES → PK via ADME ensemble + ODE | **Pass** — Cmax AAFE 1.90, AUC AAFE 1.66, 70% within 2-fold |
| **2** | AAFE < 2.0, < 500ms, regression testing | **Pass** — 73ms, automated benchmark with regression detection |
| **3** | Patient covariates, few-shot adaptation | **Prototype** — allometric scaling + Bayesian individual fitting |
| -- | Multi-tier validation (Gold/Silver/Bronze/Temporal) | **Done** — 20 drugs PK, 39 drugs t_half, 151 compounds ADME, 5 temporal |
| -- | Clinical data pipeline (PK-DB + OpenFDA) | **Done** — 118 PK params from 43 drugs, SMILES mapped |
| -- | Next: P-gp transporter correction | Planned |
| -- | Next: Neural L3 with real clinical data | Planned |

See [docs/superpowers/specs/2026-03-15-omega-next-phase-design.md](docs/superpowers/specs/2026-03-15-omega-next-phase-design.md) for the detailed plan.

## Development

```bash
pip install -e ".[dev]"
pytest tests/ -v                    # Full test suite (~48K tests)
pytest tests/ml/ -v                 # ML tests only
ruff check src/                     # Lint
ruff format src/                    # Format
```

### Benchmarking & Validation

```bash
python scripts/run_full_benchmark.py                    # Gold-tier: 20 drugs, Cmax/AUC AAFE
python scripts/run_full_benchmark.py --previous prev.json  # ...with regression detection
python scripts/run_silver_benchmark.py                  # Silver-tier: 39 drugs, t_half
python scripts/run_bronze_benchmark.py                  # Bronze-tier: 151 compounds, ADME
python scripts/run_temporal_holdout.py                  # Temporal: 5 post-2022 drugs
python scripts/analyze_failures.py                      # Classify >3-fold errors by mechanism
python scripts/demo_l3_covariates.py                    # L3 demo: weight/genotype scenarios
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
