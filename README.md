# Omega PBPK

**AI/ML-driven pharmacokinetic prediction platform.**

SMILES string in → PK profile out, powered by learned models.

![Tests](https://img.shields.io/badge/tests-passing-brightgreen)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

## Overview

Omega PBPK is a hybrid neural-mechanistic pharmacokinetic prediction platform. Unlike traditional PBPK tools that require manual parameterization, Omega learns to predict drug behavior directly from molecular structure using graph neural networks, then validates predictions through a mechanistic 35-state ODE engine.

The platform operates at three levels of sophistication:

| Level | Input | Method | Output |
|-------|-------|--------|--------|
| **Level 1** | SMILES | ADMET-AI ensemble + XGBoost | ADME properties → ODE → PK profile |
| **Level 2** | SMILES | GNN encoder → parameter head → ODE | End-to-end learned PK prediction |
| **Level 3** | SMILES + patient + dosing | Foundation model with cross-attention | Personalized PK with few-shot adaptation |

The ODE engine serves as both training data generator and inference-time validator — not the product itself. The product is the learned prediction: SMILES in, PK profile out.

> **Safety Scope:** For computational research and model prototyping only. Not validated for clinical decision making.

## Architecture

```
Level 1 (Ensemble):
  SMILES → ADMET-AI (pretrained D-MPNN) → ADME properties
         → XGBoost (Morgan FP) → RBP
         → Ensemble → Drug → 35-state ODE → PK profile

Level 2 (End-to-End):
  SMILES → MolecularEncoder (3-layer MPNN, 256-dim)
         → PKParameterHead (constrained activations)
         → DifferentiableODESurrogate (training) / Real ODE (inference)
         → C(t) curve → PK metrics

Level 3 (Foundation):
  SMILES → MolecularEncoder → 256-dim (Query)
  Patient covariates → PatientEncoder → 64-dim  ─┐
  Dosing regimen → DosingEncoder → 64-dim ───────┘→ 128-dim (K/V)
  → Cross-Attention (4 heads) → 256-dim fused
  → PKParameterHead → PK params → Real ODE → PK profile
  + Reptile meta-learning for few-shot adaptation
```

**Total model: ~2.04M parameters.** Multi-fidelity curriculum training (1-compartment analytical → 35-state ODE → clinical data).

## Features

### ML Pipeline
- **ADMET-AI integration** — pretrained Chemprop v2 D-MPNN (#1 on TDC ADMET leaderboard)
- **XGBoost RBP model** — custom blood-to-plasma ratio predictor (no public model exists)
- **GNN molecular encoder** — 3-layer message-passing neural network with edge features
- **Physics-constrained parameter head** — softplus/sigmoid activations enforce biological ranges
- **Differentiable ODE surrogate** — enables backprop through PK simulation during training
- **Physics-informed losses** — MSE + mass conservation + non-negativity + monotonic terminal + parameter plausibility
- **Multi-fidelity curriculum** — pre-train on cheap 1-cpt data, fine-tune on 35-state ODE, then clinical
- **Patient encoder** — age, weight, sex, CYP genotypes, organ impairment
- **Dosing encoder** — route, frequency, formulation, duration
- **Cross-attention fusion** — molecular × patient × dosing context
- **Reptile meta-learning** — few-shot adaptation from 1-5 observed concentrations
- **Conformal prediction** — calibrated uncertainty intervals (90% coverage target)

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

## Installation

```bash
git clone https://github.com/jam-sudo/Omega.git
cd Omega
pip install -e "."
```

### ML dependencies (required for ML predictions)

```bash
pip install -e ".[ml-new]"
# Installs: admet-ai, chemprop, torchdiffeq, xgboost, optuna, PyTDC, torch-geometric
```

Additional dependencies installed separately:

```bash
pip install rdkit torch
```

### Other optional extras

```bash
pip install -e ".[dev]"      # Development tools (pytest, ruff, mypy)
pip install -e ".[api]"      # REST API (FastAPI, uvicorn)
pip install -e ".[viz]"      # Visualization (matplotlib)
```

## Quick Start

### Level 1: ADME Ensemble → PK

```python
from omega_pbpk.ml.models.adme.ensemble import EnsembleADMEPredictor
from omega_pbpk.pipeline import OmegaPipeline, SimulationRequest

# ML-based ADME prediction
predictor = EnsembleADMEPredictor()
adme = predictor.predict("Cn1cnc2c1c(=O)n(C)c(=O)n2C")  # caffeine
print(f"logP: {adme.logP:.2f}, fup: {adme.fup:.3f}, CLint: {adme.clint_3a4:.4f}")

# Full PK simulation
pipeline = OmegaPipeline()
result = pipeline.simulate(SimulationRequest(
    smiles="Cn1cnc2c1c(=O)n(C)c(=O)n2C",
    dose_mg=100.0,
    route="oral",
))
print(f"Cmax: {result.cmax_mg_L:.2f} mg/L, t½: {result.t_half_h:.1f} h")
```

### Level 3: Personalized Prediction with Patient Context

```python
from omega_pbpk.ml.models.foundation.interactive import InteractivePKPredictor

predictor = InteractivePKPredictor()

# Basic prediction
result = predictor.predict("CC(=O)Oc1ccccc1C(=O)O")  # aspirin

# With patient covariates
result = predictor.predict(
    smiles="CC(=O)Oc1ccccc1C(=O)O",
    patient={"age": 65, "weight": 55, "sex": "F", "hepatic_impairment": "mild"},
    dosing={"dose_mg": 500, "route": "oral", "frequency": "BID"},
)

# Few-shot adaptation from observed data
adapted = predictor.adapt(
    smiles="CC(=O)Oc1ccccc1C(=O)O",
    observations=[(1.0, 15.2), (2.0, 12.8), (4.0, 6.1)],  # (time_h, conc_mg_L)
)
```

### CLI

```bash
# ML-based prediction (Level 1)
omega predict --smiles "Cn1cnc2c1c(=O)n(C)c(=O)n2C" --dose 100 --model ensemble

# ADMET-AI only
omega predict --smiles "CC(=O)Oc1ccccc1C(=O)O" --model admet-ai

# Legacy polynomial predictor
omega predict --smiles "CC(=O)Oc1ccccc1C(=O)O" --model legacy

# Advanced: with patient covariates (Level 3)
omega predict-advanced --smiles "CC(=O)Oc1ccccc1C(=O)O" \
    --dose 500 --route oral --frequency BID \
    --patient-age 65 --patient-weight 55 --patient-sex F --hepatic mild

# Population PK, DDI, benchmarks, etc.
omega population --compound compounds/warfarin.yaml --n-subjects 100
omega benchmark
omega report --smiles "SMILES" --name "Drug X" --dose 100 --out report.html
```

## CLI Reference

| Command | Description |
|---------|-------------|
| `omega predict` | SMILES → PK via ML ADME ensemble (`--model`: ensemble/admet-ai/legacy) |
| `omega predict-advanced` | Level 3: SMILES + patient covariates + dosing → personalized PK |
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

## Module Map

```
src/omega_pbpk/
├── ml/                     # ML prediction pipeline (NEW)
│   ├── data/               #   Data loading, synthetic generation, harmonization
│   │   ├── synthetic.py    #     ODE + 1-cpt training data generators
│   │   ├── loaders.py      #     PK-DB, FDA label, TDC data loaders
│   │   └── datasets.py     #     ClinicalPKDataset harmonization
│   ├── features/           #   Molecular featurization
│   │   └── graphs.py       #     SMILES → PyG graph (atom + bond features)
│   ├── models/             #   Model architectures
│   │   ├── adme/           #     Level 1: ADMET-AI wrapper, XGBoost RBP, ensemble
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
| TDC ADME benchmarks | — | 906-9,982 per endpoint | pip cached |

Generate training data:

```bash
# 1-compartment analytical (seconds)
python -c "from omega_pbpk.ml.data.synthetic import generate_1cpt_data; d = generate_1cpt_data(n_samples=100000); d.save_hdf5('data/ml/1cpt_100k.h5')"

# 35-state PBPK ODE (~80 min for 50K)
python -c "from omega_pbpk.ml.data.synthetic import generate_pbpk_data; d = generate_pbpk_data(n_samples=50000, n_workers=4); d.save_hdf5('data/ml/pbpk_50k.h5')"
```

## Tech Stack

| Purpose | Tool |
|---------|------|
| ADME prediction (Level 1) | admet-ai, xgboost, rdkit |
| GNN encoder (Level 2+) | torch, torch-geometric |
| Differentiable ODE | torchdiffeq |
| Hyperparameter search | optuna |
| ADME benchmarks | PyTDC |
| Clinical data | PK-DB REST API |
| ODE solver | scipy (LSODA) |
| Molecular features | rdkit |

## Validation

| Metric | Value | Target |
|--------|-------|--------|
| Level 1: ADME AAFE | < 3.0 | < 3.0 |
| Level 1: PK ≤2-fold | ≥ 70% of 20+ drugs | ≥ 70% |
| Level 2: SMILES → PK | < 500ms | < 500ms |
| Level 2: AAFE | < 2.0 | < 2.0 |
| Level 3: Few-shot | < 5 observations | < 5 |
| ODE mass balance (IV) | ~100% | ± 0.5% |
| ML test suite | 224 tests | all pass |

## Reference Compounds

| Compound | YAML | Primary CYP | fup | Literature CL (L/h) |
|----------|------|-------------|-----|---------------------|
| Midazolam | `compounds/midazolam.yaml` | CYP3A4 (93%) | 0.032 | 27-35 |
| Warfarin | `compounds/warfarin.yaml` | CYP2C9 (80%) | 0.005 | 0.15-0.20 |
| Propranolol | `compounds/propranolol.yaml` | CYP2D6 (70%) | 0.13 | 60-80 |
| Metformin | `compounds/metformin.yaml` | None (renal) | 0.99 | 25-35 (CLr) |
| Caffeine | `compounds/caffeine.yaml` | CYP1A2 (95%) | 0.65 | 1.5-2.0 |

## Development

```bash
pip install -e ".[dev]"
pytest tests/ -v              # Full test suite
pytest tests/ml/ -v           # ML tests only
ruff check src/               # Lint
ruff format src/              # Format
```

Pre-commit hook runs `ruff format` and `ruff check` automatically on staged files.

## License

MIT
