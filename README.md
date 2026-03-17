<div align="center">

# &Omega; Omega

### Toward a Digital General Human

**Structure-based pharmacokinetic prediction using hybrid mechanistic-ML modeling**

[![Tests](https://img.shields.io/badge/tests-48%2C671_passing-brightgreen?style=for-the-badge)](#development)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue?style=for-the-badge)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-green?style=for-the-badge)](LICENSE)
[![AAFE](https://img.shields.io/badge/Cmax_AAFE-1.72-blueviolet?style=for-the-badge)](#benchmark-results)
[![External](https://img.shields.io/badge/external-2.95-orange?style=for-the-badge)](#external-validation)
[![Speed](https://img.shields.io/badge/speed-73ms%2Fdrug-informational?style=for-the-badge)](#benchmark-results)

</div>

---

## Vision

Omega's long-term goal is a **full digital general human** — a computational model of human physiology capable of simulating any molecule's absorption, distribution, metabolism, and excretion (ADME), predicting therapeutic outcomes, and personalizing dosing.

**Current stage:** Whole-body PBPK prediction from molecular structure (SMILES).

**Roadmap:** PK &rarr; PK/PD &rarr; Systems Pharmacology &rarr; Digital Twin &rarr; Digital General Human

> [!CAUTION]
> **Research use only.** Not validated for clinical decision-making or regulatory submissions.
> In-sample benchmark metrics reflect tuning; external validation (AAFE 2.95) better represents prospective accuracy.

## How It Works

Omega predicts plasma pharmacokinetics directly from a SMILES string, without requiring measured in vitro data. The pipeline combines ML-predicted ADME properties with a mechanistic 35-state PBPK ODE:

```
SMILES
  |
  v
EnsembleADMEPredictor          XGBoost CLint/fup/rbp/VDss + polynomial logP/logS
  |
  v
Drug Object Construction       IVIVE scaling, Berezhkovskiy Kp, renal CL, P-gp correction
  |
  v
35-state ODE Simulation        Whole-body PBPK (15 organs, 8-segment ACAT GI)
  |
  v
Hybrid Cmax Selector           Adaptive-weight blend of ODE + analytical 1-compartment
  |
  v
PBPK/ML Ensemble               Confidence-weighted blend with direct XGBoost Cmax
  |
  v
Conformal UQ                   90% prediction intervals from parameter uncertainty
  |
  v
SimulationResult               Cmax, AUC, t_half, C(t) curve, confidence, intervals
```

**Key methods:** Berezhkovskiy (2004) tissue partitioning, well-stirred hepatic clearance, IVIVE with empirical scaling, Rodgers & Rowland Kp estimation, conformal prediction for uncertainty quantification.

| Level | Description | Input | Output |
|-------|-------------|-------|--------|
| **1-2** | Population PK prediction | SMILES + dose | Cmax, AUC, t&frac12;, C(t) curve |
| **3** | Patient-specific PK | SMILES + dose + covariates | Individualized PK |

## Benchmark Results

All predictions are SMILES-only — no manual parameterization, no measured in vitro data.

### Gold Tier (24 drugs, Cmax + AUC)

| Metric | Value | Notes |
|--------|-------|-------|
| Cmax AAFE | **1.72** | 12/24 drugs have CLint anchors (semi-supervised) |
| Cmax within 2-fold | **79%** | |
| AUC AAFE | **1.96** | |
| Speed (warm) | **73 ms/drug** | |

> Healthy volunteers, single oral IR dose, fasted state. Benchmark CSVs are synthetic (1-compartment generated); warfarin replaced with PK-DB clinical data (2026-03-17).
>
> Run `python scripts/run_full_benchmark.py` to reproduce.

### External Validation (8 drugs, never used in calibration)

| Metric | Value |
|--------|-------|
| Cmax AAFE | **2.95** |
| Cmax within 2-fold | **62%** |
| Median fold error | **1.47x** |

These 8 drugs were not used in any CLint anchor, IVIVE calibration, or pipeline tuning — the best available estimate of prospective accuracy.

<details>
<summary>Multi-tier validation details</summary>

| Tier | Scope | Metric | Result |
|------|-------|--------|--------|
| **Gold** | 24 drugs (Cmax + AUC) | Cmax AAFE | **1.72** |
| **Silver** | 39 drugs (t&frac12; from FDA labels) | t&frac12; AAFE | **2.42** |
| **Bronze** | 151 compounds (ADME properties) | logP / fup / peff / clint AAFE | 1.54 / 2.10 / 1.46 / 3.25 |
| **Temporal** | 5 post-2022 drugs | t&frac12; AAFE | **3.12** |
| **External** | 8 unseen drugs | Cmax AAFE | **2.95** |

</details>

### Context: Related Work

| Platform | Input | Cmax Accuracy | Drugs | Open Source |
|----------|-------|---------------|-------|-------------|
| **Omega** | SMILES only | AAFE 2.95 (external) | 8 | Yes |
| Bayer AI-PBPK (Maass 2024) | SMILES only | mfce 1.87 | 9 | No |
| Jia et al. (2025) | SMILES only | 60% 2-fold | 106 | Partial |
| Simcyp / GastroPlus | Measured in vitro | >80% 2-fold | 100+ | No |

> Note: Direct comparison is limited — each study uses different drug sets, metrics, and validation protocols. Omega's external AAFE 2.95 is the appropriate comparator for SMILES-only approaches.

<details>
<summary>Known limitations</summary>

- **CLint prediction** (AAFE 3.25): structure-based clearance prediction is the primary bottleneck; 12/24 benchmark drugs use semi-supervised anchors
- **Gut wall first-pass (Fg)**: currently ~1.0 for all drugs; CYP3A4 substrates (midazolam, nifedipine) have overestimated bioavailability
- **Vd for highly protein-bound drugs**: Berezhkovskiy Kp and XGBoost VDss both fail for fup < 0.01 (warfarin: 6.69x error)
- **Hybrid Cmax selector**: heuristic blending of ODE and analytical models; critical for accuracy but architecturally complex
- **P-gp substrates**: binary peff correction only (0.5x for substrates, skipped for substrate+inhibitors)
- **No Phase II metabolism**: UGT, NAT2, SULT enzymes not represented
- **No dissolution model**: BCS Class II drugs assume pre-dissolved drug
- **Benchmark data quality**: all CSVs are synthetic (1-compartment model, 20% constant SD)

</details>

## Installation

```bash
git clone https://github.com/jam-sudo/Omega.git
cd Omega
pip install -e ".[ml-new]"
pip install rdkit torch
```

<details>
<summary>Optional extras</summary>

```bash
pip install -e ".[dev]"      # Development tools (pytest, ruff)
pip install -e ".[api]"      # REST API (FastAPI)
pip install -e ".[viz]"      # Visualization (matplotlib)
pip install -e "."           # Base install (ODE engine only)
```

</details>

## Quick Start

### Population PK Prediction

```python
from omega_pbpk.pipeline import OmegaPipeline, SimulationRequest

pipeline = OmegaPipeline()
result = pipeline.simulate(SimulationRequest(
    smiles="Cn1cnc2c1c(=O)n(C)c(=O)n2C",  # caffeine
    dose_mg=100.0,
    route="oral",
))

print(f"Cmax: {result.cmax_mg_L:.2f} mg/L")
print(f"AUC:  {result.auc0t_mg_h_L:.2f} mg*h/L")
print(f"t1/2: {result.t_half_h:.1f} h")

# 90% prediction intervals
if result.cmax_ci90:
    lo, hi = result.cmax_ci90
    print(f"Cmax 90% CI: [{lo:.2f}, {hi:.2f}] mg/L")
```

### Batch Screening

```python
from omega_pbpk.screening.batch import batch_predict, rank_results

smiles_list = [
    "CC(C)Cc1ccc(C(C)C(=O)O)cc1",      # ibuprofen
    "CN(C)C(=N)NC(=N)N",                 # metformin
    "CC(=O)Nc1ccc(O)cc1",                # acetaminophen
]
results = batch_predict(smiles_list, dose_mg=100.0)
ranked = rank_results(results, objective="cmax")

for r in ranked:
    print(f"Rank {r['rank']}: Cmax={r['cmax_mg_L']:.2f} mg/L")
```

### Patient-Specific Prediction & CLI

```python
# Weight + CYP genotype adjustment
result = pipeline.simulate(SimulationRequest(
    smiles="CC(=O)CC(c1ccccc1)c1c(O)c2ccccc2oc1=O",  # warfarin
    dose_mg=5.0,
    subject_weight_kg=40.0,
    cyp2c9_genotype="*1/*3",
))

# Bayesian individual fitting from sparse C(t) observations
fit = pipeline.fit_individual(
    SimulationRequest(smiles="...", dose_mg=5.0),
    observations=[(1.0, 0.15), (4.0, 0.13), (12.0, 0.05)],
)
```

```bash
omega predict --smiles "Cn1cnc2c1c(=O)n(C)c(=O)n2C" --dose 100 --model ensemble
omega benchmark                                      # Multi-drug validation
```

## Architecture

```
src/omega_pbpk/
├── pipeline/               # OmegaPipeline: SMILES → PK
│   ├── __init__.py         #   Main pipeline (simulate, fit_individual)
│   └── pk_engine.py        #   Analytical 1-compartment PK engine
├── ml/                     # ML prediction modules
│   ├── models/adme/        #   XGBoost (CLint, fup, rbp, VDss), polynomial, ensemble
│   ├── models/direct_pk/   #   Direct Cmax predictor + PBPK/ML ensemble
│   ├── models/foundation/  #   Level 3: patient encoder, covariate scaling, Bayesian fitting
│   ├── applicability.py    #   Applicability domain filter (prodrug detection)
│   └── evaluation/         #   Benchmarks, metrics, conformal calibration
├── screening/              # Batch screening engine (batch_predict, rank_results)
├── uncertainty/            # Conformal UQ (LHS parameter sampling)
├── core/                   # 35-state ODE engine (body.py, organ.py)
├── prediction/             # Bioavailability prediction, legacy ADME
├── clinical/               # NCA, DDI, allometry, IVIVE, pharmacogenomics
├── population/             # Virtual population simulation
└── cli.py                  # CLI (typer)
```

## Training Data

| Source | Purpose | Samples |
|--------|---------|---------|
| TDC PPBR_AZ | XGBoost fup | 1,614 |
| TDC Clearance_Hepatocyte_AZ | XGBoost CLint (+18 clinical anchors) | 1,231 |
| TDC VDss_Lombardo | XGBoost VDss | 1,130 |
| adme_reference.csv | XGBoost RBP + Bronze calibration | 153 |
| PK-DB timecourses | Platinum-tier C(t) validation | 16 drugs |
| FDA label extraction | Gold/Silver-tier PK parameters | 296 drugs |
| Reference database | Unified multi-tier validation | 285 drugs |

## Roadmap

| Phase | Milestone | Status |
|-------|-----------|--------|
| **PK (current)** | SMILES &rarr; PK via hybrid mechanistic-ML | In-sample AAFE 1.72, external 2.95 |
| **Rigor (v7)** | Bootstrap CI, ablation, error cancellation analysis | Active |
| **Structural** | pKa integration, gut wall Fg, salt form, dissolution | Planned |
| **PK/PD** | Efficacy/toxicity endpoints from PK profiles | Future |
| **Digital Twin** | Patient-specific multi-organ physiological model | Future |
| **Digital General Human** | Full computational human physiology | Long-term vision |

## Development

```bash
pip install -e ".[dev]"

# Core test suite
pytest tests/ -m "not slow and not benchmark" -q     # ~48K fast tests
pytest tests/ml/test_accuracy_regression.py -v        # Accuracy regression guard

# Benchmarking
python scripts/run_full_benchmark.py                  # 24-drug Gold benchmark
python scripts/run_expanded_benchmark.py              # 285-drug expanded
python scripts/run_ablation.py                        # Ablation study (correction impact)

# Quality
ruff check . && ruff format --check .                 # Lint + format
```

Pre-commit hook runs `ruff format` and `ruff check` automatically.

## Contributing

1. Fork and create a feature branch
2. Install dev dependencies: `pip install -e ".[dev]"`
3. Write tests first (TDD)
4. Run `ruff format . && ruff check .` before committing
5. Run accuracy regression test: `pytest tests/ml/test_accuracy_regression.py`
6. Open a PR against `main`

## License

[MIT](LICENSE)
