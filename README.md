<div align="center">

# &Omega; Omega

### Toward a Digital General Human

**Structure-based pharmacokinetic prediction using hybrid mechanistic-ML modeling**

[![Tests](https://img.shields.io/badge/tests-48K%2B_passing-brightgreen?style=for-the-badge)](#development)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue?style=for-the-badge)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-green?style=for-the-badge)](LICENSE)
[![AAFE](https://img.shields.io/badge/Cmax_AAFE-1.50_%5B1.32%2C_1.74%5D-blueviolet?style=for-the-badge)](#benchmark-results)
[![External](https://img.shields.io/badge/external_AAFE-2.95-orange?style=for-the-badge)](#external-validation)
[![Speed](https://img.shields.io/badge/latency-61ms%2Fdrug-informational?style=for-the-badge)](#benchmark-results)

</div>

---

> [!CAUTION]
> **Research use only.** Not validated for clinical decision-making or regulatory submissions.
> In-sample metrics (24 drugs) reflect tuning; external validation (AAFE 2.95 on 8 unseen drugs) provides a more realistic estimate of prospective accuracy.

## What Omega Does

Omega predicts human plasma pharmacokinetics directly from a molecular structure (SMILES string), without requiring measured in vitro data. Given a SMILES and dose, it returns C<sub>max</sub>, AUC, t&frac12;, a full C(t) concentration-time profile, and 90% prediction intervals.

**Current stage:** Whole-body PBPK prediction from molecular structure.
**Long-term vision:** PK &rarr; PK/PD &rarr; Systems Pharmacology &rarr; Digital Twin &rarr; Digital General Human.

## How It Works

The pipeline combines ML-predicted ADME properties with a mechanistic 35-state whole-body PBPK ODE system:

```
SMILES
  │
  ▼
EnsembleADMEPredictor          XGBoost CLint/fup/rbp/VDss + polynomial logP/logS
  │
  ▼
pKa & Compound Type            RDKit SMARTS functional group detection
  │
  ▼
Drug Object Construction       IVIVE scaling, Berezhkovskiy Kp (ionization-corrected
                               for acids), renal CL, P-gp correction, gut wall CYP3A4
  │
  ▼
35-state ODE Simulation        Whole-body PBPK (15 organs, 8-segment ACAT GI tract)
  │
  ▼
Hybrid Cmax Selector           Adaptive-weight blend of ODE + analytical 1-cpt model
  │
  ▼
PBPK/ML Ensemble               Confidence-weighted blend with direct XGBoost Cmax
  │
  ▼
Conformal UQ                   90% prediction intervals via parameter uncertainty
  │
  ▼
SimulationResult               Cmax, AUC, t_half, C(t), confidence, intervals
```

**Key methods:** Berezhkovskiy (2004) tissue partitioning with distribution-coefficient correction for ionized acids, well-stirred hepatic clearance, IVIVE scaling (Houston 1994), Rodgers & Rowland Kp estimation, conformal prediction for uncertainty quantification.

## Benchmark Results

All predictions are SMILES-only. No manual parameterization, no measured in vitro data. All 24 gold-tier reference values are sourced from FDA-approved labels or peer-reviewed clinical literature.

### Gold Tier (24 drugs)

| Metric | Value | 95% Bootstrap CI | Notes |
|--------|-------|-------------------|-------|
| C<sub>max</sub> AAFE | **1.50** | [1.32, 1.74] | 12/24 drugs use semi-supervised CLint anchors; inter-study floor &asymp; 1.23 |
| C<sub>max</sub> within 2-fold | **83%** | &mdash; | 20 of 24 drugs |
| AUC AAFE | **1.86** | [1.50, 2.50] | |
| AUC within 2-fold | **62%** | &mdash; | |
| &gt;3-fold C<sub>max</sub> errors | **1** | &mdash; | midazolam (3.99&times;; CYP3A4 gut wall architectural limitation) |
| Prediction latency | **61 ms/drug** | &mdash; | Warm start, single core |

> Healthy volunteers, single oral IR dose, fasted state.
>
> Run `python scripts/run_full_benchmark.py` to reproduce (includes bootstrap CI).

### External Validation

| Metric | Value |
|--------|-------|
| C<sub>max</sub> AAFE | **2.95** |
| C<sub>max</sub> within 2-fold | **62%** |

8 drugs held out from all CLint anchors, IVIVE calibration, and pipeline tuning &mdash; the best available estimate of prospective accuracy.

<details>
<summary>Multi-tier validation details</summary>

| Tier | N | Metric | Result | 95% CI |
|------|---|--------|--------|--------|
| **Gold** | 24 drugs | C<sub>max</sub> AAFE / %2-fold | **1.50** / 83% | [1.32, 1.74] |
| **Expanded Gold** | 51 drugs | C<sub>max</sub> AAFE / %2-fold | **3.40** / 39% | [2.42, 4.94] |
| **Silver** | 39 drugs | t&frac12; AAFE | **2.42** | &mdash; |
| **Bronze** | 151 compounds | logP / fup / rbp / peff AAFE | 1.54 / 2.10 / 1.09 / 1.46 | &mdash; |
| **Temporal** | 15 drugs (C<sub>max</sub>) | C<sub>max</sub> AAFE / %2-fold | **2.66** / 47% | [1.74, 4.35] |
| **Temporal (in-scope)** | 10 drugs | C<sub>max</sub> AAFE / %2-fold | **1.64** / 70% | [1.33, 2.11] |
| **External** | 8 drugs | C<sub>max</sub> AAFE / %2-fold | **2.95** / 62% | &mdash; |

</details>

<details>
<summary>Ablation study (component contributions)</summary>

Each row removes one component from the full pipeline and re-evaluates:

| Configuration | AAFE | &Delta; AAFE | %2-fold |
|---------------|------|-------------|---------|
| Full pipeline | 1.51 | &mdash; | 88% |
| No hybrid selector | 1.83 | +0.32 | 62% |
| No PBPK/ML ensemble | 1.71 | +0.20 | 71% |
| No VDss correction | 1.60 | +0.09 | 79% |
| No P-gp correction | 1.51 | &minus;0.01 | 83% |
| Bare (ODE only) | 1.93 | +0.41 | 58% |

The hybrid C<sub>max</sub> selector is the single largest contributor (&Delta; +0.32 AAFE when removed).

</details>

### Related Work

| Platform | Input | C<sub>max</sub> Accuracy | Drugs | Open Source |
|----------|-------|--------------------------|-------|-------------|
| **Omega** | SMILES only | AAFE 2.95 (external) | 8 | Yes |
| Bayer AI-PBPK (Maass 2024) | SMILES only | mfce 1.87 | 9 | No |
| Jia et al. (2025) | SMILES only | 60% 2-fold | 106 | Partial |
| Simcyp / GastroPlus | Measured in vitro | &gt;80% 2-fold | 100+ | No |

> Direct comparison across studies is limited by differences in drug sets, metrics, and validation protocols. Omega's external AAFE 2.95 on 8 held-out drugs is the appropriate comparator for SMILES-only approaches.

<details>
<summary>Known limitations</summary>

- **CLint prediction** (AAFE 3.25): structure-based clearance prediction is the primary accuracy bottleneck; 12/24 gold-tier drugs use semi-supervised anchors back-calculated from clinical CL
- **Error cancellation**: predicted ADME (AAFE 2.10) outperforms measured ADME (2.50) on the gold tier &mdash; ML prediction errors partially compensate for ODE structural biases. This is systematic (79% of drugs) and must be preserved when modifying individual ADME components
- **Gut wall first-pass (Fg)**: CYP3A4 threshold guard prevents false positives, but the CLint<sub>gut</sub> scaling formula uses a pre-inverted CLint value (known architectural issue; empirically calibrated)
- **Vd for highly protein-bound drugs**: Berezhkovskiy Kp overestimates tissue partitioning for fup &lt; 0.01; VDss anchors partially compensate for selected drugs
- **Multi-compartmental drugs**: compounds with large V<sub>c</sub>/V<sub>DSS</sub> ratios (diazepam, fluconazole) are poorly described by the analytical 1-compartment C<sub>max</sub> model
- **Data leakage**: 36/107 (34%) gold-tier drugs overlap with ADME training set
- **No transporter modeling**: P-gp uses a binary permeability correction only; OATP, OCT2, OAT are not represented
- **No Phase II metabolism**: UGT, NAT2, SULT enzymes not modeled
- **No dissolution model**: BCS Class II drugs assume pre-dissolved drug in solution
- **Expanded gold (51 drugs)**: AAFE 3.40 &mdash; performance degrades substantially beyond the tuned gold-tier set

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
pip install -e ".[dev]"      # Development tools (pytest, ruff, pint, pytest-benchmark)
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

### Patient-Specific Prediction

```python
warfarin = "CC(=O)CC(c1ccccc1)c1c(O)c2ccccc2oc1=O"

# Weight + CYP genotype adjustment
result = pipeline.simulate(SimulationRequest(
    smiles=warfarin,
    dose_mg=5.0,
    subject_weight_kg=40.0,
    cyp2c9_genotype="*1/*3",
))

# Bayesian individual fitting from sparse C(t) observations
fit = pipeline.fit_individual(
    SimulationRequest(smiles=warfarin, dose_mg=5.0),
    observations=[(1.0, 0.15), (4.0, 0.13), (12.0, 0.05)],  # (time_h, conc_mg_L)
)
```

### CLI

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
│   ├── models/foundation/  #   Patient encoder, covariate scaling, Bayesian fitting
│   ├── applicability.py    #   Applicability domain filter (prodrug detection)
│   └── evaluation/         #   Benchmarks, metrics, conformal calibration
├── screening/              # Batch screening engine (batch_predict, rank_results)
├── uncertainty/            # Conformal UQ (LHS parameter sampling)
├── core/                   # 35-state ODE engine (body.py, organ.py)
├── drugs/                  # Drug dataclass, named IVIVE scaling constants
├── prediction/             # pKa prediction (RDKit SMARTS), bioavailability
├── clinical/               # NCA, DDI, allometry, IVIVE, pharmacogenomics
├── population/             # Virtual population simulation (LHS CYP activity + allometry)
└── cli.py                  # CLI (typer)
```

## Training Data

| Source | Purpose | Samples |
|--------|---------|---------|
| TDC PPBR_AZ | XGBoost fup | 1,614 |
| TDC Clearance_Hepatocyte_AZ | XGBoost CLint (+18 clinical anchors) | 1,231 |
| TDC VDss_Lombardo | XGBoost VDss (+2 clinical anchors) | 1,130 |
| adme_reference.csv | XGBoost RBP + ADME calibration | 153 |
| PK-DB timecourses | C(t) validation | 16 drugs |
| FDA label extraction | Gold/Silver-tier PK parameters | 296 drugs |
| Reference database | Unified multi-tier validation | 285 drugs |

## Roadmap

| Phase | Milestone | Status |
|-------|-----------|--------|
| **PK (current)** | SMILES &rarr; PK via hybrid mechanistic-ML | Gold AAFE 1.50 [1.32, 1.74], external 2.95 |
| **Rigor (v7)** | Bootstrap CI, ablation, error cancellation analysis | Complete |
| **Structural** | pKa integration, acid-Kp D-fix, CYP3A4 gut wall guard | Partially complete |
| **PK/PD** | Efficacy/toxicity endpoints from PK profiles | Future |
| **Digital Twin** | Patient-specific multi-organ physiological model | Future |

## Development

```bash
pip install -e ".[dev]"

# Core test suite
pytest tests/ -m "not slow and not benchmark" -q       # ~48K fast tests
pytest tests/ml/test_accuracy_regression.py -v          # Accuracy regression (5 drugs)

# Gold-tier regression gate
pytest tests/regression/test_gold24_regression.py \
    -v -m benchmark                                     # AAFE ≤ 1.70, ≥75% 2-fold, latency < 500ms

# Benchmarking
python scripts/run_full_benchmark.py                    # 24-drug Gold benchmark (with bootstrap CI)
python scripts/run_expanded_benchmark.py                # 285-drug expanded
python scripts/run_ablation.py                          # Ablation study (component contributions)

# Quality
ruff check . && ruff format --check .                   # Lint + format
```

Pre-commit hook runs `ruff format` and `ruff check` automatically.

## Contributing

1. Fork and create a feature branch
2. Install dev dependencies: `pip install -e ".[dev]"`
3. Write tests first (TDD)
4. Run `ruff format . && ruff check .` before committing
5. Run regression gates: `pytest tests/ml/test_accuracy_regression.py && pytest tests/regression/test_gold24_regression.py -m benchmark`
6. Open a PR against `main`

## Citation

If you use Omega in your research, please cite:

```bibtex
@software{omega_pbpk,
  title  = {Omega: Structure-Based Pharmacokinetic Prediction
            via Hybrid Mechanistic-ML Modeling},
  author = {Omega Contributors},
  url    = {https://github.com/jam-sudo/Omega},
  year   = {2026}
}
```

## License

[MIT](LICENSE)
