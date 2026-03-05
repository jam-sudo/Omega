# Omega PBPK

**Whole-body physiologically-based pharmacokinetic (PBPK) simulation platform.**

![Tests](https://img.shields.io/badge/tests-passing-brightgreen)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

## Overview

Omega PBPK is a research-grade, whole-body pharmacokinetic simulation platform built around a 35-state ODE engine that models drug distribution across 15 organs, 4 permeability-limited tissues, and an 8-segment ACAT intestinal absorption model. Starting from a SMILES string or a compound YAML definition, the platform automatically predicts ADME properties, constructs a parameterized PBPK model, and integrates the ODEs using a high-accuracy LSODA solver (rtol=1e-8, atol=1e-10) to produce full plasma and tissue concentration-time profiles.

Beyond core simulation, Omega PBPK provides an end-to-end drug-development toolkit: population PK with NHANES-based virtual subjects, pediatric CYP ontogeny scaling, DDI risk assessment following the FDA 2020 static mechanistic model, non-compartmental analysis (NCA), allometric scaling, IVIVE, pharmacogenomics-stratified PBPK, a neural surrogate model, Bayesian MCMC calibration, a regulatory-grade HTML report generator, a REST API, and a multi-compound benchmark validation suite. All features are accessible through both a Python API and a unified CLI.

> **Safety Scope:** For computational research and model prototyping only. Not validated for clinical decision making. No real-world dosing recommendations are provided.

The core simulation engine integrates a 35-state ODE system representing 13 perfusion-limited organs (lung, brain, heart, kidney, liver, spleen, gut wall, pancreas, thymus, reproductive, rest) plus 4 permeability-limited tissues (adipose, muscle, bone, skin) and an 8-segment ACAT gastrointestinal tract. Tissue-to-plasma partition coefficients are predicted using the Rodgers & Rowland method from drug physicochemical properties (logP, pKa, fup, drug type). Hepatic clearance is modeled via the well-stirred equation with unbound fraction (fup) and intrinsic clearance (CLint), gut-wall first-pass extraction (f_gut) is applied at the intestinal wall, and renal clearance routes drug directly to a urine sink compartment. Pharmacodynamic coupling is available through an Emax model with optional ke0 effect-compartment delay for hysteresis modeling.

## Features

- 35-state whole-body PBPK ODE engine (LSODA, rtol=1e-8, atol=1e-10)
- SMILES -> ADME prediction -> full PK simulation pipeline
- Population PK with NHANES-based virtual subjects
- Pediatric dosing with CYP ontogeny (CYP3A4/2D6/1A2)
- DDI risk assessment (FDA 2020 static mechanistic model)
- Non-compartmental analysis (NCA): AUC, Cmax, t1/2, CL, Vss
- Allometric scaling (Boxenbaum 1982, multi-species log-log)
- In vitro-in vivo extrapolation (IVIVE): microsomal and hepatocyte CLint -> CLh
- Pharmacogenomics: CYP2D6/2C19/2C9/3A5/1A2 genotype -> PK stratification
- Neural surrogate model (NumPy MLP) trained on real PBPK ODE simulation data
- Bayesian MCMC parameter calibration against observed clinical data
- Multi-compound benchmark validation suite (5 reference compounds)
- HTML regulatory report generator
- REST API via FastAPI
- Multi-species support: human, rat, mouse, dog

## Installation

```bash
git clone https://github.com/jam-sudo/Omega.git
cd Omega
pip install -e ".[dev]"
```

To enable the REST API:

```bash
pip install -e ".[api]"
```

## Quick Start

### Python API

```python
from omega_pbpk.pipeline import OmegaPipeline, SimulationRequest

pipeline = OmegaPipeline()
result = pipeline.simulate(SimulationRequest(
    smiles="Cn1cnc2c1c(=O)n(C)c(=O)n2C",  # caffeine
    dose_mg=100.0,
    route="oral",
    duration_h=24.0,
))
print(f"Cmax: {result.cmax_mg_L:.3f} mg/L")
print(f"AUC:  {result.auc0t_mg_h_L:.2f} mg·h/L")
print(f"t½:   {result.t_half_h:.1f} h")
```

### CLI

```bash
# SMILES -> full PK
omega predict --smiles "Cn1cnc2c1c(=O)n(C)c(=O)n2C" --dose 100 --route oral

# Simulate from compound YAML
omega simulate --compound compounds/midazolam.yaml --dose-mg 2 --route oral

# Population PK (100 virtual subjects)
omega population --compound compounds/warfarin.yaml --n-subjects 100 --dose 5

# Generate regulatory HTML report
omega report --smiles "SMILES" --name "Drug X" --dose 100 --out report.html

# DDI risk assessment / PGx stratified simulation
omega pgx-sim --smiles "SMILES" --gene CYP2D6 --dose 100

# Bayesian calibration against observed data
omega calibrate --compound compounds/midazolam.yaml --observed data/observed.csv --dose 2

# Benchmark validation (5 reference compounds)
omega benchmark
```

## CLI Reference

| Command | Description |
|---------|-------------|
| `omega simulate` | Run PBPK simulation (IV or oral) from a compound YAML file |
| `omega predict` | SMILES -> full PK simulation via ADME prediction + PBPK engine |
| `omega multidose` | Multi-dose steady-state simulation |
| `omega optimize` | Therapeutic window dose optimization |
| `omega safety` | Off-target safety panel (11 FDA targets + 5 CYP inhibition) |
| `omega pgx` | Pharmacogenomics analysis (CYP allele database, CPIC standard) |
| `omega calibrate` | Bayesian MCMC parameter calibration against observed PK data |
| `omega benchmark` | Multi-drug benchmark validation suite (5 reference compounds) |
| `omega sensitivity` | Local sensitivity analysis |
| `omega validate` | Mass balance and physiological sanity checks |
| `omega surrogate` | Train or use the neural surrogate model on real PBPK data |
| `omega uncertainty` | Monte Carlo uncertainty propagation |
| `omega evaluate` | Integrated drug candidate evaluation with risk flags |
| `omega population` | Population PK simulation with NHANES-based virtual subjects |
| `omega report` | Generate HTML regulatory report |
| `omega pgx-sim` | PGx-stratified PBPK: PM/IM/NM/UM AUC profiles |
| `omega test` | Run the test suite |

## Python API Reference

### 1. Core PBPK: WholeBodyPBPK

The low-level 35-state ODE engine. Use this when you have a fully parameterized `Drug` object.

```python
from omega_pbpk.core.body import WholeBodyPBPK
from omega_pbpk.drugs.drug import Drug

drug = Drug(name="Midazolam", mw=325.77, logP=3.89, fup=0.032, ...)
model = WholeBodyPBPK(drug=drug, body_weight=70.0)
model.setup_oral(dose_mg=5.0)   # or setup_iv(), setup_sc()
result = model.simulate(t_end_h=24.0)

cp = result.plasma_concentration()   # NDArray of mg/L values
pk = result.pk_summary()             # dict: Cmax, Tmax, AUC, t½, CL, Vss
```

### 2. Pipeline: OmegaPipeline

High-level SMILES-to-PK pipeline. Predicts ADME properties automatically then runs PBPK.

```python
from omega_pbpk.pipeline import OmegaPipeline, SimulationRequest

pipeline = OmegaPipeline()
result = pipeline.simulate(SimulationRequest(
    smiles="CC(C)NCC(O)COc1cccc2ccccc12",  # propranolol
    dose_mg=80.0,
    route="oral",
    duration_h=24.0,
))
# Available on result:
# result.cmax_mg_L, result.tmax_h, result.auc0t_mg_h_L, result.t_half_h
# result.adme_properties, result.confidence, result.warnings
```

Evaluate a candidate drug for development:

```python
from omega_pbpk.pipeline import evaluate_candidate

report = evaluate_candidate(drug=drug, dose_mg=10.0, route="oral")
# report.overall_score, report.pk_summary, report.risk_flags
```

### 3. Clinical Tools

#### Non-Compartmental Analysis (NCA)

```python
from omega_pbpk.clinical import run_nca

result = run_nca(time_h, conc_mg_L, dose_mg=100.0)
# result.auc0t_mg_h_L — AUC from 0 to t (linear-log trapezoidal)
# result.cmax_mg_L    — peak plasma concentration
# result.t_half_h     — terminal elimination half-life
# result.cl_L_per_h   — apparent oral clearance
```

#### DDI Risk Assessment

```python
from omega_pbpk.clinical import assess_ddi_risk, DDIInhibitor

inhibitor = DDIInhibitor(name="Itraconazole", cmax_uM=0.2, ki_3a4_uM=0.0013)
report = assess_ddi_risk(inhibitor)
# FDA 2020 static mechanistic model: competitive inhibition + MBI
```

#### Allometric Scaling

```python
from omega_pbpk.clinical import predict_human_from_preclinical

pred = predict_human_from_preclinical(
    smiles="CC(C)NCC(O)COc1cccc2ccccc12",
    preclinical_data={"rat": (0.8, 0.25), "dog": (5.0, 12.0)},
)
# Boxenbaum 1982 method + multi-species log-log regression
```

#### IVIVE: Microsomal CLint Scaling

```python
from omega_pbpk.clinical import scale_microsomal_clint

result = scale_microsomal_clint(
    clint_uL_min_mg=12.0,
    fup=0.1,
    q_liver_L_per_h=96.6,
)
# Well-stirred model: in vitro CLint -> in vivo hepatic CLh
```

### 4. Population PK

```python
from omega_pbpk.population.pop_simulator import PopulationSimulator

sim = PopulationSimulator(drug)
result = sim.run(
    n_subjects=100,
    dose_mg=10.0,
    route="oral",
    t_end_h=24.0,
)
# result contains per-subject PK profiles and summary statistics
```

### 5. Bayesian MCMC Calibration

```python
from omega_pbpk.calibration import run_mh_calibration

samples = run_mh_calibration(
    drug=drug,
    observed_time_h=time_array,
    observed_conc_mg_L=conc_array,
    n_iterations=5000,
)
# Metropolis-Hastings MCMC; returns posterior parameter samples
```

## Supported Species

| Species | Body Weight | Reference |
|---------|-------------|-----------|
| Human | 70 kg | ICRP Publication 89 (2002) |
| Rat | 0.25 kg | Brown et al., Toxicol Sci, 1997 |
| Mouse | 0.02 kg | Brown et al., Toxicol Sci, 1997 |
| Dog | 10 kg | Davies & Morris, Pharm Res, 1993 |

Species selection via `get_species_physiology(species="rat")`. Organ volumes scale as BW^1.0 (brain: BW^0.7); blood flows scale as BW^0.75.

## Reference Compounds

| Compound | YAML | Primary CYP | fup | Literature CL (L/h) |
|----------|------|-------------|-----|---------------------|
| Midazolam | `compounds/midazolam.yaml` | CYP3A4 (93%) | 0.032 | 27-35 |
| Warfarin | `compounds/warfarin.yaml` | CYP2C9 (80%) | 0.005 | 0.15-0.20 |
| Propranolol | `compounds/propranolol.yaml` | CYP2D6 (70%) | 0.13 | 60-80 |
| Metformin | `compounds/metformin.yaml` | None (renal) | 0.99 | 25-35 (CLr) |
| Caffeine | `compounds/caffeine.yaml` | CYP1A2 (95%) | 0.65 | 1.5-2.0 |

## Architecture

### 35-State ODE Model

The core ODE engine (`src/omega_pbpk/core/body.py`) integrates a 35-state system using `scipy.integrate.solve_ivp` with method='LSODA', rtol=1e-8, atol=1e-10.

**State vector layout:**

| Index | Compartment | Type |
|-------|-------------|------|
| 0 | venous_blood | Blood |
| 1 | arterial_blood | Blood |
| 2-12 | lung, brain, heart, kidney, liver, spleen, gut_wall, pancreas, thymus, reproductive, rest | Perfusion-limited organs |
| 13-20 | adipose_vasc/extra, muscle_vasc/extra, bone_vasc/extra, skin_vasc/extra | Permeability-limited (vascular + extravascular) |
| 21-28 | stomach through colon lumen | ACAT 8-segment absorption |
| 29 | portal_vein | Portal circulation |
| 30 | metabolized_hepatic | Mass balance sink |
| 31 | excreted_renal | Mass balance sink |
| 32 | metabolized_gut | Mass balance sink |
| 33 | excreted_fecal | Mass balance sink |
| 34 | sc_depot | Subcutaneous absorption depot |

**Key physiological parameters (70 kg human, ICRP Reference Man):**

- Cardiac output: 390 L/h
- Liver volume: 1.80 L
- Muscle volume: 28.0 L
- Adipose volume: 14.5 L
- GFR: 7.5 L/h (~125 mL/min)

**Mass balance:** For IV dosing, sum of all 35 states equals dose at all times (verified to within ±0.5%).

### Module Map

```
src/omega_pbpk/
├── core/               # 35-state ODE engine (body.py, organ.py)
├── drugs/              # Drug dataclass
├── pipeline/           # OmegaPipeline, SimulationRequest, SimulationResult
├── features/           # Molecular feature extraction from SMILES
├── clinical/           # NCA, DDI, allometry, IVIVE, HTML report, PGx PBPK, ontogeny
├── prediction/         # ADMEPredictor (SMILES -> 9 ADME properties)
├── population/         # ICRP physiology, virtual population, PopulationSimulator
├── surrogate/          # Neural surrogate model (NumPy MLP) training
├── visualization/      # VPC and forest plots
├── api/                # FastAPI REST endpoints
├── config.py           # YAML loader (yaml.safe_load)
└── cli.py              # 17-command CLI (typer)
```

### REST API

Start the server:

```bash
uvicorn omega_pbpk.api.server:app --reload
```

Key endpoints:

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/predict/new-molecule` | SMILES -> PK simulation via OmegaPipeline |
| POST | `/predict/uncertainty` | Monte Carlo uncertainty propagation |

## Validation

| Metric | Value | Target |
|--------|-------|--------|
| Midazolam IV AAFE | 1.62 | < 2.0 |
| Midazolam DDI AUC ratio | 9.4x | obs 10-16x |
| PopPK CV (Cmax) | 27% | 20-40% |
| Mass balance (IV) | ~100% | ±0.5% |
| Test suite | 400+ tests | all pass |

## Development

```bash
pip install -e ".[dev]"
pytest tests/ -v
ruff check src/
ruff format src/
mypy src/omega_pbpk/ --ignore-missing-imports
```

End-to-end integration tests:

```bash
pytest tests/test_e2e.py -v
```

## License

MIT
