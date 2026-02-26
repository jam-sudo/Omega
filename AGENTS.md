# Omega PBPK — Agent Instructions

## Project

Omega PBPK v0.9.0 — Whole-body PBPK simulation platform.

## Structure

- `src/omega_pbpk/` — Main package (35-state ODE engine)
  - `core/body.py` — `WholeBodyPBPK` class (35-state LSODA ODE solver)
  - `drugs/drug.py` — `Drug` dataclass
  - `pipeline/__init__.py` — `OmegaPipeline`, `CandidateReport`, `SimulationRequest`, `SimulationResult`, `SimulationResult2`
  - `features/` — Molecular feature extraction
  - `clinical/ontogeny.py` — Ontogeny (pediatric scaling)
  - `clinical/allometry.py` — Allometric scaling (Boxenbaum 1982, multi-species log-log)
  - `clinical/ivive.py` — IVIVE: microsomal/hepatocyte → in vivo CLh (well-stirred model)
  - `clinical/report.py` — HTML regulatory report generator
  - `clinical/pgx_pbpk.py` — PGx-stratified PBPK simulation
  - `prediction/adme_predictor.py` — `ADMEPredictor` (SMILES → ADME properties)
  - `population/physiology.py` — `get_species_physiology()` (human, rat, mouse, dog)
  - `population/pop_simulator.py` — PopulationSimulator (covariate-scaled parallel PBPK)
  - `surrogate/train.py` — PKSurrogate training on real PBPK ODE data
  - `visualization/vpc.py` — VPC and forest plots
- `compounds/` — YAML compound definitions (midazolam, warfarin, propranolol, metformin, caffeine)
- `tests/test_all.py` — 400+ tests
- `tests/test_e2e.py` — 13 end-to-end integration tests (SMILES → PK → report)
- `pyproject.toml` — Package config, entry point: `omega = omega_pbpk.cli:main`

## CLI Commands

```
omega simulate     — Run PBPK simulation (IV or oral) from compound YAML
omega predict      — SMILES → full PK simulation (ADME + PBPK)
omega multidose    — Multi-dose steady-state simulation
omega optimize     — Therapeutic window dose optimization
omega safety       — Off-target safety panel
omega pgx          — Pharmacogenomics analysis
omega calibrate    — Bayesian MCMC parameter calibration
omega benchmark    — Multi-drug benchmark validation suite
omega sensitivity  — Local sensitivity analysis
omega validate     — Mass balance and physiological sanity checks
omega surrogate    — Train neural surrogate model on real PBPK data
omega uncertainty  — Monte Carlo uncertainty propagation
omega evaluate     — Integrated drug candidate evaluation
omega population   — Population PK simulation (virtual subjects)
omega report       — Generate HTML regulatory report
omega pgx-sim      — PGx-stratified PBPK (PM/IM/NM/UM AUC ratios)
omega test         — Run test suite
```

### New CLI: omega predict

```bash
omega predict --smiles "Cn1cnc2c1c(=O)n(C)c(=O)n2C" --dose 100 --route oral --duration 24
```

Options:
- `--smiles / -s` — SMILES string of the drug (required)
- `--dose / -d` — Dose in mg (default: 100.0)
- `--route / -r` — Route: oral or iv (default: oral)
- `--duration` — Simulation duration in hours (default: 24.0)

## New API Endpoints (v0.8)

- `POST /predict/new-molecule` — SMILES → PK simulation via OmegaPipeline
- `POST /predict/uncertainty` — Monte Carlo uncertainty propagation for new molecules

## Development

```bash
pip install -e ".[dev]"
pytest tests/ -v
ruff check src/
ruff format src/
```

## Technical Rules

- ODE solver: `solve_ivp`, method='LSODA', rtol=1e-8, atol=1e-10
- State vector: 35 states (see `core/body.py` header for layout)
- Mass balance: IV dose = sum(all states) ± 0.5%
- Units: mg/L (concentration), L (volume), L/h (flow/clearance), h (time), mg (dose)
- Drug parameters: always use Drug dataclass, never raw dicts
- YAML loading: yaml.safe_load only, through config.py
- All new features require tests
- Species support: human (default), rat, mouse, dog via `get_species_physiology()`

## Key Classes & API

### WholeBodyPBPK

```python
from omega_pbpk.core.body import WholeBodyPBPK
model = WholeBodyPBPK(drug=drug, body_weight=70.0)
model.setup_oral(dose_mg=100.0)   # or setup_iv(), setup_sc()
result = model.simulate(t_end_h=24.0)
cp = result.plasma_concentration()   # method → NDArray mg/L
pk = result.pk_summary()             # dict with Cmax, Tmax, AUC, t½, CL, Vss
```

### OmegaPipeline (SMILES → PK)

```python
from omega_pbpk.pipeline import OmegaPipeline, SimulationRequest
pipeline = OmegaPipeline()
result = pipeline.simulate(SimulationRequest(
    smiles="Cn1cnc2c1c(=O)n(C)c(=O)n2C",
    dose_mg=100.0,
    route="oral",
    duration_h=24.0,
))
# result.cmax_mg_L, result.tmax_h, result.auc0t_mg_h_L, result.t_half_h
# result.adme_properties, result.confidence, result.warnings
```

### CandidateReport (evaluate_candidate)

```python
from omega_pbpk.pipeline import evaluate_candidate
report = evaluate_candidate(drug=drug, dose_mg=10.0, route="oral")
# report.overall_score, report.pk_summary, report.risk_flags, etc.
```

### Key APIs — Clinical Tools (v0.9)

```python
# NCA
from omega_pbpk.clinical import run_nca
result = run_nca(time_h, conc_mg_L, dose_mg=100.0)
# result.auc0t_mg_h_L, result.cmax_mg_L, result.t_half_h, result.cl_L_per_h

# DDI
from omega_pbpk.clinical import assess_ddi_risk, DDIInhibitor
inhibitor = DDIInhibitor(name="Itraconazole", cmax_uM=0.2, ki_3a4_uM=0.0013)
report = assess_ddi_risk(inhibitor)

# Population PK
from omega_pbpk.population.pop_simulator import PopulationSimulator
result = PopulationSimulator(drug).run(n_subjects=100, dose_mg=10.0, route="oral", t_end_h=24.0)

# Allometry
from omega_pbpk.clinical import predict_human_from_preclinical
pred = predict_human_from_preclinical(
    smiles="...", preclinical_data={"rat": (0.8, 0.25), "dog": (5.0, 12.0)}
)

# IVIVE
from omega_pbpk.clinical import scale_microsomal_clint
result = scale_microsomal_clint(clint_uL_min_mg=12.0, fup=0.1, q_liver_L_per_h=96.6)
```

## Coding Conventions

- Python 3.10+ with full type hints on public functions.
- Keep functions small, side effects explicit, module boundaries clear.
- Run formatting/lint with `ruff` and type checks with `mypy`.

## Commit Convention

`type(scope): description`
- type: feat, fix, refactor, test, docs, chore
- scope: core, drugs, population, prediction, docking, qsp, pgx, clinical, api, cli, pipeline, features, data
