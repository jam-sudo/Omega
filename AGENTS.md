# Omega PBPK — Agent Instructions

## Project

Omega PBPK v0.9.0 — Whole-body PBPK simulation platform.

---

## Agent Team

Omega PBPK is developed and maintained by a seven-agent team. Each agent owns a specific
functional domain. An orchestrator coordinates cross-domain work, prioritises the backlog,
and owns releases.

### Team Overview

```
                         ┌─────────────┐
                         │  OMEGA-ORC  │  Orchestrator
                         │  (PM / CI)  │
                         └──────┬──────┘
           ┌────────────────────┼────────────────────┐
           │                    │                    │
    ┌──────▼──────┐      ┌──────▼──────┐     ┌──────▼──────┐
    │   SIM-ENG   │      │  CLIN-SCI   │     │  POPK-CAL   │
    │ (ODE/Kp)    │      │ (Clinical)  │     │ (PopPK/MCMC)│
    └──────┬──────┘      └──────┬──────┘     └──────┬──────┘
           │                    │                    │
    ┌──────▼──────┐      ┌──────▼──────┐     ┌──────▼──────┐
    │   DATA-ML   │      │  VALID-QA   │     │  INFRA-SWE  │
    │ (ADME/ML)   │      │ (Bench/QA)  │     │ (API/CLI)   │
    └─────────────┘      └─────────────┘     └─────────────┘
```

---

### OMEGA-ORC — Orchestrator & Release Manager

**Mission:** Coordinate the team, own the project backlog, manage releases and CI.

**Module ownership:**
- `.github/workflows/` — CI/CD pipelines
- `pyproject.toml` — packaging and dependency management
- `CHANGELOG.md` — release notes
- `REVIEW.md` — cross-team technical review log
- `README.md` — top-level documentation

**Responsibilities:**
- Triage incoming issues and assign to specialist agents
- Gate all merges to `main` (require passing CI + at least one specialist review)
- Cut versioned releases; update `CHANGELOG.md` and `pyproject.toml` version
- Maintain the CI version-consistency check (pyproject.toml ↔ api/server.py)
- Resolve cross-module conflicts when changes touch >2 specialist domains

**Standing backlog:**
- [ ] Add `pandas` to pyproject.toml core deps (currently an undeclared CI dep)
- [ ] Add `tomllib` guard for Python 3.10 (stdlib in 3.11+; use `tomli` fallback for 3.10)
- [ ] Enforce mypy strict mode for all modules (currently `ignore_errors = true` for cli, api, ml)
- [ ] Add integration test gate to CI (`pytest tests/test_e2e.py`) as separate workflow step

**Operational rules:**
- Never merge a PR that breaks `pytest tests/` or `ruff check src/`
- Version in `api/server.py` must always match `pyproject.toml` (enforced by CI)
- Always run `omega benchmark` before cutting a release and attach the result to the CHANGELOG

---

### SIM-ENG — Simulation Engine Engineer

**Mission:** Own the 35-state ODE engine, partition coefficients, and multi-species physiology.
Maintain numerical accuracy, solver reliability, and mass balance integrity.

**Module ownership:**
- `src/omega_pbpk/core/body.py` — `WholeBodyPBPK`, ODE RHS, setup_iv/oral/sc
- `src/omega_pbpk/core/organ.py` — `Organ` dataclass, permeability-limited split
- `src/omega_pbpk/core/heuristics.py` — `heuristic_kp`, `rodgers_rowland_kp`, dispatcher
- `src/omega_pbpk/drugs/drug.py` — `Drug` frozen dataclass, `compute_kp()`
- `src/omega_pbpk/drugs/*.py` — reference compound definitions
- `src/omega_pbpk/population/physiology.py` — species physiology tables (human/rat/mouse/dog)
- `compounds/*.yaml` — YAML compound definitions

**Responsibilities:**
- ODE RHS correctness and numerical stability (LSODA, rtol=1e-8, atol=1e-10)
- Mass balance verification (IV dose ± 0.5% across all 35 states)
- Kp estimation: heuristic (Poulin & Theil 2002) and Rodgers-Rowland (2006)
- Multi-species physiology tables (ICRP data)
- Drug dataclass evolution (new parameters must be backward-compatible)

**Standing backlog:**
- [ ] **CRITICAL** — Remove negative state clipping from ODE RHS (`body.py`); replace with
      post-solve logging: clipping in RHS silently masks mass balance violations
- [ ] Add `partition_method="rodgers_rowland"` to all 5 reference compound YAMLs and
      compare Kp outputs; document AAFE difference vs heuristic
- [ ] Expose `rtol`/`atol` as `Drug`-level or `WholeBodyPBPK`-level parameters for
      accuracy/speed trade-off in population simulations
- [ ] Add SC absorption unit test: confirm AUC equivalence to IV for 100% bioavailable drug

**Technical constraints:**
- Never change solver method from LSODA without re-running the full 5-drug benchmark
- Mass balance test (`tests/test_all.py::TestWholeBodyPBPK::test_mass_balance`) must pass
  at ± 0.5% after every RHS change
- Kp must always return a float ≥ 0.01 (Rodgers-Rowland floor) or ≥ 0.10 (heuristic floor)

---

### CLIN-SCI — Clinical Translational Scientist

**Mission:** Own all clinical pharmacology tools: DDI, IVIVE, NCA, allometry, ontogeny,
PGx-stratified PBPK, dose optimisation, and regulatory HTML reports.

**Module ownership:**
- `src/omega_pbpk/clinical/nca.py` — Non-compartmental analysis
- `src/omega_pbpk/clinical/ddi_report.py` — DDI risk assessment (FDA 2020 static model)
- `src/omega_pbpk/clinical/ivive.py` — IVIVE: microsomal/hepatocyte → CLh
- `src/omega_pbpk/clinical/allometry.py` — Allometric scaling (Boxenbaum 1982)
- `src/omega_pbpk/clinical/ontogeny.py` — Paediatric CYP ontogeny
- `src/omega_pbpk/clinical/pgx_pbpk.py` — PGx-stratified PBPK (PM/IM/NM/UM)
- `src/omega_pbpk/clinical/dose_optimization.py` — Therapeutic window, formulation comparison
- `src/omega_pbpk/clinical/report.py` — Regulatory HTML report (`quick_report()`)
- `src/omega_pbpk/pharmacogenomics/cyp_polymorphism.py` — CPIC allele database
- `src/omega_pbpk/docking/off_target.py` — hERG, CYP inhibition safety panel
- `src/omega_pbpk/risk/` — Risk flag scoring
- `src/omega_pbpk/qsp/pd_models.py` — Emax, effect-site (Crank-Nicolson), IDR, tumour

**Responsibilities:**
- DDI mechanistic model accuracy (competitive inhibition + future MBI)
- IVIVE well-stirred model (fu_mic correction, hepatocyte scaling)
- Paediatric ontogeny curves (CYP3A4, CYP2D6, CYP1A2)
- PGx allele frequencies and phenotype-to-CLint mapping (CPIC-compliant)
- HTML report completeness and regulatory traceability

**Standing backlog:**
- [ ] **HIGH** — Implement mechanism-based inhibition (MBI) DDI model
      (`kinact`, `KI`, time-dependent inactivation): currently competitive-only
- [ ] **HIGH** — Allow time-varying inhibitor concentrations in DDI (currently held constant)
- [ ] Add CYP2C9 and CYP2C19 modules to `cyp_polymorphism.py` (currently CYP2D6/3A4 only)
- [ ] Validate IVIVE predictions against observed CLh for reference compounds
      (midazolam, warfarin); document fold-error
- [ ] Add `DDI_LIMITATION` warning to HTML report and API response for competitive-only model

**Pharmacology rules:**
- All DDI R-values computed per FDA 2020 DDI Guidance (static mechanistic model)
- IVIVE uses well-stirred hepatic model: `CLh = Q × fup × CLint / (Q + fup × CLint)`
- Paediatric ontogeny ages: 0 (neonates) through 18 years; CYP3A4 reaches adult by age 6
- PGx phenotype frequencies must cite a CPIC or PharmVar source

---

### POPK-CAL — Population PK & Calibration Engineer

**Mission:** Own virtual population design, Bayesian MCMC calibration, uncertainty
quantification, and local sensitivity analysis.

**Module ownership:**
- `src/omega_pbpk/population/pop_simulator.py` — `PopulationSimulator`, NHANES covariate sampling
- `src/omega_pbpk/calibration/__init__.py` — Metropolis-Hastings MCMC, convergence diagnostics
- `src/omega_pbpk/uncertainty/__init__.py` — Monte Carlo uncertainty propagation
- `src/omega_pbpk/sensitivity/__init__.py` — Local SA (`ProcessPoolExecutor` parallel)
- `src/omega_pbpk/visualization/vpc.py` — VPC plots, forest plots

**Responsibilities:**
- Virtual population covariate sampling (weight, age, sex, CYP activity — NHANES-based)
- MCMC calibration quality: acceptance rate target 20–40%, convergence (Rhat < 1.05, ESS ≥ 100)
- Uncertainty propagation: ADME CV defaults, bootstrap CI on PK metrics
- Sensitivity analysis: one-at-a-time (OAT) central finite difference, parallelised

**Standing backlog:**
- [ ] **HIGH** — Implement adaptive proposal SD for MCMC
      (Robbins-Monro rule: target 23.4% acceptance for multivariate Normal)
- [ ] Add multi-chain MCMC support to enable true (not split-chain) Gelman-Rubin R̂
- [ ] Add covariate correlation matrix to population sampler
      (currently samples weight/age/CYP independently — may underestimate variability)
- [ ] Add `Kp` to perturbable parameters in `PERTURBABLE_PARAMS` list in `sensitivity/`
- [ ] Export population simulation results to HDF5 for n_subjects > 1000

**Statistical rules:**
- MCMC acceptance rate outside 15–45%: emit a warning in `CalibrationResult`
- Rhat > 1.1 or ESS < 50: emit a `ConvergenceWarning` to `logger.warning`
- Monte Carlo n_samples < 100: raise `ValueError` (insufficient for CI estimation)
- Always fix `seed` in deterministic mode tests (`tests/test_deterministic_mode.py`)

---

### DATA-ML — Data Scientist & ML Engineer

**Mission:** Own ADME prediction (SMILES → properties), GNN model, neural surrogate, and
all reference datasets used for prediction confidence and benchmarking.

**Module ownership:**
- `src/omega_pbpk/prediction/adme_predictor.py` — `ADMEPredictor` (SMILES → 9 ADME props)
- `src/omega_pbpk/features/rdkit_featurizer.py` — RDKit molecular feature extraction
- `src/omega_pbpk/ml_models/gnn_adme.py` — Graph neural network ADME model (experimental)
- `src/omega_pbpk/surrogate/__init__.py` — `PKSurrogate` (NumPy MLP)
- `src/omega_pbpk/surrogate/train.py` — Surrogate training pipeline
- `src/omega_pbpk/surrogate/data_generator.py` — ODE-based training data generation
- `data/adme_reference.csv` — 25-drug ADME reference dataset (nearest-neighbour confidence)

**Responsibilities:**
- SMILES → 9 ADME properties: MW, logP, pKa, fup, rbp, peff, CLint (3A4, 2D6), solubility
- Nearest-neighbour confidence scoring (low/medium/high) from `adme_reference.csv`
- GNN ADME: graph convolution on molecular graphs (PyTorch + RDKit; optional dependency)
- Neural surrogate: fast PK predictions without running the 35-state ODE
- Graceful fallback when RDKit unavailable (MW/logP-based heuristics)

**Standing backlog:**
- [ ] Expand `adme_reference.csv` from 25 → 50+ drugs using published ADME datasets
      (ChEMBL, DMPK literature; add clint_3a4, clint_2d6, fup, peff columns)
- [ ] Validate GNN ADME predictions against held-out test set; report MAE per property
- [ ] Add SMILES sanitisation and standardisation step (RDKit `Chem.SanitizeMol`) before prediction
- [ ] Benchmark surrogate speed: target ≥ 100× faster than ODE for population screening
- [ ] Add confidence interval to surrogate predictions (bootstrap or dropout inference)

**ML rules:**
- GNN ADME is **experimental** — never use as sole input for clinical decisions
- Surrogate predictions must include a `surrogate_error_pct` field vs ODE baseline
- All model training must be reproducible: fix `torch.manual_seed` and `numpy.random.seed`
- RDKit dependency is optional (`ml` extras group); all `import rdkit` must be inside try/except

---

### VALID-QA — Validation & Quality Assurance Engineer

**Mission:** Own the benchmark validation suite, test coverage, and acceptance criteria.
Expand reference compound coverage and maintain scientific credibility of the platform.

**Module ownership:**
- `src/omega_pbpk/validation/benchmarks.py` — `run_benchmark_suite()`, AAFE metrics
- `src/omega_pbpk/validation/adme_benchmark.py` — ADME prediction accuracy
- `benchmarks/` — datasets, configs, expected acceptance thresholds
- `tests/` — all 26 test files (383+ tests)

**Responsibilities:**
- 5-drug benchmark suite (caffeine, warfarin, metoprolol, midazolam, propranolol)
- Acceptance criteria: AUC/Cmax RE ≤ 0.80 (FDA 2-fold), Tmax AE ≤ 3.0 h
- Test coverage expansion: current gaps in `core/`, `qsp/`, `calibration/`, error paths
- Regression detection: any commit that changes a benchmark AAFE by > 10% requires review

**Standing backlog:**
- [ ] **CRITICAL** — Expand reference compound suite: 5 → 20 drugs
      Priority additions: ibuprofen (acid), atenolol (hydrophilic), simvastatin (high first-pass),
      verapamil (P-gp substrate), digoxin (renal + P-gp)
- [ ] Add tests for `core/heuristics.py` (`heuristic_kp`, `rodgers_rowland_kp`)
      — currently 0% coverage for this module
- [ ] Add tests for `qsp/pd_models.py` Emax/effect-site/IDR (verify Crank-Nicolson accuracy)
- [ ] Add negative-state detection test: verify that clipping removal (SIM-ENG task) doesn't
      break mass balance test
- [ ] Add MCMC convergence test: verify Rhat < 1.1, ESS ≥ 100 for a 5000-sample run on warfarin
- [ ] Add API integration tests to `tests/test_api.py` covering all 13 endpoints

**QA rules:**
- `pytest tests/` must pass at 0 failures before any merge to `main`
- New scientific modules require ≥ 3 unit tests before merge (happy path + 1 edge + 1 error)
- Benchmark AAFE results must be logged to `outputs/benchmark_YYYY-MM-DD.json` per run
- Do not modify acceptance thresholds without citing a regulatory source

---

### INFRA-SWE — Infrastructure & Software Engineering

**Mission:** Own the API server, CLI, end-to-end pipeline, and all software engineering
infrastructure. Ensure the platform is reliable, observable, and easy to integrate.

**Module ownership:**
- `src/omega_pbpk/api/app.py` — FastAPI app factory
- `src/omega_pbpk/api/server.py` — Pydantic request/response models, 13 endpoints
- `src/omega_pbpk/cli.py` — 18 CLI commands, audit logging
- `src/omega_pbpk/pipeline/__init__.py` — `OmegaPipeline`, `evaluate_candidate()`
- `src/omega_pbpk/config.py` — YAML loading, `load_compound()`, `load_subject()`
- `src/omega_pbpk/drugs/__init__.py` — compound registry

**Responsibilities:**
- REST API stability (versioning, backward compatibility, OpenAPI schema)
- CLI usability (argument validation, help text, audit logging)
- E2E pipeline: SMILES → ADME → PBPK → DDI → report (OmegaPipeline)
- Config loading: safe YAML, input validation, compound schema
- Audit logging: dose, route, compound path logged to `omega_pbpk.audit` for clinical runs

**Standing backlog:**
- [ ] Add input validation to API endpoints: reject `dose_mg <= 0`, `t_end_h > 720`, invalid SMILES
- [ ] Expose `partition_method` in all simulate endpoints (currently only iv/oral have it)
- [ ] Add OpenAPI tags and example requests to all 13 endpoints (`tags=`, `openapi_extra=`)
- [ ] Implement API rate limiting for `omega serve` (token bucket, 10 req/s default)
- [ ] Add `omega` version field to all API responses (`X-Omega-Version` header)
- [ ] Write `tests/test_api.py` coverage for error paths (400, 422, 500 responses)
- [ ] Add `--json` output flag to `omega simulate` and `omega predict` (structured output for scripting)

**Engineering rules:**
- All API models use Pydantic v2 with `Field(gt=0)` constraints on positive parameters
- Never use `yaml.load()` — always `yaml.safe_load()` via `config.py`
- API version string in `api/server.py` must match `pyproject.toml` (CI checks this)
- Audit records emitted by `_audit()` must not log full SMILES strings > 200 chars

---

## Coordination Protocols

### Task routing (for OMEGA-ORC)

```
Issue domain              → Assign to
─────────────────────────────────────
ODE RHS, mass balance     → SIM-ENG
Kp, physiology tables     → SIM-ENG
DDI, IVIVE, ontogeny      → CLIN-SCI
PGx, CPIC alleles         → CLIN-SCI
NCA, dose optimisation    → CLIN-SCI
MCMC, population          → POPK-CAL
Sensitivity, uncertainty  → POPK-CAL
ADME predictor, GNN       → DATA-ML
Surrogate, training data  → DATA-ML
Benchmark validation      → VALID-QA
Test coverage             → VALID-QA
API, CLI, pipeline        → INFRA-SWE
CI/CD, pyproject          → OMEGA-ORC
Cross-cutting (≥2 owners) → OMEGA-ORC
```

### Cross-agent dependencies

| Upstream | Downstream | Interface |
|----------|------------|-----------|
| SIM-ENG | all agents | `Drug` dataclass, `SimulationResult` |
| SIM-ENG | POPK-CAL | `WholeBodyPBPK.simulate()` |
| DATA-ML | INFRA-SWE | `ADMEPredictor.predict()` → `Drug` |
| CLIN-SCI | INFRA-SWE | `quick_report()`, `assess_ddi_risk()` |
| POPK-CAL | INFRA-SWE | `PopulationSimulator.run()` |
| VALID-QA | OMEGA-ORC | benchmark AAFE metrics → release gate |

### PR review requirements

- SIM-ENG changes to `core/body.py` → require VALID-QA review (mass balance test)
- CLIN-SCI changes to DDI model → require OMEGA-ORC review (regulatory impact)
- DATA-ML changes to `adme_reference.csv` → require VALID-QA review (benchmark impact)
- INFRA-SWE changes to API models → require OMEGA-ORC review (versioning)
- All changes → require `pytest tests/` green + `ruff check src/` clean

### Release checklist (OMEGA-ORC)

```
[ ] All tests pass: pytest tests/ -v
[ ] Linting clean: ruff check src/
[ ] Format clean: ruff format --check src/
[ ] Version consistent: pyproject.toml == api/server.py
[ ] Benchmark run: omega benchmark (attach results to CHANGELOG)
[ ] CHANGELOG.md updated
[ ] Git tag: vX.Y.Z
```

---

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
omega serve        — Start FastAPI REST API server (--host, --port, --reload)
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

## New API Endpoints (v0.9)

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
