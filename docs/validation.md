# Validation — Omega PBPK v0.9

## Test suite

48,000+ unit/integration/E2E tests (as of March 2026). Key test modules:

| Test file | Count | Coverage |
|-----------|-------|----------|
| test_all.py | 87 | Drug dataclass, organs, PBPK engine, DDI, population, config, ADME, PD, safety, PGx, visualization |
| test_e2e.py | 13 | Full pipeline: SMILES → ADME → PBPK → DDI → PopPK → report |
| test_pgx_pbpk.py | 28 | PGx-stratified PBPK (PM/IM/NM/UM phenotypes) |
| test_pharmacogenomics.py | 25 | CYP phenotype analysis, CPIC allele database |
| test_pd_models.py | 28 | Emax, effect-site, indirect response PD models |
| test_calibration.py | 13 | Bayesian MCMC calibration |
| test_ddi.py | 13 | Competitive inhibition, MBI, induction |
| test_off_target.py | 21 | hERG, CYP inhibition safety panel |
| test_allometry.py / test_allometry_ivive.py | — | Allometric scaling, IVIVE |
| test_clinical.py | — | Dose optimization, formulation comparison |
| test_benchmark.py | — | Literature-derived benchmark validation |
| (other files) | — | API, CLI, pop simulator, surrogate, sensitivity |

## Mass balance validation

IV bolus: `dose = sum(all 35 states)` verified within ± 0.5% at all time points.

## Benchmark datasets

Literature-derived clinical concentration–time profiles in `benchmarks/datasets/`:

- `caffeine_oral_100mg.csv` — Arnaud (1993), Nehlig (2016)
- `warfarin_oral_5mg.csv` — Holford (1986)
- `metoprolol_oral_100mg.csv` — Regardh (1980)
- `midazolam_oral_2mg.csv` — Greenblatt (1992), Thummel (1996)
- `propranolol_oral_80mg.csv` — Walle (1985)

These datasets are generated from published PK parameters with ±15% lognormal variability
(seed=0) to simulate published mean ± SD profiles.

## Acceptance criteria

Thresholds in `benchmarks/expected/acceptance.json` follow FDA/EMA PBPK guidelines:

| Metric | Threshold | Basis |
|--------|-----------|-------|
| AUC relative error | ≤ 0.80 (2-fold) | EMA/FDA PBPK guideline (2018) |
| Cmax relative error | ≤ 0.80 (2-fold) | EMA/FDA PBPK guideline (2018) |
| Tmax absolute error | ≤ 3.0 h | Practical clinical relevance |

A 2-fold accuracy criterion (50–200% of observed) is the accepted standard for PBPK model
validation per the FDA Draft Guidance on PBPK (2018) and EMA Guideline on the reporting of
PBPK modelling and simulation (2018).

## Running tests

```bash
# Unit tests
python -m pytest tests/ -v --tb=short

# Benchmark suite
omega benchmark

# Linting
python -m ruff check .
python -m ruff format --check .
```
