# Validation — Omega PBPK v0.7

## Test suite

50 unit/integration tests in `tests/test_all.py` covering:

| Test class | Count | Coverage |
|------------|-------|----------|
| TestDrug | 5 | Drug dataclass, IVIVE scaling, Kp defaults |
| TestMidazolam | 3 | Reference compound parameters |
| TestOrgan | 3 | Organ volumes, permeability-limited split |
| TestWholeBodyPBPK | 8 | IV/oral simulation, mass balance, Cmax/AUC |
| TestDDI | 3 | Competitive inhibition, MBI, induction |
| TestPopulation | 3 | Virtual population generation, statistics |
| TestConfig | 3 | YAML loading, Drug construction |
| TestADMEPredictor | 3 | Property prediction, confidence levels |
| TestPDModels | 5 | Emax, effect compartment, indirect response, tumor |
| TestSafety | 3 | Safety panel assessment, risk classification |
| TestPGx | 3 | Allele analysis, phenotype distribution |
| TestClinical | 3 | Dose optimization, multi-dose, formulation |
| TestGNN | 1 | MPNN scaffold instantiation |
| TestVisualization | 1 | Plot generation |
| TestIntegration | 3 | End-to-end oral simulation with PK/PD |

## Mass balance validation

IV bolus: `dose = sum(all 34 states)` verified within ± 0.5% at all time points.

## Benchmark datasets

Synthetic reference datasets in `benchmarks/datasets/`:

- `caffeine_oral_100mg.csv`
- `warfarin_oral_5mg.csv`
- `metoprolol_oral_100mg.csv`

These are synthetic datasets generated for software testing, not clinical data.

Acceptance criteria (from `benchmarks/expected/acceptance.json`):
- AUC relative error ≤ 0.30
- Cmax relative error ≤ 0.30
- Tmax absolute error ≤ 1.0 h

## Running tests

```bash
# Unit tests
python -m pytest tests/ -v --tb=short

# Linting
python -m ruff check .
python -m ruff format --check .
```
