# Roadmap

Omega remains the **mechanistic engine**. Product/service layers stay separate.

## M2 — Real-World Candidate Evaluation (Tool Mode)

- Population simulation for `N` virtual subjects
- Genotype modifiers (for example CYP polymorphism effects as CLint multipliers)
- Risk flags and scoring expansion
- Standardized product-facing evaluation report (non-academic)

## M3 — QSP Layer (Mechanistic Response)

- QSP plugin system (`base`, `registry`, `models/`)
- Turnover biomarker model as first QSP module
- Optional coupled PK-QSP solving
- QSP validation against biomarker datasets

## M4 — Productization Readiness

- API service wrapper in a separate module
- Job queue and persistent artifact storage
- Model registry and strict version pinning
- Audit trail and permissions model
- Optional ML surrogate model integration as a separate package

The architecture boundary is explicit: Omega continues as the mechanistic simulation core, while deployment/product concerns live outside the core engine package.
