# Changelog

## v0.9.0 (2026-02-26)

### Added
- 35-state ODE engine with SC depot route (Phase 1)
- Multi-species physiology: rat, mouse, dog (Phase 1)
- Virtual population with NHANES-based covariate sampling (Phase 1)
- Pediatric CYP ontogeny: CYP3A4/2D6/1A2 (Phase 1)
- OmegaPipeline: SMILES -> ADME -> PBPK (Phase 2)
- DDI risk assessment (FDA 2020 static model) (Phase 2)
- Non-compartmental analysis (NCA) (Phase 2)
- HTML regulatory report generator (Phase 2)
- Neural surrogate model trained on real PBPK data (Phase 3)
- PopulationSimulator with parallel covariate-scaled PBPK (Phase 3)
- Allometric scaling: Boxenbaum + multi-species log-log (Phase 3)
- IVIVE: microsomal/hepatocyte CLint -> CLh (Phase 3)
- VPC and forest plots (Phase 3)
- 25-drug ADME reference dataset (Phase 4)
- 5-compound benchmark validation suite (Phase 4)
- PGx-stratified PBPK: PM/IM/NM/UM profiles (Phase 4)
- End-to-end integration tests (Phase 4)
- FastAPI REST server (Phase 5)
- Literature-derived benchmark clinical PK data (Phase 5)

### Fixed
- `np.trapz` -> `np.trapezoid` deprecation warnings
- `_REPO_ROOT` path resolution in surrogate training
- Caffeine CLint calibrated to literature values (40.0 -> 2.3 L/h)
- Drug.logP field name (capital P)

## v0.8.0
- Initial multi-phase PBPK platform
