# Publication Outline

## Abstract draft
We present a reproducible PBPK/PD simulation framework implementing perfusion-limited tissue exchange, dual-inlet hepatic disposition, and explicit gut-wall first-pass handling, with deterministic and sensitivity workflows suitable for translational decision support.

## Methods template
1. Model structure and assumptions
2. Governing equations and units
3. Numerical solution settings
4. Calibration and uncertainty procedures
5. Sensitivity analysis method
6. Candidate evaluation scoring and risk flag definitions

## Results structure
1. Base PK and PD profiles
2. Internal validation and mass-balance checks
3. Sensitivity ranking (`dCmax/dp`, `dAUC/dp`)
4. Uncertainty and population variability summaries
5. Candidate risk-scoring outputs

## Discussion positioning
- Emphasize pragmatic translational use with transparent equations.
- Distinguish this platform from full mechanistic QSP by scope.
- Highlight deterministic reproducibility for publication workflows.

## Limitations
- Simplified tissue partition and elimination assumptions.
- Limited QSP pathway depth.
- Population variability is currently parameterized with fixed lognormal assumptions.

## Future work
- Extend mechanistic pathway depth and biomarker linkage.
- Add external validation cohorts and V&V benchmarks.
- Integrate formal model qualification reports.
