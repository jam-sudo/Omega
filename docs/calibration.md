# Calibration — Omega PBPK

## Individual parameter estimation (implemented)

Omega supports Bayesian individual PK parameter estimation from sparse concentration-time observations via `OmegaPipeline.fit_individual()`. Given 1-5 observed plasma concentrations, the system estimates individual clearance and volume scaling factors.

```python
from omega_pbpk.pipeline import OmegaPipeline, SimulationRequest

pipeline = OmegaPipeline()
fit = pipeline.fit_individual(
    SimulationRequest(smiles="CC(=O)CC(c1ccccc1)c1c(O)c2ccccc2oc1=O", dose_mg=5.0),
    observations=[(1.0, 0.15), (4.0, 0.13), (12.0, 0.05)],
)
print(f"CL scale: {fit['cl_scale']:.2f}, Vd scale: {fit['vd_scale']:.2f}")
```

**Method:** L-BFGS-B optimization in log-concentration space, using an analytical 1-compartment oral model. Scaling factors bounded to [0.05, 20.0].

**Status:** Implemented and unit-tested. Not yet validated against real patient data.

## Confidence calibration (partial)

Conformal prediction intervals are calibrated on a scaffold-split holdout (30 compounds). Coverage varies by property:

| Property | Coverage (target: 90%) | Status |
|----------|----------------------|--------|
| fup | 100% | Over-covered |
| rbp | 100% | Over-covered |
| peff | 97% | OK |
| clint | 37% | Under-covered |

CLint interval calibration is a known gap (see docs/paper for details).

## Forward simulation

```bash
omega simulate \
  --compound compounds/midazolam.yaml \
  --dose-mg 7.5 \
  --route oral \
  --t-end-h 24.0
```
