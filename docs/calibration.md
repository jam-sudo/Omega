# Calibration — Omega PBPK v0.7

Calibration against observed clinical PK data is planned for a future release.

## Current capabilities

The v0.7 engine supports forward simulation with literature-based parameters.
Compound YAML files define all PK parameters (CLint, Kp, Peff, etc.) from published sources.

## CLI usage (simulation)

```bash
omega simulate \
  --compound compounds/midazolam.yaml \
  --dose-mg 7.5 \
  --route oral \
  --t-end-h 24.0
```

## Planned: Bayesian calibration

Future versions will implement parameter estimation against observed plasma data:

- **Parameters**: CLint, Peff, Kp values
- **Method**: Metropolis-Hastings or NUTS sampling in log-space
- **Likelihood**: Gaussian residual model on plasma concentrations
- **Priors**: Weak lognormal priors centered on compound YAML values

Observed CSV format:

- `time_h`
- `Cp_mg_L`
