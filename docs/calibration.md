# Bayesian Calibration

`physio_sim.calibration` implements a lightweight Metropolis-Hastings sampler to calibrate:

- `CLint` (`clint_L_per_h`)
- `ka` (`ka_per_h`)

from observed plasma concentration data.

## CLI usage

```bash
python -m physio_sim.cli calibrate \
  --data observed.csv \
  --compound examples/compound_template.yaml \
  --subject examples/subject_default.yaml
```

Observed CSV must contain columns:

- `time_h`
- `C_plasma_mg_per_L`

## Outputs

The command writes to `--out` (default: `outputs/calibration`):

- `posterior_samples.csv`
- `trace_plots.png`
- `posterior_predictive_overlay.png`
- `calibration_summary.json`

## Method summary

- Proposal: random walk in log-space for positive parameters.
- Likelihood: Gaussian residual model on plasma concentrations.
- Priors: weak lognormal priors centered on input YAML values.
