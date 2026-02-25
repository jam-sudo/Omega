# Local Sensitivity Analysis

Omega supports local finite-difference sensitivity analysis on core PK metrics.

## Supported metrics

For each selected parameter, Omega computes:

- `dCmax/dparam`
- `dAUC/dparam`

using centered finite differences.

## CLI usage

```bash
python -m physio_sim.cli simulate \
  --compound examples/compound_caffeine.yaml \
  --subject examples/subject_default.yaml \
  --dose-mg 100 --route oral --t-end-h 24 \
  --sensitivity \
  --sensitivity-params clint_L_per_h,clr_L_per_h,ka_per_h,fu_plasma \
  --sensitivity-eps 0.05 \
  --out outputs/run_sensitivity
```

## Outputs

When `--sensitivity` is enabled, run artifacts include:

- `sensitivity.csv` (raw local derivatives and influence values)
- `sensitivity_ranked.json` (ranked parameter influence)

The benchmark and publication examples typically use `--deterministic` for stable sensitivity outputs.
