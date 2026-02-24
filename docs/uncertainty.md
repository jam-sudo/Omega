# Uncertainty Propagation

`physio_sim.uncertainty` adds Monte Carlo uncertainty propagation for PK summary metrics.

## Supported parameter specifications

Each uncertain parameter can be provided as:

- **Fixed value**: `1.2`
- **Normal**: `{"dist": "normal", "mean": 1.2, "sd": 0.2}`
- **Lognormal**: `{"dist": "lognormal", "mean": 0.0, "sd": 0.3}`

Notes:
- For `lognormal`, `mean` and `sd` are parameters of the underlying normal in log-space.
- `sd` must be positive.

## Monte Carlo API

```python
from physio_sim.uncertainty import monte_carlo_propagation

result = monte_carlo_propagation(
    subject=subject_cfg,
    compound=compound_cfg,
    dose_mg=100,
    route="oral",
    t_end_h=24,
    dt_out_h=0.1,
    n_samples=500,
    parameter_specs={
        "ka_per_h": {"dist": "lognormal", "mean": 0.2, "sd": 0.2},
        "clint_L_per_h": {"dist": "normal", "mean": 8.0, "sd": 1.0},
    },
    seed=42,
)
```

Output `result.sample_metrics` includes one row per simulation and aggregates:

- `Cmax_mg_per_L`
- `AUC0_tend_mg_h_per_L`
- sampled parameter values
