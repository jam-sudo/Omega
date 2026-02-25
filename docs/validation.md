# Validation and benchmark suite

This repository includes a reproducible benchmark suite under `benchmarks/` to check whether the current simulator behavior remains consistent against fixed reference PK curves.

## Philosophy

- Benchmarks validate **model behavior stability**, not clinical correctness.
- Acceptance gates are intentionally broad for an MVP model:
  - AUC relative error <= 0.30
  - Cmax relative error <= 0.30
  - Tmax absolute error <= 1.0 h
- RMSE is also reported for trend tracking.

Thresholds are defined in `benchmarks/expected/acceptance.json`.

## Dataset format

Each CSV in `benchmarks/datasets/` must contain:

- `time_h`
- `C_plasma_mg_per_L`

Optional columns are allowed (for example `std_mg_per_L`).

## Dataset provenance

The included datasets (`caffeine`, `warfarin`, and `metoprolol`) are **SYNTHETIC EXAMPLE DATASETS** generated from smooth simulator curves with deterministic perturbations. They are not sourced from clinical trial publications.

## Running benchmarks

```bash
python -m physio_sim.cli benchmark --suite benchmarks --out outputs/benchmarks --deterministic --seed 0
```

The command writes:

- `outputs/benchmarks/<drug>/overlay.png`
- `outputs/benchmarks/<drug>/metrics.json`
- `outputs/benchmarks/summary.json`
- `outputs/benchmarks/report.md`

The command exits with status code `1` if any benchmark case fails acceptance.

## Adding a new drug case

1. Add dataset CSV in `benchmarks/datasets/`.
2. Add a config YAML in `benchmarks/configs/` with:
   - subject path
   - compound parameters
   - dosing + run controls
   - dataset path
3. If needed, tune per-suite thresholds in `benchmarks/expected/acceptance.json`.
4. Re-run the benchmark command and review overlays + metrics.
