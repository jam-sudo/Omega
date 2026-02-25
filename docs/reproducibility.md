# Reproducibility

Omega includes a deterministic execution mode for simulation and benchmark commands.

## Deterministic mode

Use `--deterministic` to force reproducible solver and stochastic behavior:

- fixed seed (`0` by default unless overridden with `--seed`)
- fixed ODE solver settings (`BDF`, `rtol=1e-8`, `atol=1e-10`)
- stable output time-grid construction via fixed `t_end_h` / `dt_out_h`
- no multiprocessing in deterministic workflows

## CLI examples

```bash
python -m physio_sim.cli simulate \
  --compound examples/compound_caffeine.yaml \
  --subject examples/subject_default.yaml \
  --dose-mg 100 --route oral --t-end-h 24 \
  --deterministic --seed 0 --out outputs/run_det
```

```bash
python -m physio_sim.cli benchmark \
  --suite benchmarks --out outputs/benchmarks --deterministic --seed 0
```

## Metadata capture

`summary.json` files include:

- `deterministic`
- `seed`
- `solver` (`method`, `rtol`, `atol`)
- `config_hash`
- `model_metadata` (`package_version`, `git_commit`, `timestamp_utc`)

This metadata supports exact reruns and publication-grade method traceability.
