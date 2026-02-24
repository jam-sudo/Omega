# Human Physiology Drug Response Simulator (MVP)

Research software prototype for **whole-body PBPK-like PK** + **minimal PD (Emax)** simulation.

## Safety scope

- For computational research and model prototyping only.
- Not validated for clinical decision making.
- No real-world dosing recommendations are provided.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

## Formatting

```bash
python -m ruff format .
python -m ruff check .
```

## Run example

```bash
python -m physio_sim.cli simulate \
  --compound examples/compound_caffeine.yaml \
  --subject examples/subject_default.yaml \
  --dose-mg 100 \
  --route oral \
  --t-end-h 24 \
  --out outputs/run1
```

Expected outputs:

- `outputs/run1/timecourse.csv`
- `outputs/run1/summary.json`
- `outputs/run1/plots.png`

## Model in scope

- 13 compartments: GI lumen, gut wall, portal vein, liver, plasma, kidney, lung, muscle, fat, brain, rest, urine sink, and gut metabolism sink.
- Perfusion-limited distribution with tissue Kp.
- Hepatic clearance via well-stirred formula.
- Renal clearance from plasma to urine sink.
- Gut-wall first-pass escape via `f_gut` (fraction escaping gut-wall metabolism).
- PD via direct Emax with optional effect-compartment delay (`ke0`).

## Deprecations

- `first_pass_extraction` is deprecated in compound YAML and internally mapped to `f_gut = 1 - first_pass_extraction` during config load.
- If both are provided, `f_gut` is used and a deprecation warning is emitted.

## Model out of scope (current MVP)

- Population variability calibration (now available as lightweight Bayesian calibration for CLint and ka; see docs)
- Mechanistic transporter/enzyme networks
- Full validated organ-specific physiology

See docs:

- `docs/model_equations.md`
- `docs/assumptions_and_limits.md`
- `docs/roadmap.md`
- `docs/uncertainty.md`
- `docs/calibration.md`
