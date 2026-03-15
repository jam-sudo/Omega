# QSP layer

Omega now supports an optional QSP plugin layer on top of PBPK + PD.

## Separation of concerns

- PBPK: mass transport and concentrations.
- PD: effect transforms (e.g., Emax) from concentration signals.
- QSP: mechanistic biomarker/cytokine state models driven by PBPK signals.

## Coupling modes

- `posthoc` (default): solve PBPK first, then solve QSP on the same output time grid using interpolated concentration signal.
- `coupled`: solve PBPK and QSP together as one ODE state vector with `solve_ivp(..., method="BDF")`.

The coupled mode is intentionally optional and currently implemented as a minimal state augmentation shortcut in `omega_pbpk.pbpk.solver.simulate`.

## Extension pattern

1. Create a model class implementing `BaseQSPModel` (`state_names`, `initial_state`, `rhs`).
2. Register with `@register_qsp_model("name")`.
3. Reference it via CLI `--qsp-model name` and YAML `model: name`.

Registry helpers:

- `get_qsp_model(name)`
- `list_qsp_models()`
