# Assumptions and Limits

## Explicit assumptions

- PBPK-like structure with perfusion-limited exchange and lumped compartments.
- Default Kp may come from a simple logP/pKa heuristic if not provided.
- DDI in MVP is simplified to a constant inhibitor proxy (`Iu`) and competitive inhibition formula.
- PD is a single Emax relationship, optionally delayed by one effect compartment.
- Gut-wall first-pass behavior is represented with `f_gut`, the fraction of gut-wall outflow that escapes gut-wall metabolism.

## Heuristic components

- Kp heuristic is intentionally non-validated and only for prototyping.
- Example compound values are placeholders for software checks.

## Numerical considerations

- ODE solved by `solve_ivp(method="BDF")` with fixed output grid.
- Negative numerical states are clipped to zero when exporting and in RHS concentrations.

## Not covered yet

- Parameter fitting/calibration against clinical datasets
- Enzyme-transporter mechanistic detail
- Population Monte Carlo variability
- Regulatory validation package
