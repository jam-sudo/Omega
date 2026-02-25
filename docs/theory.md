# Formal Theory and Equation Specification

## State vector and units
State vector `y` is in mg for all amount states:

`[A_GI_lumen, A_Gut_wall, A_Portal_vein, A_Liver, A_Plasma, A_Kidney, A_Lung, A_Muscle, A_Fat, A_Brain, A_Rest, A_Urine, A_Gut_metabolism]`.

Units:
- time: h
- amount: mg
- concentration: mg/L (`C_x = A_x / V_x`)
- flow and clearance: L/h
- partition coefficients `Kp`: unitless
- fractions (`fu_plasma`, `f_gut`): unitless

## Full ODE system definition

Oral dose input:
- `dA_GI_lumen/dt = -ka * A_GI_lumen` (mg/h)
- `dA_Gut_wall/dt += ka * A_GI_lumen` (mg/h)

Exchange tissues `t in {Kidney, Lung, Muscle, Fat, Brain, Rest}` (perfusion-limited):
- `dA_t/dt += Q_t * (C_plasma - C_t/Kp_t)`
- `dA_Plasma/dt -= Q_t * (C_plasma - C_t/Kp_t)`

Gut wall perfusion-limited exchange:
- `dA_Gut_wall/dt += Q_gut * (C_plasma - C_gut/Kp_gut)`
- `dA_Plasma/dt -= Q_gut * (C_plasma - C_gut/Kp_gut)`

Portal transfer + gut-wall metabolism:
- `Portal_in_raw = Q_portal * (C_gut/Kp_gut)`
- `Portal_in = f_gut * Portal_in_raw`
- `Gut_metabolism_loss = (1-f_gut) * Portal_in_raw`
- `Portal_out = Q_portal * C_portal`
- `dA_Gut_wall/dt -= Portal_in_raw`
- `dA_Gut_metabolism/dt += Gut_metabolism_loss`
- `dA_Portal_vein/dt += Portal_in - Portal_out`

Liver dual inflow and outflow:
- `Q_ha = max(Q_liver - Q_portal, 0)`
- `C_liver_venous = C_liver/Kp_liver`
- `Liver_out = Q_liver * C_liver_venous`
- `dA_Liver/dt += Q_ha*C_plasma + Portal_out - Liver_out`
- `dA_Plasma/dt += Liver_out - Q_ha*C_plasma`

Hepatic elimination (well-stirred-derived `CLh`):
- `CLint_eff = CLint/(1 + Iu/Ki)` if DDI inhibition is enabled and inputs exist, else `CLint`
- `CLh = Q_liver * fu_plasma * CLint_eff / (Q_liver + fu_plasma * CLint_eff)`
- `Hepatic_elim = CLh * (fu_plasma * C_liver_venous)`
- `dA_Liver/dt -= Hepatic_elim`

Renal elimination:
- `C_u_plasma = fu_plasma * C_plasma`
- `Renal_elim = CLr * C_u_plasma`
- `dA_Plasma/dt -= Renal_elim`
- `dA_Urine/dt += Renal_elim`

## Well-stirred equation (explicit)
`CLh = Qh * fu * CLint_eff / (Qh + fu*CLint_eff)`.

## Perfusion-limited equation (explicit)
`Rate_t = Q_t * (C_plasma - C_t/Kp_t)`.

## PD and QSP equations
PD uses Emax with optional effect-site link:
- `dCe/dt = ke0 * (C_plasma - Ce)` when `ke0` is provided; otherwise `Ce = C_plasma`
- `Effect = E0 + (Emax * Ce^h)/(EC50^h + Ce^h)`

QSP in this implementation is represented by this minimal PK-PD coupling (effect-site transit + Emax response) without additional mechanistic pathway states.

## Deterministic numerical mode
When deterministic mode is enabled:
- ODE solver uses fixed tolerances `rtol=1e-8`, `atol=1e-10`
- random seed defaults to `0` if not user-provided
- uncertainty worker count is fixed to `1`
- solver tolerances are written to `summary.json`
