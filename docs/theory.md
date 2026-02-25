# Formal Theory and Equation Specification — Omega PBPK v0.7

## State vector and units

State vector `y` has 34 states in mg:

```
[venous_blood, arterial_blood, lung,
 brain, heart, kidney, liver, spleen, gut_wall, pancreas, thymus, reproductive, rest,
 adipose_vasc, muscle_vasc, bone_vasc, skin_vasc,
 adipose_extra, muscle_extra, bone_extra, skin_extra,
 stomach, duodenum, jejunum1, jejunum2, ileum1, ileum2, ileum3, colon,
 portal_vein, metabolized_hepatic, excreted_renal, metabolized_gut, excreted_fecal]
```

Units:
- time: h
- amount: mg
- concentration: mg/L (`C_x = A_x / V_x`)
- flow and clearance: L/h
- partition coefficients `Kp`: unitless
- fractions (`fup`, `Fg`): unitless

## Circulation model

Closed-loop circulatory system:

1. **Venous blood** collects outflow from all non-lung organs
2. **Lung** receives venous blood (CO × C_ven), outputs to arterial
3. **Arterial blood** distributes to all non-lung organs

Conservation: CO in = CO out at every node.

## Full ODE system definition

### Dosing
- IV bolus: `A_venous(0) = dose_mg`
- Oral: `A_stomach(0) = dose_mg`

### Lung and circulation
- `dA_lung/dt = CO × C_ven − CO × (A_lung × Rbp / (V_lung × Kp_lung))`
- `dA_art/dt = CO × C_lung_out − CO × C_art`
- `dA_ven/dt = Σ venous_inflows − CO × C_ven`

### Perfusion-limited organs
For tissue `t` in {brain, heart, thymus, reproductive, rest}:
- `dA_t/dt = Q_t × (C_art − A_t × Rbp / (V_t × Kp_t))`

### Portal organs (spleen, gut_wall, pancreas)
These drain to portal vein instead of venous blood:
- `dA_t/dt = Q_t × C_art − Q_t × C_out_t − [gut_metabolism if gut_wall]`
- `portal_inflow += Q_t × C_out_t`

### Gut wall with metabolism
- `gut_met_rate = CLint_gut × fup × A_gut / (V_gut × Kp_gut)`
- `dA_gut/dt = Q_gw × C_art − Q_gw × C_out − gut_met_rate + Σ ACAT_absorption`

### Portal vein
- `dA_portal/dt = portal_inflow − Q_portal_total × C_portal`

### Liver with dual inflow
- `dA_liver/dt = Q_ha × C_art + Q_portal × C_portal − Q_total × C_liver_out − met_rate`
- `met_rate = CLint_eff × fup × A_liver / (V_liver × Kp_liver)`

### Kidney with renal clearance
- `dA_kidney/dt = Q_kidney × C_art − Q_kidney × C_out − GFR × fup × C_kidney_unbound`
- `GFR = 7.5 × (BW/70) L/h`

### Permeability-limited organs (adipose, muscle, bone, skin)
- `dA_vasc/dt = Q_t × C_art − Q_t × C_vasc − PS × (C_u_vasc − C_u_extra)`
- `dA_extra/dt = PS × (C_u_vasc − C_u_extra)`

### ACAT absorption (8 segments)
For segment `i`:
- `dA_i/dt += −ka_i × A_i − kt_i × A_i`
- Transit: drug moves from segment i to i+1 (colon → fecal)
- Absorption: drug moves from segment i to gut wall

## Well-stirred equation
`CLh = Qh × fu × CLint_eff / (Qh + fu × CLint_eff)`

## IVIVE scaling
`CLint_scaled = CLint × MPPGL(40) × mg_protein/g_liver(45) × liver_weight(1800g) / 10⁶ / 60`

## Qgut model for Fg (Yang 2007)
`Fg = Qgut / (Qgut + fup × CLint_gut)`

## DDI mechanisms
1. **Competitive**: `1/(1 + [I]/Ki)` fold change on fm-weighted CLint
2. **MBI**: `kinact × [I] / (kdeg × (Ki + [I]) + kinact × [I])` fraction inactivated
3. **Induction**: `1 + fm × (fold_induction − 1)` multiplier

## PD models
- **Direct Emax**: `E = E0 + Emax × Ce^h / (EC50^h + Ce^h)`
- **Effect compartment** (Crank-Nicolson implicit): unconditionally stable scheme
- **Indirect response**: 4 types (inhibit/stimulate Kin/Kout)
- **Simeoni tumor growth**: transit damage compartments

## Numerical methods
- Solver: LSODA (scipy), stiff-aware with automatic Jacobian estimation
- Default: `rtol=1e-8`, `atol=1e-10`, `max_step=0.1 h`
- No negative state clipping inside RHS (only post-hoc with warning)
