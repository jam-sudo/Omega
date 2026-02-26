# Model Equations and Units — Omega PBPK v0.7

## Units

- time: h
- amount: mg
- volume: L
- flow/clearance: L/h
- concentration: mg/L
- partition coefficients (Kp): unitless
- fractions (fup, Fg): unitless

For each compartment `x`, concentration is `C_x = A_x / V_x`.

## State Vector (34 states)

| Index | State | Description |
|-------|-------|-------------|
| 0 | venous_blood | Venous blood pool |
| 1 | arterial_blood | Arterial blood pool |
| 2 | lung | Lung (perfusion-limited) |
| 3 | brain | Brain (perfusion-limited) |
| 4 | heart | Heart (perfusion-limited) |
| 5 | kidney | Kidney (perfusion-limited) |
| 6 | liver | Liver (perfusion-limited) |
| 7 | spleen | Spleen → portal vein |
| 8 | gut_wall | Gut wall → portal vein |
| 9 | pancreas | Pancreas → portal vein |
| 10 | thymus | Thymus (perfusion-limited) |
| 11 | reproductive | Reproductive (perfusion-limited) |
| 12 | rest | Rest-of-body (perfusion-limited) |
| 13–16 | adipose/muscle/bone/skin_vasc | Vascular space (permeability-limited) |
| 17–20 | adipose/muscle/bone/skin_extra | Extravascular space (permeability-limited) |
| 21–28 | ACAT segments | stomach, duodenum, jejunum1/2, ileum1/2/3, colon |
| 29 | portal_vein | Portal vein mixing compartment |
| 30 | metabolized_hepatic | Cumulative hepatic metabolism sink |
| 31 | excreted_renal | Cumulative renal excretion sink |
| 32 | metabolized_gut | Cumulative gut wall metabolism sink |
| 33 | excreted_fecal | Cumulative fecal excretion sink |

Mass balance (IV): `dose = sum(all 34 states)` at all times.

## Dosing

- IV bolus: `A_venous(0) = dose_mg`
- Oral bolus: `A_stomach(0) = dose_mg`

## Plasma protein binding

- Blood concentration: `C_blood = A_venous / V_venous`
- Plasma concentration: `C_plasma = C_blood / Rbp`
- Unbound plasma: `C_u = fup × C_plasma`

## Circulation: Lung ↔ Arterial ↔ Venous

- `dA_lung/dt = CO × C_venous − CO × C_lung_out`
- `C_lung_out = A_lung × Rbp / (V_lung × Kp_lung)`
- `dA_arterial/dt = CO × C_lung_out − CO × C_arterial`
- `dA_venous/dt = Σ(organ venous outflows) − CO × C_venous`

CO = cardiac output (390 L/h at 70 kg, linear scaling with BW).

## Perfusion-limited tissue distribution

For tissue `t` in {brain, heart, kidney, thymus, reproductive, rest}:

- `C_out_t = A_t × Rbp / (V_t × Kp_t)`
- `dA_t/dt = Q_t × C_arterial − Q_t × C_out_t`
- Venous return: `+= Q_t × C_out_t`

Blood flow fractions (fraction of CO, excluding lung):

| Organ | %CO |
|-------|-----|
| brain | 0.12 |
| heart | 0.04 |
| kidney | 0.19 |
| liver (HA) | 0.065 |
| spleen | 0.03 |
| gut_wall | 0.15 |
| pancreas | 0.01 |
| thymus | 0.002 |
| reproductive | 0.002 |
| rest | 0.069 |
| adipose | 0.052 |
| muscle | 0.17 |
| bone | 0.05 |
| skin | 0.05 |
| **Total** | **1.000** |

Source: ICRP Publication 89 (2002).

## Permeability-limited tissue distribution

For tissue `t` in {adipose, muscle, bone, skin}, with vascular (V) and extravascular (E) subcompartments:

- `C_u_vasc = fup × C_vasc / Rbp`
- `C_u_extra = fup × C_extra / Kp_t`
- `dA_vasc/dt = Q_t × C_art − Q_t × C_vasc − PS × (C_u_vasc − C_u_extra)`
- `dA_extra/dt = PS × (C_u_vasc − C_u_extra)`

PS = permeability-surface area product (L/h), scales linearly with body weight.

## ACAT 8-segment absorption model

8 GI segments with segment-specific transit rates and absorption fractions:

| Segment | Transit time (h) | ka fraction |
|---------|------------------|-------------|
| stomach | 0.25 | 0.0 |
| duodenum | 0.26 | 1.0 |
| jejunum1 | 0.475 | 1.0 |
| jejunum2 | 0.475 | 1.0 |
| ileum1 | 0.68 | 0.8 |
| ileum2 | 0.68 | 0.6 |
| ileum3 | 0.68 | 0.3 |
| colon | 13.5 | 0.05 |

For each segment `i`:

- `kt_i = 1 / transit_time_i` (transit rate, h⁻¹)
- `ka_i = 2 × Peff × 3600 × 10⁻⁴ / r × ka_fraction_i × 0.01`
- `dA_i/dt += −ka_i × A_i − kt_i × A_i`
- `dA_{i+1}/dt += kt_i × A_i` (transit to next segment)
- `dA_gut_wall/dt += ka_i × A_i` (absorption to gut wall)
- Colon outflow: `dA_excreted_fecal/dt += kt_colon × A_colon`

## Portal transfer and gut-wall metabolism

Portal organs (spleen, gut_wall, pancreas) drain into portal vein:

- `portal_inflow = Σ Q_portal_organ × C_out_portal_organ`
- `dA_portal/dt = portal_inflow − Q_portal_total × C_portal`

Gut wall metabolism (enterocyte CYP3A4):

- `CLint_gut = CLint × gut_clint_multiplier × IVIVE_scaling`
- `gut_met_rate = CLint_gut × fup × A_gut / (V_gut × Kp_gut)`
- `dA_metabolized_gut/dt += gut_met_rate`

## Liver with dual blood supply

Liver receives from:
- Hepatic artery: `Q_ha × C_arterial`
- Portal vein: `Q_portal × C_portal`
- `Q_total = Q_ha + Q_portal`

Outflow:
- `C_liver_out = A_liver × Rbp / (V_liver × Kp_liver)`
- `dA_liver/dt = Q_ha × C_art + Q_portal × C_portal − Q_total × C_liver_out − met_rate`
- Venous return: `+= Q_total × C_liver_out`

## Hepatic clearance (well-stirred model)

IVIVE scaling from in vitro CLint:

```
CLint_scaled (L/h) = CLint (µL/min/pmol) × MPPGL (40) × microsomal_protein (45 mg/g liver)
                     × liver_weight (1800 g) / 10⁶ / 60
                   ≈ CLint × 0.054 L/h
```

Well-stirred model conversion (CLint → CLh):

```
CLh = (Q_liver × fup × CLint_effective) / (Q_liver + fup × CLint_effective)
```

CLh is a blood clearance parameter bounded by liver blood flow (flow-limited ceiling).

Hepatic metabolism rate:

- `Hepatic_elim = CLh × C_liver,venous`
- `dA_metabolized_hepatic/dt += Hepatic_elim`

DDI modifiers to `CLint_effective`:

1. **Competitive**: `CLint_eff *= 1 − fm × (1 − 1/(1 + [I]/Ki))`
2. **MBI**: `CLint_eff *= 1 − fm × kinact×[I] / (kdeg×(Ki+[I]) + kinact×[I])`
3. **Induction**: `CLint_eff *= 1 + fm × (fold_induction − 1)`

## Renal elimination

Renal clearance on total plasma concentration basis:

- `Renal_elim = CLr × C_plasma,total`
- `dA_kidney/dt = Q_kidney × C_art − Q_kidney × C_out − Renal_elim`
- `dA_excreted_renal/dt += Renal_elim`

CLr (L/h) is a drug-specific parameter set in the compound YAML. When CLr = 0 (default), no renal elimination occurs.

## PK parameter calculation

From plasma concentration-time profile:

- `Cmax = max(Cp)`
- `Tmax = time at Cmax`
- `AUC = ∫ Cp dt` (trapezoidal rule)
- `t½`: log-linear regression on terminal phase (last 50% of profile, Cp > 1% of Cmax)
- `CL = dose / AUC`
- `Vss = CL × t½ / 0.693`

## PD models

### Direct Emax
- `Effect = E0 + (Emax × Ce^hill) / (EC50^hill + Ce^hill)`

### Effect compartment (Crank-Nicolson implicit)
- `Ce[i] = (Ce[i-1] + 0.5 × dt × ke0 × (Cp[i-1] + Cp[i])) / (1 + 0.5 × dt × ke0)`

### Indirect response (4 types)
- Type I: inhibition of Kin
- Type II: inhibition of Kout
- Type III: stimulation of Kin
- Type IV: stimulation of Kout

### Tumor growth (Simeoni)
- `dT_grow/dt = (2×λ0×λ1×T) / (λ0×T + λ1) − k2×C×T`
- Transit damage chain: `dD_i/dt = k2×C×D_{i-1} − k1×D_i`

## Numerical methods

- Solver: `scipy.integrate.solve_ivp`, method=`LSODA`
- Default: `rtol=1e-8`, `atol=1e-10`, `max_step=0.1 h`
- Negative states: logged and clipped to zero after integration (never inside RHS)
