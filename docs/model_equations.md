# Model Equations and Units

Units:

- time: h
- amount: mg
- volume: L
- flow/clearance: L/h
- concentration: mg/L

For each compartment `x`, concentration is `C_x = A_x / V_x`.

## Dosing

- Oral bolus: `A_GI_lumen(0) = dose_mg`
- IV bolus: `A_Plasma(0) = dose_mg`

## Plasma protein binding

- Total plasma concentration: `C_plasma,total = A_Plasma / V_plasma`
- Unbound plasma concentration: `C_u,plasma = fu_plasma * C_plasma,total`

Unbound concentration is used for clearance terms where appropriate.

## Oral absorption

- `dA_GI_lumen/dt = -ka * A_GI_lumen`
- `dA_Gut_wall/dt += ka * A_GI_lumen`

## Perfusion-limited tissue distribution

For tissue `t` in `{Gut_wall, Kidney, Lung, Muscle, Fat, Brain, Rest}`:

- `Exchange_t = Q_t * (C_plasma,total - C_t / Kp_t)`
- `dA_t/dt += Exchange_t`
- `dA_Plasma/dt -= Exchange_t`

## Portal transfer and gut-wall metabolism escape (`f_gut`)

- `Portal_in_raw = Q_portal * (C_gut_wall / Kp_gut_wall)`
- `Portal_in = f_gut * Portal_in_raw`
- `Gut_metabolism_loss = (1 - f_gut) * Portal_in_raw`
- `Portal_out = Q_portal * C_portal`
- `dA_Gut_wall/dt -= Portal_in_raw`
- `dA_Portal_vein/dt += Portal_in - Portal_out`
- `dA_Gut_metabolism/dt += Gut_metabolism_loss`

Backward-compatibility note: legacy `first_pass_extraction` is mapped to
`f_gut = 1 - first_pass_extraction` during config load.

## Liver with dual blood supply

Liver receives inflow from:

- Portal vein: `Portal_out`
- Hepatic artery: `Q_ha * C_plasma,total`, where `Q_ha = max(Q_liver - Q_portal, 0)`

Liver outflow to plasma:

- `C_liver,venous = C_liver / Kp_liver`
- `Liver_out = Q_liver * C_liver,venous`

Mass balance terms:

- `dA_Liver/dt += Q_ha * C_plasma,total + Portal_out - Liver_out`
- `dA_Plasma/dt += Liver_out - Q_ha * C_plasma,total`

## Hepatic intrinsic clearance (well-stirred) and elimination location

Effective intrinsic clearance:

- `CLint_eff = CLint / (1 + Iu/Ki)` if DDI inhibition enabled and `Ki, Iu` provided
- otherwise `CLint_eff = CLint`

Well-stirred hepatic blood clearance parameter:

- `CLh = Q_liver * fu_plasma * CLint_eff / (Q_liver + fu_plasma * CLint_eff)`

Liver-only elimination term:

- `Hepatic_elim = CLh * (fu_plasma * C_liver,venous)`
- `dA_Liver/dt -= Hepatic_elim`

## Renal elimination to urine sink

- `Renal_elim = CLr * C_u,plasma`
- `dA_Plasma/dt -= Renal_elim`
- `dA_Urine/dt += Renal_elim`

## PD model

Direct Emax model:

- `Effect = E0 + (Emax * Ce^hill) / (EC50^hill + Ce^hill)`

Optional effect compartment:

- `dCe/dt = ke0 * (C_plasma,total - Ce)`

## Partition method interface

The model supports modular tissue partition method selection via `partition_method`.

- Current implemented method: `heuristic`
- Extension point reserved for future mechanistic options (e.g., Rodgers & Rowland, Poulin & Theil).
