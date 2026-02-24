# Model Equations and Units

Units:
- time: h
- amount: mg
- volume: L
- flow/clearance: L/h
- concentration: mg/L

Let `C_x = A_x / V_x` for each compartment.

## Dosing
- Oral: `A_GI(0) = dose_mg`
- IV bolus: `A_Plasma(0) = dose_mg`

## Absorption
- `dA_GI/dt = -ka * A_GI`
- `dA_GutWall/dt += ka * A_GI`

## Perfusion-limited distribution
For tissue `t`:
- `Exchange_t = Q_t * (C_plasma - C_t / Kp_t)`
- `dA_t/dt += Exchange_t`
- `dA_plasma/dt -= Exchange_t`

Applied to gut wall, kidney, lung, muscle, fat, brain, rest, and liver.

## Portal-liver transfer
- `Portal_in = Q_portal * (C_gut / Kp_gut)`
- `Portal_out = Q_portal * C_portal`
- `dA_portal/dt = Portal_in - Portal_out`
- `dA_liver/dt += Portal_out`

## Hepatic clearance (well-stirred)
- `CLh = Qh * fu * CLint_eff / (Qh + fu * CLint_eff)`
- `CLint_eff = CLint / (1 + Iu/Ki)` if DDI enabled, else `CLint`
- `Hepatic_elim = CLh * (C_liver / Kp_liver)`
- `dA_liver/dt -= Hepatic_elim`

## Renal clearance
- `Renal_elim = CLr * C_plasma`
- `dA_plasma/dt -= Renal_elim`
- `dA_urine/dt += Renal_elim`

## PD
Direct Emax:
- `Effect = E0 + (Emax * Ce^hill) / (EC50^hill + Ce^hill)`

Effect compartment (optional):
- `dCe/dt = ke0 * (C_plasma - Ce)`
