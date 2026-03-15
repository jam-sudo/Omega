# Assumptions and Limits — Omega PBPK v0.7

## Model structure

- 35-state ODE with 15 organs in closed-loop arterial-venous circulation.
- 11 perfusion-limited organs (instantaneous tissue-blood equilibrium within each organ).
- 4 permeability-limited organs (adipose, muscle, bone, skin) with PS barrier.
- ACAT 8-segment GI absorption with segment-specific transit rates and ka fractions.
- Portal vein explicitly modeled — spleen, gut wall, pancreas drain into liver via portal vein.

## Key assumptions

- Blood flow fractions from ICRP Publication 89 (2002) reference man (70 kg adult male).
- Linear body weight scaling for organ volumes and blood flows.
- Cardiac output scales linearly with BW in ODE (population generator uses BW^0.75).
- Well-stirred hepatic clearance model — assumes rapid equilibrium within liver.
- IVIVE: 40 pmol CYP/mg protein (MPPGL), 45 mg protein/g liver, 1800 g liver weight.
- Renal clearance: GFR-based filtration with heuristic active secretion (OCT2/OAT for hydrophilic compounds with TPSA > 74 A^2). No tubular reabsorption modeling.
- Drug-drug interactions use static inhibitor concentrations (no time-varying [I]).

## Heuristic components

- Default Kp values from a logP-based heuristic when not provided in compound YAML.
- ADME predictor uses simplified QSPR models; RDKit path for better accuracy when available.
- Safety panel IC50 predictions are QSPR-based, not docking-based.

## Pharmacogenomics

- CPIC allele frequency database for 5 CYP genes.
- Activity score to phenotype mapping per CPIC guidelines.
- CLint scaling factors are empirical (UM=1.5, NM=1.0, IM=0.5, PM=0.1).

## Numerical considerations

- ODE solved by `solve_ivp(method='LSODA')` with `rtol=1e-8, atol=1e-10, max_step=0.1h`.
- Negative numerical states are clipped to zero **after** integration (never inside RHS).
- Mass balance verified for IV bolus: dose = sum(all 35 states) ± 0.5%.

## Not yet covered

- Time-varying inhibitor PK for DDI
- Tubular reabsorption modeling
- Multi-zonal liver metabolism and transporter effects (P-gp, OATP1B1)
- Regulatory V&V package
- Nonlinear (saturable) metabolism (Michaelis-Menten kinetics)

## Implemented but not in production pipeline

- GNN molecular encoder (training unsuccessful — distillation ceiling)
- Cross-attention foundation model (Level 3 neural architecture, awaiting clinical data)
- Reptile meta-learning for few-shot adaptation (code ready, no training data)
