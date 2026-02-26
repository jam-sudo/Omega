# Publication Outline — Omega PBPK v0.7

## Abstract draft

We present Omega PBPK, a 34-state ODE whole-body pharmacokinetic simulation platform
implementing 15-organ perfusion- and permeability-limited tissue distribution, ACAT
8-segment absorption, dual-inlet hepatic disposition with IVIVE-scaled clearance,
drug-drug interaction modeling (competitive, MBI, induction), and QSP/PD coupling
including Emax, indirect response, and tumor growth models. The platform features
Monte Carlo virtual population generation, CPIC pharmacogenomics for 5 CYP genes,
and off-target safety assessment, providing a comprehensive translational toolkit for
preclinical-to-clinical PK prediction.

## Methods template

1. Model structure: 34-state ODE, 15 organs, closed-loop circulation
2. Tissue distribution: 11 perfusion-limited + 4 permeability-limited organs
3. Absorption: ACAT 8-segment model with Peff-based ka
4. Hepatic clearance: well-stirred model with IVIVE scaling
5. DDI mechanisms: competitive inhibition, MBI, induction (fm-weighted)
6. PD models: Emax, effect compartment (Crank-Nicolson), indirect response, Simeoni
7. Population variability: Monte Carlo + PGx (CPIC allele frequencies)
8. Numerical methods: LSODA solver, rtol=1e-8, atol=1e-10
9. Validation: mass balance, 50 unit tests, synthetic benchmark datasets

## Results structure

1. Single-dose PK profiles (IV and oral) with mass balance verification
2. Multi-dose steady-state simulation (Css,max, Css,min, accumulation ratio)
3. DDI AUC ratio predictions
4. Population PK variability (median, 5th-95th percentile bands)
5. PGx-stratified clearance distributions
6. Safety panel risk classification

## Discussion positioning

- Full mechanistic PBPK with explicit circulation (not simplified 1/2-compartment)
- Transparent equations — every ODE term traceable to physiological parameter
- Extensible architecture — new organs, DDI mechanisms, PD models
- Open-source, reproducible simulation results

## Limitations

- Default Kp from logP heuristic (not Rodgers & Rowland)
- Static DDI inhibitor concentrations (no perpetrator PK)
- GFR-only renal clearance (no active transport)
- GNN ADME model is scaffold only (untrained)

## Future work

- Clinical validation against published PK datasets (midazolam, caffeine, etc.)
- Mechanistic Kp prediction methods
- Time-varying DDI with perpetrator simulation
- Formal regulatory V&V package
