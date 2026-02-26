# Roadmap — Omega PBPK v0.7+

## Completed in v0.7

- 34-state ODE engine with 15 organs
- ACAT 8-segment absorption model
- Permeability-limited distribution (adipose, muscle, bone, skin)
- Portal vein with dual-inlet liver
- Well-stirred hepatic clearance with IVIVE scaling
- GFR-based renal clearance
- DDI: competitive inhibition, MBI, induction
- PD models: Emax, effect compartment, indirect response, Simeoni tumor growth
- CPIC pharmacogenomics for 5 CYP genes
- Monte Carlo virtual population generator
- Dose optimization and multi-dose simulation
- Off-target safety panel
- GNN MPNN scaffold for ADME prediction
- FastAPI REST endpoint scaffold
- CLI with 7 commands

## Future work

1. **Oral calibration**
   - Midazolam oral validation (AAFE < 2.0 target)
   - Multi-compound validation (caffeine, warfarin, metoprolol)
2. **DDI expansion**
   - Time-varying inhibitor concentrations (simulate perpetrator PK)
   - Transporter-mediated DDI (OATP1B1, P-gp)
3. **Renal model refinement**
   - Active tubular secretion and reabsorption
4. **Liver model refinement**
   - Multi-zonal metabolism and transporter effects
5. **GNN training**
   - Train ADME MPNN on ChEMBL/internal datasets
   - Validate prediction accuracy vs. ADME predictor QSPR
6. **Population simulation**
   - Age/sex/ethnicity covariates
   - Pediatric/geriatric physiological scaling
7. **QSP integration**
   - Pathway-level downstream biomarkers
   - Mechanistic disease models
8. **Calibration and validation**
   - Bayesian parameter estimation against clinical PK data
   - Formal V&V report for regulatory submission
9. **Frontend and deployment**
   - Web UI (Streamlit/React)
   - Docker containerization
   - PyPI publication
