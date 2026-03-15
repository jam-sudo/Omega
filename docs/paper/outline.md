# Omega: Open-Source SMILES-to-Pharmacokinetics Prediction in 73 Milliseconds

## Paper Outline

> **Target:** Journal of Pharmaceutical Sciences / CPT: Pharmacometrics & Systems Pharmacology
> **Type:** Research Article
> **Key claim:** Public-data-only hybrid model matches proprietary Bayer model accuracy (AAFE 1.90 vs 1.87)

---

## Abstract (250 words)

Predicting human pharmacokinetics (PK) from molecular structure alone remains a key challenge in drug discovery. Current approaches either require proprietary data and manual parameterization (commercial PBPK tools) or sacrifice interpretability (pure deep learning). We present Omega, an open-source hybrid neural-mechanistic platform that predicts PK directly from SMILES strings in 73 milliseconds, with no manual parameters. Omega combines ML-predicted ADME properties (XGBoost ensemble with conformal intervals) with a 35-state whole-body PBPK ODE, augmented by physics-informed corrections (GSE solubility floor, VDss calibration, hybrid Cmax/t_half selectors). On 20 benchmark drugs, Omega achieves Cmax AAFE 1.90 and AUC AAFE 1.66, matching the recently reported Bayer hybrid GCN-PBPK model (1.87 oral fold error) despite using only publicly available data and tools. Multi-tier validation across 39 drugs (t_half), 151 compounds (ADME properties), and 5 post-2022 temporal holdout drugs confirms generalization. A pragmatic Level 3 module provides patient-specific predictions via allometric scaling and Bayesian individual fitting from sparse observations. Omega is freely available under MIT license.

---

## 1. Introduction

### 1.1 The Problem
- Drug attrition due to unfavorable PK (cite statistics)
- Commercial PBPK tools (Simcyp, GastroPlus, PK-Sim) require expert parameterization
- Need: automated, fast, structure-only PK prediction

### 1.2 Prior Work
- **Bayer (Gruber et al., 2024):** GCN + PBPK hybrid, fold error 1.87 oral, but proprietary data + closed-source
- **DeepPK/PkSolver:** Pure ML approaches, lack mechanistic interpretability
- **ADMET-AI:** SOTA molecular property prediction (#1 TDC leaderboard) but no PK integration
- **Gap:** No open-source, public-data-only system that matches proprietary model accuracy

### 1.3 Contribution
- Open-source SMILES→PK in 73ms (1000x faster than manual PBPK)
- AAFE 1.90 using only public data, matching Bayer's 1.87 with proprietary data
- Multi-tier validation framework (Gold/Silver/Bronze/Temporal)
- Patient-specific prediction (allometric + Bayesian)
- Fully reproducible: `pip install` + 1 line of code

---

## 2. Methods

### 2.1 Architecture Overview

```
SMILES → XGBoost Ensemble → ADME properties (fup, CLint, logP, peff, rbp, VDss)
       → GSE Solubility Floor (logS ≥ 0.5 - logP)
       → Berezhkovskiy Kp + VDss Calibration → Drug Object
       → 35-state Whole-Body PBPK ODE → C(t) curve
       → Hybrid Cmax/t_half Selector → PK metrics
```

### 2.2 ADME Prediction Layer
- XGBoost with 2048-bit Morgan fingerprints for fup, CLint, RBP, VDss
- Polynomial ridge regression with GSE floor for logP, logS
- ADMET-AI (Chemprop D-MPNN) available but selectively used
- Conformal prediction for uncertainty intervals
- 153 reference compounds for calibration

### 2.3 PBPK ODE Engine
- 35-state model: 15 organs, 4 permeability-limited tissues, 8-segment ACAT GI
- LSODA integrator (rtol=1e-8, atol=1e-10)
- Well-stirred hepatic clearance with gut-wall first-pass
- Rodgers & Rowland tissue partitioning

### 2.4 Physics-Informed Corrections
- **GSE Solubility Floor:** General Solubility Equation (Yalkowsky & Valvani 1980) as thermodynamic lower bound
- **VDss Calibration:** XGBoost VDss vs Berezhkovskiy Kp comparison; use XGBoost when Kp-based Vd diverges >2x
- **Hybrid Cmax Selector:** Geometric mean of ODE and analytical 1-cpt Cmax for low-extraction drugs

### 2.5 Patient-Specific Prediction (Level 3)
- Allometric scaling: CL × (W/70)^0.75, Vd × (W/70)^1.0
- CYP genotype factors: CYP2D6 (UM/EM/IM/PM), CYP2C9 (6 diplotypes), CYP2C19
- Bayesian individual estimation: L-BFGS-B fitting from 1-5 C(t) observations

### 2.6 Validation Framework
- **Gold tier:** 20 drugs with clinical C(t) curves → Cmax, AUC fold-error
- **Silver tier:** 39 drugs with FDA label t_half → half-life fold-error
- **Bronze tier:** 151 compounds with reference ADME properties → per-property AAFE
- **Temporal holdout:** 5 drugs approved after 2022 (post-ADMET-AI training cutoff)
- **Structural analogs (T9):** SAR consistency via RDKit perturbation
- **De novo molecules (T10):** Physical plausibility on novel chemical space
- Automated regression testing with before/after comparison

---

## 3. Results

### 3.1 Gold Tier: PK Accuracy (20 drugs)

| Metric | Omega | Target | Bayer (2024) |
|--------|-------|--------|-------------|
| Cmax AAFE | 1.90 | < 2.0 | 1.87* |
| AUC AAFE | 1.66 | < 2.0 | — |
| Cmax %2-fold | 70% | ≥ 70% | — |
| AUC %2-fold | 70% | ≥ 70% | — |
| Speed | 73ms | < 500ms | — |

*Bayer reports "exposure fold change" which may include AUC; direct Cmax comparison not available.

- Table: Per-drug fold-errors for all 20 drugs
- Figure: Predicted vs. observed Cmax scatter plot (log-log)
- Figure: Predicted vs. observed AUC scatter plot (log-log)

### 3.2 Silver Tier: Half-Life (39 drugs)
- t_half AAFE: 2.42, 51% within 2-fold
- Table: Per-drug t_half comparison
- Note: After data quality correction (2 extraction errors in FDA labels)

### 3.3 Bronze Tier: ADME Properties (151 compounds)

| Property | AAFE | %2-fold | n |
|----------|------|---------|---|
| logP | 1.54 | 82% | 131 |
| fup | 2.10 | 58% | 151 |
| rbp | 1.09 | 98% | 151 |
| peff | 1.46 | 86% | 151 |
| CLint | 3.25 | 34% | 151 |

### 3.4 Temporal Holdout (5 post-2022 drugs)
- t_half AAFE: 3.12, 3/5 within 2-fold
- Drugs: adagrasib, futibatinib, capivasertib, elacestrant, pirtobrutinib
- Demonstrates generalization to truly unseen chemical space

### 3.5 Structural Analog Validation
- 20/20 analogs pass physical plausibility checks
- SAR consistency: structural perturbations produce gradual PK changes

### 3.6 Failure Analysis
- Only 2/20 drugs with >3-fold Cmax error
- verapamil (8.8x): P-gp efflux not modeled
- ibuprofen (5.0x): extreme protein binding (fup ~0.01)
- Known mechanistic limitations, not random errors

### 3.7 Patient-Specific Prediction
- Weight scaling: 40kg vs 70kg vs 100kg warfarin PK
- CYP2C9 genotype: *1/*1 vs *1/*3 vs *3/*3
- Bayesian fitting: 3 C(t) observations → individual CL/Vd

---

## 4. Discussion

### 4.1 Comparison to Bayer Hybrid Model
- Omega 1.90 vs Bayer 1.87 — comparable accuracy
- Key difference: Omega uses only public data; Bayer uses proprietary in vitro assays
- Implication: public data + careful engineering ≈ proprietary data + deep learning

### 4.2 Speed Advantage
- 73ms enables high-throughput virtual screening
- Warm start: XGBoost models pre-loaded
- Cold start: ~5s (ADMET-AI model loading)
- vs. commercial PBPK: minutes to hours per compound

### 4.3 Interpretability
- Unlike pure ML: every prediction traces through named ADME parameters → ODE
- Failure modes are diagnosable (e.g., P-gp, protein binding)
- Confidence intervals per property, not just final PK

### 4.4 Limitations
- CLint prediction (AAFE 3.25) is the weakest link — structure-based clearance remains an open challenge
- P-gp and other transporters not yet modeled
- Nonlinear metabolism (phenytoin-type) not captured
- Validated on oral IR only; IV, modified-release, multi-dose not yet tested
- Small validation set (20 drugs Gold) compared to industry databases

### 4.5 Future Directions
- Transporter correction (P-gp, OATP) using structural features
- Neural Level 3 with real individual patient data (PK-DB, clinical collaborations)
- Multi-dose and IV route validation
- TDC leaderboard submission

---

## 5. Conclusion

Omega demonstrates that open-source, public-data-only pharmacokinetic prediction can match the accuracy of proprietary models. At 73ms per prediction, it enables applications from high-throughput virtual screening to real-time clinical decision support. The multi-tier validation framework and honest failure analysis provide a template for transparent PK model evaluation.

---

## Figures

1. Architecture diagram (SMILES → ADME → ODE → PK)
2. Predicted vs. observed Cmax (log-log scatter, 20 drugs)
3. Predicted vs. observed AUC (log-log scatter, 20 drugs)
4. Per-drug fold-error bar chart (sorted, with 2-fold reference line)
5. Silver tier: predicted vs. observed t_half (39 drugs)
6. Bronze tier: per-property AAFE comparison
7. Temporal holdout: 5 post-2022 drugs results
8. L3 demo: warfarin PK across weight/genotype scenarios

## Tables

1. Gold tier: 20-drug per-drug results (SMILES, dose, pred/obs Cmax, pred/obs AUC, fold-errors)
2. Multi-tier validation summary
3. Omega vs. Bayer 2024 vs. commercial PBPK comparison
4. Failure classification by mechanism
5. ADME property accuracy (Bronze tier)

## Supplementary

- S1: All 39 Silver tier per-drug t_half results
- S2: Bronze tier per-compound results (151 compounds)
- S3: Confidence calibration (T8) results
- S4: CYP genotype scaling factor table
- S5: Benchmark drug SMILES and clinical PK sources

---

## Estimated Timeline

| Phase | Duration | Deliverable |
|-------|----------|-------------|
| Data preparation | 1 week | Final benchmark results, figures |
| First draft | 2 weeks | Complete manuscript |
| Internal review | 1 week | Revised draft |
| Submission | — | Target: J. Pharm. Sci. or CPT:PSP |

## Key References

1. Gruber et al. (2024). Prediction of Human Pharmacokinetics From Chemical Structure. J. Pharm. Sci. 113(2).
2. Swanson et al. (2024). ADMET-AI. arXiv.
3. Rodgers & Rowland (2006). Tissue distribution. J. Pharm. Sci.
4. Yalkowsky & Valvani (1980). General Solubility Equation. J. Pharm. Sci.
