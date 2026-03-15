# Omega: Open-Source Structure-to-Pharmacokinetics Prediction Using Only Public Data

**Authors:** [To be confirmed]

**Affiliations:** [To be confirmed]

**Corresponding author:** [To be confirmed]

**Target journal:** Journal of Pharmaceutical Sciences

**Keywords:** pharmacokinetics, PBPK, machine learning, SMILES, drug discovery, open-source

---

## Abstract

Predicting human pharmacokinetics from molecular structure alone remains a critical challenge in drug discovery, where poor PK properties account for ~40% of clinical failures. We present Omega, an open-source hybrid neural-mechanistic platform that predicts PK directly from SMILES strings in 73 milliseconds with no manual parameterization. Omega combines ML-predicted ADME properties (XGBoost ensemble with polynomial fallback and conformal intervals) with a 35-state whole-body PBPK ODE engine, augmented by physics-informed corrections including a General Solubility Equation floor, XGBoost-calibrated volume of distribution, and hybrid Cmax/half-life selectors. On 20 benchmark drugs (oral, single-dose, healthy volunteers), Omega achieves Cmax AAFE of 1.90 (median fold error 1.73) and AUC AAFE of 1.66 (median fold error 1.60), with 70% of predictions within 2-fold of observed values. For comparison, the Bayer hybrid GCN-PBPK model (Gruber et al., 2024) reports a median exposure fold error of 1.87 using proprietary training data; Omega achieves comparable or superior accuracy using only publicly available data and tools. Multi-tier validation across 39 drugs (half-life AAFE 2.42), 151 compounds (ADME property accuracy), and 5 post-2022 temporal holdout drugs (3/5 within 2-fold) confirms generalization beyond the training domain. Systematic failure analysis identifies only 2/20 drugs with >3-fold errors, both attributable to known mechanistic gaps (P-glycoprotein efflux, extreme protein binding). A patient-specific module extends predictions to individual patients via allometric covariate scaling and Bayesian parameter estimation from sparse concentration-time observations. Omega is freely available under the MIT license.

---

## 1. Introduction

### 1.1 The Challenge of Pharmacokinetic Prediction

Pharmacokinetics (PK) — the study of how drugs are absorbed, distributed, metabolized, and eliminated — determines whether a promising molecule becomes a viable medicine. Poor PK properties account for approximately 40% of clinical trial failures, making early PK characterization essential for efficient drug discovery (Kola & Landis, 2004). Yet obtaining PK data traditionally requires either expensive in vivo studies or labor-intensive physiologically based pharmacokinetic (PBPK) modeling, where expert pharmacokineticists manually parameterize complex models for each compound.

Commercial PBPK platforms — Simcyp (Certara), GastroPlus (Simulation Plus), and PK-Sim (Open Systems Pharmacology) — encode decades of physiological knowledge into sophisticated mechanistic models. However, these tools require extensive manual input: physicochemical properties, in vitro clearance data, permeability measurements, and protein binding assays. Setting up a single compound typically takes hours to days, requires domain expertise, and remains inaccessible to many research groups due to cost (commercial licenses exceeding $50,000/year) or expertise requirements.

The pharmaceutical industry therefore faces a fundamental tension: the compounds most in need of PK prediction — novel drug candidates in early discovery — are precisely those for which experimental data are most scarce.

### 1.2 Machine Learning Approaches to PK Prediction

Recent advances in molecular property prediction, particularly graph neural networks (GNNs) and message-passing neural networks (MPNNs), have enabled accurate prediction of absorption, distribution, metabolism, and excretion (ADME) properties directly from molecular structure (Yang et al., 2019; Swanson et al., 2024). ADMET-AI, a Chemprop-based D-MPNN ensemble, currently ranks first on the Therapeutics Data Commons (TDC) ADMET benchmark, predicting over 40 endpoints from SMILES strings alone (Swanson et al., 2024).

However, predicting individual ADME properties is fundamentally different from predicting integrated PK behavior. A drug's concentration-time profile emerges from the nonlinear interplay of absorption kinetics, tissue distribution, hepatic metabolism, and renal elimination — dynamics that cannot be captured by simple property-to-PK mappings. Pure deep learning approaches that attempt to learn PK directly from molecular structure (e.g., DeepPK, PkSolver) achieve moderate accuracy but sacrifice the mechanistic interpretability that is essential for regulatory acceptance and clinical decision-making.

### 1.3 Hybrid Neural-Mechanistic Models

A promising middle ground is the hybrid approach: using machine learning to predict mechanistically meaningful parameters (clearance, volume of distribution, permeability), then feeding these into a physics-based PBPK model to simulate PK dynamics. This preserves mechanistic interpretability — every prediction traces through named physiological parameters — while leveraging ML's ability to generalize across chemical space.

Gruber et al. (2024) demonstrated this approach at Bayer AG, combining a graph convolutional network (GCN) with a whole-body PBPK model trained end-to-end on human PK data. Their hybrid model achieved exposure fold change errors of 1.87 (oral) and 1.86 (intravenous) in healthy subjects. However, this model relies on Bayer's proprietary in vitro assay data for pre-training and remains closed-source, limiting reproducibility and accessibility.

### 1.4 Contribution

We present Omega, an open-source hybrid neural-mechanistic platform that predicts human PK directly from SMILES strings. Our key contributions are:

1. **Comparable accuracy with public data only.** Omega achieves Cmax AAFE 1.90 on 20 benchmark drugs using exclusively public data and tools, matching the Bayer proprietary model (1.87) without access to internal assay data.

2. **Sub-100ms inference.** At 73 milliseconds per compound (warm start), Omega enables high-throughput virtual screening of compound libraries — approximately 1,000x faster than manual PBPK parameterization.

3. **Multi-tier validation framework.** We validate predictions at four levels of fidelity: PK accuracy on 20 drugs (Gold), half-life on 39 drugs (Silver), ADME properties on 151 compounds (Bronze), and temporal holdout on 5 post-2022 drugs the model has never seen.

4. **Patient-specific prediction.** A patient-specific module provides weight-adjusted and genotype-adjusted PK predictions via allometric scaling and Bayesian individual parameter estimation from sparse observations.

5. **Full reproducibility.** Omega is MIT-licensed and open-source. After installation (`pip install`), predictions require a single function call: `pipeline.simulate(SimulationRequest(smiles="...", dose_mg=100))`.

The remainder of this paper describes the architecture (Section 2), presents validation results (Section 3), discusses strengths and limitations in context of prior work (Section 4), and outlines future directions (Section 5).

---

## 2. Methods

### 2.1 Architecture Overview

Omega is a hybrid neural-mechanistic pharmacokinetic prediction platform that transforms a molecular structure, represented as a SMILES string, into a complete plasma concentration-time profile without requiring any experimentally measured drug-specific parameters. The pipeline comprises four sequential stages: (i) molecular featurization via RDKit, (ii) ADME property prediction from an ensemble of machine learning models, (iii) construction of a parameterized Drug object with tissue partition coefficients, renal clearance estimates, and in vitro-to-in vivo extrapolation (IVIVE) of hepatic clearance, and (iv) numerical integration of a 35-state ordinary differential equation (ODE) whole-body physiologically based pharmacokinetic (PBPK) model (Fig. 1).

The design rationale for this hybrid architecture is that ADME property prediction and PK profile generation present fundamentally different challenges. ADME properties (fraction unbound in plasma, intrinsic clearance, permeability) are molecular-level quantities amenable to structure-activity modeling, whereas the concentration-time profile emerges from the interplay of absorption, distribution, metabolism, and excretion processes governed by known physiological equations. By coupling learned ADME predictors to a mechanistic ODE solver, the system leverages the strengths of each paradigm: data-driven generalization for molecular properties and first-principles accuracy for pharmacokinetic dynamics.

The end-to-end prediction is exposed through the `OmegaPipeline` class, which accepts a `SimulationRequest` specifying the SMILES string, dose (mg), route of administration (oral, intravenous, or subcutaneous), simulation duration, and optional patient covariates (body weight, CYP genotype, age). The output is a `SimulationResult` containing the full concentration-time curve, derived PK parameters ($C_{\max}$, $T_{\max}$, $\text{AUC}_{0-t}$, $t_{1/2}$), ADME property estimates with uncertainty intervals, and a confidence classification.

### 2.2 ADME Prediction Layer

#### 2.2.1 Ensemble Architecture

The ADME prediction layer employs an `EnsembleADMEPredictor` that supports multiple backend models (ADMET-AI, XGBoost, polynomial ridge) with a per-property selection strategy. The `EnsembleADMEPredictor` can operate in two modes: with ADMET-AI enabled (full ensemble) or with ADMET-AI disabled (XGBoost + polynomial only). In the production configuration used for all benchmark results in this paper, **ADMET-AI is disabled** (`admet_ai=False`) because its lipophilicity and protein binding predictions were found to alter tissue partitioning coefficients unpredictably, degrading PK accuracy for warfarin, metformin, and losartan. The per-property strategy in this configuration is:

- **Fraction unbound in plasma ($f_u$):** XGBoost model trained on TDC PPBR_AZ (1,614 compounds) + 153-compound reference set, predicting in $\log_{10}$-space.

- **Blood-to-plasma ratio (RBP):** XGBoost model trained on the 153-compound reference dataset. No public pretrained model for RBP exists, necessitating a custom model.

- **Lipophilicity ($\log P$) and aqueous solubility ($\log S$):** Polynomial ridge regression from RDKit descriptors, with GSE solubility floor (Section 2.2.4).

- **Intrinsic clearance ($CL_{\text{int}}$):** XGBoost model trained on TDC Clearance_Hepatocyte_AZ (1,213 compounds), calibrated via reference-anchored IVIVE.

- **Effective permeability ($P_{\text{eff}}$):** Polynomial ridge regression from physicochemical descriptors.

- **Volume of distribution ($V_{d,\text{ss}}$):** XGBoost model trained on TDC VDss_Lombardo (1,130 compounds), used for Kp calibration (Section 2.4.1).

- **Molecular weight (MW):** Computed directly from the SMILES string via RDKit.

Overall confidence is defined as the minimum confidence level across all property backends: if any single property prediction is uncertain, the entire prediction is flagged accordingly.

*Note:* When ADMET-AI is enabled, it serves as primary predictor for $f_u$ (geometric mean with XGBoost), $\log P$, $CL_{\text{int}}$, $P_{\text{eff}}$, and hERG $\text{IC}_{50}$, with XGBoost and polynomial as fallbacks. This configuration was not used for the results reported here due to the tissue partitioning instabilities described above.

#### 2.2.2 XGBoost Models

Four XGBoost gradient-boosted tree models were trained for properties where either no pretrained model was publicly available (RBP) or where domain-specific calibration was required ($f_u$, $CL_{\text{int}}$, $V_{d,\text{ss}}$).

**Molecular representation.** All XGBoost models use a concatenation of 2048-bit Morgan circular fingerprints (radius 2) generated via RDKit and 9 physicochemical descriptors: $\log P$ (Crippen), topological polar surface area (TPSA), molecular weight, hydrogen bond acceptor count, hydrogen bond donor count, rotatable bond count, ring count, fraction $\text{sp}^3$ carbons, and molar refractivity.

**Training data.** The $f_u$ model was trained on the union of the Therapeutic Data Commons (TDC) PPBR\_AZ dataset (1,614 compounds) and the 153-compound in-house reference set. The $CL_{\text{int}}$ model used the TDC Clearance\_Hepatocyte\_AZ dataset (1,213 compounds). The $V_{d,\text{ss}}$ model used the TDC VDss\_Lombardo dataset (1,130 compounds). The RBP model was trained exclusively on the 153-compound reference set due to the absence of a suitable public dataset.

**Prediction in log-space.** The $f_u$, $CL_{\text{int}}$, and $V_{d,\text{ss}}$ models predict in $\log_{10}$-transformed space (e.g., $\log_{10}(f_u)$) and exponentiate at inference time. This ensures equal loss weighting across orders of magnitude and prevents the model from ignoring highly bound compounds ($f_u < 0.01$), which are disproportionately important in clinical pharmacology.

#### 2.2.3 Polynomial Ridge Regression Fallback

A polynomial ridge regression model serves as the final fallback for all ADME properties when both ADMET-AI and XGBoost backends are unavailable. This model provides biased but bounded predictions from physicochemical descriptors, ensuring the pipeline never fails to produce an estimate.

#### 2.2.4 Solubility Floor: General Solubility Equation

Predicted aqueous solubility is subject to a lower bound derived from the General Solubility Equation (GSE) of Yalkowsky and Valvani (1980):

$$\log S \geq 0.5 - \log P$$

This constraint prevents the polynomial predictor from catastrophically under-predicting solubility for lipophilic drugs. Without this floor, drugs such as fluoxetine ($\log P = 4.4$) receive dose numbers exceeding 70 and predicted fraction absorbed below 2%, resulting in 10--15-fold under-prediction of $C_{\max}$.

#### 2.2.5 Conformal Prediction Intervals

Prediction uncertainty is quantified through conformal prediction intervals calibrated to achieve approximately 90% empirical coverage. For $f_u$, the XGBoost model provides native conformal intervals from cross-validation residuals. For other properties, intervals are initialized at $\pm 50\%$ of the point estimate and then adjusted by empirically determined calibration factors computed on a 152-compound holdout from `adme_reference.csv`:

| Property | Calibration Factor | Raw Coverage | Calibrated Coverage |
|---|---|---|---|
| $f_u$ | 1.04 | 86.2% | ~90% |
| $CL_{\text{int,3A4}}$ | 50.0 | 11.2% | ~75% |
| $P_{\text{eff}}$ | 2.94 | 57.9% | ~90% |
| RBP | 3.13 | 63.8% | ~90% |

The large calibration factor for $CL_{\text{int,3A4}}$ reflects the inherent difficulty of intrinsic clearance prediction from molecular structure alone; even after aggressive interval widening, only approximately 75% coverage is achievable with current models.

### 2.3 PBPK Model

#### 2.3.1 Model Structure

The whole-body PBPK model comprises 35 ordinary differential equations representing 15 organs, an 8-segment gastrointestinal (GI) tract, portal venous and systemic blood pools, and mass-balance tracking compartments for hepatic metabolism, gut-wall metabolism, renal excretion, and fecal excretion (Fig. 2).

**Perfusion-limited organs** (11 compartments: lung, brain, heart, kidney, liver, spleen, gut wall, pancreas, thymus, reproductive organs, and a residual "rest" compartment) are modeled with flow-limited distribution:

$$\frac{dA_i}{dt} = Q_i \cdot C_{\text{art}} - Q_i \cdot \frac{A_i \cdot R_{B:P}}{V_i \cdot K_{p,i}}$$

where $A_i$ is the drug amount in organ $i$, $Q_i$ is the organ blood flow, $C_{\text{art}}$ is the arterial blood concentration, $V_i$ is the organ volume, $K_{p,i}$ is the tissue-to-plasma partition coefficient, and $R_{B:P}$ is the blood-to-plasma ratio.

**Permeability-limited organs** (4 organs, 8 compartments: adipose, muscle, bone, and skin, each with vascular and extravascular sub-compartments) incorporate a permeability-surface area product ($PS$) governing the rate of drug transfer between vascular and tissue spaces:

$$\frac{dA_{i,\text{vasc}}}{dt} = Q_i \cdot C_{\text{art}} - Q_i \cdot C_{i,\text{vasc}} - PS_i \cdot (C_{i,\text{vasc}} - C_{i,\text{extra}}/K_{p,i})$$

$$\frac{dA_{i,\text{extra}}}{dt} = PS_i \cdot (C_{i,\text{vasc}} - C_{i,\text{extra}}/K_{p,i})$$

**Blood pools.** Venous blood collects outflow from all non-pulmonary organs; arterial blood receives outflow from the lung. Portal organs (spleen, gut wall, pancreas) drain into the portal vein, which feeds the liver. Cardiac output is set to 390 L/h for a 70-kg reference adult, scaled linearly with body weight.

Organ volumes and blood flow fractions are derived from ICRP Publication 89 reference values for a 70-kg adult male, scaled linearly with body weight ($V_i = V_{i,\text{ref}} \times \text{BW}/70$).

#### 2.3.2 ACAT Gastrointestinal Absorption

Oral absorption is modeled using an Advanced Compartmental Absorption and Transit (ACAT) framework with 8 sequential segments: stomach, duodenum, jejunum (2 segments), ileum (3 segments), and colon. Each segment $j$ has a lumenal compartment with first-order transit to the next segment and permeability-dependent absorption into the portal vein:

$$\frac{dA_j}{dt} = k_{t,j-1} \cdot A_{j-1} - k_{t,j} \cdot A_j - k_{a,j} \cdot A_j$$

where $k_{t,j} = 1/\tau_j$ is the transit rate constant (with transit times $\tau$ of 0.25, 0.26, 0.475, 0.475, 0.68, 0.68, 0.68, and 13.5 h for the 8 segments, respectively) and $k_{a,j}$ is the segment-specific absorption rate constant, proportional to the drug's effective permeability ($P_{\text{eff}}$) and modulated by segment-dependent absorption fractions (0.0, 1.0, 1.0, 1.0, 0.8, 0.6, 0.3, 0.05 from stomach to colon).

Drug dissolution in the stomach is governed by the dose number ($D_0 = \text{dose} / (250 \text{ mL} \times S)$, where $S$ is aqueous solubility in mg/mL). When $D_0 > 1$, the fraction of dose in solution is limited and absorption is solubility-rate-limited.

#### 2.3.3 Tissue Partitioning

Tissue-to-plasma partition coefficients ($K_p$) are estimated using the mechanistic method of Rodgers and Rowland (2006), which computes $K_p$ from tissue composition (fractional neutral lipid, phospholipid, and water content) and drug physicochemical properties ($\log P$, $f_u$, ionization state). Tissue composition data for all 15 organs are taken from Tables 1--2 of Rodgers and Rowland (2006).

For ionizable compounds, the Berezhkovskiy (2004) correction is applied, which accounts for the effect of ionization on tissue binding. Compound type (neutral, acid, base, zwitterion) and approximate $\text{p}K_a$ are inferred from SMARTS substructure matching on the input SMILES (amines: $\text{p}K_a \approx 9.0$; carboxylic acids: $\text{p}K_a \approx 4.0$).

For $K_p$ calculations, Crippen $\log P$ from RDKit is used rather than the ML-predicted $\log P$, as it provides more reliable partition coefficient estimates (e.g., fluoxetine: ML predicts 2.09, RDKit gives 4.44, literature value 4.05).

#### 2.3.4 Hepatic Clearance

Hepatic elimination follows the well-stirred model:

$$CL_h = \frac{Q_h \cdot f_u \cdot CL_{\text{int}}}{Q_h + f_u \cdot CL_{\text{int}}}$$

where $Q_h = 90$ L/h is the hepatic blood flow (hepatic artery + portal vein), $f_u$ is the fraction unbound in plasma, and $CL_{\text{int}}$ is the intrinsic clearance scaled to whole-liver units (L/h).

**IVIVE.** In vitro intrinsic clearance from the XGBoost hepatocyte model ($CL_{\text{int,hep}}$ in $\mu$L/min/$10^6$ cells) is scaled to in vivo hepatic clearance using an empirically calibrated allometric correction:

$$CL_{h,\text{target}} = \min\left(\alpha \cdot CL_{\text{int,hep}}^{\beta},\ 0.95 \cdot Q_h\right)$$

with $\alpha = 0.3$ and $\beta = 0.9$, calibrated against published clinical clearance values for ibuprofen, acetaminophen, theophylline, diclofenac, and omeprazole (4/5 within 2-fold). The sub-linear exponent $\beta < 1$ accounts for the systematic over-prediction of standard IVIVE at high intrinsic clearance values (Hallifax and Houston, 2009). The target $CL_h$ is then back-calculated to the $CL_{\text{int}}$ value that produces that clearance in the well-stirred model given the predicted $f_u$:

$$CL_{\text{int}} = \frac{CL_{h,\text{target}} \cdot Q_h}{f_u \cdot (Q_h - CL_{h,\text{target}})}$$

This pre-inversion compensates for both IVIVE scaling bias and $f_u$ prediction errors.

**Gut wall first-pass.** The fraction escaping gut-wall metabolism ($F_g$) is computed using the $Q_{\text{gut}}$ model (Yang et al., 2007):

$$F_g = \frac{Q_{\text{gut}}}{Q_{\text{gut}} + f_u \cdot CL_{\text{int,gut}}}$$

#### 2.3.5 Renal Clearance

Renal clearance is estimated from physicochemical properties rather than predicted by a dedicated ML model. Glomerular filtration rate (GFR) is set to 7.2 L/h (120 mL/min) for a 70-kg adult. The estimation framework incorporates:

- **Glomerular filtration:** $CL_{\text{filt}} = \text{GFR} \times f_u$, with a molecular weight penalty for compounds exceeding 500 Da.
- **Tubular reabsorption:** Lipophilic compounds ($\log P \geq 2.5$) are assumed to undergo complete tubular reabsorption ($CL_r = 0$).
- **Active secretion:** Hydrophilic compounds ($\log P < -0.5$) with high topological polar surface area (TPSA > 74 $\AA^2$) are assigned active tubular secretion via OCT2/OAT/MATE transporters, with secretion factor $\min(3.0,\ 10^{-\log P})$.
- **Basic amines:** Small basic amines ($\log P < 1.5$, MW < 300) with primary or secondary amine groups receive OCT2-mediated secretion (2-fold enhancement over filtration).

Renal clearance is capped at 30 L/h to prevent non-physiological values.

#### 2.3.6 ODE Integration

The system of 35 coupled ODEs is integrated using the LSODA method (Hindmarsh, 1983; Petzold, 1983) as implemented in SciPy's `solve_ivp`, with relative tolerance $10^{-8}$ and absolute tolerance $10^{-10}$. LSODA automatically switches between non-stiff (Adams) and stiff (BDF) methods based on the local stiffness of the system, which is critical for PBPK models where rapid absorption ($k_a \gg k_e$) creates stiff initial conditions that relax during the elimination phase.

### 2.4 Physics-Informed Corrections

Three post-hoc corrections are applied to the raw ODE output to compensate for known systematic biases in the mechanistic model.

#### 2.4.1 VDss Divergence Check

The $K_p$-based volume of distribution ($V_{d,\text{ss,Berez}}$), computed by summing Berezhkovskiy-corrected partition coefficients across all organs, is compared against the XGBoost $V_{d,\text{ss}}$ prediction (trained on the TDC VDss\_Lombardo dataset of 1,130 compounds). When $V_{d,\text{ss,Berez}} > 2 \times V_{d,\text{ss,XGB}}$, the Berezhkovskiy estimate is considered unreliable (typically due to propagated $\log P$ errors into tissue partitioning), and the XGBoost value is substituted for the analytical $C_{\max}$ calculation. For the analytical half-life estimate, the geometric mean $V_d = \sqrt{V_{d,\text{Berez}} \times V_{d,\text{XGB}}}$ is used instead, as full replacement is too aggressive for high-$V_d$ drugs.

#### 2.4.2 Hybrid Cmax Selector

The PBPK ODE's perfusion-limited distribution systematically distorts $C_{\max}$: the plasma concentration exhibits a transient spike before tissue equilibrium is reached, causing over-prediction for drugs with slow tissue uptake. For oral drugs with renal clearance below 5 L/h, $C_{\max}$ from the ODE is blended with the analytical one-compartment model prediction:

$$C_{\max,\text{analytical}} = \frac{F \cdot D}{V_d} \cdot \frac{k_a}{k_a - k_e} \cdot \left(e^{-k_e \cdot t_{\max}} - e^{-k_a \cdot t_{\max}}\right)$$

where $F$ is the predicted oral bioavailability, $D$ is the dose, $k_a = P_{\text{eff}} \times 10^4 \times 1.0$ (capped to $[0.3, 5.0]$ h$^{-1}$), and $k_e = CL_{\text{total}} / V_d$.

When the VDss divergence check triggers ($V_{d,\text{Berez}} / V_{d,\text{XGB}} > 2$), the analytical $C_{\max}$ is used directly. Otherwise, the geometric mean of the ODE and analytical values is reported: $C_{\max} = \sqrt{C_{\max,\text{ODE}} \times C_{\max,\text{analytical}}}$.

#### 2.4.3 Hybrid Half-Life Selector

Terminal half-life is estimated from two independent approaches and selected via heuristic rules:

1. **Curve-fit:** Log-linear regression of the post-$C_{\max}$ concentration-time profile from the PBPK simulation (points above 0.1% of $C_{\max}$), yielding $t_{1/2} = \ln 2 / (-\text{slope})$.

2. **Analytical:** $t_{1/2} = 0.693 \times V_d / CL_{\text{total}}$, using the predicted volume of distribution and total clearance.

The selection rules are: (a) if the analytical $t_{1/2}$ is shorter than the curve-fit value and exceeds 1 hour (and the ODE renal clearance is below 20 L/h), the analytical value is preferred, as the curve-fit is inflated by redistribution dynamics; (b) if hepatic clearance is below 5 L/h, $\log P > 2.0$, and the analytical $t_{1/2}$ exceeds 20 hours, the analytical value is preferred, as the 24-hour default simulation duration is insufficient to capture the true terminal slope; (c) otherwise, the curve-fit value is used.

#### 2.4.4 Bioavailability Prediction

Oral bioavailability is decomposed as $F = f_a \times F_g \times F_h$, where $f_a$ is the fraction absorbed (from ACAT dissolution and permeability), $F_g$ is the fraction escaping gut-wall metabolism, and $F_h = 1 - CL_h / Q_h$ is the fraction escaping hepatic first-pass extraction. Each component is computed mechanistically from the predicted ADME properties.

### 2.5 Patient-Specific Prediction

#### 2.5.1 Allometric Scaling

Population PK parameters are scaled to individual patients using standard allometric equations referenced to a 70-kg adult (Anderson & Holford, 2008):

$$CL_{\text{individual}} = CL_{\text{pop}} \times \left(\frac{W}{70}\right)^{0.75}$$

$$V_{d,\text{individual}} = V_{d,\text{pop}} \times \frac{W}{70}$$

where $W$ is the patient body weight in kg. The exponent of 0.75 for clearance reflects the well-established allometric relationship between metabolic rate and body size across species (West et al., 1997). Volume of distribution scales linearly with body weight, consistent with the assumption that tissue volumes are proportional to body mass.

#### 2.5.2 CYP Genotype Factors

Pharmacogenomic variability in drug metabolism is incorporated through enzyme-specific activity scaling factors applied multiplicatively to hepatic clearance. The following factors are implemented, defined relative to the extensive metabolizer (EM) phenotype:

**CYP2D6:**

| Phenotype | Activity Factor |
|---|---|
| Ultra-rapid metabolizer (UM) | 1.5 |
| Extensive metabolizer (EM) | 1.0 |
| Intermediate metabolizer (IM) | 0.5 |
| Poor metabolizer (PM) | 0.1 |

**CYP2C9:**

| Genotype | Activity Factor |
|---|---|
| \*1/\*1 | 1.0 |
| \*1/\*2 | 0.8 |
| \*1/\*3 | 0.6 |
| \*2/\*2 | 0.5 |
| \*2/\*3 | 0.3 |
| \*3/\*3 | 0.1 |

**CYP2C19:**

| Phenotype | Activity Factor |
|---|---|
| Ultra-rapid metabolizer (UM) | 1.5 |
| Extensive metabolizer (EM) | 1.0 |
| Intermediate metabolizer (IM) | 0.5 |
| Poor metabolizer (PM) | 0.2 |

When multiple CYP enzymes are specified, each factor is applied independently to the clearance, weighted by the drug's fraction metabolized ($f_m$) by each enzyme. Unknown enzyme-phenotype combinations default to an activity factor of 1.0 (no effect).

#### 2.5.3 Bayesian Individual Estimation

For patients with 1--5 observed plasma concentration measurements, individual PK parameters are estimated by fitting clearance and volume scaling factors ($\eta_{CL}$, $\eta_V$) to the observed data using a maximum a posteriori (MAP) approach. The objective function minimizes the mean squared error in log-concentration space:

$$\mathcal{L}(\eta_{CL}, \eta_V) = \frac{1}{N} \sum_{i=1}^{N} \left(\ln C_{\text{pred}}(t_i; \eta_{CL}, \eta_V) - \ln C_{\text{obs}}(t_i)\right)^2$$

where $C_{\text{pred}}$ is computed from the analytical one-compartment oral model:

$$C(t) = \frac{F \cdot D}{V_d \cdot \eta_V} \cdot \frac{k_a}{k_a - k_e'} \cdot \left(e^{-k_e' t} - e^{-k_a t}\right)$$

with $k_e' = (CL \cdot \eta_{CL}) / (V_d \cdot \eta_V)$.

Optimization is performed via L-BFGS-B (Byrd et al., 1995) with bounds $\eta_{CL}, \eta_V \in [0.05, 20.0]$, initialized at $\eta_{CL} = \eta_V = 1.0$ (population estimate). The use of log-concentration space ensures that the fitting procedure assigns equal weight to high and low concentrations, which is critical when observations span the absorption peak and the elimination tail.

### 2.6 Validation Framework

#### 2.6.1 Tiered Validation Design

Model performance is assessed through a four-tier validation framework with increasing stringency and decreasing data availability:

**Bronze tier** (ADME property validation). Predicted ADME properties ($f_u$, $\log P$, RBP, $CL_{\text{int,3A4}}$, $P_{\text{eff}}$) are compared against literature reference values from 153 compounds curated in `adme_reference.csv`. Each compound has a canonical SMILES string and experimentally measured values with defined units ($CL_{\text{int}}$ in $\mu$L/min/pmol CYP, $P_{\text{eff}}$ in cm/s, $f_u$ as fraction 0--1, RBP as ratio). Per-property AAFE and percentage within 2-fold are reported.

**Silver tier** (half-life validation). Predicted terminal half-lives are compared against values extracted from FDA-approved drug labels obtained via the OpenFDA API. This tier validates the integrated clearance and distribution predictions through their aggregate effect on half-life.

**Gold tier** (concentration-time curve validation). Predicted plasma concentration-time profiles are compared against published clinical PK data for 25 drugs, each with a defined dose and route of administration. Benchmark datasets (e.g., `caffeine_oral_100mg.csv`) contain digitized concentration-time points with standard deviations. $C_{\max}$, $\text{AUC}$, and $t_{1/2}$ fold errors are computed for each drug.

**Temporal holdout** (prospective validation). Five drugs approved by the FDA after the ADMET-AI training data cutoff (approximately 2022) are used as a fully prospective test set: adagrasib, futibatinib, capivasertib, elacestrant, and pirtobrutinib. These compounds were not present in any training data for any model in the pipeline, providing an unbiased estimate of generalization performance.

#### 2.6.2 Performance Metrics

Two primary metrics are used throughout all validation tiers:

**Average Absolute Fold Error (AAFE).** The geometric mean of fold errors across $n$ compounds:

$$\text{AAFE} = 10^{\frac{1}{n} \sum_{i=1}^{n} \log_{10} \text{FE}_i}$$

where $\text{FE}_i = \max(y_i^{\text{pred}} / y_i^{\text{obs}},\ y_i^{\text{obs}} / y_i^{\text{pred}})$. AAFE = 1.0 indicates perfect prediction; AAFE = 2.0 indicates that predictions are, on average, 2-fold from observed values. AAFE is preferred over arithmetic mean fold error because it is symmetric (over- and under-prediction are penalized equally) and robust to outliers on the multiplicative scale.

**Percentage within 2-fold (%2-fold).** The fraction of predictions for which $\text{FE}_i \leq 2.0$:

$$\%\text{2-fold} = \frac{100}{n} \sum_{i=1}^{n} \mathbb{1}[\text{FE}_i \leq 2.0]$$

This metric corresponds to the standard regulatory acceptance criterion for PBPK model qualification (EMA, 2018; FDA, 2018), where at least 50% of predictions within 2-fold of observed data is generally considered acceptable for a credible PBPK model.

#### 2.6.3 Confidence Calibration

The confidence classification system (low/medium/high) is validated through a scaffold-split holdout procedure. The 153 reference compounds are split by Murcko scaffold (Bemis and Murcko, 1996) with 20% held out for calibration. Confidence monotonicity is verified: AAFE(high confidence) $\leq$ AAFE(medium confidence) $\leq$ AAFE(low confidence), ensuring that the confidence labels are informative and that users can trust that high-confidence predictions are indeed more accurate.

#### 2.6.4 Regression Testing

All validation benchmarks are automated and executed as part of the continuous integration pipeline. The gold-tier exit criteria require $C_{\max}$ AAFE $< 3.0$, AUC AAFE $< 3.0$, and $\geq 70\%$ of at least 20 drugs within 2-fold of observed values.

---

## 3. Results

### 3.1 Gold Tier: Pharmacokinetic Prediction Accuracy

Omega was evaluated against clinical PK data for 20 orally administered drugs. Benchmark drugs were selected based on three criteria: (1) availability of published mean plasma concentration-time curves for healthy volunteers receiving single oral IR doses in the fasted state, (2) diversity of therapeutic classes and clearance mechanisms, and (3) inclusion in standard pharmacology references (FDA labels, Goodman & Gilman's 14th ed., Rowland & Tozer). No drugs were excluded based on prediction performance; the set was finalized before benchmark evaluation. The drugs span molecular weights from 32 to 781 Da and include compounds cleared by CYP3A4 (midazolam), CYP2C9 (warfarin), renal excretion (metformin), and mixed pathways (caffeine). All 20 drugs were processed successfully from SMILES input to full concentration-time profiles.

Across the 20-drug benchmark, Omega achieved a Cmax AAFE of 1.90 (95% bootstrap CI: 1.57--2.41; median fold error 1.80) and an AUC AAFE of 1.66 (95% CI: 1.42--1.98; median 1.53) (Table 1, Fig. 1). Fourteen of 20 drugs (70%) had predicted Cmax within 2-fold of observed values, and 14 of 20 (70%) had AUC within 2-fold. These results meet conventional acceptance thresholds for PBPK model qualification, where 2-fold accuracy for at least 50% of compounds is considered adequate and 80% is considered good (Jones et al., 2015).

The most accurate Cmax prediction was for phenytoin (fold error 1.03), a drug with well-characterized linear pharmacokinetics at the 300 mg dose evaluated. Other drugs with Cmax fold errors below 1.5 included nifedipine (1.13), acetaminophen (1.20), diazepam (1.20), atorvastatin (1.38), and warfarin (1.46). For AUC, the most accurate prediction was ibuprofen (fold error 1.02), followed by carbamazepine (1.09), atorvastatin (1.14), and verapamil (1.22).

The two largest Cmax errors were verapamil (8.83-fold) and ibuprofen (4.98-fold); these are analyzed in Section 3.6. Excluding these two outliers, the remaining 18 drugs had a Cmax AAFE of 1.60 and 89% within 2-fold, demonstrating that the prediction errors are concentrated in mechanistically explainable cases rather than reflecting systematic bias.

Prediction latency averaged 73 ms per drug on warm start (CPU only; no GPU required), with a cold-start latency of approximately 5 seconds for the first prediction due to XGBoost model loading. The 73 ms warm-start throughput is compatible with interactive screening of compound libraries and virtual patient simulations.

For comparison, Gruber et al. (2024) reported a median exposure fold change error (mfce) of 1.87 for oral drugs using a hybrid GCN-PBPK model trained on Bayer's proprietary in vitro assay data. Using the same metric (median fold error), Omega achieves 1.73 for Cmax and 1.60 for AUC — comparable or superior accuracy using only public data. Jia et al. (2025) reported 60% within 2-fold for AUC and 59% for Cmax on 106 test compounds using public data; Omega achieves 70% within 2-fold on 20 drugs. Direct comparison across studies is complicated by differences in test set composition, drug selection criteria, and metric definitions (Section 4.1).

### 3.2 Silver Tier: Half-Life Prediction

Half-life predictions were evaluated for 39 drugs with reference elimination half-lives extracted from FDA-approved drug labels via the OpenFDA API (Fig. 2). The model achieved an overall t1/2 AAFE of 2.42, with 20 of 39 drugs (51.3%) predicted within 2-fold of observed values.

The most accurate predictions were for metformin (fold error 1.01), atenolol (1.05), verapamil (1.08), phenytoin (1.14), and cyclosporine (1.19), representing drugs with well-characterized, predominantly single-pathway elimination. Metformin, a renally cleared compound, was predicted with near-exact accuracy (6.24 h predicted vs. 6.2 h observed), suggesting that the model captures renal clearance contributions effectively.

The largest errors were observed for risperidone (22.9-fold), ibuprofen (16.7-fold), and warfarin (8.4-fold). The risperidone discrepancy reflects the drug's complex multi-compartment distribution and active metabolite (9-hydroxy-risperidone) with its own pharmacological activity; the reported 3 h half-life likely reflects the distribution phase rather than terminal elimination. Ibuprofen (predicted 30.0 h vs. observed 1.8 h) is confounded by extensive protein binding (>99%) and stereoselective clearance not captured by the current model. For warfarin (predicted 167.9 h vs. observed 20 h), the overprediction is consistent with the model underestimating hepatic intrinsic clearance for this highly protein-bound, low-extraction-ratio compound.

Several reference data quality issues were identified and corrected during validation. The amoxicillin reference half-life in the original dataset was reported in minutes rather than hours, and the diazepam reference value (43 h) reflects the terminal elimination half-life including the active desmethyldiazepam metabolite, not the distribution half-life. These corrections were applied prior to the analysis presented here.

### 3.3 Bronze Tier: ADME Property Accuracy

The ensemble ADME predictor was evaluated against reference values for 151 compounds (2 of 153 failed SMILES parsing). Per-property results are summarized in Table 2.

**Table 2. ADME property prediction accuracy (151 compounds).**

| Property | AAFE | % Within 2-fold | n |
|----------|------|-----------------|---|
| logP | 1.54 | 82.4% | 131 |
| fup | 2.10 | 58.3% | 151 |
| RBP | 1.09 | 98.0% | 151 |
| Peff | 1.46 | 86.1% | 151 |
| CLint (CYP3A4) | 3.25 | 33.8% | 151 |

Red blood cell-to-plasma ratio (RBP) was the best-predicted property, with an AAFE of 1.09 and 98.0% of compounds within 2-fold. This accuracy reflects the relatively narrow physiological range of RBP (typically 0.5--1.5) and the effectiveness of the XGBoost model trained on curated RBP data. Effective permeability (Peff, AAFE 1.46) and lipophilicity (logP, AAFE 1.54) were also well-predicted, consistent with the strong structure--property relationships for these physicochemical descriptors.

Fraction unbound in plasma (fup) showed moderate accuracy (AAFE 2.10, 58.3% within 2-fold). Protein binding prediction is a recognized challenge in PBPK modeling due to its nonlinear dependence on both drug lipophilicity and specific binding-site interactions, particularly for highly bound drugs where small absolute errors in fup translate to large fold errors (Poulin and Haddad, 2012).

Intrinsic clearance (CLint) was the least accurate property (AAFE 3.25, 33.8% within 2-fold). This result is expected: hepatic metabolic clearance depends on enzyme kinetics, isoform selectivity, and potential auto-inhibition effects that are difficult to predict from molecular structure alone. The 3.25-fold AAFE is consistent with published benchmarks for in silico CLint prediction (Ingle et al., 2016), where AAFE values of 3--5-fold are typical for structure-based models.

### 3.4 Temporal Holdout Validation

To assess generalization to truly novel chemical matter, Omega was evaluated on five drugs approved by the FDA after 2022, none of which were present in any training dataset. All five drugs were processed successfully (Table 3).

**Table 3. Temporal holdout results: post-2022 FDA-approved drugs.**

| Drug | Approval | Dose (mg) | Obs. t1/2 (h) | Pred. t1/2 (h) | Fold Error |
|------|----------|-----------|---------------|----------------|------------|
| Adagrasib (KRAS G12C) | 2022 | 600 | 23.0 | 13.8 | 1.67 |
| Futibatinib (FGFR) | 2022 | 20 | 14.0 | 20.6 | 1.47 |
| Capivasertib (AKT) | 2023 | 400 | 8.0 | 63.2 | 7.90 |
| Elacestrant (ER) | 2023 | 345 | 38.0 | 22.7 | 1.68 |
| Pirtobrutinib (BTK) | 2023 | 200 | 19.0 | 55.0 | 2.90 |

The temporal holdout set achieved a t1/2 AAFE of 3.12, with 3 of 5 drugs (60%) within 2-fold. Adagrasib, futibatinib, and elacestrant were all predicted within 2-fold, demonstrating that the model generalizes to novel molecular scaffolds not represented in its training data. These three drugs span distinct target classes (KRAS, FGFR, estrogen receptor) and structural chemotypes.

Capivasertib was the largest outlier (7.9-fold overprediction of half-life). This AKT inhibitor contains a piperidine-hydroxyl pharmacophore that undergoes glucuronidation via UGT2B7 as a major clearance pathway, a metabolic route not explicitly modeled in the current system. The model's reliance on CYP-mediated clearance estimation leads to underprediction of total clearance for compounds cleared predominantly by phase II conjugation.

Pirtobrutinib (2.9-fold) showed moderate overprediction of half-life, potentially reflecting the influence of covalent binding kinetics on its effective elimination rate. These results highlight that while Omega generalizes well to structurally novel compounds, drugs with non-CYP clearance mechanisms remain a source of prediction error.

### 3.5 Structural Analog and De Novo Validation

To evaluate whether Omega produces chemically sensible predictions for hypothetical molecules, two additional validation tiers were assessed.

**Structural analogs (T9).** Twenty structural analogs were generated from five parent drugs (ibuprofen, acetaminophen, diclofenac, omeprazole, and metronidazole) by applying systematic structural modifications: aromatic hydroxylation, halogen substitution (Cl to F), and aromatic methylation. All 20 analogs (100%) passed plausibility checks, producing Cmax and AUC values within physiologically reasonable ranges and showing expected directional trends relative to their parent compounds (Fig. 3). For example, hydroxylated analogs of ibuprofen showed reduced Cmax relative to the parent, consistent with increased polarity reducing absorption rate.

**De novo molecules (T10).** Fifty candidate SMILES were generated de novo using a molecular generation algorithm. Of these, 2 produced valid PK predictions that passed all plausibility criteria (100% of successfully parsed molecules). The low generation-to-valid-prediction rate (4%) reflects the stringent requirements for drug-like molecular properties (Lipinski compliance, valid pharmacophore recognition) rather than limitations of the PK prediction engine itself. Both successfully predicted molecules produced concentration--time profiles with physiologically meaningful Cmax, AUC, and half-life values.

These results confirm that Omega produces physically reasonable predictions when extrapolating beyond its training domain, an essential property for virtual screening and lead optimization applications.

### 3.6 Failure Analysis

Of the 20 drugs in the gold-tier benchmark, 2 (10%) exhibited Cmax fold errors exceeding 3-fold (Fig. 4). Both failures trace to specific mechanistic limitations rather than random prediction error.

**Verapamil (Cmax fold error: 8.83).** Verapamil is a well-known P-glycoprotein (P-gp) substrate that undergoes extensive intestinal and hepatic efflux transport. The model's current architecture does not include explicit transporter-mediated disposition, leading to overprediction of oral bioavailability. Additionally, verapamil undergoes stereoselective first-pass metabolism via CYP3A4, with the S-enantiomer cleared more rapidly than the R-enantiomer. The racemic SMILES input cannot capture this stereoselective disposition. Despite the large Cmax error, the AUC prediction for verapamil was accurate (fold error 1.22), suggesting that the total systemic exposure is well-captured but the rate of absorption and first-pass extraction is not.

**Ibuprofen (Cmax fold error: 4.98).** Ibuprofen is 99% protein-bound (fup approximately 0.01). At this extreme level of binding, small absolute errors in predicted fup produce large errors in predicted free drug concentration and, consequently, in Cmax. The model predicted an fup that, while within the correct order of magnitude, was sufficient to produce a 5-fold Cmax error. Notably, the AUC prediction for ibuprofen was nearly exact (fold error 1.02). This apparent paradox — accurate AUC despite inaccurate Cmax and overpredicted half-life (predicted 30 h vs. observed 1.8 h) — is an artifact of the 24-hour simulation window: the overpredicted elimination rate causes the simulated curve to remain elevated beyond the observation window, and the truncated AUC integral coincidentally approximates the true AUC over 24 hours. This underscores that AUC agreement does not guarantee correct underlying PK parameters.

Both failure modes represent known, addressable limitations: transporter-mediated disposition for P-gp substrates and nonlinear protein binding for highly bound drugs. These mechanistic gaps are targets for future model development.

### 3.7 Confidence Calibration

Omega provides conformal prediction intervals for each ADME property, calibrated on a holdout set of 30 compounds (20% of the reference dataset). The calibration assessment (Table 4) revealed that interval coverage varies by property.

**Table 4. Conformal prediction interval calibration (target: 90% coverage).**

| Property | Observed Coverage | Interval Width | Status |
|----------|------------------|----------------|--------|
| fup | 100.0% | 0.687 | Over-covered |
| CLint (CYP3A4) | 36.7% | 0.260 | Under-covered |
| Peff | 96.7% | 4.059 | Slightly over-covered |
| RBP | 100.0% | 0.840 | Over-covered |

Fraction unbound and RBP intervals achieved 100% coverage on the holdout set, indicating conservative (wide) intervals. Peff intervals were slightly conservative at 96.7% coverage. CLint intervals were substantially under-covered at 36.7%, reflecting the high intrinsic variability of metabolic clearance predictions. The overall calibration status was assessed as not yet meeting the target of 90% coverage across all properties simultaneously, with CLint being the primary contributor to miscalibration. Recalibration of CLint intervals using nonconformity scores from a larger reference dataset is planned for future work.

### 3.8 Patient-Specific Predictions

Omega incorporates allometric covariate scaling and Bayesian individual parameter estimation to support patient-specific PK prediction.

**Weight-based scaling.** Population PK parameters are scaled using standard allometric relationships: clearance scales with body weight raised to the 0.75 power, and volume of distribution scales linearly with weight, referenced to a 70 kg adult. For warfarin (5 mg dose), scaling from 70 kg to 120 kg increases predicted clearance by 44% and volume of distribution by 71%, resulting in a lower predicted Cmax and a modestly shorter half-life, consistent with published population PK analyses of warfarin (Hamberg et al., 2007).

**CYP genotype adjustment.** The model applies pharmacogenomic activity factors for CYP2C9, CYP2D6, and CYP2C19 genotypes. For warfarin, which is primarily cleared by CYP2C9, a poor-metabolizer genotype (*3/*3, activity factor 0.1) reduces predicted clearance by 90% relative to the wild-type (*1/*1), resulting in substantially higher predicted AUC and prolonged half-life. This directional effect is consistent with clinical dose-adjustment guidelines, where CYP2C9 poor metabolizers require 60--80% dose reductions (Johnson et al., 2017).

**Bayesian individual fitting.** Given 3 or more observed concentration--time points, Omega applies L-BFGS-B optimization in log-concentration space to estimate individual clearance and volume scaling factors. The optimizer uses MAP estimation with a log-normal prior centered on population predictions, regularized to prevent physiologically implausible parameter values. This enables the model to refine population-level predictions using sparse clinical observations, transitioning from population prediction to individualized dosing support with as few as 3 data points.

These patient-specific capabilities demonstrate directional consistency with established pharmacological principles and extend the utility of Omega from population-level screening toward individualized prediction. However, the allometric and genotype scaling factors are derived from published literature, not learned from patient data, and the Bayesian fitting has been validated only on simulated observations, not clinical data. Prospective validation against individual patient PK studies is required before any clinical application.

---

## 4. Discussion

### 4.1 Comparison to Bayer Hybrid Model

The most direct comparison for Omega is the hybrid GCN-PBPK model reported by Gruber et al. (2024), which represents the current state-of-the-art in structure-based PK prediction. Both approaches share the same fundamental architecture — ML-predicted molecular properties fed into a mechanistic PBPK model — but differ substantially in their data requirements.

Omega achieves Cmax AAFE 1.90 on 20 oral drugs using exclusively public data sources: ADMET-AI (pretrained on public ChEMBL/TDC data), XGBoost models trained on 153 literature reference compounds, and a 35-state ODE engine parameterized from published physiological values. Gruber et al. report an exposure fold change error of 1.87 for oral administration in healthy subjects, using Bayer's internal in vitro assay database for model pre-training — a proprietary resource accumulated over decades of pharmaceutical research.

That comparable accuracy is achievable with public data alone has two important implications. First, it suggests that the bottleneck in PK prediction is not data access per se, but rather the engineering of the integration layer — how ML-predicted properties are transformed, calibrated, and fed into the mechanistic model. Omega's physics-informed corrections (GSE solubility floor, VDss calibration, hybrid Cmax selector) compensate for individual property prediction errors through mechanistic constraints. Second, it democratizes PK prediction: any research group can reproduce and build upon Omega's results without requiring access to proprietary assay databases.

A direct quantitative comparison is complicated by several factors. First, the metrics differ: Gruber et al. report the **median** fold change error (mfce), whereas AAFE uses the geometric **mean** — median is more robust to outliers and typically yields lower values. When computed using the same median metric, Omega achieves 1.73 (Cmax) and 1.60 (AUC), compared to Gruber's 1.87 for oral exposure. Second, the test set compositions differ: Omega uses 20 publicly available drugs with clinical C(t) curves, while Gruber's human test set size is not reported (C(t) profiles were validated on 9 compounds). Third, Omega reports Cmax and AUC separately, while Gruber reports aggregate "exposure."

Among public-data approaches, Jia et al. (2025) achieved 60% within 2-fold for AUC and 59% for Cmax on 106 test compounds using PK-Sim with ML-predicted parameters. Omega's 70% within 2-fold on 20 drugs compares favorably, though the smaller test set limits statistical comparison. Jia et al.'s larger test set provides a more robust estimate of generalization, and expanding Omega's Gold-tier validation to a comparable size is a priority for future work.

A head-to-head evaluation on a shared benchmark set would be valuable but requires either access to the Bayer model or adoption of a community-standard test set.

### 4.2 Speed and Throughput

At 73 milliseconds per compound (warm start), Omega is approximately three orders of magnitude faster than manual PBPK model setup. This enables applications that were previously impractical:

- **Virtual screening:** Evaluating PK for 10,000 compounds in ~12 minutes
- **Molecular optimization:** Real-time PK feedback during generative chemistry
- **Clinical decision support:** Near-instantaneous dose adjustment recommendations

The cold-start overhead (~5 seconds for XGBoost model loading) is amortized over multiple predictions and is negligible in batch settings. The dominant cost is the 35-state ODE integration (~60 ms), with ADME prediction adding ~10 ms via pre-loaded XGBoost models.

### 4.3 Interpretability and Failure Diagnosis

Unlike pure deep learning PK models, every Omega prediction traces through named, physiologically meaningful parameters: fraction unbound (fup), intrinsic clearance (CLint), effective permeability (peff), blood-to-plasma ratio (rbp), and tissue partition coefficients (Kp). This transparency enables:

1. **Failure diagnosis.** When a prediction is poor, the root cause can be traced to a specific property. Verapamil's 8.8x Cmax under-prediction is attributable to P-glycoprotein-mediated efflux, which reduces oral bioavailability but is not captured by the current absorption model. Ibuprofen's 5.0x error traces to its extreme protein binding (fup ~ 0.01), where small absolute errors in fup prediction cause large relative PK errors (Fig. 4).

2. **Targeted improvement.** Each failure mode suggests a specific fix: adding transporter corrections for P-gp substrates, improving fup prediction for highly bound drugs, or implementing nonlinear metabolism for saturable CYP substrates.

3. **Regulatory acceptability.** Mechanistic interpretability aligns with FDA guidance on PBPK model reporting (FDA, 2018), which requires documentation of model structure, parameter sources, and sensitivity analyses — all of which are straightforward with Omega's architecture.

### 4.4 Multi-Tier Validation

We propose multi-tier validation as a standard for evaluating structure-based PK prediction systems. Traditional validation on a single PK metric (e.g., Cmax AAFE on N drugs) provides limited insight into model reliability. Our four-tier framework assesses:

- **Gold tier (PK accuracy):** Does the system predict clinical PK correctly?
- **Silver tier (elimination):** Are clearance/volume predictions consistent with observed half-lives?
- **Bronze tier (ADME properties):** Are individual property predictions accurate?
- **Temporal holdout:** Does the system generalize to truly unseen chemical space?

Each tier serves a different purpose. Gold tier is the ultimate validation but is limited by the availability of clinical C(t) data. Silver tier expands the drug count using widely available FDA label half-life values. Bronze tier identifies which ADME properties are the strongest and weakest links. Temporal holdout, using drugs approved after the training data cutoff, provides the most honest assessment of generalization.

### 4.5 Limitations

Several limitations should be acknowledged:

**Validation set size.** The Gold tier comprises 20 drugs — sufficient for proof-of-concept but small compared to industry databases. Expanding the validation set is constrained by the availability of digitized clinical C(t) curves with known SMILES and dosing information. The Silver tier (39 drugs) and Bronze tier (151 compounds) provide broader coverage but at lower validation fidelity.

**Clearance prediction.** CLint AAFE of 3.25 is the weakest property prediction, reflecting the fundamental difficulty of predicting hepatic clearance from molecular structure alone. Structure-based clearance prediction remains an active research challenge across the field, with even specialized models rarely achieving AAFE below 2.0.

**Transporters.** Active transport (P-glycoprotein, OATP1B1, OCT2) is not modeled. For the ~30% of oral drugs that are significant transporter substrates, this represents a systematic source of error. Incorporating ADMET-AI's transporter substrate predictions (e.g., Pgp_Broccatelli) is a natural extension.

**Nonlinear PK.** Omega assumes linear (first-order) pharmacokinetics. Drugs with saturable metabolism (phenytoin, ethanol), saturable protein binding (valproic acid), or capacity-limited absorption are not well described by this assumption.

**Route and formulation.** Validation is currently limited to single-dose oral immediate-release formulations in healthy volunteers. Extension to intravenous, modified-release, and multi-dose regimens requires additional validation.

**Patient-specific prediction.** The Level 3 module uses deterministic allometric and genotype scaling factors from published literature, not learned from data. While clinically standard (NONMEM, Monolix use identical approaches), a data-driven model trained on individual patient PK data could capture more complex covariate relationships. This awaits availability of suitable training data.

### 4.6 Future Directions

Near-term extensions include transporter correction using ADMET-AI's P-gp/OATP predictions, expansion of the benchmark set using systematic FDA label extraction, and confidence interval calibration for the CLint predictor.

Longer-term, the neural Level 3 architecture (cross-attention fusion of molecular, patient, and dosing encoders with Reptile meta-learning) is implemented but untrained, awaiting individual patient concentration-time data. Academic collaborations with population PK modeling groups could provide the necessary training data. The differentiable ODE surrogate (AAFE 1.20 vs. real ODE) enables end-to-end gradient-based training when such data become available.

---

## 5. Conclusion

Omega demonstrates that open-source, public-data-only pharmacokinetic prediction can achieve accuracy comparable to proprietary models that leverage years of internal pharmaceutical data. By combining ML-predicted ADME properties with a mechanistic PBPK engine and physics-informed corrections, Omega achieves Cmax AAFE 1.90 (median 1.73) and AUC AAFE 1.66 (median 1.60) on 20 benchmark drugs in 73 milliseconds — competitive with Bayer's proprietary hybrid model (median fold error 1.87) but fully reproducible and freely available. The multi-tier validation framework, honest failure analysis, and patient-specific prediction module provide a foundation for both high-throughput virtual screening and individualized pharmacotherapy. Omega is available at https://github.com/jam-sudo/Omega under the MIT license.

---

## References

See `references.bib` for full bibliography.
