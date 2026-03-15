# 2. Methods

## 2.1 Architecture Overview

Omega is a hybrid neural-mechanistic pharmacokinetic prediction platform that
transforms a molecular structure, represented as a SMILES string, into a
complete plasma concentration-time profile without requiring any
experimentally measured drug-specific parameters. The pipeline comprises four
sequential stages: (i) molecular featurization via RDKit, (ii) ADME property
prediction from an ensemble of machine learning models, (iii) construction of
a parameterized Drug object with tissue partition coefficients, renal
clearance estimates, and in vitro-to-in vivo extrapolation (IVIVE) of hepatic
clearance, and (iv) numerical integration of a 35-state ordinary differential
equation (ODE) whole-body physiologically based pharmacokinetic (PBPK) model.

The design rationale for this hybrid architecture is that ADME property
prediction and PK profile generation present fundamentally different
challenges. ADME properties (fraction unbound in plasma, intrinsic clearance,
permeability) are molecular-level quantities amenable to structure-activity
modeling, whereas the concentration-time profile emerges from the interplay
of absorption, distribution, metabolism, and excretion processes governed by
known physiological equations. By coupling learned ADME predictors to a
mechanistic ODE solver, the system leverages the strengths of each paradigm:
data-driven generalization for molecular properties and first-principles
accuracy for pharmacokinetic dynamics.

The end-to-end prediction is exposed through the `OmegaPipeline` class, which
accepts a `SimulationRequest` specifying the SMILES string, dose (mg), route
of administration (oral, intravenous, or subcutaneous), simulation duration,
and optional patient covariates (body weight, CYP genotype, age). The output
is a `SimulationResult` containing the full concentration-time curve, derived
PK parameters ($C_{\max}$, $T_{\max}$, $\text{AUC}_{0-t}$, $t_{1/2}$), ADME
property estimates with uncertainty intervals, and a confidence classification.

## 2.2 ADME Prediction Layer

### 2.2.1 Ensemble Architecture

The ADME prediction layer employs an `EnsembleADMEPredictor` that combines
multiple backend models with a per-property selection strategy:

- **Fraction unbound in plasma ($f_u$):** Geometric mean of ADMET-AI and
  XGBoost predictions in log-space, computed as
  $f_u = \sqrt{f_u^{\text{ADMET-AI}} \cdot f_u^{\text{XGBoost}}}$. This
  ensemble gives equal weight to both predictors in the
  pharmacokinetically relevant log-scale, where a 10-fold difference between
  $f_u = 0.01$ and $f_u = 0.001$ is as consequential as the difference
  between $f_u = 0.1$ and $f_u = 1.0$.

- **Blood-to-plasma ratio (RBP):** XGBoost model trained on the 153-compound
  reference dataset (primary), with polynomial regression fallback. No public
  pretrained model for RBP exists, necessitating a custom model.

- **Lipophilicity ($\log P$), aqueous solubility ($\log S$), intrinsic
  clearance ($CL_{\text{int,3A4}}$), effective permeability ($P_{\text{eff}}$),
  hERG $\text{IC}_{50}$:** ADMET-AI (primary), polynomial ridge regression
  (fallback).

- **CYP2D6 intrinsic clearance ($CL_{\text{int,2D6}}$):** ADMET-AI
  categorical prediction with heuristic scaling.

- **Molecular weight (MW):** Computed directly from the SMILES string via
  RDKit (always available, no model required).

Overall confidence is defined as the minimum confidence level across all
property backends, following a conservative principle: if any single property
prediction is uncertain, the entire prediction is flagged accordingly.

### 2.2.2 XGBoost Models

Four XGBoost gradient-boosted tree models were trained for properties
where either no pretrained model was publicly available (RBP) or where
domain-specific calibration was required ($f_u$, $CL_{\text{int}}$,
$V_{d,\text{ss}}$).

**Molecular representation.** All XGBoost models use a concatenation of
2048-bit Morgan circular fingerprints (radius 2) generated via RDKit and
9 physicochemical descriptors: $\log P$ (Crippen), topological polar surface
area (TPSA), molecular weight, hydrogen bond acceptor count, hydrogen bond
donor count, rotatable bond count, ring count, fraction $\text{sp}^3$
carbons, and molar refractivity.

**Training data.** The $f_u$ model was trained on the union of the
Therapeutic Data Commons (TDC) PPBR\_AZ dataset (1,614 compounds) and the
153-compound in-house reference set. The $CL_{\text{int}}$ model used the TDC
Clearance\_Hepatocyte\_AZ dataset (1,213 compounds). The $V_{d,\text{ss}}$
model used the TDC VDss\_Lombardo dataset (1,130 compounds). The RBP model
was trained exclusively on the 153-compound reference set due to the absence
of a suitable public dataset.

**Prediction in log-space.** The $f_u$, $CL_{\text{int}}$, and
$V_{d,\text{ss}}$ models predict in $\log_{10}$-transformed space (e.g.,
$\log_{10}(f_u)$) and exponentiate at inference time. This ensures equal
loss weighting across orders of magnitude and prevents the model from
ignoring highly bound compounds ($f_u < 0.01$), which are disproportionately
important in clinical pharmacology.

### 2.2.3 Polynomial Ridge Regression Fallback

A polynomial ridge regression model serves as the final fallback for all
ADME properties when both ADMET-AI and XGBoost backends are unavailable.
This model provides biased but bounded predictions from physicochemical
descriptors, ensuring the pipeline never fails to produce an estimate.

### 2.2.4 Solubility Floor: General Solubility Equation

Predicted aqueous solubility is subject to a lower bound derived from the
General Solubility Equation (GSE) of Yalkowsky and Valvani (1980):

$$\log S \geq 0.5 - \log P$$

This constraint prevents the polynomial predictor from catastrophically
under-predicting solubility for lipophilic drugs. Without this floor, drugs
such as fluoxetine ($\log P = 4.4$) receive dose numbers exceeding 70 and
predicted fraction absorbed below 2%, resulting in 10--15-fold
under-prediction of $C_{\max}$.

### 2.2.5 Conformal Prediction Intervals

Prediction uncertainty is quantified through conformal prediction intervals
calibrated to achieve approximately 90% empirical coverage. For $f_u$, the
XGBoost model provides native conformal intervals from cross-validation
residuals. For other properties, intervals are initialized at $\pm 50\%$ of
the point estimate and then adjusted by empirically determined calibration
factors computed on a 152-compound holdout from `adme_reference.csv`:

| Property | Calibration Factor | Raw Coverage | Calibrated Coverage |
|---|---|---|---|
| $f_u$ | 1.04 | 86.2% | ~90% |
| $CL_{\text{int,3A4}}$ | 50.0 | 11.2% | ~75% |
| $P_{\text{eff}}$ | 2.94 | 57.9% | ~90% |
| RBP | 3.13 | 63.8% | ~90% |

The large calibration factor for $CL_{\text{int,3A4}}$ reflects the inherent
difficulty of intrinsic clearance prediction from molecular structure alone;
even after aggressive interval widening, only approximately 75% coverage is
achievable with current models.

## 2.3 PBPK Model

### 2.3.1 Model Structure

The whole-body PBPK model comprises 35 ordinary differential equations
representing 15 organs, an 8-segment gastrointestinal (GI) tract, portal
venous and systemic blood pools, and mass-balance tracking compartments for
hepatic metabolism, gut-wall metabolism, renal excretion, and fecal
excretion.

**Perfusion-limited organs** (11 compartments: lung, brain, heart, kidney,
liver, spleen, gut wall, pancreas, thymus, reproductive organs, and a
residual "rest" compartment) are modeled with flow-limited distribution:

$$\frac{dA_i}{dt} = Q_i \cdot C_{\text{art}} - Q_i \cdot \frac{A_i \cdot R_{B:P}}{V_i \cdot K_{p,i}}$$

where $A_i$ is the drug amount in organ $i$, $Q_i$ is the organ blood flow,
$C_{\text{art}}$ is the arterial blood concentration, $V_i$ is the organ
volume, $K_{p,i}$ is the tissue-to-plasma partition coefficient, and
$R_{B:P}$ is the blood-to-plasma ratio.

**Permeability-limited organs** (4 organs, 8 compartments: adipose, muscle,
bone, and skin, each with vascular and extravascular sub-compartments)
incorporate a permeability-surface area product ($PS$) governing the rate of
drug transfer between vascular and tissue spaces:

$$\frac{dA_{i,\text{vasc}}}{dt} = Q_i \cdot C_{\text{art}} - Q_i \cdot C_{i,\text{vasc}} - PS_i \cdot (C_{i,\text{vasc}} - C_{i,\text{extra}}/K_{p,i})$$

$$\frac{dA_{i,\text{extra}}}{dt} = PS_i \cdot (C_{i,\text{vasc}} - C_{i,\text{extra}}/K_{p,i})$$

**Blood pools.** Venous blood collects outflow from all non-pulmonary organs;
arterial blood receives outflow from the lung. Portal organs (spleen, gut
wall, pancreas) drain into the portal vein, which feeds the liver. Cardiac
output is set to 390 L/h for a 70-kg reference adult, scaled linearly with
body weight.

Organ volumes and blood flow fractions are derived from ICRP Publication 89
reference values for a 70-kg adult male, scaled linearly with body weight
($V_i = V_{i,\text{ref}} \times \text{BW}/70$).

### 2.3.2 ACAT Gastrointestinal Absorption

Oral absorption is modeled using an Advanced Compartmental Absorption and
Transit (ACAT) framework with 8 sequential segments: stomach, duodenum,
jejunum (2 segments), ileum (3 segments), and colon. Each segment $j$ has a
lumenal compartment with first-order transit to the next segment and
permeability-dependent absorption into the portal vein:

$$\frac{dA_j}{dt} = k_{t,j-1} \cdot A_{j-1} - k_{t,j} \cdot A_j - k_{a,j} \cdot A_j$$

where $k_{t,j} = 1/\tau_j$ is the transit rate constant (with transit times
$\tau$ of 0.25, 0.26, 0.475, 0.475, 0.68, 0.68, 0.68, and 13.5 h for the
8 segments, respectively) and $k_{a,j}$ is the segment-specific absorption
rate constant, proportional to the drug's effective permeability
($P_{\text{eff}}$) and modulated by segment-dependent absorption fractions
(0.0, 1.0, 1.0, 1.0, 0.8, 0.6, 0.3, 0.05 from stomach to colon).

Drug dissolution in the stomach is governed by the dose number
($D_0 = \text{dose} / (250 \text{ mL} \times S)$, where $S$ is aqueous
solubility in mg/mL). When $D_0 > 1$, the fraction of dose in solution is
limited and absorption is solubility-rate-limited.

### 2.3.3 Tissue Partitioning

Tissue-to-plasma partition coefficients ($K_p$) are estimated using the
mechanistic method of Rodgers and Rowland (2006), which computes $K_p$ from
tissue composition (fractional neutral lipid, phospholipid, and water content)
and drug physicochemical properties ($\log P$, $f_u$, ionization state).
Tissue composition data for all 15 organs are taken from Tables 1--2 of
Rodgers and Rowland (2006).

For ionizable compounds, the Berezhkovskiy (2004) correction is applied,
which accounts for the effect of ionization on tissue binding. Compound type
(neutral, acid, base, zwitterion) and approximate $\text{p}K_a$ are inferred
from SMARTS substructure matching on the input SMILES (amines: $\text{p}K_a
\approx 9.0$; carboxylic acids: $\text{p}K_a \approx 4.0$).

For $K_p$ calculations, Crippen $\log P$ from RDKit is used rather than
the ML-predicted $\log P$, as it provides more reliable partition coefficient
estimates (e.g., fluoxetine: ML predicts 2.09, RDKit gives 4.44, literature
value 4.05).

### 2.3.4 Hepatic Clearance

Hepatic elimination follows the well-stirred model:

$$CL_h = \frac{Q_h \cdot f_u \cdot CL_{\text{int}}}{Q_h + f_u \cdot CL_{\text{int}}}$$

where $Q_h = 90$ L/h is the hepatic blood flow (hepatic artery + portal
vein), $f_u$ is the fraction unbound in plasma, and $CL_{\text{int}}$ is the
intrinsic clearance scaled to whole-liver units (L/h).

**IVIVE.** In vitro intrinsic clearance from the XGBoost hepatocyte model
($CL_{\text{int,hep}}$ in $\mu$L/min/$10^6$ cells) is scaled to in vivo
hepatic clearance using an empirically calibrated allometric correction:

$$CL_{h,\text{target}} = \min\left(\alpha \cdot CL_{\text{int,hep}}^{\beta},\ 0.95 \cdot Q_h\right)$$

with $\alpha = 0.3$ and $\beta = 0.9$, calibrated against published clinical
clearance values for ibuprofen, acetaminophen, theophylline, diclofenac, and
omeprazole (4/5 within 2-fold). The sub-linear exponent $\beta < 1$ accounts
for the systematic over-prediction of standard IVIVE at high intrinsic
clearance values (Hallifax and Houston, 2009). The target $CL_h$ is then
back-calculated to the $CL_{\text{int}}$ value that produces that clearance
in the well-stirred model given the predicted $f_u$:

$$CL_{\text{int}} = \frac{CL_{h,\text{target}} \cdot Q_h}{f_u \cdot (Q_h - CL_{h,\text{target}})}$$

This pre-inversion compensates for both IVIVE scaling bias and $f_u$
prediction errors.

**Gut wall first-pass.** The fraction escaping gut-wall metabolism ($F_g$)
is computed using the $Q_{\text{gut}}$ model (Yang et al., 2007):

$$F_g = \frac{Q_{\text{gut}}}{Q_{\text{gut}} + f_u \cdot CL_{\text{int,gut}}}$$

### 2.3.5 Renal Clearance

Renal clearance is estimated from physicochemical properties rather than
predicted by a dedicated ML model. Glomerular filtration rate (GFR) is set to
7.2 L/h (120 mL/min) for a 70-kg adult. The estimation framework
incorporates:

- **Glomerular filtration:** $CL_{\text{filt}} = \text{GFR} \times f_u$,
  with a molecular weight penalty for compounds exceeding 500 Da.
- **Tubular reabsorption:** Lipophilic compounds ($\log P \geq 2.5$) are
  assumed to undergo complete tubular reabsorption ($CL_r = 0$).
- **Active secretion:** Hydrophilic compounds ($\log P < -0.5$) with high
  topological polar surface area (TPSA > 74 \AA$^2$) are assigned active
  tubular secretion via OCT2/OAT/MATE transporters, with secretion factor
  $\min(3.0,\ 10^{-\log P})$.
- **Basic amines:** Small basic amines ($\log P < 1.5$, MW < 300) with
  primary or secondary amine groups receive OCT2-mediated secretion
  (2-fold enhancement over filtration).

Renal clearance is capped at 30 L/h to prevent non-physiological values.

### 2.3.6 ODE Integration

The system of 35 coupled ODEs is integrated using the LSODA method
(Hindmarsh, 1983; Petzold, 1983) as implemented in SciPy's `solve_ivp`,
with relative tolerance $10^{-8}$ and absolute tolerance $10^{-10}$. LSODA
automatically switches between non-stiff (Adams) and stiff (BDF) methods
based on the local stiffness of the system, which is critical for PBPK
models where rapid absorption ($k_a \gg k_e$) creates stiff initial
conditions that relax during the elimination phase.

## 2.4 Physics-Informed Corrections

Three post-hoc corrections are applied to the raw ODE output to compensate
for known systematic biases in the mechanistic model.

### 2.4.1 VDss Divergence Check

The $K_p$-based volume of distribution ($V_{d,\text{ss,Berez}}$), computed by
summing Berezhkovskiy-corrected partition coefficients across all organs, is
compared against the XGBoost $V_{d,\text{ss}}$ prediction (trained on the
TDC VDss\_Lombardo dataset of 1,130 compounds). When $V_{d,\text{ss,Berez}} >
2 \times V_{d,\text{ss,XGB}}$, the Berezhkovskiy estimate is considered
unreliable (typically due to propagated $\log P$ errors into tissue
partitioning), and the XGBoost value is substituted for the analytical
$C_{\max}$ calculation. For the analytical half-life estimate, the geometric
mean $V_d = \sqrt{V_{d,\text{Berez}} \times V_{d,\text{XGB}}}$ is used
instead, as full replacement is too aggressive for high-$V_d$ drugs.

### 2.4.2 Hybrid Cmax Selector

The PBPK ODE's perfusion-limited distribution systematically distorts
$C_{\max}$: the plasma concentration exhibits a transient spike before tissue
equilibrium is reached, causing over-prediction for drugs with slow tissue
uptake. For oral drugs with renal clearance below 5 L/h, $C_{\max}$ from the
ODE is blended with the analytical one-compartment model prediction:

$$C_{\max,\text{analytical}} = \frac{F \cdot D}{V_d} \cdot \frac{k_a}{k_a - k_e} \cdot \left(e^{-k_e \cdot t_{\max}} - e^{-k_a \cdot t_{\max}}\right)$$

where $F$ is the predicted oral bioavailability, $D$ is the dose,
$k_a = P_{\text{eff}} \times 10^4 \times 1.0$ (capped to $[0.3, 5.0]$
h$^{-1}$), and $k_e = CL_{\text{total}} / V_d$.

When the VDss divergence check triggers ($V_{d,\text{Berez}} / V_{d,\text{XGB}}
> 2$), the analytical $C_{\max}$ is used directly. Otherwise, the geometric
mean of the ODE and analytical values is reported:
$C_{\max} = \sqrt{C_{\max,\text{ODE}} \times C_{\max,\text{analytical}}}$.

### 2.4.3 Hybrid Half-Life Selector

Terminal half-life is estimated from two independent approaches and selected
via heuristic rules:

1. **Curve-fit:** Log-linear regression of the post-$C_{\max}$
   concentration-time profile from the PBPK simulation (points above 0.1% of
   $C_{\max}$), yielding $t_{1/2} = \ln 2 / (-\text{slope})$.

2. **Analytical:** $t_{1/2} = 0.693 \times V_d / CL_{\text{total}}$, using
   the predicted volume of distribution and total clearance.

The selection rules are: (a) if the analytical $t_{1/2}$ is shorter than the
curve-fit value and exceeds 1 hour (and the ODE renal clearance is below
20 L/h), the analytical value is preferred, as the curve-fit is inflated by
redistribution dynamics; (b) if hepatic clearance is below 5 L/h, $\log P >
2.0$, and the analytical $t_{1/2}$ exceeds 20 hours, the analytical value is
preferred, as the 24-hour default simulation duration is insufficient to
capture the true terminal slope; (c) otherwise, the curve-fit value is used.

### 2.4.4 Bioavailability Prediction

Oral bioavailability is decomposed as $F = f_a \times F_g \times F_h$, where
$f_a$ is the fraction absorbed (from ACAT dissolution and permeability),
$F_g$ is the fraction escaping gut-wall metabolism, and $F_h = 1 - CL_h /
Q_h$ is the fraction escaping hepatic first-pass extraction. Each component
is computed mechanistically from the predicted ADME properties.

## 2.5 Patient-Specific Prediction

### 2.5.1 Allometric Scaling

Population PK parameters are scaled to individual patients using standard
allometric equations referenced to a 70-kg adult:

$$CL_{\text{individual}} = CL_{\text{pop}} \times \left(\frac{W}{70}\right)^{0.75}$$

$$V_{d,\text{individual}} = V_{d,\text{pop}} \times \frac{W}{70}$$

where $W$ is the patient body weight in kg. The exponent of 0.75 for
clearance reflects the well-established allometric relationship between
metabolic rate and body size across species (West et al., 1997). Volume of
distribution scales linearly with body weight, consistent with the
assumption that tissue volumes are proportional to body mass.

### 2.5.2 CYP Genotype Factors

Pharmacogenomic variability in drug metabolism is incorporated through
enzyme-specific activity scaling factors applied multiplicatively to hepatic
clearance. The following factors are implemented, defined relative to the
extensive metabolizer (EM) phenotype:

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

When multiple CYP enzymes are specified, each factor is applied
independently to the clearance, weighted by the drug's fraction metabolized
($f_m$) by each enzyme. Unknown enzyme-phenotype combinations default to an
activity factor of 1.0 (no effect).

### 2.5.3 Bayesian Individual Estimation

For patients with 1--5 observed plasma concentration measurements, individual
PK parameters are estimated by fitting clearance and volume scaling factors
($\eta_{CL}$, $\eta_V$) to the observed data using a maximum a posteriori
(MAP) approach. The objective function minimizes the mean squared error in
log-concentration space:

$$\mathcal{L}(\eta_{CL}, \eta_V) = \frac{1}{N} \sum_{i=1}^{N} \left(\ln C_{\text{pred}}(t_i; \eta_{CL}, \eta_V) - \ln C_{\text{obs}}(t_i)\right)^2$$

where $C_{\text{pred}}$ is computed from the analytical one-compartment oral
model:

$$C(t) = \frac{F \cdot D}{V_d \cdot \eta_V} \cdot \frac{k_a}{k_a - k_e'} \cdot \left(e^{-k_e' t} - e^{-k_a t}\right)$$

with $k_e' = (CL \cdot \eta_{CL}) / (V_d \cdot \eta_V)$.

Optimization is performed via L-BFGS-B (Byrd et al., 1995) with bounds
$\eta_{CL}, \eta_V \in [0.05, 20.0]$, initialized at $\eta_{CL} =
\eta_V = 1.0$ (population estimate). The use of log-concentration space
ensures that the fitting procedure assigns equal weight to high and low
concentrations, which is critical when observations span the absorption
peak and the elimination tail.

## 2.6 Validation Framework

### 2.6.1 Tiered Validation Design

Model performance is assessed through a four-tier validation framework with
increasing stringency and decreasing data availability:

**Bronze tier** (ADME property validation). Predicted ADME properties ($f_u$,
$\log P$, RBP, $CL_{\text{int,3A4}}$, $P_{\text{eff}}$) are compared against
literature reference values from 153 compounds curated in
`adme_reference.csv`. Each compound has a canonical SMILES string and
experimentally measured values with defined units ($CL_{\text{int}}$ in
$\mu$L/min/pmol CYP, $P_{\text{eff}}$ in cm/s, $f_u$ as fraction 0--1, RBP
as ratio). Per-property AAFE and percentage within 2-fold are reported.

**Silver tier** (half-life validation). Predicted terminal half-lives are
compared against values extracted from FDA-approved drug labels obtained via
the OpenFDA API. This tier validates the integrated clearance and
distribution predictions through their aggregate effect on half-life.

**Gold tier** (concentration-time curve validation). Predicted plasma
concentration-time profiles are compared against published clinical PK data
for 25 drugs, each with a defined dose and route of administration.
Benchmark datasets (e.g., `caffeine_oral_100mg.csv`) contain digitized
concentration-time points with standard deviations. $C_{\max}$, $\text{AUC}$,
and $t_{1/2}$ fold errors are computed for each drug.

**Temporal holdout** (prospective validation). Five drugs approved by the FDA
after the ADMET-AI training data cutoff (approximately 2022) are used as a
fully prospective test set: adagrasib, futibatinib, capivasertib,
elacestrant, and pirtobrutinib. These compounds were not present in any
training data for any model in the pipeline, providing an unbiased estimate
of generalization performance.

### 2.6.2 Performance Metrics

Two primary metrics are used throughout all validation tiers:

**Average Absolute Fold Error (AAFE).** The geometric mean of fold errors
across $n$ compounds:

$$\text{AAFE} = 10^{\frac{1}{n} \sum_{i=1}^{n} \log_{10} \text{FE}_i}$$

where $\text{FE}_i = \max(y_i^{\text{pred}} / y_i^{\text{obs}},\
y_i^{\text{obs}} / y_i^{\text{pred}})$. AAFE = 1.0 indicates perfect
prediction; AAFE = 2.0 indicates that predictions are, on average, 2-fold
from observed values. AAFE is preferred over arithmetic mean fold error
because it is symmetric (over- and under-prediction are penalized equally)
and robust to outliers on the multiplicative scale.

**Percentage within 2-fold (%2-fold).** The fraction of predictions for which
$\text{FE}_i \leq 2.0$:

$$\%\text{2-fold} = \frac{100}{n} \sum_{i=1}^{n} \mathbb{1}[\text{FE}_i \leq 2.0]$$

This metric corresponds to the standard regulatory acceptance criterion for
PBPK model qualification (EMA, 2018; FDA, 2018), where at least 50% of
predictions within 2-fold of observed data is generally considered acceptable
for a credible PBPK model.

### 2.6.3 Confidence Calibration

The confidence classification system (low/medium/high) is validated through a
scaffold-split holdout procedure. The 153 reference compounds are split by
Murcko scaffold (Bemis and Murcko, 1996) with 20% held out for calibration.
Confidence monotonicity is verified: AAFE(high confidence) $\leq$ AAFE(medium
confidence) $\leq$ AAFE(low confidence), ensuring that the confidence labels
are informative and that users can trust that high-confidence predictions are
indeed more accurate.

### 2.6.4 Regression Testing

All validation benchmarks are automated and executed as part of the
continuous integration pipeline. The gold-tier exit criteria require
$C_{\max}$ AAFE $< 3.0$, AUC AAFE $< 3.0$, and $\geq 70\%$ of at least
20 drugs within 2-fold of observed values.
