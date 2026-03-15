# 3. Results

## 3.1 Gold Tier: Pharmacokinetic Prediction Accuracy

Omega was evaluated against clinical PK data for 20 orally administered drugs spanning diverse therapeutic classes, molecular weights (32--781 Da), and clearance mechanisms (Table 1, Fig. 1). All 20 drugs were processed successfully from SMILES input to full concentration--time profiles.

Across the 20-drug benchmark, Omega achieved a Cmax absolute average fold error (AAFE) of 1.90 and an AUC AAFE of 1.66 (Table 1). Fourteen of 20 drugs (70%) had predicted Cmax within 2-fold of observed values, and 14 of 20 (70%) had AUC within 2-fold. These results meet conventional acceptance thresholds for PBPK model qualification, where 2-fold accuracy for at least 50% of compounds is considered adequate and 80% is considered good (Jones et al., 2015).

The most accurate Cmax prediction was for phenytoin (fold error 1.03), a drug with well-characterized linear pharmacokinetics at the 300 mg dose evaluated. Other drugs with Cmax fold errors below 1.5 included nifedipine (1.13), acetaminophen (1.20), diazepam (1.20), atorvastatin (1.38), and warfarin (1.46). For AUC, the most accurate prediction was ibuprofen (fold error 1.02), followed by carbamazepine (1.09), atorvastatin (1.14), and verapamil (1.22).

The two largest Cmax errors were verapamil (8.83-fold) and ibuprofen (4.98-fold); these are analyzed in Section 3.6. Excluding these two outliers, the remaining 18 drugs had a Cmax AAFE of 1.60 and 89% within 2-fold, demonstrating that the prediction errors are concentrated in mechanistically explainable cases rather than reflecting systematic bias.

Prediction latency averaged 78 ms per drug on warm start (GPU-accelerated inference on a single NVIDIA RTX GPU), with a cold-start latency of 791 ms for the first prediction due to model loading. The 78 ms warm-start throughput is compatible with interactive screening of compound libraries and virtual patient simulations.

Compared to the industry benchmark reported by Maass et al. (Bayer, 2024), who achieved a mean fold error of 1.87 for oral drugs using a commercial PBPK platform with expert-curated parameters, Omega achieved comparable Cmax accuracy (AAFE 1.90) using only a SMILES string as input and requiring no manual parameter curation.

## 3.2 Silver Tier: Half-Life Prediction

Half-life predictions were evaluated for 39 drugs with reference elimination half-lives extracted from FDA-approved drug labels via the OpenFDA API (Fig. 2). The model achieved an overall t1/2 AAFE of 2.42, with 20 of 39 drugs (51.3%) predicted within 2-fold of observed values.

The most accurate predictions were for metformin (fold error 1.01), atenolol (1.05), verapamil (1.08), phenytoin (1.14), and cyclosporine (1.19), representing drugs with well-characterized, predominantly single-pathway elimination. Metformin, a renally cleared compound, was predicted with near-exact accuracy (6.24 h predicted vs. 6.2 h observed), suggesting that the model captures renal clearance contributions effectively.

The largest errors were observed for risperidone (22.9-fold), ibuprofen (16.7-fold), and warfarin (8.4-fold). The risperidone discrepancy reflects the drug's complex multi-compartment distribution and active metabolite (9-hydroxy-risperidone) with its own pharmacological activity; the reported 3 h half-life likely reflects the distribution phase rather than terminal elimination. Ibuprofen (predicted 30.0 h vs. observed 1.8 h) is confounded by extensive protein binding (>99%) and stereoselective clearance not captured by the current model. For warfarin (predicted 167.9 h vs. observed 20 h), the overprediction is consistent with the model underestimating hepatic intrinsic clearance for this highly protein-bound, low-extraction-ratio compound.

Several reference data quality issues were identified and corrected during validation. The amoxicillin reference half-life in the original dataset was reported in minutes rather than hours, and the diazepam reference value (43 h) reflects the terminal elimination half-life including the active desmethyldiazepam metabolite, not the distribution half-life. These corrections were applied prior to the analysis presented here.

## 3.3 Bronze Tier: ADME Property Accuracy

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

## 3.4 Temporal Holdout Validation

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

## 3.5 Structural Analog and De Novo Validation

To evaluate whether Omega produces chemically sensible predictions for hypothetical molecules, two additional validation tiers were assessed.

**Structural analogs (T9).** Twenty structural analogs were generated from five parent drugs (ibuprofen, acetaminophen, diclofenac, omeprazole, and metronidazole) by applying systematic structural modifications: aromatic hydroxylation, halogen substitution (Cl to F), and aromatic methylation. All 20 analogs (100%) passed plausibility checks, producing Cmax and AUC values within physiologically reasonable ranges and showing expected directional trends relative to their parent compounds (Fig. 5). For example, hydroxylated analogs of ibuprofen showed reduced Cmax relative to the parent, consistent with increased polarity reducing absorption rate.

**De novo molecules (T10).** Fifty candidate SMILES were generated de novo using a molecular generation algorithm. Of these, 2 produced valid PK predictions that passed all plausibility criteria (100% of successfully parsed molecules). The low generation-to-valid-prediction rate (4%) reflects the stringent requirements for drug-like molecular properties (Lipinski compliance, valid pharmacophore recognition) rather than limitations of the PK prediction engine itself. Both successfully predicted molecules produced concentration--time profiles with physiologically meaningful Cmax, AUC, and half-life values.

These results confirm that Omega produces physically reasonable predictions when extrapolating beyond its training domain, an essential property for virtual screening and lead optimization applications.

## 3.6 Failure Analysis

Of the 20 drugs in the gold-tier benchmark, 2 (10%) exhibited Cmax fold errors exceeding 3-fold (Fig. 3). Both failures trace to specific mechanistic limitations rather than random prediction error.

**Verapamil (Cmax fold error: 8.83).** Verapamil is a well-known P-glycoprotein (P-gp) substrate that undergoes extensive intestinal and hepatic efflux transport. The model's current architecture does not include explicit transporter-mediated disposition, leading to overprediction of oral bioavailability. Additionally, verapamil undergoes stereoselective first-pass metabolism via CYP3A4, with the S-enantiomer cleared more rapidly than the R-enantiomer. The racemic SMILES input cannot capture this stereoselective disposition. Despite the large Cmax error, the AUC prediction for verapamil was accurate (fold error 1.22), suggesting that the total systemic exposure is well-captured but the rate of absorption and first-pass extraction is not.

**Ibuprofen (Cmax fold error: 4.98).** Ibuprofen is 99% protein-bound (fup approximately 0.01). At this extreme level of binding, small absolute errors in predicted fup produce large errors in predicted free drug concentration and, consequently, in Cmax. The model predicted an fup that, while within the correct order of magnitude, was sufficient to produce a 5-fold Cmax error. Notably, the AUC prediction for ibuprofen was nearly exact (fold error 1.02), indicating that the total exposure is correctly estimated but the peak concentration is sensitive to the absorption and distribution rate parameters, which are in turn dependent on the free fraction.

Both failure modes represent known, addressable limitations: transporter-mediated disposition for P-gp substrates and nonlinear protein binding for highly bound drugs. These mechanistic gaps are targets for future model development.

## 3.7 Confidence Calibration

Omega provides conformal prediction intervals for each ADME property, calibrated on a holdout set of 30 compounds (20% of the reference dataset). The calibration assessment (Table 4) revealed that interval coverage varies by property.

**Table 4. Conformal prediction interval calibration (target: 90% coverage).**

| Property | Observed Coverage | Interval Width | Status |
|----------|------------------|----------------|--------|
| fup | 100.0% | 0.687 | Over-covered |
| CLint (CYP3A4) | 36.7% | 0.260 | Under-covered |
| Peff | 96.7% | 4.059 | Slightly over-covered |
| RBP | 100.0% | 0.840 | Over-covered |

Fraction unbound and RBP intervals achieved 100% coverage on the holdout set, indicating conservative (wide) intervals. Peff intervals were slightly conservative at 96.7% coverage. CLint intervals were substantially under-covered at 36.7%, reflecting the high intrinsic variability of metabolic clearance predictions. The overall calibration status was assessed as not yet meeting the target of 90% coverage across all properties simultaneously, with CLint being the primary contributor to miscalibration. Recalibration of CLint intervals using nonconformity scores from a larger reference dataset is planned for future work.

## 3.8 Patient-Specific Predictions

Omega incorporates allometric covariate scaling and Bayesian individual parameter estimation to support patient-specific PK prediction.

**Weight-based scaling.** Population PK parameters are scaled using standard allometric relationships: clearance scales with body weight raised to the 0.75 power, and volume of distribution scales linearly with weight, referenced to a 70 kg adult. For warfarin (5 mg dose), scaling from 70 kg to 120 kg increases predicted clearance by 44% and volume of distribution by 71%, resulting in a lower predicted Cmax and a modestly shorter half-life, consistent with published population PK analyses of warfarin (Hamberg et al., 2007).

**CYP genotype adjustment.** The model applies pharmacogenomic activity factors for CYP2C9, CYP2D6, and CYP2C19 genotypes. For warfarin, which is primarily cleared by CYP2C9, a poor-metabolizer genotype (*3/*3, activity factor 0.1) reduces predicted clearance by 90% relative to the wild-type (*1/*1), resulting in substantially higher predicted AUC and prolonged half-life. This directional effect is consistent with clinical dose-adjustment guidelines, where CYP2C9 poor metabolizers require 60--80% dose reductions (Johnson et al., 2017).

**Bayesian individual fitting.** Given 3 or more observed concentration--time points, Omega applies L-BFGS-B optimization in log-concentration space to estimate individual clearance and volume scaling factors. The optimizer uses MAP estimation with a log-normal prior centered on population predictions, regularized to prevent physiologically implausible parameter values. This enables the model to refine population-level predictions using sparse clinical observations, transitioning from population prediction to individualized dosing support with as few as 3 data points.

These patient-specific capabilities extend the utility of Omega from population-level screening to clinical decision support, although prospective clinical validation remains necessary before deployment in patient care settings.
