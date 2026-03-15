# Supplementary Materials

## Table S1: Gold Tier -- Per-Drug PK Results (20 drugs)

Full concentration-time profile validation against digitized clinical PK curves.
Aggregate: Cmax AAFE = 1.90, AUC AAFE = 1.66, 70% within 2-fold for both endpoints.

| Drug | Dose (mg) | Pred Cmax (mg/L) | Obs Cmax (mg/L) | FE Cmax | Pred AUC (mg-h/L) | Obs AUC (mg-h/L) | FE AUC | Latency (ms) |
|------|-----------|-------------------|------------------|---------|---------------------|-------------------|--------|--------------|
| Acetaminophen | 1000 | 9.15 | 10.96 | 1.20 | 21.3 | 49.5 | 2.32 | 89.9 |
| Amoxicillin | 500 | 5.50 | 9.52 | 1.73 | 33.9 | 42.1 | 1.24 | 85.5 |
| Atorvastatin | 40 | 0.00850 | 0.0117 | 1.38 | 0.200 | 0.176 | 1.14 | 95.8 |
| Caffeine | 100 | 1.36 | 2.04 | 1.50 | 9.39 | 12.5 | 1.33 | 790.8 |
| Carbamazepine | 200 | 0.595 | 1.36 | 2.29 | 30.3 | 27.7 | 1.09 | 86.5 |
| D-Amphetamine | 20 | 0.0962 | 0.0504 | 1.91 | 1.19 | 0.717 | 1.66 | 67.5 |
| Diazepam | 10 | 0.146 | 0.121 | 1.20 | 3.31 | 2.44 | 1.36 | 64.3 |
| Digoxin | 0.5 | 0.00191 | 0.000694 | 2.75 | 0.00894 | 0.0136 | 1.52 | 60.5 |
| Fluoxetine | 20 | 0.0131 | 0.00629 | 2.08 | 0.325 | 0.131 | 2.48 | 76.6 |
| Ibuprofen | 400 | 3.82 | 19.0 | 4.98 | 96.4 | 94.4 | 1.02 | 92.9 |
| Methanol | 100 | 1.02 | 1.99 | 1.95 | 6.36 | 7.87 | 1.24 | 74.1 |
| Metoprolol | 100 | 0.150 | 0.0663 | 2.27 | 0.516 | 0.426 | 1.21 | 67.5 |
| Midazolam | 2 | 0.0108 | 0.00577 | 1.87 | 0.0812 | 0.0210 | 3.86 | 70.2 |
| Nifedipine | 10 | 0.0845 | 0.0751 | 1.13 | 0.464 | 0.261 | 1.78 | 78.8 |
| Omeprazole | 20 | 0.305 | 0.595 | 1.95 | 0.473 | 1.47 | 3.10 | 68.4 |
| Phenytoin | 300 | 5.16 | 5.29 | 1.03 | 45.3 | 97.6 | 2.15 | 88.1 |
| Propranolol | 80 | 0.129 | 0.0822 | 1.56 | 0.306 | 0.579 | 1.89 | 82.2 |
| Theophylline | 300 | 4.52 | 7.23 | 1.60 | 31.7 | 83.2 | 2.62 | 87.5 |
| Verapamil | 80 | 0.00630 | 0.0556 | 8.83 | 0.444 | 0.540 | 1.22 | 77.0 |
| Warfarin | 5 | 0.112 | 0.162 | 1.46 | 4.89 | 3.16 | 1.54 | 76.3 |

FE = fold error (max of pred/obs, obs/pred). Latency = wall-clock time for SMILES-to-PK prediction.
Median latency: 78.8 ms (excluding caffeine first-call warm-up of 791 ms).

---

## Table S2: Silver Tier -- Per-Drug Half-Life Results (39 drugs)

Half-life validation against OpenFDA-extracted PK parameters.
Aggregate: AAFE = 2.42, 51.3% within 2-fold (20/39 drugs).

| Drug | Pred t1/2 (h) | Obs t1/2 (h) | Fold Error | Within 2-fold |
|------|---------------|--------------|------------|---------------|
| Acetaminophen | 3.63 | 2.9 | 1.25 | Yes |
| Alprazolam | 39.3 | 11.2 | 3.51 | No |
| Amlodipine | 11.2 | 30.0 | 2.68 | No |
| Amoxicillin | 1.94 | 1.02 | 1.90 | Yes |
| Atenolol | 6.33 | 6.0 | 1.05 | Yes |
| Atorvastatin | 4.00 | 14.0 | 3.50 | No |
| Carbamazepine | 70.7 | 25.0 | 2.83 | No |
| Ciprofloxacin | 6.25 | 4.0 | 1.56 | Yes |
| Clarithromycin | 4.49 | 3.0 | 1.50 | Yes |
| Cyclosporine | 7.04 | 8.4 | 1.19 | Yes |
| Diazepam | 83.6 | 43.0 | 1.94 | Yes |
| Digoxin | 4.07 | 1.5 | 2.71 | No |
| Ezetimibe | 13.0 | 22.0 | 1.69 | Yes |
| Fluconazole | 10.4 | 30.0 | 2.89 | No |
| Fluoxetine | 3.87 | 1.0 | 3.87 | No |
| Furosemide | 5.38 | 2.0 | 2.69 | No |
| Gabapentin | 7.06 | 5.0 | 1.41 | Yes |
| Hydroxychloroquine | 32.5 | 40.0 | 1.23 | Yes |
| Ibuprofen | 30.0 | 1.8 | 16.7 | No |
| Itraconazole | 9.36 | 16.0 | 1.71 | Yes |
| Lamotrigine | 98.0 | 31.2 | 3.14 | No |
| Levofloxacin | 12.4 | 6.0 | 2.06 | No |
| Lisinopril | 2.30 | 12.0 | 5.22 | No |
| Lorazepam | 71.0 | 12.0 | 5.92 | No |
| Losartan | 3.98 | 2.0 | 1.99 | Yes |
| Metformin | 6.24 | 6.2 | 1.01 | Yes |
| Metoprolol | 4.40 | 3.0 | 1.47 | Yes |
| Morphine | 7.52 | 2.0 | 3.76 | No |
| Olanzapine | 14.2 | 21.0 | 1.48 | Yes |
| Oxazepam | 39.7 | 8.2 | 4.84 | No |
| Phenytoin | 11.4 | 10.0 | 1.14 | Yes |
| Propranolol | 3.53 | 10.0 | 2.83 | No |
| Risperidone | 68.6 | 3.0 | 22.9 | No |
| Sertraline | 16.0 | 26.0 | 1.62 | Yes |
| Sitagliptin | 19.4 | 12.4 | 1.56 | Yes |
| Tacrolimus | 7.69 | 25.0 | 3.25 | No |
| Theophylline | 6.42 | 8.0 | 1.25 | Yes |
| Verapamil | 4.16 | 4.5 | 1.08 | Yes |
| Warfarin | 168 | 20.0 | 8.40 | No |

---

## Table S3: Temporal Holdout -- Per-Drug Results (5 drugs)

Prospective validation on drugs approved after 2022 (post-training data cutoff).
Aggregate: AAFE = 3.12, 60% within 2-fold (3/5 drugs).

| Drug | Year Approved | Pred t1/2 (h) | Obs t1/2 (h) | Fold Error |
|------|---------------|---------------|--------------|------------|
| Adagrasib | 2022 | 13.8 | 23.0 | 1.67 |
| Futibatinib | 2022 | 20.6 | 14.0 | 1.47 |
| Capivasertib | 2023 | 63.2 | 8.0 | 7.90 |
| Elacestrant | 2023 | 22.7 | 38.0 | 1.68 |
| Pirtobrutinib | 2023 | 55.0 | 19.0 | 2.90 |

---

## Table S4: CYP Genotype Scaling Factors

Activity factors relative to the reference phenotype (extensive/normal metabolizer = 1.0),
used for patient-specific clearance adjustment via `CL_adjusted = CL_pop x factor`.

**CYP2D6**

| Phenotype | Activity Factor |
|-----------|----------------|
| UM (ultra-rapid) | 1.5 |
| EM (extensive/normal) | 1.0 |
| IM (intermediate) | 0.5 |
| PM (poor) | 0.1 |

**CYP2C9**

| Genotype | Activity Factor |
|----------|----------------|
| \*1/\*1 | 1.0 |
| \*1/\*2 | 0.8 |
| \*1/\*3 | 0.6 |
| \*2/\*2 | 0.5 |
| \*2/\*3 | 0.3 |
| \*3/\*3 | 0.1 |

**CYP2C19**

| Phenotype | Activity Factor |
|-----------|----------------|
| UM (ultra-rapid) | 1.5 |
| EM (extensive/normal) | 1.0 |
| IM (intermediate) | 0.5 |
| PM (poor) | 0.2 |

---

## Table S5: Benchmark Drug SMILES and Dosing

All 20 Gold Tier benchmark drugs with canonical SMILES, dose, route, and clinical PK data source.
Clinical reference data consists of digitized concentration-time curves from published PK studies,
stored as CSV files in `benchmarks/datasets/`.

| Drug | SMILES | Dose (mg) | Route |
|------|--------|-----------|-------|
| Acetaminophen | `CC(=O)Nc1ccc(O)cc1` | 1000 | Oral |
| Amoxicillin | `CC1(C)SC2C(NC(=O)C(N)c3ccc(O)cc3)C(=O)N2C1C(=O)O` | 500 | Oral |
| Atorvastatin | `CC(C)c1n(CC[C@@H](O)C[C@@H](O)CC(=O)O)c(-c2ccccc2)c(-c2ccc(F)cc2)c1C(=O)Nc1ccccc1` | 40 | Oral |
| Caffeine | `Cn1c(=O)c2c(ncn2C)n(C)c1=O` | 100 | Oral |
| Carbamazepine | `NC(=O)N1c2ccccc2C=Cc2ccccc21` | 200 | Oral |
| D-Amphetamine | `C[C@@H](N)Cc1ccccc1` | 20 | Oral |
| Diazepam | `CN1C(=O)CN=C(c2ccccc2)c2cc(Cl)ccc21` | 10 | Oral |
| Digoxin | `C[C@@H]1O[C@@H](O[C@@H]2C[C@H](O)[C@@H](O[C@@H]3C[C@H](O)[C@@H](O[C@@H]4C[C@H](O)[C@@H](OC5CC(CO)=CC(=O)O5)C(C)O4)C(C)O3)C(C)O2)C[C@H](O)[C@H]1O` | 0.5 | Oral |
| Fluoxetine | `CNCCC(Oc1ccc(C(F)(F)F)cc1)c1ccccc1` | 20 | Oral |
| Ibuprofen | `CC(C)Cc1ccc(C(C)C(=O)O)cc1` | 400 | Oral |
| Methanol | `CO` | 100 | Oral |
| Metoprolol | `COCCc1ccc(OCC(O)CNC(C)C)cc1` | 100 | Oral |
| Midazolam | `Clc1ccc2c(c1)C(=NCc1nccn1C)c1ccccc1N2` | 2 | Oral |
| Nifedipine | `COC(=O)C1=C(C)NC(C)=C(C(=O)OC)C1c1ccccc1[N+](=O)[O-]` | 10 | Oral |
| Omeprazole | `COc1ccc2[nH]c(S(=O)Cc3ncc(C)c(OC)c3C)nc2c1` | 20 | Oral |
| Phenytoin | `O=C1NC(=O)C(c2ccccc2)(c2ccccc2)N1` | 300 | Oral |
| Propranolol | `CC(C)NCC(O)COc1cccc2ccccc12` | 80 | Oral |
| Theophylline | `Cn1c(=O)c2[nH]cnc2n(C)c1=O` | 300 | Oral |
| Verapamil | `COc1ccc(CCN(C)CCCC(C#N)(c2ccc(OC)c(OC)c2)C(C)C)cc1OC` | 80 | Oral |
| Warfarin | `CC(=O)CC(c1ccccc1)c1c(O)c2ccccc2oc1=O` | 5 | Oral |

---

## Table S6: Conformal Calibration Report

90% conformal prediction intervals evaluated on a 20% holdout set (n=30 compounds).

| Property | Coverage (%) | Target (%) | Interval Width | Status |
|----------|-------------|------------|----------------|--------|
| fup | 100.0 | 90.0 | 0.687 | Conservative |
| clint_3a4 | 36.7 | 90.0 | 0.260 | Under-covered |
| peff | 96.7 | 90.0 | 4.059 | Conservative |
| rbp | 100.0 | 90.0 | 0.840 | Conservative |

Confidence monotonicity: MONOTONIC (higher-confidence predictions have lower AAFE).
Medium-confidence fup AAFE on holdout: 1.98 (n=30).

---

## Table S7: Bronze Tier -- ADME Property Prediction Accuracy

Per-property accuracy on 153 reference compounds (151 successful predictions).

| Property | AAFE | % Within 2-fold | n |
|----------|------|-----------------|---|
| logP | 1.54 | 82.4 | 131 |
| fup | 2.10 | 58.3 | 151 |
| rbp | 1.09 | 98.0 | 151 |
| clint_3a4 | 3.25 | 33.8 | 151 |
| peff | 1.46 | 86.1 | 151 |
