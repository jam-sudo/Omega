<div align="center">

# &Omega; Omega

### Toward a Digital General Human

**Structure-based pharmacokinetic prediction using hybrid mechanistic-ML modeling**

[![Tests](https://img.shields.io/badge/tests-48K%2B_passing-brightgreen?style=for-the-badge)](#development)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue?style=for-the-badge)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-green?style=for-the-badge)](LICENSE)
[![Pre-curation AAFE](https://img.shields.io/badge/pre--curation_AAFE-2.90_%5B2.34%2C_3.67%5D-critical?style=for-the-badge)](#scientific-integrity-disclosure)
[![Post-curation AAFE](https://img.shields.io/badge/post--curation_AAFE-2.45_%5B2.09%2C_2.89%5D-orange?style=for-the-badge)](#held-out-validation-pre-curation-vs-post-curation)
[![Spearman ALL](https://img.shields.io/badge/Spearman_%CF%81_(all)-0.86-success?style=for-the-badge)](#held-out-validation-pre-curation-vs-post-curation)
[![Speed](https://img.shields.io/badge/latency-73ms%2Fdrug-informational?style=for-the-badge)](#benchmark-results)

</div>

---

> [!CAUTION]
> **Research use only.** Not validated for clinical decision-making or regulatory submissions.
>
> **The currently reported holdout AAFE 2.45 reflects iterative data curation and AD-filter tuning informed by holdout failures** &mdash; it is not a fully prospective estimate. The honest pre-curation baseline on the same 99-drug scaffold-stratified holdout was **AAFE 2.90 [2.34, 3.67]** (commit [`f59fd9c`](https://github.com/jam-sudo/Omega/commit/f59fd9c), 2026-03-20). The core-24 in-sample number (1.88) reflects tuning on that set, with 12/24 drugs using CLint anchors back-calculated from the clinical clearance being predicted (semi-supervised circularity).
>
> **Estimated prospective AAFE on a novel, curation-blind drug set: 2.5&ndash;3.0.** See [Scientific Integrity Disclosure](#scientific-integrity-disclosure) for a full audit of bias sources, each quantified where possible.

## What Omega Does

Omega predicts human plasma pharmacokinetics directly from a molecular structure (SMILES string), without requiring measured in vitro data. Given a SMILES and dose, it returns C<sub>max</sub>, AUC, t&frac12;, a full C(t) concentration-time profile, and 90% prediction intervals.

**Current stage:** Whole-body PBPK prediction from molecular structure.
**Long-term vision:** PK &rarr; PK/PD &rarr; Systems Pharmacology &rarr; Digital Twin &rarr; Digital General Human.

## How It Works

The pipeline combines ML-predicted ADME properties with a mechanistic 35-state whole-body PBPK ODE system:

```
SMILES
  │
  ▼
EnsembleADMEPredictor          XGBoost CLint/fup/rbp/VDss + polynomial logP/logS
  │
  ▼
pKa & Compound Type            RDKit SMARTS functional group detection
  │
  ▼
Drug Object Construction       IVIVE scaling, Berezhkovskiy Kp (ionization-corrected
                               for acids), renal CL, P-gp correction, gut wall CYP3A4
  │
  ▼
35-state ODE Simulation        Whole-body PBPK (15 organs, 8-segment ACAT GI tract)
  │
  ▼
PBPK/ML Ensemble               Confidence-weighted blend with direct XGBoost Cmax
  │                            (hybrid Cmax selector disabled — overfitted to N=24)
  │
  ▼
VDss Correction                Weighted geomean (XGBoost^0.7 * Berezhkovskiy^0.3)
  │                            for t1/2; ODE Kp preserved for Cmax
  │
  ▼
Applicability Domain           SMARTS-based prodrug / extreme-property flagging
  │                            (val-ester, thienopyridine, pivoxil, nucleoside ester,
  │                             quaternary amine, inorganic, P-gp efflux risk)
  │
  ▼
Adaptive Conformal UQ          90% prediction intervals (k-NN local conformal,
  │                            calibrated on 68 clean drugs)
  │
  ▼
SimulationResult               Cmax, AUC, t_half, C(t), in_applicability_domain,
                               ad_flags, confidence, intervals
```

**Key methods:** Berezhkovskiy (2004) tissue partitioning with distribution-coefficient correction for ionized acids, well-stirred hepatic clearance, IVIVE scaling (Houston 1994), Rodgers & Rowland Kp estimation, adaptive (k-NN local) conformal prediction for uncertainty quantification.

## Benchmark Results

All predictions are SMILES-only. No manual parameterization, no measured in vitro data. All reference values are sourced from FDA-approved labels or peer-reviewed clinical literature. **Read the [Scientific Integrity Disclosure](#scientific-integrity-disclosure) below before citing any of these numbers.**

### Held-Out Validation: Pre-Curation vs Post-Curation

100 drugs held out from training via Murcko generic-scaffold-stratified split (seed=42). 29 of these drugs were added from an automated OpenFDA extraction (2026-03-20) to reduce selection bias. The same holdout set was subsequently **re-examined** to identify failing drugs, and 12+ reference-data fixes plus 4 AD-filter SMARTS patterns were added in response &mdash; so the "current" metrics below are **not fully prospective**.

| Stage | N | C<sub>max</sub> AAFE | 95% CI | %2-fold | %3-fold | Spearman &rho; | Provenance |
|-------|---|----------------------|--------|---------|---------|---------------|------------|
| **Pre-curation honest** | 99 | **2.90** | [2.34, 3.67] | &mdash; | &mdash; | &mdash; | commit [`f59fd9c`](../../commit/f59fd9c), 2026-03-20 |
| Post-curation ALL (decontaminated CLint) | 100 | 2.45 | [2.09, 2.89] | 48% | 76% | **0.86** | current; 12+ data fixes + AD filter; CLint anchors decontaminated 2026-04-22 |
| Post-curation in-domain | 79 | 1.98 | [1.77, 2.23] | 57% | 84% | 0.94 | 21 drugs excluded by SMARTS/property AD filter |

> **Which number should you compare against a competitor's externally validated AAFE?**
>
> The most defensible comparator is the **pre-curation 2.90**, because it was measured before the reference-data fixes and AD SMARTS patterns were developed in response to seeing which holdout drugs failed. The "post-curation ALL 2.45" is the best we can currently re-measure, but it has known test-set leakage from the curation loop. The "in-domain 1.98" additionally excludes 21 drugs that were identified retrospectively as failing &mdash; compare to an external model's full-set AAFE, not its filtered subset.

**Multi-metric honesty (holdout):**

| Metric | N | AAFE | Spearman &rho; | Notes |
|--------|---|------|---------------|-------|
| C<sub>max</sub> (headline) | 100 | 2.45 | 0.86 | decontaminated CLint, 2026-04-22 |
| AUC | 32 (dose-matched MMPK) | **3.21** | 0.77 | VDss + CLint errors compound through ODE |
| VDss (Lombardo cross-val) | 17 | 3.71 (Berez) / 1.31 (XGB) | **0.27** | essentially **no ranking** via Berezhkovskiy Kp |
| t&frac12; | (derived from AUC/VDss) | &mdash; | &mdash; | dominated by VDss and CLint errors |

**UQ calibration:** 90% C<sub>max</sub> CI coverage = 94% (in-domain), median width = 21&times;. AUC/t&frac12; CIs use heuristic scaling from the C<sub>max</sub> q-value (not independently calibrated). Calibration set = 68 drugs, 67/68 overlap with platinum-train (not a held-out calibration fold).

**Latency:** **73 ms/drug** (warm start, single core).

Reproduce: `python scripts/run_holdout_benchmark.py`.

### Diagnostic: Core-24 (in-sample, do not compare to other models' holdout)

| Metric | Value | 95% Bootstrap CI | Notes |
|--------|-------|-------------------|-------|
| C<sub>max</sub> AAFE | 1.88 | [1.64, 2.20] | **In-sample**; 14/24 drugs still use CLint anchors back-calculated from clinical CL (decontamination only removed the 3 anchors that were in *holdout*) |
| AUC AAFE | 2.19 | [1.81, 2.68] | After XGBoost VDss geomean (XGB^0.7 &times; Berez^0.3) |
| Cmax %2-fold | 58% | &mdash; | |

Used as a regression gate, **not** a comparator. Hybrid C<sub>max</sub> selector is **disabled** (KD#3, overfitted to N=24). Reproduce: `python scripts/run_full_benchmark.py`.

<details>
<summary>Multi-tier validation details</summary>

| Tier | N | Metric | Result | 95% CI |
|------|---|--------|--------|--------|
| **Core-24** (in-sample) | 24 drugs | C<sub>max</sub> AAFE / %2-fold | **1.88** / 54% | [1.63, 2.19] |
| **Holdout in-domain** | 79 drugs | C<sub>max</sub> AAFE / %3-fold | **1.97** / 85% | [1.76, 2.22] |
| **Holdout all** | 100 drugs | C<sub>max</sub> AAFE / %2-fold | **2.44** / 50% | [2.08, 2.88] |
| **MMPK in-domain** | 850 drugs | C<sub>max</sub> AAFE / %3-fold | **2.22** / 82% | [2.07, 2.39] |
| **MMPK no-prodrug** | 743 drugs | C<sub>max</sub> AAFE | **1.91** | &mdash; |
| **AUC (holdout, dose-matched)** | 32 drugs | AUC AAFE / Spearman &rho; | **3.21** / 0.77 | &mdash; |
| **VDss (Lombardo cross-val)** | 17 drugs | VDss AAFE (XGBoost) | **1.31** vs 4.11 (Berez) | &mdash; |

</details>

<details>
<summary>Ablation study (component contributions)</summary>

Holdout (100 drugs, scaffold-stratified):

| Configuration | AAFE | 95% CI | %2-fold |
|---------------|------|--------|---------|
| Pipeline (selector OFF) &mdash; default | **2.78** | [2.28, 3.44] | 48% |
| Pipeline (selector ON) | 3.06 | [2.46, 3.88] | 45% |

The hybrid C<sub>max</sub> selector was previously reported as the largest contributor on the synthetic 24-drug benchmark. Holdout ablation shows it **worsens** AAFE by +0.28 &mdash; the selector was overfitted to the synthetic CSV via LOO-CV. Disabled by default since 2026-03-22 (CLAUDE.md KD#3).

Other ablations performed (see commit history for details):
- VDss XGB+Berez geomean: core-24 AUC 2.34 &rarr; **2.14** (-9%); Cmax unchanged
- Acid-Kp D-fix (Berezhkovskiy ionization correction): core-24 AAFE 1.75 &rarr; **1.67** (-5%)
- CLint reference anchors: ANCHORED 1.81 vs CLEAN 1.74 (&Delta; +0.08, **not** the dominant inflator)

</details>

### Related Work

| Platform | Input | C<sub>max</sub> Accuracy | Drugs | Open Source |
|----------|-------|--------------------------|-------|-------------|
| **Omega** (pre-curation, expanded holdout) | SMILES only | AAFE **2.90** [2.34, 3.67] | 99 (scaffold-stratified) | Yes |
| **Omega** (post-curation, ALL) | SMILES only | AAFE 2.44 [2.08, 2.88] | 100 | Yes |
| **Omega** (post-curation, in-domain) | SMILES only | AAFE 1.97 [1.76, 2.22] | 79 (AD-filtered) | Yes |
| Bayer AI-PBPK (Maass 2024) | SMILES only | mfce 1.87 | 9 | No |
| Jia et al. (2025) | SMILES only | 60% 2-fold | 106 | Partial |
| Simcyp / GastroPlus | Measured in vitro | &gt;80% 2-fold | 100+ | No |

> Direct comparison across studies is limited by drug-set, metric, and protocol differences. The most defensible Omega comparator is the **pre-curation 2.90** because it precedes the test-set-informed data fixes and AD SMARTS patterns. "Post-curation" numbers have known test-set leakage from the iterative-curation loop; in-domain additionally excludes 21 drugs identified *retrospectively* as failing.

## Scientific Integrity Disclosure

This section documents known biases in the reported metrics. We list them in order of severity and quantify each where possible. None of these make the pipeline "wrong", but they do mean the headline numbers overstate prospective performance.

### Tier 1: Test-Set Leakage Through Iterative Curation (HIGH severity)

**1.1 &nbsp; Reference data was iteratively fixed after examining holdout failures.**
From commit [`2f6d21e`](../../commit/2f6d21e)'s own message: *"Session: 3.520 &rarr; 1.847 (&minus;47.5%). 12 data fixes + AD filter. **Zero model changes.**"* The fixes targeted drugs observed to be failing on the holdout; each fix individually may be a legitimate data error correction, but the process is indistinguishable from test-set tuning. The pre-curation baseline on the expanded holdout is **AAFE 2.90 [2.34, 3.67]** (commit [`f59fd9c`](../../commit/f59fd9c)); the current 2.44 is 0.46 AAFE below that, attributable almost entirely to curation.

**1.2 &nbsp; Applicability-domain SMARTS were added in response to specific holdout failures.**
Commit history documents the pattern:
- [`b98bea3`](../../commit/b98bea3): *"Thienopyridine SMARTS fixed: catches clopidogrel. logP threshold 6.0&rarr;5.5 catches sonidegib."*
- [`2f6d21e`](../../commit/2f6d21e): *"Added pivoxil ester + isopropyl ester SMARTS patterns"* (for adefovir, molnupiravir)
- [`9a10e4a`](../../commit/9a10e4a): *"Add nucleoside 5'-ester SMARTS for molnupiravir prodrug detection"*

The 21 drugs excluded by the AD filter are therefore not a random OOD sample; they are drugs **selected retrospectively** because the pipeline failed on them. OOD (N=21) AAFE = 5.50, Spearman &rho; = 0.46; in-domain (N=79) AAFE = 1.97, &rho; = 0.94. Structurally: OOD median MW = 561, logP = 4.57; in-domain MW = 327, logP = 2.67.

**1.3 &nbsp; 3 CLint anchors were in the holdout set (FIXED 2026-04-22).**
Previously, `src/omega_pbpk/ml/models/adme/xgboost_clint.py` called `_get_clint_reference_anchors()` directly, so **ciprofloxacin, losartan, and ranitidine** appeared in the CLint XGBoost training data (5&times; weight, back-calculated from clinical CL) AND in the scaffold-stratified holdout. The training path now reads `holdout_split.json` and excludes these 3 anchors by default (`train(exclude_holdout=True)`). Measured impact of the fix: holdout ALL AAFE 2.440 &rarr; 2.452 (+0.012), in-domain 1.966 &rarr; 1.978 (+0.012). The leak existed architecturally but inflated headline numbers by &lt;1% &mdash; consistent with the pre-fix ablation prediction (&Delta;&nbsp;+0.08).

### Tier 2: Model Selection via Holdout Signal (MODERATE)

**2.1 &nbsp; The hybrid C<sub>max</sub> selector was disabled because it hurt the holdout** (KD#3 in `CLAUDE.md`). The selector was originally tuned via LOO-CV on a 24-drug synthetic benchmark. When the holdout showed &Delta;+0.28 AAFE with the selector on, it was disabled. This is not hyperparameter tuning per se, but it *is* model selection guided by the holdout.

**2.2 &nbsp; Five Optuna-tuned constants were reverted when they hurt the holdout** (KD#33). The revert decision used holdout feedback.

### Tier 3: Selection Biases in the Benchmark Set (MODERATE)

**3.1 &nbsp; Platinum inclusion criterion is narrow:** `oral_IR_fasted_healthy_single_dose`. Excluded: IV, SC, transdermal, fed state, controlled release, pediatric / geriatric / renal-impaired, multi-dose PK, protein biologics. Real-world PK is often messier.

**3.2 &nbsp; Scaffold split uses Murcko *generic* scaffolds** (ring-system topology only; atom types stripped). Analogs that differ only in substituents can land in both train and holdout. Stricter splits (Murcko pharmacophore, atom-path fingerprints, time-based) would likely show worse generalization.

**3.3 &nbsp; Single-metric headline.** C<sub>max</sub> is foregrounded (AAFE 1.97 in-domain / 2.44 all). AUC (AAFE 3.21, &rho;=0.77 on 32 drugs) and VDss ranking (&rho;=0.27, *essentially no correlation* via Berezhkovskiy Kp on Lombardo 17-drug set) are weaker and were previously buried in a collapsed section. They are now in the main [multi-metric honesty](#held-out-validation-pre-curation-vs-post-curation) table.

### Tier 4: Structural Risks (LOW but Consequential)

**4.1 &nbsp; Error cancellation is load-bearing.** Mean cancellation index = 0.30; 79% of drugs cancel ADME errors against ODE structural biases. Predicted ADME (AAFE 2.10 on core) beats *measured* ADME (AAFE 2.50) &mdash; the pipeline reaches correct C<sub>max</sub> via wrong intermediates. This pattern may not survive a distribution shift.

**4.2 &nbsp; Conformal calibration is not held-out.** 67/68 calibration drugs overlap with the platinum-train set. Coverage (94%) and width (21&times;) are empirically measured on the holdout so the coverage claim is valid, but the inflated width partially reflects the calibration set being in-distribution for the pipeline.

### What Would a Fully Prospective Number Look Like?

A defensible prospective AAFE would require:
1. A new drug set curated **without looking at Omega's predictions**
2. No SMARTS patterns added in response to failures on that set
3. No pipeline config changes (selector on/off, Optuna constants) made in response to the set

Our best current estimate, triangulating pre-curation 2.90 (expanded holdout), 3.52 (pre-expansion 71-drug baseline), and the MMPK 850 in-domain 2.22: **prospective AAFE ~2.5&ndash;3.0** on a curation-blind drug set.

<details>
<summary>Known limitations</summary>

- **CLint anchors decontaminated (2026-04-22)**: `train(exclude_holdout=True)` now reads `holdout_split.json` and excludes ciprofloxacin/losartan/ranitidine from anchor training. Measured impact on aggregate metrics: +0.012 AAFE (holdout ALL 2.440 &rarr; 2.452, in-domain 1.966 &rarr; 1.978). See [Scientific Integrity Disclosure §1.3](#tier-1-test-set-leakage-through-iterative-curation-high-severity)
- **Reference-data fixes and AD-filter SMARTS were developed by examining holdout failures** &mdash; current "post-curation" metrics are not fully prospective. Pre-curation honest baseline = 2.90 [2.34, 3.67]
- **CLint prediction is the primary AUC bottleneck**: 12/24 core drugs use semi-supervised anchors back-calculated from clinical CL. Holdout AUC AAFE 3.21 (vs C<sub>max</sub> 1.97) reflects CLint + VDss errors compounding through the ODE
- **Error cancellation**: predicted ADME outperforms measured ADME on the core tier &mdash; ML prediction errors partially compensate for ODE structural biases (mean cancellation index 0.30; 79% of drugs). Must be preserved when modifying individual ADME components
- **VDss systematically over-predicted by Berezhkovskiy** (Lombardo cross-val: AAFE 4.11). Mitigated by weighted geomean (XGBoost^0.7 &times; Berez^0.3) for t&frac12;; ODE Kp preserved for C<sub>max</sub>
- **Gut wall first-pass (Fg)**: CYP3A4 threshold guard prevents fm false positives, but the CLint<sub>gut</sub> scaling formula uses a pre-inverted CLint value (known architectural issue; empirically calibrated K=1.7)
- **Vd for highly protein-bound drugs (fup &lt; 0.01)**: Berezhkovskiy Kp overestimates tissue partitioning; VDss anchors partially compensate for selected drugs
- **Out-of-domain (21/100 holdout drugs)**: prodrugs, DDI-boosted (ritonavir-boosted PIs), extreme lipophilic (logP &gt; 5.5), high-MW + P-gp efflux risk &mdash; flagged via SMARTS / property thresholds in `SimulationResult.in_applicability_domain`
- **Data leakage**: 36/107 (34%) core-tier drugs overlap with ADME training set; the 100-drug scaffold-stratified holdout is leak-free
- **All synthetic CSV benchmarks are deprecated** (KD#32): inflated accuracy by ~0.5 AAFE vs clinical reference. Use clinical reference (`platinum_reference.json`) only
- **No transporter modeling**: P-gp uses a binary permeability correction only; OATP, OCT2, OAT are not represented
- **No Phase II metabolism**: UGT, NAT2, SULT enzymes not modeled
- **No dissolution model**: BCS Class II drugs assume pre-dissolved drug in solution
- **AUC/t&frac12; UQ intervals use heuristic scaling** from the C<sub>max</sub> conformal q-value (q&times;1.35 for AUC, q&times;1.0 for t&frac12;) rather than independently calibrated conformal models

</details>

## Installation

```bash
git clone https://github.com/jam-sudo/Omega.git
cd Omega
pip install -e ".[ml-new]"
pip install rdkit torch
```

<details>
<summary>Optional extras</summary>

```bash
pip install -e ".[dev]"      # Development tools (pytest, ruff, pint, pytest-benchmark)
pip install -e ".[api]"      # REST API (FastAPI)
pip install -e ".[viz]"      # Visualization (matplotlib)
pip install -e "."           # Base install (ODE engine only)
```

</details>

## Quick Start

### Population PK Prediction

```python
from omega_pbpk.pipeline import OmegaPipeline, SimulationRequest

pipeline = OmegaPipeline()
result = pipeline.simulate(SimulationRequest(
    smiles="Cn1cnc2c1c(=O)n(C)c(=O)n2C",  # caffeine
    dose_mg=100.0,
    route="oral",
))

print(f"Cmax: {result.cmax_mg_L:.2f} mg/L")
print(f"AUC:  {result.auc0t_mg_h_L:.2f} mg*h/L")
print(f"t1/2: {result.t_half_h:.1f} h")

# 90% prediction intervals
if result.cmax_ci90:
    lo, hi = result.cmax_ci90
    print(f"Cmax 90% CI: [{lo:.2f}, {hi:.2f}] mg/L")

# Applicability domain (true = in-domain; flags list reasons if out-of-domain)
if not result.in_applicability_domain:
    print(f"WARNING: out-of-domain ({', '.join(result.ad_flags)})")
```

### Batch Screening

```python
from omega_pbpk.screening.batch import batch_predict, rank_results

smiles_list = [
    "CC(C)Cc1ccc(C(C)C(=O)O)cc1",      # ibuprofen
    "CN(C)C(=N)NC(=N)N",                 # metformin
    "CC(=O)Nc1ccc(O)cc1",                # acetaminophen
]
results = batch_predict(smiles_list, dose_mg=100.0)
ranked = rank_results(results, objective="cmax")

for r in ranked:
    print(f"Rank {r['rank']}: Cmax={r['cmax_mg_L']:.2f} mg/L")
```

### Patient-Specific Prediction

```python
warfarin = "CC(=O)CC(c1ccccc1)c1c(O)c2ccccc2oc1=O"

# Weight + CYP genotype adjustment
result = pipeline.simulate(SimulationRequest(
    smiles=warfarin,
    dose_mg=5.0,
    subject_weight_kg=40.0,
    cyp2c9_genotype="*1/*3",
))

# Bayesian individual fitting from sparse C(t) observations
fit = pipeline.fit_individual(
    SimulationRequest(smiles=warfarin, dose_mg=5.0),
    observations=[(1.0, 0.15), (4.0, 0.13), (12.0, 0.05)],  # (time_h, conc_mg_L)
)
```

### CLI

```bash
omega predict --smiles "Cn1cnc2c1c(=O)n(C)c(=O)n2C" --dose 100 --model ensemble
omega benchmark                                      # Multi-drug validation
```

## Architecture

```
src/omega_pbpk/
├── pipeline/               # OmegaPipeline: SMILES → PK
│   ├── __init__.py         #   Main pipeline (simulate, fit_individual)
│   └── pk_engine.py        #   Analytical 1-compartment PK engine
├── ml/                     # ML prediction modules
│   ├── models/adme/        #   XGBoost (CLint, fup, rbp, VDss), polynomial, ensemble
│   ├── models/direct_pk/   #   Direct Cmax predictor + PBPK/ML ensemble
│   ├── models/foundation/  #   Patient encoder, covariate scaling, Bayesian fitting
│   ├── applicability.py    #   Applicability domain filter (prodrug detection)
│   └── evaluation/         #   Benchmarks, metrics, conformal calibration
├── screening/              # Batch screening engine (batch_predict, rank_results)
├── uncertainty/            # Conformal UQ (LHS parameter sampling)
├── core/                   # 35-state ODE engine (body.py, organ.py)
├── drugs/                  # Drug dataclass, named IVIVE scaling constants
├── prediction/             # pKa prediction (RDKit SMARTS), bioavailability
├── clinical/               # NCA, DDI, allometry, IVIVE, pharmacogenomics
├── population/             # Virtual population simulation (LHS CYP activity + allometry)
└── cli.py                  # CLI (typer)
```

## Training & Validation Data

| Source | Purpose | Samples |
|--------|---------|---------|
| TDC PPBR_AZ | XGBoost fup | 1,614 |
| TDC Clearance_Hepatocyte_AZ | XGBoost CLint (+18 clinical anchors @ 50&times;) | 1,231 |
| TDC VDss_Lombardo | XGBoost VDss (+2 clinical anchors) | 1,130 |
| adme_reference.csv | XGBoost RBP + ADME calibration | 153 |
| PK-DB timecourses | C(t) validation | 16 drugs |
| FDA label + literature extraction | Platinum-tier C<sub>max</sub> reference | 176 drugs |
| Murcko-scaffold split (seed=42) | Train (76) / holdout (100) | 176 drugs |
| MMPK Zenodo | Cross-validation (large-scale) | 850 in-domain |

## Roadmap

| Phase | Milestone | Status |
|-------|-----------|--------|
| **PK (current)** | SMILES &rarr; PK via hybrid mechanistic-ML | Pre-curation AAFE **2.90** [2.34, 3.67]; post-curation 2.44; &rho;=0.86 (all) / 0.94 (in-dom) |
| **Rigor (v7-v9)** | Bootstrap CI, scaffold-stratified holdout, applicability domain, UQ recalibration | Complete |
| **Structural** | pKa integration, acid-Kp D-fix, CYP3A4 gut wall guard, VDss XGB+Berez geomean | Complete |
| **AUC accuracy** | Improve VDss + CLint joint balance (current holdout AUC AAFE 3.21) | In progress |
| **PK/PD** | Efficacy/toxicity endpoints from PK profiles | Future |
| **Digital Twin** | Patient-specific multi-organ physiological model | Future |

## Development

```bash
pip install -e ".[dev]"

# Core test suite
pytest tests/ -m "not slow and not benchmark" -q       # ~48K fast tests
pytest tests/ml/test_accuracy_regression.py -v          # Accuracy regression (5 drugs)

# Gold-tier regression gate
pytest tests/regression/test_gold24_regression.py \
    -v -m benchmark                                     # AAFE ≤ 1.70, ≥75% 2-fold, latency < 500ms

# Benchmarking
python scripts/run_full_benchmark.py                    # 24-drug core benchmark (with bootstrap CI)
python scripts/run_holdout_benchmark.py                 # 100-drug scaffold-stratified holdout (in-domain + all)
python scripts/run_expanded_benchmark.py                # Expanded reference benchmark
python scripts/run_ablation.py                          # Ablation study (component contributions)
python scripts/ablation_hybrid_selector_holdout.py      # Selector ablation on holdout (KD#3 verification)
python scripts/run_measured_ablation.py                 # Error cancellation check (measured vs predicted ADME)

# Quality
ruff check . && ruff format --check .                   # Lint + format
```

Pre-commit hook runs `ruff format` and `ruff check` automatically.

## Contributing

1. Fork and create a feature branch
2. Install dev dependencies: `pip install -e ".[dev]"`
3. Write tests first (TDD)
4. Run `ruff format . && ruff check .` before committing
5. Run regression gates: `pytest tests/ml/test_accuracy_regression.py && pytest tests/regression/test_gold24_regression.py -m benchmark`
6. Open a PR against `main`

## Citation

If you use Omega in your research, please cite:

```bibtex
@software{omega_pbpk,
  title  = {Omega: Structure-Based Pharmacokinetic Prediction
            via Hybrid Mechanistic-ML Modeling},
  author = {Omega Contributors},
  url    = {https://github.com/jam-sudo/Omega},
  year   = {2026}
}
```

## License

[MIT](LICENSE)
