# Omega PBPK — Benchmark Suite

> [!WARNING]
> **Deprecated for headline accuracy claims (2026-03-22, CLAUDE.md KD#32).**
> The 5-drug suite in this directory uses **synthetic 1-compartment-back-calculated** time courses (parameters → 1-cpt absorption model → ±15% lognormal noise). Reporting AAFE on these CSVs inflated accuracy by ~0.5 vs. the clinical reference and overfitted the hybrid C<sub>max</sub> selector via LOO-CV.
>
> **Use for:** quick smoke testing, reproducibility of legacy results, CLI demonstration.
> **Do not use for:** headline accuracy claims, model selection, hyperparameter tuning.
> **For honest accuracy:** run `python scripts/run_full_benchmark.py` (core-24 with clinical reference) and `python scripts/run_holdout_benchmark.py` (100-drug scaffold-stratified holdout).

## Overview

This benchmark suite validates the Omega PBPK model against five well-characterised reference compounds. The original literature parameters (C<sub>max</sub>, T<sub>max</sub>, t&frac12;, AUC) are sourced from peer-reviewed clinical studies and FDA labels, but the time-course CSVs are reconstructed via a 1-compartment oral absorption model with added noise — they are **not** raw clinical observations.

## What the Benchmark Validates

For each compound the benchmark:
1. Runs the full Omega PBPK simulation using the compound parameters in `configs/`.
2. Compares simulated plasma concentration–time profiles to the observed clinical data in `datasets/`.
3. Computes three acceptance metrics:
   - **AUC relative error**: |AUC_sim - AUC_obs| / AUC_obs
   - **Cmax relative error**: |Cmax_sim - Cmax_obs| / Cmax_obs
   - **Tmax absolute error**: |Tmax_sim - Tmax_obs| in hours

## Compounds and Data Sources

### Caffeine — 100 mg oral (healthy adults)
- **Cmax**: ~1.5–2.0 mg/L at Tmax ~0.5–1.0 h
- **t½**: ~4–6 h
- **AUC0-∞**: ~10–14 mg·h/L
- **Bioavailability**: ~100%
- **References**: Arnaud MJ. Pharmacol Ther. 1993;60(2):289-392; Nehlig A. Pharmacol Ther. 2016;163:79-92; FDA caffeine label.
- **PK parameters used**: kel = 0.1386 h⁻¹, ka = 4.89 h⁻¹, V = 60 L, F = 1.0

### Warfarin — 10 mg oral (S-warfarin, healthy adults)
- **Source**: PK-DB (replaced original Holford 5mg data due to corrupt Tmax=18h)
- **References**: PK-DB database; FDA warfarin label.

### Metoprolol — 100 mg oral tartrate (healthy adults)
- **Cmax**: ~0.04–0.08 mg/L at Tmax ~1.5–2.0 h
- **t½**: ~3–4 h
- **AUC0-24h**: ~0.2–0.4 mg·h/L
- **Bioavailability**: ~50% (moderate first-pass)
- **References**: Regardh CG et al. Eur J Clin Pharmacol. 1980;18(4):321-9; FDA metoprolol label.
- **PK parameters used**: kel = 0.198 h⁻¹, ka = 1.252 h⁻¹, V = 589 L, F = 0.50

### Midazolam — 2 mg oral (healthy adults)
- **Cmax**: ~0.003–0.008 mg/L (3–8 ng/mL) at Tmax ~0.5–1.0 h
- **t½**: ~2–3 h
- **AUC0-∞**: ~0.015–0.025 mg·h/L
- **Bioavailability**: ~30–40% (high first-pass / CYP3A4 gut wall metabolism)
- **References**: Greenblatt DJ et al. Clin Pharmacol Ther. 1992;52(2):157-68; Thummel KE et al. J Pharmacol Exp Ther. 1996;277(1):114-21.
- **PK parameters used**: kel = 0.2772 h⁻¹, ka = 3.750 h⁻¹, V = 113.7 L, F = 0.35

### Propranolol — 80 mg oral (healthy adults)
- **Cmax**: ~0.05–0.12 mg/L (50–120 ng/mL) at Tmax ~1.5–2.0 h
- **t½**: ~3.5–5 h
- **AUC0-24h**: ~0.4–0.8 mg·h/L
- **Bioavailability**: ~25–30% (extensive first-pass)
- **References**: Walle T et al. Clin Pharmacol Ther. 1985;37(6):598-606; FDA propranolol label.
- **PK parameters used**: kel = 0.1732 h⁻¹, ka = 1.344 h⁻¹, V = 199 L, F = 0.27

## Dataset Generation

Each `datasets/*.csv` file was generated with:
- A 1-compartment oral absorption model: C(t) = (F·dose·ka) / (V·(ka − kel)) · (e^(−kel·t) − e^(−ka·t))
- Parameters back-calculated from published Cmax, Tmax, t½, and AUC values (see above)
- ±15% lognormal inter-individual noise (numpy.random seed=0) to simulate published mean data variability
- Standard deviation column = 20% of observed concentration (typical published mean ± SD)
- Timepoints (h): 0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 18.0, 24.0
- Units: mg/L throughout (1 mg/L = 1000 ng/mL)

## Acceptance Criteria

The acceptance thresholds in `expected/acceptance.json` reflect the standard regulatory acceptance criterion for PBPK models:

| Metric | Threshold | Basis |
|--------|-----------|-------|
| AUC relative error | ≤ 0.80 (2-fold) | EMA/FDA PBPK guideline |
| Cmax relative error | ≤ 0.80 (2-fold) | EMA/FDA PBPK guideline |
| Tmax absolute error | ≤ 3.0 h | Practical clinical relevance |

A 2-fold accuracy criterion (50–200% of observed) is the accepted standard for PBPK model validation per the FDA Draft Guidance on PBPK (2018) and EMA Guideline on the reporting of PBPK modelling and simulation (2018). Tighter criteria appropriate only for data-trained empirical PK models are not applicable here.

## Benchmark Results Interpretation

Because these datasets now reflect real (rather than synthetic) clinical data, some compounds may fail the acceptance criteria. This is scientifically expected and informative:
- **PASS** (RE ≤ 0.80): The PBPK model adequately predicts this compound's in vivo PK from first principles.
- **FAIL** (RE > 0.80): The PBPK model has systematic error for this compound — a meaningful signal for model refinement (e.g., adjust fup, Clint, absorption parameters).
- **RE > 2.0**: Suggests a potential unit mismatch, parameter error, or structural model inadequacy — warrants investigation.

## How to Run

```bash
# Via the CLI
omega benchmark

# Via Python
python3 -c "
from omega_pbpk.validation.benchmarks import run_benchmark_suite
summary = run_benchmark_suite('benchmarks', '/tmp/benchmark_results')
print('n_drugs:', summary['n_drugs'])
for r in summary['results']:
    m = r['metrics']
    print(f\"{r['drug']:15s}  AUC_RE={m['auc_relative_error']:.3f}  Cmax_RE={m['cmax_relative_error']:.3f}  {'PASS' if r['pass'] else 'FAIL'}\")
"
```

## Directory Structure

```
benchmarks/
  configs/          — Per-compound simulation config (YAML)
  datasets/         — Observed clinical PK data (CSV, mg/L)
  expected/         — Acceptance thresholds (JSON)
  README.md         — This file
```
