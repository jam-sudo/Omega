# 2026-03-19 Session Analysis & Next Steps Plan

> **Session duration:** Full day
> **Commits:** 17 pushed to origin/main
> **Key finding:** Pipeline is tightly coupled — individual ML changes consistently fail. Only domain knowledge, data quality, and joint training can improve it.

---

## 1. What Was Accomplished

### Infrastructure (Permanent Value)

| Deliverable | Impact |
|-------------|--------|
| RDKit SMARTS pKa detection | Fixes caffeine/fluconazole/amine misclassifications |
| Named IVIVE constants + Pint proof | Prevents silent unit errors |
| Drug registry module (`drug_registry.py`) | Clean imports, removed sys.path hacks |
| Platinum schema + validation | Enforced data contract for benchmark |
| **Platinum benchmark (147 drugs)** | Single honest evaluation replacing tier system |
| **Two-level regression gate** | Core-24 (strict) + platinum (loose) in pytest |
| **Adaptive conformal UQ** | k-NN local intervals, 150-drug calibration |
| Quaternary amine + inorganic detection | Applicability domain improvements |
| **External datasets acquired** | MMPK 1,144 Cmax, PharmaBench 52K, TDC 40K |

### Metrics

| Set | N | AAFE | %2-fold |
|-----|---|------|---------|
| Core-24 | 24 | **1.502** [1.32, 1.74] | **83.3%** |
| Platinum (all) | 147 | **2.95** [2.49, 3.53] | 49.0% |
| Tractable only | 120 | **2.65** | 53% |
| Tractable FE<=20x | 113 | **2.19** | 57% |
| Untractable (flagged) | 27 | 4.80 | 30% |

---

## 2. ML Experiments — 8 Attempts, 8 Failures

| # | Attempt | Individual Metric | End-to-End Impact |
|---|---------|------------------|-------------------|
| 1 | ChemProp D-MPNN (1.2K CLint) | R² 0.218 < XGB 0.236 | Not deployed |
| 2 | ChemProp D-MPNN (3.7K CLint) | R² 0.491 < XGB 0.582 | Not deployed (XGB still wins) |
| 3 | XGBoost CLint + microsome data | CLint R² +21% | **%2-fold 83→75% (WORSE)** |
| 4 | DirectCmax MMPK 1,144 drugs | CV AAFE 10.8→3.3 | **AAFE 1.50→1.71 (GATE FAIL)** |
| 5 | Post-pipeline correction | MMPK AAFE -0.41 | **Core-24 1.50→1.83 (GATE FAIL)** |
| 6 | Pre/Post-ODE corrections (prior) | Infrastructure built | AAFE 1.65→2.69 |
| 7 | fup calibration (prior) | fup accuracy improved | AAFE +0.088 |
| 8 | ADMET-AI (prior) | Better logP/fup | Broke Vd predictions |

### Root Cause: Tightly Coupled Error Compensation

```
XGBoost CLint (wrong by X)
    ↓ anchors encode fup error
XGBoost fup (wrong by ~1/X)
    ↓ thresholds tuned for this error pattern
Hybrid selector (compensates Kp over-prediction)
    ↓ ensemble weights tuned for current DirectCmax
DirectCmax + Ensemble blend
    ↓
Final Cmax = all errors cancel → AAFE 1.50
```

**Changing ANY component breaks the co-adaptation chain.** Even improving a component's individual accuracy worsens the final output because downstream components were calibrated for the original error pattern.

### What DID Work

| Action | Type | AAFE Effect |
|--------|------|-------------|
| Acid-Kp D-fix | Domain knowledge | -0.082 |
| CYP3A4 threshold guard | Domain knowledge | -0.046 |
| VDss anchors (2 drugs) | Clinical data | -0.15 |
| Data quality fix (6 drugs) | Data quality | **-0.44** |
| Adaptive conformal | UQ improvement | 0 (by design) |
| Applicability domain | Outlier detection | 0 (flags only) |

---

## 3. External Data Acquired (Saved for Future Use)

| Source | N | Type | Location |
|--------|---|------|----------|
| **MMPK** (Li 2025) | 1,144 drugs | Oral Cmax, AUC | `data/external/mmpk/` |
| **PharmaBench** | 52,482 entries | 11 ADMET endpoints | `data/external/pharmabench/` |
| **TDC microsome CLint** | 1,102 | CLint (microsome) | `data/clearance_microsome_az.tab` |
| **TDC HLM** | 6,013 | Microsome stability | `data/hlm.tab` |
| **TDC half-life** | 667 | Human t1/2 | `data/half_life_obach.tab` |
| **TDC LogD** | 4,200 | Lipophilicity | `data/lipophilicity_astrazeneca.tab` |
| **TDC solubility** | 9,982 | Aqueous solubility | `data/solubility_aqsoldb.tab` |
| **TDC Caco-2** | 910 | Permeability | `data/caco2_wang.tab` |
| **TDC bioavailability** | 640 | Oral F (binary) | `data/bioavailability_ma.tab` |
| **TDC HIA** | 578 | Intestinal absorption | `data/hia_hou.tab` |
| **Combined CLint** | 3,712 | Merged hepatocyte+microsome+PB | `data/external/combined_clint_dataset.csv` |

**Total: ~70K+ data points across ADME endpoints.** This is sufficient for multi-task deep learning if a joint training framework is built.

### Not Yet Acquired (Blocked)

| Source | Why Blocked | Potential |
|--------|-------------|-----------|
| Lombardo/Obach 1352 | Journal paywall (403) | +680 CL+VDss |
| EPA httk fuinc | R package extraction needed | 553 fuinc values |
| ADMETlab 3.0 | Email request required | 400K molecules |
| EMA EPAR | PDF NLP extraction | +200-400 drugs |

---

## 4. Architectural Analysis

### Current Architecture (Fragile Equilibrium)

```
SMILES → [XGBoost CLint] → [XGBoost fup] → [XGBoost VDss]
              ↓                  ↓                ↓
         All trained independently on TDC data
              ↓
         Drug Object (IVIVE scaling with co-adapted anchors)
              ↓
         ODE (35-state PBPK) → Cmax_ODE
              ↓
         Analytical 1-cpt → Cmax_analytical
              ↓
         Hybrid Selector (hand-tuned thresholds)
              ↓
         DirectCmax + Ensemble (hand-tuned weights)
              ↓
         Final Cmax (error-cancelled, AAFE 1.50)
```

**Problem:** Each layer is calibrated to compensate errors from the layer below. Changing any layer cascades upward.

### Target Architecture (Joint Training)

```
SMILES → [Shared Molecular Encoder]
              ↓
         [Multi-task ADME heads: CLint, fup, VDss, RBP]
              ↓
         [Differentiable PBPK / ODE Surrogate] → Cmax
              ↓
         End-to-end Cmax loss ← observed Cmax (MMPK 1,144 drugs)
              ↓
         Gradient flows back through ENTIRE pipeline
              ↓
         Error cancellation is LEARNED, not accidental
```

**Requirements:**
- Differentiable ODE solver (adjoint method) or neural ODE surrogate
- Multi-task encoder: shared molecular representation
- End-to-end training data: 1,144+ drugs with SMILES + dose + observed Cmax
- All acquired (MMPK data + surrogate code exists)

### What's Missing

| Component | Status | Effort |
|-----------|--------|--------|
| Differentiable ODE surrogate | Code exists (`models/level2/final.pt`) but untested at scale | 2-3 weeks |
| Multi-task ADME encoder | Not built | 2-3 weeks |
| End-to-end training loop | Not built | 1-2 weeks |
| MMPK training data | **Acquired** (1,144 drugs) | Done |
| ADME training data | **Acquired** (70K+ from TDC/PharmaBench) | Done |

---

## 5. Next Steps (Priority Order)

### Tier 1: Safe, Proven Approaches (1-2 weeks each)

1. **More data quality fixes on platinum**
   - Verify remaining 147 drugs against known clinical values
   - Several drugs likely have wrong Cmax (extraction artifacts like the 6 we already fixed)
   - Expected: AAFE 2.95 → 2.5-2.7 with better reference data

2. **Multi-compartment Cmax model**
   - Diazepam (2.20x) and fluconazole (2.81x) are multi-compartment drugs
   - Vc << VDss; analytical 1-cpt with VDss over-distributes
   - Adding 2-cpt model for drugs where Vc/VDss < 0.3 could fix both
   - Expected: 2 drugs improved, core-24 AAFE ~1.45

3. **Platinum expansion to 200+**
   - Manual curation from literature (high-value approved drugs)
   - EMA EPAR extraction (European labels, different drugs)
   - Expected: 200+ drugs, more robust AAFE estimate

### Tier 2: Moderate Risk, High Potential (3-4 weeks)

4. **Joint end-to-end training framework**
   - Build multi-task ADME encoder + differentiable ODE surrogate
   - Train on MMPK 1,144 drugs with end-to-end Cmax loss
   - Validate on platinum benchmark (dual gate: core-24 + full)
   - This is the ONLY viable path for ML improvement
   - Data is ready (MMPK + PharmaBench + TDC all acquired)

### Tier 3: Research Directions (4-8 weeks)

5. **AlphaFold + DiffDock docking features**
   - CYP enzyme structures → docking scores as XGBoost features
   - Speculative but physically grounded
   - Compute-intensive (10-20 hours for 1,144 drugs)

6. **Phase II metabolism modeling**
   - UGT, SULT, NAT2 for drugs like acetaminophen, morphine
   - Requires new enzyme kinetics in ODE engine

---

## 6. Key Principles (Learned from 8 Failed ML Experiments)

1. **Never change a single ML model independently** — the pipeline is co-adapted
2. **Always test on BOTH core-24 AND platinum** — individual metrics are misleading
3. **Data quality > ML architecture** — 6 drug fixes gave more improvement than all ML combined
4. **Domain knowledge > neural networks** — at current data scale (150 PK drugs)
5. **Joint training is the only ML path** — error cancellation must be learned, not accidental
6. **The regression gate is essential** — it prevented 3 deployments that would have worsened the system
7. **Applicability domain matters** — tractable-only AAFE 2.65 vs all 2.95

---

## 7. Files Modified/Created (Summary)

### New Files (22)
- `src/omega_pbpk/data/drug_registry.py` — canonical drug list
- `src/omega_pbpk/data/platinum_schema.py` — schema + validation
- `tests/data/test_drug_registry.py` — 4 tests
- `tests/data/test_platinum_schema.py` — 7 tests
- `tests/prediction/test_pka_smarts.py` — 15 tests
- `tests/unit/test_ivive_units.py` — 7 tests
- `tests/regression/test_gold24_regression.py` — 5 tests
- `tests/regression/test_platinum_regression.py` — 6 tests
- `scripts/migrate_to_platinum.py`
- `scripts/run_platinum_benchmark.py`
- `scripts/fetch_dailymed_pk.py`
- `scripts/merge_platinum_sources.py`
- `data/clinical/platinum_reference.json` — 147 drugs
- `data/external/mmpk/` — MMPK oral PK (1,144 drugs)
- `data/external/pharmabench/` — PharmaBench ADMET (52K)
- `data/external/combined_clint_dataset.csv` — 3,712 CLint compounds
- 9 TDC .tab files (CLint, HLM, t1/2, LogD, solubility, etc.)

### Modified Files (8)
- `src/omega_pbpk/prediction/pka_predictor.py` — RDKit SMARTS
- `src/omega_pbpk/drugs/drug.py` — IVIVE constants
- `src/omega_pbpk/pipeline/__init__.py` — IVIVE import + adaptive conformal
- `src/omega_pbpk/ml/applicability.py` — quaternary amine + inorganic
- `src/omega_pbpk/data/__init__.py` — drug_registry exports
- `scripts/run_l1_benchmarks.py` — import from registry
- `scripts/run_full_benchmark.py` — import from registry
- `tests/test_pka_predictor.py` — fixed imidazole SMILES
- `README.md` — full rewrite
