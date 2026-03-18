# ML-Physics Hybrid PBPK Improvement — Design Spec

**Date:** 2026-03-18
**Status:** Approved (10 rounds of self-feedback convergence)

## Problem

Current pipeline: Gold-24 Cmax AAFE 1.665 [1.44, 1.98], 88% 2-fold. Expanded-51: AAFE 4.30, 32% 2-fold.

Root causes:
1. **Warfarin compound_type bug:** pKa predictor detects enol_lactone (pKa=5.0) but pipeline passes compound_type="neutral" to Berezhkovskiy Kp → logD correction skipped → Vd inflated 5-7x → Cmax 6.95x under-predicted.
2. **Simulation time hardcoded 24h:** Fluconazole t½=30h → AUC only captures ~50% → AUC 16.59x error.
3. **CLint over-prediction for stable drugs:** Fluconazole CLint 20.7x over-predicted.
4. **Error cancellation dependence:** 79% of drugs rely on accidental error cancellation between fup, CLint, Fg, and ODE biases. Fixing individual params worsens AAFE.
5. **No transporter modeling:** P-gp only; OATP/OAT/OCT2/PepT1 missing.
6. **Data scarcity:** Only 66 Cmax drugs for ML training.

## Solution: 4-Phase ML-Physics Hybrid

### Phase 0: Immediate Bug Fixes
- Fix compound_type mapping for enol_lactone → "acid"
- Adaptive simulation time: t_sim = max(24h, 5 × t½_predicted)

### Phase 1: Data Foundation
- Expand Cmax reference from 66 → 150+ drugs via PK-DB + FDA extraction
- Quality filters (IR, single-dose, healthy adult)
- Strict train/validation/test split

### Phase 2: ML Correction Layer
- **Pre-ODE ADME Corrector:** Learn δ_fup, δ_CLint via finite-difference gradient optimization on end-to-end Cmax loss. Replaces accidental error cancellation with learned optimization.
- **Post-ODE Residual Corrector:** XGBoost/Ridge on molecular features + ODE output → log(obs/pred) residual. Replaces hybrid selector heuristics.
- **Transporter classifiers:** 6 binary models (P-gp, OATP1B1, BCRP, OCT2, OAT1/3, PepT1)
- **Adaptive Conformal UQ:** Molecular similarity-based variable-width intervals.

### Phase 3: Structural Physics-ML
- Learned VDss correction
- Renal pharmacology ML
- Multi-task ADME+PK (shared encoder, GradNorm loss)
- BCS classification + dissolution

## Expected Performance

| Phase | Gold AAFE | %2-fold | External AAFE |
|-------|----------|---------|--------------|
| Current | 1.665 | 88% | 2.95 |
| Phase 0 | ~1.58 | 88% | ~2.8 |
| Phase 2 | ~1.35 | 92% | ~2.1 |
| Phase 3 | ~1.25 | 95% | ~1.8 |

## Key Design Decisions
- ODE remains physics backbone (not replaced by pure ML, per Decision 4)
- Pre-ODE corrector manages error cancellation explicitly
- Post-ODE corrector has fallback to ODE when |correction| > 1.0 log unit
- Phase 2/3 are independent branches; Phase 3 triggers Phase 2 re-training
