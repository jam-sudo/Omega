# Plan v7: Scientific Rigor & Structural Fixes

> **Created:** 2026-03-17
> **Supersedes:** Plan v6 (2026-03-16, screening platform — all chunks DONE)
> **Origin:** 50-iteration computational biology self-critique revealing structural weaknesses
> **Status:** Phase 0 ready to execute

---

## Motivation

Plan v6 built the screening platform. Plan v7 addresses **scientific rigor issues** discovered during deep review:

1. **Error cancellation** — Gold tier AAFE 1.70 partly reflects Fg↑ × Fh↓ cancellation, not genuine accuracy
2. **pKa not connected** — `pka_predictor.py` exists but pipeline hardcodes `pka: [7.0]` for all drugs
3. **Gut wall Fg ≈ 1.0** — 1% factor dramatically underpredicts CYP3A4 first-pass (midazolam measured Fg ~0.44)
4. **CLint circular reasoning** — reference anchors back-calculated with same IVIVE formula used for scaling
5. **No bootstrap CI** — AAFE 1.70 reported without confidence interval (95% CI likely [1.3, 2.4])
6. **Ridge correction statistically insignificant** — 3.7% LOO improvement from 53 drugs, 6 features (below chance-level R²)
7. **Conformal CLint coverage 37%** — target 90%, interval 50x wide but still under-covering
8. **Salt form solubility** — `salt_form_prediction.py` exists but unconnected; BCS II drugs get free-base logS
9. **No ER-stratified validation** — high-extraction drugs (CLh ≈ Q_liver) inflate AAFE by not testing ML

## Design Principles

```
1. Diagnose before prescribe (Phase 0 results determine everything)
2. Benchmark gate at every phase (regression → rollback)
3. CLint retraining changes bundled (one retrain, not many)
4. Error cancellation monitored at every step
5. Existing modules first, new development last
6. Feature flags on all structural changes
```

---

## Phase 0: Diagnostic Sprint — NO CODE CHANGES

**Purpose:** Data-driven investment decisions. No guessing.

### 0.1 Run Existing Ablation Study

```bash
python scripts/run_ablation.py --all
# 8 configs: FULL, NO_RIDGE, NO_VDSS, NO_HYBRID, NO_PGP, NO_ENSEMBLE, NO_ENS_RIDGE, BARE
```

**Key question:** Which corrections contribute most? If BARE ≈ FULL, corrections are noise.

### 0.2 Measured-ADME Ablation (NEW script ~50 lines)

`scripts/run_measured_ablation.py`:
- Gold tier drugs with measured ADME in `data/adme_reference.csv`
- A: measured ADME → pipeline → Cmax
- B: predicted ADME → pipeline → Cmax (current)
- **Decision logic:**
  - AAFE_A ≈ AAFE_B → ODE model is bottleneck → Phase 3a priority
  - AAFE_A << AAFE_B → ADME prediction is bottleneck → Phase 3b priority
  - AAFE_A > AAFE_B → Error cancellation confirmed → fix structure before ADME

### 0.3 Bootstrap CI (NEW script ~30 lines)

`scripts/compute_bootstrap_ci.py`:
- 10,000 bootstrap resamples of gold tier fold errors
- Output: AAFE [lo, hi] 95% CI
- Check if Bayer 1.87 falls inside CI

### 0.4 ER-Stratified Validation (NEW script ~40 lines)

`scripts/run_stratified_validation.py`:
- Classify gold tier by hepatic extraction ratio (Low/Mid/High)
- Report AAFE per stratum
- **Expected:** Low-ER AAFE >> High-ER AAFE (ML actually tested only on low-ER drugs)

### 0.5 Sobol GSA

```bash
# sobol_gsa.py already exists
python scripts/run_sobol_gold_tier.py  # NEW, ~40 lines
# 5 representative drugs × Sobol analysis
# Key output: is logP total-order > CLint total-order?
```

### 0.6 Reference Data Audit (NEW script ~60 lines)

`scripts/audit_reference_data.py`:
1. Gold tier ∩ adme_reference.csv → data leakage quantification
2. Single-dose vs steady-state source verification (carbamazepine, warfarin, diazepam)
3. pKa mismatch: actual pKa vs default 7.0 for each gold tier drug
4. Salt form identification

### Phase 0 Output

```json
{
  "ablation": {"FULL": 1.70, "BARE": "?", "NO_RIDGE": "?", ...},
  "measured_vs_predicted": {"midazolam": [1.2, 1.5], ...},
  "bootstrap_ci": {"aafe_cmax": [1.3, 2.4]},
  "er_stratified": {"low": "?", "mid": "?", "high": "?"},
  "sobol": {"logP_ST": "?", "clint_ST": "?", ...},
  "data_audit": {"leakage_count": "?", "ss_drugs": [], "pka_mismatches": []}
}
```

**GATE 0:** Review diagnostic_report → confirm/adjust Phase 1-5 priorities.

---

## Phase 1: Statistical Rigor — Reporting Only

**Risk:** Very low (no science changes)
**Depends on:** Phase 0

### 1.1 Bootstrap CI in Benchmark Output

**File:** `scripts/run_full_benchmark.py`
**Change:** Add 95% CI to output JSON for AAFE, %2-fold

### 1.2 ER-Stratified Reporting

**File:** `src/omega_pbpk/validation/benchmarks.py`
**Change:** `BenchmarkSummary` gains `stratified` dict (low/mid/high ER)

### 1.3 Ridge Permutation Test

**File:** `scripts/test_ridge_significance.py` (NEW)
- 1000 permutations → p-value
- **If p ≥ 0.05:** disable Ridge in pipeline (keep code, set flag off)

### 1.4 Applicability Domain Analysis

**File:** `src/omega_pbpk/ml/evaluation/novel_validator.py` (EXISTS)
**Change:** Generate Tanimoto-vs-AAFE data for gold tier drugs

**GATE 1:** `python scripts/run_full_benchmark.py` — identical to baseline (reporting only changed)

---

## Phase 2: Integrate Existing Modules — Low Risk

**Risk:** Medium (gold tier results may change)
**Depends on:** Phase 0 (pKa mismatch data), Phase 1
**Strategy:** Each integration tested independently, benchmark before/after

### 2.1 Connect pKa Predictor

`pka_predictor.py` already exists (rule-based). Pipeline hardcodes `pka: [7.0]`.

**File:** `src/omega_pbpk/pipeline/__init__.py` (`_predict_adme` or `_build_drug`)
**Change:**
```python
from omega_pbpk.prediction.pka_predictor import predict_pka
pka_result = predict_pka(smiles)
drug_pka = [pka_result.pka] if pka_result else [7.0]
drug_type = pka_result.molecule_type if pka_result else 'neutral'
```

**Feature flag:** `USE_PREDICTED_PKA = True`
**Impact:** pKa → compound_type → Kp method → Vd → Cmax, t_half
**Risk:** May break error cancellation. Run Phase 2.3 monitor.

### 2.2 Connect Salt Form Solubility

`salt_form_prediction.py` already exists.

**File:** `src/omega_pbpk/pipeline/__init__.py` (`_build_drug`)
**Change:** Apply salt-form solubility adjustment to logS-derived solubility
**Feature flag:** `USE_SALT_CORRECTION = True`
**Impact:** BCS II drugs only (fluoxetine, verapamil, ibuprofen)

### 2.3 Error Cancellation Monitor (NEW)

`scripts/error_cancellation_analysis.py` (~80 lines):
- For drugs with measured ADME: compute parameter-level FE, intermediate FE (Fg, Fh, Vd), final FE
- Cancellation Index = |log(fe_cmax)| / Σ|log(fe_param_i)|
- CI < 0.5 → strong cancellation (fragile)
- Run after every Phase to track cancellation evolution

**GATE 2:** Benchmark per integration. Rollback if AAFE worsens > 0.3 from baseline.

---

## Phase 3a: Structural Fixes — No CLint Retrain

**Risk:** High (ODE equation changes)
**Depends on:** Phase 0 ablation results, Phase 2 complete
**Strategy:** Feature flags, independent testing

### 3a.1 Gut Wall First-Pass Correction

**Current:** `gut_clint_multiplier = 1.0` + 18g mass → Fg ≈ 1.0 for all drugs
**Measured:** midazolam Fg ≈ 0.44

**Fix 1 — Q_gut for Fg calculation:**
**File:** `src/omega_pbpk/core/body.py`
- Use villous blood flow (~23 L/h, 6% CO) instead of total mesenteric flow (~58 L/h, 15% CO)
- Only for gut wall well-stirred Fg calculation

**Fix 2 — Drug-specific gut_clint_multiplier:**
**File:** `src/omega_pbpk/pipeline/__init__.py` (`_build_drug`)
```python
# Scale by CYP3A4 contribution
gut_clint_multiplier = max(1.0, 50.0 * fm.get('CYP3A4', 0))
```

**Feature flag:** `ENABLE_GUT_WALL_FIX = True`

### 3a.2 Hybrid Cmax Selector Threshold Sweep

**File:** `src/omega_pbpk/pipeline/__init__.py`
- Sweep threshold [1.5, 2.0, 2.5, 3.0, 4.0, 5.0]
- LOO-CV on gold tier for threshold selection
- Document chosen threshold with rationale

### 3a.3 1-cpt Fallback Volume Audit

**File:** `src/omega_pbpk/pipeline/pk_engine.py`
- Verify: does 1-cpt use Vd_ss or Vc?
- If Vd_ss: fix to use Vc (blood volume-based) for Cmax calculation

**GATE 3a:** Benchmark + error cancellation monitor. AAFE must not worsen > 0.3.

---

## Phase 3b: IVIVE Fixes — CLint Retrain Required

**Risk:** High (full ML pipeline retrain)
**Depends on:** Phase 0.2 result (proceed only if AAFE_measured << AAFE_predicted)

### 3b.1 fuinc Correction

**File:** `src/omega_pbpk/ml/models/adme/xgboost_clint.py`
```python
def fuinc(logP):
    """Austin et al. 2002 — fraction unbound in incubation."""
    log_ratio = 0.072 * logP**2 + 0.067 * logP - 1.126
    return 1.0 / (1.0 + 10**log_ratio)
```
- Apply to reference anchor back-calculation
- Apply to prediction output (CLint_true = CLint_apparent / fuinc)

### 3b.2 Training Data Expansion

- Lombardo 2018 (~1,300 compounds with measured hepatocyte CLint)
- Reference DB in vivo CL for ~50 drugs (already available)
- Deduplicate against TDC AZ dataset

### 3b.3 Single Retrain

```bash
python -m omega_pbpk.ml.training.train_clint \
    --data tdc+lombardo+reference \
    --fuinc-correction \
    --cv-folds 5 \
    --output models/xgboost_clint_v2/
```

### 3b.4 Local Conformal Intervals

**File:** `src/omega_pbpk/ml/models/adme/ensemble.py`
- Replace global multiplier (50x for CLint) with k-NN local conformal
- k=50 nearest training compounds in fingerprint space
- Local residual distribution → [q5, q95] interval

**GATE 3b:** CLint AAFE (Bronze) must improve. Conformal coverage must increase.

---

## Phase 4: Mechanistic Extensions — ODE Changes

**Risk:** High (core ODE modifications)
**Depends on:** Phase 3a complete (gut wall fix is baseline)
**Strategy:** Opt-in per extension, default off, verify then activate

### 4.1 P-gp Mechanistic Efflux in ACAT

**File:** `src/omega_pbpk/core/body.py` (ACAT section)
- Replace binary `peff *= 0.5` with concentration-dependent efflux
- `ka_eff = ka / (1 + Jmax / (Km + C_lumen))`
- Drug dataclass: add `pgp_km_uM`, `pgp_jmax`
- Parameters from `transporter_reference.csv` + literature defaults

### 4.2 Extended Hepatic Clearance (OATP)

**File:** `src/omega_pbpk/core/body.py` (liver section)
- For OATP substrates: extended clearance model (Shitara 2006)
- `CLh = Q × fup × PS_uptake × CLint / (Q × (PS_uptake + PS_eff) + fup × PS_uptake × CLint)`
- Drug dataclass: add `oatp_substrate`, `ps_uptake_L_per_h`

### 4.3 Renal Transporter Terms

**File:** `src/omega_pbpk/core/body.py` (kidney section)
- `CLr = GFR × fup + CL_secretion` (OCT2, OAT1/3)
- Drug dataclass: add `oct2_substrate`, `oat_substrate`

**GATE 4:** Benchmark per extension independently, then combined.

---

## Phase 5: Validation Expansion & Paper

**Depends on:** Phases 1-4 stabilized

### 5.1 Gold Tier N=50+

- Select from 285 drugs: Platinum+Gold tier, non-overlapping with ADME training
- Single-dose data only (exclude steady-state references)
- Balanced by BCS class, ER, drug_type

### 5.2 Temporal Holdout N=15-20

- FDA 2023-2025 approved drugs with label PK data
- SMILES from PubChem/DrugBank
- Must be post-training-cutoff

### 5.3 Error Cancellation Formal Analysis

- Full parameter-level decomposition for all gold tier drugs
- Cancellation Index table for paper supplementary
- Discuss implications for prospective use

### 5.4 Population Simulation for Gold Tier

- CYP2D6/2C19 substrate drugs: simulate N=100 virtual subjects
- Compare simulated geometric mean to clinical geometric mean
- Eliminates reference-man vs population-mean bias

### 5.5 Paper Rewrite

| Section | Addition |
|---------|----------|
| Methods | pKa prediction, salt form, fuinc, gut wall model |
| Results | Bootstrap CI, ER-stratified table, ablation figure |
| Results | Applicability domain (Tanimoto vs AAFE scatter) |
| Discussion | Error cancellation analysis, information-theoretic floor (~1.23) |
| Discussion | Reframe: "predicted-input achieves measured-input accuracy" |
| Limitations | CLint UQ coverage, transporter gaps, Phase II |
| Supplementary | Sobol indices, full drug-level results, cancellation table |

---

## Dependency Graph

```
Phase 0 (Diagnostic)
  │
  ├──→ Phase 1 (Stats)        ── low risk, do immediately
  │     └── Ridge decision
  │
  ├──→ Phase 2 (Integration)  ── connect existing modules
  │     ├── 2.1 pKa
  │     ├── 2.2 Salt form
  │     └── 2.3 Cancellation monitor
  │
  ├──→ Phase 3a (Structural)  ── no retrain
  │     ├── Gut wall Fg
  │     ├── Hybrid threshold
  │     └── 1-cpt Vc audit
  │
  ├──→ Phase 3b (IVIVE)       ── retrain once (conditional on Phase 0.2)
  │     ├── fuinc
  │     ├── Training data
  │     ├── Retrain
  │     └── Local conformal
  │
  ├──→ Phase 4 (Mechanistic)  ── after Phase 3a stable
  │     ├── P-gp ACAT
  │     ├── OATP liver
  │     └── Renal transport
  │
  └──→ Phase 5 (Validation)   ── after all phases
        ├── Gold N=50+
        ├── Temporal N=15+
        ├── Cancellation analysis
        ├── Population simulation
        └── Paper rewrite
```

## Success Metrics

| Metric | Current | After Ph2 | After Ph3 | After Ph4 | Final |
|--------|---------|-----------|-----------|-----------|-------|
| Gold AAFE (N=25) | 1.70 | ≤1.70 | ≤1.60 | ≤1.55 | ≤1.50 |
| Gold %2-fold | 80% | ≥80% | ≥85% | ≥85% | ≥85% |
| External/Temporal AAFE | 2.95/3.12 | ≤3.00 | ≤2.50 | ≤2.30 | ≤2.50 |
| CLint coverage | 37% | 37% | ≥60% | ≥70% | ≥75% |
| Gold N | 25 | 25 | 25 | 25 | 50+ |
| Bootstrap CI | N/A | reported | narrower | narrower | published |
| Low-ER AAFE | unknown | reported | improved | improved | ≤2.0 |

## Rollback Rule

> If any Phase increases gold tier AAFE by > 0.3 above baseline (1.70),
> rollback that Phase's changes and investigate root cause before retrying.
> Error cancellation monitor (`scripts/error_cancellation_analysis.py`)
> must be run after every structural change.

---

## Team Deployment Strategy (`/team`)

Use `/team` to spawn Agent Teams (via `TeamCreate`) for parallelizable work within each Phase.
Teammates communicate via `SendMessage`, report to `docs/team/findings.md`, blockers to `docs/team/blockers.md`.
Team-lead (Claude) handles git, resolves blockers, synthesizes findings.

### Phase 0 — 4 teammates in parallel

| Role | Task | Key Files (owns) | Output |
|------|------|-------------------|--------|
| **data-engineer** | Audit reference data: leakage check (gold ∩ adme_reference), single-dose vs steady-state, pKa mismatches, salt forms | `scripts/audit_reference_data.py` (NEW), `data/` | `diagnostic_report.data_audit` |
| **domain-scientist** | Run ablation study (8 configs) + design measured-ADME ablation script | `scripts/run_ablation.py`, `scripts/run_measured_ablation.py` (NEW) | `diagnostic_report.ablation`, `diagnostic_report.measured_vs_predicted` |
| **ml-engineer** | Run Sobol GSA on 5 representative drugs + compute bootstrap CI | `scripts/run_sobol_gold_tier.py` (NEW), `scripts/compute_bootstrap_ci.py` (NEW) | `diagnostic_report.sobol`, `diagnostic_report.bootstrap_ci` |
| **ci-auditor** | Verify full test suite passes, check ER-stratified validation | `scripts/run_stratified_validation.py` (NEW), `tests/` | `diagnostic_report.er_stratified`, CI health report |

**File conflict avoidance:** Each teammate creates NEW scripts only; nobody edits shared files.

### Phase 1 — 2 teammates

| Role | Task | Key Files |
|------|------|-----------|
| **ml-engineer** | Add bootstrap CI to benchmark output, implement Ridge permutation test | `scripts/run_full_benchmark.py`, `scripts/test_ridge_significance.py` (NEW) |
| **domain-scientist** | Add ER-stratified reporting to BenchmarkSummary, applicability domain analysis | `src/omega_pbpk/validation/benchmarks.py` |

**Cross-review:** ci-auditor reviews both after completion.

### Phase 2 — 3 teammates

| Role | Task | Key Files |
|------|------|-----------|
| **infra-engineer** | Connect pKa predictor to pipeline (_predict_adme or _build_drug) | `src/omega_pbpk/pipeline/__init__.py` (pKa section only) |
| **domain-scientist** | Connect salt form prediction to solubility in _build_drug | `src/omega_pbpk/pipeline/__init__.py` (solubility section only) |
| **ml-engineer** | Build error cancellation monitor script | `scripts/error_cancellation_analysis.py` (NEW) |

**File conflict:** infra-engineer and domain-scientist both touch `pipeline/__init__.py` — assign separate sections (pKa vs solubility) or serialize. Safer: infra-engineer does both pipeline changes, domain-scientist validates results.

**Revised assignment to avoid conflict:**

| Role | Task | Key Files |
|------|------|-----------|
| **infra-engineer** | Connect pKa + salt form to pipeline (both changes in __init__.py) | `src/omega_pbpk/pipeline/__init__.py` |
| **ml-engineer** | Build error cancellation monitor | `scripts/error_cancellation_analysis.py` (NEW) |
| **domain-scientist** | Validate pKa predictions vs literature for gold tier, prepare expected-impact analysis | `docs/team/findings.md` |

### Phase 3a — 3 teammates

| Role | Task | Key Files |
|------|------|-----------|
| **ode-engineer** | Gut wall Fg fix: Q_villous + drug-specific gut_clint_multiplier | `src/omega_pbpk/core/body.py`, `src/omega_pbpk/drugs/drug.py` |
| **ml-engineer** | Hybrid Cmax threshold sweep (LOO-CV on gold tier) | `src/omega_pbpk/pipeline/__init__.py` (hybrid section) |
| **infra-engineer** | 1-cpt Vc audit + fix if needed | `src/omega_pbpk/pipeline/pk_engine.py` |

**Cross-review:** domain-scientist reviews ode-engineer's Fg fix for physiological correctness.

### Phase 3b — 3 teammates

| Role | Task | Key Files |
|------|------|-----------|
| **ml-engineer** | fuinc correction + XGBoost retrain + local conformal intervals | `src/omega_pbpk/ml/models/adme/xgboost_clint.py`, `src/omega_pbpk/ml/models/adme/ensemble.py` |
| **data-engineer** | Acquire + clean Lombardo 2018 data, merge with TDC + reference | `src/omega_pbpk/ml/data/`, `data/` |
| **ci-auditor** | Run full test suite after retrain, verify no regression | `tests/`, benchmark scripts |

### Phase 4 — 2 teammates

| Role | Task | Key Files |
|------|------|-----------|
| **ode-engineer** | P-gp ACAT efflux + OATP extended clearance + renal transporters | `src/omega_pbpk/core/body.py`, `src/omega_pbpk/drugs/drug.py` |
| **domain-scientist** | Curate transporter Km/Jmax from literature for ~20 P-gp substrates | `data/transporter_reference.csv`, `docs/team/findings.md` |

**Cross-review:** ml-engineer validates that new Drug fields don't break ADME pipeline.

### Phase 5 — 3 teammates

| Role | Task | Key Files |
|------|------|-----------|
| **data-engineer** | Gold tier expansion (N=50+), temporal holdout (N=15+) | `data/clinical/reference_database.json` |
| **domain-scientist** | Error cancellation formal analysis, population simulation for CYP2D6 drugs | `scripts/`, `docs/paper/` |
| **ml-engineer** | Paper figure generation (ablation, ER-stratified, Tanimoto scatter) | `docs/paper/figures/` |

### Team Spawn Template

```
/team
Task: Plan v7 Phase {N} — {description}
Roles needed: {role1}, {role2}, {role3}
Context: See docs/superpowers/plans/2026-03-17-omega-v7-scientific-rigor.md Phase {N}
```

### Team Coordination Rules

1. **No two teammates edit the same file** — break work by file ownership
2. **ci-auditor runs after every Phase** — `pytest tests/ -m "not slow" -q` + `ruff check .`
3. **domain-scientist validates every structural change** — physiological correctness check
4. **Team-lead (Claude) handles all git** — teammates report findings, don't commit
5. **Error cancellation monitor runs after Phase 2, 3a, 3b, 4** — ml-engineer's responsibility
