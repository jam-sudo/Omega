# Platinum Benchmark: Unified Clinical Reference System

> **Status:** Design (pending approval)
> **Goal:** Single clinical benchmark replacing the tier system, targeting 150-200 drugs.
> **Scope:** Data acquisition pipeline, quality framework, unified benchmark, regression gate.

---

## 1. Motivation

The current tier system (gold/silver/bronze/temporal/external) is an artifact of incomplete data collection, not a design choice. Gold-24 AAFE 1.50 reflects tuning on a small curated set; expanded-51 AAFE 3.40 shows how fragile that is. The gap isn't model quality — it's data coverage.

**Core insight:** The pipeline (`OmegaPipeline.simulate`) is fully generic. It works for any SMILES. The bottleneck is reference data, not code.

**What "Platinum" means:** Every drug in the benchmark meets the same inclusion standard:

```
Drug in Platinum  <=>  has {
    smiles            : valid RDKit SMILES
    dose_mg           : single oral IR dose (mg)
    cmax_mg_L         : observed Cmax (mg/L)
    source            : PMID or FDA NDA number
    fasted_confidence : confirmed_fasted | assumed_fasted
    data_quality      : fda_label_exact | clinical_exact | clinical_dose_normalized | fda_label_dose_normalized | fda_label_median
}
```

**Honest framing:** The core-24 drugs remain a high-confidence subset for regression detection. This is a practical engineering choice, not a tier — the same pipeline, same metrics, same thresholds apply to all drugs. The core-24 simply has tighter regression bounds because its reference data has been manually verified.

---

## 2. Current State & Gap Analysis

### What exists

| Source | Drugs with Cmax | SMILES | Usable | Bottleneck |
|--------|----------------|--------|--------|------------|
| gold24_reference_cmax.json | 24 | 24 | 24 | Manual curation |
| expanded_cmax.csv | 129 | 129 | ~120 | Mixed quality, typos (ciprofolxacin) |
| reference_database.json | 96 (Cmax) | 266 | ~89 | Silver tier has no Cmax |
| OpenFDA PK extracted | 10 Cmax | 43 | 10 | Regex can't parse tables |
| PK-DB expanded | 3 | 3 | 0 | Mock data only |

**Current usable total after dedup: ~148 unique drugs** (cross-referenced across all sources; includes quality issues in some entries)

### What's missing

| Gap | Impact | Fix |
|-----|--------|-----|
| FDA table parsing | +30-80 drugs (see Phase 0 caveat) | DailyMed SPL XML + improved regex |
| PK-DB integration dormant | +15-25 drugs | Fix API queries, remove mock data |
| No fasted/fed metadata | +/-50% Cmax noise for lipophilic drugs | Extract from label text |
| No IR/ER distinction | Wrong baseline for ER drugs | Formulation field + filter |
| No route tracking | IV drugs may contaminate | Add route field, filter oral |
| No population filter | Disease state alters PK | Extract "healthy volunteers" |
| No tuning contamination tracking | Biased CV results | Track which drugs have anchors |
| Non-linear PK drugs unhandled | Wrong Cmax at extracted dose | Annotate and handle separately |

---

## 3. Data Acquisition Pipeline

### 3.0 Phase 0: Proof-of-Concept Yield Measurement (2-3 days)

**Before committing to the full pipeline, validate DailyMed yield on 20 drugs.**

Many FDA labels present PK data as narrative prose, not structured `<table>` elements. A live test of atorvastatin's DailyMed XML showed Cmax embedded in text, not in a table. This is representative of many labels.

- [ ] Select 20 drugs: 10 known high-quality labels (e.g., metformin, atorvastatin, omeprazole) + 10 random from expanded set
- [ ] Query DailyMed API for SPL XML
- [ ] Count: how many have Cmax in `<table>` elements vs narrative text
- [ ] Measure extraction success rate

**Gate:** If `<table>` extraction succeeds for >= 10/20 drugs (50%), proceed with DailyMed as primary source. If < 50%, fall back to improved regex (Option A) and adjust drug count target to 150.

### 3.1 Source Priority

```
Priority 1: FDA Labels (DailyMed XML tables + OpenFDA regex fallback)
    |
    v
Priority 2: PK-DB (literature C(t) curves, computable Cmax)
    |
    v
Priority 3: Manual Curation (high-value drugs not in above)
```

### 3.2 FDA Label Extraction (Primary Source)

**Current problem:** `extract_openfda_pk.py` uses regex on text blobs. Only 10/43 drugs successfully extracted.

**Dual strategy (informed by Phase 0 results):**

| Approach | Expected Yield | Effort | When |
|----------|---------------|--------|------|
| DailyMed SPL XML table parsing | +30-80 drugs | 4-5 days | If Phase 0 >= 50% success |
| Improved OpenFDA regex | +20-40 drugs | 2-3 days | Always (fallback) |

**DailyMed XML parser:** `scripts/fetch_dailymed_pk.py`
- Query DailyMed API for SPL XML
- Parse `<table>` elements in "Clinical Pharmacology" section
- Extract column headers -> identify Cmax, AUC, Tmax columns
- **Label selection criteria:**
  - Prefer reference-listed drug (original NDA holder) over generic repackagers
  - Prefer labels with longest "Clinical Pharmacology" section
  - Prefer tablet/capsule over solution/injection
  - Filter for oral dosage forms specifically
- Filter: fasted state, oral IR, healthy volunteers, single dose
- Output: structured JSON per drug

**Improved regex:** enhance existing `extract_openfda_pk.py`
- Add patterns for tabular text (column-aligned numbers)
- Add context-aware dose extraction (reject daily max, bid/tid doses)
- Add fasted-state keyword detection in surrounding text

### 3.3 PK-DB Integration (Secondary Source)

**Fix:**
- Remove mock data from `expand_pkdb_cmax.py`, enable real PK-DB API calls
- Extend MW table (60 -> 200 drugs via PubChem API)
- Compute Cmax from C(t) timecourse data (21 drugs in pkdb_timecourses.json)
- Add study selection scoring: prefer single-dose, oral, fasted, healthy, N>6

**Expected yield:** 15-25 additional drugs not in FDA labels

### 3.4 Manual Curation (Gap Fill)

For high-value drugs without automated extraction:
- PubMed search for "drug name pharmacokinetics healthy volunteers"
- Extract from paper tables (human judgment)
- Document: PMID, Table/Figure number, dose, formulation, fasted state

**Target:** ~10-20 drugs

---

## 4. Quality Assurance Framework

### 4.1 Mandatory Fields (Platinum Entry Requirements)

| Field | Type | Validation | Source |
|-------|------|-----------|--------|
| `drug_name` | str | Normalized (lowercase, no salts) | All |
| `smiles` | str | RDKit parseable, >5 heavy atoms | PubChem |
| `dose_mg` | float | 0.1-5000 | Label/paper |
| `cmax_mg_L` | float | >0, ratio check | Label/paper |
| `source_type` | enum | fda_label, literature, pkdb | Extraction |
| `source_id` | str | NDA number or PMID | Extraction |
| `fasted_confidence` | enum | confirmed_fasted, assumed_fasted | Extracted or inferred |
| `formulation` | enum | IR, ER, solution, other | Extracted |
| `route` | enum | oral (required) | Extracted |
| `population` | str | "healthy" (required for primary AAFE) | Extracted |
| `single_dose` | bool | True required | Extracted |
| `tuning_contaminated` | bool | True if drug has CLint/VDss anchor or hand-tuned params | Manual annotation |
| `nonlinear_pk` | bool | True if known dose-dependent PK | Manual annotation |

### 4.2 Optional Fields

| Field | Type | Notes |
|-------|------|-------|
| `auc_mg_h_L` | float | AUC if available; validated same as Cmax (>0, ratio check) |
| `thalf_h` | float | Terminal half-life |
| `tmax_h` | float | Time to Cmax |
| `f_oral` | float | Oral bioavailability (0-1) |

### 4.3 Fasted State Classification

| Level | Definition | Benchmark Role |
|-------|-----------|----------------|
| `confirmed_fasted` | Label explicitly states "fasted" or "after overnight fast" | Primary AAFE |
| `assumed_fasted` | Label says "single dose" without specifying food; IR formulation | Primary AAFE (annotated) |
| `fed_only` | Only fed-state data available | Excluded from primary AAFE; reported in exploratory set |

### 4.4 Non-Linear PK Drug Handling

Drugs with known non-linear PK (autoinhibition, saturable absorption, Michaelis-Menten clearance):
- **Include in benchmark** at a dose where PK is approximately linear (typically lowest recommended dose)
- **Annotate** `nonlinear_pk: true`
- **Report separately** in benchmark output (AAFE with/without non-linear drugs)
- Known non-linear drugs: omeprazole, gabapentin, phenytoin (high dose), tacrolimus, cyclosporine, valproic acid (high dose)

### 4.5 Automated Sanity Checks

```python
# Cmax/dose ratio plausibility (oral PK typical range)
assert 1e-6 < cmax_mg_L / dose_mg < 1.0

# Cross-check: run pipeline prediction, flag if >10x discrepancy
pred = pipeline.simulate(smiles, dose_mg)
ratio = max(pred.cmax / obs_cmax, obs_cmax / pred.cmax)
if ratio > 10.0:
    flag_for_manual_review(drug)

# Molecular weight sanity
mw = Descriptors.MolWt(Chem.MolFromSmiles(smiles))
assert 100 < mw < 1500

# Name-based dedup (Levenshtein distance, salt stripping)
check_name_duplicates(drug_name, existing_names, threshold=0.85)

# Structure-based near-duplicate detection
# Flag at Tanimoto > 0.90, auto-reject at > 0.99 (same compound, different names)
check_structural_duplicates(smiles, existing_smiles, flag=0.90, reject=0.99)
```

### 4.6 Manual Review Queue

Drugs automatically flagged for human review:
- Prediction/observation ratio > 10x
- Cmax/dose ratio outside 1e-5 to 0.5
- Non-linear PK drugs
- Multiple conflicting Cmax values from different sources
- Name-based near-duplicates (Levenshtein > 0.85)

---

## 5. Platinum Reference Schema

**File:** `data/clinical/platinum_reference.json`

```json
{
  "metadata": {
    "version": "1.0",
    "n_drugs": 200,
    "created": "2026-03-19",
    "inclusion_standard": "oral_IR_fasted_healthy_single_dose"
  },
  "drugs": {
    "caffeine": {
      "smiles": "Cn1cnc2c1c(=O)n(C)c(=O)n2C",
      "dose_mg": 100.0,
      "cmax_mg_L": 1.74,
      "auc_mg_h_L": 14.2,
      "source_type": "fda_label",
      "source_id": "NDA 020863",
      "fasted_confidence": "confirmed_fasted",
      "formulation": "IR",
      "route": "oral",
      "population": "healthy",
      "single_dose": true,
      "tuning_contaminated": false,
      "nonlinear_pk": false,
      "data_quality": "fda_label_exact",
      "notes": ""
    }
  }
}
```

**Migration:** Gold-24 entries are absorbed into platinum. The `BENCHMARK_DRUGS` dict currently in `run_l1_benchmarks.py` is extracted into a shared module (`src/omega_pbpk/data/drug_registry.py`) before any script deprecation, since `test_gold24_regression.py` imports it.

---

## 6. Unified Benchmark System

### 6.1 Prerequisite: Extract Drug Registry

Before replacing benchmark scripts, extract `BENCHMARK_DRUGS` from `scripts/run_l1_benchmarks.py` into:
- `src/omega_pbpk/data/drug_registry.py` — canonical drug list with SMILES and doses
- Both old scripts and new platinum script import from here
- `test_gold24_regression.py` updated to import from new location

### 6.2 Single Benchmark Script

**File:** `scripts/run_platinum_benchmark.py`

Supplements (not immediately replaces) existing scripts. Old scripts kept for reproducibility.

```bash
# Default: all platinum drugs
python scripts/run_platinum_benchmark.py

# Subset: only core-24 (backward compatibility)
python scripts/run_platinum_benchmark.py --subset core24

# With bootstrap CI
python scripts/run_platinum_benchmark.py --bootstrap 10000

# Cross-validation mode
python scripts/run_platinum_benchmark.py --cv 5

# Exclude tuning-contaminated drugs
python scripts/run_platinum_benchmark.py --clean-only
```

**Outputs:**
- Console: AAFE, %2-fold, bootstrap CI, per-drug fold errors
- JSON: `outputs/platinum_benchmark_YYYY-MM-DD.json`
- CSV: `outputs/platinum_per_drug_YYYY-MM-DD.csv`

### 6.3 Cross-Validation Strategy

**Decision: Stratified 5-fold CV for development, temporal holdout for publication.**

| Mode | Train | Test | Use |
|------|-------|------|-----|
| 5-fold CV | 160 (stratified by compound type, log dose) | 40 per fold | Development iteration |
| Temporal holdout | Pre-2024 approval | 2024+ approval | Publication claims |
| Clean holdout | All drugs | Non-contaminated only | Honest generalization |

**Critical rules:**
- CLint anchor drugs (19, not 18 — verified count from `_get_clint_reference_anchors`) and VDss anchor drugs (2) must always be in the training fold
- `tuning_contaminated` drugs: always report AAFE separately with and without these drugs
- Non-linear PK drugs: report AAFE with and without

### 6.4 Metric Reporting

| Metric | Definition |
|--------|-----------|
| AAFE | exp(mean(\|log(pred/obs)\|)) |
| %2-fold | fraction with fold error <= 2.0 |
| %3-fold | fraction with fold error <= 3.0 |
| Bootstrap 95% CI | 10,000 resamples of AAFE |
| Max fold error | worst single drug |
| Median fold error | robust central tendency |
| AAFE (clean subset) | excluding tuning-contaminated drugs |
| AAFE (linear PK only) | excluding nonlinear_pk drugs |

---

## 7. Regression Gate

### 7.1 Two-Level Gate

**Level 1: Core-24 (strict — regression detection for curated drugs)**

```python
CORE24_AAFE_MAX = 1.70      # current: 1.50
CORE24_PCT2FOLD_MIN = 75.0   # current: 83%
CORE24_MAX_SINGLE_FE = 6.0   # current: max 3.99 (midazolam)
```

**Level 2: Full Platinum (looser — catastrophic regression prevention)**

```python
PLATINUM_AAFE_MAX = 3.50      # start loose; tighten after baseline run
PLATINUM_PCT2FOLD_MIN = 40.0  # expected: ~40-55% initially
PLATINUM_MAX_SINGLE_FE = 10.0 # some drugs will be far off
```

Level 2 thresholds tighten as the pipeline improves. They start loose because the full set includes noisier references and more chemical diversity.

### 7.2 Test File

**File:** `tests/regression/test_platinum_regression.py`

Loads from `platinum_reference.json`. Runs both gates. Includes latency benchmark.

---

## 8. ML Retraining (Follow-On, Not Core)

With 150-200 drugs, existing ML infrastructure becomes more effective:

| Component | Current N | New N | Action |
|-----------|----------|-------|--------|
| DirectCmaxPredictor | 75 | ~200 | Retrain, expect CV AAFE improvement |
| Pre-ODE corrector | 127 (mismatched) | ~200 (uniform quality) | LOO-CV retrain |
| Post-ODE corrector | 127 (mismatched) | ~200 | LOO-CV retrain |
| Adaptive conformal | ~24 calibration | ~200 | Deploy with proper calibration |
| CLint anchors | 19 | 19-30 | Add anchors for new high-confidence drugs |

**Not in scope:** Multi-task GNN, foundation model, differentiable ODE. These require 500+ drugs.

---

## 9. Migration Plan

### Phase 0: Yield Prototype (2-3 days)

- [ ] Test DailyMed XML extraction on 20 drugs
- [ ] Measure table-vs-prose Cmax distribution
- [ ] Decision gate: DailyMed primary (>= 50% table success) or regex-only fallback
- [ ] Revised drug count target

### Phase 1: FDA Extraction Upgrade (Week 1)

- [ ] Build `scripts/fetch_dailymed_pk.py` (if Phase 0 passes) or improve regex
- [ ] Extract Cmax/AUC/dose with fasted_confidence, formulation, population metadata
- [ ] Quality filter: fasted, healthy volunteers, single dose, IR
- [ ] Deduplicate: name-based (Levenshtein) + structure-based (Tanimoto)
- [ ] Yield target: ~150-170 drugs with Cmax (adjusted per Phase 0)

### Phase 2: PK-DB & Gap Fill (Week 2)

- [ ] Fix `expand_pkdb_cmax.py` (remove mock, enable real API)
- [ ] Extend MW table via PubChem API (60 -> 200 drugs)
- [ ] Compute Cmax from C(t) timecourses for 21 drugs in pkdb_timecourses.json
- [ ] Manual curation for 10-20 high-value drugs
- [ ] Yield target: 150-200 drugs total

### Phase 3: Unified Benchmark (Week 2-3)

- [ ] Extract `BENCHMARK_DRUGS` to `src/omega_pbpk/data/drug_registry.py`
- [ ] Update `test_gold24_regression.py` import path
- [ ] Create `data/clinical/platinum_reference.json` (unified schema)
- [ ] Annotate `tuning_contaminated` and `nonlinear_pk` fields
- [ ] Build `scripts/run_platinum_benchmark.py`
- [ ] Create `tests/regression/test_platinum_regression.py` (two-level gate)
- [ ] Run baseline: AAFE on full platinum set + clean subset + core-24
- [ ] Run 5-fold CV: report mean +/- std

### Phase 4: ML Retraining (Week 3-4)

- [ ] Retrain DirectCmaxPredictor on platinum set
- [ ] Retrain Pre/Post-ODE correctors with LOO-CV
- [ ] Deploy adaptive conformal with full calibration set
- [ ] Run ablation on platinum set

### Phase 5: Cleanup (Week 4)

- [ ] Update CLAUDE.md, README with platinum numbers
- [ ] Update paper if applicable
- [ ] Old scripts kept for reproducibility (not deleted)

---

## 10. Success Criteria

| Criterion | Target |
|-----------|--------|
| Platinum drug count | >= 150 (stretch: 200) |
| All drugs have: SMILES + dose + Cmax + source | 100% |
| `fasted_confidence` annotated for all drugs | 100% |
| `tuning_contaminated` annotated for all drugs | 100% |
| Core-24 AAFE | <= 1.70 (no regression) |
| Platinum AAFE (full set) | Report honestly, no pre-set target |
| Platinum AAFE (clean subset) | Report honestly |
| 5-fold CV AAFE std | < 0.5 (stability across folds) |
| Benchmark runtime | < 10 min for 200 drugs |

---

## 11. Risks & Mitigations

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| DailyMed table yield < expected | High | Fewer drugs | Phase 0 prototype measures actual yield; fallback to regex |
| FDA table parsing inaccuracy | Medium | Wrong Cmax | Automated sanity checks + 10x prediction-discrepancy flag |
| Platinum AAFE much worse than gold-24 | High | Honest but unflattering | Expected and acceptable; separate clean-subset reporting |
| < 150 drugs after quality filters | Medium | Below target | Accept `assumed_fasted` drugs (annotated), relax to 150 minimum |
| Error cancellation breaks at scale | High | Higher AAFE | Information, not a problem; reveals true limitations |
| Existing data has typos/duplicates | High | Noisy benchmark | Name-based + structure-based dedup in pipeline |
| Label selection ambiguity (multiple labels per drug) | Medium | Wrong reference | Explicit criteria: original NDA, longest PK section, oral formulation |
| PK-DB API changes/downtime | Low | Blocks Phase 2 | Cache everything, manual fallback |
| Non-linear PK drugs inflate AAFE | Medium | Misleading metric | Annotate and report separately |

---

## 12. What This Design Does NOT Include

- **Neural network / GNN / Transformer** — insufficient data at 200 drugs; XGBoost remains appropriate
- **Differentiable ODE** — premature; requires 500+ drugs with C(t) curves
- **Phase II metabolism** — mechanistic improvement, separate design needed
- **Multi-compartment Cmax model** — separate design, addresses diazepam/fluconazole
- **Automatic label NLP** — beyond regex/XML parsing; revisit at 500+ drugs
- **EMA/PMDA labels** — future expansion source, not in scope for v1
