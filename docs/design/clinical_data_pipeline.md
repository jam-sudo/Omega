# Clinical Data Pipeline Design (Branch D)

> **Status:** Design complete | **Author:** data-engineer | **Date:** 2026-03-10

## 1. Overview

The clinical data pipeline collects, harmonizes, and serves real-world PK data for
Level 3 training and benchmarking. It combines three data sources:

| Source | Data Type | Coverage | Access |
|--------|-----------|----------|--------|
| **PK-DB** | C(t) curves + PK params | 10 substances, 3148 timecourses | REST API (free) |
| **DailyMed/FDA** | PK params from labels | ~100K drug labels | REST API (free) |
| **PyTDC** | ADME benchmarks (SMILES → value) | 640-1614 compounds/endpoint | Python package |

## 2. Current State (what exists)

### Implemented (`src/omega_pbpk/ml/data/`)

| Module | Class | Status | Notes |
|--------|-------|--------|-------|
| `loaders.py` | `PKDBLoader` | **Functional** | Caching, pagination, rate-limiting |
| `loaders.py` | `FDALabelExtractor` | **Functional** | Regex extraction for 6 PK params |
| `loaders.py` | `TDCLoader` | **Partial** | Missing 4 key datasets |
| `datasets.py` | `ClinicalPKDataset` | **Functional** | Unit conversion, SMILES lookup, scaffold split |
| `datasets.py` | `_BUILTIN_SMILES` | **Partial** | Only 10 drugs, needs 20+ |

### Gaps Identified

1. **TDCLoader missing endpoints:** `VDss_Lombardo`, `Half_Life_Obach`, `Bioavailability_Ma`, `Clearance_Microsome_AZ`
2. **No C(t) curve extraction:** `PKDBLoader.get_concentration_time()` exists but `ClinicalPKDataset.from_pkdb()` only pulls scalar PK params, not timecourse curves
3. **No orchestration CLI:** No way to run the full pipeline end-to-end
4. **SMILES table incomplete:** Missing 13 new benchmark drugs
5. **No data quality checks:** No validation of extracted values against expected ranges
6. **No integration tests:** Loaders untested against live APIs

## 3. Architecture

```
┌─────────────┐    ┌──────────────────┐    ┌──────────────────┐
│   PK-DB     │───→│   PKDBLoader     │───→│                  │
│  REST API   │    │  (timecourses +  │    │                  │
│             │    │   PK params)     │    │                  │
└─────────────┘    └──────────────────┘    │                  │
                                           │  ClinicalPK      │    ┌────────────┐
┌─────────────┐    ┌──────────────────┐    │  Dataset         │───→│ Parquet /  │
│  DailyMed   │───→│ FDALabelExtract  │───→│                  │    │ CSV output │
│  REST API   │    │  (NLP/regex)     │    │  - unit norm     │    └────────────┘
└─────────────┘    └──────────────────┘    │  - SMILES match  │
                                           │  - scaffold split│    ┌────────────┐
┌─────────────┐    ┌──────────────────┐    │  - QC filters    │───→│ ML training│
│   PyTDC     │───→│   TDCLoader      │───→│                  │    │ (Level 3)  │
│  (pip pkg)  │    │  (ADME datasets) │    │                  │    └────────────┘
└─────────────┘    └──────────────────┘    └──────────────────┘
```

## 4. Implementation Plan

### Phase D.1: Fix TDCLoader (1 task, ~30 min)

Add missing TDC endpoints to `_SUPPORTED_TDC_ENDPOINTS`:

```python
_SUPPORTED_TDC_ENDPOINTS = {
    "Caco2_Wang": "Caco2_Wang",
    "Lipophilicity_AstraZeneca": "Lipophilicity_AstraZeneca",
    "Solubility_AqSolDB": "Solubility_AqSolDB",
    "PPBR_AZ": "PPBR_AZ",
    "Clearance_Hepatocyte_AZ": "Clearance_Hepatocyte_AZ",
    "Clearance_Microsome_AZ": "Clearance_Microsome_AZ",  # NEW
    "Half_Life_Obach": "Half_Life_Obach",                 # NEW
    "VDss_Lombardo": "VDss_Lombardo",                     # NEW
    "Bioavailability_Ma": "Bioavailability_Ma",            # NEW
    "hERG": "hERG",
}
```

### Phase D.2: C(t) Curve Extraction from PK-DB (~1h)

PK-DB has 3,148 timecourses across 10 substances. Add a method to extract these as
DataFrames with standardized columns:

```python
class PKDBLoader:
    def get_timecourses_for_substance(self, drug_name: str) -> list[dict]:
        """Get all timecourse data for a substance.

        Returns list of dicts with keys:
        - study_id, group_id, substance, route, dose_mg
        - timepoints: list of (time_h, conc_mg_L, std_mg_L)
        """
```

Update `ClinicalPKDataset` to store timecourse data alongside scalar params.

### Phase D.3: Expand SMILES Lookup (~30 min)

Add all 20 benchmark drugs to `_BUILTIN_SMILES` in `datasets.py`:

```python
# Missing drugs to add:
"propranolol": "CC(C)NCC(O)c1cccc2ccccc12",
"metoprolol": "COCCCN(CC)CC(O)c1ccc(OC)cc1",  # simplified
"amphetamine": "CC(N)Cc1ccccc1",
"omeprazole": "COc1ccc2[nH]c(S(=O)Cc3ncc(C)c(OC)c3C)nc2c1",
"amoxicillin": "CC1(C)SC2C(NC(=O)C(N)c3ccc(O)cc3)C(=O)N2C1C(O)=O",
"atorvastatin": "CC(C)c1n(CC[C@@H](O)C[C@@H](O)CC(O)=O)c(c2ccc(F)cc2)c(c1c1ccccc1)C(=O)Nc1ccccc1",
"fluoxetine": "CNCCC(Oc1ccc(C(F)(F)F)cc1)c1ccccc1",
"carbamazepine": "NC(=O)N1c2ccccc2C=Cc2ccccc21",
"phenytoin": "O=C1NC(=O)C(c2ccccc2)(c2ccccc2)N1",
"verapamil": "COc1ccc(CCN(C)CCCC(C#N)(c2ccc(OC)c(OC)c2)C(C)C)cc1OC",
"nifedipine": "COC(=O)C1=C(C)NC(C)=C(C(=O)OC)C1c1ccccc1[N+]([O-])=O",
"digoxin": "<complex glycoside SMILES>",  # needs careful lookup
```

### Phase D.4: Data Quality Layer (~1h)

Add validation ranges for extracted PK parameters:

```python
_PK_EXPECTED_RANGES = {
    "cmax": (0.0001, 1000.0, "mg/L"),    # ng/mL to g/L range
    "auc": (0.001, 10000.0, "mg*h/L"),
    "t_half": (0.05, 1000.0, "h"),        # 3 min to 42 days
    "bioavailability": (0.0, 1.0, "fraction"),
    "vd": (0.01, 50000.0, "L"),           # 10 mL to 50K L
    "clearance": (0.001, 1000.0, "L/h"),
}
```

Records outside these ranges get flagged (not removed) for manual review.

### Phase D.5: Pipeline Orchestration (~1h)

Create `src/omega_pbpk/ml/data/pipeline.py`:

```python
def run_clinical_data_pipeline(
    output_dir: Path = Path("data/ml/clinical"),
    use_pkdb: bool = True,
    use_fda: bool = True,
    use_tdc: bool = True,
    drug_list: list[str] | None = None,
    seed: int = 42,
) -> ClinicalPKDataset:
    """Run the full clinical data collection pipeline.

    Steps:
    1. Collect from PK-DB (timecourses + PK params)
    2. Collect from DailyMed (PK params from labels)
    3. Collect from TDC (ADME benchmarks)
    4. Harmonize units
    5. Match SMILES
    6. QC filter
    7. Scaffold split
    8. Save to parquet
    """
```

### Phase D.6: Tests (~1h)

Create `tests/ml/test_data_pipeline.py`:
- Unit tests for unit conversion functions
- Unit tests for SMILES lookup
- Integration tests (mocked HTTP) for PKDBLoader
- Integration tests (mocked HTTP) for FDALabelExtractor
- End-to-end test for ClinicalPKDataset

## 5. PK-DB API Reference

**Base URL:** `https://pk-db.com/api/v1/`

| Endpoint | Purpose | Key Params |
|----------|---------|------------|
| `studies/` | List all studies | `page` |
| `pkdata/` | PK output parameters | `substance` |
| `timecourses/` | C(t) curve data | `study` |
| `statistics/` | DB overview/counts | — |
| `filter/` | Advanced search → UUID | `studies__*`, `groups__*` |

**Current DB contents (v0.9.3):** 10 substances, 512 studies, 3148 timecourses

**Relevant substances in PK-DB:**
acetaminophen, caffeine, diazepam, midazolam (overlap with our 20 benchmarks)

## 6. DailyMed API Reference

**Base URL:** `https://dailymed.nlm.nih.gov/dailymed/services/v2/`

| Endpoint | Purpose | Key Params |
|----------|---------|------------|
| `spls.json` | Search drug labels | `drug_name` |
| `spls/{setid}.json` | Get full label | — |

**PK Section LOINC codes:** `34090-1` (Clinical Pharmacology), `43682-4` (Pharmacokinetics)

**Regex extraction targets:** Cmax, AUC, t½, bioavailability, Vd, clearance

## 7. TDC Datasets for Omega

| Dataset | Task | N | Units | Code |
|---------|------|---|-------|------|
| `Caco2_Wang` | Regression | 906 | cm/s | `ADME(name='Caco2_Wang')` |
| `PPBR_AZ` | Regression | 1,614 | % bound | `ADME(name='PPBR_AZ')` |
| `Clearance_Hepatocyte_AZ` | Regression | 1,102 | µL/min/10^6 cells | `ADME(name='Clearance_Hepatocyte_AZ')` |
| `Clearance_Microsome_AZ` | Regression | 1,020 | µL/min/mg | `ADME(name='Clearance_Microsome_AZ')` |
| `Half_Life_Obach` | Regression | 667 | h | `ADME(name='Half_Life_Obach')` |
| `VDss_Lombardo` | Regression | 1,130 | L/kg | `ADME(name='VDss_Lombardo')` |
| `Bioavailability_Ma` | Classification | 640 | binary | `ADME(name='Bioavailability_Ma')` |
| `Lipophilicity_AstraZeneca` | Regression | 4,200 | logD | `ADME(name='Lipophilicity_AstraZeneca')` |
| `Solubility_AqSolDB` | Regression | 9,982 | logS | `ADME(name='Solubility_AqSolDB')` |
| `hERG` | Classification | 648 | binary | `ADME(name='hERG')` |

## 8. Dependencies

```
# Already in project or planned
requests>=2.28
pandas>=1.5
PyTDC>=0.4.0        # for TDCLoader
rdkit>=2022.09      # for SMILES/scaffold
pyarrow>=10.0       # for parquet (optional, CSV fallback)
```

## 9. Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| PK-DB API down/changed | No timecourse data | JSON cache persists; check at startup |
| DailyMed label format changes | Regex extraction breaks | Version-pin patterns; manual review flag |
| TDC dataset deprecated | Missing ADME training data | Pin version; local copy fallback |
| Only 10 PK-DB substances | Limited C(t) coverage | Supplement with benchmark CSVs (Task #6) |
| Regex misses PK values | Incomplete FDA extraction | QC ranges flag suspicious gaps |

## 10. Priority & Sequencing

```
D.1 (TDC fix)     ─┐
D.3 (SMILES)       ├──→ D.5 (orchestration) ──→ D.6 (tests)
D.4 (QC)           ┘
D.2 (C(t) extract) ─────→ D.5
```

**Blocked by:** Task #1 (dev env setup) for live API testing and PyTDC installation.
**Blocks:** Level 3 training (needs real clinical data).
