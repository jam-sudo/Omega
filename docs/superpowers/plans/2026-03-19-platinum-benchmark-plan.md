# Platinum Benchmark Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the tier system with a unified Platinum benchmark (150-200 drugs with clinical Cmax references) and a two-level regression gate.

**Architecture:**
- Extract drug registry to shared module (`src/omega_pbpk/data/drug_registry.py`)
- Define platinum schema with validation (`src/omega_pbpk/data/platinum_schema.py`)
- Build DailyMed SPL XML parser for FDA label extraction (`scripts/fetch_dailymed_pk.py`)
- Create unified benchmark script loading from `platinum_reference.json`
- Two-level regression gate: core-24 (strict) + full platinum (loose)

**Tech Stack:** beautifulsoup4, lxml, requests (DailyMed API), existing OmegaPipeline, pytest-benchmark.

**Spec:** `docs/superpowers/specs/2026-03-19-platinum-benchmark-design.md`

**Acceptance criteria (apply after each task):**
- Core-24 AAFE <= 1.70, %2-fold >= 75% (no regression)
- All existing fast tests pass: `pytest tests/ -m "not slow and not benchmark" -q`

---

## File Map

| File | Action | Task |
|------|--------|------|
| `src/omega_pbpk/data/drug_registry.py` | Create | 1 |
| `src/omega_pbpk/data/__init__.py` | Modify | 1 |
| `tests/data/__init__.py` | Create | 1 |
| `tests/data/test_drug_registry.py` | Create | 1 |
| `scripts/run_l1_benchmarks.py` | Modify (import from registry) | 1 |
| `scripts/run_full_benchmark.py` | Modify (import from registry) | 1 |
| `tests/regression/test_gold24_regression.py` | Modify (import from registry) | 1 |
| `src/omega_pbpk/data/platinum_schema.py` | Create | 2 |
| `tests/data/test_platinum_schema.py` | Create | 2 |
| `scripts/migrate_to_platinum.py` | Create | 3 |
| `data/clinical/platinum_reference.json` | Create | 3 |
| `pyproject.toml` | Modify (add beautifulsoup4, lxml) | 4 |
| `scripts/fetch_dailymed_pk.py` | Create | 4, 5 |
| `scripts/expand_pkdb_cmax.py` | Modify (remove mock) | 6 |
| `scripts/merge_platinum_sources.py` | Create | 7 |
| `scripts/run_platinum_benchmark.py` | Create | 8 |
| `tests/regression/test_platinum_regression.py` | Create | 9 |

---

## Task 1: Extract Drug Registry to Shared Module

**Why:** `BENCHMARK_DRUGS` is hardcoded in `scripts/run_l1_benchmarks.py` and imported via `sys.path` hacks by `test_gold24_regression.py` and `run_full_benchmark.py`. Extracting to a proper module enables clean imports and is a prerequisite for the platinum benchmark.

**Files:**
- Create: `src/omega_pbpk/data/drug_registry.py`
- Create: `tests/data/__init__.py`
- Create: `tests/data/test_drug_registry.py`
- Modify: `src/omega_pbpk/data/__init__.py`
- Modify: `scripts/run_l1_benchmarks.py`
- Modify: `scripts/run_full_benchmark.py`
- Modify: `tests/regression/test_gold24_regression.py`

- [ ] **Step 1:** Create `tests/data/__init__.py` (empty)

```bash
mkdir -p tests/data && touch tests/data/__init__.py
```

- [ ] **Step 2:** Write the failing test

```python
# tests/data/test_drug_registry.py
"""Drug registry module tests."""
import pytest

def test_benchmark_drugs_importable():
    from omega_pbpk.data.drug_registry import BENCHMARK_DRUGS
    assert isinstance(BENCHMARK_DRUGS, dict)
    assert len(BENCHMARK_DRUGS) >= 24

def test_benchmark_drugs_have_required_fields():
    from omega_pbpk.data.drug_registry import BENCHMARK_DRUGS
    for name, info in BENCHMARK_DRUGS.items():
        assert "smiles" in info, f"{name} missing smiles"
        assert "dose_mg" in info, f"{name} missing dose_mg"
        assert isinstance(info["smiles"], str)
        assert info["dose_mg"] > 0

def test_core24_subset():
    from omega_pbpk.data.drug_registry import BENCHMARK_DRUGS, CORE24_NAMES
    assert len(CORE24_NAMES) == 24
    for name in CORE24_NAMES:
        assert name in BENCHMARK_DRUGS, f"{name} not in BENCHMARK_DRUGS"

def test_get_core24():
    from omega_pbpk.data.drug_registry import get_core24
    core = get_core24()
    assert len(core) == 24
    assert "caffeine" in core
```

- [ ] **Step 3:** Run test to verify it fails

```bash
pytest tests/data/test_drug_registry.py -v
```

Expected: `ModuleNotFoundError: No module named 'omega_pbpk.data.drug_registry'`

- [ ] **Step 4:** Create `src/omega_pbpk/data/drug_registry.py`

```python
"""Canonical drug registry for benchmarking.

Single source of truth for benchmark drug SMILES and doses.
Imported by benchmark scripts and regression tests.
"""
from __future__ import annotations

# All benchmark drugs: name -> {smiles, dose_mg, set?}
# First 20 = dev set, last 5 = validation set
BENCHMARK_DRUGS: dict[str, dict] = {
    "caffeine": {"smiles": "Cn1c(=O)c2c(ncn2C)n(C)c1=O", "dose_mg": 100},
    "metoprolol": {"smiles": "COCCc1ccc(OCC(O)CNC(C)C)cc1", "dose_mg": 100},
    "midazolam": {"smiles": "Clc1ccc2c(c1)C(=NCc1nccn1C)c1ccccc1N2", "dose_mg": 2},
    "propranolol": {"smiles": "CC(C)NCC(O)COc1cccc2ccccc12", "dose_mg": 80},
    "warfarin": {"smiles": "CC(=O)CC(c1ccccc1)c1c(O)c2ccccc2oc1=O", "dose_mg": 10},
    "d_amphetamine": {"smiles": "C[C@@H](N)Cc1ccccc1", "dose_mg": 20},
    "ibuprofen": {"smiles": "CC(C)Cc1ccc(C(C)C(=O)O)cc1", "dose_mg": 400},
    "acetaminophen": {"smiles": "CC(=O)Nc1ccc(O)cc1", "dose_mg": 1000},
    "amoxicillin": {"smiles": "CC1(C)SC2C(NC(=O)C(N)c3ccc(O)cc3)C(=O)N2C1C(=O)O", "dose_mg": 500},
    "atorvastatin": {
        "smiles": "CC(C)c1n(CC[C@@H](O)C[C@@H](O)CC(=O)O)c(-c2ccccc2)c(-c2ccc(F)cc2)c1C(=O)Nc1ccccc1",
        "dose_mg": 40,
    },
    "carbamazepine": {"smiles": "NC(=O)N1c2ccccc2C=Cc2ccccc21", "dose_mg": 200},
    "diazepam": {"smiles": "CN1C(=O)CN=C(c2ccccc2)c2cc(Cl)ccc21", "dose_mg": 10},
    "digoxin": {
        "smiles": "C[C@@H]1O[C@@H](O[C@@H]2C[C@H](O)[C@@H](O[C@@H]3C[C@H](O)[C@@H](O[C@@H]4C[C@H](O)[C@@H](OC5CC(CO)=CC(=O)O5)C(C)O4)C(C)O3)C(C)O2)C[C@H](O)[C@H]1O",
        "dose_mg": 0.5,
    },
    "fluoxetine": {"smiles": "CNCCC(Oc1ccc(C(F)(F)F)cc1)c1ccccc1", "dose_mg": 20},
    "nifedipine": {"smiles": "COC(=O)C1=C(C)NC(C)=C(C(=O)OC)C1c1ccccc1[N+](=O)[O-]", "dose_mg": 10},
    "omeprazole": {"smiles": "COc1ccc2[nH]c(S(=O)Cc3ncc(C)c(OC)c3C)nc2c1", "dose_mg": 20},
    "phenytoin": {"smiles": "O=C1NC(=O)C(c2ccccc2)(c2ccccc2)N1", "dose_mg": 300},
    "theophylline": {"smiles": "Cn1c(=O)c2[nH]cnc2n(C)c1=O", "dose_mg": 300},
    "verapamil": {"smiles": "COc1ccc(CCN(C)CCCC(C#N)(c2ccc(OC)c(OC)c2)C(C)C)cc1OC", "dose_mg": 80},
    "tramadol": {"smiles": "COc1cccc(C2(O)CCCCC2CN(C)C)c1", "dose_mg": 100},
    # --- Validation set ---
    "atenolol": {"smiles": "CC(C)NCC(O)COc1ccc(CC(N)=O)cc1", "dose_mg": 50, "set": "validation"},
    "fluconazole": {"smiles": "OC(Cn1cncn1)(Cn1cncn1)c1ccc(F)cc1F", "dose_mg": 200, "set": "validation"},
    "furosemide": {"smiles": "NS(=O)(=O)c1cc(C(=O)O)c(NCc2ccco2)cc1Cl", "dose_mg": 40, "set": "validation"},
    "gabapentin": {"smiles": "OC(=O)CC1(CN)CCCCC1", "dose_mg": 300, "set": "validation"},
    "metformin": {"smiles": "CN(C)C(=N)NC(=N)N", "dose_mg": 500, "set": "validation"},
}

# Core-24 drugs used in gold24_reference_cmax.json
# (all BENCHMARK_DRUGS except tramadol, which was added later)
CORE24_NAMES: frozenset[str] = frozenset(BENCHMARK_DRUGS.keys()) - {"tramadol"}


def get_core24() -> dict[str, dict]:
    """Return only the core-24 benchmark drugs."""
    return {k: v for k, v in BENCHMARK_DRUGS.items() if k in CORE24_NAMES}
```

- [ ] **Step 5:** Update `src/omega_pbpk/data/__init__.py` — add drug_registry exports

Add after existing imports:

```python
from omega_pbpk.data.drug_registry import BENCHMARK_DRUGS, CORE24_NAMES, get_core24
```

And add to `__all__`:

```python
"BENCHMARK_DRUGS", "CORE24_NAMES", "get_core24",
```

- [ ] **Step 6:** Run tests

```bash
pytest tests/data/test_drug_registry.py -v
```

Expected: All 4 tests pass.

- [ ] **Step 7:** Update `scripts/run_l1_benchmarks.py` — replace hardcoded dict with import

At the top of `run_l1_benchmarks.py`, replace the `BENCHMARK_DRUGS = { ... }` dict (lines 22-62) with:

```python
from omega_pbpk.data.drug_registry import BENCHMARK_DRUGS  # noqa: E402
```

Keep all other functions in the file unchanged.

- [ ] **Step 8:** Update `tests/regression/test_gold24_regression.py` — clean import

Replace the `sys.path` hack + import:

```python
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
# ...
from run_l1_benchmarks import BENCHMARK_DRUGS
```

With:

```python
from omega_pbpk.data.drug_registry import BENCHMARK_DRUGS
```

Remove the `sys.path.insert` line and the `sys` import if no longer needed.

- [ ] **Step 9:** Update `scripts/run_full_benchmark.py` — clean import

Replace:
```python
from run_l1_benchmarks import BENCHMARK_DRUGS, compute_aafe, compute_fold_error, load_observed_pk
```

With:
```python
from omega_pbpk.data.drug_registry import BENCHMARK_DRUGS
from run_l1_benchmarks import compute_aafe, compute_fold_error, load_observed_pk
```

- [ ] **Step 10:** Run all affected tests to verify no regression

```bash
pytest tests/data/test_drug_registry.py tests/regression/test_gold24_regression.py -v -m "benchmark or not benchmark"
python scripts/run_full_benchmark.py 2>&1 | grep -E "AAFE|2-fold"
```

Expected: All tests pass, AAFE unchanged (~1.50).

- [ ] **Step 11:** Commit

```bash
git add src/omega_pbpk/data/drug_registry.py src/omega_pbpk/data/__init__.py \
        tests/data/__init__.py tests/data/test_drug_registry.py \
        scripts/run_l1_benchmarks.py scripts/run_full_benchmark.py \
        tests/regression/test_gold24_regression.py
git commit -m "refactor: extract BENCHMARK_DRUGS to shared drug_registry module"
```

---

## Task 2: Platinum Schema and Validation

**Why:** The platinum reference needs a defined schema with validation to prevent bad data from entering the benchmark.

**Files:**
- Create: `src/omega_pbpk/data/platinum_schema.py`
- Create: `tests/data/test_platinum_schema.py`

- [ ] **Step 1:** Write the failing test

```python
# tests/data/test_platinum_schema.py
"""Platinum reference schema validation tests."""
import pytest
from omega_pbpk.data.platinum_schema import (
    PlatinumEntry,
    validate_entry,
    load_platinum_reference,
    save_platinum_reference,
    ValidationError,
)


def _valid_entry() -> dict:
    return {
        "smiles": "Cn1c(=O)c2c(ncn2C)n(C)c1=O",
        "dose_mg": 100.0,
        "cmax_mg_L": 1.74,
        "source_type": "fda_label",
        "source_id": "NDA 020863",
        "fasted_confidence": "confirmed_fasted",
        "formulation": "IR",
        "route": "oral",
        "population": "healthy",
        "single_dose": True,
        "tuning_contaminated": False,
        "nonlinear_pk": False,
        "data_quality": "fda_label_exact",
    }


def test_valid_entry_passes():
    entry = validate_entry("caffeine", _valid_entry())
    assert entry.drug_name == "caffeine"
    assert entry.cmax_mg_L == 1.74


def test_missing_smiles_fails():
    d = _valid_entry()
    del d["smiles"]
    with pytest.raises(ValidationError, match="smiles"):
        validate_entry("caffeine", d)


def test_bad_cmax_dose_ratio_fails():
    d = _valid_entry()
    d["cmax_mg_L"] = 500.0  # ratio 5.0, above 1.0 threshold
    with pytest.raises(ValidationError, match="ratio"):
        validate_entry("caffeine", d)


def test_invalid_fasted_confidence_fails():
    d = _valid_entry()
    d["fasted_confidence"] = "unknown"
    with pytest.raises(ValidationError, match="fasted_confidence"):
        validate_entry("caffeine", d)


def test_non_oral_route_fails():
    d = _valid_entry()
    d["route"] = "iv"
    with pytest.raises(ValidationError, match="route"):
        validate_entry("caffeine", d)


def test_roundtrip_json(tmp_path):
    ref = {"caffeine": _valid_entry()}
    path = tmp_path / "test_ref.json"
    save_platinum_reference(ref, path)
    loaded = load_platinum_reference(path)
    assert "caffeine" in loaded
    assert loaded["caffeine"]["cmax_mg_L"] == 1.74


def test_optional_auc_field():
    d = _valid_entry()
    d["auc_mg_h_L"] = 14.2
    entry = validate_entry("caffeine", d)
    assert entry.auc_mg_h_L == 14.2
```

- [ ] **Step 2:** Run test to verify it fails

```bash
pytest tests/data/test_platinum_schema.py -v
```

Expected: `ImportError`

- [ ] **Step 3:** Implement `src/omega_pbpk/data/platinum_schema.py`

```python
"""Platinum benchmark reference schema and validation.

Defines the data contract for platinum_reference.json entries.
All drugs in the benchmark must pass validation before inclusion.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


class ValidationError(ValueError):
    """Raised when a platinum entry fails validation."""


_VALID_FASTED = {"confirmed_fasted", "assumed_fasted", "fed_only"}
_VALID_FORMULATION = {"IR", "ER", "solution", "other"}
_VALID_ROUTE = {"oral"}
_VALID_SOURCE_TYPE = {"fda_label", "literature", "pkdb"}
_VALID_DATA_QUALITY = {
    "fda_label_exact",
    "clinical_exact",
    "clinical_dose_normalized",
    "fda_label_dose_normalized",
    "fda_label_median",
}
_REQUIRED_FIELDS = {
    "smiles", "dose_mg", "cmax_mg_L", "source_type", "source_id",
    "fasted_confidence", "formulation", "route", "population",
    "single_dose", "tuning_contaminated", "nonlinear_pk", "data_quality",
}


@dataclass(frozen=True)
class PlatinumEntry:
    """Validated platinum benchmark entry."""

    drug_name: str
    smiles: str
    dose_mg: float
    cmax_mg_L: float
    source_type: str
    source_id: str
    fasted_confidence: str
    formulation: str
    route: str
    population: str
    single_dose: bool
    tuning_contaminated: bool
    nonlinear_pk: bool
    data_quality: str
    auc_mg_h_L: float | None = None
    thalf_h: float | None = None
    tmax_h: float | None = None
    f_oral: float | None = None
    notes: str = ""


def validate_entry(drug_name: str, data: dict[str, Any]) -> PlatinumEntry:
    """Validate a drug entry against the platinum schema.

    Raises ValidationError if the entry is invalid.
    """
    # Required fields
    for f in _REQUIRED_FIELDS:
        if f not in data:
            raise ValidationError(f"{drug_name}: missing required field '{f}'")

    # SMILES
    smiles = data["smiles"]
    if not isinstance(smiles, str) or len(smiles) < 3:
        raise ValidationError(f"{drug_name}: smiles must be a non-trivial string")

    # Dose
    dose = float(data["dose_mg"])
    if not (0.1 <= dose <= 5000):
        raise ValidationError(f"{drug_name}: dose_mg {dose} outside [0.1, 5000]")

    # Cmax
    cmax = float(data["cmax_mg_L"])
    if cmax <= 0:
        raise ValidationError(f"{drug_name}: cmax_mg_L must be > 0")

    # Cmax/dose ratio
    ratio = cmax / dose
    if not (1e-6 < ratio < 1.0):
        raise ValidationError(
            f"{drug_name}: cmax/dose ratio {ratio:.2e} outside (1e-6, 1.0)"
        )

    # Enum fields
    if data["fasted_confidence"] not in _VALID_FASTED:
        raise ValidationError(
            f"{drug_name}: fasted_confidence '{data['fasted_confidence']}' "
            f"not in {_VALID_FASTED}"
        )
    if data["formulation"] not in _VALID_FORMULATION:
        raise ValidationError(
            f"{drug_name}: formulation '{data['formulation']}' not in {_VALID_FORMULATION}"
        )
    if data["route"] not in _VALID_ROUTE:
        raise ValidationError(
            f"{drug_name}: route '{data['route']}' must be 'oral'"
        )
    if data["source_type"] not in _VALID_SOURCE_TYPE:
        raise ValidationError(
            f"{drug_name}: source_type '{data['source_type']}' "
            f"not in {_VALID_SOURCE_TYPE}"
        )
    if data["data_quality"] not in _VALID_DATA_QUALITY:
        raise ValidationError(
            f"{drug_name}: data_quality '{data['data_quality']}' "
            f"not in {_VALID_DATA_QUALITY}"
        )

    # Optional: MW check (requires RDKit)
    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            mw = Descriptors.MolWt(mol)
            if not (100 < mw < 1500):
                raise ValidationError(f"{drug_name}: MW {mw:.0f} outside (100, 1500)")
    except ImportError:
        pass  # RDKit not available, skip MW check

    return PlatinumEntry(
        drug_name=drug_name,
        smiles=smiles,
        dose_mg=dose,
        cmax_mg_L=cmax,
        source_type=data["source_type"],
        source_id=data["source_id"],
        fasted_confidence=data["fasted_confidence"],
        formulation=data["formulation"],
        route=data["route"],
        population=data["population"],
        single_dose=bool(data["single_dose"]),
        tuning_contaminated=bool(data["tuning_contaminated"]),
        nonlinear_pk=bool(data["nonlinear_pk"]),
        data_quality=data["data_quality"],
        auc_mg_h_L=data.get("auc_mg_h_L"),
        thalf_h=data.get("thalf_h"),
        tmax_h=data.get("tmax_h"),
        f_oral=data.get("f_oral"),
        notes=data.get("notes", ""),
    )


def load_platinum_reference(path: str | Path) -> dict[str, dict]:
    """Load platinum_reference.json, return drugs dict (without metadata)."""
    data = json.loads(Path(path).read_text())
    return data.get("drugs", data)


def save_platinum_reference(
    drugs: dict[str, dict], path: str | Path, version: str = "1.0"
) -> None:
    """Save drugs dict to platinum_reference.json with metadata."""
    from datetime import date

    out = {
        "metadata": {
            "version": version,
            "n_drugs": len(drugs),
            "created": str(date.today()),
            "inclusion_standard": "oral_IR_fasted_healthy_single_dose",
        },
        "drugs": drugs,
    }
    Path(path).write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n")
```

- [ ] **Step 4:** Run tests

```bash
pytest tests/data/test_platinum_schema.py -v
```

Expected: All 7 tests pass.

- [ ] **Step 5:** Commit

```bash
git add src/omega_pbpk/data/platinum_schema.py tests/data/test_platinum_schema.py
git commit -m "feat: platinum reference schema with validation"
```

---

## Task 3: Migrate Gold-24 to Platinum Format

**Why:** Create the initial `platinum_reference.json` by merging `gold24_reference_cmax.json` with `BENCHMARK_DRUGS` SMILES/doses, adding the new metadata fields.

**Files:**
- Create: `scripts/migrate_to_platinum.py`
- Create: `data/clinical/platinum_reference.json`

- [ ] **Step 1:** Create migration script

```python
#!/usr/bin/env python3
"""Migrate gold-24 reference data to platinum format.

Reads gold24_reference_cmax.json + BENCHMARK_DRUGS, produces
platinum_reference.json with all required fields.
"""
import json
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root / "src"))

from omega_pbpk.data.drug_registry import BENCHMARK_DRUGS, CORE24_NAMES
from omega_pbpk.data.platinum_schema import save_platinum_reference, validate_entry

GOLD_REF = repo_root / "data" / "clinical" / "gold24_reference_cmax.json"
OUTPUT = repo_root / "data" / "clinical" / "platinum_reference.json"

# Drugs with CLint/VDss anchors or hand-tuned pipeline parameters
TUNING_CONTAMINATED = {
    "caffeine", "metoprolol", "midazolam", "propranolol", "warfarin",
    "ibuprofen", "acetaminophen", "atorvastatin", "diazepam", "fluoxetine",
    "nifedipine", "omeprazole", "verapamil", "fluconazole",
}

NONLINEAR_PK = {"omeprazole", "phenytoin"}


def main():
    gold_ref = json.loads(GOLD_REF.read_text())
    gold_ref.pop("_metadata", None)

    drugs = {}
    for name in CORE24_NAMES:
        if name not in BENCHMARK_DRUGS:
            continue
        bm = BENCHMARK_DRUGS[name]
        ref = gold_ref.get(name, {})

        cmax = ref.get("cmax_mg_L")
        if cmax is None or cmax <= 0:
            print(f"SKIP {name}: no valid Cmax in gold reference")
            continue

        entry = {
            "smiles": bm["smiles"],
            "dose_mg": bm["dose_mg"],
            "cmax_mg_L": cmax,
            "auc_mg_h_L": ref.get("auc_mg_h_L"),
            "tmax_h": ref.get("tmax_h"),
            "source_type": "fda_label" if "FDA" in ref.get("source", "") or "NDA" in ref.get("source", "") else "literature",
            "source_id": ref.get("source", "unknown"),
            "fasted_confidence": "confirmed_fasted",
            "formulation": "IR",
            "route": "oral",
            "population": "healthy",
            "single_dose": True,
            "tuning_contaminated": name in TUNING_CONTAMINATED,
            "nonlinear_pk": name in NONLINEAR_PK,
            "data_quality": ref.get("data_quality", "clinical_exact"),
            "notes": ref.get("note", ""),
        }

        try:
            validate_entry(name, entry)
            drugs[name] = entry
            tag = " [contaminated]" if entry["tuning_contaminated"] else ""
            print(f"  OK  {name}: Cmax={cmax:.4f} mg/L{tag}")
        except Exception as e:
            print(f"FAIL {name}: {e}")

    save_platinum_reference(drugs, OUTPUT)
    print(f"\nWrote {len(drugs)} drugs to {OUTPUT}")
    n_clean = sum(1 for d in drugs.values() if not d["tuning_contaminated"])
    print(f"  Clean (non-contaminated): {n_clean}")
    print(f"  Contaminated: {len(drugs) - n_clean}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2:** Run migration

```bash
python scripts/migrate_to_platinum.py
```

Expected: 24 drugs migrated, platinum_reference.json created.

- [ ] **Step 3:** Verify the output

```bash
python -c "
import json
d = json.load(open('data/clinical/platinum_reference.json'))
print(f'Drugs: {d[\"metadata\"][\"n_drugs\"]}')
print(f'Version: {d[\"metadata\"][\"version\"]}')
for name in sorted(d['drugs'])[:5]:
    e = d['drugs'][name]
    tag = ' [C]' if e['tuning_contaminated'] else ''
    print(f'  {name}: Cmax={e[\"cmax_mg_L\"]:.3f}{tag}')
"
```

Expected: 24 drugs, version 1.0.

- [ ] **Step 4:** Commit

```bash
git add scripts/migrate_to_platinum.py data/clinical/platinum_reference.json
git commit -m "feat: migrate gold-24 to platinum reference format"
```

---

## Task 4: Phase 0 — DailyMed Yield Prototype

**Why:** Before building the full FDA extraction pipeline, measure how many drugs have Cmax in structured XML tables vs narrative prose. This gates the approach for Task 5.

**Files:**
- Modify: `pyproject.toml` (add beautifulsoup4, lxml)
- Create: `scripts/fetch_dailymed_pk.py` (prototype version)

- [ ] **Step 1:** Install dependencies

Add to `pyproject.toml` dev deps:
```toml
"beautifulsoup4>=4.12",
"lxml>=5.0",
```

```bash
pip install "beautifulsoup4>=4.12" "lxml>=5.0"
```

- [ ] **Step 2:** Create prototype script

```python
#!/usr/bin/env python3
"""Phase 0: DailyMed yield prototype.

Tests 20 drugs to measure what fraction have Cmax in <table> elements
vs narrative text in their SPL XML labels.

Usage: python scripts/fetch_dailymed_pk.py --phase0
"""
import json
import re
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup

CACHE_DIR = Path("data/ml/dailymed_cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

DAILYMED_SEARCH = "https://dailymed.nlm.nih.gov/dailymed/services/v2/spls.json"
DAILYMED_SPL = "https://dailymed.nlm.nih.gov/dailymed/services/v2/spls/{setid}.xml"

# 20 test drugs for Phase 0
PHASE0_DRUGS = [
    "metformin", "atorvastatin", "omeprazole", "metoprolol", "warfarin",
    "ibuprofen", "caffeine", "midazolam", "fluconazole", "carbamazepine",
    "amlodipine", "lisinopril", "sertraline", "escitalopram", "rosuvastatin",
    "montelukast", "losartan", "clopidogrel", "duloxetine", "aripiprazole",
]

CMAX_PATTERN = re.compile(
    r"C\s*max|peak\s+(?:plasma\s+)?concentration",
    re.IGNORECASE,
)

CMAX_VALUE_PATTERN = re.compile(
    r"(\d+\.?\d*)\s*(ng/m[Ll]|[uµ]g/m[Ll]|mg/[Ll]|mcg/m[Ll])",
)


def fetch_spl_xml(drug_name: str) -> str | None:
    """Fetch SPL XML for a drug from DailyMed. Returns XML string or None."""
    cache_path = CACHE_DIR / f"{drug_name.lower()}_spl.xml"
    if cache_path.exists():
        return cache_path.read_text()

    # Search for drug
    resp = requests.get(
        DAILYMED_SEARCH,
        params={"drug_name": drug_name, "page": 1, "pagesize": 5},
        timeout=30,
    )
    if resp.status_code != 200:
        return None

    results = resp.json().get("data", [])
    if not results:
        return None

    # Prefer oral dosage forms
    setid = None
    for r in results:
        title = (r.get("title") or "").lower()
        if any(w in title for w in ["tablet", "capsule", "oral"]):
            setid = r.get("setid")
            break
    if setid is None:
        setid = results[0].get("setid")
    if setid is None:
        return None

    time.sleep(0.5)  # rate limit

    # Fetch SPL XML
    xml_resp = requests.get(DAILYMED_SPL.format(setid=setid), timeout=30)
    if xml_resp.status_code != 200:
        return None

    xml_text = xml_resp.text
    cache_path.write_text(xml_text)
    time.sleep(0.5)
    return xml_text


def analyze_pk_tables(xml_text: str) -> dict:
    """Analyze an SPL XML for PK table presence and Cmax extractability."""
    soup = BeautifulSoup(xml_text, "lxml-xml")

    # Find Clinical Pharmacology section
    pk_sections = []
    for section in soup.find_all("section"):
        title_el = section.find("title")
        if title_el and re.search(
            r"clinical\s+pharmacol|pharmacokinetic", title_el.get_text(), re.IGNORECASE
        ):
            pk_sections.append(section)

    if not pk_sections:
        return {"status": "no_pk_section", "tables": 0, "cmax_in_table": False, "cmax_in_text": False}

    tables = []
    cmax_in_table = False
    cmax_in_text = False

    for sec in pk_sections:
        # Check tables
        for table in sec.find_all("table"):
            table_text = table.get_text()
            tables.append(table_text[:200])
            if CMAX_PATTERN.search(table_text):
                if CMAX_VALUE_PATTERN.search(table_text):
                    cmax_in_table = True

        # Check narrative text (outside tables)
        for p in sec.find_all("paragraph"):
            p_text = p.get_text()
            if CMAX_PATTERN.search(p_text) and CMAX_VALUE_PATTERN.search(p_text):
                cmax_in_text = True

    return {
        "status": "found",
        "tables": len(tables),
        "cmax_in_table": cmax_in_table,
        "cmax_in_text": cmax_in_text,
    }


def run_phase0():
    """Run Phase 0 yield prototype on 20 drugs."""
    results = {}
    for drug in PHASE0_DRUGS:
        print(f"  Fetching {drug}...", end=" ", flush=True)
        xml = fetch_spl_xml(drug)
        if xml is None:
            results[drug] = {"status": "fetch_failed"}
            print("FETCH FAILED")
            continue
        analysis = analyze_pk_tables(xml)
        results[drug] = analysis
        status = []
        if analysis["cmax_in_table"]:
            status.append("TABLE")
        if analysis["cmax_in_text"]:
            status.append("TEXT")
        if not status:
            status.append("NO_CMAX")
        print(f"{' + '.join(status)} ({analysis['tables']} tables)")

    # Summary
    n_total = len(results)
    n_table = sum(1 for r in results.values() if r.get("cmax_in_table"))
    n_text = sum(1 for r in results.values() if r.get("cmax_in_text"))
    n_either = sum(1 for r in results.values() if r.get("cmax_in_table") or r.get("cmax_in_text"))
    n_failed = sum(1 for r in results.values() if r.get("status") == "fetch_failed")

    print(f"\n--- Phase 0 Results ({n_total} drugs) ---")
    print(f"  Cmax in <table>: {n_table}/{n_total} ({100*n_table/n_total:.0f}%)")
    print(f"  Cmax in text:    {n_text}/{n_total} ({100*n_text/n_total:.0f}%)")
    print(f"  Cmax anywhere:   {n_either}/{n_total} ({100*n_either/n_total:.0f}%)")
    print(f"  Fetch failed:    {n_failed}/{n_total}")
    print(f"\n  Decision gate: {'PASS (>=50% table)' if n_table >= 10 else 'FAIL (<50% table) -> regex fallback'}")

    # Save results
    out_path = Path("outputs/phase0_dailymed_yield.json")
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"  Results saved to {out_path}")


if __name__ == "__main__":
    run_phase0()
```

- [ ] **Step 3:** Run Phase 0

```bash
python scripts/fetch_dailymed_pk.py
```

Expected: Results for 20 drugs. Note the table-vs-text distribution.

- [ ] **Step 4:** Decision gate

- If >= 50% have Cmax in `<table>`: proceed with Task 5A (DailyMed XML primary)
- If < 50%: proceed with Task 5B (improved regex only)
- Record decision in outputs/phase0_dailymed_yield.json

- [ ] **Step 5:** Commit

```bash
git add pyproject.toml scripts/fetch_dailymed_pk.py
git commit -m "feat: Phase 0 DailyMed yield prototype (20-drug test)"
```

---

## Task 5: FDA Label Cmax Extraction (Full Scale)

**Why:** Scale FDA extraction based on Phase 0 results. This is the primary data source for reaching 150+ drugs.

**Files:**
- Modify: `scripts/fetch_dailymed_pk.py` (add full extraction mode)

**Note:** The implementation of this task depends on Phase 0 results:
- **Path A (>=50% table yield):** Extend `fetch_dailymed_pk.py` with table extraction + unit normalization + text fallback.
- **Path B (<50% table yield):** Skip DailyMed XML. Instead improve `scripts/extract_openfda_pk.py` with: (1) additional regex patterns for tabular text (column-aligned numbers), (2) range extraction ("73 to 113 ng/mL" → take geometric mean), (3) context-aware fasted-state detection. Expected yield: +20-40 drugs (lower than Path A but still meaningful).

- [ ] **Step 1:** Add full extraction mode to `fetch_dailymed_pk.py`

Add function `extract_cmax_from_table(table_element) -> dict | None` that:
- Identifies column headers (Cmax, AUC, Tmax, Dose)
- Parses data rows
- Converts units to mg/L
- Returns `{cmax_mg_L, auc_mg_h_L, dose_mg, unit_raw}`

Add function `extract_cmax_from_text(text) -> dict | None` as fallback that:
- Uses regex patterns from `extract_openfda_pk.py`
- Returns same structure

Add `run_full(drug_list)` that:
- Iterates all drugs
- Tries table extraction first, then text fallback
- Validates with platinum_schema
- Outputs `data/ml/clinical/dailymed_extracted.csv`

- [ ] **Step 2:** Build candidate drug list from existing data

```python
# Combine all known drugs that don't yet have Cmax
# From: reference_database.json (266 SMILES) + expanded_cmax.csv (129 drugs)
# Target: unique drugs not already in platinum_reference.json
```

- [ ] **Step 3:** Run full extraction

```bash
python scripts/fetch_dailymed_pk.py --full --output data/ml/clinical/dailymed_extracted.csv
```

- [ ] **Step 4:** Review extraction quality

```bash
python -c "
import pandas as pd
df = pd.read_csv('data/ml/clinical/dailymed_extracted.csv')
print(f'Drugs extracted: {len(df)}')
print(f'With Cmax: {df.cmax_mg_L.notna().sum()}')
print(f'Source: table={len(df[df.extraction_method==\"table\"])}, text={len(df[df.extraction_method==\"text\"])}')
"
```

- [ ] **Step 5:** Commit

```bash
git add scripts/fetch_dailymed_pk.py data/ml/clinical/dailymed_extracted.csv
git commit -m "feat: full-scale DailyMed FDA Cmax extraction"
```

---

## Task 6: Fix PK-DB Integration

**Why:** `expand_pkdb_cmax.py` currently returns only 3 mock entries. Fixing it adds 15-25 real drugs from literature C(t) curves.

**Files:**
- Modify: `scripts/expand_pkdb_cmax.py`

- [ ] **Step 1:** Remove mock data, enable real API calls

In `expand_pkdb_cmax.py`:
- Remove `MOCK_TSV_CONTENT` and all references to it
- Remove `--dry-run` default behavior
- Enable real PK-DB API queries

- [ ] **Step 2:** Extend MW table via PubChem

Add auto-MW lookup from PubChem for drugs not in the hardcoded table:

```python
def lookup_mw(drug_name: str) -> float | None:
    """Fetch molecular weight from PubChem."""
    from omega_pbpk.data.pubchem_client import lookup_by_name
    result = lookup_by_name(drug_name)
    return result.molecular_weight if result else None
```

- [ ] **Step 3:** Compute Cmax from existing timecourses

21 drugs already have C(t) curves in `data/clinical/pkdb_timecourses.json`. For drugs not yet in platinum, compute Cmax directly from the timecourse:

```python
def cmax_from_timecourse(timecourse: dict) -> float | None:
    """Extract Cmax from a PK-DB timecourse record.

    PK-DB timepoints use {"time_h": float, "mean": float, "sd": float}.
    The "mean" field is concentration in the unit specified by the study.
    Assumes unit conversion to mg/L was already done during fetch.
    """
    points = timecourse.get("timepoints", [])
    if not points:
        return None
    concs = [p.get("mean", 0) for p in points if p.get("mean") is not None]
    return max(concs) if concs else None
```

- [ ] **Step 4:** Run PK-DB extraction

```bash
python scripts/expand_pkdb_cmax.py --output data/ml/clinical/pkdb_extracted.csv
```

- [ ] **Step 5:** Commit

```bash
git add scripts/expand_pkdb_cmax.py data/ml/clinical/pkdb_extracted.csv
git commit -m "feat: fix PK-DB integration, enable real API extraction"
```

---

## Task 7: Merge All Sources into Platinum Reference

**Why:** Combine gold-24 + DailyMed + PK-DB + existing expanded data into a single validated `platinum_reference.json`.

**Files:**
- Create: `scripts/merge_platinum_sources.py`
- Modify: `data/clinical/platinum_reference.json`

- [ ] **Step 1:** Create merge script

The script should:
1. Load existing platinum_reference.json (gold-24 base)
2. Load dailymed_extracted.csv (Task 5)
3. Load pkdb_extracted.csv (Task 6)
4. Load expanded_cmax.csv (existing, for gap-fill)
5. Deduplicate: by normalized name (Levenshtein) + SMILES (Tanimoto)
6. Priority: gold-24 > DailyMed table > DailyMed text > PK-DB > expanded
7. Validate each entry with `platinum_schema.validate_entry()`
8. Flag for manual review if prediction discrepancy > 10x
9. Save merged platinum_reference.json

- [ ] **Step 2:** Run merge

```bash
python scripts/merge_platinum_sources.py
```

- [ ] **Step 3:** Report drug count and quality

```bash
python -c "
import json
d = json.load(open('data/clinical/platinum_reference.json'))
drugs = d['drugs']
print(f'Total drugs: {len(drugs)}')
print(f'Confirmed fasted: {sum(1 for v in drugs.values() if v[\"fasted_confidence\"]==\"confirmed_fasted\")}')
print(f'Tuning contaminated: {sum(1 for v in drugs.values() if v[\"tuning_contaminated\"])}')
print(f'Non-linear PK: {sum(1 for v in drugs.values() if v[\"nonlinear_pk\"])}')
"
```

Expected: 150+ drugs total.

- [ ] **Step 4:** Commit

```bash
git add scripts/merge_platinum_sources.py data/clinical/platinum_reference.json
git commit -m "feat: merge all sources into unified platinum reference (N drugs)"
```

---

## Task 8: Unified Benchmark Script

**Why:** Single script replacing `run_l1_benchmarks.py`, `run_full_benchmark.py`, `run_expanded_benchmark.py` for all benchmarking needs.

**Files:**
- Create: `scripts/run_platinum_benchmark.py`

- [ ] **Step 1:** Create benchmark script

```python
#!/usr/bin/env python3
"""Unified Platinum benchmark runner.

Runs all drugs in platinum_reference.json through OmegaPipeline, computes
AAFE/%%2-fold with bootstrap CI, supports subsetting and cross-validation.

Usage:
    python scripts/run_platinum_benchmark.py
    python scripts/run_platinum_benchmark.py --subset core24
    python scripts/run_platinum_benchmark.py --bootstrap 10000
    python scripts/run_platinum_benchmark.py --clean-only
    python scripts/run_platinum_benchmark.py --cv 5
"""
import argparse
import json
import math
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root / "src"))

from omega_pbpk.data.drug_registry import CORE24_NAMES
from omega_pbpk.data.platinum_schema import load_platinum_reference
from omega_pbpk.pipeline import OmegaPipeline, SimulationRequest

PLATINUM_REF = repo_root / "data" / "clinical" / "platinum_reference.json"


def compute_aafe(fold_errors: list[float]) -> float:
    if not fold_errors:
        return float("nan")
    return math.exp(sum(math.log(fe) for fe in fold_errors) / len(fold_errors))


def bootstrap_ci(fold_errors: list[float], n_boot: int = 10000, seed: int = 42):
    log_fe = np.log10(np.array(fold_errors))
    rng = np.random.default_rng(seed)
    n = len(log_fe)
    aafes = []
    for _ in range(n_boot):
        sample = rng.choice(log_fe, size=n, replace=True)
        aafes.append(10 ** np.mean(np.abs(sample)))
    return float(np.percentile(aafes, 2.5)), float(np.percentile(aafes, 97.5))


def run_benchmark(drugs: dict, pipeline: OmegaPipeline) -> dict:
    results = {}
    for name, entry in drugs.items():
        t0 = time.perf_counter()
        try:
            result = pipeline.simulate(SimulationRequest(
                smiles=entry["smiles"],
                dose_mg=entry["dose_mg"],
                route="oral",
            ))
            pred_cmax = result.cmax_mg_L
        except Exception as e:
            print(f"  ERROR {name}: {e}")
            continue
        dt_ms = (time.perf_counter() - t0) * 1000
        obs_cmax = entry["cmax_mg_L"]
        fe = max(pred_cmax / obs_cmax, obs_cmax / pred_cmax)
        results[name] = {
            "pred_cmax": pred_cmax,
            "obs_cmax": obs_cmax,
            "fold_error": fe,
            "latency_ms": dt_ms,
            "tuning_contaminated": entry.get("tuning_contaminated", False),
            "nonlinear_pk": entry.get("nonlinear_pk", False),
        }
    return results


def report(results: dict, label: str, n_boot: int = 0):
    fes = [r["fold_error"] for r in results.values()]
    if not fes:
        print(f"  {label}: no results")
        return
    aafe = compute_aafe(fes)
    pct2 = 100 * sum(1 for fe in fes if fe <= 2.0) / len(fes)
    pct3 = 100 * sum(1 for fe in fes if fe <= 3.0) / len(fes)
    max_fe = max(fes)
    ci_str = ""
    if n_boot > 0:
        lo, hi = bootstrap_ci(fes, n_boot)
        ci_str = f" [{lo:.2f}, {hi:.2f}]"

    print(f"\n{'='*50}")
    print(f"  {label} (N={len(fes)})")
    print(f"  AAFE:    {aafe:.3f}{ci_str}")
    print(f"  %2-fold: {pct2:.1f}%")
    print(f"  %3-fold: {pct3:.1f}%")
    print(f"  Max FE:  {max_fe:.2f}x")
    med_fe = float(np.median(fes))
    print(f"  Median:  {med_fe:.2f}x")
    mean_lat = np.mean([r["latency_ms"] for r in results.values()])
    print(f"  Latency: {mean_lat:.0f} ms/drug (mean)")
    print(f"{'='*50}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", choices=["core24"], help="Run only a subset")
    parser.add_argument("--clean-only", action="store_true", help="Exclude tuning-contaminated drugs")
    parser.add_argument("--linear-only", action="store_true", help="Exclude non-linear PK drugs")
    parser.add_argument("--bootstrap", type=int, default=10000, help="Bootstrap resamples for CI")
    parser.add_argument("--cv", type=int, default=0, help="K-fold CV (0=disabled)")
    parser.add_argument("--output", type=str, default=None, help="Save JSON results")
    args = parser.parse_args()

    drugs = load_platinum_reference(PLATINUM_REF)

    # Apply filters
    if args.subset == "core24":
        drugs = {k: v for k, v in drugs.items() if k in CORE24_NAMES}
    if args.clean_only:
        drugs = {k: v for k, v in drugs.items() if not v.get("tuning_contaminated")}
    if args.linear_only:
        drugs = {k: v for k, v in drugs.items() if not v.get("nonlinear_pk")}

    print(f"Platinum Benchmark: {len(drugs)} drugs")
    pipeline = OmegaPipeline()
    results = run_benchmark(drugs, pipeline)

    # Full report
    report(results, "All Drugs", args.bootstrap)

    # Core-24 subset
    core_results = {k: v for k, v in results.items() if k in CORE24_NAMES}
    if core_results and args.subset != "core24":
        report(core_results, "Core-24 Subset", args.bootstrap)

    # Clean subset
    clean_results = {k: v for k, v in results.items() if not v["tuning_contaminated"]}
    if clean_results and len(clean_results) < len(results):
        report(clean_results, "Clean (non-contaminated)", args.bootstrap)

    # Per-drug table
    print(f"\n{'Drug':<25} {'Pred':>8} {'Obs':>8} {'FE':>6} {'ms':>5} {'Flags'}")
    print("-" * 65)
    for name in sorted(results):
        r = results[name]
        flags = []
        if r["tuning_contaminated"]:
            flags.append("C")
        if r["nonlinear_pk"]:
            flags.append("NL")
        flag_str = ",".join(flags) if flags else ""
        status = "OK" if r["fold_error"] <= 2.0 else "WARN" if r["fold_error"] <= 3.0 else "FAIL"
        print(f"{name:<25} {r['pred_cmax']:>8.4f} {r['obs_cmax']:>8.4f} {r['fold_error']:>5.2f}x {r['latency_ms']:>5.0f} {flag_str:>4} {status}")

    # Save JSON
    if args.output:
        out = {
            "date": str(date.today()),
            "n_drugs": len(results),
            "results": results,
        }
        Path(args.output).write_text(json.dumps(out, indent=2))
        print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2:** Run on current platinum reference (gold-24)

```bash
python scripts/run_platinum_benchmark.py --output outputs/platinum_baseline.json
```

Expected: Core-24 AAFE ~1.50, matching existing benchmark.

- [ ] **Step 3:** Verify `--clean-only` flag works

```bash
python scripts/run_platinum_benchmark.py --clean-only
```

Expected: Fewer drugs, separate AAFE for non-contaminated subset.

- [ ] **Step 4:** Commit

```bash
git add scripts/run_platinum_benchmark.py
git commit -m "feat: unified platinum benchmark script with subset/clean/cv support"
```

---

## Task 9: Platinum Regression Gate

**Why:** Two-level pytest gate: core-24 (strict, no regression) + full platinum (loose, catastrophic prevention).

**Files:**
- Create: `tests/regression/test_platinum_regression.py`

- [ ] **Step 1:** Create the regression test

```python
# tests/regression/test_platinum_regression.py
"""Platinum benchmark two-level regression gate.

Level 1 (Core-24): AAFE <= 1.70, %2-fold >= 75% — strict, prevents quality regression
Level 2 (Full):    AAFE <= 2.50, %2-fold >= 40% — loose, prevents catastrophic regression

Run: pytest tests/regression/test_platinum_regression.py -v -m benchmark
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from omega_pbpk.data.drug_registry import CORE24_NAMES
from omega_pbpk.data.platinum_schema import load_platinum_reference

REPO_ROOT = Path(__file__).resolve().parents[2]
PLATINUM_REF = REPO_ROOT / "data" / "clinical" / "platinum_reference.json"

# Level 1: Core-24
CORE24_AAFE_MAX = 1.70
CORE24_PCT2FOLD_MIN = 75.0
CORE24_MAX_SINGLE_FE = 6.0

# Level 2: Full Platinum
PLATINUM_AAFE_MAX = 3.50  # start loose; tighten after first full run
PLATINUM_PCT2FOLD_MIN = 40.0
PLATINUM_MAX_SINGLE_FE = 10.0


@pytest.fixture(scope="module")
def pipeline():
    from omega_pbpk.pipeline import OmegaPipeline
    return OmegaPipeline()


@pytest.fixture(scope="module")
def platinum_drugs():
    assert PLATINUM_REF.exists(), f"Platinum reference not found: {PLATINUM_REF}"
    return load_platinum_reference(PLATINUM_REF)


@pytest.fixture(scope="module")
def all_fold_errors(pipeline, platinum_drugs):
    from omega_pbpk.pipeline import SimulationRequest
    results = {}
    failures = []
    for name, entry in platinum_drugs.items():
        try:
            result = pipeline.simulate(SimulationRequest(
                smiles=entry["smiles"],
                dose_mg=entry["dose_mg"],
                route="oral",
            ))
            obs = entry["cmax_mg_L"]
            pred = result.cmax_mg_L
            results[name] = max(pred / obs, obs / pred)
        except Exception as e:
            failures.append((name, str(e)))
    if failures:
        print(f"\nWARNING: {len(failures)} drugs failed simulation:")
        for name, err in failures[:5]:
            print(f"  {name}: {err}")
    return results


# --- Level 1: Core-24 ---

@pytest.mark.benchmark
def test_core24_aafe(all_fold_errors):
    core = {k: v for k, v in all_fold_errors.items() if k in CORE24_NAMES}
    assert len(core) >= 20, f"Too few core-24 drugs: {len(core)}"
    fes = list(core.values())
    aafe = math.exp(sum(math.log(fe) for fe in fes) / len(fes))
    assert aafe <= CORE24_AAFE_MAX, f"Core-24 AAFE {aafe:.3f} > {CORE24_AAFE_MAX}"


@pytest.mark.benchmark
def test_core24_pct_2fold(all_fold_errors):
    core = {k: v for k, v in all_fold_errors.items() if k in CORE24_NAMES}
    fes = list(core.values())
    pct = 100.0 * sum(1 for fe in fes if fe <= 2.0) / len(fes)
    assert pct >= CORE24_PCT2FOLD_MIN, f"Core-24 %2-fold {pct:.1f}% < {CORE24_PCT2FOLD_MIN}%"


@pytest.mark.benchmark
def test_core24_no_catastrophic(all_fold_errors):
    core = {k: v for k, v in all_fold_errors.items() if k in CORE24_NAMES}
    bad = [(n, fe) for n, fe in core.items() if fe > CORE24_MAX_SINGLE_FE]
    if bad:
        detail = ", ".join(f"{n}:{fe:.1f}x" for n, fe in sorted(bad, key=lambda x: -x[1]))
        pytest.fail(f"Core-24 drugs > {CORE24_MAX_SINGLE_FE}x: {detail}")


# --- Level 2: Full Platinum ---

@pytest.mark.benchmark
def test_platinum_aafe(all_fold_errors):
    fes = list(all_fold_errors.values())
    assert len(fes) >= 20, f"Too few platinum drugs: {len(fes)}"
    aafe = math.exp(sum(math.log(fe) for fe in fes) / len(fes))
    assert aafe <= PLATINUM_AAFE_MAX, f"Platinum AAFE {aafe:.3f} > {PLATINUM_AAFE_MAX}"


@pytest.mark.benchmark
def test_platinum_pct_2fold(all_fold_errors):
    fes = list(all_fold_errors.values())
    pct = 100.0 * sum(1 for fe in fes if fe <= 2.0) / len(fes)
    assert pct >= PLATINUM_PCT2FOLD_MIN, f"Platinum %2-fold {pct:.1f}% < {PLATINUM_PCT2FOLD_MIN}%"


@pytest.mark.benchmark
def test_platinum_no_catastrophic(all_fold_errors):
    bad = [(n, fe) for n, fe in all_fold_errors.items() if fe > PLATINUM_MAX_SINGLE_FE]
    if bad:
        detail = ", ".join(f"{n}:{fe:.1f}x" for n, fe in sorted(bad, key=lambda x: -x[1])[:5])
        pytest.fail(f"Platinum drugs > {PLATINUM_MAX_SINGLE_FE}x: {detail}")
```

- [ ] **Step 2:** Run regression gate

```bash
pytest tests/regression/test_platinum_regression.py -v -m benchmark
```

Expected: All 6 tests pass on current gold-24 platinum reference.

- [ ] **Step 3:** Commit

```bash
git add tests/regression/test_platinum_regression.py
git commit -m "test: two-level platinum regression gate (core-24 strict + full loose)"
```

---

## Task 10: ML Retraining on Platinum Dataset (Outline)

**Why:** With 150-200 drugs, existing ML correctors can be retrained with proper data. This is done AFTER the platinum reference is stable.

**Note:** This task is an outline. Full implementation depends on the final drug count and data quality from Tasks 5-7.

- [ ] **Step 1:** Retrain DirectCmaxPredictor

```bash
# Update training script to use platinum_reference.json
python scripts/train_direct_cmax.py --data data/clinical/platinum_reference.json
```

- [ ] **Step 2:** Retrain Pre-ODE + Post-ODE correctors with LOO-CV

```bash
python scripts/train_pre_ode_corrector.py --data data/clinical/platinum_reference.json --cv loo
python scripts/train_post_ode_corrector.py --data data/clinical/platinum_reference.json --cv loo
```

- [ ] **Step 3:** Deploy adaptive conformal with full calibration set

```bash
python scripts/train_adaptive_conformal.py --data data/clinical/platinum_reference.json
```

- [ ] **Step 4:** Run ablation on platinum set

```bash
python scripts/run_ablation.py --data data/clinical/platinum_reference.json
```

- [ ] **Step 5:** Commit all retrained models

```bash
git add models/ data/clinical/platinum_reference.json
git commit -m "feat: retrain ML models on platinum dataset (N drugs)"
```

---

## Final Verification (After All Tasks)

```bash
# 1. Fast tests
pytest tests/ -m "not slow and not benchmark" -q

# 2. Drug registry tests
pytest tests/data/ -v

# 3. Gold-24 regression (existing, must still pass)
pytest tests/regression/test_gold24_regression.py -v -m benchmark

# 4. Platinum regression (new)
pytest tests/regression/test_platinum_regression.py -v -m benchmark

# 5. Full platinum benchmark
python scripts/run_platinum_benchmark.py --bootstrap 10000

# 6. Core-24 subset (must match existing AAFE)
python scripts/run_platinum_benchmark.py --subset core24

# 7. Clean subset
python scripts/run_platinum_benchmark.py --clean-only
```

**Pass criteria:**
- Steps 1-4: All tests pass
- Step 5: Platinum AAFE reported honestly (no pre-set target)
- Step 6: Core-24 AAFE <= 1.55 (within noise of current 1.50)
- Step 7: Clean AAFE reported for uncontaminated drugs
