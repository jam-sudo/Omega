"""Harmonized clinical PK dataset combining PK-DB and FDA label data.

This module standardizes units, matches compounds to SMILES, and provides
train/val/test splitting for downstream ML training.
"""

from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pandas as pd

    from omega_pbpk.ml.data.loaders import FDALabelExtractor, PKDBLoader

logger = logging.getLogger(__name__)

_DEFAULT_OUTPUT_DIR = Path("data/ml/clinical")

# ---------------------------------------------------------------------------
# Unit conversion factors → canonical units
#   concentration → mg/L
#   time → h
#   clearance → L/h
#   volume → L
# ---------------------------------------------------------------------------

_CONCENTRATION_TO_MG_L: dict[str, float] = {
    "mg/l": 1.0,
    "mg/L": 1.0,
    "mg/ml": 1_000.0,
    "mg/mL": 1_000.0,
    "µg/ml": 1.0,
    "ug/ml": 1.0,
    "µg/mL": 1.0,
    "ug/mL": 1.0,
    "ng/ml": 0.001,
    "ng/mL": 0.001,
    "g/l": 1_000.0,
    "g/L": 1_000.0,
}

_TIME_TO_H: dict[str, float] = {
    "h": 1.0,
    "hr": 1.0,
    "hrs": 1.0,
    "hours": 1.0,
    "hour": 1.0,
    "min": 1.0 / 60.0,
    "minutes": 1.0 / 60.0,
    "minute": 1.0 / 60.0,
    "day": 24.0,
    "days": 24.0,
}

_CLEARANCE_TO_L_PER_H: dict[str, float] = {
    "L/h": 1.0,
    "l/h": 1.0,
    "L/hr": 1.0,
    "l/hr": 1.0,
    "mL/min": 0.06,
    "ml/min": 0.06,
}

_VOLUME_TO_L: dict[str, float] = {
    "L": 1.0,
    "l": 1.0,
    "mL": 0.001,
    "ml": 0.001,
}


def convert_concentration(value: float, unit: str) -> float | None:
    """Convert a concentration value to mg/L. Returns None if unit unknown."""
    factor = _CONCENTRATION_TO_MG_L.get(unit)
    return value * factor if factor is not None else None


def convert_time(value: float, unit: str) -> float | None:
    """Convert a time value to hours. Returns None if unit unknown."""
    factor = _TIME_TO_H.get(unit)
    return value * factor if factor is not None else None


def convert_clearance(value: float, unit: str) -> float | None:
    """Convert clearance to L/h. Returns None if unit unknown."""
    # Strip /kg suffix for now — per-kg values are kept as-is
    base_unit = unit.replace("/kg", "")
    factor = _CLEARANCE_TO_L_PER_H.get(base_unit)
    return value * factor if factor is not None else None


def convert_volume(value: float, unit: str) -> float | None:
    """Convert volume to L. Returns None if unit unknown."""
    base_unit = unit.replace("/kg", "")
    factor = _VOLUME_TO_L.get(base_unit)
    return value * factor if factor is not None else None


# ---------------------------------------------------------------------------
# SMILES lookup — best-effort name → SMILES mapping
# ---------------------------------------------------------------------------

# A small built-in table for common drugs (extendable).
_BUILTIN_SMILES: dict[str, str] = {
    "caffeine": "Cn1c(=O)c2c(ncn2C)n(C)c1=O",
    "acetaminophen": "CC(=O)Nc1ccc(O)cc1",
    "paracetamol": "CC(=O)Nc1ccc(O)cc1",
    "ibuprofen": "CC(C)Cc1ccc(cc1)[C@@H](C)C(O)=O",
    "aspirin": "CC(=O)Oc1ccccc1C(O)=O",
    "metformin": "CN(C)C(=N)NC(=N)N",
    "warfarin": "CC(=O)CC(c1ccccc1)c1c(O)c2ccccc2oc1=O",
    "theophylline": "Cn1c(=O)c2[nH]cnc2n(C)c1=O",
    "midazolam": "Clc1ccc2c(c1)C(=NCC(=O)n2C)c1ccccc1F",
    "diazepam": "CN1C(=O)CN=C(c2ccccc2)c2cc(Cl)ccc21",
    # Benchmark drugs (added for 20-drug coverage)
    "propranolol": "CC(C)NCC(O)c1cccc2ccccc12",
    "metoprolol": "COCCOCC(CNC(C)C)c1ccc(OC)cc1",
    "amphetamine": "CC(N)Cc1ccccc1",
    "d-amphetamine": "C[C@@H](N)Cc1ccccc1",
    "methanol": "CO",
    "omeprazole": "COc1ccc2[nH]c(S(=O)Cc3ncc(C)c(OC)c3C)nc2c1",
    "amoxicillin": "CC1(C)S[C@@H]2[C@H](NC(=O)[C@@H](N)c3ccc(O)cc3)C(=O)N2[C@@H]1C(O)=O",
    "atorvastatin": "CC(C)c1n(CC[C@@H](O)C[C@@H](O)CC(O)=O)c(-c2ccc(F)cc2)c(-c2ccccc2)c1C(=O)Nc1ccccc1",
    "fluoxetine": "CNCCC(Oc1ccc(C(F)(F)F)cc1)c1ccccc1",
    "carbamazepine": "NC(=O)N1c2ccccc2C=Cc2ccccc21",
    "phenytoin": "O=C1NC(=O)C(c2ccccc2)(c2ccccc2)N1",
    "verapamil": "COc1ccc(CCN(C)CCCC(C#N)(c2ccc(OC)c(OC)c2)C(C)C)cc1OC",
    "nifedipine": "COC(=O)C1=C(C)NC(C)=C(C(=O)OC)C1c1ccccc1[N+]([O-])=O",
    "digoxin": "C[C@H]1OC(OC2CC(O)C(OC3CC(O)C(OC4CCC5(C)C(CCC6C5CC(O)C5(C)C(C7=CC(=O)OC7)CCC65)C4)C(C)O3)C(C)O2)CC(O)C1O",
}


def lookup_smiles(compound_name: str) -> str | None:
    """Best-effort compound name → SMILES lookup.

    Uses a built-in table first, then tries RDKit's name-to-SMILES
    conversion if available.
    """
    name_lower = compound_name.strip().lower()
    if name_lower in _BUILTIN_SMILES:
        return _BUILTIN_SMILES[name_lower]

    # Try RDKit name → mol → SMILES
    try:
        from rdkit import Chem  # type: ignore[import-untyped]

        mol = Chem.MolFromSmiles(compound_name)
        if mol is None:
            # Try as name (RDKit's NameToMol is limited)
            return None
        return Chem.MolToSmiles(mol)
    except ImportError:
        pass

    return None


# ---------------------------------------------------------------------------
# Scaffold splitting
# ---------------------------------------------------------------------------


def _scaffold_split(
    smiles_list: list[str],
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    seed: int = 42,
) -> dict[str, str]:
    """Assign each SMILES to train/valid/test using Murcko scaffolds.

    Falls back to random splitting if RDKit is unavailable.

    Returns mapping of SMILES → split label.
    """
    try:
        from rdkit import Chem  # type: ignore[import-untyped]
        from rdkit.Chem.Scaffolds import MurckoScaffold  # type: ignore[import-untyped]

        scaffold_to_indices: dict[str, list[int]] = {}
        for idx, smi in enumerate(smiles_list):
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
            else:
                scaffold = smi  # fallback: treat SMILES itself as scaffold
            scaffold_to_indices.setdefault(scaffold, []).append(idx)

        rng = random.Random(seed)
        scaffolds = list(scaffold_to_indices.keys())
        rng.shuffle(scaffolds)

        n = len(smiles_list)
        train_cutoff = int(n * train_frac)
        val_cutoff = train_cutoff + int(n * val_frac)

        assignments: dict[str, str] = {}
        count = 0
        for scaffold in scaffolds:
            for idx in scaffold_to_indices[scaffold]:
                if count < train_cutoff:
                    assignments[smiles_list[idx]] = "train"
                elif count < val_cutoff:
                    assignments[smiles_list[idx]] = "valid"
                else:
                    assignments[smiles_list[idx]] = "test"
                count += 1
        return assignments

    except ImportError:
        logger.warning("RDKit not available — falling back to random train/val/test split.")
        rng = random.Random(seed)
        assignments = {}
        for smi in smiles_list:
            r = rng.random()
            if r < train_frac:
                assignments[smi] = "train"
            elif r < train_frac + val_frac:
                assignments[smi] = "valid"
            else:
                assignments[smi] = "test"
        return assignments


# ===========================================================================
# ClinicalPKDataset
# ===========================================================================


@dataclass
class ClinicalPKDataset:
    """Unified clinical PK dataset combining PK-DB and FDA label data.

    Each record contains:
    - compound: drug name
    - smiles: SMILES string (may be None)
    - source: "pkdb" | "fda"
    - parameter: e.g. "cmax", "auc", "t_half", etc.
    - value: numeric value in canonical units
    - unit: canonical unit string
    - split: "train" | "valid" | "test"
    """

    records: list[dict[str, Any]] = field(default_factory=list)

    # -- Construction helpers -----------------------------------------------

    @classmethod
    def from_pkdb(cls, loader: PKDBLoader) -> ClinicalPKDataset:
        """Build dataset from PK-DB study data.

        Iterates over all studies and extracts PK output records.
        """
        dataset = cls()
        studies = loader.list_studies()
        logger.info("Processing %d PK-DB studies", len(studies))

        for study in studies:
            study_name = study.get("name", study.get("sid", "unknown"))
            pk_data = loader.get_pk_data(study_name)
            for record in pk_data:
                compound = record.get("substance", record.get("drug", ""))
                value = record.get("value")
                unit = record.get("unit", "")
                param = record.get("measurement_type", record.get("pktype", ""))

                if value is None:
                    continue

                try:
                    value = float(value)
                except (TypeError, ValueError):
                    continue

                smiles = lookup_smiles(compound)
                dataset.records.append(
                    {
                        "compound": compound,
                        "smiles": smiles,
                        "source": "pkdb",
                        "parameter": param,
                        "value": value,
                        "unit": unit,
                        "original_value": value,
                        "original_unit": unit,
                    }
                )

        return dataset

    @classmethod
    def from_fda(cls, extractor: FDALabelExtractor) -> ClinicalPKDataset:
        """Build dataset from FDA label extractions.

        Requires that labels have already been searched and cached.
        This method walks the cache directory for previously downloaded SPLs.
        """
        dataset = cls()
        cache_files = list(extractor.cache_dir.glob("*.json"))
        logger.info("Processing %d cached FDA label files", len(cache_files))

        for cache_file in cache_files:
            try:
                data = json.loads(cache_file.read_text())
            except (json.JSONDecodeError, OSError):
                continue

            if not isinstance(data, dict):
                continue

            title = data.get("title", "")
            # Try to extract drug name from title
            compound = title.split(" - ")[0].strip() if title else ""
            if not compound:
                continue

            # Walk sections for PK text
            sections = data.get("sections", [])
            pk_text = _extract_pk_text(sections)
            if not pk_text:
                continue

            from omega_pbpk.ml.data.loaders import FDALabelExtractor as _FDAExtractor

            params = _FDAExtractor.extract_pk_params(pk_text)
            smiles = lookup_smiles(compound)

            for param_name, matches in params.items():
                for match in matches:
                    dataset.records.append(
                        {
                            "compound": compound,
                            "smiles": smiles,
                            "source": "fda",
                            "parameter": param_name,
                            "value": match["value"],
                            "unit": match["unit"],
                            "original_value": match["value"],
                            "original_unit": match["unit"],
                        }
                    )

        return dataset

    # -- Harmonization ------------------------------------------------------

    def standardize_units(self) -> ClinicalPKDataset:
        """Convert all values to canonical units in-place and return self.

        Canonical units:
        - concentrations → mg/L
        - time → h
        - clearance → L/h
        - volume → L
        """
        concentration_params = {"cmax"}
        time_params = {"t_half", "tmax"}
        clearance_params = {"clearance", "cl"}
        volume_params = {"vd", "vss"}

        new_records = []
        for rec in self.records:
            rec = dict(rec)  # copy
            param = rec["parameter"].lower()
            value = rec["value"]
            unit = rec["unit"]

            converted: float | None = None
            canonical_unit = unit

            if param in concentration_params:
                converted = convert_concentration(value, unit)
                if converted is not None:
                    canonical_unit = "mg/L"
            elif param in time_params:
                converted = convert_time(value, unit)
                if converted is not None:
                    canonical_unit = "h"
            elif param in clearance_params:
                converted = convert_clearance(value, unit)
                if converted is not None:
                    canonical_unit = "L/h"
            elif param in volume_params:
                converted = convert_volume(value, unit)
                if converted is not None:
                    canonical_unit = "L"

            if converted is not None:
                rec["value"] = converted
                rec["unit"] = canonical_unit

            new_records.append(rec)

        return ClinicalPKDataset(records=new_records)

    def merge(self, other: ClinicalPKDataset) -> ClinicalPKDataset:
        """Merge two datasets, returning a new combined dataset."""
        return ClinicalPKDataset(records=list(self.records) + list(other.records))

    def assign_splits(self, seed: int = 42) -> ClinicalPKDataset:
        """Assign train/valid/test splits based on scaffold (or random).

        Only records with non-None SMILES are split by scaffold.
        Records without SMILES are assigned to 'train'.
        Returns a new dataset with 'split' column populated.
        """
        smiles_set = [r["smiles"] for r in self.records if r.get("smiles") is not None]
        unique_smiles = list(set(smiles_set))

        if unique_smiles:
            split_map = _scaffold_split(unique_smiles, seed=seed)
        else:
            split_map = {}

        new_records = []
        for rec in self.records:
            rec = dict(rec)
            smi = rec.get("smiles")
            rec["split"] = split_map.get(smi, "train") if smi else "train"
            new_records.append(rec)

        return ClinicalPKDataset(records=new_records)

    # -- I/O ---------------------------------------------------------------

    def to_dataframe(self) -> pd.DataFrame:
        """Convert records to a pandas DataFrame."""
        import pandas as pd

        return pd.DataFrame(self.records)

    def save(self, path: str | Path | None = None) -> Path:
        """Save dataset as parquet (preferred) or CSV (fallback).

        Parameters
        ----------
        path:
            Output file path.  Defaults to ``data/ml/clinical/clinical_pk.parquet``.
            If pyarrow/fastparquet are unavailable and the path ends with
            ``.parquet``, the suffix is changed to ``.csv`` automatically.
        """
        out = Path(path or _DEFAULT_OUTPUT_DIR / "clinical_pk.parquet")
        out.parent.mkdir(parents=True, exist_ok=True)
        df = self.to_dataframe()

        try:
            df.to_parquet(out, index=False)
        except ImportError:
            # Parquet engine not installed — fall back to CSV
            if out.suffix == ".parquet":
                out = out.with_suffix(".csv")
            df.to_csv(out, index=False)
            logger.warning("pyarrow/fastparquet not installed; saved as CSV: %s", out)

        logger.info("Saved %d records to %s", len(df), out)
        return out

    @classmethod
    def load(cls, path: str | Path) -> ClinicalPKDataset:
        """Load dataset from a parquet or CSV file."""
        import pandas as pd

        path = Path(path)
        if path.suffix == ".csv":
            df = pd.read_csv(path)
        else:
            df = pd.read_parquet(path)
        records = df.to_dict(orient="records")
        return cls(records=records)


def _extract_pk_text(sections: list[dict[str, Any]]) -> str | None:
    """Walk DailyMed section tree looking for PK content."""
    pk_codes = {"34090-1", "43682-4"}
    for sec in sections:
        code = sec.get("loinc_code") or sec.get("code")
        if code in pk_codes:
            return sec.get("text", "")
        children = sec.get("children_sections", [])
        if children:
            result = _extract_pk_text(children)
            if result is not None:
                return result
    return None


# ---------------------------------------------------------------------------
# Data Quality Validation (Phase D.4)
# ---------------------------------------------------------------------------

# Expected ranges for PK parameters in canonical units.
# Records outside these ranges are flagged, not removed.
_PK_EXPECTED_RANGES: dict[str, tuple[float, float, str]] = {
    "cmax": (0.0001, 1000.0, "mg/L"),
    "auc": (0.001, 10000.0, "mg*h/L"),
    "t_half": (0.05, 1000.0, "h"),
    "tmax": (0.01, 72.0, "h"),
    "bioavailability": (0.0, 1.0, "fraction"),
    "vd": (0.01, 50000.0, "L"),
    "vss": (0.01, 50000.0, "L"),
    "clearance": (0.001, 1000.0, "L/h"),
    "cl": (0.001, 1000.0, "L/h"),
}


def validate_pk_record(record: dict[str, Any]) -> dict[str, Any]:
    """Check a PK record against expected ranges and add QC flags.

    Adds ``qc_flag`` key: ``"pass"``, ``"out_of_range"``, or ``"missing_value"``.
    Also adds ``qc_detail`` with explanation when flagged.

    Parameters
    ----------
    record:
        A single PK record dict with at least ``parameter``, ``value``, ``unit``.

    Returns
    -------
    dict
        The same record with ``qc_flag`` and optionally ``qc_detail`` added.
    """
    rec = dict(record)
    param = rec.get("parameter", "").lower()
    value = rec.get("value")

    if value is None:
        rec["qc_flag"] = "missing_value"
        rec["qc_detail"] = "No numeric value"
        return rec

    try:
        value = float(value)
    except (TypeError, ValueError):
        rec["qc_flag"] = "missing_value"
        rec["qc_detail"] = f"Non-numeric value: {value!r}"
        return rec

    if param in _PK_EXPECTED_RANGES:
        lo, hi, expected_unit = _PK_EXPECTED_RANGES[param]
        if value < lo or value > hi:
            rec["qc_flag"] = "out_of_range"
            rec["qc_detail"] = f"{param}={value} outside [{lo}, {hi}] {expected_unit}"
            return rec

    rec["qc_flag"] = "pass"
    return rec


def validate_dataset(dataset: ClinicalPKDataset) -> ClinicalPKDataset:
    """Run QC validation on all records, adding flags.

    Returns a new dataset with ``qc_flag`` on every record.
    """
    flagged = [validate_pk_record(r) for r in dataset.records]
    n_pass = sum(1 for r in flagged if r.get("qc_flag") == "pass")
    n_flag = len(flagged) - n_pass
    logger.info("QC: %d pass, %d flagged out of %d total", n_pass, n_flag, len(flagged))
    return ClinicalPKDataset(records=flagged)


# ---------------------------------------------------------------------------
# Pipeline Orchestration (Phase D.5)
# ---------------------------------------------------------------------------


def run_clinical_data_pipeline(
    output_dir: Path | str | None = None,
    use_pkdb: bool = True,
    use_fda: bool = True,
    use_tdc: bool = True,
    drug_list: list[str] | None = None,
    tdc_endpoints: list[str] | None = None,
    seed: int = 42,
) -> ClinicalPKDataset:
    """Run the full clinical data collection and harmonization pipeline.

    Steps:
        1. Collect PK params (+ timecourses) from PK-DB
        2. Collect PK params from FDA DailyMed labels
        3. Collect ADME benchmark data from TDC
        4. Merge all sources
        5. Standardize units
        6. Run QC validation
        7. Assign train/valid/test splits
        8. Save to output_dir

    Parameters
    ----------
    output_dir:
        Where to save the final dataset. Defaults to ``data/ml/clinical/``.
    use_pkdb:
        Whether to query PK-DB.
    use_fda:
        Whether to query FDA DailyMed labels.
    use_tdc:
        Whether to load TDC ADME datasets.
    drug_list:
        List of drug names to collect from PK-DB and FDA.
        If ``None``, uses the PK-DB substances available.
    tdc_endpoints:
        Which TDC endpoints to load. Defaults to all supported.
    seed:
        Random seed for scaffold splitting.

    Returns
    -------
    ClinicalPKDataset
        Harmonized, QC-flagged, split dataset.
    """
    from omega_pbpk.ml.data.loaders import FDALabelExtractor, PKDBLoader, TDCLoader

    out = Path(output_dir or _DEFAULT_OUTPUT_DIR)
    out.mkdir(parents=True, exist_ok=True)

    combined = ClinicalPKDataset()

    # Default drug list: PK-DB's 10 substances
    if drug_list is None:
        drug_list = [
            "acetaminophen",
            "caffeine",
            "codeine",
            "diazepam",
            "glucose",
            "midazolam",
            "morphine",
            "oxazepam",
            "simvastatin",
            "torasemide",
        ]

    # --- Step 1: PK-DB ---
    if use_pkdb:
        logger.info("=== Step 1: Collecting from PK-DB ===")
        pkdb = PKDBLoader()
        pkdb_dataset = ClinicalPKDataset.from_pkdb(pkdb)
        logger.info("PK-DB: %d records", len(pkdb_dataset.records))
        combined = combined.merge(pkdb_dataset)

    # --- Step 2: FDA DailyMed ---
    if use_fda:
        logger.info("=== Step 2: Collecting from FDA DailyMed ===")
        fda = FDALabelExtractor()
        for drug in drug_list:
            try:
                spls = fda.search_drug(drug)
                if not spls:
                    logger.warning("FDA: No labels found for %s", drug)
                    continue
                # Use first result
                setid = spls[0].get("setid", spls[0].get("spl_set_id"))
                if not setid:
                    continue
                pk_text = fda.get_pk_section(setid)
                if not pk_text:
                    logger.debug("FDA: No PK section for %s", drug)
                    continue
                params = fda.extract_pk_params(pk_text)
                smiles = lookup_smiles(drug)
                for param_name, matches in params.items():
                    for match in matches:
                        combined.records.append(
                            {
                                "compound": drug,
                                "smiles": smiles,
                                "source": "fda",
                                "parameter": param_name,
                                "value": match["value"],
                                "unit": match["unit"],
                                "original_value": match["value"],
                                "original_unit": match["unit"],
                            }
                        )
            except Exception:
                logger.exception("FDA: Error processing %s", drug)
                continue
        logger.info("Combined after FDA: %d records", len(combined.records))

    # --- Step 3: TDC ---
    if use_tdc:
        logger.info("=== Step 3: Collecting from TDC ===")
        tdc = TDCLoader()
        endpoints = tdc_endpoints or tdc.supported_endpoints()
        for ep in endpoints:
            try:
                df = tdc.load_adme(ep)
                for _, row in df.iterrows():
                    combined.records.append(
                        {
                            "compound": "",
                            "smiles": row["smiles"],
                            "source": f"tdc:{ep}",
                            "parameter": ep.lower(),
                            "value": row["value"],
                            "unit": "",
                            "split": row.get("split", "train"),
                        }
                    )
                logger.info("TDC %s: %d records", ep, len(df))
            except (ImportError, ValueError):
                logger.warning("TDC: Skipping %s (not available)", ep)
            except Exception:
                logger.exception("TDC: Error loading %s", ep)

    logger.info("=== Total raw records: %d ===", len(combined.records))

    # --- Step 4-5: Standardize units ---
    combined = combined.standardize_units()

    # --- Step 6: QC validation ---
    combined = validate_dataset(combined)

    # --- Step 7: Assign splits (only non-TDC records need splitting) ---
    combined = combined.assign_splits(seed=seed)

    # --- Step 8: Save ---
    save_path = combined.save(out / "clinical_pk.parquet")
    logger.info("Pipeline complete. Saved to %s", save_path)

    return combined
