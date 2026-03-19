"""Platinum benchmark reference schema and validation.

Defines the data contract for platinum_reference.json entries.
All drugs in the benchmark must pass validation before inclusion.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
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
    "smiles",
    "dose_mg",
    "cmax_mg_L",
    "source_type",
    "source_id",
    "fasted_confidence",
    "formulation",
    "route",
    "population",
    "single_dose",
    "tuning_contaminated",
    "nonlinear_pk",
    "data_quality",
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
        raise ValidationError(f"{drug_name}: cmax/dose ratio {ratio:.2e} outside (1e-6, 1.0)")

    # Enum fields
    if data["fasted_confidence"] not in _VALID_FASTED:
        raise ValidationError(
            f"{drug_name}: fasted_confidence '{data['fasted_confidence']}' not in {_VALID_FASTED}"
        )
    if data["formulation"] not in _VALID_FORMULATION:
        raise ValidationError(
            f"{drug_name}: formulation '{data['formulation']}' not in {_VALID_FORMULATION}"
        )
    if data["route"] not in _VALID_ROUTE:
        raise ValidationError(f"{drug_name}: route '{data['route']}' must be 'oral'")
    if data["source_type"] not in _VALID_SOURCE_TYPE:
        raise ValidationError(
            f"{drug_name}: source_type '{data['source_type']}' not in {_VALID_SOURCE_TYPE}"
        )
    if data["data_quality"] not in _VALID_DATA_QUALITY:
        raise ValidationError(
            f"{drug_name}: data_quality '{data['data_quality']}' not in {_VALID_DATA_QUALITY}"
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


def save_platinum_reference(drugs: dict[str, dict], path: str | Path, version: str = "1.0") -> None:
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
