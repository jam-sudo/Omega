"""YAML configuration loader for compound and subject files.

Converts YAML compound specs → Drug dataclass.
Converts YAML subject specs → body weight and optional physiology overrides.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

from omega_pbpk.drugs.drug import Drug

logger = logging.getLogger(__name__)


def load_compound(path: str | Path) -> Drug:
    """Load a compound YAML file and return a Drug dataclass.

    Args:
        path: Path to compound YAML file.

    Returns:
        Drug dataclass with all parameters.

    Raises:
        FileNotFoundError: If file doesn't exist.
        ValueError: If required fields are missing.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Compound file not found: {path}")

    with open(path) as f:
        raw: dict[str, Any] = yaml.safe_load(f) or {}

    # Extract clearance
    clearance = raw.get("clearance", {})
    clint = clearance.get("clint_uL_min_pmol", {})
    fm = clearance.get("fm", {})

    # Extract absorption
    absorption = raw.get("absorption", {})

    # Extract distribution
    distribution = raw.get("distribution", {})
    kp = distribution.get("kp", {})
    permeability_limited = distribution.get("permeability_limited", {})

    # Gut metabolism
    gut_met = raw.get("gut_metabolism", {})

    drug = Drug(
        name=raw.get("name", path.stem),
        mw=float(raw.get("mw", 300.0)),
        logP=float(raw.get("logP", 2.0)),
        pka=raw.get("pka", [7.0]),
        drug_type=raw.get("drug_type", "neutral"),
        fup=float(raw.get("fup", 0.5)),
        rbp=float(raw.get("rbp", 1.0)),
        smiles=raw.get("smiles"),
        clint=clint,
        fm=fm,
        peff=float(absorption.get("peff", 1.0)),
        solubility_mg_mL=float(absorption.get("solubility_mg_mL", 1.0)),
        particle_radius_um=float(absorption.get("particle_radius_um", 25.0)),
        particle_density=float(absorption.get("particle_density", 1.2)),
        kp=kp,
        permeability_limited=permeability_limited,
        gut_clint_multiplier=float(gut_met.get("gut_clint_multiplier", 1.0)),
    )

    logger.info(
        "Loaded compound: %s (MW=%.1f, logP=%.2f, fup=%.3f)",
        drug.name,
        drug.mw,
        drug.logP,
        drug.fup,
    )
    return drug


def load_subject(path: str | Path) -> dict[str, Any]:
    """Load a subject YAML file.

    Args:
        path: Path to subject YAML file.

    Returns:
        Dict with body_weight_kg and optional physiology overrides.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Subject file not found: {path}")

    with open(path) as f:
        raw: dict[str, Any] = yaml.safe_load(f) or {}

    return {
        "body_weight_kg": float(raw.get("body_weight_kg", 70.0)),
        "age_years": int(raw.get("age_years", 30)),
        "sex": raw.get("sex", "male"),
        "cardiac_output_L_h": float(raw.get("cardiac_output_L_h", 390.0)),
        "gfr_L_h": float(raw.get("gfr_L_h", 7.5)),
    }
