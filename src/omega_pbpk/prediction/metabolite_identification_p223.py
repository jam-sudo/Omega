"""
Phase 223 — Metabolite Identification
Predict major metabolic pathways and metabolite structures for drug candidates.
Pure Python (stdlib: math, dataclasses) only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class MetaboliteIDResult:
    """Result of metabolite identification prediction."""

    drug_name: str
    parent_mw_Da: float
    parent_logP: float
    metabolite_names: tuple
    metabolite_mw_Da: tuple
    metabolite_pathways: tuple
    metabolite_enzymes: tuple
    n_metabolites: int
    primary_pathway: str
    notes: str


def identify_metabolites(
    drug_name: str,
    mw_Da: float,
    logP: float,
    has_aromatic: bool = False,
    has_aliphatic_ch: bool = False,
    has_amine: bool = False,
    has_ether: bool = False,
    has_sulfur: bool = False,
) -> MetaboliteIDResult:
    """
    Predict major metabolic pathways and metabolite structures.

    Parameters
    ----------
    drug_name : str
        Name of the drug.
    mw_Da : float
        Molecular weight in Daltons.
    logP : float
        Lipophilicity (log P).
    has_aromatic : bool
        Drug contains aromatic ring(s).
    has_aliphatic_ch : bool
        Drug contains aliphatic C-H bond(s).
    has_amine : bool
        Drug contains amine group(s).
    has_ether : bool
        Drug contains ether group(s).
    has_sulfur : bool
        Drug contains sulfur atom(s).

    Returns
    -------
    MetaboliteIDResult

    Raises
    ------
    ValueError
        If mw_Da <= 0.
    """
    if mw_Da <= 0:
        raise ValueError(f"mw_Da must be > 0, got {mw_Da}")

    names: list[str] = []
    mws: list[float] = []
    pathways: list[str] = []
    enzymes: list[str] = []

    # Aromatic hydroxylation (CYP3A4/1A2)
    if has_aromatic:
        names.append(f"{drug_name}_aromatic_OH")
        mws.append(mw_Da + 16.0)
        pathways.append("Aromatic hydroxylation")
        enzymes.append("CYP3A4/CYP1A2")

    # Aliphatic hydroxylation (CYP3A4)
    if has_aliphatic_ch:
        names.append(f"{drug_name}_aliphatic_OH")
        mws.append(mw_Da + 16.0)
        pathways.append("Aliphatic hydroxylation")
        enzymes.append("CYP3A4")

    # N-dealkylation (CYP2D6/3A4)
    if has_amine:
        names.append(f"{drug_name}_N_dealkyl")
        mws.append(mw_Da - 14.0)
        pathways.append("N-dealkylation")
        enzymes.append("CYP2D6/CYP3A4")

    # O-dealkylation (CYP2D6)
    if has_ether:
        names.append(f"{drug_name}_O_dealkyl")
        mws.append(mw_Da - 14.0)
        pathways.append("O-dealkylation")
        enzymes.append("CYP2D6")

    # Sulfoxidation (CYP3A4/FMO)
    if has_sulfur:
        names.append(f"{drug_name}_sulfoxide")
        mws.append(mw_Da + 16.0)
        pathways.append("Sulfoxidation")
        enzymes.append("CYP3A4/FMO")

    # Glucuronidation (UGT) — always possible
    names.append(f"{drug_name}_glucuronide")
    mws.append(mw_Da + 176.0)
    pathways.append("Glucuronidation")
    enzymes.append("UGT")

    # Sulfation (SULT) — only if mw < 500
    if mw_Da < 500:
        names.append(f"{drug_name}_sulfate")
        mws.append(mw_Da + 80.0)
        pathways.append("Sulfation")
        enzymes.append("SULT")

    n = len(names)

    # Determine primary pathway by functional group priority
    if has_aromatic:
        primary = "Aromatic hydroxylation"
    elif has_amine:
        primary = "N-dealkylation"
    elif has_aliphatic_ch:
        primary = "Aliphatic hydroxylation"
    elif has_ether:
        primary = "O-dealkylation"
    elif has_sulfur:
        primary = "Sulfoxidation"
    else:
        primary = "Glucuronidation"

    notes = (
        f"Identified {n} metabolite(s) via rule-based prediction. "
        f"logP={logP:.2f}. Primary pathway: {primary}."
    )

    return MetaboliteIDResult(
        drug_name=drug_name,
        parent_mw_Da=mw_Da,
        parent_logP=logP,
        metabolite_names=tuple(names),
        metabolite_mw_Da=tuple(mws),
        metabolite_pathways=tuple(pathways),
        metabolite_enzymes=tuple(enzymes),
        n_metabolites=n,
        primary_pathway=primary,
        notes=notes,
    )


def predict_metabolic_soft_spots(
    drug_name: str,
    mw_Da: float,
    logP: float,
    functional_groups: dict[str, bool],
) -> list[dict[str, Any]]:
    """
    Predict metabolic soft spots sorted by vulnerability score.

    Parameters
    ----------
    drug_name : str
        Name of the drug.
    mw_Da : float
        Molecular weight in Daltons.
    logP : float
        Lipophilicity.
    functional_groups : dict
        Keys: "has_aromatic", "has_aliphatic_ch", "has_amine", "has_ether", "has_sulfur".

    Returns
    -------
    List of dicts with keys: site, pathway, enzyme, vulnerability_score.
    """
    if mw_Da <= 0:
        raise ValueError(f"mw_Da must be > 0, got {mw_Da}")

    spots: list[dict[str, Any]] = []

    # Vulnerability scoring: higher logP → more CYP susceptibility
    logp_factor = min(max(logP / 5.0, 0.2), 2.0)  # clamp [0.2, 2.0]
    low_mw_factor = 1.2 if mw_Da < 400 else 1.0

    if functional_groups.get("has_aromatic", False):
        spots.append(
            {
                "site": "Aromatic ring",
                "pathway": "Aromatic hydroxylation",
                "enzyme": "CYP3A4/CYP1A2",
                "vulnerability_score": round(0.85 * logp_factor * low_mw_factor, 4),
            }
        )

    if functional_groups.get("has_amine", False):
        spots.append(
            {
                "site": "Amine nitrogen",
                "pathway": "N-dealkylation",
                "enzyme": "CYP2D6/CYP3A4",
                "vulnerability_score": round(0.80 * logp_factor * low_mw_factor, 4),
            }
        )

    if functional_groups.get("has_aliphatic_ch", False):
        spots.append(
            {
                "site": "Aliphatic C-H",
                "pathway": "Aliphatic hydroxylation",
                "enzyme": "CYP3A4",
                "vulnerability_score": round(0.70 * logp_factor * low_mw_factor, 4),
            }
        )

    if functional_groups.get("has_ether", False):
        spots.append(
            {
                "site": "Ether oxygen",
                "pathway": "O-dealkylation",
                "enzyme": "CYP2D6",
                "vulnerability_score": round(0.65 * logp_factor * low_mw_factor, 4),
            }
        )

    if functional_groups.get("has_sulfur", False):
        spots.append(
            {
                "site": "Sulfur atom",
                "pathway": "Sulfoxidation",
                "enzyme": "CYP3A4/FMO",
                "vulnerability_score": round(0.60 * logp_factor * low_mw_factor, 4),
            }
        )

    # Phase II always present but lower vulnerability compared to CYP
    spots.append(
        {
            "site": "Hydroxyl/carboxyl",
            "pathway": "Glucuronidation",
            "enzyme": "UGT",
            "vulnerability_score": round(0.50 * low_mw_factor, 4),
        }
    )

    if mw_Da < 500:
        spots.append(
            {
                "site": "Phenol/alcohol",
                "pathway": "Sulfation",
                "enzyme": "SULT",
                "vulnerability_score": round(0.40 * low_mw_factor, 4),
            }
        )

    spots.sort(key=lambda x: x["vulnerability_score"], reverse=True)
    return spots
