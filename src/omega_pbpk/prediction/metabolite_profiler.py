"""Drug metabolite profiling from structure — Phase 501.

Predicts major metabolic pathways and relative abundance of metabolites
based on fractional metabolism (fm) parameters for major enzyme systems.

References
----------
- FDA 2020 MIST Guidance for Industry: Safety Testing of Drug Metabolites
- Ziegler DM, Annu Rev Pharmacol Toxicol. 1993
- Miners JO, Mackenzie PI, Adv Pharmacol. 1991
"""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = [
    "MetaboliteProfileResult",
    "profile_metabolites",
    "predict_metabolite_mw",
    "calculate_mist_coverage",
]

# ---------------------------------------------------------------------------
# MW changes for common reaction types (Da)
# ---------------------------------------------------------------------------

REACTION_MW_DELTA: dict[str, float] = {
    "hydroxylation": 16.0,
    "demethylation": -14.0,
    "glucuronidation": 176.0,
    "sulfation": 80.0,
    "n_oxidation": 16.0,
    "dealkylation": -28.0,  # typical -28 to -56 Da; use midpoint conservative estimate
    "epoxidation": 16.0,
}

# ---------------------------------------------------------------------------
# Enzyme → primary reaction mapping
# ---------------------------------------------------------------------------

ENZYME_REACTIONS: dict[str, list[str]] = {
    "CYP3A4": ["hydroxylation", "dealkylation"],
    "CYP2D6": ["hydroxylation", "demethylation"],
    "CYP1A2": ["n_oxidation"],
    "CYP2C9": ["hydroxylation"],
    "UGT": ["glucuronidation"],
}

# Known active metabolite production likelihood per enzyme
ACTIVE_METABOLITE_RISK: dict[str, str] = {
    "CYP2D6": "moderate",  # codeine→morphine; tamoxifen→endoxifen
    "CYP3A4": "low",
    "CYP1A2": "low",
    "CYP2C9": "low",
    "UGT": "low",
    "other": "low",
}


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------


@dataclass
class MetaboliteProfileResult:
    """Result of metabolite pathway profiling.

    Attributes
    ----------
    compound_name : Compound identifier.
    smiles : SMILES string (may be empty).
    parent_mw : Molecular weight of parent compound (Da).
    pathways : List of predicted metabolic pathway dicts, each with keys:
        'enzyme', 'reaction', 'estimated_fraction', 'metabolite_mw',
        'active_risk'.
    n_pathways : Number of pathways with fm > 0.
    major_pathway : Enzyme with highest fm.
    metabolic_soft_spots : List of major enzyme names (fm >= 0.1).
    active_metabolite_risk : Overall active metabolite risk
        ('low', 'moderate', 'high').
    notes : Free-text summary.
    """

    compound_name: str
    smiles: str
    parent_mw: float
    pathways: list[dict] = field(default_factory=list)
    n_pathways: int = 0
    major_pathway: str = ""
    metabolic_soft_spots: list[str] = field(default_factory=list)
    active_metabolite_risk: str = "low"
    notes: str = ""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def predict_metabolite_mw(parent_mw: float, reaction_type: str) -> float:
    """Predict metabolite MW from parent MW and reaction type.

    Parameters
    ----------
    parent_mw:
        Parent compound molecular weight in Da.
    reaction_type:
        Reaction type string; must be a key in ``REACTION_MW_DELTA``.

    Returns
    -------
    float
        Estimated metabolite MW in Da.

    Raises
    ------
    ValueError
        If *reaction_type* is not recognised.
    """
    if parent_mw <= 0:
        raise ValueError(f"parent_mw must be positive, got {parent_mw}")
    reaction_type = reaction_type.strip().lower()
    if reaction_type not in REACTION_MW_DELTA:
        raise ValueError(
            f"Unknown reaction type '{reaction_type}'. "
            f"Valid types: {sorted(REACTION_MW_DELTA)}"
        )
    return parent_mw + REACTION_MW_DELTA[reaction_type]


def profile_metabolites(
    compound_name: str,
    smiles: str,
    mw: float,
    logP: float,  # noqa: N803 — domain convention
    fm_cyp3a4: float = 0.4,
    fm_cyp2d6: float = 0.2,
    fm_cyp1a2: float = 0.1,
    fm_cyp2c9: float = 0.1,
    fm_ugt: float = 0.1,
    fm_other: float = 0.1,
) -> MetaboliteProfileResult:
    """Profile drug metabolites from physicochemical and fm parameters.

    Parameters
    ----------
    compound_name:
        Human-readable compound identifier.
    smiles:
        SMILES string (used for record-keeping; not parsed).
    mw:
        Molecular weight of parent compound (Da).
    logP:
        Partition coefficient (influences soft-spot annotation).
    fm_cyp3a4:
        Fraction metabolised by CYP3A4 (0–1).
    fm_cyp2d6:
        Fraction metabolised by CYP2D6 (0–1).
    fm_cyp1a2:
        Fraction metabolised by CYP1A2 (0–1).
    fm_cyp2c9:
        Fraction metabolised by CYP2C9 (0–1).
    fm_ugt:
        Fraction metabolised by UGT (0–1).
    fm_other:
        Fraction via other pathways (0–1).

    Returns
    -------
    MetaboliteProfileResult
    """
    # ------------------------------------------------------------------
    # Input validation
    # ------------------------------------------------------------------
    if mw <= 0:
        raise ValueError(f"mw must be positive, got {mw}")

    fm_values = {
        "CYP3A4": fm_cyp3a4,
        "CYP2D6": fm_cyp2d6,
        "CYP1A2": fm_cyp1a2,
        "CYP2C9": fm_cyp2c9,
        "UGT": fm_ugt,
        "other": fm_other,
    }
    for name, val in fm_values.items():
        if val < 0 or val > 1:
            raise ValueError(
                f"fm_{name.lower()} must be between 0 and 1, got {val}"
            )

    total_fm = sum(fm_values.values())
    if total_fm > 1.0 + 1e-9:
        raise ValueError(
            f"Sum of all fm values must be ≤ 1.0, got {total_fm:.4f}"
        )

    # ------------------------------------------------------------------
    # Build pathway list
    # ------------------------------------------------------------------
    pathways: list[dict] = []

    for enzyme, fm in fm_values.items():
        if fm <= 0:
            continue

        if enzyme in ENZYME_REACTIONS:
            reactions = ENZYME_REACTIONS[enzyme]
        else:
            reactions = ["hydroxylation"]  # fallback for "other"

        # Use first (primary) reaction for this enzyme
        primary_reaction = reactions[0]
        met_mw = predict_metabolite_mw(mw, primary_reaction)
        active_risk = ACTIVE_METABOLITE_RISK.get(enzyme, "low")

        pathways.append(
            {
                "enzyme": enzyme,
                "reaction": primary_reaction,
                "estimated_fraction": round(fm, 4),
                "metabolite_mw": round(met_mw, 2),
                "active_risk": active_risk,
            }
        )

    # Sort by fraction descending
    pathways.sort(key=lambda p: p["estimated_fraction"], reverse=True)

    # ------------------------------------------------------------------
    # Derive summary fields
    # ------------------------------------------------------------------
    n_pathways = len(pathways)

    major_pathway = pathways[0]["enzyme"] if pathways else "none"

    # Metabolic soft spots: enzymes with fm ≥ 0.1
    soft_spots = [p["enzyme"] for p in pathways if p["estimated_fraction"] >= 0.1]

    # Active metabolite risk: highest risk among active pathways
    risk_rank = {"low": 0, "moderate": 1, "high": 2}
    overall_risk = "low"
    for p in pathways:
        er = p["active_risk"]
        if risk_rank.get(er, 0) > risk_rank.get(overall_risk, 0):
            overall_risk = er

    # Notes
    note_parts = [
        f"Major pathway: {major_pathway} ({fm_values.get(major_pathway, 0):.0%}).",
        f"Soft spots: {', '.join(soft_spots) if soft_spots else 'none'}.",
        f"Active metabolite risk: {overall_risk}.",
    ]
    if logP > 3.5:
        note_parts.append(
            "High logP suggests extensive CYP oxidative metabolism likely."
        )
    if fm_ugt >= 0.2:
        note_parts.append(
            "Significant glucuronidation — monitor UGT polymorphisms."
        )

    return MetaboliteProfileResult(
        compound_name=compound_name,
        smiles=smiles,
        parent_mw=mw,
        pathways=pathways,
        n_pathways=n_pathways,
        major_pathway=major_pathway,
        metabolic_soft_spots=soft_spots,
        active_metabolite_risk=overall_risk,
        notes=" ".join(note_parts),
    )


def calculate_mist_coverage(metabolite_profiles: list[dict]) -> dict:
    """Determine FDA MIST coverage for a set of metabolite exposures.

    A metabolite requires MIST safety testing when its AUC is ≥ 10 % of
    parent AUC (FDA 2020 MIST guidance).

    Parameters
    ----------
    metabolite_profiles:
        List of dicts, each with keys:
        - 'name' (str): metabolite identifier
        - 'fraction' (float): estimated fraction of parent AUC (0–1)

    Returns
    -------
    dict
        Keys:
        - 'mist_required' (list[str]): metabolite names requiring testing
        - 'mist_not_required' (list[str]): metabolite names below threshold
        - 'n_mist_required' (int)
        - 'threshold' (float): 0.10
        - 'notes' (str)

    Raises
    ------
    ValueError
        If any fraction is negative.
    """
    if not metabolite_profiles:
        return {
            "mist_required": [],
            "mist_not_required": [],
            "n_mist_required": 0,
            "threshold": 0.10,
            "notes": "No metabolite profiles provided.",
        }

    mist_threshold = 0.10
    mist_required: list[str] = []
    mist_not_required: list[str] = []

    for entry in metabolite_profiles:
        name = entry.get("name", "unknown")
        fraction = entry.get("fraction", 0.0)
        if fraction < 0:
            raise ValueError(
                f"Fraction for metabolite '{name}' must be ≥ 0, got {fraction}"
            )
        if fraction >= mist_threshold:
            mist_required.append(name)
        else:
            mist_not_required.append(name)

    n_required = len(mist_required)
    if n_required == 0:
        notes = "No metabolites meet FDA MIST threshold (≥10% of parent AUC)."
    else:
        listed = ", ".join(mist_required)
        notes = (
            f"{n_required} metabolite(s) require MIST safety testing: {listed}."
        )

    return {
        "mist_required": mist_required,
        "mist_not_required": mist_not_required,
        "n_mist_required": n_required,
        "threshold": mist_threshold,
        "notes": notes,
    }
