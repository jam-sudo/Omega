"""Reactive metabolite burden score across CYP pathways.

Models the cumulative burden of reactive metabolite formation as a function of
fraction metabolized (fm) per CYP isoform, intrinsic clearance, daily dose, and
lipophilicity.

References
----------
Nakayama 2009 (Drug Metab Dispos), Smith 2016 (Expert Opin Drug Metab Toxicol),
FDA DILI guidance 2009.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "MetaboliteBurdenResult",
    "compute_metabolite_burden",
    "screen_compounds",
]

# CYP pathways considered
_PATHWAYS: tuple[str, ...] = ("CYP3A4", "CYP2D6", "CYP2C9", "CYP1A2")


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MetaboliteBurdenResult:
    """Result of reactive metabolite burden assessment."""

    compound_name: str
    daily_dose_mg: float
    fm_cyp3a4: float
    fm_cyp2d6: float
    fm_cyp2c9: float
    fm_cyp1a2: float
    clint_L_per_h: float
    mw: float
    logP: float
    pathway_burdens: dict  # dict[str, float] per CYP pathway
    total_burden: float
    risk_category: str  # low / moderate / high
    dominant_pathway: str  # CYP with highest burden
    notes: str


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_metabolite_burden(
    compound_name: str,
    daily_dose_mg: float,
    fm_cyp3a4: float,
    fm_cyp2d6: float,
    fm_cyp2c9: float,
    fm_cyp1a2: float,
    clint_L_per_h: float,
    mw: float,
    logP: float,
) -> MetaboliteBurdenResult:
    """Compute reactive metabolite burden across CYP pathways.

    The pathway burden for each CYP is:

        burden_i = fm_i * clint * daily_dose_mg * logP_factor

    where:

        logP_factor = 1 + max(0, logP - 2) * 0.1

    Lipophilic drugs (logP > 2) tend to form more reactive metabolites due to
    increased CYP-mediated oxidation of hydrophobic regions.

    Risk categories are based on total burden:
        - low      : total_burden < 1.0
        - moderate : 1.0 <= total_burden < 5.0
        - high     : total_burden >= 5.0

    Parameters
    ----------
    compound_name : str
        Name or identifier of the compound.
    daily_dose_mg : float
        Daily dose in mg (must be > 0).
    fm_cyp3a4 : float
        Fraction metabolized by CYP3A4 (0-1).
    fm_cyp2d6 : float
        Fraction metabolized by CYP2D6 (0-1).
    fm_cyp2c9 : float
        Fraction metabolized by CYP2C9 (0-1).
    fm_cyp1a2 : float
        Fraction metabolized by CYP1A2 (0-1).
    clint_L_per_h : float
        Total intrinsic clearance (L/h, must be > 0).
    mw : float
        Molecular weight in Da (must be > 0).
    logP : float
        Octanol-water partition coefficient.

    Returns
    -------
    MetaboliteBurdenResult
    """
    # --- Validation ---
    if daily_dose_mg <= 0:
        raise ValueError(f"daily_dose_mg must be > 0, got {daily_dose_mg}")
    if clint_L_per_h <= 0:
        raise ValueError(f"clint_L_per_h must be > 0, got {clint_L_per_h}")
    if mw <= 0:
        raise ValueError(f"mw must be > 0, got {mw}")

    fm_vals = {
        "CYP3A4": fm_cyp3a4,
        "CYP2D6": fm_cyp2d6,
        "CYP2C9": fm_cyp2c9,
        "CYP1A2": fm_cyp1a2,
    }
    for cyp, fm in fm_vals.items():
        if not (0.0 <= fm <= 1.0):
            raise ValueError(f"{cyp} fm must be in [0, 1], got {fm}")

    fm_sum = sum(fm_vals.values())
    if fm_sum > 1.0 + 1e-9:
        raise ValueError(f"Sum of fm values must be <= 1.0, got {fm_sum:.6f}")

    # --- logP factor ---
    logp_factor = 1.0 + max(0.0, logP - 2.0) * 0.1

    # --- Per-pathway burdens ---
    pathway_burdens: dict[str, float] = {}
    for cyp, fm in fm_vals.items():
        pathway_burdens[cyp] = fm * clint_L_per_h * daily_dose_mg * logp_factor

    total_burden = sum(pathway_burdens.values())

    # --- Risk category ---
    if total_burden < 1.0:
        risk_category = "low"
    elif total_burden < 5.0:
        risk_category = "moderate"
    else:
        risk_category = "high"

    # --- Dominant pathway ---
    dominant_pathway = max(pathway_burdens, key=lambda k: pathway_burdens[k])

    # --- Notes ---
    note_parts: list[str] = [
        f"total_burden={total_burden:.3f}",
        f"risk={risk_category}",
        f"dominant={dominant_pathway} ({pathway_burdens[dominant_pathway]:.3f})",
        f"logP_factor={logp_factor:.3f}",
    ]
    if fm_sum < 0.5:
        note_parts.append("Low total fm (<50%): non-CYP pathways may dominate")
    notes = "; ".join(note_parts)

    return MetaboliteBurdenResult(
        compound_name=compound_name,
        daily_dose_mg=daily_dose_mg,
        fm_cyp3a4=fm_cyp3a4,
        fm_cyp2d6=fm_cyp2d6,
        fm_cyp2c9=fm_cyp2c9,
        fm_cyp1a2=fm_cyp1a2,
        clint_L_per_h=clint_L_per_h,
        mw=mw,
        logP=logP,
        pathway_burdens=pathway_burdens,
        total_burden=total_burden,
        risk_category=risk_category,
        dominant_pathway=dominant_pathway,
        notes=notes,
    )


def screen_compounds(compounds: list[dict]) -> list[MetaboliteBurdenResult]:
    """Screen a list of compounds for reactive metabolite burden.

    Parameters
    ----------
    compounds : list[dict]
        Each dict must contain keys:
        ``compound_name``, ``daily_dose_mg``, ``fm_cyp3a4``, ``fm_cyp2d6``,
        ``fm_cyp2c9``, ``fm_cyp1a2``, ``clint_L_per_h``, ``mw``, ``logP``.

    Returns
    -------
    list[MetaboliteBurdenResult]
        Sorted by ``total_burden`` descending.
    """
    results: list[MetaboliteBurdenResult] = []
    for c in compounds:
        result = compute_metabolite_burden(
            compound_name=c["compound_name"],
            daily_dose_mg=c["daily_dose_mg"],
            fm_cyp3a4=c["fm_cyp3a4"],
            fm_cyp2d6=c["fm_cyp2d6"],
            fm_cyp2c9=c["fm_cyp2c9"],
            fm_cyp1a2=c["fm_cyp1a2"],
            clint_L_per_h=c["clint_L_per_h"],
            mw=c["mw"],
            logP=c["logP"],
        )
        results.append(result)

    return sorted(results, key=lambda r: r.total_burden, reverse=True)
