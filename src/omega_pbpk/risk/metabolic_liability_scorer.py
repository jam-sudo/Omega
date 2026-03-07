"""
Phase 660 — Metabolic Liability Scorer

Score metabolic liability of drugs based on structural features and in vitro data.

Metabolic liability = susceptibility to rapid metabolism (high CLint) and
reactive metabolite formation risk.  High liability → rapid clearance + DILI risk.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MetabolicLiabilityResult:
    """Comprehensive metabolic liability assessment for a single compound."""

    compound_name: str
    logP: float
    mw: float
    n_metabolic_soft_spots: int         # estimated number of CYP-labile groups
    n_reactive_alerts: int              # alerts for reactive metabolite formation
    clint_predicted_uL_per_min: float   # predicted microsomal CLint (µL/min/mg protein)
    hepatic_er_predicted: float         # predicted hepatic extraction ratio [0, 1]
    t_half_in_vitro_min: float          # predicted in vitro t½ (min)
    liability_score: float              # 0–100 (higher = more liable)
    liability_class: str                # "low" / "moderate" / "high" / "very_high"
    rapid_clearance_risk: bool          # True if ER > 0.7
    reactive_metabolite_risk: bool      # True if n_reactive_alerts > 0
    primary_liability: str              # "rapid_clearance", "reactive_metabolite", "both", "none"
    optimization_suggestions: str       # human-readable notes on how to reduce liability
    notes: str


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _predict_clint(
    logP: float,
    n_unshielded_positions: int,
    n_aromatic_rings: int,
    psa: float,
) -> float:
    """Predict microsomal CLint (µL/min/mg protein) from physicochemical descriptors."""
    base = max(1.0, logP * 8.0 + n_unshielded_positions * 5.0)
    aromatic_penalty = n_aromatic_rings * 3.0
    clint = base + aromatic_penalty - psa * 0.05
    return max(0.5, min(clint, 500.0))


def _count_soft_spots(
    n_unshielded_positions: int,
    n_aromatic_rings: int,
    logP: float,
) -> int:
    """Estimate number of CYP-labile (soft-spot) positions."""
    base = n_unshielded_positions
    # aromatic rings are labile
    base += n_aromatic_rings
    # high logP increases lipophilic soft spots
    if logP > 3.0:
        base += 1
    return max(0, base)


def _hepatic_er(clint_uL_min: float) -> float:
    """Compute hepatic extraction ratio using the well-stirred model.

    Q_H = 90 L/h = 1500 mL/min
    CLint_body (mL/min) = clint (µL/min/mg) * 40 mg protein/g liver
                          * 25.7 g liver (per 70 kg) → scale to per-person
    Simplified: CLint_body_mLmin = clint * 0.06 * 40
    ER = CLint_body / (CLint_body + Q_H_mLmin)
    """
    q_h_mL_min = 1500.0  # hepatic blood flow in mL/min
    clint_body = clint_uL_min * 0.06 * 40.0  # mL/min/person
    er = clint_body / (clint_body + q_h_mL_min)
    return max(0.0, min(er, 1.0))


def _t_half_in_vitro(clint_uL_min: float) -> float:
    """In vitro half-life (min) from CLint.

    Assume 1 mg/mL protein, 0.5 mL incubation volume:
        k_el = CLint * V_inc_mL / (protein_mg * 1000)  [1/min]  -- actually simpler:
    Use: t½ = ln(2) * 1000 / clint
    """
    if clint_uL_min <= 0.0:
        return float("inf")
    return math.log(2) * 1000.0 / clint_uL_min


def _liability_class(score: float) -> str:
    if score < 25.0:
        return "low"
    if score < 60.0:
        return "moderate"
    if score < 80.0:
        return "high"
    return "very_high"


def _primary_liability(rapid: bool, reactive: bool) -> str:
    if rapid and reactive:
        return "both"
    if rapid:
        return "rapid_clearance"
    if reactive:
        return "reactive_metabolite"
    return "none"


def _optimization_suggestions(
    logP: float,
    n_aromatic_rings: int,
    n_unshielded_positions: int,
    has_michael_acceptor: bool,
    has_epoxide_precursor: bool,
    has_aniline: bool,
    has_nitro_group: bool,
    has_hydrazine: bool,
    er: float,
    n_reactive_alerts: int,
) -> str:
    suggestions: list[str] = []

    if logP > 3.0:
        suggestions.append("Reduce logP (<3) to decrease CYP-mediated oxidative metabolism.")
    if n_aromatic_rings > 1:
        suggestions.append(
            "Replace one or more aromatic rings with saturated/heteroaromatic analogues "
            "('magic methyl' or ring saturation) to reduce metabolic soft spots."
        )
    if n_unshielded_positions > 1:
        suggestions.append(
            "Block unshielded metabolic positions with fluorine or deuterium substitution."
        )
    if has_aniline:
        suggestions.append(
            "Eliminate aniline motif or add electron-withdrawing groups to reduce N-hydroxylation risk."
        )
    if has_nitro_group:
        suggestions.append(
            "Replace nitro group with isostere (e.g., CN, CF3) to prevent nitroreduction."
        )
    if has_michael_acceptor:
        suggestions.append(
            "Remove or cap Michael acceptor to reduce covalent protein binding."
        )
    if has_epoxide_precursor:
        suggestions.append(
            "Modify epoxide-forming substructure (e.g., block oxidation site) to prevent "
            "reactive epoxide formation."
        )
    if has_hydrazine:
        suggestions.append(
            "Replace hydrazine with bioisostere; hydrazines form reactive diazonium ions."
        )
    if er > 0.7:
        suggestions.append(
            "High hepatic extraction: consider prodrug strategy or alternative route of "
            "administration to improve oral bioavailability."
        )

    if not suggestions:
        suggestions.append(
            "No major liability flags identified; maintain current scaffold during optimization."
        )

    return " | ".join(suggestions)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def score_metabolic_liability(
    compound_name: str,
    logP: float,
    mw: float,
    psa: float,
    n_hbd: int = 2,
    n_hba: int = 4,
    n_aromatic_rings: int = 1,
    n_unshielded_positions: int = 2,
    has_michael_acceptor: bool = False,
    has_epoxide_precursor: bool = False,
    has_aniline: bool = False,
    has_nitro_group: bool = False,
    has_hydrazine: bool = False,
    measured_clint: float | None = None,
) -> MetabolicLiabilityResult:
    """Score metabolic liability for a single compound.

    Parameters
    ----------
    compound_name:
        Identifier / name for the compound.
    logP:
        Logarithm of octanol-water partition coefficient.
    mw:
        Molecular weight (Da).  Must be > 0.
    psa:
        Polar surface area (Å²).  Must be >= 0.
    n_hbd:
        Number of hydrogen bond donors.
    n_hba:
        Number of hydrogen bond acceptors.
    n_aromatic_rings:
        Number of aromatic rings.
    n_unshielded_positions:
        Number of positions vulnerable to CYP oxidative attack.
    has_michael_acceptor:
        True if the compound contains a Michael acceptor motif.
    has_epoxide_precursor:
        True if the compound can form reactive epoxides in vivo.
    has_aniline:
        True if an aniline (primary aromatic amine) moiety is present.
    has_nitro_group:
        True if a nitro group is present.
    has_hydrazine:
        True if a hydrazine moiety is present.
    measured_clint:
        If provided, overrides the predicted CLint (µL/min/mg protein).

    Returns
    -------
    MetabolicLiabilityResult
    """
    # Validation
    if mw <= 0.0:
        raise ValueError(f"mw must be > 0, got {mw}")
    if psa < 0.0:
        raise ValueError(f"psa must be >= 0, got {psa}")
    if n_hbd < 0:
        raise ValueError(f"n_hbd must be >= 0, got {n_hbd}")
    if n_aromatic_rings < 0:
        raise ValueError(f"n_aromatic_rings must be >= 0, got {n_aromatic_rings}")
    if measured_clint is not None and measured_clint < 0.0:
        raise ValueError(f"measured_clint must be >= 0 if provided, got {measured_clint}")

    # CLint
    if measured_clint is not None:
        clint = measured_clint
    else:
        clint = _predict_clint(logP, n_unshielded_positions, n_aromatic_rings, psa)

    # Soft spots
    n_soft = _count_soft_spots(n_unshielded_positions, n_aromatic_rings, logP)

    # Reactive alerts
    n_reactive = (
        (1 if has_michael_acceptor else 0)
        + (1 if has_epoxide_precursor else 0)
        + (1 if has_aniline else 0)
        + (1 if has_nitro_group else 0)
        + (1 if has_hydrazine else 0)
    )

    # Hepatic ER and t½
    er = _hepatic_er(clint)
    t_half = _t_half_in_vitro(clint)

    # Liability score (0–100)
    s_clint = min(40.0, clint / 10.0)
    s_react = n_reactive * 10.0
    s_er = max(0.0, (er - 0.3) / 0.7 * 20.0)
    score = min(100.0, s_clint + s_react + s_er)

    # Classification
    l_class = _liability_class(score)
    rapid = er > 0.7
    reactive = n_reactive > 0
    primary = _primary_liability(rapid, reactive)

    suggestions = _optimization_suggestions(
        logP=logP,
        n_aromatic_rings=n_aromatic_rings,
        n_unshielded_positions=n_unshielded_positions,
        has_michael_acceptor=has_michael_acceptor,
        has_epoxide_precursor=has_epoxide_precursor,
        has_aniline=has_aniline,
        has_nitro_group=has_nitro_group,
        has_hydrazine=has_hydrazine,
        er=er,
        n_reactive_alerts=n_reactive,
    )

    notes = (
        f"Compound '{compound_name}': logP={logP:.2f}, MW={mw:.1f} Da, PSA={psa:.1f} Å², "
        f"CLint={'measured' if measured_clint is not None else 'predicted'}={clint:.2f} µL/min/mg, "
        f"ER={er:.3f}, t½_iv={t_half:.1f} min, score={score:.1f} ({l_class})"
    )

    return MetabolicLiabilityResult(
        compound_name=compound_name,
        logP=logP,
        mw=mw,
        n_metabolic_soft_spots=n_soft,
        n_reactive_alerts=n_reactive,
        clint_predicted_uL_per_min=clint,
        hepatic_er_predicted=er,
        t_half_in_vitro_min=t_half,
        liability_score=score,
        liability_class=l_class,
        rapid_clearance_risk=rapid,
        reactive_metabolite_risk=reactive,
        primary_liability=primary,
        optimization_suggestions=suggestions,
        notes=notes,
    )


def screen_metabolic_liability(compounds: list[dict]) -> list[MetabolicLiabilityResult]:
    """Screen multiple compounds and rank by liability_score descending.

    Parameters
    ----------
    compounds:
        List of dicts, each with keyword arguments for :func:`score_metabolic_liability`.

    Returns
    -------
    list[MetabolicLiabilityResult]
        Sorted by liability_score descending.
    """
    if not compounds:
        return []

    results = [score_metabolic_liability(**c) for c in compounds]
    results.sort(key=lambda r: r.liability_score, reverse=True)
    return results
