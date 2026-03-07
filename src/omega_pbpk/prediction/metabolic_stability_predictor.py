"""Metabolic stability prediction — Phase 798.

Predicts microsomal half-life and intrinsic clearance from physicochemical
descriptors using an empirical model.

References
----------
- Houston JB & Carlile DJ, Drug Metab Rev. 1997 (CLint scaling)
- Sun H et al., AAPS J. 2004 (MPPGL estimates)
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "MetabolicStabilityResult",
    "predict_metabolic_stability",
    "screen_metabolic_stability",
]

# Physiological constants for scaling to in vivo
_MPPGL_MG_PER_G = 45.0  # mg microsomal protein per g liver
_LIVER_WEIGHT_G = 1500.0  # human liver weight in grams
_QH_L_PER_H = 97.0  # hepatic blood flow (L/h)

# CLint_mic limits (µL/min/mg protein)
_CLINT_MIN = 0.5
_CLINT_MAX = 5000.0


@dataclass(frozen=True)
class MetabolicStabilityResult:
    """Metabolic stability prediction result (Phase 798)."""

    compound_name: str
    logp: float
    mw_da: float
    n_cyp3a4_sites: int
    n_cyp2d6_sites: int
    n_cyp2c9_sites: int
    predicted_clint_mic_uL_per_min_per_mg: float
    predicted_t_half_mic_min: float
    metabolic_stability_class: str  # "high" (>60 min), "moderate" (30-60 min), "low" (<30 min)
    primary_cyp: str
    n_metabolic_alerts: int
    scalable_to_human_cl_L_per_h: float
    notes: str


def _classify_stability(t_half_min: float) -> str:
    """Classify metabolic stability based on microsomal half-life."""
    if t_half_min > 60.0:
        return "high"
    elif t_half_min >= 30.0:
        return "moderate"
    else:
        return "low"


def _determine_primary_cyp(
    n_cyp3a4: int,
    n_cyp2d6: int,
    n_cyp2c9: int,
    has_ester: bool,
) -> str:
    """Determine primary metabolizing enzyme."""
    # Weight contributions
    contributions = {
        "CYP3A4": n_cyp3a4 * 0.3,
        "CYP2D6": n_cyp2d6 * 0.2,
        "CYP2C9": n_cyp2c9 * 0.15,
    }
    if has_ester:
        contributions["Esterase"] = 0.1

    max_contrib = max(contributions.values())
    if max_contrib <= 0.0:
        return "Non-CYP"

    for enzyme, contrib in contributions.items():
        if contrib == max_contrib:
            return enzyme
    return "Non-CYP"


def predict_metabolic_stability(
    compound_name: str,
    logp: float,
    mw_da: float,
    fup: float = 0.1,
    n_cyp3a4_sites: int = 0,
    n_cyp2d6_sites: int = 0,
    n_cyp2c9_sites: int = 0,
    has_ester: bool = False,
    has_amide: bool = False,
    has_michael_acceptor: bool = False,
    n_aromatic_rings: int = 1,
) -> MetabolicStabilityResult:
    """Predict microsomal metabolic stability.

    Parameters
    ----------
    compound_name : Compound identifier.
    logp : Calculated logP (octanol-water partition coefficient).
    mw_da : Molecular weight in Daltons. Must be > 0.
    fup : Fraction unbound in plasma (0, 1]. Must be in (0, 1].
    n_cyp3a4_sites : Number of predicted CYP3A4 metabolic soft spots.
    n_cyp2d6_sites : Number of predicted CYP2D6 metabolic soft spots.
    n_cyp2c9_sites : Number of predicted CYP2C9 metabolic soft spots.
    has_ester : Whether the compound contains an ester group.
    has_amide : Whether the compound contains an amide group (metabolically resistant).
    has_michael_acceptor : Whether the compound is a Michael acceptor (rapid metabolism).
    n_aromatic_rings : Number of aromatic rings.

    Returns
    -------
    MetabolicStabilityResult
    """
    if mw_da <= 0:
        raise ValueError("mw_da must be > 0")
    if fup <= 0 or fup > 1:
        raise ValueError("fup must be in (0, 1]")

    # Base CLint from logP and MW
    # CLint_base = 10^(0.4*logP - 0.005*mw + 1.0)
    clint_base = 10.0 ** (0.4 * logp - 0.005 * mw_da + 1.0)

    # CYP contributions
    clint_total = clint_base
    clint_total += n_cyp3a4_sites * clint_base * 0.3
    clint_total += n_cyp2d6_sites * clint_base * 0.2
    clint_total += n_cyp2c9_sites * clint_base * 0.15

    # Structural modifiers
    if has_ester:
        clint_total += clint_base * 0.1  # esterase cleavage
    if has_amide:
        clint_total -= clint_base * 0.1  # metabolic resistance
    if has_michael_acceptor:
        clint_total *= 1.5  # rapid metabolism flag

    # Clamp to physiological range
    clint_mic = max(_CLINT_MIN, min(_CLINT_MAX, clint_total))

    # Microsomal half-life: t½ = 693 / CLint (µL/min/mg → min)
    # Derived from: t½ = 0.693 / k_elim, k_elim = CLint / V_mic, V_mic=1 mL/mg protein
    t_half_mic = 693.0 / clint_mic

    # Scale to in vivo hepatic CL (well-stirred model)
    # CLint_scaled [µL/min/mg] * MPPGL [mg/g] * liver_weight [g] = CLint_liver [µL/min]
    # Convert to L/h: * 60 min/h / 1e6 µL/L
    clint_liver_uL_per_min = clint_mic * _MPPGL_MG_PER_G * _LIVER_WEIGHT_G
    clint_liver_L_per_h = clint_liver_uL_per_min * 60.0 / 1e6

    # Well-stirred model: CLhep = fup * CLint_liver / (Qh + fup * CLint_liver)
    clhep_L_per_h = (_QH_L_PER_H * fup * clint_liver_L_per_h) / (
        _QH_L_PER_H + fup * clint_liver_L_per_h
    )

    # Stability classification
    stability_class = _classify_stability(t_half_mic)

    # Primary CYP
    primary_cyp = _determine_primary_cyp(n_cyp3a4_sites, n_cyp2d6_sites, n_cyp2c9_sites, has_ester)

    # Count metabolic alerts
    n_alerts = 0
    if has_ester:
        n_alerts += 1
    if has_michael_acceptor:
        n_alerts += 1
    if n_aromatic_rings >= 3:
        n_alerts += 1
    if logp > 4.0:
        n_alerts += 1

    notes = (
        f"{compound_name}: CLint={round(clint_mic, 2)} µL/min/mg, "
        f"t½={round(t_half_mic, 1)} min ({stability_class}), "
        f"primary: {primary_cyp}, "
        f"CLhep={round(clhep_L_per_h, 3)} L/h."
    )

    return MetabolicStabilityResult(
        compound_name=compound_name,
        logp=logp,
        mw_da=mw_da,
        n_cyp3a4_sites=n_cyp3a4_sites,
        n_cyp2d6_sites=n_cyp2d6_sites,
        n_cyp2c9_sites=n_cyp2c9_sites,
        predicted_clint_mic_uL_per_min_per_mg=round(clint_mic, 4),
        predicted_t_half_mic_min=round(t_half_mic, 4),
        metabolic_stability_class=stability_class,
        primary_cyp=primary_cyp,
        n_metabolic_alerts=n_alerts,
        scalable_to_human_cl_L_per_h=round(clhep_L_per_h, 6),
        notes=notes,
    )


def screen_metabolic_stability(compounds: list) -> list:
    """Screen a list of compounds for metabolic stability.

    Parameters
    ----------
    compounds : List of dicts, each with keys matching predict_metabolic_stability
                parameters. Required keys: compound_name, logp, mw_da.

    Returns
    -------
    List of MetabolicStabilityResult sorted by predicted_t_half_mic_min descending
    (most stable first).
    """
    results: list[MetabolicStabilityResult] = []
    for cpd in compounds:
        result = predict_metabolic_stability(
            compound_name=cpd["compound_name"],
            logp=cpd["logp"],
            mw_da=cpd["mw_da"],
            fup=cpd.get("fup", 0.1),
            n_cyp3a4_sites=cpd.get("n_cyp3a4_sites", 0),
            n_cyp2d6_sites=cpd.get("n_cyp2d6_sites", 0),
            n_cyp2c9_sites=cpd.get("n_cyp2c9_sites", 0),
            has_ester=cpd.get("has_ester", False),
            has_amide=cpd.get("has_amide", False),
            has_michael_acceptor=cpd.get("has_michael_acceptor", False),
            n_aromatic_rings=cpd.get("n_aromatic_rings", 1),
        )
        results.append(result)

    # Sort by t_half descending (most stable first)
    results.sort(key=lambda r: r.predicted_t_half_mic_min, reverse=True)
    return results
