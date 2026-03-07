"""Phase 1087 — Drug Metabolic Switching (CYP to UGT).

Predicts fractional metabolic pathway contributions when CYP enzymes
are saturated or inhibited, leading to increased UGT/SULT/NAT2 activity.
"""

from dataclasses import dataclass

# Default enzyme kinetic parameters (Vmax nmol/min, Km µM)
_ENZYME_PARAMS: dict[str, tuple[float, float]] = {
    "cyp3a4": (100.0, 10.0),
    "cyp2d6": (50.0, 5.0),
    "ugt1a1": (200.0, 100.0),
    "sult1a1": (30.0, 2.0),
    "nat2": (80.0, 20.0),
}

_CYP_PATHWAYS = {"cyp3a4", "cyp2d6"}
_ALT_PATHWAYS = {"ugt1a1", "sult1a1", "nat2"}
_ALL_PATHWAYS = list(_ENZYME_PARAMS.keys())

_VALID_PATHWAY_NAMES = {
    "cyp3a4": "cyp3a4",
    "cyp2d6": "cyp2d6",
    "ugt1a1": "ugt1a1",
    "sult1a1": "sult1a1",
    "nat2": "nat2",
}


@dataclass(frozen=True)
class MetabolicSwitchingResult:
    """Result of metabolic pathway switching prediction."""

    drug_name: str
    drug_conc_uM: float
    f_cyp3a4: float
    f_cyp2d6: float
    f_ugt1a1: float
    f_sult1a1: float
    f_nat2: float
    dominant_pathway: str
    cyp_fraction_total: float
    alt_pathway_fraction: float
    switching_occurred: bool
    metabolite_toxicity_risk: str
    notes: str


def _mm_velocity(vmax: float, km: float, conc_uM: float, inhib_fraction: float) -> float:
    """Michaelis-Menten velocity with competitive inhibition expressed as fraction inhibited."""
    # inhib_fraction is the fraction inhibited (0=none, 1=complete)
    # We interpret inhibition_fraction as: inhibition_factor = 1 - inhib_fraction
    # (i.e., the user says "I've inhibited 50% of this enzyme")
    inhibition_factor = 1.0 - inhib_fraction
    # Michaelis-Menten: v = Vmax / (1 + Km/C) * inhibition_factor
    if conc_uM <= 0.0:
        return 0.0
    v = vmax / (1.0 + km / conc_uM) * inhibition_factor
    return max(0.0, v)


def _compute_fractions(
    conc_uM: float,
    inhib: dict[str, float],
) -> dict[str, float]:
    """Compute fractional contribution of each pathway at given concentration."""
    velocities: dict[str, float] = {}
    for pathway, (vmax, km) in _ENZYME_PARAMS.items():
        inh = inhib.get(pathway, 0.0)
        velocities[pathway] = _mm_velocity(vmax, km, conc_uM, inh)

    total_v = sum(velocities.values())
    if total_v <= 0.0:
        # All pathways fully inhibited — distribute equally
        n = len(_ENZYME_PARAMS)
        return {p: 1.0 / n for p in _ENZYME_PARAMS}

    return {p: v / total_v for p, v in velocities.items()}


def _determine_switching(
    conc_uM: float,
    inhib: dict[str, float],
) -> bool:
    """Determine if metabolic switching occurred: alt_pathway_fraction at 10x conc > baseline by >20%."""
    baseline_fracs = _compute_fractions(conc_uM, inhib)
    highconc_fracs = _compute_fractions(conc_uM * 10.0, inhib)

    baseline_alt = sum(baseline_fracs[p] for p in _ALT_PATHWAYS)
    highconc_alt = sum(highconc_fracs[p] for p in _ALT_PATHWAYS)

    return (highconc_alt - baseline_alt) > 0.20


def predict_metabolic_switching(
    drug_name: str,
    drug_conc_uM: float,
    cyp3a4_inhibition_fraction: float = 0.0,
    cyp2d6_inhibition_fraction: float = 0.0,
    ugt_inhibition_fraction: float = 0.0,
    sult_inhibition_fraction: float = 0.0,
    nat2_inhibition_fraction: float = 0.0,
) -> MetabolicSwitchingResult:
    """Predict fractional metabolic pathway contributions for a drug.

    Parameters
    ----------
    drug_name : str
        Name of the drug.
    drug_conc_uM : float
        Drug concentration in µM (must be > 0).
    cyp3a4_inhibition_fraction : float
        Fraction of CYP3A4 inhibited [0, 1].
    cyp2d6_inhibition_fraction : float
        Fraction of CYP2D6 inhibited [0, 1].
    ugt_inhibition_fraction : float
        Fraction of UGT1A1 inhibited [0, 1].
    sult_inhibition_fraction : float
        Fraction of SULT1A1 inhibited [0, 1].
    nat2_inhibition_fraction : float
        Fraction of NAT2 inhibited [0, 1].

    Returns
    -------
    MetabolicSwitchingResult
    """
    # Validation
    if drug_conc_uM <= 0.0:
        raise ValueError(f"drug_conc_uM must be > 0, got {drug_conc_uM}")

    inhibition_map = {
        "cyp3a4": cyp3a4_inhibition_fraction,
        "cyp2d6": cyp2d6_inhibition_fraction,
        "ugt1a1": ugt_inhibition_fraction,
        "sult1a1": sult_inhibition_fraction,
        "nat2": nat2_inhibition_fraction,
    }
    for name, frac in inhibition_map.items():
        if not (0.0 <= frac <= 1.0):
            raise ValueError(f"{name}_inhibition_fraction must be in [0, 1], got {frac}")

    fracs = _compute_fractions(drug_conc_uM, inhibition_map)

    f_cyp3a4 = fracs["cyp3a4"]
    f_cyp2d6 = fracs["cyp2d6"]
    f_ugt1a1 = fracs["ugt1a1"]
    f_sult1a1 = fracs["sult1a1"]
    f_nat2 = fracs["nat2"]

    cyp_fraction_total = f_cyp3a4 + f_cyp2d6
    alt_pathway_fraction = f_ugt1a1 + f_sult1a1 + f_nat2

    # Dominant pathway
    dominant_pathway = max(fracs, key=lambda p: fracs[p])

    # Switching occurred
    switching_occurred = _determine_switching(drug_conc_uM, inhibition_map)

    # Metabolite toxicity risk based on alt_pathway_fraction
    if alt_pathway_fraction < 0.3:
        metabolite_toxicity_risk = "low"
    elif alt_pathway_fraction < 0.6:
        metabolite_toxicity_risk = "moderate"
    else:
        metabolite_toxicity_risk = "high"

    # Build notes
    notes_parts = []
    if switching_occurred:
        notes_parts.append("Metabolic switching detected at 10x concentration.")
    if cyp3a4_inhibition_fraction > 0.5:
        notes_parts.append("Significant CYP3A4 inhibition present.")
    if cyp2d6_inhibition_fraction > 0.5:
        notes_parts.append("Significant CYP2D6 inhibition present.")
    if alt_pathway_fraction > 0.5:
        notes_parts.append("Alternative pathways (UGT/SULT/NAT2) are dominant.")
    if not notes_parts:
        notes_parts.append("CYP pathways dominant; no significant switching predicted.")
    notes = " ".join(notes_parts)

    return MetabolicSwitchingResult(
        drug_name=drug_name,
        drug_conc_uM=drug_conc_uM,
        f_cyp3a4=f_cyp3a4,
        f_cyp2d6=f_cyp2d6,
        f_ugt1a1=f_ugt1a1,
        f_sult1a1=f_sult1a1,
        f_nat2=f_nat2,
        dominant_pathway=dominant_pathway,
        cyp_fraction_total=cyp_fraction_total,
        alt_pathway_fraction=alt_pathway_fraction,
        switching_occurred=switching_occurred,
        metabolite_toxicity_risk=metabolite_toxicity_risk,
        notes=notes,
    )


def compare_concentrations(
    drug_name: str,
    conc_list: list[float],
    cyp3a4_inhibition_fraction: float = 0.0,
    cyp2d6_inhibition_fraction: float = 0.0,
    ugt_inhibition_fraction: float = 0.0,
    sult_inhibition_fraction: float = 0.0,
    nat2_inhibition_fraction: float = 0.0,
) -> list[MetabolicSwitchingResult]:
    """Compare metabolic switching across a list of drug concentrations.

    Returns results sorted by alt_pathway_fraction descending
    (higher concentration → more switching).
    """
    results = []
    for conc in conc_list:
        result = predict_metabolic_switching(
            drug_name=drug_name,
            drug_conc_uM=conc,
            cyp3a4_inhibition_fraction=cyp3a4_inhibition_fraction,
            cyp2d6_inhibition_fraction=cyp2d6_inhibition_fraction,
            ugt_inhibition_fraction=ugt_inhibition_fraction,
            sult_inhibition_fraction=sult_inhibition_fraction,
            nat2_inhibition_fraction=nat2_inhibition_fraction,
        )
        results.append(result)

    results.sort(key=lambda r: r.alt_pathway_fraction, reverse=True)
    return results
