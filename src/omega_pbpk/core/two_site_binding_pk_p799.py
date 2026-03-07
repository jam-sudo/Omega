"""Two-site receptor binding pharmacodynamics — Phase 799.

Models competitive and allosteric two-site receptor binding with combined
effect calculations using Bliss independence and Hill equation occupancy.

Pure Python only: stdlib math, dataclasses.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "TwoSiteBindingResult",
    "calculate_two_site_binding",
    "scan_two_site_binding",
]

_VALID_INTERACTION_TYPES = {
    "competitive",
    "allosteric_positive",
    "allosteric_negative",
    "independent",
}


@dataclass(frozen=True)
class TwoSiteBindingResult:
    """Result of two-site receptor binding PD calculation.

    Fields
    ------
    drug_name : str
        Name of the drug.
    conc_mg_L : float
        Drug concentration in mg/L.
    site1_occupancy_pct : float
        Occupancy at site 1 (%).
    site2_occupancy_pct : float
        Occupancy at site 2 (%).
    combined_effect_pct : float
        Combined pharmacodynamic effect (%).
    interaction_type : str
        Type of interaction: competitive, allosteric_positive,
        allosteric_negative, or independent.
    emax_combined : float
        Maximum combined effect achievable (%).
    cooperativity_index : float
        Cooperativity index: >1 positive, <1 negative.
    notes : str
        Summary notes.
    """

    drug_name: str
    conc_mg_L: float
    site1_occupancy_pct: float
    site2_occupancy_pct: float
    combined_effect_pct: float
    interaction_type: str
    emax_combined: float
    cooperativity_index: float
    notes: str


def _hill_occupancy(conc: float, ec50: float, hill: float) -> float:
    """Hill equation occupancy fraction (0 to 1).

    Parameters
    ----------
    conc : Drug concentration (mg/L), must be >= 0.
    ec50 : EC50 (mg/L), must be > 0.
    hill : Hill coefficient.

    Returns
    -------
    float
        Occupancy fraction in [0, 1].
    """
    if conc <= 0.0:
        return 0.0
    c_h = conc**hill
    ec50_h = ec50**hill
    return c_h / (c_h + ec50_h)


def calculate_two_site_binding(
    drug_name: str,
    conc_mg_L: float,
    ec50_site1_mg_L: float,
    ec50_site2_mg_L: float,
    emax1_pct: float = 100.0,
    emax2_pct: float = 50.0,
    hill1: float = 1.0,
    hill2: float = 1.0,
    interaction_type: str = "independent",
    alpha: float = 1.0,
) -> TwoSiteBindingResult:
    """Calculate two-site receptor binding pharmacodynamics.

    Parameters
    ----------
    drug_name : str
        Drug identifier.
    conc_mg_L : float
        Drug concentration in mg/L (must be >= 0).
    ec50_site1_mg_L : float
        EC50 at site 1 in mg/L (must be > 0).
    ec50_site2_mg_L : float
        EC50 at site 2 in mg/L (must be > 0).
    emax1_pct : float
        Maximum effect at site 1 (%).
    emax2_pct : float
        Maximum effect at site 2 (%).
    hill1 : float
        Hill coefficient for site 1.
    hill2 : float
        Hill coefficient for site 2.
    interaction_type : str
        One of: "independent", "competitive", "allosteric_positive",
        "allosteric_negative".
    alpha : float
        Allosteric factor (>1 = positive cooperativity, <1 = negative).
        Used for allosteric_positive and allosteric_negative types.

    Returns
    -------
    TwoSiteBindingResult

    Raises
    ------
    ValueError
        If conc_mg_L < 0, ec50_site1_mg_L <= 0, or ec50_site2_mg_L <= 0.
    """
    if conc_mg_L < 0.0:
        raise ValueError(f"conc_mg_L must be >= 0, got {conc_mg_L}")
    if ec50_site1_mg_L <= 0.0:
        raise ValueError(f"ec50_site1_mg_L must be > 0, got {ec50_site1_mg_L}")
    if ec50_site2_mg_L <= 0.0:
        raise ValueError(f"ec50_site2_mg_L must be > 0, got {ec50_site2_mg_L}")
    if interaction_type not in _VALID_INTERACTION_TYPES:
        raise ValueError(
            f"interaction_type must be one of {_VALID_INTERACTION_TYPES}, got '{interaction_type}'"
        )

    # Site 1 occupancy — always standard Hill
    occ1 = _hill_occupancy(conc_mg_L, ec50_site1_mg_L, hill1)
    site1_occupancy_pct = occ1 * 100.0

    # Site 2 occupancy depends on interaction type
    if interaction_type == "independent":
        occ2 = _hill_occupancy(conc_mg_L, ec50_site2_mg_L, hill2)
        # Bliss independence: E = E1 + E2 - E1*E2/100
        e1 = emax1_pct * occ1
        e2 = emax2_pct * occ2
        combined_effect = e1 + e2 - e1 * e2 / 100.0

    elif interaction_type == "competitive":
        # Drug competes at both sites; site2 occupancy reduced by site1 occupancy
        # Effective concentration available for site 2 is scaled by (1 - occ1)
        ec50_site2_eff = ec50_site2_mg_L / max(1.0 - occ1, 1e-9)
        occ2 = _hill_occupancy(conc_mg_L, ec50_site2_eff, hill2)
        e1 = emax1_pct * occ1
        e2 = emax2_pct * occ2
        # Combined is additive (competition reduces site2)
        combined_effect = e1 + e2 - e1 * e2 / 100.0

    elif interaction_type == "allosteric_positive":
        # alpha > 1 → EC50_site2_eff = EC50_site2 / alpha (increases affinity)
        if alpha <= 0.0:
            raise ValueError(f"alpha must be > 0 for allosteric interaction, got {alpha}")
        ec50_site2_eff = ec50_site2_mg_L / alpha
        occ2 = _hill_occupancy(conc_mg_L, ec50_site2_eff, hill2)
        e1 = emax1_pct * occ1
        e2 = emax2_pct * occ2
        combined_effect = e1 + e2 - e1 * e2 / 100.0

    else:  # allosteric_negative
        # alpha > 1 → EC50_site2_eff = EC50_site2 * alpha (decreases affinity)
        if alpha <= 0.0:
            raise ValueError(f"alpha must be > 0 for allosteric interaction, got {alpha}")
        ec50_site2_eff = ec50_site2_mg_L * alpha
        occ2 = _hill_occupancy(conc_mg_L, ec50_site2_eff, hill2)
        e1 = emax1_pct * occ1
        e2 = emax2_pct * occ2
        combined_effect = e1 + e2 - e1 * e2 / 100.0

    site2_occupancy_pct = occ2 * 100.0

    # Emax combined: theoretical maximum at saturation (conc → infinity)
    emax_combined = emax1_pct + emax2_pct - emax1_pct * emax2_pct / 100.0

    # Cooperativity index: ratio of effective affinity improvement
    # For independent/competitive: alpha is 1.0 by definition
    # For allosteric: alpha reflects cooperativity
    if interaction_type == "allosteric_positive":
        cooperativity_index = alpha
    elif interaction_type == "allosteric_negative":
        cooperativity_index = 1.0 / alpha if alpha != 0.0 else 0.0
    elif interaction_type == "competitive":
        # Cooperative index < 1 because site1 reduces site2 availability
        cooperativity_index = max(1.0 - occ1, 0.0)
    else:
        cooperativity_index = 1.0

    notes = (
        f"{drug_name}: [{conc_mg_L:.3f} mg/L] "
        f"site1_occ={site1_occupancy_pct:.1f}%, site2_occ={site2_occupancy_pct:.1f}%, "
        f"combined_effect={combined_effect:.1f}%, "
        f"type={interaction_type}, cooperativity={cooperativity_index:.3f}"
    )

    return TwoSiteBindingResult(
        drug_name=drug_name,
        conc_mg_L=conc_mg_L,
        site1_occupancy_pct=site1_occupancy_pct,
        site2_occupancy_pct=site2_occupancy_pct,
        combined_effect_pct=combined_effect,
        interaction_type=interaction_type,
        emax_combined=emax_combined,
        cooperativity_index=cooperativity_index,
        notes=notes,
    )


def scan_two_site_binding(
    drug_name: str,
    conc_range_mg_L: list,
    ec50_site1_mg_L: float,
    ec50_site2_mg_L: float,
    emax1_pct: float = 100.0,
    emax2_pct: float = 50.0,
    hill1: float = 1.0,
    hill2: float = 1.0,
    interaction_type: str = "independent",
    alpha: float = 1.0,
) -> list:
    """Scan two-site binding across a range of concentrations.

    Parameters
    ----------
    drug_name : str
        Drug identifier.
    conc_range_mg_L : list[float]
        List of drug concentrations in mg/L to evaluate.
    ec50_site1_mg_L : float
        EC50 at site 1 in mg/L.
    ec50_site2_mg_L : float
        EC50 at site 2 in mg/L.
    emax1_pct : float
        Maximum effect at site 1 (%).
    emax2_pct : float
        Maximum effect at site 2 (%).
    hill1 : float
        Hill coefficient for site 1.
    hill2 : float
        Hill coefficient for site 2.
    interaction_type : str
        Interaction type (see calculate_two_site_binding).
    alpha : float
        Allosteric factor.

    Returns
    -------
    list[TwoSiteBindingResult]
        Results sorted by concentration ascending.
    """
    sorted_concs = sorted(conc_range_mg_L)
    results = []
    for conc in sorted_concs:
        result = calculate_two_site_binding(
            drug_name=drug_name,
            conc_mg_L=conc,
            ec50_site1_mg_L=ec50_site1_mg_L,
            ec50_site2_mg_L=ec50_site2_mg_L,
            emax1_pct=emax1_pct,
            emax2_pct=emax2_pct,
            hill1=hill1,
            hill2=hill2,
            interaction_type=interaction_type,
            alpha=alpha,
        )
        results.append(result)
    return results
