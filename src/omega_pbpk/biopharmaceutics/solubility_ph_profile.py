"""
Drug solubility pH profile computation using Henderson-Hasselbalch equations.

Covers weak acids, weak bases, neutral compounds, and amphoteric drugs
across the physiological pH range (1–8).
"""

from __future__ import annotations

_VALID_DRUG_TYPES = {"weak_acid", "weak_base", "neutral", "amphoteric"}


def solubility_at_ph(
    s0_mg_mL: float,
    pka: float,
    drug_type: str,
    ph: float,
    pka2: float | None = None,
) -> float:
    """Compute solubility at a given pH using Henderson-Hasselbalch.

    Parameters
    ----------
    s0_mg_mL : float
        Intrinsic solubility (solubility of the neutral form) in mg/mL.
    pka : float
        pKa of the drug (for amphoteric: pKa of the acidic group).
    drug_type : str
        One of 'weak_acid', 'weak_base', 'neutral', 'amphoteric'.
    ph : float
        Target pH.
    pka2 : float, optional
        pKa of the basic group (required for amphoteric).

    Returns
    -------
    float
        Solubility at the given pH in mg/mL.
    """
    if drug_type not in _VALID_DRUG_TYPES:
        raise ValueError(f"drug_type must be one of {sorted(_VALID_DRUG_TYPES)}, got {drug_type!r}")
    if s0_mg_mL < 0:
        raise ValueError("s0_mg_mL must be non-negative")

    if drug_type == "neutral":
        return s0_mg_mL
    elif drug_type == "weak_acid":
        return s0_mg_mL * (1.0 + 10.0 ** (ph - pka))
    elif drug_type == "weak_base":
        return s0_mg_mL * (1.0 + 10.0 ** (pka - ph))
    else:  # amphoteric
        if pka2 is None:
            raise ValueError("pka2 (basic pKa) is required for amphoteric drugs")
        # pka = pKa_acid, pka2 = pKa_base
        return s0_mg_mL * (1.0 + 10.0 ** (ph - pka) + 10.0 ** (pka2 - ph))


def ph_solubility_profile(
    s0_mg_mL: float,
    pka: float,
    drug_type: str,
    pka2: float | None = None,
    ph_range: tuple[float, float] = (1.0, 8.0),
    n_points: int = 71,
) -> dict:
    """Generate a pH vs solubility curve.

    Parameters
    ----------
    s0_mg_mL : float
        Intrinsic solubility in mg/mL.
    pka : float
        pKa (acid group for amphoteric).
    drug_type : str
        One of 'weak_acid', 'weak_base', 'neutral', 'amphoteric'.
    pka2 : float, optional
        Basic pKa for amphoteric drugs.
    ph_range : tuple
        (pH_min, pH_max) for the sweep.
    n_points : int
        Number of pH points.

    Returns
    -------
    dict with keys:
        ph_values, solubility_mg_mL, ph_min_solubility, s_min_mg_mL, s_max_mg_mL
    """
    if n_points < 2:
        raise ValueError("n_points must be >= 2")
    ph_lo, ph_hi = ph_range
    if ph_lo >= ph_hi:
        raise ValueError("ph_range[0] must be less than ph_range[1]")

    step = (ph_hi - ph_lo) / (n_points - 1)
    ph_values = [ph_lo + i * step for i in range(n_points)]
    sol_values = [solubility_at_ph(s0_mg_mL, pka, drug_type, ph, pka2=pka2) for ph in ph_values]

    s_min = min(sol_values)
    s_max = max(sol_values)
    idx_min = sol_values.index(s_min)
    ph_min_sol = ph_values[idx_min]

    return {
        "ph_values": ph_values,
        "solubility_mg_mL": sol_values,
        "ph_min_solubility": ph_min_sol,
        "s_min_mg_mL": s_min,
        "s_max_mg_mL": s_max,
    }


def dose_number(
    dose_mg: float,
    volume_mL: float = 250.0,
    s0_mg_mL: float | None = None,
    ph: float = 6.5,
    pka: float | None = None,
    drug_type: str = "weak_acid",
) -> dict:
    """Compute BCS dose number at a given pH.

    Dose number Dn = dose / (volume * S), where S is solubility at the given pH.

    Parameters
    ----------
    dose_mg : float
        Dose in mg.
    volume_mL : float
        GI fluid volume (default 250 mL per BCS guidance).
    s0_mg_mL : float, optional
        Intrinsic solubility. Required if pka is None.
    ph : float
        pH at which to evaluate solubility.
    pka : float, optional
        pKa; if provided, computes pH-adjusted solubility from s0_mg_mL.
    drug_type : str
        Drug ionization type.

    Returns
    -------
    dict with keys: dose_number, solubility_at_ph_mg_mL, bcs_solubility_class, notes
    """
    if dose_mg <= 0:
        raise ValueError("dose_mg must be positive")
    if volume_mL <= 0:
        raise ValueError("volume_mL must be positive")

    if pka is not None and s0_mg_mL is not None:
        sol = solubility_at_ph(s0_mg_mL, pka, drug_type, ph)
    elif s0_mg_mL is not None:
        sol = s0_mg_mL
    else:
        raise ValueError("Either s0_mg_mL or (s0_mg_mL + pka) must be provided")

    if sol <= 0:
        dn = float("inf")
    else:
        dn = dose_mg / (volume_mL * sol)

    if dn <= 1.0:
        bcs_class = "high_solubility"
        notes = "BCS I or III candidate (high solubility)"
    else:
        bcs_class = "low_solubility"
        notes = "BCS II or IV candidate (low solubility)"

    return {
        "dose_number": dn,
        "solubility_at_ph_mg_mL": sol,
        "bcs_solubility_class": bcs_class,
        "notes": notes,
    }


def minimum_volume_for_dissolution(
    dose_mg: float,
    pka: float,
    drug_type: str,
    ph: float = 6.5,
    s0_mg_mL: float = 1.0,
) -> float:
    """Compute minimum GI fluid volume required to dissolve the full dose.

    V_min = dose / S(pH)  [mL]

    Parameters
    ----------
    dose_mg : float
        Dose in mg.
    pka : float
        pKa of the drug.
    drug_type : str
        Drug ionization type.
    ph : float
        pH at dissolution site.
    s0_mg_mL : float
        Intrinsic solubility in mg/mL.

    Returns
    -------
    float
        Minimum dissolution volume in mL.
    """
    if dose_mg <= 0:
        raise ValueError("dose_mg must be positive")

    sol = solubility_at_ph(s0_mg_mL, pka, drug_type, ph)
    if sol <= 0:
        return float("inf")
    return dose_mg / sol


def solubility_bcs_classification(
    s_mg_mL_ph65: float,
    peff_cm_s: float,
    dose_mg: float,
    volume_mL: float = 250.0,
) -> dict:
    """Classify a drug according to the Biopharmaceutics Classification System.

    Parameters
    ----------
    s_mg_mL_ph65 : float
        Solubility at pH 6.5 in mg/mL.
    peff_cm_s : float
        Effective intestinal permeability in cm/s.
    dose_mg : float
        Highest dose strength in mg.
    volume_mL : float
        Reference GI fluid volume (default 250 mL).

    Returns
    -------
    dict with keys: bcs_class, is_high_solubility, is_high_permeability,
                    dose_number, notes
    """
    if s_mg_mL_ph65 < 0:
        raise ValueError("s_mg_mL_ph65 must be non-negative")
    if peff_cm_s < 0:
        raise ValueError("peff_cm_s must be non-negative")
    if dose_mg <= 0:
        raise ValueError("dose_mg must be positive")
    if volume_mL <= 0:
        raise ValueError("volume_mL must be positive")

    # Dose number
    if s_mg_mL_ph65 <= 0:
        dn = float("inf")
    else:
        dn = dose_mg / (volume_mL * s_mg_mL_ph65)

    is_high_sol = dn <= 1.0
    is_high_perm = peff_cm_s >= 2e-4

    if is_high_sol and is_high_perm:
        bcs_class = "I"
        notes = "High solubility, high permeability — biowaiver eligible"
    elif not is_high_sol and is_high_perm:
        bcs_class = "II"
        notes = "Low solubility, high permeability — dissolution rate-limited"
    elif is_high_sol and not is_high_perm:
        bcs_class = "III"
        notes = "High solubility, low permeability — permeability rate-limited"
    else:
        bcs_class = "IV"
        notes = "Low solubility, low permeability — poor oral absorption"

    return {
        "bcs_class": bcs_class,
        "is_high_solubility": is_high_sol,
        "is_high_permeability": is_high_perm,
        "dose_number": dn,
        "notes": notes,
    }
