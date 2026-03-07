"""Drug salivary excretion PK model.

Saliva-to-plasma (S/P) ratio depends on protein binding and ionization.
Provides non-invasive monitoring alternative to plasma for TDM.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SalivaryExcretionResult:
    """Result of salivary excretion PK simulation."""

    drug_name: str
    dose_mg: float
    times_h: list
    c_plasma_mg_L: list
    c_saliva_mg_L: list
    cmax_plasma: float
    cmax_saliva: float
    auc_plasma: float
    auc_saliva: float
    sp_ratio: float
    reliability_for_tdm: str
    f_neutral_at_saliva_ph: float
    notes: str


def _compute_f_neutral(compound_type: str, pka: float, ph_saliva: float) -> float:
    """Compute fraction of drug in neutral (unionized) form at saliva pH."""
    if compound_type == "acid":
        # Henderson-Hasselbalch for acid: f_neutral = 1 / (1 + 10^(pH - pKa))
        return 1.0 / (1.0 + 10.0 ** (ph_saliva - pka))
    elif compound_type == "base":
        # Henderson-Hasselbalch for base: f_neutral = 1 / (1 + 10^(pKa - pH))
        return 1.0 / (1.0 + 10.0 ** (pka - ph_saliva))
    else:
        # Neutral compound: fully unionized
        return 1.0


def _compute_sp_ratio(fup: float, f_neutral: float) -> float:
    """Compute salivary-to-plasma ratio."""
    return fup * f_neutral


def _reliability_label(sp_ratio: float) -> str:
    """Classify TDM reliability based on S/P ratio."""
    if sp_ratio > 0.8:
        return "excellent"
    elif sp_ratio >= 0.5:
        return "good"
    elif sp_ratio >= 0.1:
        return "acceptable"
    else:
        return "unreliable"


def _trapezoidal_auc(times: list, concentrations: list) -> float:
    """Compute AUC using manual trapezoidal integration."""
    auc = 0.0
    for i in range(1, len(times)):
        dt = times[i] - times[i - 1]
        auc += 0.5 * (concentrations[i] + concentrations[i - 1]) * dt
    return auc


def simulate_salivary_pk(
    drug_name: str,
    dose_mg: float,
    cl_L_per_h: float = 5.0,
    vd_L: float = 50.0,
    fup: float = 0.5,
    pka: float = -1.0,
    compound_type: str = "neutral",
    ph_saliva: float = 6.8,
    t_end_h: float = 12.0,
    dt_h: float = 0.1,
) -> SalivaryExcretionResult:
    """Simulate 1-compartment PK and compute salivary concentration profile.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    dose_mg:
        IV bolus dose in mg.
    cl_L_per_h:
        Total plasma clearance in L/h.
    vd_L:
        Volume of distribution in L.
    fup:
        Fraction unbound in plasma (0 < fup <= 1).
    pka:
        Drug pKa value. Use -1.0 (default) if not applicable (neutral compound).
    compound_type:
        One of "acid", "base", or "neutral".
    ph_saliva:
        Salivary pH (default 6.8).
    t_end_h:
        Simulation end time in hours.
    dt_h:
        Time step in hours.

    Returns
    -------
    SalivaryExcretionResult
    """
    # Validation
    if dose_mg <= 0:
        raise ValueError(f"dose_mg must be > 0, got {dose_mg}")
    if cl_L_per_h <= 0:
        raise ValueError(f"cl_L_per_h must be > 0, got {cl_L_per_h}")
    if vd_L <= 0:
        raise ValueError(f"vd_L must be > 0, got {vd_L}")
    if not (0.0 < fup <= 1.0):
        raise ValueError(f"fup must be in (0, 1], got {fup}")
    if not (0.0 < ph_saliva < 14.0):
        raise ValueError(f"ph_saliva must be in (0, 14), got {ph_saliva}")

    compound_type = compound_type.lower()
    if compound_type not in ("acid", "base", "neutral"):
        raise ValueError(f"compound_type must be 'acid', 'base', or 'neutral', got {compound_type}")

    # If pKa not provided (sentinel -1.0), treat as neutral regardless
    if pka == -1.0 or pka is None:
        effective_type = "neutral"
        effective_pka = 7.0  # placeholder, won't be used
    else:
        effective_type = compound_type
        effective_pka = pka

    # Compute ionization-based fraction and S/P ratio
    f_neutral = _compute_f_neutral(effective_type, effective_pka, ph_saliva)
    sp_ratio = _compute_sp_ratio(fup, f_neutral)

    # 1-compartment IV bolus PK: forward Euler
    ke = cl_L_per_h / vd_L  # elimination rate constant (h^-1)
    c0 = dose_mg / vd_L  # initial plasma concentration (mg/L)

    times_h: list = []
    c_plasma: list = []
    c_saliva: list = []

    t = 0.0
    c = c0
    n_steps = int(round(t_end_h / dt_h)) + 1

    for _step in range(n_steps):
        times_h.append(t)
        c_plasma.append(max(c, 0.0))
        c_saliva.append(max(c * sp_ratio, 0.0))

        # Forward Euler step
        dc_dt = -ke * c
        c = c + dc_dt * dt_h
        t += dt_h

    cmax_plasma = max(c_plasma)
    cmax_saliva = max(c_saliva)
    auc_plasma = _trapezoidal_auc(times_h, c_plasma)
    auc_saliva = _trapezoidal_auc(times_h, c_saliva)

    reliability = _reliability_label(sp_ratio)

    # Build notes
    notes_parts = [
        f"S/P ratio = {sp_ratio:.4f}",
        f"Compound type: {compound_type}",
        f"fup = {fup:.3f}",
        f"f_neutral at pH {ph_saliva} = {f_neutral:.4f}",
        f"TDM reliability: {reliability}",
    ]
    if reliability == "unreliable":
        notes_parts.append("Plasma sampling is preferred over saliva for TDM.")
    elif reliability == "excellent":
        notes_parts.append("Saliva is an excellent surrogate for plasma TDM.")
    notes = "; ".join(notes_parts)

    return SalivaryExcretionResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        times_h=times_h,
        c_plasma_mg_L=c_plasma,
        c_saliva_mg_L=c_saliva,
        cmax_plasma=cmax_plasma,
        cmax_saliva=cmax_saliva,
        auc_plasma=auc_plasma,
        auc_saliva=auc_saliva,
        sp_ratio=sp_ratio,
        reliability_for_tdm=reliability,
        f_neutral_at_saliva_ph=f_neutral,
        notes=notes,
    )


def assess_salivary_tdm(result: SalivaryExcretionResult) -> dict:
    """Assess whether salivary TDM is feasible for this drug.

    Parameters
    ----------
    result:
        SalivaryExcretionResult from simulate_salivary_pk.

    Returns
    -------
    dict with keys:
        usable (bool): True if S/P ratio >= 0.1
        sp_ratio (float): Salivary-to-plasma ratio
        recommendation (str): Sampling guidance or note
    """
    usable = result.sp_ratio >= 0.1

    if usable:
        recommendation = (
            f"Salivary sampling is feasible (S/P = {result.sp_ratio:.3f}). "
            f"Collect saliva at steady-state trough or peak, depending on therapeutic window. "
            f"TDM reliability: {result.reliability_for_tdm}."
        )
    else:
        recommendation = (
            f"Plasma sampling preferred. S/P ratio = {result.sp_ratio:.4f} is below 0.1 "
            f"threshold; salivary concentrations are too low for reliable TDM."
        )

    return {
        "usable": usable,
        "sp_ratio": result.sp_ratio,
        "recommendation": recommendation,
    }
