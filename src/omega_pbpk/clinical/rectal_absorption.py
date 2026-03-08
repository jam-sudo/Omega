"""Phase 912 — Drug Rectal/Colonic Absorption.

Models drug absorption via rectal/suppository route — relevant for
anti-nausea drugs, analgesics, and fever reducers in pediatrics.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RectalAbsorptionResult:
    """Result of rectal drug absorption simulation."""

    drug_name: str
    dose_mg: float
    formulation: str
    rectal_ph: float
    times_h: tuple
    c_plasma_mg_L: tuple
    cmax_mg_L: float
    tmax_h: float
    auc_mg_h_per_L: float
    f_rectal: float
    first_pass_bypass_fraction: float
    compared_to_oral_ratio: float
    notes: str


# Formulation parameters
_FORMULATION_K_REL = {
    "suppository_wax": 0.5,
    "suppository_PEG": 1.0,
    "enema": 2.0,
    "gel": 0.8,
}

_FORMULATION_F_ABS = {
    "suppository_wax": 0.5,
    "suppository_PEG": 0.6,
    "enema": 0.8,
    "gel": 0.7,
}

_VALID_FORMULATIONS = frozenset(_FORMULATION_K_REL.keys())


def simulate_rectal_absorption(
    drug_name: str,
    dose_mg: float,
    formulation: str = "suppository_wax",
    rectal_ph: float = 7.4,
    pka: float | None = None,
    ionization_type: str = "neutral",
    cl_L_per_h: float = 10.0,
    vd_L: float = 50.0,
    t_end_h: float = 12.0,
    dt_h: float = 0.1,
) -> RectalAbsorptionResult:
    """Simulate rectal drug absorption using a two-compartment depot model.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    dose_mg:
        Dose administered rectally in mg.
    formulation:
        Rectal formulation type: "suppository_wax", "suppository_PEG",
        "enema", or "gel".
    rectal_ph:
        Rectal fluid pH (default 7.4).
    pka:
        Drug pKa (currently unused in ODE; reserved for future pH correction).
    ionization_type:
        Ionization type (reserved for future use).
    cl_L_per_h:
        Systemic clearance in L/h.
    vd_L:
        Volume of distribution in L.
    t_end_h:
        Simulation end time in hours.
    dt_h:
        Forward Euler time step in hours.

    Returns
    -------
    RectalAbsorptionResult
    """
    if dose_mg <= 0:
        raise ValueError(f"dose_mg must be > 0, got {dose_mg}")
    if formulation not in _VALID_FORMULATIONS:
        raise ValueError(
            f"formulation must be one of {set(_VALID_FORMULATIONS)}, got {formulation!r}"
        )

    k_rel = _FORMULATION_K_REL[formulation]
    f_absorbed_rectal = _FORMULATION_F_ABS[formulation]
    first_pass_bypass_fraction = 0.5

    # Effective fraction reaching systemic circulation
    # Portal portion (1 - bypass) loses 30% to hepatic first-pass
    f_effective = f_absorbed_rectal * (1.0 - (1.0 - first_pass_bypass_fraction) * 0.3)

    ke = cl_L_per_h / vd_L

    # Forward Euler ODE integration
    a_rect = dose_mg  # drug in rectal depot (mg)
    a_plasma = 0.0  # drug amount in plasma (mg)

    n_steps = int(t_end_h / dt_h) + 1
    times = []
    c_plasma_list = []

    for i in range(n_steps):
        t = i * dt_h
        c_plasma = a_plasma / vd_L
        times.append(t)
        c_plasma_list.append(c_plasma)

        # Derivatives
        da_rect = -k_rel * a_rect
        da_plasma = k_rel * a_rect * f_effective - ke * a_plasma

        # Update state
        a_rect = a_rect + da_rect * dt_h
        a_plasma = a_plasma + da_plasma * dt_h

        # Clamp to non-negative
        if a_rect < 0.0:
            a_rect = 0.0
        if a_plasma < 0.0:
            a_plasma = 0.0

    # Compute PK metrics
    cmax_mg_L = max(c_plasma_list)
    tmax_h = times[c_plasma_list.index(cmax_mg_L)]

    # Trapezoidal AUC
    auc = 0.0
    for i in range(1, len(times)):
        auc += 0.5 * (c_plasma_list[i - 1] + c_plasma_list[i]) * (times[i] - times[i - 1])

    # Rectal bioavailability
    f_rectal = auc * cl_L_per_h / dose_mg
    f_rectal = max(0.0, min(1.0, f_rectal))

    # Compared to oral (assume oral F = 0.6 as reference)
    compared_to_oral_ratio = f_rectal / 0.6
    compared_to_oral_ratio = max(0.1, min(5.0, compared_to_oral_ratio))

    # Notes
    note_parts = []
    if compared_to_oral_ratio > 0.8:
        note_parts.append("Rectal bioavailability comparable to oral — viable alternative route")
    if first_pass_bypass_fraction > 0:
        note_parts.append("Partial hepatic first-pass bypass — useful for high first-pass drugs")
    if not note_parts:
        note_parts.append("Standard rectal absorption")
    notes = "; ".join(note_parts)

    return RectalAbsorptionResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        formulation=formulation,
        rectal_ph=rectal_ph,
        times_h=tuple(times),
        c_plasma_mg_L=tuple(c_plasma_list),
        cmax_mg_L=cmax_mg_L,
        tmax_h=tmax_h,
        auc_mg_h_per_L=auc,
        f_rectal=f_rectal,
        first_pass_bypass_fraction=first_pass_bypass_fraction,
        compared_to_oral_ratio=compared_to_oral_ratio,
        notes=notes,
    )


def compare_rectal_formulations(
    drug_name: str,
    dose_mg: float,
    **kwargs,
) -> list[RectalAbsorptionResult]:
    """Compare all four rectal formulations for a given drug.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    dose_mg:
        Dose in mg.
    **kwargs:
        Additional keyword arguments forwarded to simulate_rectal_absorption
        (e.g., cl_L_per_h, vd_L, pka, ionization_type).

    Returns
    -------
    List of 4 RectalAbsorptionResult objects sorted by auc_mg_h_per_L descending.
    """
    results = []
    for formulation in _VALID_FORMULATIONS:
        result = simulate_rectal_absorption(
            drug_name=drug_name,
            dose_mg=dose_mg,
            formulation=formulation,
            **kwargs,
        )
        results.append(result)
    results.sort(key=lambda r: r.auc_mg_h_per_L, reverse=True)
    return results
