"""P-glycoprotein (P-gp) efflux model for BBB and intestinal epithelium.

Models P-gp efflux ratio prediction, CNS penetration impact,
intestinal absorption modulation, and inhibitor effects.
"""

from __future__ import annotations


def predict_pgp_efflux_ratio(
    mw: float,
    logP: float,
    psa_ang2: float,
    n_hbd: int,
    n_hba: int,
) -> float:
    """Predict P-gp efflux ratio (Papp_BA / Papp_AB) from physicochemical properties.

    Parameters
    ----------
    mw: molecular weight (g/mol)
    logP: lipophilicity
    psa_ang2: polar surface area (Angstrom^2)
    n_hbd: number of hydrogen bond donors
    n_hba: number of hydrogen bond acceptors

    Returns
    -------
    efflux_ratio (float, clamped to [1.0, 20.0])
    """
    if mw <= 0:
        raise ValueError("mw must be positive")
    if psa_ang2 < 0:
        raise ValueError("psa_ang2 must be non-negative")
    if n_hbd < 0:
        raise ValueError("n_hbd must be non-negative")
    if n_hba < 0:
        raise ValueError("n_hba must be non-negative")

    er = 1.0

    if mw > 400:
        er += (mw - 400) / 200

    if psa_ang2 > 60:
        er += (psa_ang2 - 60) / 40

    if n_hbd > 3:
        er += (n_hbd - 3) * 0.3

    if n_hba > 5:
        er += (n_hba - 5) * 0.1

    # Clamp to [1.0, 20.0]
    er = max(1.0, min(20.0, er))

    return er


def cns_penetration_with_pgp(
    kp_passive: float,
    efflux_ratio: float,
    pgp_expression: float = 1.0,
) -> float:
    """Estimate CNS penetration (Kp,uu,brain) accounting for P-gp efflux.

    Parameters
    ----------
    kp_passive: passive brain-to-plasma Kp without P-gp (dimensionless)
    efflux_ratio: P-gp efflux ratio (Papp_BA/Papp_AB)
    pgp_expression: relative P-gp expression level (1.0 = normal)

    Returns
    -------
    kp_uu_brain: unbound brain-to-plasma concentration ratio
    """
    if kp_passive <= 0:
        raise ValueError("kp_passive must be positive")
    if efflux_ratio < 1.0:
        raise ValueError("efflux_ratio must be >= 1.0")
    if pgp_expression < 0:
        raise ValueError("pgp_expression must be non-negative")

    denominator = 1.0 + (efflux_ratio - 1.0) * pgp_expression
    if denominator <= 0:
        denominator = 1.0

    kp_uu = kp_passive / denominator

    # Clamp to [0.01, kp_passive]
    kp_uu = max(0.01, min(kp_passive, kp_uu))

    return kp_uu


def intestinal_absorption_with_pgp(
    fa_passive: float,
    efflux_ratio: float,
    dose_mg: float,
    km_pgp: float = 10.0,
    vmax_pgp: float = 100.0,
) -> dict:
    """Estimate intestinal absorption fraction accounting for saturable P-gp efflux.

    Parameters
    ----------
    fa_passive: passive fraction absorbed (0-1)
    efflux_ratio: P-gp efflux ratio (Papp_BA/Papp_AB)
    dose_mg: dose in milligrams
    km_pgp: P-gp Michaelis constant (mg/L)
    vmax_pgp: P-gp maximum transport rate (mg/L/h, not used in simple model)

    Returns
    -------
    dict with fa_passive, fa_adjusted, effective_er, saturation_factor, pgp_impact
    """
    if not (0.0 <= fa_passive <= 1.0):
        raise ValueError("fa_passive must be between 0 and 1")
    if efflux_ratio < 1.0:
        raise ValueError("efflux_ratio must be >= 1.0")
    if dose_mg < 0:
        raise ValueError("dose_mg must be non-negative")
    if km_pgp <= 0:
        raise ValueError("km_pgp must be positive")

    # Luminal concentration approximation: dissolve dose in 250 mL
    c = dose_mg / 0.25  # mg/L

    saturation_factor = 1.0 - km_pgp / (km_pgp + c)

    # Effective ER decreases as P-gp saturates
    effective_er = efflux_ratio * (1.0 - saturation_factor * 0.8)
    # Ensure effective_er >= 1.0
    effective_er = max(1.0, effective_er)

    # Absorption reduction from P-gp
    fa_adjusted = fa_passive / (1.0 + (effective_er - 1.0) * 0.3)

    # Clamp to [0.01, fa_passive]
    fa_adjusted = max(0.01, min(fa_passive, fa_adjusted))

    reduction_fraction = (fa_passive - fa_adjusted) / fa_passive if fa_passive > 0 else 0.0

    if reduction_fraction > 0.30:
        pgp_impact = "high"
    elif reduction_fraction > 0.10:
        pgp_impact = "moderate"
    else:
        pgp_impact = "low"

    return {
        "fa_passive": fa_passive,
        "fa_adjusted": fa_adjusted,
        "effective_er": effective_er,
        "saturation_factor": saturation_factor,
        "pgp_impact": pgp_impact,
    }


def inhibition_effect_on_pgp(
    kp_uu_baseline: float,
    efflux_ratio: float,
    inhibitor_ki_uM: float,
    inhibitor_conc_uM: float,
) -> dict:
    """Estimate the effect of a P-gp inhibitor on CNS penetration.

    Parameters
    ----------
    kp_uu_baseline: baseline Kp,uu,brain without inhibitor
    efflux_ratio: P-gp efflux ratio (Papp_BA/Papp_AB)
    inhibitor_ki_uM: P-gp inhibitor Ki (micromolar)
    inhibitor_conc_uM: inhibitor concentration at site (micromolar)

    Returns
    -------
    dict with kp_uu_baseline, kp_uu_inhibited, fold_increase_penetration,
    effective_er, fractional_inhibition
    """
    if kp_uu_baseline <= 0:
        raise ValueError("kp_uu_baseline must be positive")
    if efflux_ratio < 1.0:
        raise ValueError("efflux_ratio must be >= 1.0")
    if inhibitor_ki_uM <= 0:
        raise ValueError("inhibitor_ki_uM must be positive")
    if inhibitor_conc_uM < 0:
        raise ValueError("inhibitor_conc_uM must be non-negative")

    # Competitive inhibition of P-gp
    effective_er = efflux_ratio / (1.0 + inhibitor_conc_uM / inhibitor_ki_uM)
    # Effective ER cannot go below 1.0
    effective_er = max(1.0, effective_er)

    # kp_uu increases proportionally to ER reduction
    kp_uu_inhibited = kp_uu_baseline * (efflux_ratio / effective_er)

    fold_increase_penetration = kp_uu_inhibited / kp_uu_baseline

    fractional_inhibition = 1.0 - effective_er / efflux_ratio

    return {
        "kp_uu_baseline": kp_uu_baseline,
        "kp_uu_inhibited": kp_uu_inhibited,
        "fold_increase_penetration": fold_increase_penetration,
        "effective_er": effective_er,
        "fractional_inhibition": fractional_inhibition,
    }
