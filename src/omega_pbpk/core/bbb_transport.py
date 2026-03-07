"""Phase 576 — Blood-brain barrier transport model with active efflux."""

from __future__ import annotations


def passive_bbb_permeability(logP: float, mw: float, psa: float) -> dict:
    """Estimate passive BBB permeability from physicochemical properties.

    Empirical equation:
        log(Papp_bbb) = 0.5*logP - 0.01*psa - 0.003*mw - 4.0
        Papp_bbb = 10^(log_papp)

    Classification:
        'high'     if Papp > 1e-6 cm/s
        'moderate' if 1e-7 <= Papp <= 1e-6 cm/s
        'low'      if Papp < 1e-7 cm/s

    Args:
        logP: Octanol-water partition coefficient
        mw: Molecular weight (Da)
        psa: Polar surface area (Angstrom^2)

    Returns:
        dict with logP, psa, mw, papp_bbb_cm_s, class_bbb
    """
    log_papp = 0.5 * logP - 0.01 * psa - 0.003 * mw - 4.0
    papp = 10.0**log_papp

    if papp > 1e-6:
        class_bbb = "high"
    elif papp > 1e-7:
        class_bbb = "moderate"
    else:
        class_bbb = "low"

    return {
        "logP": logP,
        "psa": psa,
        "mw": mw,
        "papp_bbb_cm_s": papp,
        "class_bbb": class_bbb,
    }


def cns_mpp_ratio(logP: float, psa: float) -> float:
    """CNS membrane permeability at pH 7.4 (MPP ratio).

    Empirical equation:
        MPP = 10 ^ (0.298 * logP - 0.0148 * psa - 0.987)

    CNS MPP > -2 (log scale) suggests CNS permeability.

    Args:
        logP: Octanol-water partition coefficient
        psa: Polar surface area (Angstrom^2)

    Returns:
        MPP value (linear scale)
    """
    log_mpp = 0.298 * logP - 0.0148 * psa - 0.987
    return 10.0**log_mpp


def pgp_efflux_correction(kp_brain_passive: float, pgp_efflux_ratio: float) -> float:
    """Correct brain Kp for P-glycoprotein efflux.

    P-gp reduces brain exposure:
        kp_brain = kp_brain_passive / pgp_efflux_ratio

    Args:
        kp_brain_passive: Passive brain-to-plasma partition coefficient
        pgp_efflux_ratio: P-gp efflux ratio (>= 1; 1 = no P-gp effect)

    Returns:
        Corrected brain Kp (> 0)
    """
    if pgp_efflux_ratio < 1.0:
        raise ValueError(f"pgp_efflux_ratio must be >= 1, got {pgp_efflux_ratio}")
    if kp_brain_passive <= 0:
        raise ValueError(f"kp_brain_passive must be > 0, got {kp_brain_passive}")

    kp_brain = kp_brain_passive / pgp_efflux_ratio
    return kp_brain


def simulate_bbb_pk(
    drug_name: str,
    dose_mg: float,
    vd_plasma_L: float,
    cl_plasma_L_per_h: float,
    papp_bbb_cm_s: float,
    pgp_efflux_ratio: float = 1.0,
    vd_brain_L: float = 1.4,
    t_end_h: float = 24.0,
    dt_h: float = 0.1,
) -> dict:
    """Simulate 2-compartment plasma-brain PK with BBB transport.

    Compartments:
        Cp = plasma concentration (mg/L)
        Cb = brain concentration (mg/L)

    Rate constants:
        ke_plasma = cl_plasma / vd_plasma
        k_brain_in = papp_bbb_cm_s * 3600 * 150 / (vd_brain_L * 1000)
            (150 cm^2 BBB surface area, convert cm/s to 1/h, mL to L)
        k_brain_out = k_brain_in * pgp_efflux_ratio

    ODEs (Forward Euler, IV bolus):
        dCp/dt = -ke_plasma * Cp - k_brain_in * Cp + k_brain_out * Cb
        dCb/dt = k_brain_in * Cp - k_brain_out * Cb

    Initial conditions:
        Cp0 = dose_mg / vd_plasma_L
        Cb0 = 0.0

    Args:
        drug_name: Name of the drug
        dose_mg: IV bolus dose (mg)
        vd_plasma_L: Plasma volume of distribution (L)
        cl_plasma_L_per_h: Plasma clearance (L/h)
        papp_bbb_cm_s: BBB passive permeability (cm/s)
        pgp_efflux_ratio: P-gp efflux ratio (>= 1.0)
        vd_brain_L: Brain volume of distribution (L)
        t_end_h: Simulation end time (h)
        dt_h: Time step (h)

    Returns:
        dict with times_h, c_plasma_mg_L, c_brain_mg_L,
              kp_brain_ss, auc_brain, notes
    """
    if vd_plasma_L <= 0:
        raise ValueError("vd_plasma_L must be > 0")
    if cl_plasma_L_per_h < 0:
        raise ValueError("cl_plasma_L_per_h must be >= 0")
    if papp_bbb_cm_s < 0:
        raise ValueError("papp_bbb_cm_s must be >= 0")
    if pgp_efflux_ratio < 1.0:
        raise ValueError("pgp_efflux_ratio must be >= 1")
    if vd_brain_L <= 0:
        raise ValueError("vd_brain_L must be > 0")

    ke_plasma = cl_plasma_L_per_h / vd_plasma_L

    # k_brain_in: papp (cm/s) * 3600 (s/h) * 150 (cm^2 surface) / (vd_brain mL)
    # vd_brain_L * 1000 = volume in mL
    k_brain_in = papp_bbb_cm_s * 3600.0 * 150.0 / (vd_brain_L * 1000.0)
    k_brain_out = k_brain_in * pgp_efflux_ratio

    cp = dose_mg / vd_plasma_L
    cb = 0.0

    n_steps = int(t_end_h / dt_h) + 1
    times_h = []
    c_plasma = []
    c_brain = []

    t = 0.0
    for _ in range(n_steps):
        times_h.append(t)
        c_plasma.append(cp)
        c_brain.append(cb)

        dcp = -ke_plasma * cp - k_brain_in * cp + k_brain_out * cb
        dcb = k_brain_in * cp - k_brain_out * cb

        cp = max(0.0, cp + dcp * dt_h)
        cb = max(0.0, cb + dcb * dt_h)
        t += dt_h

    # kp_brain_ss: Cb_max / Cp at time of Cb_max
    cb_max = max(c_brain)
    idx_max = c_brain.index(cb_max)
    cp_at_cbmax = c_plasma[idx_max]
    if cp_at_cbmax > 0:
        kp_brain_ss = cb_max / cp_at_cbmax
    else:
        kp_brain_ss = 0.0

    # AUC brain via trapezoidal
    auc_brain = 0.0
    for i in range(1, len(times_h)):
        auc_brain += 0.5 * (c_brain[i] + c_brain[i - 1]) * (times_h[i] - times_h[i - 1])

    notes = (
        f"Drug: {drug_name}; BBB Papp={papp_bbb_cm_s:.2e} cm/s; "
        f"P-gp efflux ratio={pgp_efflux_ratio:.1f}"
    )

    return {
        "times_h": times_h,
        "c_plasma_mg_L": c_plasma,
        "c_brain_mg_L": c_brain,
        "kp_brain_ss": kp_brain_ss,
        "auc_brain": auc_brain,
        "notes": notes,
    }


def kp_uu_brain(kp_brain: float, fu_plasma: float, fu_brain: float) -> float:
    """Calculate unbound brain-to-plasma partition coefficient.

    Kp,uu = Kp * (fu_brain / fu_plasma)

    Args:
        kp_brain: Total brain-to-plasma partition coefficient
        fu_plasma: Unbound fraction in plasma (0 < fu <= 1)
        fu_brain: Unbound fraction in brain (0 < fu <= 1)

    Returns:
        Kp,uu (unbound partition coefficient)
    """
    if kp_brain <= 0:
        raise ValueError(f"kp_brain must be > 0, got {kp_brain}")
    if fu_plasma <= 0:
        raise ValueError(f"fu_plasma must be > 0, got {fu_plasma}")
    if fu_brain <= 0:
        raise ValueError(f"fu_brain must be > 0, got {fu_brain}")

    return kp_brain * (fu_brain / fu_plasma)
