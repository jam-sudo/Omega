"""Phase 1143: Prostate tissue PK model.

Models drug distribution in prostate tissue accounting for the blood-prostate
barrier (BPB), ion trapping in acidic prostatic fluid (pH ~6.5), and
lipophilicity-driven penetration.
"""

from dataclasses import dataclass


@dataclass
class ProstatePKResult:
    drug_name: str
    dose_mg: float
    logP: float
    ionization_type: str
    kp_prostate: float
    times_h: list
    c_plasma_mg_L: list
    c_prostate_mg_L: list
    cmax_prostate_mg_L: float
    tmax_prostate_h: float
    auc_prostate_mg_h_per_L: float
    auc_plasma_mg_h_per_L: float
    prostate_to_plasma_auc_ratio: float
    prostate_penetration_class: str
    clinical_utility: str
    notes: str


_VALID_IONIZATION = {"acid", "base", "neutral"}
_V_PROSTATE_L = 0.03  # ~30 mL adult prostate


def _compute_logp_factor(logP: float) -> float:
    if logP <= 0:
        return 0.2
    return max(0.2, min(3.0, logP / 2.0))


def _compute_ionization_factor(ionization_type: str) -> float:
    if ionization_type == "base":
        return 1.5  # ion trapping in acidic prostatic fluid
    if ionization_type == "acid":
        return 0.7
    return 1.0  # neutral


def _trapz(y: list, x: list) -> float:
    """Manual trapezoidal integration."""
    total = 0.0
    for i in range(len(x) - 1):
        total += 0.5 * (y[i] + y[i + 1]) * (x[i + 1] - x[i])
    return total


def _penetration_class(ratio: float) -> str:
    if ratio < 0.3:
        return "poor"
    if ratio <= 1.0:
        return "moderate"
    return "good"


def _clinical_utility(pen_class: str, ionization_type: str) -> str:
    if pen_class == "good":
        return "recommended"
    if pen_class == "moderate":
        return "borderline"
    return "not_recommended"


def simulate_prostate_pk(
    drug_name: str,
    dose_mg: float,
    logP: float,
    ionization_type: str = "neutral",
    cl_L_per_h: float = 5.0,
    vd_L: float = 50.0,
    t_end_h: float = 12.0,
    dt_h: float = 0.05,
) -> ProstatePKResult:
    """Simulate drug distribution in prostate tissue using a 2-compartment ODE.

    Compartments:
        C_plasma  (mg/L): systemic plasma
        C_prostate (mg/L): prostate tissue

    Parameters
    ----------
    drug_name : str
        Identifier for the drug.
    dose_mg : float
        IV bolus dose in mg.
    logP : float
        Octanol-water partition coefficient (log scale).
    ionization_type : str
        One of "acid", "base", "neutral".
    cl_L_per_h : float
        Total systemic clearance (L/h).
    vd_L : float
        Volume of distribution (L).
    t_end_h : float
        Simulation end time (h).
    dt_h : float
        Forward-Euler step size (h).

    Returns
    -------
    ProstatePKResult
    """
    # --- Validation ---
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if cl_L_per_h <= 0:
        raise ValueError("cl_L_per_h must be > 0")
    if vd_L <= 0:
        raise ValueError("vd_L must be > 0")
    if not (-5 <= logP <= 10):
        raise ValueError("logP must be in [-5, 10]")
    if ionization_type not in _VALID_IONIZATION:
        raise ValueError(
            f"ionization_type must be one of {_VALID_IONIZATION}, got {ionization_type!r}"
        )

    # --- Parameters ---
    logP_factor = _compute_logp_factor(logP)
    ion_factor = _compute_ionization_factor(ionization_type)

    kpr = 0.1 * logP_factor * ion_factor  # plasma → prostate (1/h)
    krp = 0.05  # prostate → plasma (1/h)
    kp_prostate = max(0.1, min(10.0, logP_factor * ion_factor))
    ke = cl_L_per_h / vd_L  # elimination rate (1/h)

    # --- Initial conditions ---
    cp = dose_mg / vd_L  # C_plasma(0)
    cpr = 0.0  # C_prostate(0)

    times_h = [0.0]
    c_plasma = [cp]
    c_prostate = [cpr]

    n_steps = int(round(t_end_h / dt_h))

    for _ in range(n_steps):
        # Volume-scaled back-transfer ensures mass balance:
        # prostate loses krp*Cpr/Kp (mg/L/h in prostate units),
        # plasma gains same amount corrected for volume ratio.
        back_term = krp * cpr / kp_prostate  # mg/L/h in prostate concentration units
        dcp = -ke * cp - kpr * cp + back_term * (_V_PROSTATE_L / vd_L)
        dcpr = kpr * (vd_L / _V_PROSTATE_L) * cp - back_term

        cp = max(0.0, cp + dcp * dt_h)
        cpr = max(0.0, cpr + dcpr * dt_h)

        times_h.append(times_h[-1] + dt_h)
        c_plasma.append(cp)
        c_prostate.append(cpr)

    # --- Derived metrics ---
    cmax_prostate = max(c_prostate)
    tmax_prostate = times_h[c_prostate.index(cmax_prostate)]
    auc_prostate = _trapz(c_prostate, times_h)
    auc_plasma = _trapz(c_plasma, times_h)
    ratio = auc_prostate / auc_plasma if auc_plasma > 0 else 0.0

    pen_class = _penetration_class(ratio)
    utility = _clinical_utility(pen_class, ionization_type)

    notes_parts = [
        f"logP_factor={logP_factor:.3f}",
        f"ion_factor={ion_factor:.2f}",
        f"kpr={kpr:.4f}/h",
        f"krp={krp:.4f}/h",
    ]
    if ionization_type == "base":
        notes_parts.append("ion trapping in acidic prostatic fluid (pH~6.5)")
    if pen_class == "poor":
        notes_parts.append("limited penetration through blood-prostate barrier")

    return ProstatePKResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        logP=logP,
        ionization_type=ionization_type,
        kp_prostate=kp_prostate,
        times_h=times_h,
        c_plasma_mg_L=c_plasma,
        c_prostate_mg_L=c_prostate,
        cmax_prostate_mg_L=cmax_prostate,
        tmax_prostate_h=tmax_prostate,
        auc_prostate_mg_h_per_L=auc_prostate,
        auc_plasma_mg_h_per_L=auc_plasma,
        prostate_to_plasma_auc_ratio=ratio,
        prostate_penetration_class=pen_class,
        clinical_utility=utility,
        notes="; ".join(notes_parts),
    )
