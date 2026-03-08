"""Phase 679 — Rectal Drug Absorption model."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RectalAbsorptionResult:
    """Result of rectal drug absorption simulation."""

    drug_name: str
    dose_mg: float
    formulation_type: str
    times_h: list = field(default_factory=list)
    c_rectal_depot_mg: list = field(default_factory=list)
    c_portal_mg_L: list = field(default_factory=list)
    c_systemic_mg_L: list = field(default_factory=list)
    cmax_systemic_mg_L: float = 0.0
    tmax_systemic_h: float = 0.0
    auc_systemic_mg_h_per_L: float = 0.0
    f_absorbed: float = 0.0
    f_portal_pct: float = 0.0
    f_systemic_direct_pct: float = 0.0
    first_pass_avoided_pct: float = 0.0
    bioavailability_pct: float = 0.0
    notes: str = ""


_FORMULATION_PARAMS = {
    "suppository": {"ka": 0.5, "f_absorbed": 0.7, "portal_fraction": 0.5},
    "enema": {"ka": 1.5, "f_absorbed": 0.85, "portal_fraction": 0.4},
    "gel": {"ka": 0.8, "f_absorbed": 0.75, "portal_fraction": 0.45},
}

_VALID_FORMULATIONS = set(_FORMULATION_PARAMS.keys())


def simulate_rectal_absorption(
    drug_name: str,
    dose_mg: float,
    logP: float,
    mw_Da: float,
    cl_sys_L_per_h: float,
    vd_sys_L: float,
    formulation_type: str = "suppository",
    t_end_h: float = 12.0,
    dt_h: float = 0.1,
) -> RectalAbsorptionResult:
    """Simulate rectal drug absorption."""
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if cl_sys_L_per_h <= 0:
        raise ValueError("cl_sys_L_per_h must be > 0")
    if vd_sys_L <= 0:
        raise ValueError("vd_sys_L must be > 0")
    if formulation_type not in _VALID_FORMULATIONS:
        raise ValueError(f"formulation_type must be one of {sorted(_VALID_FORMULATIONS)}")

    params = _FORMULATION_PARAMS[formulation_type]
    ka = params["ka"]
    f_abs = params["f_absorbed"]
    portal_fraction = params["portal_fraction"]

    f_direct = 1.0 - portal_fraction
    f_portal = portal_fraction

    # Hepatic extraction
    hepatic_blood_flow = 90.0
    E_h = cl_sys_L_per_h / (cl_sys_L_per_h + hepatic_blood_flow)
    E_h = min(0.99, E_h)

    # logP adjustment
    ka_adjusted = ka * max(0.5, min(2.0, 1.0 + logP * 0.2))

    # Portal compartment
    V_portal = 0.5
    k_portal_clear = hepatic_blood_flow / V_portal

    # Initial conditions
    A_depot = dose_mg * f_abs
    C_portal = 0.0
    C_sys = 0.0

    times_h = []
    c_rectal_depot_mg = []
    c_portal_mg_L = []
    c_systemic_mg_L = []

    cmax = 0.0
    tmax = 0.0

    n_steps = int(t_end_h / dt_h) + 1

    for i in range(n_steps):
        t = i * dt_h
        times_h.append(t)
        c_rectal_depot_mg.append(A_depot)
        c_portal_mg_L.append(C_portal)
        c_systemic_mg_L.append(C_sys)

        if C_sys > cmax:
            cmax = C_sys
            tmax = t

        # Forward Euler
        dA_depot = -ka_adjusted * A_depot
        dC_portal = ka_adjusted * A_depot * f_portal / V_portal - k_portal_clear * C_portal
        dC_sys = (
            ka_adjusted * A_depot * f_direct / vd_sys_L
            + k_portal_clear * C_portal * (1.0 - E_h) / vd_sys_L
            - (cl_sys_L_per_h / vd_sys_L) * C_sys
        )

        A_depot = max(0.0, A_depot + dA_depot * dt_h)
        C_portal = max(0.0, C_portal + dC_portal * dt_h)
        C_sys = max(0.0, C_sys + dC_sys * dt_h)

    # Manual trapezoidal AUC
    auc = sum(
        0.5 * (c_systemic_mg_L[i] + c_systemic_mg_L[i - 1]) * (times_h[i] - times_h[i - 1])
        for i in range(1, len(times_h))
    )

    first_pass_avoided_pct = f_direct * 100.0
    bioavailability_pct = f_abs * (f_direct + f_portal * (1.0 - E_h)) * 100.0
    bioavailability_pct = max(0.0, min(100.0, bioavailability_pct))

    return RectalAbsorptionResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        formulation_type=formulation_type,
        times_h=times_h,
        c_rectal_depot_mg=c_rectal_depot_mg,
        c_portal_mg_L=c_portal_mg_L,
        c_systemic_mg_L=c_systemic_mg_L,
        cmax_systemic_mg_L=cmax,
        tmax_systemic_h=tmax,
        auc_systemic_mg_h_per_L=auc,
        f_absorbed=f_abs,
        f_portal_pct=f_portal * 100.0,
        f_systemic_direct_pct=f_direct * 100.0,
        first_pass_avoided_pct=first_pass_avoided_pct,
        bioavailability_pct=bioavailability_pct,
        notes=f"Rectal {formulation_type} absorption; MW={mw_Da} Da, logP={logP}",
    )
