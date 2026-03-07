"""Phase 413 — Enterohepatic cycling driven by bile acid transport (OATP/NTCP)."""

from __future__ import annotations

from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class BileAcidTransportResult:
    """Result of bile acid transporter-driven enterohepatic cycling simulation."""

    drug_name: str
    dose_mg: float
    km_OATP_mg_L: float
    vmax_OATP_mg_h: float

    # Time-course arrays
    times_h: list[float] = field(default_factory=list)
    c_portal_mg_L: list[float] = field(default_factory=list)
    c_hep_mg_L: list[float] = field(default_factory=list)
    a_bile_mg: list[float] = field(default_factory=list)
    a_intestine_mg: list[float] = field(default_factory=list)

    # Summary metrics
    auc_portal: float = 0.0
    n_cycles_estimated: float = 0.0
    notes: str = ""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_VD_PORTAL_L = 5.0   # portal vein apparent volume (L)
_VD_HEP_L = 2.0      # hepatocyte apparent volume (L)
_K_BILE_PER_H = 0.5  # first-order bile → intestine transit rate (1/h)
_K_REABS_PER_H = 2.0  # first-order intestinal reabsorption rate (1/h)
_REABS_FRACTION = 0.95  # fraction of intestinal amount returning to portal


def _mm(vmax: float, km: float, conc: float) -> float:
    """Michaelis-Menten rate (mg/h)."""
    return vmax * conc / (km + conc) if (km + conc) > 0.0 else 0.0


def _auc_trapz(times: list[float], values: list[float]) -> float:
    """Manual trapezoidal AUC."""
    auc = 0.0
    for i in range(1, len(times)):
        auc += 0.5 * (values[i] + values[i - 1]) * (times[i] - times[i - 1])
    return auc


# ---------------------------------------------------------------------------
# Main simulation
# ---------------------------------------------------------------------------

def simulate_bile_acid_transport(
    drug_name: str,
    dose_mg: float,
    cl_hep_L_per_h: float,
    vd_L: float,
    km_OATP_mg_L: float,
    vmax_OATP_mg_h: float,
    km_NTCP_mg_L: float,
    vmax_NTCP_mg_h: float,
    bile_flow_L_per_h: float,
    t_end_h: float,
    dt_h: float,
) -> BileAcidTransportResult:
    """Simulate enterohepatic cycling driven by OATP/NTCP bile acid transporters.

    Parameters
    ----------
    drug_name : str
        Name of the drug/compound.
    dose_mg : float
        Oral dose administered into the portal compartment (mg).
    cl_hep_L_per_h : float
        Passive hepatic elimination clearance (L/h).
    vd_L : float
        Total volume of distribution (L; used for reference, not ODE compartments).
    km_OATP_mg_L : float
        Km for OATP hepatic uptake (mg/L), must be > 0.
    vmax_OATP_mg_h : float
        Vmax for OATP hepatic uptake (mg/h), must be >= 0.
    km_NTCP_mg_L : float
        Km for NTCP biliary excretion (mg/L), must be > 0.
    vmax_NTCP_mg_h : float
        Vmax for NTCP biliary excretion (mg/h), must be >= 0.
    bile_flow_L_per_h : float
        Bile flow rate (L/h); informational, not used directly in ODE.
    t_end_h : float
        Simulation end time (h).
    dt_h : float
        Forward Euler time step (h).

    Returns
    -------
    BileAcidTransportResult
    """
    # --- Input validation ---
    if not drug_name:
        raise ValueError("drug_name must be a non-empty string")
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if cl_hep_L_per_h < 0:
        raise ValueError("cl_hep_L_per_h must be >= 0")
    if vd_L <= 0:
        raise ValueError("vd_L must be > 0")
    if km_OATP_mg_L <= 0:
        raise ValueError("km_OATP_mg_L must be > 0")
    if vmax_OATP_mg_h < 0:
        raise ValueError("vmax_OATP_mg_h must be >= 0")
    if km_NTCP_mg_L <= 0:
        raise ValueError("km_NTCP_mg_L must be > 0")
    if vmax_NTCP_mg_h < 0:
        raise ValueError("vmax_NTCP_mg_h must be >= 0")
    if bile_flow_L_per_h < 0:
        raise ValueError("bile_flow_L_per_h must be >= 0")
    if t_end_h <= 0:
        raise ValueError("t_end_h must be > 0")
    if dt_h <= 0:
        raise ValueError("dt_h must be > 0")

    # --- Initial conditions ---
    # Dose administered directly into portal vein compartment
    c_portal = dose_mg / _VD_PORTAL_L   # mg/L
    c_hep = 0.0                          # mg/L
    a_bile = 0.0                         # mg (amount)
    a_intestine = 0.0                    # mg (amount)

    times: list[float] = []
    cp_list: list[float] = []
    ch_list: list[float] = []
    ab_list: list[float] = []
    ai_list: list[float] = []

    t = 0.0
    n_steps = int(round(t_end_h / dt_h))

    for _ in range(n_steps + 1):
        times.append(t)
        cp_list.append(c_portal)
        ch_list.append(c_hep)
        ab_list.append(a_bile)
        ai_list.append(a_intestine)

        # Rates (mg/h)
        hepatic_uptake = _mm(vmax_OATP_mg_h, km_OATP_mg_L, c_portal)
        biliary_excretion = _mm(vmax_NTCP_mg_h, km_NTCP_mg_L, c_hep)
        passive_elim = cl_hep_L_per_h * c_hep  # passive hepatic elimination

        bile_transit = _K_BILE_PER_H * a_bile
        intestinal_removal = _K_REABS_PER_H * a_intestine
        reabsorption_rate_mg_h = _REABS_FRACTION * intestinal_removal

        # ODEs (forward Euler)
        dc_portal = (
            -hepatic_uptake / _VD_PORTAL_L
            + reabsorption_rate_mg_h / _VD_PORTAL_L
        )
        dc_hep = (hepatic_uptake - biliary_excretion - passive_elim) / _VD_HEP_L
        da_bile = biliary_excretion - bile_transit
        da_intestine = bile_transit - intestinal_removal

        c_portal = max(0.0, c_portal + dc_portal * dt_h)
        c_hep = max(0.0, c_hep + dc_hep * dt_h)
        a_bile = max(0.0, a_bile + da_bile * dt_h)
        a_intestine = max(0.0, a_intestine + da_intestine * dt_h)

        t += dt_h

    auc_portal = _auc_trapz(times, cp_list)

    # Estimate number of enterohepatic cycles: each intestinal-to-portal transfer
    # is one cycle; estimate from cumulative reabsorbed / original dose
    total_reabsorbed = sum(
        _REABS_FRACTION * _K_REABS_PER_H * ai_list[i] * dt_h
        for i in range(len(ai_list))
    )
    n_cycles = total_reabsorbed / dose_mg if dose_mg > 0 else 0.0

    notes = (
        f"OATP-driven hepatic uptake, NTCP biliary excretion; "
        f"k_bile={_K_BILE_PER_H}/h, k_reabs={_K_REABS_PER_H}/h, "
        f"reabs_fraction={_REABS_FRACTION}"
    )

    return BileAcidTransportResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        km_OATP_mg_L=km_OATP_mg_L,
        vmax_OATP_mg_h=vmax_OATP_mg_h,
        times_h=times,
        c_portal_mg_L=cp_list,
        c_hep_mg_L=ch_list,
        a_bile_mg=ab_list,
        a_intestine_mg=ai_list,
        auc_portal=auc_portal,
        n_cycles_estimated=n_cycles,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Transporter inhibition effect
# ---------------------------------------------------------------------------

def transporter_inhibition_effect(
    drug_name: str,
    dose_mg: float,
    oatp_inhibition_pct: float,
    ntcp_inhibition_pct: float,
) -> dict:
    """Compare portal AUC under transporter inhibition vs normal.

    Parameters
    ----------
    drug_name : str
        Drug name.
    dose_mg : float
        Dose (mg), must be > 0.
    oatp_inhibition_pct : float
        Percent inhibition of OATP Vmax (0–100).
    ntcp_inhibition_pct : float
        Percent inhibition of NTCP Vmax (0–100).

    Returns
    -------
    dict with keys:
        auc_portal_normal, auc_portal_inhibited, auc_ratio,
        oatp_inhibition_pct, ntcp_inhibition_pct, interpretation
    """
    if not drug_name:
        raise ValueError("drug_name must be a non-empty string")
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if not (0.0 <= oatp_inhibition_pct <= 100.0):
        raise ValueError("oatp_inhibition_pct must be in [0, 100]")
    if not (0.0 <= ntcp_inhibition_pct <= 100.0):
        raise ValueError("ntcp_inhibition_pct must be in [0, 100]")

    # Default reference parameters
    _DEFAULT_PARAMS = dict(
        cl_hep_L_per_h=1.0,
        vd_L=50.0,
        km_OATP_mg_L=0.5,
        vmax_OATP_mg_h=5.0,
        km_NTCP_mg_L=0.5,
        vmax_NTCP_mg_h=3.0,
        bile_flow_L_per_h=0.9,
        t_end_h=24.0,
        dt_h=0.05,
    )

    # Normal simulation
    normal = simulate_bile_acid_transport(
        drug_name=drug_name,
        dose_mg=dose_mg,
        **_DEFAULT_PARAMS,  # type: ignore[arg-type]
    )

    # Inhibited: reduce Vmax values
    inh_params = dict(_DEFAULT_PARAMS)
    inh_params["vmax_OATP_mg_h"] = _DEFAULT_PARAMS["vmax_OATP_mg_h"] * (  # type: ignore[assignment]
        1.0 - oatp_inhibition_pct / 100.0
    )
    inh_params["vmax_NTCP_mg_h"] = _DEFAULT_PARAMS["vmax_NTCP_mg_h"] * (  # type: ignore[assignment]
        1.0 - ntcp_inhibition_pct / 100.0
    )

    inhibited = simulate_bile_acid_transport(
        drug_name=drug_name,
        dose_mg=dose_mg,
        **inh_params,  # type: ignore[arg-type]
    )

    auc_ratio = (
        inhibited.auc_portal / normal.auc_portal
        if normal.auc_portal > 0.0
        else float("nan")
    )

    if auc_ratio > 2.0:
        interpretation = "Strong DDI: portal AUC >2-fold increase under transporter inhibition"
    elif auc_ratio > 1.25:
        interpretation = "Moderate DDI: portal AUC 1.25–2-fold increase"
    else:
        interpretation = "Minimal DDI: portal AUC change <25%"

    return {
        "drug_name": drug_name,
        "dose_mg": dose_mg,
        "auc_portal_normal": normal.auc_portal,
        "auc_portal_inhibited": inhibited.auc_portal,
        "auc_ratio": auc_ratio,
        "oatp_inhibition_pct": oatp_inhibition_pct,
        "ntcp_inhibition_pct": ntcp_inhibition_pct,
        "n_cycles_normal": normal.n_cycles_estimated,
        "n_cycles_inhibited": inhibited.n_cycles_estimated,
        "interpretation": interpretation,
    }
