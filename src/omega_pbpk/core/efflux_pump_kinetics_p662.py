"""Drug efflux pump kinetics model for epithelial cell transport.

Models P-gp, BCRP, and MRP2 efflux pump kinetics in a three-compartment
transcellular transport system (apical, intracellular, basolateral).

References
----------
- Giacomini KM et al., Nat Rev Drug Discov. 2010;9(3):215-236 (transporter pharmacology)
- Polli JW et al., J Pharmacol Exp Ther. 2001;299(2):620-628 (efflux transport)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

VALID_PUMPS = {"Pgp", "BCRP", "MRP2"}


@dataclass
class EffluxPumpKineticsResult:
    """Result container for efflux pump kinetics simulation."""

    drug_name: str
    pump_name: str
    times_h: list[float] = field(default_factory=list)
    c_intracellular_mg_L: list[float] = field(default_factory=list)
    c_extracellular_mg_L: list[float] = field(default_factory=list)
    c_basolateral_mg_L: list[float] = field(default_factory=list)
    efflux_rate_mg_h: list[float] = field(default_factory=list)
    net_transport_ratio: float = 0.0
    efflux_efficiency_pct: float = 0.0
    ic50_inhibition_mg_L: float = 0.0
    steady_state_intracellular_mg_L: float = 0.0
    permeability_ab_cm_s: float = 0.0
    permeability_ba_cm_s: float = 0.0
    efflux_ratio: float = 0.0
    transporter_saturation_pct: float = 0.0
    notes: str = ""


def simulate_efflux_pump(
    drug_name: str,
    pump_name: str,
    initial_apical_mg_L: float,
    vmax_mg_h: float,
    km_mg_L: float,
    passive_perm_cm_s: float = 1e-6,
    cell_volume_mL: float = 0.001,
    t_end_h: float = 4.0,
    dt_h: float = 0.01,
) -> EffluxPumpKineticsResult:
    """Simulate drug efflux pump kinetics in epithelial cells.

    Parameters
    ----------
    drug_name : str
        Name of the drug.
    pump_name : str
        Efflux pump name: "Pgp", "BCRP", or "MRP2".
    initial_apical_mg_L : float
        Initial apical (donor) concentration in mg/L.
    vmax_mg_h : float
        Maximum efflux rate (mg/h).
    km_mg_L : float
        Michaelis-Menten constant (mg/L).
    passive_perm_cm_s : float
        Passive permeability (cm/s).
    cell_volume_mL : float
        Cell monolayer volume (mL).
    t_end_h : float
        Simulation duration in hours.
    dt_h : float
        Time step for forward Euler integration.

    Returns
    -------
    EffluxPumpKineticsResult
    """
    # Validation
    if pump_name not in VALID_PUMPS:
        raise ValueError(f"pump_name must be one of {sorted(VALID_PUMPS)}")
    if initial_apical_mg_L <= 0:
        raise ValueError("initial_apical_mg_L must be positive")
    if vmax_mg_h <= 0:
        raise ValueError("vmax_mg_h must be positive")
    if km_mg_L <= 0:
        raise ValueError("km_mg_L must be positive")

    # Volumes in L
    v_ap_l = 0.001  # 1 mL apical
    v_cell_l = cell_volume_mL / 1000.0
    v_baso_l = 0.001  # 1 mL basolateral

    # Passive permeability rate constant: P * A * 3600 (L/h)
    # area = 10 cm2, P in cm/s, A in cm2 → P*A = cm3/s = mL/s
    # * 3600 s/h = mL/h; / 1000 = L/h
    kp = passive_perm_cm_s * 10.0 * 3600.0 / 1000.0  # L/h

    # Initial state (all in mg/L)
    c_ap = initial_apical_mg_L
    c_in = 0.0
    c_ba = 0.0

    n_steps = int(math.ceil(t_end_h / dt_h))

    times: list[float] = []
    c_intra_list: list[float] = []
    c_extra_list: list[float] = []
    c_baso_list: list[float] = []
    efflux_rate_list: list[float] = []

    total_j_efflux = 0.0
    total_j_passive_in = 0.0
    max_j_efflux = 0.0

    for i in range(n_steps + 1):
        t = i * dt_h

        # Fluxes (mg/h)
        j_passive_in = kp * (c_ap - c_in)
        j_efflux = vmax_mg_h * c_in / (km_mg_L + c_in) if c_in > 0 else 0.0
        j_passive_baso = kp * 0.5 * (c_in - c_ba)

        times.append(round(t, 6))
        c_intra_list.append(max(c_in, 0.0))
        c_extra_list.append(max(c_ap, 0.0))
        c_baso_list.append(max(c_ba, 0.0))
        efflux_rate_list.append(max(j_efflux, 0.0))

        if j_efflux > max_j_efflux:
            max_j_efflux = j_efflux

        if i == n_steps:
            break

        # Accumulate for efficiency calc
        total_j_efflux += j_efflux * dt_h
        total_j_passive_in += j_passive_in * dt_h if j_passive_in > 0 else 0.0

        # ODEs (forward Euler)
        dc_ap = (-j_passive_in + j_efflux) / v_ap_l
        dc_in = (j_passive_in - j_efflux - j_passive_baso) / v_cell_l
        dc_ba = j_passive_baso / v_baso_l

        c_ap += dc_ap * dt_h
        c_in += dc_in * dt_h
        c_ba += dc_ba * dt_h

        # Clamp
        c_ap = max(c_ap, 0.0)
        c_in = max(c_in, 0.0)
        c_ba = max(c_ba, 0.0)

    # Post-processing
    ss_intra = c_intra_list[-1] if c_intra_list else 0.0

    # Net transport ratio
    c_ap_final = c_extra_list[-1] if c_extra_list else 1.0
    c_ba_final = c_baso_list[-1] if c_baso_list else 0.0
    net_ratio = c_ba_final / c_ap_final if c_ap_final > 0 else 0.0

    # Efflux efficiency
    eff_pct = total_j_efflux / (total_j_passive_in + 1e-9) * 100.0
    eff_pct = max(0.0, min(100.0, eff_pct))

    # Efflux ratio (total efflux / total passive in)
    efflux_r = total_j_efflux / (total_j_passive_in + 1e-9) if total_j_passive_in > 0 else 0.0

    # Permeability estimates
    perm_ab = passive_perm_cm_s
    perm_ba = passive_perm_cm_s + vmax_mg_h / (km_mg_L * 10.0 * 3600.0 + 1e-9) * 1e-4

    # IC50 approximation
    ic50 = km_mg_L * 2.0

    # Transporter saturation
    sat_pct = (max_j_efflux / vmax_mg_h) * 100.0 if vmax_mg_h > 0 else 0.0
    sat_pct = max(0.0, min(100.0, sat_pct))

    notes_parts = []
    if efflux_r > 2.0:
        notes_parts.append("Strong efflux substrate")
    if sat_pct > 80.0:
        notes_parts.append("Transporter near saturation")

    return EffluxPumpKineticsResult(
        drug_name=drug_name,
        pump_name=pump_name,
        times_h=times,
        c_intracellular_mg_L=c_intra_list,
        c_extracellular_mg_L=c_extra_list,
        c_basolateral_mg_L=c_baso_list,
        efflux_rate_mg_h=efflux_rate_list,
        net_transport_ratio=net_ratio,
        efflux_efficiency_pct=eff_pct,
        ic50_inhibition_mg_L=ic50,
        steady_state_intracellular_mg_L=ss_intra,
        permeability_ab_cm_s=perm_ab,
        permeability_ba_cm_s=perm_ba,
        efflux_ratio=efflux_r,
        transporter_saturation_pct=sat_pct,
        notes="; ".join(notes_parts),
    )
