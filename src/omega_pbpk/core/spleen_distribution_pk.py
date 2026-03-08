"""Phase 1031 — Drug Spleen Distribution PK.

Models drug distribution into spleen — important for nanoparticles,
liposomes, and immune cells (MPS uptake).
"""

from __future__ import annotations

from dataclasses import dataclass
from math import exp


def _trapz(times: list[float], values: list[float]) -> float:
    """Manual trapezoidal AUC integration."""
    if len(times) < 2:
        return 0.0
    auc = 0.0
    for i in range(len(times) - 1):
        auc += 0.5 * (values[i] + values[i + 1]) * (times[i + 1] - times[i])
    return auc


@dataclass
class SpleenDistributionResult:
    """Result of spleen distribution PK simulation."""

    drug_name: str
    dose_mg: float
    times_h: list
    c_plasma_mg_L: list
    c_spleen_mg_L: list
    c_red_pulp_mg_L: list
    auc_plasma: float
    auc_spleen: float
    cmax_spleen: float
    tmax_spleen_h: float
    spleen_to_plasma_ratio: float
    mps_uptake_fraction: float
    notes: str


def simulate_spleen_distribution(
    drug_name: str,
    dose_mg: float,
    particle_size_nm: float = 100.0,
    cl_sys_L_per_h: float = 5.0,
    vd_sys_L: float = 30.0,
    q_spleen_L_h: float = 0.077,
    v_spleen_L: float = 0.2,
    kp_spleen: float = 2.0,
    mps_rate_per_h: float = 0.0,
    t_end_h: float = 48.0,
    dt_h: float = 0.1,
) -> SpleenDistributionResult:
    """Simulate drug distribution into spleen.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    dose_mg:
        IV bolus dose in mg.
    particle_size_nm:
        Nanoparticle size (nm); use 0 for small molecules.
    cl_sys_L_per_h:
        Systemic clearance (L/h).
    vd_sys_L:
        Volume of distribution (L).
    q_spleen_L_h:
        Splenic blood flow (L/h); default ~1.28 mL/min.
    v_spleen_L:
        Spleen volume (L); default 200 mL.
    kp_spleen:
        Spleen/plasma partition coefficient.
    mps_rate_per_h:
        MPS phagocytosis rate (1/h); 0 for small molecules.
    t_end_h:
        Simulation end time (h).
    dt_h:
        ODE integration step (h).

    Returns
    -------
    SpleenDistributionResult

    Notes
    -----
    Mass-balanced 3-compartment ODE (plasma, spleen tissue, red pulp MPS):

    Plasma:
        vd  * dCp/dt = -CL*Cp - q*Cp + q*Cs/kp
        dCp/dt = -(CL/vd)*Cp - (q/vd)*Cp + (q/(vd*kp))*Cs

    Spleen:
        v_s * dCs/dt = q*Cp - q*Cs/kp - k_mps*v_s*Cs
        dCs/dt = (q/v_s)*Cp - (q/(v_s*kp))*Cs - k_mps*Cs

    Red pulp (irreversible MPS sequestration):
        dCrp/dt = k_mps * Cs   (accumulates)
    """
    # Validation
    if dose_mg <= 0:
        raise ValueError(f"dose_mg must be > 0, got {dose_mg}")
    if cl_sys_L_per_h <= 0:
        raise ValueError(f"cl_sys_L_per_h must be > 0, got {cl_sys_L_per_h}")
    if vd_sys_L <= 0:
        raise ValueError(f"vd_sys_L must be > 0, got {vd_sys_L}")
    if v_spleen_L <= 0:
        raise ValueError(f"v_spleen_L must be > 0, got {v_spleen_L}")
    if kp_spleen <= 0:
        raise ValueError(f"kp_spleen must be > 0, got {kp_spleen}")
    if particle_size_nm < 0:
        raise ValueError(f"particle_size_nm must be >= 0, got {particle_size_nm}")

    # Adjust MPS rate for nanoparticles (Gaussian peak at 150 nm)
    effective_mps_rate = mps_rate_per_h
    if particle_size_nm > 0:
        size_factor = exp(-((particle_size_nm - 150.0) ** 2) / (2.0 * 100.0**2))
        effective_mps_rate = max(mps_rate_per_h, 0.3 * size_factor)

    # Rate constants (mass-balanced)
    # Plasma
    ke = cl_sys_L_per_h / vd_sys_L  # systemic elimination rate (/h)
    k_sp_in = q_spleen_L_h / vd_sys_L  # plasma → spleen flow rate (/h)
    k_sp_ret = q_spleen_L_h / (vd_sys_L * kp_spleen)  # spleen → plasma return (/h, per vd)

    # Spleen
    k_in_sp = q_spleen_L_h / v_spleen_L  # plasma → spleen, per spleen volume (/h)
    k_out_sp = q_spleen_L_h / (v_spleen_L * kp_spleen)  # spleen → plasma (/h)
    k_mps = effective_mps_rate

    # Initial conditions: IV bolus → plasma only
    c_plasma = dose_mg / vd_sys_L
    c_spleen = 0.0
    c_red_pulp = 0.0

    # Storage
    times: list[float] = [0.0]
    c_plasma_list: list[float] = [c_plasma]
    c_spleen_list: list[float] = [c_spleen]
    c_red_pulp_list: list[float] = [c_red_pulp]

    # Forward Euler integration
    n_steps = int(round(t_end_h / dt_h))
    t = 0.0
    for _i in range(n_steps):
        dcp = -ke * c_plasma - k_sp_in * c_plasma + k_sp_ret * c_spleen
        dcs = k_in_sp * c_plasma - k_out_sp * c_spleen - k_mps * c_spleen
        drp = k_mps * c_spleen

        c_plasma = max(0.0, c_plasma + dt_h * dcp)
        c_spleen = max(0.0, c_spleen + dt_h * dcs)
        c_red_pulp = max(0.0, c_red_pulp + dt_h * drp)
        t += dt_h

        times.append(t)
        c_plasma_list.append(c_plasma)
        c_spleen_list.append(c_spleen)
        c_red_pulp_list.append(c_red_pulp)

    # Derived PK metrics
    auc_plasma = _trapz(times, c_plasma_list)
    auc_spleen = _trapz(times, c_spleen_list)

    cmax_spleen = max(c_spleen_list)
    tmax_spleen_h = times[c_spleen_list.index(cmax_spleen)]

    spleen_to_plasma_ratio = auc_spleen / auc_plasma if auc_plasma > 0 else 0.0

    # MPS uptake fraction: fraction of dose sequestered in red pulp (cumulative)
    final_red_pulp_mass = c_red_pulp_list[-1] * v_spleen_L
    mps_uptake_fraction = min(1.0, max(0.0, final_red_pulp_mass / dose_mg))

    # Notes
    note_parts = [f"Simulated {drug_name} spleen distribution over {t_end_h} h."]
    if particle_size_nm > 0:
        note_parts.append(
            f"Nanoparticle ({particle_size_nm} nm): effective MPS rate = {effective_mps_rate:.4f}/h."
        )
    else:
        note_parts.append("Small molecule: MPS uptake negligible.")
    note_parts.append(f"Spleen/plasma AUC ratio = {spleen_to_plasma_ratio:.3f}.")
    if mps_uptake_fraction > 0.1:
        note_parts.append(
            f"Significant MPS sequestration: {mps_uptake_fraction * 100:.1f}% of dose."
        )
    notes = " ".join(note_parts)

    return SpleenDistributionResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        times_h=times,
        c_plasma_mg_L=c_plasma_list,
        c_spleen_mg_L=c_spleen_list,
        c_red_pulp_mg_L=c_red_pulp_list,
        auc_plasma=auc_plasma,
        auc_spleen=auc_spleen,
        cmax_spleen=cmax_spleen,
        tmax_spleen_h=tmax_spleen_h,
        spleen_to_plasma_ratio=spleen_to_plasma_ratio,
        mps_uptake_fraction=mps_uptake_fraction,
        notes=notes,
    )
