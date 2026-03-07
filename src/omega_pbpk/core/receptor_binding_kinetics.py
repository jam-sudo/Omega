"""Phase 602 — Receptor binding kinetics.

Simulates kon/koff/KD dynamics, occupancy profiles, competitive binding,
and residence time using forward Euler integration.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class ReceptorBindingResult:
    drug_name: str
    receptor: str
    kon_per_M_per_s: float
    koff_per_s: float
    kd_nM: float  # equilibrium dissociation constant
    residence_time_s: float  # 1/koff
    occupancy_at_kd: float  # 0.5 (always)
    times_s: list
    occupancy_profile: list  # fraction occupied [0, 1]
    auc_occupancy: float  # trapezoidal AUC of occupancy vs time (seconds)
    t_half_dissociation_s: float  # 0.693/koff
    drug_receptor_complex_nM: list  # amount of DR complex (nM)
    notes: str


def compute_kd(kon_per_nM_per_s: float, koff_per_s: float) -> float:
    """KD = koff / kon (in nM)."""
    return koff_per_s / kon_per_nM_per_s


def equilibrium_occupancy(drug_conc_nM: float, kd_nM: float) -> float:
    """RO = [D] / ([D] + KD)."""
    return drug_conc_nM / (drug_conc_nM + kd_nM)


def competitive_binding(
    drug_conc_nM: float,
    competitor_conc_nM: float,
    kd_drug_nM: float,
    kd_competitor_nM: float,
) -> dict:
    """Compute equilibrium occupancy with competitive binding (Cheng-Prusoff).

    Effective KD_app for drug = KD_drug * (1 + [competitor] / KD_competitor).
    Effective KD_app for competitor = KD_competitor * (1 + [drug] / KD_drug).

    Returns drug_occupancy, competitor_occupancy, total_occupancy.
    """
    kd_app_drug = kd_drug_nM * (1.0 + competitor_conc_nM / kd_competitor_nM)
    kd_app_competitor = kd_competitor_nM * (1.0 + drug_conc_nM / kd_drug_nM)

    drug_occupancy = drug_conc_nM / (drug_conc_nM + kd_app_drug)
    competitor_occupancy = competitor_conc_nM / (competitor_conc_nM + kd_app_competitor)

    # Total occupancy is the fraction of receptors bound by either molecule.
    # In a competitive system the two fractions are mutually exclusive at steady state;
    # combined they cannot exceed 1. We cap to 1 for numerical safety.
    total_occupancy = min(1.0, drug_occupancy + competitor_occupancy)

    return {
        "drug_occupancy": drug_occupancy,
        "competitor_occupancy": competitor_occupancy,
        "total_occupancy": total_occupancy,
    }


def simulate_receptor_binding(
    drug_name: str,
    receptor: str,
    drug_conc_nM: float,
    total_receptor_nM: float,
    kon_per_nM_per_s: float,
    koff_per_s: float,
    t_end_s: float = 600.0,
    dt_s: float = 1.0,
) -> ReceptorBindingResult:
    """Simulate receptor binding kinetics via forward Euler.

    ODE: dDR/dt = kon * [D] * (R_total - DR) - koff * DR
    [D] is treated as constant (drug not depleted by binding).
    """
    # --- Validation ---
    if drug_conc_nM <= 0:
        raise ValueError(f"drug_conc_nM must be > 0, got {drug_conc_nM}")
    if total_receptor_nM <= 0:
        raise ValueError(f"total_receptor_nM must be > 0, got {total_receptor_nM}")
    if kon_per_nM_per_s <= 0:
        raise ValueError(f"kon_per_nM_per_s must be > 0, got {kon_per_nM_per_s}")
    if koff_per_s <= 0:
        raise ValueError(f"koff_per_s must be > 0, got {koff_per_s}")
    if t_end_s <= 0:
        raise ValueError(f"t_end_s must be > 0, got {t_end_s}")

    kd_nM = compute_kd(kon_per_nM_per_s, koff_per_s)

    # --- Forward Euler ---
    n_steps = max(1, int(t_end_s / dt_s))
    times_s: list[float] = []
    occupancy_profile: list[float] = []
    dr_profile: list[float] = []

    dr = 0.0  # initial complex = 0
    t = 0.0

    for _ in range(n_steps + 1):
        occ = dr / total_receptor_nM
        times_s.append(t)
        occupancy_profile.append(occ)
        dr_profile.append(dr)

        if t >= t_end_s:
            break

        # dDR/dt = kon*[D]*(R_tot - DR) - koff*DR
        ddr_dt = kon_per_nM_per_s * drug_conc_nM * (total_receptor_nM - dr) - koff_per_s * dr
        dr = max(0.0, min(total_receptor_nM, dr + ddr_dt * dt_s))
        t = min(t + dt_s, t_end_s)

    # Trapezoidal AUC for occupancy
    auc = 0.0
    for i in range(1, len(times_s)):
        dt = times_s[i] - times_s[i - 1]
        auc += 0.5 * (occupancy_profile[i - 1] + occupancy_profile[i]) * dt

    residence_time_s = 1.0 / koff_per_s
    t_half_dissociation_s = math.log(2.0) / koff_per_s

    # KD in 1/nM units for converting kon to per_M_per_s
    kon_per_M_per_s = kon_per_nM_per_s * 1e9

    notes = (
        f"KD={kd_nM:.4g} nM, kon={kon_per_M_per_s:.4g} M^-1s^-1, "
        f"koff={koff_per_s:.4g} s^-1, residence_time={residence_time_s:.2f} s, "
        f"t½_dissociation={t_half_dissociation_s:.2f} s, "
        f"equilibrium_occupancy={equilibrium_occupancy(drug_conc_nM, kd_nM):.3f}"
    )

    return ReceptorBindingResult(
        drug_name=drug_name,
        receptor=receptor,
        kon_per_M_per_s=kon_per_M_per_s,
        koff_per_s=koff_per_s,
        kd_nM=kd_nM,
        residence_time_s=residence_time_s,
        occupancy_at_kd=0.5,
        times_s=times_s,
        occupancy_profile=occupancy_profile,
        auc_occupancy=auc,
        t_half_dissociation_s=t_half_dissociation_s,
        drug_receptor_complex_nM=dr_profile,
        notes=notes,
    )
