"""
Phase 1129 — Bile acid enterohepatic cycling and drug co-transport.

Models the recycling of bile-acid-conjugated drugs through the
enterohepatic circulation using a three-compartment Forward-Euler ODE.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class BileAcidPKResult:
    """Result of a bile-acid PK simulation."""

    drug_name: str
    dose_mg: float
    f_reabs: float
    times_h: list
    c_plasma_mg_L: list
    a_bile_pool_mg: list
    a_intestine_mg: list
    cmax_plasma_mg_L: float
    tmax_plasma_h: float
    auc_plasma_mg_h_per_L: float
    cycling_efficiency: float
    secondary_peak_present: bool
    fecal_recovery_pct: float
    notes: str


def _local_maxima(values: list) -> int:
    """Return the number of local maxima in a sequence."""
    count = 0
    n = len(values)
    for i in range(1, n - 1):
        if values[i] > values[i - 1] and values[i] > values[i + 1]:
            count += 1
    return count


def simulate_bile_acid_pk(
    drug_name: str,
    dose_mg: float,
    f_reabs: float = 0.95,
    k_sec_per_h: float = 0.1,
    k_reabs_per_h: float = 0.5,
    k_hepatic_per_h: float = 0.05,
    cl_L_per_h: float = 2.0,
    vd_L: float = 10.0,
    t_end_h: float = 48.0,
    dt_h: float = 0.1,
) -> BileAcidPKResult:
    """Simulate bile-acid enterohepatic cycling PK.

    Parameters
    ----------
    drug_name:
        Name of the drug / bile-acid conjugate.
    dose_mg:
        Initial amount loaded into the bile pool (mg).
    f_reabs:
        Fraction of intestinal drug reabsorbed by ASBT (0 ≤ f_reabs < 1).
    k_sec_per_h:
        First-order bile secretion rate constant (h⁻¹).
    k_reabs_per_h:
        First-order intestinal reabsorption rate constant (h⁻¹).
    k_hepatic_per_h:
        First-order hepatic uptake rate constant (h⁻¹).
    cl_L_per_h:
        Systemic clearance (L/h).
    vd_L:
        Volume of distribution (L).
    t_end_h:
        Simulation duration (h).
    dt_h:
        Forward-Euler step size (h).

    Returns
    -------
    BileAcidPKResult
    """
    # Validation
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if cl_L_per_h <= 0:
        raise ValueError("cl_L_per_h must be > 0")
    if vd_L <= 0:
        raise ValueError("vd_L must be > 0")
    if not (0.0 <= f_reabs < 1.0):
        raise ValueError("f_reabs must be in [0, 1)")
    if k_sec_per_h <= 0:
        raise ValueError("k_sec_per_h must be > 0")
    if k_reabs_per_h <= 0:
        raise ValueError("k_reabs_per_h must be > 0")
    if k_hepatic_per_h <= 0:
        raise ValueError("k_hepatic_per_h must be > 0")

    # Initial conditions
    a_bile = dose_mg
    a_int = 0.0
    c_plasma = 0.0

    times: list = []
    cp_list: list = []
    ab_list: list = []
    ai_list: list = []

    n_steps = int(round(t_end_h / dt_h))
    t = 0.0

    for _ in range(n_steps + 1):
        times.append(t)
        cp_list.append(c_plasma)
        ab_list.append(a_bile)
        ai_list.append(a_int)

        # Fluxes
        secretion = k_sec_per_h * a_bile
        reabsorption = k_reabs_per_h * f_reabs * a_int
        fecal_loss = k_reabs_per_h * (1.0 - f_reabs) * a_int
        hepatic_uptake = k_hepatic_per_h * c_plasma * vd_L

        # ODEs
        d_bile = hepatic_uptake - secretion + reabsorption
        d_int = secretion - k_reabs_per_h * a_int  # = secretion - reabs - fecal
        d_cp = (
            k_reabs_per_h * (1.0 - f_reabs) * a_int / vd_L
            - (k_hepatic_per_h + cl_L_per_h / vd_L) * c_plasma
        )

        # Guard against unused variable warning (fecal_loss is part of d_int already)
        _ = fecal_loss  # used implicitly in d_int derivation above

        # Forward Euler update (skip on last iteration)
        if _ < n_steps:
            a_bile = max(0.0, a_bile + d_bile * dt_h)
            a_int = max(0.0, a_int + d_int * dt_h)
            c_plasma = max(0.0, c_plasma + d_cp * dt_h)

        t += dt_h

    # Derived metrics
    cmax = max(cp_list)
    tmax = times[cp_list.index(cmax)]

    # Manual trapz AUC
    auc = 0.0
    for i in range(len(times) - 1):
        auc += 0.5 * (cp_list[i] + cp_list[i + 1]) * (times[i + 1] - times[i])

    n_peaks = _local_maxima(cp_list)
    secondary_peak = n_peaks >= 2

    fecal_recovery_pct = (1.0 - f_reabs) * 100.0
    cycling_efficiency = f_reabs

    notes = (
        f"Simulated {t_end_h} h; "
        f"k_sec={k_sec_per_h}/h, k_reabs={k_reabs_per_h}/h, "
        f"k_hepatic={k_hepatic_per_h}/h; "
        f"CL={cl_L_per_h} L/h, Vd={vd_L} L"
    )

    return BileAcidPKResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        f_reabs=f_reabs,
        times_h=times,
        c_plasma_mg_L=cp_list,
        a_bile_pool_mg=ab_list,
        a_intestine_mg=ai_list,
        cmax_plasma_mg_L=cmax,
        tmax_plasma_h=tmax,
        auc_plasma_mg_h_per_L=auc,
        cycling_efficiency=cycling_efficiency,
        secondary_peak_present=secondary_peak,
        fecal_recovery_pct=fecal_recovery_pct,
        notes=notes,
    )
