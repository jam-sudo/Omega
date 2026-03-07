"""Mucus barrier pharmacokinetics — models drug diffusion through intestinal mucus layer."""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["MucusBarrierPKResult", "simulate_mucus_barrier_pk", "compare_mucus_barriers"]


@dataclass
class MucusBarrierPKResult:
    """Result of mucus barrier PK simulation."""

    drug_name: str
    dose_mg: float
    mw: float
    logP: float
    charge: int
    times_h: list
    a_lumen_mg: list
    a_mucus_mg: list
    c_plasma_mg_L: list
    cmax_mg_L: float
    tmax_h: float
    auc_mg_h_per_L: float
    t_half_h: float
    d_eff_cm2_per_s: float
    k_me_per_h: float
    partition_mucus: float
    mucus_delay_h: float
    notes: str


def simulate_mucus_barrier_pk(
    drug_name: str,
    dose_mg: float,
    mw: float = 500.0,
    logP: float = 1.0,
    charge: int = 0,
    mucus_thickness_um: float = 200.0,
    cl_L_per_h: float = 5.0,
    vd_L: float = 50.0,
    t_end_h: float = 8.0,
    dt_h: float = 0.05,
) -> MucusBarrierPKResult:
    """Simulate mucus barrier pharmacokinetics using a 2-layer ODE model.

    Args:
        drug_name: Name of the drug compound.
        dose_mg: Dose in milligrams.
        mw: Molecular weight (g/mol).
        logP: Lipophilicity (log octanol-water partition).
        charge: Formal charge (integer).
        mucus_thickness_um: Mucus layer thickness in micrometers.
        cl_L_per_h: Systemic clearance (L/h).
        vd_L: Volume of distribution (L).
        t_end_h: Simulation end time (hours).
        dt_h: Time step (hours).

    Returns:
        MucusBarrierPKResult with simulation outputs.
    """
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if mw <= 0:
        raise ValueError("mw must be > 0")
    if cl_L_per_h <= 0:
        raise ValueError("cl_L_per_h must be > 0")
    if vd_L <= 0:
        raise ValueError("vd_L must be > 0")
    if mucus_thickness_um <= 0:
        raise ValueError("mucus_thickness_um must be > 0")

    # Physicochemical-based diffusivity
    d_free_cm2_per_s = 4.4e-6 * ((500.0 / mw) ** 0.5)

    # Partition into mucus: charged/hydrophilic drugs partition poorly
    hydrophilic_penalty = max(0.0, -logP) * 0.3
    charge_penalty = abs(charge) * 0.5
    partition_mucus = 1.0 / (1.0 + charge_penalty + hydrophilic_penalty)

    d_eff = d_free_cm2_per_s * partition_mucus

    mucus_thickness_cm = mucus_thickness_um / 1.0e4

    # k_me: mucus transit rate (1/h), from diffusion physics
    k_me_raw = d_eff / (mucus_thickness_cm**2) * 3600.0
    k_me = max(0.1, min(k_me_raw, 20.0))

    # Fixed lumen-to-mucus entry rate
    k_lm = 2.0

    # Epithelial absorption rate from mucus
    ka_epi = 2.0 * partition_mucus

    # Elimination rate
    ke = cl_L_per_h / vd_L

    # Initial conditions
    a_lumen = dose_mg
    a_mucus = 0.0
    c_plasma = 0.0

    times_h: list = []
    a_lumen_list: list = []
    a_mucus_list: list = []
    c_plasma_list: list = []

    t = 0.0
    n_steps = int(t_end_h / dt_h) + 1

    # Track mucus peak for delay calculation
    mucus_peak_t = 0.0
    mucus_peak_val = 0.0

    for _ in range(n_steps):
        times_h.append(t)
        a_lumen_list.append(a_lumen)
        a_mucus_list.append(a_mucus)
        c_plasma_list.append(c_plasma)

        # Track mucus peak
        if a_mucus > mucus_peak_val:
            mucus_peak_val = a_mucus
            mucus_peak_t = t

        # Forward Euler ODEs
        da_lumen = -k_lm * a_lumen
        da_mucus = k_lm * a_lumen - k_me * a_mucus
        dc_plasma = ka_epi * a_mucus / vd_L - ke * c_plasma

        a_lumen = max(0.0, a_lumen + da_lumen * dt_h)
        a_mucus = max(0.0, a_mucus + da_mucus * dt_h)
        c_plasma = max(0.0, c_plasma + dc_plasma * dt_h)

        t += dt_h

    # PK metrics
    cmax_mg_L = max(c_plasma_list)
    tmax_h = times_h[c_plasma_list.index(cmax_mg_L)]

    # Trapezoidal AUC
    auc = 0.0
    for i in range(1, len(times_h)):
        auc += 0.5 * (c_plasma_list[i - 1] + c_plasma_list[i]) * (times_h[i] - times_h[i - 1])

    # Terminal half-life from CL/Vd
    t_half_h = 0.693 / ke

    # Mucus delay: time from mucus peak to plasma tmax
    mucus_delay_h = max(0.0, tmax_h - mucus_peak_t)

    notes_parts = []
    if partition_mucus < 0.5:
        notes_parts.append("Low mucus partition due to charge/hydrophilicity")
    if mucus_thickness_um > 300:
        notes_parts.append("Thick mucus layer may significantly reduce absorption")
    if k_me < 1.0:
        notes_parts.append("Slow mucus transit rate")
    if not notes_parts:
        notes_parts.append("Standard mucus barrier conditions")
    notes = "; ".join(notes_parts)

    return MucusBarrierPKResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        mw=mw,
        logP=logP,
        charge=charge,
        times_h=times_h,
        a_lumen_mg=a_lumen_list,
        a_mucus_mg=a_mucus_list,
        c_plasma_mg_L=c_plasma_list,
        cmax_mg_L=cmax_mg_L,
        tmax_h=tmax_h,
        auc_mg_h_per_L=auc,
        t_half_h=t_half_h,
        d_eff_cm2_per_s=d_eff,
        k_me_per_h=k_me,
        partition_mucus=partition_mucus,
        mucus_delay_h=mucus_delay_h,
        notes=notes,
    )


def compare_mucus_barriers(
    drug_name: str,
    dose_mg: float,
    scenarios: list,
) -> list:
    """Compare drug PK across different mucus barrier scenarios.

    Args:
        drug_name: Name of the drug compound.
        dose_mg: Dose in milligrams.
        scenarios: List of dicts with keys: label, mucus_thickness_um, charge.
            Optional keys: mw, logP, cl_L_per_h, vd_L, t_end_h, dt_h.

    Returns:
        List of MucusBarrierPKResult, one per scenario.
    """
    results = []
    for scenario in scenarios:
        result = simulate_mucus_barrier_pk(
            drug_name=scenario.get("label", drug_name),
            dose_mg=dose_mg,
            mw=scenario.get("mw", 500.0),
            logP=scenario.get("logP", 1.0),
            charge=scenario.get("charge", 0),
            mucus_thickness_um=scenario.get("mucus_thickness_um", 200.0),
            cl_L_per_h=scenario.get("cl_L_per_h", 5.0),
            vd_L=scenario.get("vd_L", 50.0),
            t_end_h=scenario.get("t_end_h", 8.0),
            dt_h=scenario.get("dt_h", 0.05),
        )
        results.append(result)
    return results
