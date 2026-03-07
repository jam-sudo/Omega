"""
Phase 732 — Transdermal Iontophoresis PK
Models current-enhanced transdermal drug delivery (iontophoresis) using a
2-compartment model (skin depot + systemic plasma) with on/off current schedule.
"""

from __future__ import annotations

from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Result dataclass (plain — has list fields)
# ---------------------------------------------------------------------------


@dataclass
class IontophoresisPKResult:
    """Result of iontophoresis PK simulation."""

    drug_name: str
    dose_mg: float
    times_h: list
    c_plasma_mg_L: list
    a_skin_mg: list
    cmax_mg_L: float
    tmax_h: float
    auc_mg_h_per_L: float
    enhancement_factor: float
    auc_passive_estimate: float
    auc_enhancement_ratio: float
    f_absorbed: float
    notes: str


# ---------------------------------------------------------------------------
# Core simulation
# ---------------------------------------------------------------------------


def simulate_iontophoresis_pk(
    drug_name: str,
    dose_mg: float,
    ka_passive_per_h: float = 0.02,
    iono_factor: float = 0.1,
    current_ma: float = 1.0,
    current_on_h: float = 8.0,
    cl_L_per_h: float = 5.0,
    vd_L: float = 50.0,
    t_end_h: float = 24.0,
    dt_h: float = 0.05,
) -> IontophoresisPKResult:
    """
    Simulate 2-compartment iontophoresis PK.

    ODEs (Forward Euler):
      dA_skin/dt = -k_total(t) * A_skin
      dC_plasma/dt = k_total(t) * A_skin / Vd - ke * C_plasma

    Enhancement:
      k_total = ka_passive * (1 + iono_factor * current_mA)  when current on
      k_total = ka_passive                                     when current off
    """
    # --- Validation ---
    if dose_mg <= 0:
        raise ValueError("dose_mg must be positive")
    if ka_passive_per_h <= 0:
        raise ValueError("ka_passive_per_h must be positive")
    if current_ma < 0:
        raise ValueError("current_ma must be non-negative")
    if cl_L_per_h <= 0:
        raise ValueError("cl_L_per_h must be positive")
    if vd_L <= 0:
        raise ValueError("vd_L must be positive")

    ke = cl_L_per_h / vd_L  # elimination rate constant (1/h)
    enhancement_factor = 1.0 + iono_factor * current_ma

    # --- Initial conditions ---
    a_skin = dose_mg  # all drug in skin depot
    c_plasma = 0.0

    times_h: list = []
    c_plasma_list: list = []
    a_skin_list: list = []

    n_steps = int(t_end_h / dt_h) + 1
    t = 0.0

    for _step in range(n_steps):
        times_h.append(t)
        c_plasma_list.append(c_plasma)
        a_skin_list.append(a_skin)

        # Determine active absorption rate
        if t < current_on_h:
            k_total = ka_passive_per_h * enhancement_factor
        else:
            k_total = ka_passive_per_h

        # Forward Euler
        da_skin = -k_total * a_skin
        dc_plasma = k_total * a_skin / vd_L - ke * c_plasma

        a_skin = max(0.0, a_skin + da_skin * dt_h)
        c_plasma = max(0.0, c_plasma + dc_plasma * dt_h)
        t = round(t + dt_h, 10)

    # --- Derived metrics ---
    cmax_mg_L = max(c_plasma_list)
    tmax_h = times_h[c_plasma_list.index(cmax_mg_L)]
    auc_mg_h_per_L = _trapz(times_h, c_plasma_list)

    # Fraction absorbed: drug that left skin depot
    a_skin_final = a_skin_list[-1]
    f_absorbed = 1.0 - a_skin_final / dose_mg if dose_mg > 0 else 0.0
    f_absorbed = max(0.0, min(1.0, f_absorbed))

    # AUC passive estimate (ka_passive << ke approximation):
    # AUC = dose * ka_passive / (Vd * ke)   [for flip-flop when ka<<ke]
    # More generally: dose * ka / (Vd * (ke - ka)) for ka != ke; use simpler form
    if abs(ke - ka_passive_per_h) > 1e-9:
        auc_passive_estimate = dose_mg * ka_passive_per_h / (vd_L * (ke - ka_passive_per_h))
    else:
        # ke == ka_passive; use dose/(Vd*ke) as limit
        auc_passive_estimate = dose_mg / (vd_L * ke)

    # Ensure positive
    if auc_passive_estimate <= 0:
        auc_passive_estimate = dose_mg * ka_passive_per_h / (vd_L * ke)

    auc_enhancement_ratio = (
        auc_mg_h_per_L / auc_passive_estimate if auc_passive_estimate > 0 else 1.0
    )

    notes = (
        f"Iontophoresis PK for {drug_name}. "
        f"Enhancement factor={enhancement_factor:.2f} (current={current_ma} mA). "
        f"Current applied for {current_on_h}h of {t_end_h}h total. "
        f"f_absorbed={f_absorbed:.3f}."
    )

    return IontophoresisPKResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        times_h=times_h,
        c_plasma_mg_L=c_plasma_list,
        a_skin_mg=a_skin_list,
        cmax_mg_L=cmax_mg_L,
        tmax_h=tmax_h,
        auc_mg_h_per_L=auc_mg_h_per_L,
        enhancement_factor=enhancement_factor,
        auc_passive_estimate=auc_passive_estimate,
        auc_enhancement_ratio=auc_enhancement_ratio,
        f_absorbed=f_absorbed,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Comparison helper
# ---------------------------------------------------------------------------


def compare_current_levels(
    drug_name: str,
    dose_mg: float,
    currents_ma: list,
    ka_passive_per_h: float = 0.02,
    iono_factor: float = 0.1,
    current_on_h: float = 8.0,
    cl_L_per_h: float = 5.0,
    vd_L: float = 50.0,
    t_end_h: float = 24.0,
    dt_h: float = 0.05,
) -> list:
    """
    Compare iontophoresis PK across different applied current levels.

    Returns list of IontophoresisPKResult, one per current level.
    """
    results = []
    for current_ma in currents_ma:
        result = simulate_iontophoresis_pk(
            drug_name=drug_name,
            dose_mg=dose_mg,
            ka_passive_per_h=ka_passive_per_h,
            iono_factor=iono_factor,
            current_ma=current_ma,
            current_on_h=current_on_h,
            cl_L_per_h=cl_L_per_h,
            vd_L=vd_L,
            t_end_h=t_end_h,
            dt_h=dt_h,
        )
        results.append(result)
    return results


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _trapz(x: list, y: list) -> float:
    """Manual trapezoidal integration."""
    total = 0.0
    for i in range(1, len(x)):
        total += 0.5 * (y[i - 1] + y[i]) * (x[i] - x[i - 1])
    return total
