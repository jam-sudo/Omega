"""
Phase 928 — Nanosuspension PK Model

Nanosuspension dissolution and oral absorption PK model — particle size-dependent
dissolution driving absorption using Noyes-Whitney model.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["NanosuspensionPKResult", "simulate_nanosuspension", "compare_particle_sizes"]

# Base dissolution constant for reference particle radius 1.0 µm
# Value chosen so that nanosuspension particles (~0.2-0.5 µm) dissolve rapidly,
# while conventional particles (~50 µm) dissolve slowly, enabling linear dose-response.
_KDISS_BASE = 200.0


@dataclass
class NanosuspensionPKResult:
    drug_name: str
    particle_radius_um: float
    dose_mg: float
    times_h: list
    c_plasma_mg_L: list
    m_undissolved_mg: list
    m_dissolved_gut_mg: list
    cmax_mg_L: float
    tmax_h: float
    auc_mg_h_per_L: float
    fraction_dissolved_at_2h: float
    fraction_absorbed: float
    dissolution_rate_constant: float
    notes: str


def simulate_nanosuspension(
    drug_name: str,
    dose_mg: float,
    particle_radius_um: float = 0.5,
    cs_mg_mL: float = 0.1,
    vgut_L: float = 0.25,
    ka_per_h: float = 1.0,
    cl_L_per_h: float = 10.0,
    vd_L: float = 50.0,
    t_end_h: float = 24.0,
    dt_h: float = 0.1,
) -> NanosuspensionPKResult:
    """
    Simulate nanosuspension dissolution and oral absorption PK.

    State variables:
    - M_undissolved (mg): undissolved solid remaining
    - M_gut (mg): dissolved drug mass in gut fluid
    - C_plasma (mg/L): plasma drug concentration

    Dissolution (Noyes-Whitney with shrinking sphere):
      dissolution_rate (mg/h) = kdiss * Cs * (M_undissolved/M0)^(2/3) * M0^(1/3)

    The driving force uses saturation solubility Cs (mg/mL) as the concentration gradient
    (thin-film approximation: drug is rapidly absorbed so gut concentration << Cs).
    This yields a linear model where doubling dose doubles Cmax proportionally.

    Gut:
      dM_gut/dt = dissolution_rate - ka * M_gut

    Plasma:
      dC_plasma/dt = ka * M_gut / vd_L - ke * C_plasma
    """
    # Validation
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if particle_radius_um <= 0:
        raise ValueError("particle_radius_um must be > 0")
    if cs_mg_mL <= 0:
        raise ValueError("cs_mg_mL must be > 0")
    if cl_L_per_h <= 0:
        raise ValueError("cl_L_per_h must be > 0")
    if vd_L <= 0:
        raise ValueError("vd_L must be > 0")
    if vgut_L <= 0:
        raise ValueError("vgut_L must be > 0")

    # Derived parameters
    kdiss = (
        _KDISS_BASE / particle_radius_um
    )  # smaller particle → faster dissolution (h^-1 * mg^(-1/3))
    ke = cl_L_per_h / vd_L

    # Initial conditions
    M_undissolved = dose_mg  # mg, undissolved solid
    M_gut = 0.0  # mg, dissolved drug mass in gut
    C_plasma = 0.0  # mg/L, plasma concentration

    M0 = dose_mg  # reference mass for area scaling (mg)
    # M0^(1/3) factor to make kdiss * Cs dimensionally consistent → mg/h
    M0_cbrt = M0 ** (1.0 / 3.0)

    times_h = []
    c_plasma_mg_L_list = []
    m_undissolved_mg_list = []
    m_dissolved_gut_mg_list = []

    t = 0.0
    fraction_dissolved_at_2h = 0.0
    _found_2h = False

    while t <= t_end_h + 1e-9:
        times_h.append(t)
        c_plasma_mg_L_list.append(C_plasma)
        m_undissolved_mg_list.append(M_undissolved)
        m_dissolved_gut_mg_list.append(M_gut)

        # Record fraction dissolved at ~2h
        if not _found_2h and t >= 2.0 - dt_h / 2.0:
            dissolved_so_far = dose_mg - M_undissolved
            fraction_dissolved_at_2h = min(1.0, max(0.0, dissolved_so_far / dose_mg))
            _found_2h = True

        # Shrinking sphere area factor: (M_undissolved/M0)^(2/3)
        if M_undissolved > 1e-12 and M0 > 1e-12:
            area_factor = (M_undissolved / M0) ** (2.0 / 3.0)
        else:
            area_factor = 0.0

        # Dissolution rate (mg/h):
        #   kdiss (/h / µm) * cs_mg_mL (mg/mL) * area_factor (dimensionless) * M0^(1/3) (mg^(1/3))
        # This gives mg * mg_mL^(-1) * mg^(1/3) / (h * µm) — needs unit adjustment factor.
        # In practice: treat kdiss * cs_mg_mL as an effective first-order rate with units h^-1.
        # The total dissolution rate at M_undissolved=M0 is kdiss*cs_mg_mL*M0.
        # For shrinking sphere: kdiss * cs_mg_mL * M_undissolved^(2/3) * M0^(1/3)
        dissolution_rate_mg_per_h = kdiss * cs_mg_mL * area_factor * M0_cbrt

        # Forward Euler
        dM_undissolved = -dissolution_rate_mg_per_h
        dM_gut = dissolution_rate_mg_per_h - ka_per_h * M_gut
        dC_plasma = ka_per_h * M_gut / vd_L - ke * C_plasma

        M_undissolved = max(0.0, M_undissolved + dM_undissolved * dt_h)
        M_gut = max(0.0, M_gut + dM_gut * dt_h)
        C_plasma = max(0.0, C_plasma + dC_plasma * dt_h)

        t = round(t + dt_h, 10)

    # If 2h not captured (e.g., t_end_h < 2h)
    if not _found_2h:
        dissolved_so_far = dose_mg - M_undissolved
        fraction_dissolved_at_2h = min(1.0, max(0.0, dissolved_so_far / dose_mg))

    # Compute metrics
    cmax = max(c_plasma_mg_L_list) if c_plasma_mg_L_list else 0.0
    tmax = times_h[c_plasma_mg_L_list.index(cmax)] if cmax > 0 else 0.0

    # Trapezoidal AUC
    auc = 0.0
    for i in range(1, len(times_h)):
        dt = times_h[i] - times_h[i - 1]
        auc += 0.5 * (c_plasma_mg_L_list[i - 1] + c_plasma_mg_L_list[i]) * dt

    # Fraction absorbed: amount that reached plasma = AUC * CL / dose_mg
    amount_absorbed_mg = auc * cl_L_per_h
    fraction_absorbed = min(1.0, max(0.0, amount_absorbed_mg / dose_mg))

    notes = (
        f"Nanosuspension PK for {drug_name}. "
        f"Particle radius={particle_radius_um} µm, kdiss={kdiss:.3f}/h. "
        f"Fraction dissolved at 2h={fraction_dissolved_at_2h:.3f}, "
        f"Fraction absorbed={fraction_absorbed:.3f}."
    )

    return NanosuspensionPKResult(
        drug_name=drug_name,
        particle_radius_um=particle_radius_um,
        dose_mg=dose_mg,
        times_h=times_h,
        c_plasma_mg_L=c_plasma_mg_L_list,
        m_undissolved_mg=m_undissolved_mg_list,
        m_dissolved_gut_mg=m_dissolved_gut_mg_list,
        cmax_mg_L=cmax,
        tmax_h=tmax,
        auc_mg_h_per_L=auc,
        fraction_dissolved_at_2h=fraction_dissolved_at_2h,
        fraction_absorbed=fraction_absorbed,
        dissolution_rate_constant=kdiss,
        notes=notes,
    )


def compare_particle_sizes(
    drug_name: str,
    dose_mg: float,
    radii_um: list[float] | None = None,
    cs_mg_mL: float = 0.1,
    vgut_L: float = 0.25,
    ka_per_h: float = 1.0,
    cl_L_per_h: float = 10.0,
    vd_L: float = 50.0,
    t_end_h: float = 24.0,
    dt_h: float = 0.1,
) -> list[NanosuspensionPKResult]:
    """
    Compare PK across multiple particle sizes.
    Returns list of NanosuspensionPKResult sorted by AUC descending.

    Default radii: [0.2, 0.5, 5.0, 50.0] µm.
    """
    if radii_um is None:
        radii_um = [0.2, 0.5, 5.0, 50.0]

    results = []
    for r in radii_um:
        res = simulate_nanosuspension(
            drug_name=drug_name,
            dose_mg=dose_mg,
            particle_radius_um=r,
            cs_mg_mL=cs_mg_mL,
            vgut_L=vgut_L,
            ka_per_h=ka_per_h,
            cl_L_per_h=cl_L_per_h,
            vd_L=vd_L,
            t_end_h=t_end_h,
            dt_h=dt_h,
        )
        results.append(res)

    # Sort by AUC descending (smallest particle → highest AUC)
    results.sort(key=lambda x: x.auc_mg_h_per_L, reverse=True)
    return results
