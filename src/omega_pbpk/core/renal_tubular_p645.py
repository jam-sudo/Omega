"""Phase 645 — Renal Tubular Secretion Model.

Models active renal tubular secretion (OAT/OCT transporters) and passive
glomerular filtration for drug urinary excretion via Forward Euler ODE.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RenalTubularResult:
    """Result from renal tubular secretion simulation."""

    drug_name: str
    dose_mg: float
    times_h: list
    c_plasma_mg_L: list
    c_urine_mg_mL: list
    cumulative_excreted_mg: list
    gfr_cl_L_per_h: float
    tubular_secretion_cl_L_per_h: float
    total_renal_cl_L_per_h: float
    fe_urine_pct: float
    t_half_h: float
    cmax_plasma_mg_L: float
    auc_plasma_mg_h_per_L: float
    secretion_class: str
    notes: str


_KA_PER_H = 1.2
_F_ORAL = 0.85
_URINE_FLOW_ML_PER_H = 60.0


def _validate_inputs(
    dose_mg: float,
    fup: float,
    gfr_mL_per_min: float,
    cl_sys_L_per_h: float,
    vd_sys_L: float,
) -> None:
    if dose_mg <= 0.0:
        raise ValueError(f"dose_mg must be > 0, got {dose_mg}")
    if not (0.0 < fup <= 1.0):
        raise ValueError(f"fup must be in (0, 1], got {fup}")
    if gfr_mL_per_min <= 0.0:
        raise ValueError(f"gfr_mL_per_min must be > 0, got {gfr_mL_per_min}")
    if cl_sys_L_per_h <= 0.0:
        raise ValueError(f"cl_sys_L_per_h must be > 0, got {cl_sys_L_per_h}")
    if vd_sys_L <= 0.0:
        raise ValueError(f"vd_sys_L must be > 0, got {vd_sys_L}")


def simulate_renal_tubular(
    drug_name: str,
    dose_mg: float,
    fup: float,
    gfr_mL_per_min: float,
    cl_sys_L_per_h: float,
    vd_sys_L: float,
    oat_vmax_mg_h: float | None = None,
    oat_km_mg_L: float = 1.0,
    t_end_h: float = 24.0,
    dt_h: float = 0.1,
) -> RenalTubularResult:
    """Simulate renal tubular secretion and glomerular filtration.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    dose_mg:
        Oral dose (mg).
    fup:
        Fraction unbound in plasma (0, 1].
    gfr_mL_per_min:
        Glomerular filtration rate (mL/min).
    cl_sys_L_per_h:
        Total systemic clearance (L/h).
    vd_sys_L:
        Volume of distribution (L).
    oat_vmax_mg_h:
        Vmax for OAT/OCT transporter secretion (mg/h). None = no secretion.
    oat_km_mg_L:
        Km for OAT/OCT transporter (mg/L).
    t_end_h:
        Simulation duration (h).
    dt_h:
        Forward Euler step size (h).

    Returns
    -------
    RenalTubularResult
    """
    _validate_inputs(dose_mg, fup, gfr_mL_per_min, cl_sys_L_per_h, vd_sys_L)

    # GFR-based filtration clearance (L/h)
    cl_filt = gfr_mL_per_min * 60.0 / 1000.0 * fup

    # Tubular secretion parameters
    has_secretion = oat_vmax_mg_h is not None and oat_vmax_mg_h > 0.0
    vmax = oat_vmax_mg_h if has_secretion else 0.0

    # Intrinsic secretion clearance (L/h) for reporting
    cl_sec_max = vmax / oat_km_mg_L if has_secretion else 0.0

    # Total renal clearance (steady-state approx)
    total_renal_cl = cl_filt + cl_sec_max

    # Non-renal elimination rate constant
    cl_nonrenal = max(0.0, cl_sys_L_per_h - cl_filt)
    k_el_nonrenal = cl_nonrenal / vd_sys_L

    # Overall elimination rate constant (for t_half)
    k_el = cl_sys_L_per_h / vd_sys_L
    t_half = 0.693 / k_el

    # State variables
    a_gut = dose_mg * _F_ORAL
    c_plasma = 0.0
    m_excreted = 0.0

    n_steps = max(1, int(round(t_end_h / dt_h)))

    times_h: list[float] = []
    c_plasma_mg_L: list[float] = []
    c_urine_mg_mL: list[float] = []
    excretion_rate_mg_h: list[float] = []

    for i in range(n_steps + 1):
        t = i * dt_h

        # Secretion clearance at current concentration
        if has_secretion and c_plasma > 0.0:
            cl_sec_rate = vmax / (oat_km_mg_L + c_plasma)
        else:
            cl_sec_rate = 0.0

        # Excretion rate (mg/h)
        exc_rate = (cl_filt + cl_sec_rate) * max(0.0, c_plasma)

        # Urine concentration (mg/mL)
        c_urine = exc_rate / _URINE_FLOW_ML_PER_H if _URINE_FLOW_ML_PER_H > 0 else 0.0

        times_h.append(t)
        c_plasma_mg_L.append(max(0.0, c_plasma))
        c_urine_mg_mL.append(max(0.0, c_urine))
        excretion_rate_mg_h.append(max(0.0, exc_rate))

        if i == n_steps:
            break

        # Forward Euler
        a_gut_curr = max(0.0, a_gut)
        c_plasma_curr = max(0.0, c_plasma)

        d_a_gut = -_KA_PER_H * a_gut_curr
        d_c_plasma = (
            _KA_PER_H * a_gut_curr / vd_sys_L
            - k_el_nonrenal * c_plasma_curr
            - cl_filt / vd_sys_L * c_plasma_curr
            - cl_sec_rate / vd_sys_L * c_plasma_curr
        )
        d_m_excreted = exc_rate

        a_gut += d_a_gut * dt_h
        c_plasma += d_c_plasma * dt_h
        m_excreted += d_m_excreted * dt_h

        a_gut = max(0.0, a_gut)
        c_plasma = max(0.0, c_plasma)

    # Cumulative excreted via trapezoidal integration of excretion rate
    cumulative_excreted_mg: list[float] = [0.0]
    for i in range(1, len(times_h)):
        trap = (
            0.5
            * (excretion_rate_mg_h[i] + excretion_rate_mg_h[i - 1])
            * (times_h[i] - times_h[i - 1])
        )
        cumulative_excreted_mg.append(cumulative_excreted_mg[-1] + trap)

    # PK metrics
    cmax = max(c_plasma_mg_L)
    auc = sum(
        0.5 * (c_plasma_mg_L[i] + c_plasma_mg_L[i - 1]) * (times_h[i] - times_h[i - 1])
        for i in range(1, len(times_h))
    )

    fe_urine = cumulative_excreted_mg[-1] / dose_mg * 100.0 if dose_mg > 0 else 0.0
    fe_urine = max(0.0, min(100.0, fe_urine))

    # Secretion class
    if cl_sec_max > 1.0:
        secretion_class = "high"
    elif cl_sec_max > 0.1:
        secretion_class = "moderate"
    else:
        secretion_class = "low"

    notes = (
        f"GFR={gfr_mL_per_min:.1f} mL/min; fup={fup:.3f}; "
        f"CL_filt={cl_filt:.3f} L/h; CL_sec_max={cl_sec_max:.3f} L/h; "
        f"ka={_KA_PER_H}/h; F_oral={_F_ORAL}"
    )

    return RenalTubularResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        times_h=times_h,
        c_plasma_mg_L=c_plasma_mg_L,
        c_urine_mg_mL=c_urine_mg_mL,
        cumulative_excreted_mg=cumulative_excreted_mg,
        gfr_cl_L_per_h=cl_filt,
        tubular_secretion_cl_L_per_h=cl_sec_max,
        total_renal_cl_L_per_h=total_renal_cl,
        fe_urine_pct=fe_urine,
        t_half_h=t_half,
        cmax_plasma_mg_L=cmax,
        auc_plasma_mg_h_per_L=auc,
        secretion_class=secretion_class,
        notes=notes,
    )
