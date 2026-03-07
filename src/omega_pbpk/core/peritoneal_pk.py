"""Peritoneal drug absorption and dialysis pharmacokinetics (Phase 711).

Models intraperitoneal (IP) administration, which bypasses hepatic first-pass via
lymphatic pathway. Also supports peritoneal dialysis clearance for renally-impaired
patients.
"""

from __future__ import annotations

import math as _math
from dataclasses import dataclass, field
from dataclasses import dataclass as _dataclass

__all__ = [
    "PeritonealPKResult",
    "simulate_peritoneal_pk",
    "compare_routes_ip_iv",
    # Phase 775 additions
    "IPAbsorptionResult",
    "simulate_ip_absorption",
    "compare_ip_vs_iv",
]


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class PeritonealPKResult:
    """Result from peritoneal PK simulation."""

    drug_name: str
    dose_mg: float
    route: str
    dialysis: bool
    times_h: list = field(default_factory=list)
    a_ip_mg: list = field(default_factory=list)
    c_plasma_mg_L: list = field(default_factory=list)
    cmax_mg_L: float = 0.0
    tmax_h: float = 0.0
    auc_mg_h_per_L: float = 0.0
    t_half_h: float = 0.0
    f_systemic_effective: float = 0.0
    cl_dialysis_L_per_h: float = 0.0
    notes: str = ""


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _validate_inputs(
    dose_mg: float,
    f_bypass: float,
    fh: float,
    cl_L_per_h: float,
    vd_L: float,
    fup: float,
) -> None:
    if dose_mg <= 0:
        raise ValueError(f"dose_mg must be > 0, got {dose_mg}")
    if not (0 < f_bypass < 1):
        raise ValueError(f"f_bypass must be in (0, 1), got {f_bypass}")
    if not (0 < fh <= 1):
        raise ValueError(f"fh must be in (0, 1], got {fh}")
    if cl_L_per_h <= 0:
        raise ValueError(f"cl_L_per_h must be > 0, got {cl_L_per_h}")
    if vd_L <= 0:
        raise ValueError(f"vd_L must be > 0, got {vd_L}")
    if not (0 < fup <= 1):
        raise ValueError(f"fup must be in (0, 1], got {fup}")


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------


def simulate_peritoneal_pk(
    drug_name: str,
    dose_mg: float,
    ka_ip_per_h: float = 0.5,
    f_bypass: float = 0.5,
    fh: float = 0.8,
    cl_L_per_h: float = 5.0,
    vd_L: float = 50.0,
    dialysis: bool = False,
    fup: float = 0.1,
    dialysate_flow_mL_per_min: float = 2000.0,
    t_end_h: float = 24.0,
    dt_h: float = 0.05,
) -> PeritonealPKResult:
    """Simulate peritoneal (IP) drug pharmacokinetics using forward Euler ODE.

    Model:
        dA_ip/dt = -ka_ip * A_ip
        dC_plasma/dt = ka_ip * A_ip * f_effective / Vd - ke * C_plasma

    where f_effective = f_bypass + (1 - f_bypass) * fh
    and ke = (CL_sys + CL_dial) / Vd.
    """
    _validate_inputs(dose_mg, f_bypass, fh, cl_L_per_h, vd_L, fup)

    # Effective systemic bioavailability:
    # lymphatic fraction (f_bypass) skips liver, portal fraction undergoes hepatic
    # extraction (fh = hepatic availability = 1 - ER)
    f_effective = f_bypass + (1.0 - f_bypass) * fh

    # Peritoneal dialysis clearance (L/h)
    cl_dialysis = 0.0
    if dialysis:
        cl_dialysis = dialysate_flow_mL_per_min * 60.0 * fup / 1000.0

    cl_total = cl_L_per_h + cl_dialysis
    ke = cl_total / vd_L  # 1/h

    # Initial conditions
    a_ip = dose_mg  # mg in peritoneal cavity
    c_plasma = 0.0  # mg/L

    n_steps = int(t_end_h / dt_h) + 1
    times_h: list[float] = [0.0]
    a_ip_mg: list[float] = [a_ip]
    c_plasma_mg_L: list[float] = [c_plasma]

    for i in range(1, n_steps):
        # Forward Euler
        da_ip_dt = -ka_ip_per_h * a_ip
        dc_plasma_dt = (ka_ip_per_h * a_ip * f_effective / vd_L) - ke * c_plasma

        a_ip = a_ip + da_ip_dt * dt_h
        c_plasma = c_plasma + dc_plasma_dt * dt_h

        # Clamp negatives
        if a_ip < 0.0:
            a_ip = 0.0
        if c_plasma < 0.0:
            c_plasma = 0.0

        times_h.append(i * dt_h)
        a_ip_mg.append(a_ip)
        c_plasma_mg_L.append(c_plasma)

    # PK metrics
    cmax_mg_L = max(c_plasma_mg_L)
    tmax_idx = c_plasma_mg_L.index(cmax_mg_L)
    tmax_h = times_h[tmax_idx]

    # Trapezoidal AUC
    auc = 0.0
    for i in range(1, len(times_h)):
        auc += 0.5 * (c_plasma_mg_L[i] + c_plasma_mg_L[i - 1]) * (times_h[i] - times_h[i - 1])

    t_half_h = 0.693147 / ke if ke > 0.0 else float("inf")

    notes_parts = [
        f"IP administration with {f_effective:.1%} effective systemic bioavailability.",
        f"Hepatic extraction ratio: {1.0 - fh:.2f} (fh={fh}).",
    ]
    if dialysis:
        notes_parts.append(
            f"Peritoneal dialysis CL: {cl_dialysis:.2f} L/h "
            f"(dialysate flow {dialysate_flow_mL_per_min:.0f} mL/min, fup={fup})."
        )

    return PeritonealPKResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        route="ip",
        dialysis=dialysis,
        times_h=times_h,
        a_ip_mg=a_ip_mg,
        c_plasma_mg_L=c_plasma_mg_L,
        cmax_mg_L=cmax_mg_L,
        tmax_h=tmax_h,
        auc_mg_h_per_L=auc,
        t_half_h=t_half_h,
        f_systemic_effective=f_effective,
        cl_dialysis_L_per_h=cl_dialysis,
        notes=" ".join(notes_parts),
    )


# ---------------------------------------------------------------------------
# Route comparison
# ---------------------------------------------------------------------------


def compare_routes_ip_iv(
    drug_name: str,
    dose_mg: float,
    fh: float = 0.8,
    cl_L_per_h: float = 5.0,
    vd_L: float = 50.0,
) -> dict:
    """Compare IP vs IV administration for the same drug and dose.

    Returns a dict with:
        ip_result: PeritonealPKResult
        iv_result: dict with cmax_mg_L, auc_mg_h_per_L, t_half_h
        auc_ratio: float (IP AUC / IV AUC)
        notes: str
    """
    ip_result = simulate_peritoneal_pk(
        drug_name=drug_name,
        dose_mg=dose_mg,
        fh=fh,
        cl_L_per_h=cl_L_per_h,
        vd_L=vd_L,
    )

    # IV 1-compartment bolus: analytical solution
    ke_iv = cl_L_per_h / vd_L
    c0_iv = dose_mg / vd_L  # mg/L at t=0
    t_half_iv = 0.693147 / ke_iv if ke_iv > 0.0 else float("inf")
    auc_iv = dose_mg / cl_L_per_h  # theoretical AUC = Dose/CL

    iv_result = {
        "cmax_mg_L": c0_iv,
        "auc_mg_h_per_L": auc_iv,
        "t_half_h": t_half_iv,
    }

    auc_ratio = ip_result.auc_mg_h_per_L / auc_iv if auc_iv > 0.0 else 0.0

    notes = (
        f"IP effective bioavailability: {ip_result.f_systemic_effective:.1%}. "
        f"AUC ratio IP/IV: {auc_ratio:.3f}. "
        f"IV bolus Cmax: {c0_iv:.2f} mg/L vs IP Cmax: {ip_result.cmax_mg_L:.2f} mg/L."
    )

    return {
        "ip_result": ip_result,
        "iv_result": iv_result,
        "auc_ratio": auc_ratio,
        "notes": notes,
    }


# ---------------------------------------------------------------------------
# Phase 775 — bidirectional peritoneal PK with lymphatic drain
# ---------------------------------------------------------------------------


@_dataclass(frozen=True)
class IPAbsorptionResult:
    """Result of intraperitoneal absorption simulation (Phase 775).

    Attributes
    ----------
    drug_name : str
    dose_mg : float
    ka_peritoneal_per_h : float
    f_lymph : float
    times_h : list  — simulation time points
    a_peritoneal_mg : list  — peritoneal drug amount (mg)
    c_plasma_mg_L : list  — plasma concentration (mg/L)
    cmax_mg_L : float
    tmax_h : float
    auc_mg_h_per_L : float
    f_absorbed : float  — fraction absorbed into plasma
    t_half_h : float
    notes : str
    """

    drug_name: str
    dose_mg: float
    ka_peritoneal_per_h: float
    f_lymph: float
    times_h: list
    a_peritoneal_mg: list
    c_plasma_mg_L: list
    cmax_mg_L: float
    tmax_h: float
    auc_mg_h_per_L: float
    f_absorbed: float
    t_half_h: float
    notes: str


def _trap_auc(times: list, conc: list) -> float:
    """Manual trapezoidal AUC."""
    auc = 0.0
    for i in range(len(times) - 1):
        auc += 0.5 * (conc[i] + conc[i + 1]) * (times[i + 1] - times[i])
    return auc


def simulate_ip_absorption(
    drug_name: str,
    dose_mg: float,
    ka_peritoneal_per_h: float = 0.3,
    cl_L_per_h: float = 5.0,
    vd_L: float = 50.0,
    f_lymph: float = 0.05,
    t_end_h: float = 12.0,
    dt_h: float = 0.05,
) -> IPAbsorptionResult:
    """Simulate peritoneal/intraperitoneal absorption PK.

    2-compartment model: peritoneal cavity + plasma.
    First-order absorption from peritoneum, first-order elimination from plasma.
    f_lymph fraction drains to lymphatics (does not reach plasma).

    Parameters
    ----------
    drug_name : Drug name.
    dose_mg : IP dose (mg).
    ka_peritoneal_per_h : Peritoneal absorption rate constant (h^-1).
    cl_L_per_h : Systemic clearance (L/h).
    vd_L : Volume of distribution (L).
    f_lymph : Fraction of dose absorbed via lymphatics (default 0.05).
    t_end_h : Simulation duration (h).
    dt_h : Forward Euler time step (h).

    Raises
    ------
    ValueError : On invalid parameters.
    """
    if dose_mg <= 0:
        raise ValueError(f"dose_mg must be > 0, got {dose_mg}")
    if ka_peritoneal_per_h <= 0:
        raise ValueError(f"ka_peritoneal_per_h must be > 0, got {ka_peritoneal_per_h}")
    if cl_L_per_h <= 0:
        raise ValueError(f"cl_L_per_h must be > 0, got {cl_L_per_h}")
    if vd_L <= 0:
        raise ValueError(f"vd_L must be > 0, got {vd_L}")
    if not 0.0 <= f_lymph < 1.0:
        raise ValueError(f"f_lymph must be in [0, 1), got {f_lymph}")

    f_plasma = 1.0 - f_lymph
    ke = cl_L_per_h / vd_L

    n_steps = max(int(t_end_h / dt_h), 1)
    times: list = [0.0] * (n_steps + 1)
    a_peri: list = [0.0] * (n_steps + 1)
    c_plas: list = [0.0] * (n_steps + 1)

    a_peri[0] = dose_mg
    a_central = 0.0  # mg in plasma compartment

    for i in range(n_steps):
        times[i + 1] = (i + 1) * dt_h
        abs_rate = ka_peritoneal_per_h * a_peri[i]
        abs_to_plasma = abs_rate * f_plasma

        da_peri = -abs_rate * dt_h
        da_central = (abs_to_plasma - ke * a_central) * dt_h

        a_peri[i + 1] = max(0.0, a_peri[i] + da_peri)
        a_central = max(0.0, a_central + da_central)
        c_plas[i + 1] = a_central / vd_L

    cmax = max(c_plas)
    tmax = times[c_plas.index(cmax)]
    auc = _trap_auc(times, c_plas)

    residual = a_peri[-1] / dose_mg
    f_absorbed = f_plasma * (1.0 - residual)

    t_half = _math.log(2.0) / ke

    notes = (
        f"IP absorption: {drug_name}, dose={dose_mg} mg. "
        f"ka={ka_peritoneal_per_h:.3f}/h, CL={cl_L_per_h:.2f} L/h, Vd={vd_L:.1f} L. "
        f"f_lymph={f_lymph:.2f}, f_absorbed={f_absorbed:.3f}. "
        f"Cmax={cmax:.4f} mg/L at Tmax={tmax:.2f} h. AUC={auc:.3f} mg*h/L. "
        f"t1/2={t_half:.2f} h."
    )

    return IPAbsorptionResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        ka_peritoneal_per_h=ka_peritoneal_per_h,
        f_lymph=f_lymph,
        times_h=times,
        a_peritoneal_mg=a_peri,
        c_plasma_mg_L=c_plas,
        cmax_mg_L=cmax,
        tmax_h=tmax,
        auc_mg_h_per_L=auc,
        f_absorbed=f_absorbed,
        t_half_h=t_half,
        notes=notes,
    )


def compare_ip_vs_iv(
    drug_name: str,
    dose_mg: float,
    cl_L_per_h: float = 5.0,
    vd_L: float = 50.0,
    ka_peritoneal_per_h: float = 0.3,
    f_lymph: float = 0.05,
    t_end_h: float = 12.0,
    dt_h: float = 0.05,
) -> dict:
    """Compare IP absorption vs IV bolus for same drug/dose.

    Returns dict with keys: ip_result, iv_cmax_mg_L, iv_auc_mg_h_per_L,
    ip_auc_mg_h_per_L, ip_cmax_mg_L, auc_ratio_ip_iv, cmax_ratio_ip_iv, notes.
    """
    ip_result = simulate_ip_absorption(
        drug_name=drug_name,
        dose_mg=dose_mg,
        ka_peritoneal_per_h=ka_peritoneal_per_h,
        cl_L_per_h=cl_L_per_h,
        vd_L=vd_L,
        f_lymph=f_lymph,
        t_end_h=t_end_h,
        dt_h=dt_h,
    )

    # IV bolus: forward Euler
    ke = cl_L_per_h / vd_L
    n_steps = max(int(t_end_h / dt_h), 1)
    times_iv = [i * dt_h for i in range(n_steps + 1)]
    c_iv = [0.0] * (n_steps + 1)
    c_iv[0] = dose_mg / vd_L
    for i in range(n_steps):
        c_iv[i + 1] = max(0.0, c_iv[i] - ke * c_iv[i] * dt_h)

    iv_cmax = c_iv[0]
    iv_auc = _trap_auc(times_iv, c_iv)

    auc_ratio = ip_result.auc_mg_h_per_L / iv_auc if iv_auc > 0 else 0.0
    cmax_ratio = ip_result.cmax_mg_L / iv_cmax if iv_cmax > 0 else 0.0

    notes = (
        f"IP vs IV: {drug_name} ({dose_mg} mg). "
        f"IP AUC={ip_result.auc_mg_h_per_L:.3f}, IV AUC={iv_auc:.3f} mg*h/L "
        f"(ratio={auc_ratio:.3f}). "
        f"IP Cmax={ip_result.cmax_mg_L:.4f}, IV Cmax={iv_cmax:.4f} mg/L "
        f"(ratio={cmax_ratio:.3f})."
    )

    return {
        "ip_result": ip_result,
        "iv_cmax_mg_L": iv_cmax,
        "iv_auc_mg_h_per_L": iv_auc,
        "ip_auc_mg_h_per_L": ip_result.auc_mg_h_per_L,
        "ip_cmax_mg_L": ip_result.cmax_mg_L,
        "auc_ratio_ip_iv": auc_ratio,
        "cmax_ratio_ip_iv": cmax_ratio,
        "notes": notes,
    }
