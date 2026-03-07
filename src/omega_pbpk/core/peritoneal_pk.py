"""Peritoneal drug absorption and dialysis pharmacokinetics (Phase 711).

Models intraperitoneal (IP) administration, which bypasses hepatic first-pass via
lymphatic pathway. Also supports peritoneal dialysis clearance for renally-impaired
patients.
"""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = ["PeritonealPKResult", "simulate_peritoneal_pk", "compare_routes_ip_iv"]


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
