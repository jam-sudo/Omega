"""Phase 715 — Sequential (IV → oral) dosing pharmacokinetics.

Models the transition from intravenous to oral therapy using a 1-compartment
forward-Euler ODE.
"""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = ["IVToOralResult", "simulate_iv_to_oral_switch", "optimize_switch_time"]


@dataclass
class IVToOralResult:
    """Result of an IV-to-oral switch simulation."""

    drug_name: str
    iv_dose_mg: float
    oral_dose_mg: float
    switch_time_h: float
    times_h: list = field(default_factory=list)
    c_plasma_mg_L: list = field(default_factory=list)
    cmax_iv_phase_mg_L: float = 0.0
    cmax_oral_phase_mg_L: float = 0.0
    trough_at_switch_mg_L: float = 0.0
    auc_iv_phase: float = 0.0
    auc_oral_phase: float = 0.0
    auc_total: float = 0.0
    t_half_h: float = 0.0
    therapeutic_continuity: bool = False
    notes: str = ""


def _trapezoidal_auc(times: list, concs: list) -> float:
    """Manual trapezoidal AUC."""
    auc = 0.0
    for i in range(1, len(times)):
        auc += 0.5 * (concs[i - 1] + concs[i]) * (times[i] - times[i - 1])
    return auc


def simulate_iv_to_oral_switch(
    drug_name: str,
    iv_dose_mg: float,
    oral_dose_mg: float,
    switch_time_h: float,
    iv_duration_h: float = 0.5,
    ka_per_h: float = 1.0,
    f_oral: float = 0.8,
    cl_L_per_h: float = 5.0,
    vd_L: float = 50.0,
    t_end_h: float = 48.0,
    dt_h: float = 0.05,
) -> IVToOralResult:
    """Simulate IV infusion followed by oral dosing at switch_time_h."""
    # Validation
    if iv_dose_mg <= 0:
        raise ValueError("iv_dose_mg must be positive")
    if oral_dose_mg <= 0:
        raise ValueError("oral_dose_mg must be positive")
    if switch_time_h <= 0:
        raise ValueError("switch_time_h must be positive")
    if not (0 < f_oral <= 1):
        raise ValueError("f_oral must be in (0, 1]")
    if cl_L_per_h <= 0:
        raise ValueError("cl_L_per_h must be positive")
    if vd_L <= 0:
        raise ValueError("vd_L must be positive")

    ke = cl_L_per_h / vd_L
    iv_rate = iv_dose_mg / iv_duration_h  # mg/h

    # State
    c = 0.0  # plasma concentration mg/L
    a_gut = 0.0  # gut amount mg — starts at zero, loaded at switch_time

    oral_dose_given = False

    times_h: list[float] = []
    c_plasma: list[float] = []

    t = 0.0
    n_steps = int(round(t_end_h / dt_h)) + 1

    for _ in range(n_steps):
        times_h.append(t)
        c_plasma.append(c)

        # Give oral dose exactly at switch_time (once)
        if not oral_dose_given and t >= switch_time_h - 1e-9:
            a_gut += oral_dose_mg
            oral_dose_given = True

        # IV infusion gate
        if t < switch_time_h and t < iv_duration_h:
            iv_input = iv_rate / vd_L  # mg/(L·h)
        else:
            iv_input = 0.0

        # Oral absorption contribution
        oral_input = ka_per_h * a_gut * f_oral / vd_L  # mg/(L·h)

        # Forward Euler
        dc_dt = iv_input + oral_input - ke * c
        da_gut_dt = -ka_per_h * a_gut

        c = max(0.0, c + dc_dt * dt_h)
        a_gut = max(0.0, a_gut + da_gut_dt * dt_h)

        t = round(t + dt_h, 10)

    # Split phases
    iv_times = [times_h[i] for i in range(len(times_h)) if times_h[i] <= switch_time_h]
    iv_concs = [c_plasma[i] for i in range(len(times_h)) if times_h[i] <= switch_time_h]
    oral_times = [times_h[i] for i in range(len(times_h)) if times_h[i] >= switch_time_h]
    oral_concs = [c_plasma[i] for i in range(len(times_h)) if times_h[i] >= switch_time_h]

    cmax_iv = max(iv_concs) if iv_concs else 0.0
    cmax_oral = max(oral_concs) if oral_concs else 0.0

    # Trough at switch: nearest time point
    trough = 0.0
    for i, t_val in enumerate(times_h):
        if abs(t_val - switch_time_h) < dt_h + 1e-9:
            trough = c_plasma[i]
            break

    auc_iv = _trapezoidal_auc(iv_times, iv_concs)
    auc_oral = _trapezoidal_auc(oral_times, oral_concs)
    auc_total = _trapezoidal_auc(times_h, c_plasma)

    t_half = 0.693147 / ke if ke > 0 else float("inf")

    therapeutic_continuity = trough > 0.5 * cmax_iv if cmax_iv > 0 else False

    notes_parts = [
        f"IV infusion rate {iv_rate:.1f} mg/h for {iv_duration_h}h.",
        f"Oral {oral_dose_mg}mg given at t={switch_time_h}h (F={f_oral}).",
        f"ke={ke:.4f}/h, t1/2={t_half:.2f}h.",
        "Therapeutic continuity maintained."
        if therapeutic_continuity
        else "Concentration gap at switch.",
    ]
    notes = " ".join(notes_parts)

    return IVToOralResult(
        drug_name=drug_name,
        iv_dose_mg=iv_dose_mg,
        oral_dose_mg=oral_dose_mg,
        switch_time_h=switch_time_h,
        times_h=times_h,
        c_plasma_mg_L=c_plasma,
        cmax_iv_phase_mg_L=cmax_iv,
        cmax_oral_phase_mg_L=cmax_oral,
        trough_at_switch_mg_L=trough,
        auc_iv_phase=auc_iv,
        auc_oral_phase=auc_oral,
        auc_total=auc_total,
        t_half_h=t_half,
        therapeutic_continuity=therapeutic_continuity,
        notes=notes,
    )


def optimize_switch_time(
    drug_name: str,
    iv_dose_mg: float,
    oral_dose_mg: float,
    cl_L_per_h: float,
    vd_L: float,
    ka_per_h: float,
    f_oral: float,
    target_trough_mg_L: float,
    candidate_times_h: list[float] | None = None,
) -> dict:
    """Find the optimal IV-to-oral switch time to maintain target trough concentration."""
    if candidate_times_h is None:
        candidate_times_h = [6.0, 12.0, 18.0, 24.0, 36.0, 48.0]

    candidates = []
    best_time = candidate_times_h[0]
    best_trough = None
    best_diff = None

    for switch_t in candidate_times_h:
        result = simulate_iv_to_oral_switch(
            drug_name=drug_name,
            iv_dose_mg=iv_dose_mg,
            oral_dose_mg=oral_dose_mg,
            switch_time_h=switch_t,
            cl_L_per_h=cl_L_per_h,
            vd_L=vd_L,
            ka_per_h=ka_per_h,
            f_oral=f_oral,
            t_end_h=switch_t + 24.0,
        )
        trough = result.trough_at_switch_mg_L
        candidates.append({"time": switch_t, "trough": trough})

        diff = abs(trough - target_trough_mg_L)
        if best_diff is None or diff < best_diff:
            best_diff = diff
            best_time = switch_t
            best_trough = trough

    return {
        "optimal_switch_h": best_time,
        "trough_achieved_mg_L": best_trough,
        "candidates": candidates,
    }
