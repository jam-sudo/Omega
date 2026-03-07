"""Phase 1007 — Cardiac Output Effect on Drug Distribution and PK.

Models how cardiac output (CO) determines organ perfusion rates and
thus hepatic/renal clearance and overall PK. Useful for heart failure
dose adjustment.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class CardiacOutputPKResult:
    """Result of a cardiac-output-driven 4-compartment PK simulation."""

    drug_name: str
    dose_mg: float
    cardiac_output_L_per_min: float
    times_h: list = field(default_factory=list)
    c_plasma_mg_L: list = field(default_factory=list)
    c_liver_mg_L: list = field(default_factory=list)
    c_kidney_mg_L: list = field(default_factory=list)
    c_muscle_mg_L: list = field(default_factory=list)
    cmax_plasma_mg_L: float = 0.0
    auc_plasma_mg_h_per_L: float = 0.0
    hepatic_extraction: float = 0.0
    renal_extraction: float = 0.0
    total_body_clearance_L_per_h: float = 0.0
    t_half_h: float = 0.0
    heart_failure_impact: str = "none"
    notes: str = ""


def _heart_failure_impact(co: float) -> str:
    if co >= 5.0:
        return "none"
    if co >= 3.5:
        return "mild"
    if co >= 2.5:
        return "moderate"
    return "severe"


def simulate_cardiac_output_pk(
    drug_name: str,
    dose_mg: float,
    cardiac_output_L_per_min: float = 5.0,
    logp: float = 2.0,
    fup: float = 0.5,
    clint_liver_mL_min: float = 50.0,
    fu_renal: float = 0.5,
    route: str = "iv",
    t_end_h: float = 24.0,
    dt_h: float = 0.1,
) -> CardiacOutputPKResult:
    """Simulate 4-compartment perfusion-limited PK as a function of cardiac output.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    dose_mg:
        Administered dose in mg.
    cardiac_output_L_per_min:
        Cardiac output in L/min. Normal ~5.0; heart failure 2–4.
    logp:
        Octanol-water partition coefficient (log scale).
    fup:
        Fraction unbound in plasma.
    clint_liver_mL_min:
        Intrinsic hepatic clearance in mL/min (unscaled).
    fu_renal:
        Fraction unbound in renal tubular fluid.
    route:
        "iv" or "oral".
    t_end_h:
        Simulation end time in hours.
    dt_h:
        Forward Euler step size in hours.

    Returns
    -------
    CardiacOutputPKResult
    """
    # --- Validation ---
    if dose_mg <= 0:
        raise ValueError(f"dose_mg must be > 0, got {dose_mg}")
    if cardiac_output_L_per_min <= 0:
        raise ValueError(f"cardiac_output_L_per_min must be > 0, got {cardiac_output_L_per_min}")
    route = route.lower()
    if route not in {"iv", "oral"}:
        raise ValueError(f"route must be 'iv' or 'oral', got {route!r}")

    # --- Organ flows (L/h) ---
    co_L_h = cardiac_output_L_per_min * 60.0
    q_liver = 0.25 * co_L_h
    q_kidney = 0.20 * co_L_h
    q_muscle = 0.17 * co_L_h

    # --- Organ volumes (L) ---
    v_plasma = 3.0
    v_liver = 1.5
    v_kidney = 0.3
    v_muscle = 28.0

    # --- Partition coefficient (tissue:plasma) ---
    kp = max(0.5, 10.0 ** (0.3 * logp))

    # --- Extraction ratios ---
    clint_liver_L_h = clint_liver_mL_min / 1000.0 * 60.0 * fup
    e_liver = clint_liver_L_h / (q_liver + clint_liver_L_h)
    e_kidney = fu_renal * 0.2  # simplified GFR fraction

    # --- Initial conditions ---
    c_plasma = 0.0
    c_liver = 0.0
    c_kidney = 0.0
    c_muscle = 0.0
    a_gut = 0.0  # oral depot (mg)
    ka = 0.8  # /h

    if route == "iv":
        c_plasma = dose_mg / v_plasma
    else:
        a_gut = dose_mg * 0.8  # 80% oral fraction absorbed from gut

    # --- Simulation ---
    # Use a stable internal sub-step: max rate constant determines step size.
    # Minimum time constant = min(V_organ / Q_organ); require dt_internal <= 0.3 * tau_min.
    tau_liver = v_liver / q_liver if q_liver > 0 else 1.0
    tau_kidney = v_kidney / q_kidney if q_kidney > 0 else 1.0
    tau_muscle = v_muscle / q_muscle if q_muscle > 0 else 1.0
    tau_plasma = v_plasma / (q_liver + q_kidney + q_muscle)
    tau_min = min(tau_liver, tau_kidney, tau_muscle, tau_plasma)
    dt_internal = min(dt_h, 0.2 * tau_min)
    # Reporting times
    n_report = max(1, int(t_end_h / dt_h))

    times: list[float] = []
    cp_list: list[float] = []
    cl_list: list[float] = []
    ck_list: list[float] = []
    cm_list: list[float] = []

    t = 0.0
    next_report_idx = 0
    report_times = [i * dt_h for i in range(n_report + 1)]

    def _record() -> None:
        times.append(t)
        cp_list.append(c_plasma)
        cl_list.append(c_liver)
        ck_list.append(c_kidney)
        cm_list.append(c_muscle)

    _record()
    next_report_idx = 1

    t_total_steps = max(1, int(t_end_h / dt_internal + 0.5))
    for _i in range(t_total_steps):
        # Compute actual step (don't overshoot next report time)
        t_next_report = report_times[next_report_idx] if next_report_idx <= n_report else t_end_h
        h = min(dt_internal, t_next_report - t)
        if h <= 0:
            # Record and advance report pointer
            if next_report_idx <= n_report and abs(t - t_next_report) < 1e-12:
                # already recorded if t matches; just advance pointer
                if len(times) <= next_report_idx:
                    _record()
                next_report_idx += 1
            if t >= t_end_h - 1e-12:
                break
            continue

        # Oral absorption input (mg/h → mg/L/h via v_plasma)
        if route == "oral":
            absorbed = ka * a_gut * h
            a_gut = max(0.0, a_gut - absorbed)
            oral_input = absorbed / v_plasma / h  # rate mg/L/h
        else:
            oral_input = 0.0

        # Standard perfusion-limited ODEs.
        # Venous concentration leaving each organ = C_organ/kp * (1 - E).
        # Plasma receives return flow from organs; outflow = total Q * C_plasma.
        c_liver_ven = c_liver / kp * (1.0 - e_liver)
        c_kidney_ven = c_kidney / kp * (1.0 - e_kidney)
        c_muscle_ven = c_muscle / kp  # no elimination in muscle

        dc_plasma = (
            oral_input
            + (q_liver * c_liver_ven + q_kidney * c_kidney_ven + q_muscle * c_muscle_ven) / v_plasma
            - (q_liver + q_kidney + q_muscle) / v_plasma * c_plasma
        )
        # Each organ: arterial in Q*C_plasma, venous out Q*C_organ/kp
        dc_liver = (q_liver * c_plasma - q_liver * c_liver / kp) / v_liver
        dc_kidney = (q_kidney * c_plasma - q_kidney * c_kidney / kp) / v_kidney
        dc_muscle = (q_muscle * c_plasma - q_muscle * c_muscle / kp) / v_muscle

        c_plasma = max(0.0, c_plasma + dc_plasma * h)
        c_liver = max(0.0, c_liver + dc_liver * h)
        c_kidney = max(0.0, c_kidney + dc_kidney * h)
        c_muscle = max(0.0, c_muscle + dc_muscle * h)
        t += h

        # Record at reporting times
        if next_report_idx <= n_report and t >= report_times[next_report_idx] - 1e-12:
            _record()
            next_report_idx += 1

        if t >= t_end_h - 1e-12:
            break

    # Ensure final time point is recorded if missing
    if not times or times[-1] < t_end_h - 1e-12:
        times.append(t_end_h)
        cp_list.append(c_plasma)
        cl_list.append(c_liver)
        ck_list.append(c_kidney)
        cm_list.append(c_muscle)

    # --- Derived PK metrics ---
    cmax = max(cp_list)

    # Trapezoidal AUC
    auc = 0.0
    for i in range(1, len(times)):
        auc += 0.5 * (cp_list[i - 1] + cp_list[i]) * (times[i] - times[i - 1])

    total_cl = e_liver * q_liver + e_kidney * q_kidney
    t_half = 0.693 * v_plasma / total_cl if total_cl > 0 else float("inf")

    hf_impact = _heart_failure_impact(cardiac_output_L_per_min)

    notes_parts = [
        f"CO={cardiac_output_L_per_min:.1f} L/min",
        f"Q_liver={q_liver:.1f} L/h",
        f"E_liver={e_liver:.3f}",
        f"E_kidney={e_kidney:.3f}",
        f"kp={kp:.2f}",
    ]
    if hf_impact != "none":
        notes_parts.append(f"Heart failure impact: {hf_impact}")

    return CardiacOutputPKResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        cardiac_output_L_per_min=cardiac_output_L_per_min,
        times_h=times,
        c_plasma_mg_L=cp_list,
        c_liver_mg_L=cl_list,
        c_kidney_mg_L=ck_list,
        c_muscle_mg_L=cm_list,
        cmax_plasma_mg_L=cmax,
        auc_plasma_mg_h_per_L=auc,
        hepatic_extraction=e_liver,
        renal_extraction=e_kidney,
        total_body_clearance_L_per_h=total_cl,
        t_half_h=t_half,
        heart_failure_impact=hf_impact,
        notes="; ".join(notes_parts),
    )
