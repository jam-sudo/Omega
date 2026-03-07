"""Phase 773 — Drug Formulation PK Comparison.

Compare PK profiles across different formulations (IR, ER, CR, delayed-release, biphasic)
to evaluate dosing regimens and patient outcomes using 1-cpt forward Euler simulation.
"""

from __future__ import annotations

from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FormulationPKResult:
    """PK comparison result for a single formulation."""

    drug_name: str
    formulation_type: str
    dose_mg: float
    cl_L_per_h: float
    vd_L: float
    f_oral: float
    cmax_mg_L: float
    tmax_h: float
    auc_mg_h_per_L: float
    time_above_mec_h: float
    time_above_mtc_h: float
    time_in_therapeutic_h: float
    fluctuation_pct: float
    composite_score: float
    notes: str


# ---------------------------------------------------------------------------
# Internal simulation helpers
# ---------------------------------------------------------------------------


def _simulate_1cpt(
    dose_mg: float,
    ka_eff: float,
    ke: float,
    vd_L: float,
    f_oral: float,
    lag_h: float,
    t_end_h: float,
    dt_h: float,
) -> tuple[list[float], list[float]]:
    """Forward Euler 1-cpt oral simulation with optional lag time.

    Returns (times, concentrations) in mg/L.
    """
    n_steps = int(t_end_h / dt_h) + 1
    times: list[float] = []
    concs: list[float] = []

    a_gut = 0.0
    c_plasma = 0.0
    dose_given = False

    for i in range(n_steps):
        t = i * dt_h
        times.append(t)
        concs.append(c_plasma)

        # Lag time: give dose once we reach lag_h
        if not dose_given and t >= lag_h:
            a_gut += dose_mg
            dose_given = True

        # Derivatives
        absorption = ka_eff * a_gut
        elimination = ke * c_plasma

        # Update
        a_gut = a_gut - absorption * dt_h
        if a_gut < 0.0:
            a_gut = 0.0
        c_plasma = c_plasma + (absorption * f_oral / vd_L - elimination) * dt_h
        if c_plasma < 0.0:
            c_plasma = 0.0

    return times, concs


def _simulate_biphasic(
    dose_mg: float,
    ka_base: float,
    ke: float,
    vd_L: float,
    f_oral: float,
    t_end_h: float,
    dt_h: float,
) -> tuple[list[float], list[float]]:
    """Biphasic simulation: 30% IR + 70% ER (superposition of two 1-cpt).

    Returns (times, concentrations) in mg/L.
    """
    # Fast phase: 30% of dose, ka = ka_base
    t_fast, c_fast = _simulate_1cpt(
        dose_mg=0.3 * dose_mg,
        ka_eff=ka_base,
        ke=ke,
        vd_L=vd_L,
        f_oral=f_oral,
        lag_h=0.0,
        t_end_h=t_end_h,
        dt_h=dt_h,
    )
    # Slow phase: 70% of dose, ka = ka_base * 0.25
    _t_slow, c_slow = _simulate_1cpt(
        dose_mg=0.7 * dose_mg,
        ka_eff=ka_base * 0.25,
        ke=ke,
        vd_L=vd_L,
        f_oral=f_oral,
        lag_h=0.0,
        t_end_h=t_end_h,
        dt_h=dt_h,
    )
    combined = [cf + cs for cf, cs in zip(c_fast, c_slow)]
    return t_fast, combined


# ---------------------------------------------------------------------------
# PK metric helpers
# ---------------------------------------------------------------------------


def _compute_metrics(
    times: list[float],
    concs: list[float],
    mec_mg_L: float,
    mtc_mg_L: float,
    dt_h: float,
) -> dict:
    """Compute Cmax, Tmax, AUC, time_above_mec, time_above_mtc, time_in_therapeutic."""
    cmax = 0.0
    tmax = 0.0
    for t, c in zip(times, concs):
        if c > cmax:
            cmax = c
            tmax = t

    # Trapezoidal AUC
    auc = 0.0
    for i in range(1, len(times)):
        auc += 0.5 * (concs[i - 1] + concs[i]) * (times[i] - times[i - 1])

    time_above_mec = 0.0
    time_above_mtc = 0.0
    time_in_therapeutic = 0.0
    for i in range(len(times) - 1):
        seg_dt = times[i + 1] - times[i]
        c_avg = 0.5 * (concs[i] + concs[i + 1])
        if c_avg >= mec_mg_L:
            time_above_mec += seg_dt
        if c_avg >= mtc_mg_L:
            time_above_mtc += seg_dt
        if mec_mg_L <= c_avg < mtc_mg_L:
            time_in_therapeutic += seg_dt

    return {
        "cmax": cmax,
        "tmax": tmax,
        "auc": auc,
        "time_above_mec": time_above_mec,
        "time_above_mtc": time_above_mtc,
        "time_in_therapeutic": time_in_therapeutic,
    }


def _compute_fluctuation(
    times: list[float],
    concs: list[float],
    t_start_h: float,
) -> float:
    """Compute fluctuation pct over the last portion of the profile.

    fluctuation = (Cmax - Cmin_last) / Cavg_last * 100
    """
    last_concs = [c for t, c in zip(times, concs) if t >= t_start_h]
    if not last_concs:
        last_concs = concs

    c_max_last = max(last_concs)
    c_min_last = min(last_concs)
    c_avg_last = sum(last_concs) / len(last_concs) if last_concs else 1.0

    if c_avg_last <= 0.0:
        return 0.0
    return (c_max_last - c_min_last) / c_avg_last * 100.0


# ---------------------------------------------------------------------------
# Convenience mapping
# ---------------------------------------------------------------------------

_CONVENIENCE = {
    "IR": 80.0,
    "ER": 90.0,
    "CR": 90.0,
    "DR": 70.0,
    "Biphasic": 70.0,
}

_ALL_FORMULATIONS = ["IR", "ER", "CR", "DR", "Biphasic"]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compare_formulations(
    drug_name: str,
    dose_mg: float,
    cl_L_per_h: float = 5.0,
    vd_L: float = 50.0,
    f_oral: float = 0.8,
    ka_base_per_h: float = 1.0,
    formulations: list[str] | None = None,
    mec_mg_L: float = 0.5,
    mtc_mg_L: float = 5.0,
    t_end_h: float = 24.0,
    dt_h: float = 0.05,
) -> list[FormulationPKResult]:
    """Compare PK profiles across different formulations.

    Parameters
    ----------
    drug_name : str
    dose_mg : float
        Total dose (same for all formulations).
    cl_L_per_h : float
        Systemic clearance in L/h.
    vd_L : float
        Volume of distribution in L.
    f_oral : float
        Oral bioavailability fraction.
    ka_base_per_h : float
        Base absorption rate constant (IR rate).
    formulations : list[str] | None
        Subset to simulate; defaults to all 5.
    mec_mg_L : float
        Minimum effective concentration.
    mtc_mg_L : float
        Minimum toxic concentration.
    t_end_h : float
        Simulation end time in hours.
    dt_h : float
        Time step for forward Euler.

    Returns
    -------
    list[FormulationPKResult]
        Sorted by composite_score descending.
    """
    # Validation
    if dose_mg <= 0:
        raise ValueError(f"dose_mg must be > 0, got {dose_mg}")
    if cl_L_per_h <= 0:
        raise ValueError(f"cl_L_per_h must be > 0, got {cl_L_per_h}")
    if vd_L <= 0:
        raise ValueError(f"vd_L must be > 0, got {vd_L}")
    if mec_mg_L <= 0:
        raise ValueError(f"mec_mg_L must be > 0, got {mec_mg_L}")
    if mec_mg_L >= mtc_mg_L:
        raise ValueError(f"mec_mg_L ({mec_mg_L}) must be < mtc_mg_L ({mtc_mg_L})")

    if formulations is None:
        formulations = _ALL_FORMULATIONS

    ke = cl_L_per_h / vd_L
    half_time = t_end_h / 2.0  # last half for fluctuation

    results: list[FormulationPKResult] = []

    for form in formulations:
        if form not in _ALL_FORMULATIONS:
            raise ValueError(f"Unknown formulation '{form}'. Choose from {_ALL_FORMULATIONS}")

        if form == "IR":
            times, concs = _simulate_1cpt(
                dose_mg=dose_mg,
                ka_eff=ka_base_per_h,
                ke=ke,
                vd_L=vd_L,
                f_oral=f_oral,
                lag_h=0.0,
                t_end_h=t_end_h,
                dt_h=dt_h,
            )
            notes = "Standard immediate release; rapid onset, higher fluctuation."

        elif form == "ER":
            times, concs = _simulate_1cpt(
                dose_mg=dose_mg,
                ka_eff=ka_base_per_h * 0.25,
                ke=ke,
                vd_L=vd_L,
                f_oral=f_oral,
                lag_h=0.0,
                t_end_h=t_end_h,
                dt_h=dt_h,
            )
            notes = "Extended release (4x slower ka); sustained plasma levels, lower Cmax."

        elif form == "CR":
            times, concs = _simulate_1cpt(
                dose_mg=dose_mg,
                ka_eff=ka_base_per_h * 0.1,
                ke=ke,
                vd_L=vd_L,
                f_oral=f_oral,
                lag_h=0.0,
                t_end_h=t_end_h,
                dt_h=dt_h,
            )
            notes = "Controlled release (10x slower ka); flattest profile, lowest fluctuation."

        elif form == "DR":
            times, concs = _simulate_1cpt(
                dose_mg=dose_mg,
                ka_eff=ka_base_per_h,
                ke=ke,
                vd_L=vd_L,
                f_oral=f_oral,
                lag_h=2.0,
                t_end_h=t_end_h,
                dt_h=dt_h,
            )
            notes = "Delayed release (2h lag then rapid release); useful for enteric coating."

        else:  # Biphasic
            times, concs = _simulate_biphasic(
                dose_mg=dose_mg,
                ka_base=ka_base_per_h,
                ke=ke,
                vd_L=vd_L,
                f_oral=f_oral,
                t_end_h=t_end_h,
                dt_h=dt_h,
            )
            notes = "Biphasic: 30% rapid loading + 70% extended; combines fast onset with sustained action."

        metrics = _compute_metrics(times, concs, mec_mg_L, mtc_mg_L, dt_h)
        fluctuation = _compute_fluctuation(times, concs, t_start_h=half_time)
        convenience = _CONVENIENCE[form]

        # Time in range percentage
        tir_pct = metrics["time_in_therapeutic"] / t_end_h * 100.0

        # Composite score
        fluc_component = 100.0 - fluctuation / 2.0
        if fluc_component < 0.0:
            fluc_component = 0.0

        composite = 0.5 * tir_pct + 0.3 * fluc_component + 0.2 * convenience
        composite = min(100.0, max(0.0, composite))

        result = FormulationPKResult(
            drug_name=drug_name,
            formulation_type=form,
            dose_mg=dose_mg,
            cl_L_per_h=cl_L_per_h,
            vd_L=vd_L,
            f_oral=f_oral,
            cmax_mg_L=metrics["cmax"],
            tmax_h=metrics["tmax"],
            auc_mg_h_per_L=metrics["auc"],
            time_above_mec_h=metrics["time_above_mec"],
            time_above_mtc_h=metrics["time_above_mtc"],
            time_in_therapeutic_h=metrics["time_in_therapeutic"],
            fluctuation_pct=fluctuation,
            composite_score=composite,
            notes=notes,
        )
        results.append(result)

    results.sort(key=lambda r: r.composite_score, reverse=True)
    return results


def recommend_formulation(
    drug_name: str,
    dose_mg: float,
    cl_L_per_h: float = 5.0,
    vd_L: float = 50.0,
    f_oral: float = 0.8,
    ka_base_per_h: float = 1.0,
    formulations: list[str] | None = None,
    mec_mg_L: float = 0.5,
    mtc_mg_L: float = 5.0,
    t_end_h: float = 24.0,
    dt_h: float = 0.05,
) -> dict:
    """Recommend the best formulation for a drug based on PK analysis.

    Returns a dict with:
    - best_formulation: str
    - rationale: str
    - results: list[FormulationPKResult]
    """
    results = compare_formulations(
        drug_name=drug_name,
        dose_mg=dose_mg,
        cl_L_per_h=cl_L_per_h,
        vd_L=vd_L,
        f_oral=f_oral,
        ka_base_per_h=ka_base_per_h,
        formulations=formulations,
        mec_mg_L=mec_mg_L,
        mtc_mg_L=mtc_mg_L,
        t_end_h=t_end_h,
        dt_h=dt_h,
    )

    best = results[0]
    rationale = (
        f"{best.formulation_type} formulation achieves the highest composite score "
        f"({best.composite_score:.1f}/100) for {drug_name}. "
        f"Cmax={best.cmax_mg_L:.3f} mg/L, Tmax={best.tmax_h:.1f}h, "
        f"AUC={best.auc_mg_h_per_L:.2f} mg·h/L, "
        f"time in therapeutic window={best.time_in_therapeutic_h:.1f}h, "
        f"fluctuation={best.fluctuation_pct:.1f}%. "
        f"{best.notes}"
    )

    return {
        "best_formulation": best.formulation_type,
        "rationale": rationale,
        "results": results,
    }
