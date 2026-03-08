"""
Phase 481 — Receptor Down-Regulation Model

Models receptor down-regulation during chronic drug exposure (tachyphylaxis).
Receptor density decreases with sustained drug exposure (negative feedback).
"""

from __future__ import annotations

from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Result dataclass (plain @dataclass — has list fields)
# ---------------------------------------------------------------------------


@dataclass
class ReceptorDownregulationResult:
    """Result of receptor down-regulation simulation."""

    drug_name: str
    dose_mg: float
    times_h: list
    receptor_density: list  # normalized; baseline = 1.0
    effect: list
    css_avg_mg_L: float
    initial_effect: float
    final_effect: float
    downregulation_pct: float
    tachyphylaxis_detected: bool
    t_half_downregulation_h: float  # time for R to fall to 0.5; 0.0 if not reached
    notes: str


# ---------------------------------------------------------------------------
# Core simulation
# ---------------------------------------------------------------------------


def simulate_receptor_downregulation(
    drug_name: str,
    dose_mg: float,
    cl_L_per_h: float = 5.0,
    vd_L: float = 50.0,
    f_oral: float = 0.8,
    tau_h: float = 24.0,
    Emax: float = 1.0,
    EC50_mg_L: float = 1.0,
    k_synth: float = 1.0,
    k_deg: float = 1.0,
    k_intern: float = 0.5,
    t_end_h: float = 168.0,
    dt_h: float = 0.5,
) -> ReceptorDownregulationResult:
    """Simulate receptor down-regulation under chronic drug exposure.

    Drug concentration is approximated as steady-state Css_avg throughout
    (constant-infusion approximation for chronic dosing).

    Parameters
    ----------
    drug_name : str
        Name of the drug.
    dose_mg : float
        Dose in milligrams.
    cl_L_per_h : float
        Apparent clearance (L/h).
    vd_L : float
        Volume of distribution (L).
    f_oral : float
        Oral bioavailability (0 < f_oral <= 1).
    tau_h : float
        Dosing interval (h).
    Emax : float
        Maximum effect (dimensionless).
    EC50_mg_L : float
        Drug concentration producing 50% Emax (mg/L).
    k_synth : float
        Receptor synthesis rate (/h).
    k_deg : float
        Receptor baseline degradation rate (/h).
    k_intern : float
        Drug-induced internalization rate (/h per mg/L).
    t_end_h : float
        Simulation end time (h).
    dt_h : float
        Forward Euler step size (h).

    Returns
    -------
    ReceptorDownregulationResult
    """
    # --- Validation ---
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if cl_L_per_h <= 0:
        raise ValueError("cl_L_per_h must be > 0")
    if vd_L <= 0:
        raise ValueError("vd_L must be > 0")
    if not (0 < f_oral <= 1):
        raise ValueError("f_oral must be in (0, 1]")
    if tau_h <= 0:
        raise ValueError("tau_h must be > 0")
    if Emax <= 0:
        raise ValueError("Emax must be > 0")
    if EC50_mg_L <= 0:
        raise ValueError("EC50_mg_L must be > 0")
    if k_synth <= 0:
        raise ValueError("k_synth must be > 0")
    if k_deg <= 0:
        raise ValueError("k_deg must be > 0")
    if k_intern < 0:
        raise ValueError("k_intern must be >= 0")

    # --- Steady-state average drug concentration ---
    # Css_avg = F * D / (CL * tau)
    css_avg_mg_L = (f_oral * dose_mg) / (cl_L_per_h * tau_h)

    # --- Forward Euler ODE for receptor density ---
    # dR/dt = k_synth - k_deg * R - k_intern * C_ss * R
    # At t=0, R=1.0 (baseline = k_synth / k_deg = 1.0 when k_synth == k_deg)
    # We normalize to baseline=1.0 regardless of k_synth/k_deg ratio.
    R_baseline = k_synth / k_deg  # natural steady state without drug

    R = R_baseline  # initial condition
    t = 0.0

    times_h: list = [t]
    receptor_density: list = [R / R_baseline]  # normalized

    C = css_avg_mg_L  # constant approximation

    def _effect(R_norm: float, conc: float) -> float:
        eff_conc = R_norm * conc
        return Emax * eff_conc / (EC50_mg_L + eff_conc)

    effect_vals: list = [_effect(R / R_baseline, C)]

    t_half_found = 0.0
    half_target = 0.5 * R_baseline  # absolute R value for 50% of baseline

    n_steps = int(round(t_end_h / dt_h))
    for _step in range(n_steps):
        dR_dt = k_synth - k_deg * R - k_intern * C * R
        R_new = R + dt_h * dR_dt
        if R_new < 0.0:
            R_new = 0.0
        t_new = t + dt_h

        R_norm = R_new / R_baseline
        receptor_density.append(R_norm)
        times_h.append(t_new)
        effect_vals.append(_effect(R_norm, C))

        # Detect t_half (first crossing below 0.5 of baseline)
        if t_half_found == 0.0 and R_new <= half_target and R_baseline > 0:
            t_half_found = t_new

        R = R_new
        t = t_new

    initial_effect = effect_vals[0]
    final_effect = effect_vals[-1]

    if initial_effect > 0:
        downregulation_pct = (initial_effect - final_effect) / initial_effect * 100.0
    else:
        downregulation_pct = 0.0

    downregulation_pct = max(0.0, downregulation_pct)
    tachyphylaxis_detected = downregulation_pct > 20.0

    notes_parts = []
    if tachyphylaxis_detected:
        notes_parts.append(
            f"Tachyphylaxis detected: {downregulation_pct:.1f}% reduction in effect."
        )
    if k_intern == 0.0:
        notes_parts.append("k_intern=0: no drug-induced receptor down-regulation.")
    notes = " ".join(notes_parts) if notes_parts else "Simulation completed successfully."

    return ReceptorDownregulationResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        times_h=times_h,
        receptor_density=receptor_density,
        effect=effect_vals,
        css_avg_mg_L=css_avg_mg_L,
        initial_effect=initial_effect,
        final_effect=final_effect,
        downregulation_pct=downregulation_pct,
        tachyphylaxis_detected=tachyphylaxis_detected,
        t_half_downregulation_h=t_half_found,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Compare dosing regimens
# ---------------------------------------------------------------------------


def compare_dosing_regimens(
    drug_name: str,
    dose_mg: float,
    regimens: list,
) -> list:
    """Compare multiple dosing regimens by receptor down-regulation outcome.

    Parameters
    ----------
    drug_name : str
        Name of the drug.
    dose_mg : float
        Dose in milligrams.
    regimens : list[dict]
        Each dict may contain keys: "cl_L_per_h", "tau_h", "k_intern".
        Missing keys use defaults.

    Returns
    -------
    list[ReceptorDownregulationResult]
        Sorted by final_effect descending (best preserved effect first).
    """
    results = []
    for reg in regimens:
        cl = reg.get("cl_L_per_h", 5.0)
        tau = reg.get("tau_h", 24.0)
        ki = reg.get("k_intern", 0.5)
        result = simulate_receptor_downregulation(
            drug_name=drug_name,
            dose_mg=dose_mg,
            cl_L_per_h=cl,
            tau_h=tau,
            k_intern=ki,
        )
        results.append(result)

    results.sort(key=lambda r: r.final_effect, reverse=True)
    return results
