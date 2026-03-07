"""Drug rebound and withdrawal PK/PD model -- Phase 273.

Models the overshoot (rebound) in biological response after abrupt
drug withdrawal, due to upregulation of receptor/target during chronic
drug exposure. Relevant for beta-blockers, corticosteroids, antidepressants.

References
----------
- Lohse MJ et al., Trends Pharmacol Sci. 1990;11(2):49-54
- Dayneka NL et al., J Pharmacokinet Biopharm. 1993;21(4):457-78
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ReboundResult:
    """Result from drug rebound/withdrawal simulation."""

    drug_name: str
    dose_mg: float
    n_doses_chronic: int
    times_h: list[float]
    c_plasma_mg_L: list[float]
    effect: list[float]  # Normalized PD effect (0-1, 1=baseline)
    receptor_density: list[float]  # Receptor upregulation factor (1.0 = normal)
    drug_phase: list[str]  # "dosing", "withdrawal"
    peak_rebound: float  # Max effect above baseline (overshoot)
    time_peak_rebound_h: float
    rebound_duration_h: float  # Duration effect > 1.1 * baseline
    withdrawal_onset_h: float  # Time of drug discontinuation
    notes: str


def simulate_rebound(
    drug_name: str,
    dose_mg: float,
    cl_L_per_h: float,
    vd_L: float,
    n_doses_chronic: int = 14,
    dosing_interval_h: float = 12.0,
    ec50_mg_L: float = 0.1,
    emax: float = 0.8,
    k_receptor_up: float = 0.1,  # Receptor upregulation rate (h^-1)
    k_receptor_down: float = 0.05,  # Receptor downregulation rate (h^-1)
    receptor_max: float = 3.0,  # Maximum receptor upregulation fold
    ka_per_h: float = 1.0,
    f_oral: float = 0.9,
    t_washout_h: float = 72.0,
    dt_h: float = 0.1,
) -> ReboundResult:
    """Simulate chronic dosing followed by withdrawal rebound.

    Parameters
    ----------
    drug_name : Drug name.
    dose_mg : Dose per administration (mg). Must be > 0.
    cl_L_per_h : Clearance (L/h). Must be > 0.
    vd_L : Volume of distribution (L). Must be > 0.
    n_doses_chronic : Number of doses before withdrawal. Must be >= 1.
    dosing_interval_h : Dosing interval (h). Must be > 0.
    ec50_mg_L : EC50 for drug effect (mg/L). Must be > 0.
    emax : Maximum drug effect (0, 1]. Must be in (0, 1].
    k_receptor_up : Upregulation rate during drug exposure (h^-1). Must be >= 0.
    k_receptor_down : Downregulation (recovery) rate after drug removal (h^-1). Must be >= 0.
    receptor_max : Maximum upregulation fold. Must be >= 1.
    ka_per_h : Oral absorption rate (h^-1).
    f_oral : Bioavailability (0, 1].
    t_washout_h : Post-withdrawal simulation time (h).
    dt_h : Time step (h).

    Returns
    -------
    ReboundResult
    """
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if cl_L_per_h <= 0:
        raise ValueError("cl_L_per_h must be > 0")
    if vd_L <= 0:
        raise ValueError("vd_L must be > 0")
    if n_doses_chronic < 1:
        raise ValueError("n_doses_chronic must be >= 1")
    if dosing_interval_h <= 0:
        raise ValueError("dosing_interval_h must be > 0")
    if ec50_mg_L <= 0:
        raise ValueError("ec50_mg_L must be > 0")
    if not (0 < emax <= 1):
        raise ValueError("emax must be in (0, 1]")
    if k_receptor_up < 0 or k_receptor_down < 0:
        raise ValueError("k_receptor rates must be >= 0")
    if receptor_max < 1:
        raise ValueError("receptor_max must be >= 1")
    if not (0 < f_oral <= 1):
        raise ValueError("f_oral must be in (0, 1]")

    ke = cl_L_per_h / vd_L
    withdrawal_h = n_doses_chronic * dosing_interval_h
    t_end = withdrawal_h + t_washout_h

    n_steps = int(t_end / dt_h) + 1
    times = [i * dt_h for i in range(n_steps)]
    c_plasma = [0.0] * n_steps
    receptor = [1.0] * n_steps  # normalized receptor density
    a_gut = 0.0

    # Dose schedule: doses at 0, interval, 2*interval, ...
    dose_steps = {round(d * dosing_interval_h / dt_h) for d in range(n_doses_chronic)}

    for i in range(1, n_steps):
        t_cur = times[i - 1]
        cp = c_plasma[i - 1]
        rec = receptor[i - 1]

        # Administer dose
        if i in dose_steps:
            a_gut += dose_mg * f_oral

        # Oral absorption (always oral route)
        abs_r = ka_per_h * a_gut
        a_gut = max(0.0, a_gut - abs_r * dt_h)

        dc = (abs_r - ke * cp * vd_L) / vd_L * dt_h
        c_plasma[i] = max(0.0, cp + dc)

        # Receptor dynamics
        drug_effect_on_receptor = cp / (ec50_mg_L + cp) if cp > 0 else 0.0
        if t_cur < withdrawal_h:
            # During dosing: receptors upregulate due to chronic exposure
            dr = k_receptor_up * drug_effect_on_receptor * (receptor_max - rec) * dt_h
        else:
            # After withdrawal: receptors slowly return to normal
            dr = -k_receptor_down * (rec - 1.0) * dt_h
        receptor[i] = max(1.0, min(receptor_max, rec + dr))

    # Effect: Emax model, but effective EC50 reduced by receptor upregulation
    # Rebound: high receptor density + residual drug -> overshoot
    effect = []
    for cp, rec in zip(c_plasma, receptor, strict=True):
        ec50_eff = ec50_mg_L / rec  # upregulated receptors -> more sensitive
        eff = emax * cp / (ec50_eff + cp) if cp > 0 else 0.0
        # Baseline = 1.0; during dosing effect > 0 suppresses it
        # After withdrawal, receptors remain elevated -> rebound when drug gone
        # Simplified: net_effect = 1.0 - drug_effect + receptor_excess_response
        receptor_excess = (rec - 1.0) * 0.3  # excess receptor drives rebound
        net = max(0.0, 1.0 - eff + receptor_excess)
        effect.append(net)

    # Metrics
    withdrawal_idx = round(withdrawal_h / dt_h)
    post_withdrawal_effects = effect[withdrawal_idx:]
    post_withdrawal_times = times[withdrawal_idx:]

    peak_rebound = max(post_withdrawal_effects) if post_withdrawal_effects else 1.0
    if post_withdrawal_effects:
        peak_t_idx = post_withdrawal_effects.index(peak_rebound)
        time_peak_r = post_withdrawal_times[peak_t_idx]
    else:
        time_peak_r = withdrawal_h

    rebound_dur = sum(dt_h for e in post_withdrawal_effects if e > 1.1)

    phases = ["dosing" if t < withdrawal_h else "withdrawal" for t in times]

    notes = (
        f"{drug_name}: {n_doses_chronic} doses of {dose_mg}mg q{dosing_interval_h:.0f}h. "
        f"Withdrawal at {withdrawal_h:.0f}h. "
        f"Peak rebound={peak_rebound:.2f} at {time_peak_r:.1f}h. "
        f"Rebound duration>{1.1:.1f}={rebound_dur:.1f}h."
    )

    return ReboundResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        n_doses_chronic=n_doses_chronic,
        times_h=times,
        c_plasma_mg_L=c_plasma,
        effect=effect,
        receptor_density=receptor,
        drug_phase=phases,
        peak_rebound=peak_rebound,
        time_peak_rebound_h=time_peak_r,
        rebound_duration_h=rebound_dur,
        withdrawal_onset_h=withdrawal_h,
        notes=notes,
    )


__all__ = ["ReboundResult", "simulate_rebound"]
