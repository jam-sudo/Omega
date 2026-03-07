"""Drug target engagement prediction — Phase 479.

Predicts target occupancy from plasma PK and target binding parameters using
the Hill occupancy equation (Langmuir isotherm for a 1:1 binding model).

Occupancy equation:
    RO(%) = 100 * Cu_nM / (Cu_nM + Kd_nM)

where Cu_nM is the free (unbound) drug concentration at the target site,
converted from plasma concentration via fu_plasma (and optionally Kp_brain).

References
----------
- Copeland RA et al., Nat Rev Drug Discov. 2006;5:730-739
- Danhof M et al., Annu Rev Pharmacol Toxicol. 2007;47:357-400
"""

from __future__ import annotations

from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TargetEngagementResult:
    """Result of a single-timepoint target engagement prediction.

    Attributes
    ----------
    drug_name : Drug identifier.
    plasma_conc_mg_L : Total plasma concentration (mg/L).
    free_plasma_conc_nM : Unbound plasma concentration (nM).
    kd_nM : Target dissociation constant (nM).
    target_occupancy_pct : Predicted receptor occupancy (%).
    is_above_ec50 : Whether occupancy exceeds 50% (EC50 threshold).
    safety_margin : Ratio of free concentration to Kd (dimensionless).
    notes : Human-readable summary.
    """

    drug_name: str
    plasma_conc_mg_L: float
    free_plasma_conc_nM: float
    kd_nM: float
    target_occupancy_pct: float
    is_above_ec50: bool
    safety_margin: float
    notes: str


# ---------------------------------------------------------------------------
# Unit conversion helpers
# ---------------------------------------------------------------------------


def _mg_L_to_nM(conc_mg_L: float, mw_g_mol: float) -> float:
    """Convert mg/L to nM.

    mg/L * (1 g / 1000 mg) * (1 mol / mw_g_mol g) * (1e9 nmol / mol)
    = conc_mg_L * 1000 / mw_g_mol
    """
    return conc_mg_L * 1_000.0 / mw_g_mol


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------


def predict_target_engagement(
    drug_name: str,
    plasma_conc_mg_L: float,
    fu_plasma: float,
    kd_nM: float,
    mw_g_mol: float = 500.0,
    kp_brain: float | None = None,
) -> TargetEngagementResult:
    """Predict target engagement at a single plasma concentration.

    Parameters
    ----------
    drug_name : Drug identifier string.
    plasma_conc_mg_L : Total plasma concentration (mg/L).
    fu_plasma : Unbound fraction in plasma (0 < fu_plasma <= 1).
    kd_nM : Target dissociation constant (nM); must be > 0.
    mw_g_mol : Molecular weight (g/mol); default 500.
    kp_brain : Brain-to-plasma partition coefficient for CNS targets.
        If provided, free brain concentration is used instead of free plasma.

    Returns
    -------
    TargetEngagementResult
    """
    if not (0 < fu_plasma <= 1.0):
        raise ValueError(f"fu_plasma must be in (0, 1], got {fu_plasma}")
    if kd_nM <= 0:
        raise ValueError(f"kd_nM must be > 0, got {kd_nM}")
    if mw_g_mol <= 0:
        raise ValueError(f"mw_g_mol must be > 0, got {mw_g_mol}")
    if plasma_conc_mg_L < 0:
        raise ValueError(f"plasma_conc_mg_L must be >= 0, got {plasma_conc_mg_L}")

    total_nM = _mg_L_to_nM(plasma_conc_mg_L, mw_g_mol)
    free_plasma_nM = total_nM * fu_plasma

    if kp_brain is not None:
        if kp_brain < 0:
            raise ValueError(f"kp_brain must be >= 0, got {kp_brain}")
        free_target_nM = free_plasma_nM * kp_brain
        site = "brain"
    else:
        free_target_nM = free_plasma_nM
        site = "plasma"

    # Langmuir / Hill (n=1) occupancy
    occupancy_pct = 100.0 * free_target_nM / (free_target_nM + kd_nM)
    is_above_ec50 = occupancy_pct >= 50.0
    safety_margin = free_target_nM / kd_nM

    notes = (
        f"Free {site} conc = {free_target_nM:.2f} nM, Kd = {kd_nM:.2f} nM, "
        f"occupancy = {occupancy_pct:.1f}%"
    )

    return TargetEngagementResult(
        drug_name=drug_name,
        plasma_conc_mg_L=plasma_conc_mg_L,
        free_plasma_conc_nM=free_plasma_nM,
        kd_nM=kd_nM,
        target_occupancy_pct=occupancy_pct,
        is_above_ec50=is_above_ec50,
        safety_margin=safety_margin,
        notes=notes,
    )


def te_time_course(
    plasma_concs_mg_L: list[float],
    times_h: list[float],
    fu_plasma: float,
    kd_nM: float,
    mw_g_mol: float = 500.0,
    ec50_pct: float = 50.0,
) -> dict:
    """Compute target engagement profile over a concentration-time course.

    Parameters
    ----------
    plasma_concs_mg_L : Sequence of total plasma concentrations (mg/L).
    times_h : Corresponding time points (h); must have same length.
    fu_plasma : Unbound fraction in plasma (0 < fu_plasma <= 1).
    kd_nM : Target dissociation constant (nM); must be > 0.
    mw_g_mol : Molecular weight (g/mol).
    ec50_pct : Occupancy threshold marking pharmacological effect (%).

    Returns
    -------
    dict with keys:
        'times_h'            : list[float]
        'plasma_concs_mg_L'  : list[float]
        'free_concs_nM'      : list[float]
        'occupancy_pct'      : list[float]
        'above_ec50'         : list[bool]
        'peak_occupancy_pct' : float
        'trough_occupancy_pct': float
    """
    if not (0 < fu_plasma <= 1.0):
        raise ValueError(f"fu_plasma must be in (0, 1], got {fu_plasma}")
    if kd_nM <= 0:
        raise ValueError(f"kd_nM must be > 0, got {kd_nM}")
    if len(plasma_concs_mg_L) != len(times_h):
        raise ValueError("plasma_concs_mg_L and times_h must have the same length")

    free_concs: list[float] = []
    occupancies: list[float] = []
    above_ec50: list[bool] = []

    for conc in plasma_concs_mg_L:
        total_nM = _mg_L_to_nM(conc, mw_g_mol)
        free_nM = total_nM * fu_plasma
        occ = 100.0 * free_nM / (free_nM + kd_nM)
        free_concs.append(free_nM)
        occupancies.append(occ)
        above_ec50.append(occ >= ec50_pct)

    peak = max(occupancies) if occupancies else 0.0
    trough = min(occupancies) if occupancies else 0.0

    return {
        "times_h": list(times_h),
        "plasma_concs_mg_L": list(plasma_concs_mg_L),
        "free_concs_nM": free_concs,
        "occupancy_pct": occupancies,
        "above_ec50": above_ec50,
        "peak_occupancy_pct": peak,
        "trough_occupancy_pct": trough,
    }


def minimum_effective_conc(
    kd_nM: float,
    fu_plasma: float,
    target_occupancy_pct: float = 80.0,
    mw_g_mol: float = 500.0,
) -> float:
    """Compute minimum plasma concentration to achieve target occupancy.

    Uses the inverse of the Hill/Langmuir equation:
        Cu_target = Kd * (RO / (100 - RO))
        C_plasma  = Cu_target / fu_plasma  [nM]
        C_plasma  [mg/L] = C_plasma_nM * mw_g_mol / 1000

    Parameters
    ----------
    kd_nM : Target dissociation constant (nM).
    fu_plasma : Unbound fraction in plasma (0 < fu_plasma <= 1).
    target_occupancy_pct : Desired receptor occupancy (%); must be in (0, 100).
    mw_g_mol : Molecular weight (g/mol).

    Returns
    -------
    float : Minimum total plasma concentration (mg/L).
    """
    if kd_nM <= 0:
        raise ValueError(f"kd_nM must be > 0, got {kd_nM}")
    if not (0 < fu_plasma <= 1.0):
        raise ValueError(f"fu_plasma must be in (0, 1], got {fu_plasma}")
    if not (0 < target_occupancy_pct < 100):
        raise ValueError(
            f"target_occupancy_pct must be in (0, 100), got {target_occupancy_pct}"
        )
    if mw_g_mol <= 0:
        raise ValueError(f"mw_g_mol must be > 0, got {mw_g_mol}")

    # Free concentration at target site (nM)
    cu_nM = kd_nM * target_occupancy_pct / (100.0 - target_occupancy_pct)
    # Total plasma concentration (nM)
    total_nM = cu_nM / fu_plasma
    # Convert nM -> mg/L: nM * mw_g_mol / 1e6 * 1000 = nM * mw_g_mol / 1000
    total_mg_L = total_nM * mw_g_mol / 1_000_000.0 * 1_000.0
    return total_mg_L


def duration_above_threshold(
    te_profile_pct: list[float],
    times_h: list[float],
    threshold_pct: float = 50.0,
) -> float:
    """Estimate duration during which target engagement exceeds a threshold.

    Uses trapezoidal interpolation to find the time spent above the threshold.
    For simplicity, each interval is counted if the midpoint occupancy exceeds
    the threshold (step-function approximation), which matches common PK/PD
    T>threshold calculations.

    Parameters
    ----------
    te_profile_pct : Target occupancy at each time point (%).
    times_h : Corresponding time points (h).
    threshold_pct : Occupancy threshold (%).

    Returns
    -------
    float : Cumulative duration above threshold (h).
    """
    if len(te_profile_pct) != len(times_h):
        raise ValueError("te_profile_pct and times_h must have the same length")
    if len(times_h) < 2:
        return 0.0

    total_duration = 0.0
    for i in range(len(times_h) - 1):
        dt = times_h[i + 1] - times_h[i]
        occ_start = te_profile_pct[i]
        occ_end = te_profile_pct[i + 1]

        both_above = occ_start >= threshold_pct and occ_end >= threshold_pct
        both_below = occ_start < threshold_pct and occ_end < threshold_pct

        if both_above:
            total_duration += dt
        elif not both_below:
            # Linear interpolation to find crossing point
            # occ(t) = occ_start + (occ_end - occ_start) * (t - t_i) / dt
            # Solve occ(t_cross) = threshold_pct
            frac = (threshold_pct - occ_start) / (occ_end - occ_start)
            t_cross = times_h[i] + frac * dt
            if occ_start >= threshold_pct:
                # Falling: above from t_i to t_cross
                total_duration += t_cross - times_h[i]
            else:
                # Rising: above from t_cross to t_{i+1}
                total_duration += times_h[i + 1] - t_cross

    return total_duration
