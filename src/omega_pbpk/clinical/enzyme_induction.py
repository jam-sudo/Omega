"""Enzyme induction modeling — time-dependent CYP induction via PXR/CAR pathway.

Implements analytical and numerical (Euler) solutions for enzyme induction
kinetics using a two-compartment synthesis/degradation model.

Analytical solution (constant inducer concentration):
    E(t) = E_ss - (E_ss - E_0) * exp(-kdeg * t)
    E_ss = (ksyn / kdeg) * (1 + Emax * C / (EC50 + C))

Auto-induction (coupled ODE system, Euler integration):
    dC/dt = -CL(t) * C / Vd
    dE/dt = ksyn * (1 + Emax*C/(EC50+C)) - kdeg * E
    CL(t) = CL_baseline * E / E_baseline

References:
- Obach et al. (2007) Drug Metab Dispos 35:246–255
- FDA Drug Interaction Guidance (2020)
- Yoshida et al. (2012) Clin Pharmacokinet 51:281–301
"""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = [
    "InductionParameters",
    "InductionTimePoint",
    "InductionTimeCourse",
    "AutoInductionResult",
    "NetDDIEffect",
    "compute_ksyn",
    "steady_state_induction_factor",
    "time_to_steady_state",
    "simulate_induction_time_course",
    "simulate_auto_induction",
    "compute_net_ddi_effect",
    "format_induction_summary",
]

# Default CYP3A4 kdeg: 0.0005 /min × 60 = 0.03 /h (Obach et al. 2007)
KDEG_CYP3A4_DEFAULT_PER_H: float = 0.03

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class InductionParameters:
    """Parameters for enzyme induction simulation."""

    inducer_name: str
    target_enzyme: str
    emax: float
    ec50_uM: float
    inducer_conc_uM: float
    kdeg_h: float = KDEG_CYP3A4_DEFAULT_PER_H
    baseline_activity: float = 1.0


@dataclass(frozen=True)
class InductionTimePoint:
    """Single time point in an induction time course."""

    time_h: float
    enzyme_activity: float
    induction_factor: float
    fractional_turnover: float


@dataclass(frozen=True)
class InductionTimeCourse:
    """Complete enzyme induction time course (analytical solution)."""

    inducer_name: str
    target_enzyme: str
    emax: float
    ec50_uM: float
    inducer_conc_uM: float
    kdeg_h: float
    ksyn_h: float
    steady_state_fold: float
    time_to_90pct_ss_h: float
    time_points: tuple[InductionTimePoint, ...]
    time_h: tuple[float, ...]
    enzyme_activity: tuple[float, ...]
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class AutoInductionResult:
    """Auto-induction result: coupled PK-enzyme ODE (Euler integration)."""

    drug_name: str
    target_enzyme: str
    emax: float
    ec50_uM: float
    initial_conc_uM: float
    kdeg_h: float
    cl_baseline_L_per_h: float
    cl_induced_L_per_h: float
    half_life_baseline_h: float
    half_life_induced_h: float
    fold_cl_change: float
    time_h: tuple[float, ...]
    concentration_uM: tuple[float, ...]
    enzyme_activity: tuple[float, ...]
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class NetDDIEffect:
    """Net DDI effect combining inhibition (R1) and induction (fold)."""

    target_enzyme: str
    inhibition_factor: float
    induction_factor: float
    net_effect: float
    dominant_mechanism: str
    clinical_relevance: str
    warnings: tuple[str, ...]


# ---------------------------------------------------------------------------
# Core calculation functions
# ---------------------------------------------------------------------------


def compute_ksyn(kdeg_h: float, baseline_activity: float = 1.0) -> float:
    """Compute synthesis rate constant from kdeg and baseline activity.

    At uninduced steady state: ksyn = kdeg * E_baseline
    """
    return kdeg_h * baseline_activity


def steady_state_induction_factor(emax: float, conc_uM: float, ec50_uM: float) -> float:
    """Compute steady-state induction fold change (Hill equation).

    fold = 1 + Emax * C / (EC50 + C)
    """
    ec50 = max(ec50_uM, 1e-12)
    return 1.0 + emax * conc_uM / (ec50 + conc_uM)


def time_to_steady_state(kdeg_h: float, fraction: float = 0.9) -> float:
    """Compute time to reach *fraction* of steady-state.

    t = -ln(1 - fraction) / kdeg
    """
    kdeg = max(kdeg_h, 1e-12)
    frac = min(max(fraction, 1e-9), 1.0 - 1e-9)
    return -math.log(1.0 - frac) / kdeg


# ---------------------------------------------------------------------------
# Simulation functions
# ---------------------------------------------------------------------------


def simulate_induction_time_course(
    params: InductionParameters,
    t_end_h: float = 336.0,
    n_points: int = 200,
) -> InductionTimeCourse:
    """Simulate enzyme induction time course — analytical solution.

    E(t) = E_ss - (E_ss - E_0) * exp(-kdeg * t)
    E_ss = (ksyn / kdeg) * (1 + Emax * C / (EC50 + C))

    Args:
        params: InductionParameters with inducer properties.
        t_end_h: Simulation end time (hours); default 336 = 14 days.
        n_points: Number of time points (must be >= 2).

    Returns:
        InductionTimeCourse with per-point and summary data.
    """
    warnings: list[str] = []

    kdeg = params.kdeg_h
    ksyn = compute_ksyn(kdeg, params.baseline_activity)
    e0 = params.baseline_activity

    ss_fold = steady_state_induction_factor(params.emax, params.inducer_conc_uM, params.ec50_uM)
    e_ss = e0 * ss_fold
    t90 = time_to_steady_state(kdeg, fraction=0.9)

    if params.emax <= 0:
        warnings.append("emax <= 0: no induction expected")
    if params.inducer_conc_uM <= 0:
        warnings.append("inducer_conc_uM <= 0: no induction will occur")

    n = max(n_points, 2)
    dt = t_end_h / (n - 1)

    time_list: list[float] = []
    activity_list: list[float] = []
    tp_list: list[InductionTimePoint] = []

    for i in range(n):
        t = i * dt
        e_t = e_ss - (e_ss - e0) * math.exp(-kdeg * t)
        induction_factor = e_t / max(e0, 1e-12)
        # Fraction of original enzyme pool replaced by time t
        fractional_turnover = 1.0 - math.exp(-kdeg * t)

        time_list.append(round(t, 6))
        activity_list.append(round(e_t, 8))
        tp_list.append(
            InductionTimePoint(
                time_h=round(t, 6),
                enzyme_activity=round(e_t, 8),
                induction_factor=round(induction_factor, 6),
                fractional_turnover=round(fractional_turnover, 6),
            )
        )

    return InductionTimeCourse(
        inducer_name=params.inducer_name,
        target_enzyme=params.target_enzyme,
        emax=params.emax,
        ec50_uM=params.ec50_uM,
        inducer_conc_uM=params.inducer_conc_uM,
        kdeg_h=kdeg,
        ksyn_h=round(ksyn, 8),
        steady_state_fold=round(ss_fold, 6),
        time_to_90pct_ss_h=round(t90, 4),
        time_points=tuple(tp_list),
        time_h=tuple(time_list),
        enzyme_activity=tuple(activity_list),
        warnings=tuple(warnings),
    )


def simulate_auto_induction(
    params: InductionParameters,
    initial_conc_uM: float,
    cl_baseline_L_per_h: float,
    vd_L: float,
    t_end_h: float = 336.0,
    n_points: int = 200,
) -> AutoInductionResult:
    """Simulate auto-induction via Euler integration of coupled ODEs.

    ODEs:
        dC/dt = -CL(t) * C / Vd
        dE/dt = ksyn * (1 + Emax*C/(EC50+C)) - kdeg * E
        CL(t) = CL_baseline * E / E_baseline

    Args:
        params: InductionParameters describing the auto-inducing drug.
        initial_conc_uM: Initial drug concentration (µM).
        cl_baseline_L_per_h: Baseline (uninduced) clearance (L/h).
        vd_L: Volume of distribution (L).
        t_end_h: Simulation end time (hours).
        n_points: Number of Euler steps.

    Returns:
        AutoInductionResult with concentration and enzyme profiles.
    """
    warnings: list[str] = []

    kdeg = params.kdeg_h
    ksyn = compute_ksyn(kdeg, params.baseline_activity)
    e_base = params.baseline_activity

    if vd_L <= 0:
        warnings.append("vd_L <= 0; defaulting to 1.0 L")
        vd_L = 1.0

    n = max(n_points, 2)
    dt = t_end_h / (n - 1)

    c = initial_conc_uM
    e = e_base

    time_list: list[float] = [0.0]
    conc_list: list[float] = [round(c, 8)]
    act_list: list[float] = [round(e, 8)]

    for i in range(1, n):
        ind_factor = steady_state_induction_factor(params.emax, c, params.ec50_uM)
        cl_current = cl_baseline_L_per_h * e / max(e_base, 1e-12)

        dc = -(cl_current * c / max(vd_L, 1e-12)) * dt
        de = (ksyn * ind_factor - kdeg * e) * dt

        c = max(c + dc, 0.0)
        e = max(e + de, 0.0)

        time_list.append(round(i * dt, 6))
        conc_list.append(round(c, 8))
        act_list.append(round(e, 8))

    # Induced CL: average over last 10% of time points
    n_avg = max(1, n // 10)
    cl_induced = cl_baseline_L_per_h * (sum(act_list[-n_avg:]) / n_avg / max(e_base, 1e-12))

    half_life_baseline = math.log(2) / max(cl_baseline_L_per_h / max(vd_L, 1e-12), 1e-12)
    half_life_induced = math.log(2) / max(cl_induced / max(vd_L, 1e-12), 1e-12)
    fold_cl = cl_induced / max(cl_baseline_L_per_h, 1e-12)

    if fold_cl > 10:
        warnings.append(f"Large CL fold change ({fold_cl:.1f}x); consider smaller time steps")

    return AutoInductionResult(
        drug_name=params.inducer_name,
        target_enzyme=params.target_enzyme,
        emax=params.emax,
        ec50_uM=params.ec50_uM,
        initial_conc_uM=initial_conc_uM,
        kdeg_h=kdeg,
        cl_baseline_L_per_h=round(cl_baseline_L_per_h, 6),
        cl_induced_L_per_h=round(cl_induced, 6),
        half_life_baseline_h=round(half_life_baseline, 4),
        half_life_induced_h=round(half_life_induced, 4),
        fold_cl_change=round(fold_cl, 4),
        time_h=tuple(time_list),
        concentration_uM=tuple(conc_list),
        enzyme_activity=tuple(act_list),
        warnings=tuple(warnings),
    )


def compute_net_ddi_effect(
    inhibition_r1: float,
    induction_fold: float,
    fm_enzyme: float = 1.0,
    target_enzyme: str = "CYP3A4",
) -> NetDDIEffect:
    """Compute net DDI AUCR combining reversible inhibition and induction.

    Net AUCR = fm / (R1 * induction_fold) + (1 - fm)

    FDA clinical relevance (AUCR thresholds):
        < 1.25  → no_interaction
        1.25–2  → weak
        2–5     → moderate
        > 5     → strong

    Args:
        inhibition_r1: Static inhibition R1 = 1 + [I]/Ki (1.0 = no inhibition).
        induction_fold: Fold induction of enzyme activity (1.0 = no induction).
        fm_enzyme: Fraction of victim drug metabolized by target enzyme.
        target_enzyme: Name of the target CYP enzyme.

    Returns:
        NetDDIEffect with combined AUCR and FDA classification.
    """
    warnings: list[str] = []

    r1 = max(inhibition_r1, 1e-12)
    ind_fold = max(induction_fold, 1e-12)
    fm = min(max(fm_enzyme, 0.0), 1.0)

    if fm_enzyme < 0.0 or fm_enzyme > 1.0:
        warnings.append(f"fm_enzyme={fm_enzyme:.3f} outside [0, 1]; clamped to [{fm:.3f}]")

    net_effect = fm / (r1 * ind_fold) + (1.0 - fm)

    # Determine dominant mechanism
    if r1 > 1.0 and ind_fold > 1.0:
        inhibition_component = fm * (r1 - 1.0)
        induction_component = fm * (1.0 - 1.0 / ind_fold)
        if inhibition_component > induction_component:
            dominant_mechanism = "inhibition"
        elif induction_component > inhibition_component:
            dominant_mechanism = "induction"
        else:
            dominant_mechanism = "balanced"
        warnings.append(
            f"Both inhibition (R1={r1:.2f}) and induction ({ind_fold:.2f}x) present; "
            f"net AUCR={net_effect:.3f}"
        )
    elif r1 > 1.0:
        dominant_mechanism = "inhibition"
    elif ind_fold > 1.0:
        dominant_mechanism = "induction"
    else:
        dominant_mechanism = "none"

    # FDA threshold classification
    if net_effect < 1.25:
        clinical_relevance = "no_interaction"
    elif net_effect < 2.0:
        clinical_relevance = "weak"
    elif net_effect < 5.0:
        clinical_relevance = "moderate"
    else:
        clinical_relevance = "strong"

    return NetDDIEffect(
        target_enzyme=target_enzyme,
        inhibition_factor=round(r1, 4),
        induction_factor=round(ind_fold, 4),
        net_effect=round(net_effect, 4),
        dominant_mechanism=dominant_mechanism,
        clinical_relevance=clinical_relevance,
        warnings=tuple(warnings),
    )


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------


def format_induction_summary(
    result: InductionTimeCourse | AutoInductionResult,
) -> str:
    """Format induction simulation result as a human-readable summary."""
    if isinstance(result, InductionTimeCourse):
        lines = [
            f"Enzyme Induction: {result.inducer_name} on {result.target_enzyme}",
            "=" * 55,
            f"  Emax:              {result.emax:.2f}",
            f"  EC50:              {result.ec50_uM:.2f} µM",
            f"  Inducer conc:      {result.inducer_conc_uM:.2f} µM",
            f"  kdeg:              {result.kdeg_h:.4f} /h",
            f"  ksyn:              {result.ksyn_h:.6f} /h",
            f"  Steady-state fold: {result.steady_state_fold:.3f}x",
            f"  Time to 90% SS:    {result.time_to_90pct_ss_h:.1f} h",
        ]
        if result.warnings:
            lines.append("\nWarnings:")
            for w in result.warnings:
                lines.append(f"  ! {w}")
        return "\n".join(lines)

    # AutoInductionResult
    lines = [
        f"Auto-Induction: {result.drug_name} on {result.target_enzyme}",
        "=" * 55,
        f"  Emax:              {result.emax:.2f}",
        f"  EC50:              {result.ec50_uM:.2f} µM",
        f"  Initial conc:      {result.initial_conc_uM:.2f} µM",
        f"  kdeg:              {result.kdeg_h:.4f} /h",
        f"  CL baseline:       {result.cl_baseline_L_per_h:.4f} L/h",
        f"  CL induced:        {result.cl_induced_L_per_h:.4f} L/h",
        f"  CL fold change:    {result.fold_cl_change:.3f}x",
        f"  t½ baseline:       {result.half_life_baseline_h:.2f} h",
        f"  t½ induced:        {result.half_life_induced_h:.2f} h",
    ]
    if result.warnings:
        lines.append("\nWarnings:")
        for w in result.warnings:
            lines.append(f"  ! {w}")
    return "\n".join(lines)
