"""Compare PK simulations across different scenarios and produce summary statistics."""

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class PKScenario:
    name: str
    times_h: tuple
    c_plasma_mg_L: tuple
    cmax_mg_L: float
    tmax_h: float
    auc_mg_h_L: float
    t_half_h: float
    dose_mg: float
    notes: str


@dataclass(frozen=True)
class ComparisonResult:
    scenario_names: tuple
    n_scenarios: int
    reference_name: str
    cmax_ratios: dict  # {name: cmax/ref_cmax}
    auc_ratios: dict  # {name: auc/ref_auc}
    tmax_differences_h: dict  # {name: tmax - ref_tmax}
    best_scenario: str  # scenario with highest AUC
    worst_scenario: str  # scenario with lowest AUC
    auc_cv_pct: float  # between-scenario variability
    recommendation: str
    notes: str


def simulate_scenario(
    name: str,
    dose_mg: float,
    cl_L_per_h: float,
    vd_L: float,
    route: str = "oral",
    ka_per_h: float = 1.0,
    f_oral: float = 1.0,
    t_end_h: float = 24.0,
    dt_h: float = 0.1,
) -> PKScenario:
    """Simulate a 1-cpt PK scenario and return PKScenario."""
    ke = cl_L_per_h / vd_L  # elimination rate constant (h^-1)
    t_half_h = math.log(2.0) / ke if ke > 0 else float("inf")

    # Build time grid
    n_steps = int(t_end_h / dt_h) + 1
    times = [i * dt_h for i in range(n_steps)]

    concentrations = []
    for t in times:
        if route == "iv":
            c = (dose_mg / vd_L) * math.exp(-ke * t)
        else:
            # Oral 1-cpt analytical: C = f*D*ka/(Vd*(ka-ke))*(exp(-ke*t) - exp(-ka*t))
            if abs(ka_per_h - ke) < 1e-9:
                # Avoid division by zero: use limit
                c = (f_oral * dose_mg / vd_L) * ke * t * math.exp(-ke * t)
            else:
                c = (
                    f_oral
                    * dose_mg
                    * ka_per_h
                    / (vd_L * (ka_per_h - ke))
                    * (math.exp(-ke * t) - math.exp(-ka_per_h * t))
                )
        concentrations.append(max(0.0, c))

    # Find Cmax and Tmax
    cmax = max(concentrations)
    tmax_idx = concentrations.index(cmax)
    tmax_h = times[tmax_idx]

    # Trapezoidal AUC
    auc = 0.0
    for i in range(1, len(times)):
        auc += 0.5 * (concentrations[i - 1] + concentrations[i]) * (times[i] - times[i - 1])

    notes = f"route={route}, ke={ke:.4f}/h, t_half={t_half_h:.2f}h"

    return PKScenario(
        name=name,
        times_h=tuple(times),
        c_plasma_mg_L=tuple(concentrations),
        cmax_mg_L=cmax,
        tmax_h=tmax_h,
        auc_mg_h_L=auc,
        t_half_h=t_half_h,
        dose_mg=dose_mg,
        notes=notes,
    )


def compare_scenarios(
    scenarios: list,
    reference_name: str | None = None,
) -> ComparisonResult:
    """Compare multiple PK scenarios against a reference."""
    if len(scenarios) < 2:
        raise ValueError("At least 2 scenarios are required for comparison")

    # Determine reference
    if reference_name is None:
        reference_name = scenarios[0].name

    # Check reference exists
    ref_scenario = None
    for s in scenarios:
        if s.name == reference_name:
            ref_scenario = s
            break

    if ref_scenario is None:
        raise ValueError(f"Reference scenario '{reference_name}' not found in scenarios list")

    ref_cmax = ref_scenario.cmax_mg_L
    ref_auc = ref_scenario.auc_mg_h_L
    ref_tmax = ref_scenario.tmax_h

    # Compute ratios and differences
    cmax_ratios = {}
    auc_ratios = {}
    tmax_differences = {}

    for s in scenarios:
        if ref_cmax > 0:
            cmax_ratios[s.name] = s.cmax_mg_L / ref_cmax
        else:
            cmax_ratios[s.name] = 0.0 if s.cmax_mg_L == 0 else float("inf")

        if ref_auc > 0:
            auc_ratios[s.name] = s.auc_mg_h_L / ref_auc
        else:
            auc_ratios[s.name] = 0.0 if s.auc_mg_h_L == 0 else float("inf")

        tmax_differences[s.name] = s.tmax_h - ref_tmax

    # Find best and worst by AUC
    best_scenario = max(scenarios, key=lambda s: s.auc_mg_h_L).name
    worst_scenario = min(scenarios, key=lambda s: s.auc_mg_h_L).name

    # AUC CV%
    aucs = [s.auc_mg_h_L for s in scenarios]
    n = len(aucs)
    mean_auc = sum(aucs) / n
    if mean_auc > 0:
        variance = sum((a - mean_auc) ** 2 for a in aucs) / n
        std_auc = math.sqrt(variance)
        auc_cv_pct = std_auc / mean_auc * 100.0
    else:
        auc_cv_pct = 0.0

    # Recommendation
    best_auc = max(aucs)
    worst_auc = min(aucs)

    if auc_cv_pct < 20.0:
        recommendation = "scenarios are similar"
    elif best_auc > 2.0 * worst_auc:
        recommendation = "substantial differences"
    else:
        recommendation = "moderate differences"

    notes = (
        f"AUC range: {worst_auc:.2f}–{best_auc:.2f} mg·h/L; "
        f"CV={auc_cv_pct:.1f}%; reference={reference_name}"
    )

    return ComparisonResult(
        scenario_names=tuple(s.name for s in scenarios),
        n_scenarios=n,
        reference_name=reference_name,
        cmax_ratios=cmax_ratios,
        auc_ratios=auc_ratios,
        tmax_differences_h=tmax_differences,
        best_scenario=best_scenario,
        worst_scenario=worst_scenario,
        auc_cv_pct=auc_cv_pct,
        recommendation=recommendation,
        notes=notes,
    )


def rank_scenarios_by_auc(scenarios: list) -> list:
    """Return scenarios sorted by AUC descending."""
    return sorted(scenarios, key=lambda s: s.auc_mg_h_L, reverse=True)
