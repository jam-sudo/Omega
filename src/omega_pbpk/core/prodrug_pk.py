"""Prodrug activation and parent drug PK — Phase 464."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ProdrugPKResult:
    """Simulation result for prodrug + active drug PK."""

    prodrug_name: str
    active_name: str
    dose_mg: float
    times_h: list[float]
    c_prodrug_mg_L: list[float]
    c_active_mg_L: list[float]
    cmax_prodrug: float
    cmax_active: float
    tmax_active_h: float
    auc_prodrug: float
    auc_active: float
    f_activation_pct: float
    notes: str


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _trapezoid_auc(times: list[float], concs: list[float]) -> float:
    """Linear trapezoidal AUC."""
    auc = 0.0
    for i in range(1, len(times)):
        dt = times[i] - times[i - 1]
        auc += 0.5 * (concs[i] + concs[i - 1]) * dt
    return auc


# ---------------------------------------------------------------------------
# ODE simulation
# ---------------------------------------------------------------------------


def simulate_prodrug_pk(
    prodrug_name: str,
    active_name: str,
    dose_mg: float,
    ka_per_h: float,
    f_oral: float,
    cl_prodrug_L_per_h: float,
    vd_prodrug_L: float,
    kact_per_h: float,
    cl_active_L_per_h: float,
    vd_active_L: float,
    activation_site: str = "systemic",
    t_end_h: float = 24.0,
    dt_h: float = 0.1,
) -> ProdrugPKResult:
    """Simulate prodrug and active drug PK using forward Euler ODE.

    Compartments
    ------------
    For oral route:
        A_gut  : gut depot amount (mg)
        A_prod : prodrug plasma amount (mg)
        A_act  : active drug plasma amount (mg)

    ODE system (oral, systemic activation):
        dA_gut/dt  = -ka * A_gut
        dA_prod/dt = ka * f_oral * A_gut - (CL_prod/Vd_prod) * A_prod - kact * A_prod
        dA_act/dt  = kact * A_prod - (CL_act/Vd_act) * A_act

    For IV route (f_oral=1, ka not used):
        A_prod(0) = dose_mg
        dA_prod/dt = -(CL_prod/Vd_prod + kact) * A_prod
        dA_act/dt  = kact * A_prod - (CL_act/Vd_act) * A_act

    activation_site options: 'systemic', 'gut', 'liver'
    - 'gut': conversion fraction applied at gut wall; kact acts on gut amount
    - 'liver': first-pass activation; modeled as elevated kact with hepatic extraction
    - 'systemic': default, kact acts on systemic prodrug
    """
    # Input validation
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if ka_per_h < 0:
        raise ValueError("ka_per_h must be >= 0")
    if not (0.0 <= f_oral <= 1.0):
        raise ValueError("f_oral must be in [0, 1]")
    if cl_prodrug_L_per_h < 0:
        raise ValueError("cl_prodrug_L_per_h must be >= 0")
    if vd_prodrug_L <= 0:
        raise ValueError("vd_prodrug_L must be > 0")
    if kact_per_h < 0:
        raise ValueError("kact_per_h must be >= 0")
    if cl_active_L_per_h < 0:
        raise ValueError("cl_active_L_per_h must be >= 0")
    if vd_active_L <= 0:
        raise ValueError("vd_active_L must be > 0")
    if t_end_h <= 0:
        raise ValueError("t_end_h must be > 0")
    if dt_h <= 0:
        raise ValueError("dt_h must be > 0")
    if activation_site not in ("systemic", "gut", "liver"):
        raise ValueError("activation_site must be 'systemic', 'gut', or 'liver'")

    ke_prod = cl_prodrug_L_per_h / vd_prodrug_L
    ke_act = cl_active_L_per_h / vd_active_L

    # Adjust kact for activation site
    if activation_site == "liver":
        # Hepatic extraction: scale kact by liver blood flow factor (~1.5)
        eff_kact = kact_per_h * 1.5
    else:
        eff_kact = kact_per_h

    # Initial conditions (amounts in mg)
    # Oral: gut depot gets the full dose; prodrug/active start at 0
    # IV: prodrug plasma starts at dose (via f_oral fraction)
    if ka_per_h > 0:
        # Oral administration
        A_gut = dose_mg
        A_prod = 0.0
        A_act = 0.0
        is_oral = True
    else:
        # IV bolus
        A_gut = 0.0
        A_prod = dose_mg * f_oral
        A_act = 0.0
        is_oral = False

    times: list[float] = [0.0]
    c_prod_list: list[float] = [A_prod / vd_prodrug_L]
    c_act_list: list[float] = [A_act / vd_active_L]

    n_steps = max(1, round(t_end_h / dt_h))
    t = 0.0

    for _ in range(n_steps):
        if is_oral:
            # Gut absorption
            dA_gut = -ka_per_h * A_gut

            if activation_site == "gut":
                # Partial conversion in gut wall
                absorbed = ka_per_h * A_gut * f_oral
                converted_gut = absorbed * (eff_kact / (eff_kact + ke_prod + 1e-12))
                to_prodrug = absorbed - converted_gut
                dA_prod = to_prodrug - ke_prod * A_prod - eff_kact * A_prod
                dA_act = converted_gut + eff_kact * A_prod - ke_act * A_act
            else:
                # Systemic / liver activation
                dA_prod = ka_per_h * f_oral * A_gut - (ke_prod + eff_kact) * A_prod
                dA_act = eff_kact * A_prod - ke_act * A_act
        else:
            dA_gut = 0.0
            dA_prod = -(ke_prod + eff_kact) * A_prod
            dA_act = eff_kact * A_prod - ke_act * A_act

        A_gut += dA_gut * dt_h
        A_prod += dA_prod * dt_h
        A_act += dA_act * dt_h

        # Clamp to non-negative
        A_gut = max(A_gut, 0.0)
        A_prod = max(A_prod, 0.0)
        A_act = max(A_act, 0.0)

        t += dt_h
        times.append(round(t, 6))
        c_prod_list.append(A_prod / vd_prodrug_L)
        c_act_list.append(A_act / vd_active_L)

    cmax_prod = max(c_prod_list)
    cmax_act = max(c_act_list)
    tmax_act = times[c_act_list.index(cmax_act)]

    auc_prod = _trapezoid_auc(times, c_prod_list)
    auc_act = _trapezoid_auc(times, c_act_list)

    # f_activation_pct: fraction of prodrug dose converted to active
    # Estimated via AUC ratio × CL ratio
    # Total active generated = AUC_act * CL_act; total prodrug absorbed = dose * f_oral
    if cl_active_L_per_h > 0:
        total_active_mg = auc_act * cl_active_L_per_h
    else:
        total_active_mg = auc_act * ke_act * vd_active_L
    total_prodrug_absorbed = dose_mg * (f_oral if is_oral else 1.0)
    if total_prodrug_absorbed > 0:
        f_act_pct = total_active_mg / total_prodrug_absorbed * 100.0
    else:
        f_act_pct = 0.0
    f_act_pct = min(f_act_pct, 100.0)

    notes = (
        f"{prodrug_name} → {active_name}: "
        f"Cmax_active={cmax_act:.3f} mg/L at {tmax_act:.2f} h, "
        f"AUC_active={auc_act:.2f} mg·h/L, "
        f"activation={f_act_pct:.1f}% at site='{activation_site}'."
    )

    return ProdrugPKResult(
        prodrug_name=prodrug_name,
        active_name=active_name,
        dose_mg=dose_mg,
        times_h=times,
        c_prodrug_mg_L=c_prod_list,
        c_active_mg_L=c_act_list,
        cmax_prodrug=cmax_prod,
        cmax_active=cmax_act,
        tmax_active_h=tmax_act,
        auc_prodrug=auc_prod,
        auc_active=auc_act,
        f_activation_pct=f_act_pct,
        notes=notes,
    )


def compare_prodrug_vs_parent(
    prodrug_result: ProdrugPKResult,
    parent_result: ProdrugPKResult,
) -> dict:
    """Compare prodrug active-drug PK vs. reference parent drug PK.

    Parameters
    ----------
    prodrug_result : ProdrugPKResult
        Simulation result for the prodrug (compare active drug profile).
    parent_result : ProdrugPKResult
        Simulation result treated as the parent/reference drug (uses active profile).

    Returns
    -------
    dict with keys:
        auc_ratio          : AUC_prodrug_active / AUC_parent_active
        cmax_ratio         : Cmax_prodrug_active / Cmax_parent_active
        tmax_difference_h  : tmax_prodrug_active - tmax_parent_active
        notes              : interpretation string
    """
    if parent_result.auc_active == 0:
        raise ValueError("parent_result.auc_active must be > 0 for comparison")

    if parent_result.auc_active > 0:
        auc_ratio = prodrug_result.auc_active / parent_result.auc_active
    else:
        auc_ratio = float("nan")
    if parent_result.cmax_active > 0:
        cmax_ratio = prodrug_result.cmax_active / parent_result.cmax_active
    else:
        cmax_ratio = float("nan")
    tmax_diff = prodrug_result.tmax_active_h - parent_result.tmax_active_h

    if abs(auc_ratio - 1.0) < 0.05 and abs(cmax_ratio - 1.0) < 0.05:
        interpretation = "Prodrug and parent have equivalent active drug exposure."
    elif auc_ratio > 1.0:
        interpretation = f"Prodrug delivers {auc_ratio:.2f}x higher active AUC vs parent."
    else:
        interpretation = f"Prodrug delivers {auc_ratio:.2f}x lower active AUC vs parent."

    return {
        "auc_ratio": auc_ratio,
        "cmax_ratio": cmax_ratio,
        "tmax_difference_h": tmax_diff,
        "notes": interpretation,
    }


def prodrug_benefit_score(
    f_prodrug: float,
    f_parent: float,
    solubility_ratio: float,
    stability_ratio: float,
) -> float:
    """Compute a composite prodrug benefit score (0-100).

    Score reflects the advantage of using a prodrug over the parent.

    Parameters
    ----------
    f_prodrug : float
        Oral bioavailability of the prodrug (as active drug), [0, 1].
    f_parent : float
        Oral bioavailability of the parent drug, [0, 1].
    solubility_ratio : float
        Solubility of prodrug / solubility of parent (≥0). >1 = prodrug more soluble.
    stability_ratio : float
        Chemical stability of prodrug / parent (≥0). >1 = prodrug more stable.

    Returns
    -------
    float in [0, 100].
    """
    if not (0.0 <= f_prodrug <= 1.0):
        raise ValueError("f_prodrug must be in [0, 1]")
    if not (0.0 <= f_parent <= 1.0):
        raise ValueError("f_parent must be in [0, 1]")
    if solubility_ratio < 0:
        raise ValueError("solubility_ratio must be >= 0")
    if stability_ratio < 0:
        raise ValueError("stability_ratio must be >= 0")

    # Bioavailability advantage (40% weight)
    if f_parent > 0:
        ba_advantage = min((f_prodrug / f_parent), 3.0) / 3.0 * 100.0
    else:
        # Parent has zero BA → prodrug has infinite advantage
        ba_advantage = 100.0 if f_prodrug > 0 else 0.0

    # Solubility advantage (30% weight): score = min(ratio, 5) / 5 * 100
    sol_score = min(solubility_ratio, 5.0) / 5.0 * 100.0

    # Stability advantage (30% weight)
    stab_score = min(stability_ratio, 5.0) / 5.0 * 100.0

    score = 0.40 * ba_advantage + 0.30 * sol_score + 0.30 * stab_score
    return min(max(score, 0.0), 100.0)
