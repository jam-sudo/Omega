"""Phase 1081 — Drug Intestinal Secretion (P-gp / BCRP Efflux).

3-compartment enterocyte model simulating the impact of intestinal
efflux transporters (P-gp, BCRP) on oral bioavailability.
"""

from __future__ import annotations

from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Result dataclass (plain – contains list fields)
# ---------------------------------------------------------------------------


@dataclass
class IntestinalEffluxResult:
    """Simulation result from intestinal efflux model."""

    drug_name: str
    dose_mg: float
    times_h: list
    a_lumen_mg: list
    a_enterocyte_mg: list
    a_portal_mg: list
    f_absorbed: float
    f_effluxed: float
    f_metabolized_gut: float
    auc_portal: float
    efflux_ratio: float
    bioavailability_impact: str
    inhibitor_benefit: float
    notes: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _trapz(times: list, values: list) -> float:
    """Trapezoidal integration."""
    auc = 0.0
    for i in range(1, len(times)):
        dt = times[i] - times[i - 1]
        auc += 0.5 * (values[i] + values[i - 1]) * dt
    return auc


def _compute_efflux_k(
    pgp_expression: float,
    bcrp_expression: float,
    inhibitor_fraction: float,
) -> float:
    """Compute apparent first-order efflux rate constant (1/h)."""
    return 0.3 * (pgp_expression + bcrp_expression * 0.5) * (1.0 - inhibitor_fraction)


def _run_simulation(
    dose_mg: float,
    k_enter: float,
    k_efflux: float,
    k_basolateral: float,
    k_met_ent: float,
    ke_portal: float,
    t_end_h: float,
    dt_h: float,
) -> tuple:
    """Forward-Euler ODE integration.

    Returns (times, a_lumen, a_enterocyte, a_portal, cumulative_efflux, cumulative_met).
    """
    n_steps = int(round(t_end_h / dt_h)) + 1
    times = [0.0] * n_steps
    a_lumen = [0.0] * n_steps
    a_ent = [0.0] * n_steps
    a_portal = [0.0] * n_steps

    # Tracking mass flows for fractions
    cumulative_efflux = [0.0] * n_steps  # drug returned to lumen from ent
    cumulative_met = [0.0] * n_steps  # drug metabolised in gut wall

    a_lumen[0] = dose_mg
    a_ent[0] = 0.0
    a_portal[0] = 0.0

    for i in range(1, n_steps):
        t_prev = times[i - 1]
        al = a_lumen[i - 1]
        ae = a_ent[i - 1]
        ap = a_portal[i - 1]

        times[i] = t_prev + dt_h

        # Rates (mg/h)
        r_enter = k_enter * al
        r_efflux = k_efflux * ae
        r_basolateral = k_basolateral * ae
        r_met = k_met_ent * ae
        r_portal_drain = ke_portal * ap

        # ODEs
        dal = -r_enter + r_efflux
        dae = r_enter - r_efflux - r_basolateral - r_met
        dap = r_basolateral - r_portal_drain

        a_lumen[i] = max(0.0, al + dal * dt_h)
        a_ent[i] = max(0.0, ae + dae * dt_h)
        a_portal[i] = max(0.0, ap + dap * dt_h)

        cumulative_efflux[i] = cumulative_efflux[i - 1] + r_efflux * dt_h
        cumulative_met[i] = cumulative_met[i - 1] + r_met * dt_h

    return times, a_lumen, a_ent, a_portal, cumulative_efflux[-1], cumulative_met[-1]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def simulate_intestinal_efflux(
    drug_name: str,
    dose_mg: float,
    pgp_expression: float = 1.0,
    bcrp_expression: float = 1.0,
    inhibitor_fraction: float = 0.0,
    k_enter: float = 0.5,
    k_basolateral: float = 0.1,
    k_met_ent: float = 0.05,
    t_end_h: float = 8.0,
    dt_h: float = 0.05,
) -> IntestinalEffluxResult:
    """Simulate intestinal efflux transporter impact on oral bioavailability.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    dose_mg:
        Oral dose (mg).
    pgp_expression:
        P-gp expression level; 1.0 = normal.
    bcrp_expression:
        BCRP expression level; 1.0 = normal.
    inhibitor_fraction:
        Fraction inhibition by efflux inhibitor (0–1).
    k_enter:
        Passive apical permeability rate constant (1/h).
    k_basolateral:
        Basolateral serosal absorption rate constant (1/h).
    k_met_ent:
        Gut-wall CYP3A4 first-pass metabolic rate constant (1/h).
    t_end_h:
        Simulation end time (h).
    dt_h:
        Forward-Euler step size (h).

    Returns
    -------
    IntestinalEffluxResult
    """
    # Validation
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if pgp_expression < 0:
        raise ValueError("pgp_expression must be >= 0")
    if bcrp_expression < 0:
        raise ValueError("bcrp_expression must be >= 0")
    if not (0.0 <= inhibitor_fraction <= 1.0):
        raise ValueError("inhibitor_fraction must be in [0, 1]")

    ke_portal = 0.5  # 1/h, portal vein drainage

    k_efflux = _compute_efflux_k(pgp_expression, bcrp_expression, inhibitor_fraction)

    times, a_lumen, a_ent, a_portal, total_effluxed, total_met = _run_simulation(
        dose_mg=dose_mg,
        k_enter=k_enter,
        k_efflux=k_efflux,
        k_basolateral=k_basolateral,
        k_met_ent=k_met_ent,
        ke_portal=ke_portal,
        t_end_h=t_end_h,
        dt_h=dt_h,
    )

    # Drug that entered the enterocyte over simulation
    # Total entering = dose_mg - a_lumen[-1] + total_effluxed
    # But simpler: use mass balance from lumen side
    # amount_entered_enterocyte = dose_mg - a_lumen[-1] + total_effluxed  (wrong sign)
    # amount_entered_enterocyte = k_enter integral = dose_mg - a_lumen[-1] + total_effluxed
    # More precisely: integral(k_enter * a_lumen dt) = dose_mg - a_lumen[-1] + total_effluxed
    # Just use what leaves lumen: dose_mg - a_lumen[-1] = net entering - wait:
    # dA_lumen = -k_enter*al + k_efflux*ae => dose - al_end = integral(k_enter*al) - total_effluxed
    # integral(k_enter*al) = dose - al_end + total_effluxed
    total_entered_ent = dose_mg - a_lumen[-1] + total_effluxed

    # Amount absorbed = basolateral flux = dose that went to portal (+ remaining in portal)
    # Use: f_absorbed = (amount that reached portal before portal drain) / dose
    # Actually portal accumulation + what drained out = total basolateral flux
    # mass balance: portal drain = integral(ke_portal * a_portal)
    # We track a_portal trajectory so AUC * ke_portal = total drained
    auc_portal = _trapz(times, a_portal)
    total_drained_portal = ke_portal * auc_portal + a_portal[-1]
    # total_drained_portal ≈ total basolateral flux (absorbed into systemic)

    f_absorbed = min(1.0, total_drained_portal / dose_mg) if dose_mg > 0 else 0.0
    f_effluxed = min(1.0, total_effluxed / dose_mg) if dose_mg > 0 else 0.0
    f_metabolized_gut = min(1.0, total_met / dose_mg) if dose_mg > 0 else 0.0

    # Efflux ratio: effluxed / entered enterocyte
    efflux_ratio = (
        min(1.0, total_effluxed / total_entered_ent) if total_entered_ent > 1e-12 else 0.0
    )

    # Bioavailability impact based on efflux_ratio
    if efflux_ratio > 0.5:
        bioavailability_impact = "high"
    elif efflux_ratio >= 0.2:
        bioavailability_impact = "moderate"
    else:
        bioavailability_impact = "low"

    # Inhibitor benefit: compare AUC portal with 90% inhibition vs current
    if inhibitor_fraction < 0.9:
        k_efflux_inh = _compute_efflux_k(pgp_expression, bcrp_expression, 0.9)
        _, _, _, a_portal_inh, _, _ = _run_simulation(
            dose_mg=dose_mg,
            k_enter=k_enter,
            k_efflux=k_efflux_inh,
            k_basolateral=k_basolateral,
            k_met_ent=k_met_ent,
            ke_portal=ke_portal,
            t_end_h=t_end_h,
            dt_h=dt_h,
        )
        auc_inh = _trapz(times, a_portal_inh)
        inhibitor_benefit = (auc_inh / auc_portal - 1.0) if auc_portal > 1e-12 else 0.0
    else:
        inhibitor_benefit = 0.0

    notes = (
        f"k_efflux={k_efflux:.3f}/h; efflux_ratio={efflux_ratio:.3f}; "
        f"impact={bioavailability_impact}"
    )

    return IntestinalEffluxResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        times_h=times,
        a_lumen_mg=a_lumen,
        a_enterocyte_mg=a_ent,
        a_portal_mg=a_portal,
        f_absorbed=f_absorbed,
        f_effluxed=f_effluxed,
        f_metabolized_gut=f_metabolized_gut,
        auc_portal=auc_portal,
        efflux_ratio=efflux_ratio,
        bioavailability_impact=bioavailability_impact,
        inhibitor_benefit=inhibitor_benefit,
        notes=notes,
    )


def assess_pgp_impact(
    drug_name: str,
    dose_mg: float,
    pgp_levels: list,
) -> list:
    """Run simulations across different P-gp expression levels.

    Parameters
    ----------
    drug_name:
        Drug name.
    dose_mg:
        Dose (mg).
    pgp_levels:
        List of P-gp expression values to evaluate.

    Returns
    -------
    list[IntestinalEffluxResult] sorted by f_absorbed descending.
    """
    results = [
        simulate_intestinal_efflux(drug_name, dose_mg, pgp_expression=lvl) for lvl in pgp_levels
    ]
    results.sort(key=lambda r: r.f_absorbed, reverse=True)
    return results
