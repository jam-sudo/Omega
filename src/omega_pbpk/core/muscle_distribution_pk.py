"""Phase 1082 — Muscle Compartment Drug Distribution.

4-compartment PK model: Plasma -> Skeletal muscle -> Cardiac muscle -> (implicit other).
Relevant for statins (myopathy), digoxin (cardiac distribution), local anesthetics.
"""

from __future__ import annotations

from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Result dataclass (plain – contains list fields)
# ---------------------------------------------------------------------------


@dataclass
class MuscleDistributionResult:
    """Result from muscle distribution PK simulation."""

    drug_name: str
    dose_mg: float
    logP: float
    times_h: list
    c_plasma_mg_L: list
    c_skeletal_mg_L: list
    c_cardiac_mg_L: list
    cmax_plasma: float
    cmax_skeletal: float
    cmax_cardiac: float
    auc_plasma: float
    auc_skeletal: float
    auc_cardiac: float
    cardiac_to_plasma_ratio: float
    myopathy_risk: str
    cardiac_exposure_risk: str
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


def _kp_skeletal(logP: float) -> float:
    """Skeletal muscle Kp from logP (dimensionless partition coefficient)."""
    raw = 0.5 + 0.3 * logP
    return max(0.1, min(10.0, raw))


def _kp_cardiac(logP: float) -> float:
    """Cardiac muscle Kp from logP (slightly higher than skeletal)."""
    raw = 1.0 + 0.5 * logP
    return max(0.5, min(20.0, raw))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def simulate_muscle_distribution(
    drug_name: str,
    dose_mg: float,
    logP: float = 2.0,
    cl_total_L_per_h: float = 5.0,
    v_plasma_L: float = 3.0,
    co_L_per_h: float = 300.0,
    t_end_h: float = 24.0,
    dt_h: float = 0.01,
) -> MuscleDistributionResult:
    """Simulate drug distribution into skeletal and cardiac muscle.

    IV bolus administration.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    dose_mg:
        IV bolus dose (mg).
    logP:
        Lipophilicity (log partition coefficient).
    cl_total_L_per_h:
        Total body clearance (L/h).
    v_plasma_L:
        Plasma volume (L).
    co_L_per_h:
        Cardiac output (L/h).
    t_end_h:
        Simulation end time (h).
    dt_h:
        Forward-Euler step size (h).

    Returns
    -------
    MuscleDistributionResult
    """
    # Validation
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if not (-5.0 <= logP <= 10.0):
        raise ValueError("logP must be in [-5, 10]")
    if cl_total_L_per_h <= 0:
        raise ValueError("cl_total_L_per_h must be > 0")
    if v_plasma_L <= 0:
        raise ValueError("v_plasma_L must be > 0")

    # Fixed volumes
    V_sk = 28.0  # L skeletal muscle
    V_card = 0.3  # L cardiac muscle

    # Flow rates (fraction of cardiac output)
    Q_sk = 0.15 * co_L_per_h
    Q_card = 0.04 * co_L_per_h

    # Tissue-to-plasma partition coefficients
    kp_sk = _kp_skeletal(logP)
    kp_card = _kp_cardiac(logP)

    # Ensure numerical stability: stiffest eigenvalue is Q_card/(V_card*kp_card)
    stiff_lambda = Q_card / (V_card * kp_card)
    dt_h = min(dt_h, 1.5 / stiff_lambda)  # keep |lambda * dt| < 1.5

    # Initial conditions (IV bolus)
    cp0 = dose_mg / v_plasma_L
    n_steps = int(round(t_end_h / dt_h)) + 1
    times = [0.0] * n_steps
    c_plasma = [0.0] * n_steps
    c_sk = [0.0] * n_steps
    c_card = [0.0] * n_steps

    c_plasma[0] = cp0
    c_sk[0] = 0.0
    c_card[0] = 0.0

    for i in range(1, n_steps):
        cp = c_plasma[i - 1]
        csk = c_sk[i - 1]
        cca = c_card[i - 1]

        times[i] = times[i - 1] + dt_h

        # Distribution driving forces
        j_sk = Q_sk * (cp - csk / kp_sk)  # net flux plasma->skeletal (mg/h)
        j_card = Q_card * (cp - cca / kp_card)  # net flux plasma->cardiac (mg/h)

        # ODEs (concentrations)
        dcp = -(cl_total_L_per_h / v_plasma_L) * cp - (j_sk / v_plasma_L) - (j_card / v_plasma_L)
        dcsk = j_sk / V_sk
        dcca = j_card / V_card

        c_plasma[i] = max(0.0, cp + dcp * dt_h)
        c_sk[i] = max(0.0, csk + dcsk * dt_h)
        c_card[i] = max(0.0, cca + dcca * dt_h)

    # PK metrics
    cmax_plasma = max(c_plasma)
    cmax_skeletal = max(c_sk)
    cmax_cardiac = max(c_card)

    auc_plasma = _trapz(times, c_plasma)
    auc_skeletal = _trapz(times, c_sk)
    auc_cardiac = _trapz(times, c_card)

    cardiac_to_plasma_ratio = auc_cardiac / auc_plasma if auc_plasma > 1e-12 else 0.0

    # Myopathy risk based on skeletal AUC / plasma AUC ratio
    sk_ratio = auc_skeletal / auc_plasma if auc_plasma > 1e-12 else 0.0
    if sk_ratio > 5.0:
        myopathy_risk = "high"
    elif sk_ratio >= 2.0:
        myopathy_risk = "moderate"
    else:
        myopathy_risk = "low"

    # Cardiac exposure risk based on cardiac_to_plasma_ratio
    if cardiac_to_plasma_ratio > 3.0:
        cardiac_exposure_risk = "high"
    elif cardiac_to_plasma_ratio >= 1.0:
        cardiac_exposure_risk = "moderate"
    else:
        cardiac_exposure_risk = "low"

    notes = (
        f"Kp_sk={kp_sk:.2f}, Kp_card={kp_card:.2f}; "
        f"sk_ratio={sk_ratio:.2f}; cardiac_ratio={cardiac_to_plasma_ratio:.2f}"
    )

    return MuscleDistributionResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        logP=logP,
        times_h=times,
        c_plasma_mg_L=c_plasma,
        c_skeletal_mg_L=c_sk,
        c_cardiac_mg_L=c_card,
        cmax_plasma=cmax_plasma,
        cmax_skeletal=cmax_skeletal,
        cmax_cardiac=cmax_cardiac,
        auc_plasma=auc_plasma,
        auc_skeletal=auc_skeletal,
        auc_cardiac=auc_cardiac,
        cardiac_to_plasma_ratio=cardiac_to_plasma_ratio,
        myopathy_risk=myopathy_risk,
        cardiac_exposure_risk=cardiac_exposure_risk,
        notes=notes,
    )


def compare_logp_muscle(
    drug_name: str,
    dose_mg: float,
    logp_list: list,
) -> list:
    """Compare muscle distribution across different logP values.

    Parameters
    ----------
    drug_name:
        Drug name.
    dose_mg:
        Dose (mg).
    logp_list:
        List of logP values to evaluate.

    Returns
    -------
    list[MuscleDistributionResult] sorted by cardiac_to_plasma_ratio descending.
    """
    results = [simulate_muscle_distribution(drug_name, dose_mg, logP=lp) for lp in logp_list]
    results.sort(key=lambda r: r.cardiac_to_plasma_ratio, reverse=True)
    return results
