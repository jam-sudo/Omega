"""Adaptive dose finding algorithms — Phase 490.

Implements Bayesian and rule-based dose-escalation designs for Phase I
oncology trials:

- Classic 3+3 rule
- Bayesian Optimal Interval (BOIN) design (Liu & Yuan, 2015)
- MTD estimation via isotonic regression (pool-adjacent-violators)
- Simulation engine with reproducible seeding

References
----------
- Liu S & Yuan Y. JRSS-C. 2015;64(3):507-523. (BOIN)
- Storer BE. Biometrics. 1989;45(3):925-937. (3+3 origin)
- Le Tourneau C et al. J Natl Cancer Inst. 2009;101:708-720. (review)
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# BOIN design constants
# ---------------------------------------------------------------------------

_BOIN_PHI1 = 0.6  # lower phi multiplier
_BOIN_PHI2 = 1.4  # upper phi multiplier
_BOIN_ELIM_THRESHOLD = 0.95  # posterior probability threshold for elimination


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DoseFindingResult:
    """Result of a 3+3 dose-finding step.

    Attributes
    ----------
    recommended_dose_mg : Dose recommended for next cohort or as MTD (mg).
    n_dose_levels_tested : Number of dose levels evaluated so far.
    n_total_patients : Total patients enrolled across all levels.
    dlt_rates_observed : Observed DLT rate at each tested dose level.
    decision : 'escalate', 'stay', 'mtd_found', or 'deescalate'.
    notes : Free-text description of the decision rationale.
    """

    recommended_dose_mg: float
    n_dose_levels_tested: int
    n_total_patients: int
    dlt_rates_observed: list[float]
    decision: str
    notes: str


@dataclass
class DoseEscalationResult:
    """Result of a simulated dose escalation trial.

    Attributes
    ----------
    dose_levels : Dose levels used in the simulation (mg).
    n_patients_per_level : Patients enrolled at each dose level.
    dlt_counts : DLT events observed at each dose level.
    estimated_mtd : Estimated MTD (mg).
    true_dlt_rates : True DLT probabilities used in the simulation.
    notes : Free-text summary.
    """

    dose_levels: list[float] = field(default_factory=list)
    n_patients_per_level: list[int] = field(default_factory=list)
    dlt_counts: list[int] = field(default_factory=list)
    estimated_mtd: float = 0.0
    true_dlt_rates: list[float] = field(default_factory=list)
    notes: str = ""


# ---------------------------------------------------------------------------
# BOIN decision rule
# ---------------------------------------------------------------------------


def boin_decision(
    n_patients: int,
    n_dlt: int,
    target_toxicity: float = 0.25,
    lambda1: float | None = None,
    lambda2: float | None = None,
) -> str:
    """Return the BOIN dose-escalation decision for the current cohort.

    The BOIN design uses two empirically-derived decision boundaries:

        lambda1 = target * (1 - phi1) / (1 - target * phi1)
        lambda2 = target * (1 + phi2) / ((1 - target) * phi2 + 1)

    where phi1 = 0.6 and phi2 = 1.4 (Liu & Yuan 2015 defaults).

    Parameters
    ----------
    n_patients : Number of patients treated at the current dose.
    n_dlt : Number of DLTs observed at the current dose.
    target_toxicity : Target DLT rate (default 0.25 for oncology).
    lambda1 : Lower BOIN boundary (auto-calculated if None).
    lambda2 : Upper BOIN boundary (auto-calculated if None).

    Returns
    -------
    str
        'escalate'    — observed rate < lambda1
        'stay'        — lambda1 ≤ rate ≤ lambda2
        'deescalate'  — rate > lambda2 (but below elimination threshold)
        'eliminate'   — dose too toxic; remove from trial
    """
    if n_patients <= 0:
        raise ValueError(f"n_patients must be > 0, got {n_patients}.")
    if n_dlt < 0 or n_dlt > n_patients:
        raise ValueError(f"n_dlt must be in [0, {n_patients}], got {n_dlt}.")
    if not (0.0 < target_toxicity < 1.0):
        raise ValueError(f"target_toxicity must be in (0, 1), got {target_toxicity}.")

    phi = target_toxicity
    if lambda1 is None:
        lambda1 = phi * (1.0 - _BOIN_PHI1) / (1.0 - phi * _BOIN_PHI1)
    if lambda2 is None:
        lambda2 = phi * (1.0 + _BOIN_PHI2) / ((1.0 - phi) * _BOIN_PHI2 + 1.0)

    observed_rate = n_dlt / n_patients

    # Elimination check: use exact binomial posterior (Beta(1+n_dlt, 1+n-n_dlt))
    # P(true_rate > lambda2 + margin | data) > threshold
    # Conservative approximation: ≥ 2 × target DLT rate
    if observed_rate >= 2.0 * phi and n_patients >= 3:
        return "eliminate"

    if observed_rate < lambda1:
        return "escalate"
    if observed_rate <= lambda2:
        return "stay"
    return "deescalate"


# ---------------------------------------------------------------------------
# 3+3 rule
# ---------------------------------------------------------------------------


def three_plus_three(
    doses_mg: list[float],
    dlt_counts: list[int],
    n_patients: list[int],
) -> DoseFindingResult:
    """Apply the 3+3 dose escalation rule.

    The 3+3 rule works as follows for the *current* (last tested) dose level:

    - 0/3 DLTs     → escalate to next dose level
    - 1/3 DLTs     → expand to 6 patients at current level
      - 1/6 DLTs   → escalate
      - ≥2/6 DLTs  → current level is MTD (or deescalate if not level 1)
    - ≥2/3 DLTs    → deescalate (prior level is MTD)

    Parameters
    ----------
    doses_mg : Dose levels in ascending order (mg).
    dlt_counts : DLT counts at each evaluated dose level.
    n_patients : Patients enrolled at each dose level.

    Returns
    -------
    DoseFindingResult
    """
    if not doses_mg:
        raise ValueError("doses_mg must not be empty.")
    if len(dlt_counts) != len(n_patients):
        raise ValueError("dlt_counts and n_patients must have the same length.")
    if len(dlt_counts) > len(doses_mg):
        raise ValueError("dlt_counts cannot have more entries than doses_mg.")
    for i, (nd, np_) in enumerate(zip(dlt_counts, n_patients, strict=False)):
        if np_ <= 0:
            raise ValueError(f"n_patients[{i}] must be > 0, got {np_}.")
        if nd < 0 or nd > np_:
            raise ValueError(f"dlt_counts[{i}]={nd} out of range for n_patients[{i}]={np_}.")
    for d in doses_mg:
        if d <= 0:
            raise ValueError(f"All dose levels must be > 0, got {d}.")

    n_levels = len(dlt_counts)
    if n_levels == 0:
        # No doses tested yet — start at lowest dose
        return DoseFindingResult(
            recommended_dose_mg=doses_mg[0],
            n_dose_levels_tested=0,
            n_total_patients=0,
            dlt_rates_observed=[],
            decision="escalate",
            notes="No doses tested yet. Begin at the lowest dose level.",
        )

    current_idx = n_levels - 1
    current_dose = doses_mg[current_idx]
    nd = dlt_counts[current_idx]
    np_ = n_patients[current_idx]
    rate = nd / np_
    total_pts = sum(n_patients)
    obs_rates = [dlt_counts[i] / n_patients[i] for i in range(n_levels)]

    # --- decision logic ---
    if np_ == 3:
        if nd == 0:
            # 0/3: escalate
            if current_idx + 1 < len(doses_mg):
                rec_dose = doses_mg[current_idx + 1]
                decision = "escalate"
                notes = f"0/3 DLTs at {current_dose} mg. Escalate to {rec_dose} mg."
            else:
                rec_dose = current_dose
                decision = "mtd_found"
                notes = (
                    f"0/3 DLTs at {current_dose} mg (highest dose). Declare highest dose as MTD."
                )
        elif nd == 1:
            # 1/3: expand cohort
            rec_dose = current_dose
            decision = "stay"
            notes = f"1/3 DLTs at {current_dose} mg. Expand cohort to 6 patients at this dose."
        else:
            # ≥2/3: deescalate
            if current_idx > 0:
                rec_dose = doses_mg[current_idx - 1]
                decision = "deescalate"
                notes = (
                    f"{nd}/3 DLTs at {current_dose} mg. "
                    f"Deescalate to {rec_dose} mg (previous level is MTD candidate)."
                )
            else:
                rec_dose = current_dose
                decision = "deescalate"
                notes = (
                    f"{nd}/3 DLTs at lowest dose {current_dose} mg. "
                    "Trial stopped — no safe dose identified."
                )

    elif np_ >= 6:
        if nd <= 1:
            # ≤1/6: escalate
            if current_idx + 1 < len(doses_mg):
                rec_dose = doses_mg[current_idx + 1]
                decision = "escalate"
                notes = f"{nd}/{np_} DLTs at {current_dose} mg. Escalate to {rec_dose} mg."
            else:
                rec_dose = current_dose
                decision = "mtd_found"
                notes = f"{nd}/{np_} DLTs at {current_dose} mg (highest dose). Declare as MTD."
        else:
            # ≥2/6: MTD is prior level
            if current_idx > 0:
                rec_dose = doses_mg[current_idx - 1]
                decision = "mtd_found"
                notes = f"{nd}/{np_} DLTs at {current_dose} mg. MTD declared at {rec_dose} mg."
            else:
                rec_dose = current_dose
                decision = "mtd_found"
                notes = f"{nd}/{np_} DLTs at lowest dose. No safe dose identified; trial stopped."
    else:
        # Partial cohort (1–5 patients, unusual): apply proportional logic
        if rate < 1 / 3:
            if current_idx + 1 < len(doses_mg):
                rec_dose = doses_mg[current_idx + 1]
                decision = "escalate"
                notes = f"{nd}/{np_} DLTs at {current_dose} mg. Escalate."
            else:
                rec_dose = current_dose
                decision = "mtd_found"
                notes = f"{nd}/{np_} DLTs at highest dose. Declare MTD."
        elif rate <= 1 / 3:
            rec_dose = current_dose
            decision = "stay"
            notes = f"{nd}/{np_} DLTs at {current_dose} mg. Stay."
        else:
            if current_idx > 0:
                rec_dose = doses_mg[current_idx - 1]
                decision = "deescalate"
                notes = f"{nd}/{np_} DLTs at {current_dose} mg. Deescalate."
            else:
                rec_dose = current_dose
                decision = "deescalate"
                notes = f"{nd}/{np_} DLTs at lowest dose. Trial stopped."

    return DoseFindingResult(
        recommended_dose_mg=rec_dose,
        n_dose_levels_tested=n_levels,
        n_total_patients=total_pts,
        dlt_rates_observed=[round(r, 4) for r in obs_rates],
        decision=decision,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# MTD estimation (pool-adjacent-violators isotonic regression)
# ---------------------------------------------------------------------------


def calculate_mtd(
    dose_levels: list[float],
    dlt_rates: list[float],
    target_tox: float = 0.25,
) -> float:
    """Estimate the MTD as the dose whose isotonically-smoothed DLT rate is
    closest to *target_tox*.

    Isotonic regression (pool-adjacent-violators algorithm) is applied to
    enforce a non-decreasing DLT-rate vs dose relationship.

    Parameters
    ----------
    dose_levels : Dose levels in ascending order (mg).
    dlt_rates : Observed or true DLT rates at each dose level.
    target_tox : Target DLT probability (default 0.25).

    Returns
    -------
    float
        Estimated MTD in mg.
    """
    if not dose_levels:
        raise ValueError("dose_levels must not be empty.")
    if len(dose_levels) != len(dlt_rates):
        raise ValueError("dose_levels and dlt_rates must have the same length.")
    if not (0.0 < target_tox < 1.0):
        raise ValueError(f"target_tox must be in (0, 1), got {target_tox}.")
    for r in dlt_rates:
        if not (0.0 <= r <= 1.0):
            raise ValueError(f"All dlt_rates must be in [0, 1], got {r}.")
    for d in dose_levels:
        if d <= 0:
            raise ValueError(f"All dose_levels must be > 0, got {d}.")

    # Pool-adjacent-violators isotonic regression
    rates = list(dlt_rates)
    n = len(rates)
    # Use weighted mean pools; equal weights here
    weights = [1] * n
    # Iterate until no violations remain
    changed = True
    while changed:
        changed = False
        i = 0
        while i < n - 1:
            if rates[i] > rates[i + 1]:
                # Pool i and i+1
                combined = (rates[i] * weights[i] + rates[i + 1] * weights[i + 1]) / (
                    weights[i] + weights[i + 1]
                )
                rates[i] = combined
                rates[i + 1] = combined
                weights[i] += weights[i + 1]
                weights[i + 1] = weights[i]
                changed = True
            i += 1

    # Find dose whose smoothed rate is closest to target_tox
    best_idx = min(range(n), key=lambda k: abs(rates[k] - target_tox))
    return float(dose_levels[best_idx])


# ---------------------------------------------------------------------------
# Dose escalation simulation
# ---------------------------------------------------------------------------


def simulate_dose_escalation(
    true_dlt_rates: list[float],
    n_cohorts: int = 6,
    cohort_size: int = 3,
    target_tox: float = 0.25,
    seed: int | None = None,
) -> DoseEscalationResult:
    """Simulate a BOIN dose-escalation trial.

    Uses the BOIN decision boundaries to guide dose escalation/de-escalation
    across pre-specified dose levels.

    Parameters
    ----------
    true_dlt_rates : True DLT probability at each dose level (length = number
        of dose levels).
    n_cohorts : Maximum number of cohorts to enrol (default 6).
    cohort_size : Patients per cohort (default 3).
    target_tox : Target DLT probability (default 0.25).
    seed : Random seed for reproducibility.

    Returns
    -------
    DoseEscalationResult
    """
    if not true_dlt_rates:
        raise ValueError("true_dlt_rates must not be empty.")
    for r in true_dlt_rates:
        if not (0.0 <= r <= 1.0):
            raise ValueError(f"All true_dlt_rates must be in [0, 1], got {r}.")
    if n_cohorts <= 0:
        raise ValueError(f"n_cohorts must be > 0, got {n_cohorts}.")
    if cohort_size <= 0:
        raise ValueError(f"cohort_size must be > 0, got {cohort_size}.")
    if not (0.0 < target_tox < 1.0):
        raise ValueError(f"target_tox must be in (0, 1), got {target_tox}.")

    rng = random.Random(seed)

    n_levels = len(true_dlt_rates)
    # Dose levels: 100 mg * 2^i as placeholder (absolute values don't affect
    # simulation logic which only uses indices)
    dose_levels_mg = [100.0 * (2**i) for i in range(n_levels)]

    n_patients_per_level = [0] * n_levels
    dlt_counts_level = [0] * n_levels

    # Pre-compute BOIN boundaries
    phi = target_tox
    lam1 = phi * (1.0 - _BOIN_PHI1) / (1.0 - phi * _BOIN_PHI1)
    lam2 = phi * (1.0 + _BOIN_PHI2) / ((1.0 - phi) * _BOIN_PHI2 + 1.0)

    eliminated: set[int] = set()
    current_level = 0  # start at lowest dose
    trial_stopped = False
    stop_reason = ""

    for _ in range(n_cohorts):
        if trial_stopped:
            break

        # Check current level hasn't been eliminated
        if current_level in eliminated:
            # Find nearest non-eliminated lower level
            found = False
            for lv in range(current_level - 1, -1, -1):
                if lv not in eliminated:
                    current_level = lv
                    found = True
                    break
            if not found:
                trial_stopped = True
                stop_reason = "All lower dose levels eliminated — trial stopped."
                break

        # Simulate DLTs for this cohort
        true_rate = true_dlt_rates[current_level]
        cohort_dlts = sum(1 for _ in range(cohort_size) if rng.random() < true_rate)

        n_patients_per_level[current_level] += cohort_size
        dlt_counts_level[current_level] += cohort_dlts

        total_n = n_patients_per_level[current_level]
        total_dlt = dlt_counts_level[current_level]

        decision = boin_decision(total_n, total_dlt, target_tox, lam1, lam2)

        if decision == "eliminate":
            eliminated.add(current_level)
            # Move down
            found = False
            for lv in range(current_level - 1, -1, -1):
                if lv not in eliminated:
                    current_level = lv
                    found = True
                    break
            if not found:
                trial_stopped = True
                stop_reason = "Current dose eliminated with no safe lower level."
        elif decision == "escalate":
            next_level = current_level + 1
            while next_level in eliminated:
                next_level += 1
            if next_level >= n_levels:
                # At highest level already
                stop_reason = "Highest dose level reached."
                trial_stopped = True
            else:
                current_level = next_level
        elif decision == "deescalate":
            if current_level > 0:
                prev_level = current_level - 1
                while prev_level in eliminated and prev_level > 0:
                    prev_level -= 1
                if prev_level not in eliminated:
                    current_level = prev_level
                else:
                    trial_stopped = True
                    stop_reason = "Cannot deescalate — no safe lower level."
            else:
                trial_stopped = True
                stop_reason = "DLT rate too high at lowest dose — trial stopped."
        # 'stay' → continue at same level

    # Compute observed DLT rates for tested levels
    tested_rates = [
        (dlt_counts_level[i] / n_patients_per_level[i]) if n_patients_per_level[i] > 0 else 0.0
        for i in range(n_levels)
    ]

    # Estimate MTD from levels that were actually tested
    tested_levels = [i for i in range(n_levels) if n_patients_per_level[i] > 0]
    if tested_levels:
        tl_doses = [dose_levels_mg[i] for i in tested_levels]
        tl_rates = [tested_rates[i] for i in tested_levels]
        estimated_mtd = calculate_mtd(tl_doses, tl_rates, target_tox)
    else:
        estimated_mtd = dose_levels_mg[0]

    if not stop_reason:
        stop_reason = f"Trial completed after up to {n_cohorts} cohorts."

    return DoseEscalationResult(
        dose_levels=dose_levels_mg,
        n_patients_per_level=n_patients_per_level,
        dlt_counts=dlt_counts_level,
        estimated_mtd=estimated_mtd,
        true_dlt_rates=list(true_dlt_rates),
        notes=stop_reason,
    )
