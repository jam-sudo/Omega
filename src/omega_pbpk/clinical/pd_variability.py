"""Pharmacodynamic variability model using LCG-based inter-individual simulation.

Simulates population variability in Emax pharmacodynamic response using a
log-normal between-subject variability model with LCG + Box-Muller RNG.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PDVariabilityResult:
    """Result of a pharmacodynamic variability simulation."""

    drug_name: str
    n_subjects: int
    emax: float
    ec50_mg_L: float
    hill_coef: float
    omega_emax_cv: float  # CV for Emax BSV
    omega_ec50_cv: float  # CV for EC50 BSV
    concentration_mg_L: float  # evaluated at this concentration
    # Population summary
    mean_effect_pct: float
    cv_effect_pct: float
    p5_effect_pct: float
    p95_effect_pct: float
    responder_pct: float  # % subjects with effect > responder_threshold
    responder_threshold_pct: float
    therapeutic_coverage_pct: float  # % subjects within [min_effect, max_effect]
    min_effect_threshold_pct: float
    max_effect_threshold_pct: float
    notes: str


# ---------------------------------------------------------------------------
# LCG RNG helpers
# ---------------------------------------------------------------------------


def _lcg_next(state: int) -> tuple[int, float]:
    """Advance LCG state, return (new_state, uniform_0_1)."""
    state = (1664525 * state + 1013904223) % (2**32)
    return state, state / (2**32)


def _box_muller(state: int) -> tuple[int, float, float]:
    """Generate two standard normal samples via Box-Muller."""
    state, u1 = _lcg_next(state)
    state, u2 = _lcg_next(state)
    # Avoid log(0)
    if u1 < 1e-15:
        u1 = 1e-15
    r = math.sqrt(-2.0 * math.log(u1))
    theta = 2.0 * math.pi * u2
    z0 = r * math.cos(theta)
    z1 = r * math.sin(theta)
    return state, z0, z1


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def simulate_pd_variability(
    drug_name: str,
    concentration_mg_L: float,
    emax: float,
    ec50_mg_L: float,
    hill_coef: float = 1.0,
    n_subjects: int = 100,
    omega_emax_cv: float = 0.3,
    omega_ec50_cv: float = 0.5,
    seed: int = 42,
    responder_threshold_pct: float = 50.0,
    min_effect_threshold_pct: float = 20.0,
    max_effect_threshold_pct: float = 80.0,
) -> PDVariabilityResult:
    """Simulate inter-individual variability in PD response (Emax model).

    Parameters
    ----------
    drug_name:
        Name of the drug.
    concentration_mg_L:
        Plasma drug concentration at which to evaluate response.
    emax:
        Population-typical maximum effect (%).
    ec50_mg_L:
        Population-typical EC50 (mg/L).
    hill_coef:
        Hill coefficient (steepness).
    n_subjects:
        Number of virtual subjects to simulate.
    omega_emax_cv:
        Coefficient of variation for log-normal Emax BSV.
    omega_ec50_cv:
        Coefficient of variation for log-normal EC50 BSV.
    seed:
        LCG seed for deterministic RNG.
    responder_threshold_pct:
        Effect threshold above which a subject is considered a responder.
    min_effect_threshold_pct:
        Lower bound for therapeutic coverage window.
    max_effect_threshold_pct:
        Upper bound for therapeutic coverage window.

    Returns
    -------
    PDVariabilityResult
    """
    if n_subjects < 2:
        raise ValueError("n_subjects must be >= 2")
    if emax <= 0.0:
        raise ValueError("emax must be > 0")
    if ec50_mg_L <= 0.0:
        raise ValueError("ec50_mg_L must be > 0")

    # Pre-compute log-normal sigma from CV
    # sigma = sqrt(ln(1 + CV^2))
    sigma_emax = math.sqrt(math.log(1.0 + omega_emax_cv**2))
    sigma_ec50 = math.sqrt(math.log(1.0 + omega_ec50_cv**2))

    state = seed
    effects: list[float] = []

    i = 0
    while i < n_subjects:
        state, z0, z1 = _box_muller(state)

        # Subject i
        eta_emax = sigma_emax * z0
        eta_ec50 = sigma_ec50 * z1
        emax_i = emax * math.exp(eta_emax)
        ec50_i = ec50_mg_L * math.exp(eta_ec50)

        c = concentration_mg_L
        n = hill_coef
        if ec50_i <= 0.0:
            ec50_i = 1e-12
        cn = c**n if c > 0.0 else 0.0
        ec50n = ec50_i**n
        effect_i = emax_i * cn / (ec50n + cn) if (ec50n + cn) > 0.0 else 0.0
        # Clamp to [0, 100]
        effect_i = max(0.0, min(100.0, effect_i))
        effects.append(effect_i)
        i += 1

        if i >= n_subjects:
            break

        # Use second normal from Box-Muller pair for subject i+1 if available
        if i < n_subjects:
            state2, z2, z3 = _box_muller(state)
            eta_emax2 = sigma_emax * z2
            eta_ec50_2 = sigma_ec50 * z3
            emax_j = emax * math.exp(eta_emax2)
            ec50_j = ec50_mg_L * math.exp(eta_ec50_2)
            ec50_j = max(1e-12, ec50_j)
            cn2 = c**n if c > 0.0 else 0.0
            ec50n2 = ec50_j**n
            effect_j = emax_j * cn2 / (ec50n2 + cn2) if (ec50n2 + cn2) > 0.0 else 0.0
            effect_j = max(0.0, min(100.0, effect_j))
            effects.append(effect_j)
            state = state2
            i += 1

    effects = effects[:n_subjects]

    # Population statistics
    n = len(effects)
    mean_eff = sum(effects) / n

    if n >= 2:
        var_eff = sum((e - mean_eff) ** 2 for e in effects) / (n - 1)
        std_eff = math.sqrt(var_eff)
    else:
        std_eff = 0.0

    cv_eff = (std_eff / mean_eff * 100.0) if mean_eff > 0.0 else 0.0

    sorted_eff = sorted(effects)
    p5_idx = max(0, int(0.05 * n) - 1)
    p95_idx = min(n - 1, int(0.95 * n))
    p5_eff = sorted_eff[p5_idx]
    p95_eff = sorted_eff[p95_idx]

    responder_count = sum(1 for e in effects if e > responder_threshold_pct)
    responder_pct = 100.0 * responder_count / n

    coverage_count = sum(
        1 for e in effects if min_effect_threshold_pct <= e <= max_effect_threshold_pct
    )
    therapeutic_coverage_pct = 100.0 * coverage_count / n

    notes = (
        f"C={concentration_mg_L:.3f} mg/L; "
        f"Emax={emax:.1f}%; EC50={ec50_mg_L:.3f} mg/L; "
        f"Hill={hill_coef:.2f}; "
        f"omega_Emax={omega_emax_cv:.2f}; omega_EC50={omega_ec50_cv:.2f}"
    )

    return PDVariabilityResult(
        drug_name=drug_name,
        n_subjects=n_subjects,
        emax=emax,
        ec50_mg_L=ec50_mg_L,
        hill_coef=hill_coef,
        omega_emax_cv=omega_emax_cv,
        omega_ec50_cv=omega_ec50_cv,
        concentration_mg_L=concentration_mg_L,
        mean_effect_pct=mean_eff,
        cv_effect_pct=cv_eff,
        p5_effect_pct=p5_eff,
        p95_effect_pct=p95_eff,
        responder_pct=responder_pct,
        responder_threshold_pct=responder_threshold_pct,
        therapeutic_coverage_pct=therapeutic_coverage_pct,
        min_effect_threshold_pct=min_effect_threshold_pct,
        max_effect_threshold_pct=max_effect_threshold_pct,
        notes=notes,
    )


def pd_variability_across_concentrations(
    drug_name: str,
    concentrations: list[float],
    emax: float,
    ec50_mg_L: float,
    hill_coef: float = 1.0,
    n_subjects: int = 100,
    omega_emax_cv: float = 0.3,
    omega_ec50_cv: float = 0.5,
    seed: int = 42,
) -> list[PDVariabilityResult]:
    """Run PD variability simulation for each concentration.

    Parameters
    ----------
    drug_name:
        Drug name.
    concentrations:
        List of plasma concentrations to evaluate (mg/L).
    emax:
        Population Emax (%).
    ec50_mg_L:
        Population EC50 (mg/L).
    hill_coef:
        Hill coefficient.
    n_subjects:
        Number of virtual subjects.
    omega_emax_cv:
        Emax BSV CV.
    omega_ec50_cv:
        EC50 BSV CV.
    seed:
        LCG seed.

    Returns
    -------
    list[PDVariabilityResult] sorted by concentration ascending.
    """
    results = []
    for conc in concentrations:
        result = simulate_pd_variability(
            drug_name=drug_name,
            concentration_mg_L=conc,
            emax=emax,
            ec50_mg_L=ec50_mg_L,
            hill_coef=hill_coef,
            n_subjects=n_subjects,
            omega_emax_cv=omega_emax_cv,
            omega_ec50_cv=omega_ec50_cv,
            seed=seed,
        )
        results.append(result)

    results.sort(key=lambda r: r.concentration_mg_L)
    return results
