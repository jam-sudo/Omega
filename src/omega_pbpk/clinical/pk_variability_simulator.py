"""Phase 866 — Pharmacokinetic Variability Simulator.

Simulates population PK variability with log-normal BSV using a
deterministic LCG-based RNG (seed-based). Generates summary statistics
for a virtual population using a one-compartment oral model.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# LCG-based deterministic RNG (no random module)
# ---------------------------------------------------------------------------

_LCG_A = 1664525
_LCG_C = 1013904223
_LCG_M = 2**32

# Sampling time points (hours)
_TIME_POINTS = [0.0, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 12.0, 24.0]


def _lcg_next(seed: int) -> int:
    """Advance LCG one step and return next seed."""
    return (_LCG_A * seed + _LCG_C) % _LCG_M


def _lcg_uniform(seed: int) -> tuple:
    """Return (uniform_0_1, next_seed)."""
    seed = _lcg_next(seed)
    u = seed / _LCG_M
    return u, seed


def _lcg_normal(seed: int, mu: float = 0.0, sigma: float = 1.0) -> tuple:
    """Box-Muller transform to get a normal sample. Returns (value, next_seed).

    Uses two LCG uniforms; returns only one normal to keep it simple.
    """
    u1, seed = _lcg_uniform(seed)
    u2, seed = _lcg_uniform(seed)

    # Avoid log(0)
    if u1 < 1e-12:
        u1 = 1e-12

    z = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
    return mu + sigma * z, seed


# ---------------------------------------------------------------------------
# One-compartment oral PK model
# ---------------------------------------------------------------------------


def _one_cpt_oral_conc(
    t: float,
    dose_mg: float,
    f_oral: float,
    ka: float,
    ke: float,
    vd: float,
) -> float:
    """1-cpt oral concentration at time t (mg/L).

    C(t) = (F * dose * ka) / (Vd * (ka - ke)) * (exp(-ke*t) - exp(-ka*t))
    """
    if t <= 0.0:
        return 0.0
    if abs(ka - ke) < 1e-8:
        # Flip-flop: treat as ke slightly different to avoid singularity
        ke = ke * 1.001 + 1e-8
    numerator = f_oral * dose_mg * ka
    denominator = vd * (ka - ke)
    if abs(denominator) < 1e-15:
        return 0.0
    c = (numerator / denominator) * (math.exp(-ke * t) - math.exp(-ka * t))
    return max(0.0, c)


def _compute_auc_trapezoidal(times: list[float], concs: list[float]) -> float:
    """Manual trapezoidal AUC."""
    auc = 0.0
    for i in range(len(times) - 1):
        dt = times[i + 1] - times[i]
        auc += 0.5 * (concs[i] + concs[i + 1]) * dt
    return auc


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PKVariabilityResult:
    """Result of population PK variability simulation."""

    drug_name: str
    n_subjects: int
    dose_mg: float
    cl_pop_L_per_h: float
    vd_pop_L: float
    ka_pop_per_h: float
    omega_cl_cv_pct: float
    omega_vd_cv_pct: float
    cmax_mean: float
    cmax_median: float
    cmax_cv_pct: float
    auc_mean: float
    auc_median: float
    auc_cv_pct: float
    pct_below_mec: float
    pct_above_mtc: float
    mec_mg_L: float
    mtc_mg_L: float
    notes: str


# ---------------------------------------------------------------------------
# Main simulation function
# ---------------------------------------------------------------------------


def simulate_pk_variability(
    drug_name: str,
    dose_mg: float,
    n_subjects: int = 100,
    cl_pop_L_per_h: float = 5.0,
    vd_pop_L: float = 50.0,
    ka_pop_per_h: float = 1.0,
    f_oral: float = 0.8,
    omega_cl_cv_pct: float = 30.0,
    omega_vd_cv_pct: float = 20.0,
    omega_ka_cv_pct: float = 25.0,
    mec_mg_L: float = 0.5,
    mtc_mg_L: float = 5.0,
    seed: int = 42,
) -> PKVariabilityResult:
    """Simulate population PK variability using log-normal BSV.

    Uses a deterministic LCG-based RNG for reproducibility.

    Parameters
    ----------
    drug_name:
        Name of drug.
    dose_mg:
        Oral dose (mg).
    n_subjects:
        Number of virtual subjects (>= 10).
    cl_pop_L_per_h:
        Population typical CL (L/h).
    vd_pop_L:
        Population typical Vd (L).
    ka_pop_per_h:
        Population typical ka (1/h).
    f_oral:
        Oral bioavailability fraction.
    omega_cl_cv_pct:
        BSV CV% for CL (log-normal).
    omega_vd_cv_pct:
        BSV CV% for Vd (log-normal).
    omega_ka_cv_pct:
        BSV CV% for ka (log-normal).
    mec_mg_L:
        Minimum effective concentration (mg/L).
    mtc_mg_L:
        Maximum tolerated concentration (mg/L).
    seed:
        Random seed for deterministic simulation.

    Returns
    -------
    PKVariabilityResult
    """
    # Validation
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if n_subjects < 10:
        raise ValueError("n_subjects must be >= 10")
    if omega_cl_cv_pct < 0:
        raise ValueError("omega_cl_cv_pct must be >= 0")
    if omega_vd_cv_pct < 0:
        raise ValueError("omega_vd_cv_pct must be >= 0")
    if omega_ka_cv_pct < 0:
        raise ValueError("omega_ka_cv_pct must be >= 0")
    if mec_mg_L < 0:
        raise ValueError("mec_mg_L must be >= 0")
    if mtc_mg_L < 0:
        raise ValueError("mtc_mg_L must be >= 0")

    # Convert CV% to log-normal sigma: CV% = 100*(exp(omega^2)-1)^0.5
    # Approximate: omega ~= CV% / 100 for small CV
    # Exact: omega = sqrt(log(1 + (CV/100)^2))
    def cv_to_sigma(cv_pct: float) -> float:
        cv = cv_pct / 100.0
        if cv <= 0.0:
            return 0.0
        return math.sqrt(math.log(1.0 + cv * cv))

    sigma_cl = cv_to_sigma(omega_cl_cv_pct)
    sigma_vd = cv_to_sigma(omega_vd_cv_pct)
    sigma_ka = cv_to_sigma(omega_ka_cv_pct)

    # Simulate n_subjects
    cmax_list: list[float] = []
    auc_list: list[float] = []
    n_below_mec = 0
    n_above_mtc = 0

    current_seed = seed

    for _ in range(n_subjects):
        # Sample individual parameters using log-normal variability
        eta_cl, current_seed = _lcg_normal(current_seed, 0.0, sigma_cl)
        eta_vd, current_seed = _lcg_normal(current_seed, 0.0, sigma_vd)
        eta_ka, current_seed = _lcg_normal(current_seed, 0.0, sigma_ka)

        cl_i = cl_pop_L_per_h * math.exp(eta_cl)
        vd_i = vd_pop_L * math.exp(eta_vd)
        ka_i = ka_pop_per_h * math.exp(eta_ka)
        ke_i = cl_i / vd_i

        # Compute concentration profile
        concs = [_one_cpt_oral_conc(t, dose_mg, f_oral, ka_i, ke_i, vd_i) for t in _TIME_POINTS]

        cmax_i = max(concs)
        auc_i = _compute_auc_trapezoidal(_TIME_POINTS, concs)

        cmax_list.append(cmax_i)
        auc_list.append(auc_i)

        # Classification based on Cmax
        if cmax_i < mec_mg_L:
            n_below_mec += 1
        if cmax_i > mtc_mg_L:
            n_above_mtc += 1

    # Summary statistics
    n = len(cmax_list)

    def _mean(vals: list[float]) -> float:
        return sum(vals) / len(vals)

    def _median(vals: list[float]) -> float:
        s = sorted(vals)
        mid = len(s) // 2
        if len(s) % 2 == 0:
            return (s[mid - 1] + s[mid]) / 2.0
        return s[mid]

    def _cv_pct(vals: list[float]) -> float:
        m = _mean(vals)
        if m == 0.0:
            return 0.0
        variance = sum((v - m) ** 2 for v in vals) / len(vals)
        sd = math.sqrt(variance)
        return (sd / m) * 100.0

    cmax_mean = _mean(cmax_list)
    cmax_median = _median(cmax_list)
    cmax_cv = _cv_pct(cmax_list)

    auc_mean = _mean(auc_list)
    auc_median = _median(auc_list)
    auc_cv = _cv_pct(auc_list)

    pct_below_mec = (n_below_mec / n) * 100.0
    pct_above_mtc = (n_above_mtc / n) * 100.0

    # Notes
    notes_parts = []
    if cmax_cv > 50:
        notes_parts.append("High Cmax variability (CV > 50%)")
    if pct_below_mec > 20:
        notes_parts.append(f"{pct_below_mec:.1f}% subjects below MEC")
    if pct_above_mtc > 10:
        notes_parts.append(f"{pct_above_mtc:.1f}% subjects above MTC")
    notes = "; ".join(notes_parts) if notes_parts else "Acceptable PK variability profile"

    return PKVariabilityResult(
        drug_name=drug_name,
        n_subjects=n_subjects,
        dose_mg=dose_mg,
        cl_pop_L_per_h=cl_pop_L_per_h,
        vd_pop_L=vd_pop_L,
        ka_pop_per_h=ka_pop_per_h,
        omega_cl_cv_pct=omega_cl_cv_pct,
        omega_vd_cv_pct=omega_vd_cv_pct,
        cmax_mean=cmax_mean,
        cmax_median=cmax_median,
        cmax_cv_pct=cmax_cv,
        auc_mean=auc_mean,
        auc_median=auc_median,
        auc_cv_pct=auc_cv,
        pct_below_mec=pct_below_mec,
        pct_above_mtc=pct_above_mtc,
        mec_mg_L=mec_mg_L,
        mtc_mg_L=mtc_mg_L,
        notes=notes,
    )


def compare_dose_variability(
    drug_name: str,
    doses_mg: list[float],
    n_subjects: int = 100,
    cl_pop_L_per_h: float = 5.0,
    vd_pop_L: float = 50.0,
    ka_pop_per_h: float = 1.0,
) -> list[PKVariabilityResult]:
    """Compare PK variability across multiple doses.

    Parameters
    ----------
    drug_name:
        Name of drug.
    doses_mg:
        List of oral doses to compare (mg).
    n_subjects:
        Number of virtual subjects per dose group.
    cl_pop_L_per_h:
        Population typical CL (L/h).
    vd_pop_L:
        Population typical Vd (L).
    ka_pop_per_h:
        Population typical ka (1/h).

    Returns
    -------
    List[PKVariabilityResult] sorted by dose_mg ascending.
    """
    results = []
    for dose in doses_mg:
        r = simulate_pk_variability(
            drug_name=drug_name,
            dose_mg=dose,
            n_subjects=n_subjects,
            cl_pop_L_per_h=cl_pop_L_per_h,
            vd_pop_L=vd_pop_L,
            ka_pop_per_h=ka_pop_per_h,
        )
        results.append(r)

    results.sort(key=lambda r: r.dose_mg)
    return results
