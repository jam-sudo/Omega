"""Phase 288 — Population Variability Simulator.

Simulate PK variability across a virtual patient population using
log-normal inter-individual variability (IIV). Generates summary
statistics and percentile profiles.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Random number utilities (no numpy/scipy)
# ---------------------------------------------------------------------------


def _lcg(seed: int, n: int) -> list[float]:
    """Generate n uniform(0, 1) values using a linear congruential generator."""
    a, c, m = 1664525, 1013904223, 2**32
    vals: list[float] = []
    state = seed % m
    for _ in range(n):
        state = (a * state + c) % m
        vals.append(state / m)
    return vals


def _box_muller(u1: float, u2: float) -> tuple[float, float]:
    """Box-Muller transform: map two uniform variates to two standard normals."""
    # Clamp u1 away from zero to avoid log(0)
    u1 = max(u1, 1e-12)
    r = math.sqrt(-2.0 * math.log(u1))
    theta = 2.0 * math.pi * u2
    return r * math.cos(theta), r * math.sin(theta)


def _sample_normals(seed: int, n: int) -> list[float]:
    """Sample n standard-normal values from an LCG seed."""
    # We need ceil(n/2) pairs
    n_pairs = (n + 1) // 2
    uniforms = _lcg(seed, 2 * n_pairs)
    normals: list[float] = []
    for i in range(n_pairs):
        z1, z2 = _box_muller(uniforms[2 * i], uniforms[2 * i + 1])
        normals.append(z1)
        normals.append(z2)
    return normals[:n]


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class PopulationVariabilityResult:
    """Result of population variability simulation (has list fields)."""

    drug_name: str
    n_subjects: int
    dose_mg: float
    times_h: list = field(default_factory=list)
    c_p5_mg_L: list = field(default_factory=list)
    c_p50_mg_L: list = field(default_factory=list)
    c_p95_mg_L: list = field(default_factory=list)
    mean_auc_mg_h_per_L: float = 0.0
    cv_auc_pct: float = 0.0
    p5_auc: float = 0.0
    p95_auc: float = 0.0
    mean_cmax_mg_L: float = 0.0
    cv_cmax_pct: float = 0.0
    variability_class: str = ""
    notes: str = ""


@dataclass(frozen=True)
class VariabilityRiskResult:
    """Result of therapeutic window risk assessment."""

    drug_name: str
    n_subjects: int
    pct_below_therapeutic_min: float
    pct_above_therapeutic_max: float
    pct_in_therapeutic_window: float
    risk_class: str
    notes: str


# ---------------------------------------------------------------------------
# Helper: percentile (pure Python)
# ---------------------------------------------------------------------------


def _percentile(sorted_vals: list[float], p: float) -> float:
    """Compute the p-th percentile of a sorted list (linear interpolation)."""
    n = len(sorted_vals)
    if n == 1:
        return sorted_vals[0]
    idx = p / 100.0 * (n - 1)
    lo = int(idx)
    hi = lo + 1
    if hi >= n:
        return sorted_vals[-1]
    frac = idx - lo
    return sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac


# ---------------------------------------------------------------------------
# Main simulation
# ---------------------------------------------------------------------------


def _simulate_1cpt_oral(
    cl: float,
    vd: float,
    ka: float,
    f: float,
    dose_mg: float,
    times_h: list[float],
) -> list[float]:
    """Analytical 1-compartment oral PK profile."""
    ke = cl / vd
    # Avoid division by zero when ka == ke
    if abs(ka - ke) < 1e-9:
        ka_eff = ke * 1.001
    else:
        ka_eff = ka
    scale = f * dose_mg * ka_eff / (vd * (ka_eff - ke))
    concs = []
    for t in times_h:
        if t <= 0.0:
            concs.append(0.0)
        else:
            c = scale * (math.exp(-ke * t) - math.exp(-ka_eff * t))
            concs.append(max(0.0, c))
    return concs


def simulate_population_variability(
    drug_name: str,
    dose_mg: float,
    cl_mean_L_per_h: float,
    vd_mean_L: float,
    ka_mean_per_h: float,
    f_mean: float,
    omega_cl: float,
    omega_vd: float,
    omega_ka: float,
    n_subjects: int = 100,
    t_end_h: float = 24.0,
    dt_h: float = 0.5,
    seed: int = 42,
) -> PopulationVariabilityResult:
    """Simulate PK variability across a virtual patient population.

    Parameters
    ----------
    drug_name:
        Drug identifier.
    dose_mg:
        Administered dose [mg].
    cl_mean_L_per_h:
        Population mean clearance [L/h].
    vd_mean_L:
        Population mean volume of distribution [L].
    ka_mean_per_h:
        Population mean absorption rate constant [1/h].
    f_mean:
        Population mean bioavailability fraction (0 < f <= 1).
    omega_cl, omega_vd, omega_ka:
        Log-normal CV for CL, Vd, ka (fraction, e.g. 0.3 = 30% CV).
    n_subjects:
        Number of virtual subjects (1–200).
    t_end_h:
        Simulation end time [h].
    dt_h:
        Time step [h].
    seed:
        Random seed for reproducibility.

    Returns
    -------
    PopulationVariabilityResult
    """
    # --- Validation ---
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if cl_mean_L_per_h <= 0:
        raise ValueError("cl_mean_L_per_h must be > 0")
    if vd_mean_L <= 0:
        raise ValueError("vd_mean_L must be > 0")
    if ka_mean_per_h <= 0:
        raise ValueError("ka_mean_per_h must be > 0")
    if not (0 < f_mean <= 1):
        raise ValueError("f_mean must be in (0, 1]")
    if omega_cl < 0:
        raise ValueError("omega_cl must be >= 0")
    if omega_vd < 0:
        raise ValueError("omega_vd must be >= 0")
    if omega_ka < 0:
        raise ValueError("omega_ka must be >= 0")
    if not (1 <= n_subjects <= 200):
        raise ValueError("n_subjects must be between 1 and 200 inclusive")
    if t_end_h <= 0:
        raise ValueError("t_end_h must be > 0")

    # --- Build time grid ---
    n_steps = max(2, int(round(t_end_h / dt_h)) + 1)
    times_h = [i * dt_h for i in range(n_steps)]
    # Ensure t_end_h is last point
    if times_h[-1] < t_end_h:
        times_h.append(t_end_h)

    # --- Sample individual parameters (3 normals per subject) ---
    normals = _sample_normals(seed, 3 * n_subjects)

    auc_list: list[float] = []
    cmax_list: list[float] = []
    all_concs: list[list[float]] = []

    for i in range(n_subjects):
        z_cl = normals[3 * i]
        z_vd = normals[3 * i + 1]
        z_ka = normals[3 * i + 2]

        # Median-preserving log-normal parameterisation
        cl_i = cl_mean_L_per_h * math.exp(omega_cl * z_cl - 0.5 * omega_cl**2)
        vd_i = vd_mean_L * math.exp(omega_vd * z_vd - 0.5 * omega_vd**2)
        ka_i = ka_mean_per_h * math.exp(omega_ka * z_ka - 0.5 * omega_ka**2)

        concs = _simulate_1cpt_oral(cl_i, vd_i, ka_i, f_mean, dose_mg, times_h)
        all_concs.append(concs)

        # Analytical AUC: F*D/CL
        auc_i = f_mean * dose_mg / cl_i
        auc_list.append(auc_i)
        cmax_list.append(max(concs))

    # --- Percentile profiles ---
    n_t = len(times_h)
    c_p5: list[float] = []
    c_p50: list[float] = []
    c_p95: list[float] = []
    for t_idx in range(n_t):
        col = sorted(subj[t_idx] for subj in all_concs)
        c_p5.append(_percentile(col, 5.0))
        c_p50.append(_percentile(col, 50.0))
        c_p95.append(_percentile(col, 95.0))

    # --- Summary statistics ---
    mean_auc = sum(auc_list) / n_subjects
    mean_cmax = sum(cmax_list) / n_subjects

    if n_subjects > 1:
        var_auc = sum((a - mean_auc) ** 2 for a in auc_list) / (n_subjects - 1)
        var_cmax = sum((c - mean_cmax) ** 2 for c in cmax_list) / (n_subjects - 1)
        sd_auc = math.sqrt(var_auc) if var_auc > 0 else 0.0
        sd_cmax = math.sqrt(var_cmax) if var_cmax > 0 else 0.0
        cv_auc_pct = (sd_auc / mean_auc * 100.0) if mean_auc > 0 else 0.0
        cv_cmax_pct = (sd_cmax / mean_cmax * 100.0) if mean_cmax > 0 else 0.0
    else:
        cv_auc_pct = 0.0
        cv_cmax_pct = 0.0

    auc_sorted = sorted(auc_list)
    p5_auc = _percentile(auc_sorted, 5.0)
    p95_auc = _percentile(auc_sorted, 95.0)

    # --- Variability class ---
    if cv_auc_pct < 30.0:
        variability_class = "low"
    elif cv_auc_pct <= 60.0:
        variability_class = "moderate"
    else:
        variability_class = "high"

    notes = (
        f"Simulated {n_subjects} virtual subjects; "
        f"omega_CL={omega_cl:.2f}, omega_Vd={omega_vd:.2f}, omega_ka={omega_ka:.2f}; "
        f"CV(AUC)={cv_auc_pct:.1f}% → {variability_class} variability."
    )

    return PopulationVariabilityResult(
        drug_name=drug_name,
        n_subjects=n_subjects,
        dose_mg=dose_mg,
        times_h=times_h,
        c_p5_mg_L=c_p5,
        c_p50_mg_L=c_p50,
        c_p95_mg_L=c_p95,
        mean_auc_mg_h_per_L=mean_auc,
        cv_auc_pct=cv_auc_pct,
        p5_auc=p5_auc,
        p95_auc=p95_auc,
        mean_cmax_mg_L=mean_cmax,
        cv_cmax_pct=cv_cmax_pct,
        variability_class=variability_class,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Risk assessment
# ---------------------------------------------------------------------------


def assess_variability_risk(
    drug_name: str,
    auc_list: list[float],
    cmax_list: list[float],
    therapeutic_auc_min: float,
    therapeutic_auc_max: float,
) -> VariabilityRiskResult:
    """Assess what fraction of subjects fall within the therapeutic window.

    Parameters
    ----------
    drug_name:
        Drug identifier.
    auc_list:
        Individual AUC values [mg*h/L].
    cmax_list:
        Individual Cmax values (currently unused in classification but stored).
    therapeutic_auc_min:
        Minimum AUC for efficacy.
    therapeutic_auc_max:
        Maximum safe AUC.

    Returns
    -------
    VariabilityRiskResult
    """
    if len(auc_list) == 0:
        raise ValueError("auc_list must not be empty")
    if therapeutic_auc_min <= 0:
        raise ValueError("therapeutic_auc_min must be > 0")
    if therapeutic_auc_max <= therapeutic_auc_min:
        raise ValueError("therapeutic_auc_max must be > therapeutic_auc_min")

    n = len(auc_list)
    n_below = sum(1 for a in auc_list if a < therapeutic_auc_min)
    n_above = sum(1 for a in auc_list if a > therapeutic_auc_max)
    n_in = n - n_below - n_above

    pct_below = n_below / n * 100.0
    pct_above = n_above / n * 100.0
    pct_in = n_in / n * 100.0

    if pct_in > 70.0:
        risk_class = "acceptable"
    elif pct_in >= 50.0:
        risk_class = "borderline"
    else:
        risk_class = "high_risk"

    notes = (
        f"{n} subjects assessed; "
        f"{pct_in:.1f}% in therapeutic window "
        f"[{therapeutic_auc_min:.2f}, {therapeutic_auc_max:.2f}] mg*h/L; "
        f"risk class: {risk_class}."
    )

    return VariabilityRiskResult(
        drug_name=drug_name,
        n_subjects=n,
        pct_below_therapeutic_min=pct_below,
        pct_above_therapeutic_max=pct_above,
        pct_in_therapeutic_window=pct_in,
        risk_class=risk_class,
        notes=notes,
    )
