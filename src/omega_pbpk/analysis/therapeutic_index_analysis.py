"""Therapeutic index and safety margin analysis (Phase 432).

Analyzes therapeutic index (TI), safety margins, and population variability
for drug candidates to support early-stage risk stratification.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

__all__ = [
    "TherapeuticIndexResult",
    "TIPopulationResult",
    "compute_therapeutic_index",
    "simulate_ti_population",
]

# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TherapeuticIndexResult:
    """Therapeutic index and safety margin analysis result."""

    drug_name: str
    ed50_mg: float  # ED50 by dose
    td50_mg: float  # TD50 by dose
    ti_dose: float  # TI_dose = TD50 / ED50
    ti_auc: float  # TI_auc = TD50_AUC / ED50_AUC
    safety_margin_pct: float  # (MTC - MEC) / MEC * 100
    is_narrow_ti: bool  # True if TI_dose < 2 or TI_auc < 2
    monitoring_recommendation: str
    notes: str


@dataclass
class TIPopulationResult:
    """Population-level therapeutic index simulation result."""

    drug_name: str
    n_subjects: int
    dose_mg: float
    pct_effective: float  # fraction with dose > ED50_i
    pct_toxic: float  # fraction with dose > TD50_i
    therapeutic_window_pct: float  # pct_effective - pct_toxic
    individual_ed50: list[float]
    individual_td50: list[float]
    notes: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _validate_ti_inputs(
    drug_name: str,
    ed50_mg: float,
    td50_mg: float,
    ed50_auc: float,
    td50_auc: float,
    therapeutic_range_mg_L: tuple[float, float],
) -> None:
    """Raise ValueError for invalid compute_therapeutic_index inputs."""
    if not drug_name or not drug_name.strip():
        raise ValueError("drug_name must be a non-empty string.")
    if ed50_mg <= 0:
        raise ValueError("ed50_mg must be > 0.")
    if td50_mg <= 0:
        raise ValueError("td50_mg must be > 0.")
    if td50_mg <= ed50_mg:
        raise ValueError("td50_mg must be > ed50_mg (TD50 > ED50 expected).")
    if ed50_auc <= 0:
        raise ValueError("ed50_auc must be > 0.")
    if td50_auc <= 0:
        raise ValueError("td50_auc must be > 0.")
    if td50_auc <= ed50_auc:
        raise ValueError("td50_auc must be > ed50_auc.")
    if len(therapeutic_range_mg_L) != 2:
        raise ValueError("therapeutic_range_mg_L must be a 2-element sequence.")
    mec, mtc = therapeutic_range_mg_L
    if mec <= 0:
        raise ValueError("therapeutic_range_mg_L lower bound (MEC) must be > 0.")
    if mtc <= mec:
        raise ValueError(
            "therapeutic_range_mg_L upper bound (MTC) must be > lower bound (MEC)."
        )


def _validate_population_inputs(
    drug_name: str,
    ed50_mg: float,
    td50_mg: float,
    n_subjects: int,
    ed50_cv_pct: float,
    td50_cv_pct: float,
    dose_mg: float,
) -> None:
    """Raise ValueError for invalid simulate_ti_population inputs."""
    if not drug_name or not drug_name.strip():
        raise ValueError("drug_name must be a non-empty string.")
    if ed50_mg <= 0:
        raise ValueError("ed50_mg must be > 0.")
    if td50_mg <= 0:
        raise ValueError("td50_mg must be > 0.")
    if td50_mg <= ed50_mg:
        raise ValueError("td50_mg must be > ed50_mg.")
    if n_subjects < 1:
        raise ValueError("n_subjects must be >= 1.")
    if ed50_cv_pct < 0:
        raise ValueError("ed50_cv_pct must be >= 0.")
    if td50_cv_pct < 0:
        raise ValueError("td50_cv_pct must be >= 0.")
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0.")


# ---------------------------------------------------------------------------
# Box-Muller normal random number generator (no numpy/scipy)
# ---------------------------------------------------------------------------


def _box_muller_normals(n: int, seed: int) -> list[float]:
    """Generate n standard normal samples using Box-Muller transform.

    Uses a simple LCG (linear congruential generator) for reproducible
    uniform samples, then applies Box-Muller.
    """
    # LCG parameters (Numerical Recipes)
    _MOD = 2**32
    _A = 1664525
    _C = 1013904223

    state = seed & 0xFFFFFFFF
    uniforms: list[float] = []

    # Need 2*ceil(n/2) uniform samples for n normals
    needed = 2 * math.ceil(n / 2) + 2
    for _ in range(needed):
        state = (_A * state + _C) % _MOD
        # Map to (0, 1) — avoid exact 0 or 1
        u = (state + 0.5) / _MOD
        uniforms.append(u)

    normals: list[float] = []
    for i in range(0, len(uniforms) - 1, 2):
        u1 = uniforms[i]
        u2 = uniforms[i + 1]
        # Clamp u1 away from 0 to avoid log(0)
        u1 = max(u1, 1e-15)
        z0 = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
        z1 = math.sqrt(-2.0 * math.log(u1)) * math.sin(2.0 * math.pi * u2)
        normals.append(z0)
        normals.append(z1)

    return normals[:n]


def _lognormal_samples(mean: float, cv_pct: float, n: int, seed: int) -> list[float]:
    """Generate n log-normal samples with given mean and CV (%).

    If cv_pct == 0, returns [mean] * n (deterministic).
    """
    if cv_pct == 0.0:
        return [mean] * n

    cv = cv_pct / 100.0
    sigma_ln = math.sqrt(math.log(1.0 + cv * cv))
    mu_ln = math.log(mean) - 0.5 * sigma_ln * sigma_ln

    z_vals = _box_muller_normals(n, seed)
    return [math.exp(mu_ln + sigma_ln * z) for z in z_vals]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_therapeutic_index(
    drug_name: str,
    ed50_mg: float,
    td50_mg: float,
    ed50_auc: float,
    td50_auc: float,
    therapeutic_range_mg_L: tuple[float, float],
) -> TherapeuticIndexResult:
    """Compute therapeutic index and safety margins.

    Parameters
    ----------
    drug_name:
        Drug identifier.
    ed50_mg:
        Dose producing 50% efficacy response (mg).
    td50_mg:
        Dose producing 50% toxic response (mg).
    ed50_auc:
        AUC at 50% efficacy (mg*h/L).
    td50_auc:
        AUC at 50% toxicity (mg*h/L).
    therapeutic_range_mg_L:
        (MEC, MTC) — minimum effective and minimum toxic concentration (mg/L).

    Returns
    -------
    TherapeuticIndexResult
    """
    _validate_ti_inputs(
        drug_name, ed50_mg, td50_mg, ed50_auc, td50_auc, therapeutic_range_mg_L
    )

    ti_dose = td50_mg / ed50_mg
    ti_auc = td50_auc / ed50_auc

    mec, mtc = therapeutic_range_mg_L
    safety_margin_pct = (mtc - mec) / mec * 100.0

    is_narrow_ti = ti_dose < 2.0 or ti_auc < 2.0

    # Monitoring recommendation
    if is_narrow_ti:
        monitoring = (
            "Narrow therapeutic index — TDM (therapeutic drug monitoring) strongly "
            "recommended; consider PGx screening and dose individualization."
        )
    elif ti_dose < 5.0:
        monitoring = (
            "Moderate TI — routine clinical monitoring recommended; "
            "dose adjustment for organ impairment."
        )
    else:
        monitoring = (
            "Wide therapeutic index — standard dosing monitoring is appropriate."
        )

    notes_parts: list[str] = []
    notes_parts.append(f"TI_dose = {ti_dose:.2f}, TI_AUC = {ti_auc:.2f}.")
    notes_parts.append(
        f"Safety margin = {safety_margin_pct:.1f}% "
        f"(MEC={mec} mg/L, MTC={mtc} mg/L)."
    )
    if is_narrow_ti:
        notes_parts.append(
            "Drug classified as narrow TI per FDA guidance (TI < 2)."
        )

    return TherapeuticIndexResult(
        drug_name=drug_name,
        ed50_mg=ed50_mg,
        td50_mg=td50_mg,
        ti_dose=ti_dose,
        ti_auc=ti_auc,
        safety_margin_pct=safety_margin_pct,
        is_narrow_ti=is_narrow_ti,
        monitoring_recommendation=monitoring,
        notes=" ".join(notes_parts),
    )


def simulate_ti_population(
    drug_name: str,
    ed50_mg: float,
    td50_mg: float,
    n_subjects: int = 1000,
    ed50_cv_pct: float = 30.0,
    td50_cv_pct: float = 20.0,
    dose_mg: float | None = None,
    seed: int = 42,
) -> TIPopulationResult:
    """Simulate population variability in ED50 and TD50 using log-normal distributions.

    Uses Box-Muller transform (no numpy) to generate standard normal deviates,
    then transforms to log-normal to reflect typical PK/PD variability.

    Parameters
    ----------
    drug_name:
        Drug identifier.
    ed50_mg:
        Population mean ED50 (mg).
    td50_mg:
        Population mean TD50 (mg).
    n_subjects:
        Number of virtual subjects to simulate.
    ed50_cv_pct:
        Coefficient of variation for ED50 (%).
    td50_cv_pct:
        Coefficient of variation for TD50 (%).
    dose_mg:
        Dose to evaluate efficacy/toxicity against.  Defaults to geometric
        mean of ED50 and TD50 (midpoint of therapeutic range).
    seed:
        Random seed for reproducibility.

    Returns
    -------
    TIPopulationResult
    """
    if dose_mg is None:
        dose_mg = math.sqrt(ed50_mg * td50_mg)

    _validate_population_inputs(
        drug_name, ed50_mg, td50_mg, n_subjects, ed50_cv_pct, td50_cv_pct, dose_mg
    )

    # Generate individual ED50 and TD50 using log-normal sampling
    ind_ed50 = _lognormal_samples(ed50_mg, ed50_cv_pct, n_subjects, seed)
    ind_td50 = _lognormal_samples(td50_mg, td50_cv_pct, n_subjects, seed + 1)

    # Count efficacy and toxicity
    n_effective = sum(1 for e in ind_ed50 if dose_mg > e)
    n_toxic = sum(1 for t in ind_td50 if dose_mg > t)

    pct_effective = n_effective / n_subjects
    pct_toxic = n_toxic / n_subjects
    therapeutic_window_pct = pct_effective - pct_toxic

    notes: list[str] = []
    if therapeutic_window_pct < 0:
        notes.append(
            "Negative therapeutic window: more subjects are toxic than effective "
            "at this dose — dose is too high."
        )
    elif therapeutic_window_pct < 0.10:
        notes.append(
            "Narrow therapeutic window in population (<10% therapeutic) — "
            "high risk of toxicity."
        )
    if pct_effective < 0.5:
        notes.append("Less than 50% of subjects are expected to respond at this dose.")
    if pct_toxic > 0.1:
        notes.append(
            f"{pct_toxic * 100:.1f}% of subjects expected to experience toxicity."
        )

    return TIPopulationResult(
        drug_name=drug_name,
        n_subjects=n_subjects,
        dose_mg=dose_mg,
        pct_effective=pct_effective,
        pct_toxic=pct_toxic,
        therapeutic_window_pct=therapeutic_window_pct,
        individual_ed50=ind_ed50,
        individual_td50=ind_td50,
        notes=notes,
    )
