"""Food-drug interaction simulation — Phase 509."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

__all__ = [
    "FoodEffectResult",
    "FoodInteractionSimResult",
    "predict_food_effect",
    "gastric_ph_effect",
    "fat_meal_enhancement",
    "simulate_fasted_vs_fed",
]


@dataclass(frozen=True)
class FoodEffectResult:
    """Predicted food effect on oral drug PK."""

    drug_name: str
    logP: float
    auc_ratio_fed_fasted: float
    cmax_ratio_fed_fasted: float
    tmax_difference_h: float  # fed - fasted; positive = delayed
    food_effect_category: str  # 'no_effect', 'moderate', 'high'
    recommendation: str
    notes: str


@dataclass
class FoodInteractionSimResult:
    """Simulated fasted vs fed PK profiles."""

    drug_name: str
    times_h: list = field(default_factory=list)
    c_fasted_mg_L: list = field(default_factory=list)
    c_fed_mg_L: list = field(default_factory=list)
    auc_fasted: float = 0.0
    auc_fed: float = 0.0
    cmax_fasted: float = 0.0
    cmax_fed: float = 0.0
    notes: str = ""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _trapz(y: list[float], x: list[float]) -> float:
    """Trapezoidal integration (pure Python)."""
    total = 0.0
    for i in range(len(x) - 1):
        total += 0.5 * (y[i] + y[i + 1]) * (x[i + 1] - x[i])
    return total


def _logistic(x: float, x0: float = 0.0, k: float = 1.0) -> float:
    """Logistic (sigmoid) function."""
    return 1.0 / (1.0 + math.exp(-k * (x - x0)))


def _fat_absorption_fold(logP: float, solubility_mg_mL: float) -> float:
    """Fold change in AUC due to fat-meal enhancement.

    High logP (>3): fat meal increases absorption markedly via micellar
    solubilisation.  Low logP: minimal effect.

    Returns a value >= 1.0.
    """
    if logP < 1.0:
        return 1.0

    # Solubility-limited floor: very soluble drugs already fully absorbed
    solubility_factor = max(0.2, 1.0 - math.log10(max(solubility_mg_mL, 0.001)) * 0.1)

    # Sigmoid from ~1× at logP=1 to ~3× at logP=5
    fold = 1.0 + 2.0 * _logistic(logP, x0=3.0, k=1.2) * solubility_factor
    return fold


def _ph_absorption_factor(
    pka: float | None, drug_type: str, fasted_ph: float, fed_ph: float
) -> float:
    """Ratio of absorbed fraction fed/fasted due to pH change.

    For basic drugs (pKa 6-9): higher fed pH reduces ionisation fraction
    less, but reduces solubility → lower fa.  Model: AUC ratio 0.5–0.8.
    For acid drugs (pKa 3-6): higher fed pH increases ionisation → higher
    solubility → higher fa.  Model: AUC ratio 1.1–1.3.
    Neutral: 1.0.
    """
    if pka is None or drug_type == "neutral":
        return 1.0

    delta_ph = fed_ph - fasted_ph  # positive (fed is less acidic)

    if drug_type == "base":
        # pKa 6-9 range: effect is strongest
        if 6.0 <= pka <= 9.0:
            # At fed pH, basic drug is less ionized → less soluble in GI
            # Effect scales with how much pKa is near fed pH
            proximity = 1.0 - abs(pka - fed_ph) / 5.0
            proximity = max(0.0, min(proximity, 1.0))
            reduction = 0.3 * proximity * (delta_ph / 3.5)  # up to 30% reduction
            return max(0.5, 1.0 - reduction)
        return 1.0 - 0.1 * (delta_ph / 3.5)

    if drug_type == "acid":
        # pKa 3-6 range: fed pH increases ionisation → better solubility
        if 3.0 <= pka <= 6.0:
            proximity = 1.0 - abs(pka - fasted_ph) / 4.0
            proximity = max(0.0, min(proximity, 1.0))
            increase = 0.2 * proximity * (delta_ph / 3.5)
            return min(1.3, 1.0 + increase)
        return 1.0 + 0.05 * (delta_ph / 3.5)

    return 1.0


def _tmax_delay_h(logP: float, pka: float | None, drug_type: str) -> float:
    """Estimate Tmax delay (hours) in fed vs fasted state.

    Delayed gastric emptying contributes +1-2 h.
    """
    base_delay = 1.0  # delayed gastric emptying
    # Lipophilic drug: more delayed due to fat digestion
    if logP > 3.0:
        base_delay += 0.5
    # Basic drugs: precipitation/re-dissolution may add delay
    if drug_type == "base" and pka is not None and 6.0 <= pka <= 9.0:
        base_delay += 0.5
    return min(base_delay, 2.0)


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------


def fat_meal_enhancement(logP: float, solubility_mg_mL: float) -> float:
    """Fold change in AUC for a fat meal vs fasted state.

    Parameters
    ----------
    logP:
        Lipophilicity.
    solubility_mg_mL:
        Aqueous solubility (mg/mL). Must be > 0.

    Returns
    -------
    float
        Fold change >= 1.0.  Higher logP gives larger enhancement.

    Raises
    ------
    ValueError
        If ``solubility_mg_mL`` <= 0.
    """
    if solubility_mg_mL <= 0:
        raise ValueError(f"solubility_mg_mL must be positive, got {solubility_mg_mL}")
    return _fat_absorption_fold(logP, solubility_mg_mL)


def gastric_ph_effect(
    pka: float,
    drug_type: str,
    fasted_ph: float = 1.5,
    fed_ph: float = 5.0,
) -> dict:
    """Assess the impact of gastric pH change (fasted → fed) on drug solubility.

    Uses Henderson-Hasselbalch solubility model:
    - Weak acid:  S = S0 * (1 + 10^(pH - pKa))  → higher pH increases solubility
    - Weak base:  S = S0 * (1 + 10^(pKa - pH))  → lower pH increases solubility

    Parameters
    ----------
    pka:
        Drug pKa.
    drug_type:
        One of 'neutral', 'acid', 'base'.
    fasted_ph:
        Gastric pH in fasted state (default 1.5).
    fed_ph:
        Gastric pH in fed state (default 5.0).

    Returns
    -------
    dict with keys:
        - ``fasted_ionized_fraction`` (0–1)
        - ``fed_ionized_fraction`` (0–1)
        - ``solubility_ratio_fed_fasted`` (fed solubility / fasted)
        - ``absorption_impact`` ('increased', 'decreased', 'unchanged')

    Raises
    ------
    ValueError
        If ``drug_type`` is not one of 'neutral', 'acid', 'base'.
    """
    valid_types = {"neutral", "acid", "base"}
    if drug_type not in valid_types:
        raise ValueError(f"drug_type must be one of {valid_types}, got {drug_type!r}")
    if fasted_ph <= 0:
        raise ValueError(f"fasted_ph must be positive, got {fasted_ph}")
    if fed_ph <= 0:
        raise ValueError(f"fed_ph must be positive, got {fed_ph}")

    if drug_type == "acid":
        # H-H: ionized fraction of weak acid at given pH
        # fraction_ionized = 1 / (1 + 10^(pKa - pH))
        def _ionized_acid(ph: float) -> float:
            exp_val = pka - ph
            if exp_val > 30:
                return 0.0
            if exp_val < -30:
                return 1.0
            return 1.0 / (1.0 + 10**exp_val)

        fi_fasted = _ionized_acid(fasted_ph)
        fi_fed = _ionized_acid(fed_ph)

        # H-H solubility for weak acid: S = S0 * (1 + 10^(pH - pKa))
        # Ratio fed/fasted (capped to [0.1, 20] for realism)
        s_fasted = 1.0 + 10 ** min(fasted_ph - pka, 10.0)
        s_fed = 1.0 + 10 ** min(fed_ph - pka, 10.0)
        raw_ratio = s_fed / max(s_fasted, 1e-30)
        solubility_ratio = max(0.1, min(raw_ratio, 20.0))
        impact = "increased" if fi_fed > fi_fasted + 0.02 else "unchanged"

    elif drug_type == "base":
        # H-H: ionized fraction of weak base at given pH
        # fraction_ionized = 1 / (1 + 10^(pH - pKa))
        def _ionized_base(ph: float) -> float:
            exp_val = ph - pka
            if exp_val > 30:
                return 0.0
            if exp_val < -30:
                return 1.0
            return 1.0 / (1.0 + 10**exp_val)

        fi_fasted = _ionized_base(fasted_ph)
        fi_fed = _ionized_base(fed_ph)

        # H-H solubility for weak base: S = S0 * (1 + 10^(pKa - pH))
        # Fasted (low pH) → much higher solubility; Fed (higher pH) → lower
        s_fasted = 1.0 + 10 ** min(pka - fasted_ph, 10.0)
        s_fed = 1.0 + 10 ** min(pka - fed_ph, 10.0)
        raw_ratio = s_fed / max(s_fasted, 1e-30)
        # Cap to realistic range; for basic drug ratio < 1.0 when pH increases
        solubility_ratio = max(1e-3, min(raw_ratio, 10.0))
        impact = "decreased" if fi_fasted > fi_fed + 0.02 else "unchanged"

    else:
        fi_fasted = 0.0
        fi_fed = 0.0
        solubility_ratio = 1.0
        impact = "unchanged"

    return {
        "fasted_ionized_fraction": fi_fasted,
        "fed_ionized_fraction": fi_fed,
        "solubility_ratio_fed_fasted": solubility_ratio,
        "absorption_impact": impact,
    }


def predict_food_effect(
    drug_name: str,
    logP: float,
    mw: float,
    pka: float | None = None,
    drug_type: str = "neutral",
    solubility_mg_mL: float = 1.0,
    cl_L_per_h: float = 10.0,
    vd_L: float = 70.0,
    dose_mg: float = 100.0,
) -> FoodEffectResult:
    """Predict food effect on oral drug PK (Cmax, AUC, Tmax changes).

    Parameters
    ----------
    drug_name:
        Drug identifier.
    logP:
        Lipophilicity.
    mw:
        Molecular weight (g/mol). Must be > 0.
    pka:
        Drug pKa (optional).
    drug_type:
        One of 'neutral', 'acid', 'base'.
    solubility_mg_mL:
        Aqueous solubility (mg/mL). Must be > 0.
    cl_L_per_h:
        Clearance (L/h). Must be > 0.
    vd_L:
        Volume of distribution (L). Must be > 0.
    dose_mg:
        Dose (mg). Must be > 0.

    Returns
    -------
    FoodEffectResult
    """
    if mw <= 0:
        raise ValueError(f"mw must be positive, got {mw}")
    if solubility_mg_mL <= 0:
        raise ValueError(f"solubility_mg_mL must be positive, got {solubility_mg_mL}")
    if cl_L_per_h <= 0:
        raise ValueError(f"cl_L_per_h must be positive, got {cl_L_per_h}")
    if vd_L <= 0:
        raise ValueError(f"vd_L must be positive, got {vd_L}")
    if dose_mg <= 0:
        raise ValueError(f"dose_mg must be positive, got {dose_mg}")

    fasted_ph = 1.5
    fed_ph = 5.0

    # Fat-meal fold change (logP-driven)
    fat_fold = _fat_absorption_fold(logP, solubility_mg_mL)

    # pH effect on absorption fraction
    ph_factor = _ph_absorption_factor(pka, drug_type, fasted_ph, fed_ph)

    # Combined AUC ratio
    auc_ratio = fat_fold * ph_factor

    # Cmax ratio: fed delays absorption, so Cmax typically lower than AUC ratio
    # Use a damping factor (0.85 of AUC ratio, but capped)
    cmax_ratio = auc_ratio * 0.85

    # Tmax delay
    tmax_diff = _tmax_delay_h(logP, pka, drug_type)

    # Categorise
    if auc_ratio >= 1.5 or auc_ratio <= 0.67:
        category = "high"
    elif abs(auc_ratio - 1.0) >= 0.2:
        category = "moderate"
    else:
        category = "no_effect"

    # Recommendation
    if category == "high" and auc_ratio > 1.0:
        recommendation = "Administer with food to improve absorption."
    elif category == "high" and auc_ratio < 1.0:
        recommendation = "Administer in fasted state; food reduces absorption."
    elif category == "moderate":
        recommendation = "Consistent administration with/without food recommended."
    else:
        recommendation = "May be taken without regard to food."

    notes_parts = [f"logP={logP:.1f}"]
    if pka is not None:
        notes_parts.append(f"pKa={pka:.1f} ({drug_type})")
    notes_parts.append(f"fat_fold={fat_fold:.2f}")
    notes_parts.append(f"ph_factor={ph_factor:.2f}")
    notes = "; ".join(notes_parts)

    return FoodEffectResult(
        drug_name=drug_name,
        logP=logP,
        auc_ratio_fed_fasted=auc_ratio,
        cmax_ratio_fed_fasted=cmax_ratio,
        tmax_difference_h=tmax_diff,
        food_effect_category=category,
        recommendation=recommendation,
        notes=notes,
    )


def simulate_fasted_vs_fed(
    drug_name: str,
    logP: float,
    mw: float,
    solubility_mg_mL: float,
    cl_L_per_h: float,
    vd_L: float,
    dose_mg: float,
    pka: float | None = None,
    drug_type: str = "neutral",
    t_end_h: float = 12.0,
    dt_h: float = 0.1,
) -> FoodInteractionSimResult:
    """Simulate fasted and fed PK profiles using a 1-compartment model.

    Uses forward Euler integration.  The fed state modifies:
    - Absorption rate constant (ka): slowed by delayed gastric emptying
    - Bioavailability fraction (F): modified by fat-meal & pH effects

    Parameters
    ----------
    drug_name:
        Drug identifier.
    logP:
        Lipophilicity.
    mw:
        Molecular weight. Must be > 0.
    solubility_mg_mL:
        Aqueous solubility (mg/mL). Must be > 0.
    cl_L_per_h:
        Clearance (L/h). Must be > 0.
    vd_L:
        Volume of distribution (L). Must be > 0.
    dose_mg:
        Dose (mg). Must be > 0.
    pka:
        Drug pKa (optional).
    drug_type:
        One of 'neutral', 'acid', 'base'.
    t_end_h:
        Simulation end time (hours).
    dt_h:
        Integration time step (hours).

    Returns
    -------
    FoodInteractionSimResult
    """
    if mw <= 0:
        raise ValueError(f"mw must be positive, got {mw}")
    if solubility_mg_mL <= 0:
        raise ValueError(f"solubility_mg_mL must be positive, got {solubility_mg_mL}")
    if cl_L_per_h <= 0:
        raise ValueError(f"cl_L_per_h must be positive, got {cl_L_per_h}")
    if vd_L <= 0:
        raise ValueError(f"vd_L must be positive, got {vd_L}")
    if dose_mg <= 0:
        raise ValueError(f"dose_mg must be positive, got {dose_mg}")
    if dt_h <= 0:
        raise ValueError(f"dt_h must be positive, got {dt_h}")
    if t_end_h <= 0:
        raise ValueError(f"t_end_h must be positive, got {t_end_h}")

    ke = cl_L_per_h / vd_L  # elimination rate constant (1/h)

    # Fasted state: ka_fasted, F_fasted=1.0 baseline
    ka_fasted = 1.0  # 1/h (typical moderate absorption)
    F_fasted = 1.0

    # Fed state modifications
    fat_fold = _fat_absorption_fold(logP, solubility_mg_mL)
    ph_factor = _ph_absorption_factor(pka, drug_type, 1.5, 5.0)
    F_fed = F_fasted * fat_fold * ph_factor

    # Delayed gastric emptying: ka slowed
    tmax_delay = _tmax_delay_h(logP, pka, drug_type)
    # ka_fed reduced (delay factor): higher delay → lower ka
    delay_factor = 1.0 / (1.0 + tmax_delay * 0.5)
    ka_fed = ka_fasted * delay_factor

    n_steps = max(int(t_end_h / dt_h), 1)
    times_h: list[float] = []
    c_fasted: list[float] = []
    c_fed: list[float] = []

    # Fasted simulation
    gut_f = dose_mg * F_fasted
    central_f = 0.0
    for i in range(n_steps + 1):
        t = i * dt_h
        times_h.append(t)
        c_fasted.append(central_f / vd_L)
        if i < n_steps:
            abs_rate = ka_fasted * gut_f
            elim_rate = ke * central_f
            gut_f = max(0.0, gut_f - abs_rate * dt_h)
            central_f = max(0.0, central_f + (abs_rate - elim_rate) * dt_h)

    # Fed simulation (separate pass)
    gut_fed = dose_mg * F_fed
    central_fed = 0.0
    for i in range(n_steps + 1):
        c_mg_L = central_fed / vd_L
        c_fed.append(c_mg_L)
        if i < n_steps:
            abs_rate = ka_fed * gut_fed
            elim_rate = ke * central_fed
            gut_fed = max(0.0, gut_fed - abs_rate * dt_h)
            central_fed = max(0.0, central_fed + (abs_rate - elim_rate) * dt_h)

    auc_fasted = _trapz(c_fasted, times_h)
    auc_fed = _trapz(c_fed, times_h)
    cmax_fasted = max(c_fasted)
    cmax_fed = max(c_fed)

    notes = (
        f"ka_fasted={ka_fasted:.2f}/h, ka_fed={ka_fed:.2f}/h; "
        f"F_fed/F_fasted={F_fed:.2f}; fat_fold={fat_fold:.2f}"
    )

    return FoodInteractionSimResult(
        drug_name=drug_name,
        times_h=times_h,
        c_fasted_mg_L=c_fasted,
        c_fed_mg_L=c_fed,
        auc_fasted=auc_fasted,
        auc_fed=auc_fed,
        cmax_fasted=cmax_fasted,
        cmax_fed=cmax_fed,
        notes=notes,
    )
