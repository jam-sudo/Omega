"""Fed vs fasted state PK comparison across BCS classes — Phase 444.

Simulates oral drug pharmacokinetics under fed and fasted conditions using a
one-compartment model with forward Euler integration. Food effects include
delayed gastric emptying (longer lag time) and altered absorption rate/extent.

References
----------
- FDA Guidance: Food-Effect Bioavailability and Fed Bioequivalence Studies. 2002.
- Fleisher D et al. Drug, meal and formulation interactions.
  Clin Pharmacokinet. 1999;36(3):233-54.
- Pouton CW. Formulation of poorly water-soluble drugs.
  Eur J Pharm Sci. 2006;29(3-4):278-87.
"""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = [
    "FedFastedResult",
    "simulate_fed_fasted_pk",
    "bcs_food_effect_prediction",
]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_LAG_FED_H = 1.5    # gastric emptying lag in fed state (hours)
_LAG_FASTED_H = 0.25  # gastric emptying lag in fasted state (hours)

# AUC ratio thresholds for food effect classification
_POSITIVE_THRESHOLD = 1.25
_NEGATIVE_THRESHOLD = 0.80

# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class FedFastedResult:
    """Result from fed/fasted PK comparison simulation."""

    drug_name: str
    bcs_class: int
    dose_mg: float
    times_h: list[float] = field(default_factory=list)
    c_fasted_mg_L: list[float] = field(default_factory=list)
    c_fed_mg_L: list[float] = field(default_factory=list)
    auc_fasted: float = 0.0
    auc_fed: float = 0.0
    auc_ratio_fed_fasted: float = 1.0
    cmax_fasted: float = 0.0
    cmax_fed: float = 0.0
    cmax_ratio: float = 1.0
    tmax_fasted_h: float = 0.0
    tmax_fed_h: float = 0.0
    food_effect_classification: str = "no_effect"
    notes: str = ""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _validate_positive(name: str, value: float) -> None:
    """Raise ValueError if value is not positive."""
    if value <= 0.0:
        raise ValueError(f"{name} must be positive (> 0), got {value}")


def _trapezoidal_auc(times: list[float], conc: list[float]) -> float:
    """Compute AUC0-last using the trapezoidal rule."""
    auc = 0.0
    for i in range(1, len(times)):
        auc += 0.5 * (conc[i] + conc[i - 1]) * (times[i] - times[i - 1])
    return auc


def _simulate_one_cpt_oral(
    dose_mg: float,
    ka_per_h: float,
    ke_per_h: float,
    vd_L: float,
    f: float,
    lag_h: float,
    t_end_h: float,
    dt_h: float,
) -> tuple[list[float], list[float]]:
    """Forward Euler 1-cpt oral PK simulation.

    Compartments:
        A_gut(t)  = drug mass in GI tract (mg)
        C_plasma(t) = plasma concentration (mg/L)

    ODEs:
        dA_gut/dt  = -ka * A_gut   (for t > lag_h, else 0)
        dC/dt      = (ka * A_gut) / Vd - ke * C

    Parameters
    ----------
    dose_mg: absorbed dose (F * nominal_dose handled via f parameter)
    ka_per_h: first-order absorption rate constant
    ke_per_h: first-order elimination rate constant
    vd_L: apparent volume of distribution (L)
    f: oral bioavailability fraction
    lag_h: absorption lag time (h)
    t_end_h: simulation end time (h)
    dt_h: integration step size (h)

    Returns
    -------
    (times, concentrations) — both as list[float]
    """
    n_steps = max(1, int(round(t_end_h / dt_h)))
    times: list[float] = []
    concentrations: list[float] = []

    a_gut = f * dose_mg  # total absorbable drug mass (mg)
    c_plasma = 0.0       # plasma concentration (mg/L)

    t = 0.0
    for _ in range(n_steps + 1):
        times.append(t)
        concentrations.append(max(c_plasma, 0.0))

        # Forward Euler step
        if t >= lag_h and a_gut > 0.0:
            da_gut = -ka_per_h * a_gut
            dc = (ka_per_h * a_gut) / vd_L - ke_per_h * c_plasma
        else:
            da_gut = 0.0
            dc = -ke_per_h * c_plasma

        a_gut = max(a_gut + da_gut * dt_h, 0.0)
        c_plasma = max(c_plasma + dc * dt_h, 0.0)
        t += dt_h

    return times, concentrations


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------


def simulate_fed_fasted_pk(
    drug_name: str,
    bcs_class: int,
    dose_mg: float,
    cl_L_per_h: float,
    vd_L: float,
    fasted_solubility_mg_mL: float,
    fed_solubility_mg_mL: float,
    fasted_ka_per_h: float,
    fed_ka_per_h: float,
    fasted_f: float,
    fed_f: float,
    t_end_h: float = 24.0,
    dt_h: float = 0.05,
) -> FedFastedResult:
    """Simulate and compare oral drug PK in fed and fasted states.

    Uses a one-compartment model with forward Euler integration. Gastric
    emptying delay is applied as a lag time: 0.25 h (fasted) vs 1.5 h (fed).

    Parameters
    ----------
    drug_name:
        Identifier for the drug compound.
    bcs_class:
        BCS classification (1, 2, 3, or 4).
    dose_mg:
        Nominal oral dose in milligrams.
    cl_L_per_h:
        Total body clearance (L/h).
    vd_L:
        Apparent volume of distribution (L).
    fasted_solubility_mg_mL:
        Aqueous solubility in fasted GI fluid (mg/mL).
    fed_solubility_mg_mL:
        Aqueous solubility in fed GI fluid (mg/mL).
    fasted_ka_per_h:
        First-order absorption rate constant under fasted conditions (h⁻¹).
    fed_ka_per_h:
        First-order absorption rate constant under fed conditions (h⁻¹).
    fasted_f:
        Oral bioavailability fraction under fasted conditions (0–1).
    fed_f:
        Oral bioavailability fraction under fed conditions (0–1).
    t_end_h:
        Simulation duration in hours (default 24 h).
    dt_h:
        Forward Euler integration step size in hours (default 0.05 h).

    Returns
    -------
    FedFastedResult
        Both PK profiles and food-effect metrics.

    Raises
    ------
    ValueError
        If any required numeric inputs are out of range.
    """
    # --- Input validation --------------------------------------------------
    if not drug_name or not isinstance(drug_name, str):
        raise ValueError("drug_name must be a non-empty string")
    if bcs_class not in (1, 2, 3, 4):
        raise ValueError(f"bcs_class must be 1, 2, 3 or 4, got {bcs_class}")
    _validate_positive("dose_mg", dose_mg)
    _validate_positive("cl_L_per_h", cl_L_per_h)
    _validate_positive("vd_L", vd_L)
    _validate_positive("fasted_solubility_mg_mL", fasted_solubility_mg_mL)
    _validate_positive("fed_solubility_mg_mL", fed_solubility_mg_mL)
    _validate_positive("fasted_ka_per_h", fasted_ka_per_h)
    _validate_positive("fed_ka_per_h", fed_ka_per_h)
    if not (0.0 < fasted_f <= 1.0):
        raise ValueError(f"fasted_f must be in (0, 1], got {fasted_f}")
    if not (0.0 < fed_f <= 1.0):
        raise ValueError(f"fed_f must be in (0, 1], got {fed_f}")
    _validate_positive("t_end_h", t_end_h)
    _validate_positive("dt_h", dt_h)
    if dt_h >= t_end_h:
        raise ValueError("dt_h must be smaller than t_end_h")

    ke = cl_L_per_h / vd_L

    # --- Simulate fasted state -------------------------------------------
    times_fasted, c_fasted = _simulate_one_cpt_oral(
        dose_mg=dose_mg,
        ka_per_h=fasted_ka_per_h,
        ke_per_h=ke,
        vd_L=vd_L,
        f=fasted_f,
        lag_h=_LAG_FASTED_H,
        t_end_h=t_end_h,
        dt_h=dt_h,
    )

    # --- Simulate fed state ----------------------------------------------
    times_fed, c_fed = _simulate_one_cpt_oral(
        dose_mg=dose_mg,
        ka_per_h=fed_ka_per_h,
        ke_per_h=ke,
        vd_L=vd_L,
        f=fed_f,
        lag_h=_LAG_FED_H,
        t_end_h=t_end_h,
        dt_h=dt_h,
    )

    # Both simulations share the same time grid
    times = times_fasted

    # --- PK metrics -------------------------------------------------------
    auc_fasted = _trapezoidal_auc(times_fasted, c_fasted)
    auc_fed = _trapezoidal_auc(times_fed, c_fed)
    auc_ratio = auc_fed / max(auc_fasted, 1e-12)

    cmax_fasted = max(c_fasted)
    cmax_fed = max(c_fed)
    cmax_ratio = cmax_fed / max(cmax_fasted, 1e-12)

    tmax_fasted_h = times_fasted[c_fasted.index(cmax_fasted)]
    tmax_fed_h = times_fed[c_fed.index(cmax_fed)]

    # --- Food effect classification --------------------------------------
    if auc_ratio > _POSITIVE_THRESHOLD:
        food_effect = "positive"
    elif auc_ratio < _NEGATIVE_THRESHOLD:
        food_effect = "negative"
    else:
        food_effect = "no_effect"

    notes_parts = [
        f"BCS {bcs_class}",
        f"AUC ratio={auc_ratio:.3f}",
        f"Cmax ratio={cmax_ratio:.3f}",
    ]
    notes = "; ".join(notes_parts)

    result = FedFastedResult(
        drug_name=drug_name,
        bcs_class=bcs_class,
        dose_mg=dose_mg,
        times_h=times,
        c_fasted_mg_L=c_fasted,
        c_fed_mg_L=c_fed,
        auc_fasted=auc_fasted,
        auc_fed=auc_fed,
        auc_ratio_fed_fasted=auc_ratio,
        cmax_fasted=cmax_fasted,
        cmax_fed=cmax_fed,
        cmax_ratio=cmax_ratio,
        tmax_fasted_h=tmax_fasted_h,
        tmax_fed_h=tmax_fed_h,
        food_effect_classification=food_effect,
        notes=notes,
    )
    return result


def bcs_food_effect_prediction(
    bcs_class: int,
    logP: float,
    solubility_mg_mL: float,
    dose_mg: float,
) -> dict:
    """Predict expected food effect direction based on BCS class and physicochemistry.

    Uses BCS class, lipophilicity (logP), solubility and dose number to
    predict whether food is likely to increase, decrease or have no effect on
    oral bioavailability.

    Parameters
    ----------
    bcs_class:
        BCS classification (1, 2, 3 or 4).
    logP:
        Logarithm of the octanol/water partition coefficient.
    solubility_mg_mL:
        Aqueous solubility in mg/mL.
    dose_mg:
        Nominal oral dose in milligrams.

    Returns
    -------
    dict with keys:
        - ``bcs_class`` (int)
        - ``predicted_direction`` (str): ``"positive"``, ``"negative"``,
          ``"minimal"``, or ``"variable"``
        - ``mechanism`` (str): brief mechanistic explanation
        - ``confidence`` (str): ``"high"``, ``"moderate"``, or ``"low"``
        - ``recommendation`` (str): brief clinical/regulatory note

    Raises
    ------
    ValueError
        If bcs_class is not 1–4 or dose_mg is not positive.
    """
    if bcs_class not in (1, 2, 3, 4):
        raise ValueError(f"bcs_class must be 1, 2, 3 or 4, got {bcs_class}")
    if dose_mg <= 0.0:
        raise ValueError(f"dose_mg must be positive, got {dose_mg}")
    if not isinstance(logP, (int, float)):
        raise ValueError("logP must be a numeric value")
    if solubility_mg_mL <= 0.0:
        raise ValueError(f"solubility_mg_mL must be positive, got {solubility_mg_mL}")

    # Dose number (Do): dose / (solubility * 250 mL) — unitless
    # Do > 1 indicates dissolution-limited absorption
    dose_number = dose_mg / (solubility_mg_mL * 250.0)

    if bcs_class == 1:
        # High solubility + high permeability → food effect minimal
        direction = "minimal"
        mechanism = (
            "BCS I drugs are highly soluble and permeable; "
            "food has minimal impact on dissolution or permeability."
        )
        confidence = "high"
        recommendation = (
            "Biowaiver eligible. Food effect study may not be required "
            "unless excipients are known to interact."
        )
    elif bcs_class == 2:
        if logP > 3.0:
            # Lipophilic BCS II: food (bile salts, fat) enhances solubilisation
            direction = "positive"
            mechanism = (
                "Lipophilic BCS II drug (logP > 3). Fed-state bile salts "
                "and dietary lipids markedly increase micellar solubilisation, "
                "raising effective luminal concentration and fa."
            )
            confidence = "high"
            recommendation = (
                "Administer with food unless reformulated for fasted-state "
                "delivery. FDA food-effect study strongly recommended."
            )
        else:
            # Low-logP, low-solubility BCS II: variable
            direction = "variable"
            mechanism = (
                "BCS II drug with low-to-moderate logP. Food may increase "
                "solubility via bile salts (positive) but GI motility changes "
                "could reduce absorption rate (negative). Net effect is variable."
            )
            confidence = "moderate"
            recommendation = (
                "Food effect study required. Consider formulation strategies "
                "(e.g., amorphous solid dispersion, nano-milling)."
            )
    elif bcs_class == 3:
        # High solubility + low permeability → food may slow transit (negative)
        direction = "minimal"
        if dose_number < 0.5:
            # Well solubilised; permeability is the rate-limiting step
            mechanism = (
                "BCS III drug: highly soluble but permeability-limited. "
                "Food rarely changes permeability markedly; effect is minimal."
            )
            confidence = "moderate"
        else:
            direction = "negative"
            mechanism = (
                "BCS III drug with borderline dose number. "
                "Food-induced GI motility reduction may decrease absorption "
                "by shortening contact time with absorptive membrane."
            )
            confidence = "low"
        recommendation = (
            "Typically no food restriction required; monitor absorption "
            "window effects for highly soluble/poorly permeable compounds."
        )
    else:
        # BCS IV: low solubility + low permeability → unpredictable
        direction = "variable"
        mechanism = (
            "BCS IV drug has low solubility AND low permeability. "
            "Food can increase solubilisation (positive) while also "
            "reducing intestinal transit rate (negative); effect is highly variable."
        )
        confidence = "low"
        recommendation = (
            "Extensive food-effect characterisation required. "
            "Formulation enhancement (LBF, nano, amorphous) strongly advised."
        )

    return {
        "bcs_class": bcs_class,
        "logP": logP,
        "solubility_mg_mL": solubility_mg_mL,
        "dose_mg": dose_mg,
        "dose_number": round(dose_number, 4),
        "predicted_direction": direction,
        "mechanism": mechanism,
        "confidence": confidence,
        "recommendation": recommendation,
    }
