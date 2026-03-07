"""Drug interactions with nutraceuticals and food components.

Models key food-drug interactions:
- Grapefruit juice: irreversible intestinal CYP3A4 inhibition
- St. John's Wort: CYP3A4/P-gp induction
- Mineral chelation: calcium, magnesium, iron, aluminum, zinc

References
----------
- Dresser GK et al., Clin Pharmacokinet. 2000;38(1):41-57 (GFJ)
- Piscitelli SC et al., Lancet. 2000;355(9203):547-8 (SJW)
- Neuvonen PJ, J Antimicrob Chemother. 1976;2(3):303-9 (chelation)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "NutraDDIResult",
    "ChelationResult",
    "grapefruit_juice_interaction",
    "st_johns_wort_interaction",
    "mineral_chelation",
    "classify_nutra_interaction_risk",
]

# ---------------------------------------------------------------------------
# Mineral chelation constants (dimensionless binding affinity per mineral)
# ---------------------------------------------------------------------------
_MINERAL_MW = {
    "calcium": 40.08,
    "magnesium": 24.31,
    "iron": 55.85,
    "aluminum": 26.98,
    "zinc": 65.38,
}

# K_chelation index: drug_class -> mineral -> K (L/mol)
# tetracyclines > fluoroquinolones > bisphosphonates > others
_K_CHELATION: dict[str, dict[str, float]] = {
    "tetracycline": {
        "calcium": 500.0,
        "magnesium": 400.0,
        "iron": 1000.0,
        "aluminum": 800.0,
        "zinc": 600.0,
    },
    "fluoroquinolone": {
        "calcium": 200.0,
        "magnesium": 150.0,
        "iron": 400.0,
        "aluminum": 300.0,
        "zinc": 180.0,
    },
    "bisphosphonate": {
        "calcium": 800.0,
        "magnesium": 600.0,
        "iron": 700.0,
        "aluminum": 900.0,
        "zinc": 500.0,
    },
    "other": {
        "calcium": 50.0,
        "magnesium": 40.0,
        "iron": 80.0,
        "aluminum": 70.0,
        "zinc": 60.0,
    },
}

_VALID_MINERALS = frozenset(_MINERAL_MW)


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class NutraDDIResult:
    """Result of a nutraceutical-drug interaction simulation.

    Attributes
    ----------
    drug_name : str
    nutraceutical : str
        Name of the food/nutraceutical component.
    dose_mg : float
        Drug dose (mg).
    aucr : float
        AUC ratio (affected / normal). >1 means AUC increased.
    cmax_ratio : float
        Cmax ratio (affected / normal).
    interaction_class : str
        Risk classification string.
    times_h_normal : tuple[float, ...]
        Time points (h) for normal simulation.
    c_plasma_normal : tuple[float, ...]
        Plasma concentration profile without interaction (mg/L).
    times_h_affected : tuple[float, ...]
        Time points (h) for interaction simulation.
    c_plasma_affected : tuple[float, ...]
        Plasma concentration profile with interaction (mg/L).
    notes : str
    """

    drug_name: str
    nutraceutical: str
    dose_mg: float
    aucr: float
    cmax_ratio: float
    interaction_class: str
    times_h_normal: tuple
    c_plasma_normal: tuple
    times_h_affected: tuple
    c_plasma_affected: tuple
    notes: str


@dataclass(frozen=True)
class ChelationResult:
    """Result of mineral-drug chelation assessment.

    Attributes
    ----------
    drug_name : str
    mineral : str
    drug_dose_mg : float
    mineral_dose_mg : float
    fraction_chelated : float
        Fraction of drug dose bound by mineral [0, 1].
    effective_dose_mg : float
        Absorbed drug dose after chelation (mg).
    notes : str
    """

    drug_name: str
    mineral: str
    drug_dose_mg: float
    mineral_dose_mg: float
    fraction_chelated: float
    effective_dose_mg: float
    notes: str


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _validate_pk(cl: float, vd: float, dose: float) -> None:
    if cl <= 0:
        raise ValueError("cl_normal_L_per_h must be > 0")
    if vd <= 0:
        raise ValueError("vd_L must be > 0")
    if dose <= 0:
        raise ValueError("dose_mg must be > 0")


def _validate_fm(fm: float, name: str) -> None:
    if not 0.0 <= fm <= 1.0:
        raise ValueError(f"{name} must be in [0, 1]")


def _simulate_1cpt_oral(
    dose_mg: float,
    ka: float,
    cl: float,
    vd: float,
    fg: float,
    fh: float,
    t_end_h: float,
    dt_h: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Forward Euler 1-cpt oral absorption simulation.

    Parameters
    ----------
    dose_mg : float
    ka : float  Absorption rate (1/h).
    cl : float  Clearance (L/h).
    vd : float  Volume of distribution (L).
    fg : float  Gut bioavailability fraction [0,1].
    fh : float  Hepatic bioavailability fraction [0,1].
    t_end_h : float
    dt_h : float

    Returns
    -------
    times : np.ndarray
    conc_mg_L : np.ndarray
    """
    n = int(round(t_end_h / dt_h)) + 1
    times = np.linspace(0.0, t_end_h, n)
    # Effective oral dose entering systemic circulation
    dose_abs = dose_mg * fg * fh

    # State: [gut_mass_mg, central_mg_L * vd]
    gut = dose_abs  # depot (all dose in gut at t=0 for simplicity)
    central = 0.0  # mg
    conc = np.zeros(n)

    ke = cl / vd

    for i, _ in enumerate(times):
        conc[i] = max(central / vd, 0.0)
        if i < n - 1:
            d_gut = -ka * gut
            d_central = ka * gut - ke * central
            gut = max(gut + d_gut * dt_h, 0.0)
            central = max(central + d_central * dt_h, 0.0)

    return times, conc


def _auc_trapz(times: np.ndarray, conc: np.ndarray) -> float:
    return float(np.trapz(conc, times))


def _cmax(conc: np.ndarray) -> float:
    return float(np.max(conc))


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------

def grapefruit_juice_interaction(
    drug_name: str,
    cl_normal_L_per_h: float,
    vd_L: float,
    fm_cyp3a4: float,
    dose_mg: float,
    gfj_effect_duration_h: float = 4.0,
    ka_per_h: float = 1.5,
    t_end_h: float = 24.0,
    dt_h: float = 0.1,
) -> NutraDDIResult:
    """Simulate grapefruit juice (GFJ) interaction via intestinal CYP3A4 inhibition.

    GFJ irreversibly inhibits intestinal CYP3A4 reducing gut extraction by 60% of
    the fm_cyp3a4_gut contribution. This increases the fraction of dose reaching
    systemic circulation (fg increases).

    Parameters
    ----------
    drug_name : str
    cl_normal_L_per_h : float
        Normal hepatic clearance (L/h). Must be > 0.
    vd_L : float
        Volume of distribution (L). Must be > 0.
    fm_cyp3a4 : float
        Fraction of drug metabolised by CYP3A4 [0, 1].
    dose_mg : float
        Oral dose (mg). Must be > 0.
    gfj_effect_duration_h : float
        Duration of irreversible CYP3A4 inhibition by GFJ (h).
    ka_per_h : float
        First-order absorption rate constant (1/h).
    t_end_h : float
        Simulation end time (h).
    dt_h : float
        Integration step (h).

    Returns
    -------
    NutraDDIResult
    """
    _validate_pk(cl_normal_L_per_h, vd_L, dose_mg)
    _validate_fm(fm_cyp3a4, "fm_cyp3a4")

    # Normal: fg = 1 - gut extraction
    # Gut extraction fraction = fm_cyp3a4 * 0.5 (50% of intestinal CYP3A4 activity)
    eg_normal = fm_cyp3a4 * 0.5
    fg_normal = 1.0 - eg_normal

    # With GFJ: gut CYP3A4 inhibited 60% => gut extraction reduced by 60%
    eg_gfj = eg_normal * (1.0 - 0.6)
    fg_gfj = 1.0 - eg_gfj

    # Hepatic bioavailability (unchanged by GFJ — only intestinal)
    # Simple estimate: fh = 1 - fm_cyp3a4 * 0.3  (hepatic contribution)
    fh = 1.0 - fm_cyp3a4 * 0.3
    fh = max(0.05, min(1.0, fh))

    t_norm, c_norm = _simulate_1cpt_oral(
        dose_mg, ka_per_h, cl_normal_L_per_h, vd_L, fg_normal, fh, t_end_h, dt_h
    )
    t_gfj, c_gfj = _simulate_1cpt_oral(
        dose_mg, ka_per_h, cl_normal_L_per_h, vd_L, fg_gfj, fh, t_end_h, dt_h
    )

    auc_norm = _auc_trapz(t_norm, c_norm)
    auc_gfj = _auc_trapz(t_gfj, c_gfj)
    aucr = auc_gfj / auc_norm if auc_norm > 0 else 1.0

    cmax_norm = _cmax(c_norm)
    cmax_gfj = _cmax(c_gfj)
    cmax_ratio = cmax_gfj / cmax_norm if cmax_norm > 0 else 1.0

    interaction_class = classify_nutra_interaction_risk(aucr, direction="increase")

    notes = (
        f"GFJ reduces intestinal CYP3A4 activity by ~60% for {gfj_effect_duration_h:.1f}h. "
        f"fg: {fg_normal:.3f} -> {fg_gfj:.3f}. AUCR={aucr:.2f}."
    )

    return NutraDDIResult(
        drug_name=drug_name,
        nutraceutical="grapefruit_juice",
        dose_mg=dose_mg,
        aucr=aucr,
        cmax_ratio=cmax_ratio,
        interaction_class=interaction_class,
        times_h_normal=tuple(t_norm.tolist()),
        c_plasma_normal=tuple(c_norm.tolist()),
        times_h_affected=tuple(t_gfj.tolist()),
        c_plasma_affected=tuple(c_gfj.tolist()),
        notes=notes,
    )


def st_johns_wort_interaction(
    drug_name: str,
    cl_normal_L_per_h: float,
    vd_L: float,
    fm_cyp3a4: float,
    fm_pgp: float,
    dose_mg: float,
    induction_days: int = 14,
    ka_per_h: float = 1.5,
    t_end_h: float = 24.0,
    dt_h: float = 0.1,
) -> NutraDDIResult:
    """Simulate St. John's Wort (SJW) interaction via CYP3A4/P-gp induction.

    SJW induces:
    - Hepatic CYP3A4: CL increases by factor (1 + 0.5 * fm_cyp3a4)
    - P-gp efflux: absorption reduced, fa multiplied by (1 - fm_pgp * 0.3)

    Parameters
    ----------
    drug_name : str
    cl_normal_L_per_h : float
        Normal hepatic clearance (L/h). Must be > 0.
    vd_L : float
        Volume of distribution (L). Must be > 0.
    fm_cyp3a4 : float
        Fraction metabolised by CYP3A4 [0, 1].
    fm_pgp : float
        Fraction of absorption limited by P-gp [0, 1].
    dose_mg : float
        Oral dose (mg). Must be > 0.
    induction_days : int
        Days of SJW pre-treatment (full induction assumed at >= 7 days).
    ka_per_h : float
        First-order absorption rate constant (1/h).
    t_end_h : float
        Simulation end time (h).
    dt_h : float
        Integration step (h).

    Returns
    -------
    NutraDDIResult
    """
    _validate_pk(cl_normal_L_per_h, vd_L, dose_mg)
    _validate_fm(fm_cyp3a4, "fm_cyp3a4")
    _validate_fm(fm_pgp, "fm_pgp")

    # Induction magnitude: scales with treatment duration (max effect at >= 14 days)
    induction_factor = min(1.0, induction_days / 14.0)

    # Induced CL: increases by up to 50% of fm_cyp3a4 contribution
    cl_sjw = cl_normal_L_per_h * (1.0 + 0.5 * fm_cyp3a4 * induction_factor)

    # Reduced absorption due to P-gp induction
    fg_sjw = max(0.05, 1.0 - fm_pgp * 0.3 * induction_factor)
    fg_normal = 1.0

    fh = 1.0  # hepatic bioavailability simplified

    t_norm, c_norm = _simulate_1cpt_oral(
        dose_mg, ka_per_h, cl_normal_L_per_h, vd_L, fg_normal, fh, t_end_h, dt_h
    )
    t_sjw, c_sjw = _simulate_1cpt_oral(
        dose_mg, ka_per_h, cl_sjw, vd_L, fg_sjw, fh, t_end_h, dt_h
    )

    auc_norm = _auc_trapz(t_norm, c_norm)
    auc_sjw = _auc_trapz(t_sjw, c_sjw)
    aucr = auc_sjw / auc_norm if auc_norm > 0 else 1.0

    cmax_norm = _cmax(c_norm)
    cmax_sjw = _cmax(c_sjw)
    cmax_ratio = cmax_sjw / cmax_norm if cmax_norm > 0 else 1.0

    interaction_class = classify_nutra_interaction_risk(aucr, direction="decrease")

    notes = (
        f"SJW induces CYP3A4 (+{50 * fm_cyp3a4 * induction_factor:.0f}% CL) "
        f"and P-gp (fg: {fg_normal:.2f} -> {fg_sjw:.3f}). "
        f"AUCR={aucr:.2f} after {induction_days}d treatment."
    )

    return NutraDDIResult(
        drug_name=drug_name,
        nutraceutical="st_johns_wort",
        dose_mg=dose_mg,
        aucr=aucr,
        cmax_ratio=cmax_ratio,
        interaction_class=interaction_class,
        times_h_normal=tuple(t_norm.tolist()),
        c_plasma_normal=tuple(c_norm.tolist()),
        times_h_affected=tuple(t_sjw.tolist()),
        c_plasma_affected=tuple(c_sjw.tolist()),
        notes=notes,
    )


def mineral_chelation(
    drug_name: str,
    drug_dose_mg: float,
    mineral: str,
    mineral_dose_mg: float,
    drug_pka: float,
    drug_class: str = "other",
) -> ChelationResult:
    """Assess mineral-drug chelation reducing effective oral dose.

    fraction_chelated = K_chelation * mineral_moles / (1 + K_chelation * mineral_moles)

    Parameters
    ----------
    drug_name : str
    drug_dose_mg : float
        Drug dose (mg). Must be > 0.
    mineral : str
        One of 'calcium', 'magnesium', 'iron', 'aluminum', 'zinc'.
    mineral_dose_mg : float
        Mineral dose (mg). Must be > 0.
    drug_pka : float
        Drug pKa (used for informational notes; chelation primarily structural).
    drug_class : str
        Drug class for K_chelation lookup: 'tetracycline', 'fluoroquinolone',
        'bisphosphonate', 'other'. Defaults to 'other'.

    Returns
    -------
    ChelationResult
    """
    if drug_dose_mg <= 0:
        raise ValueError("drug_dose_mg must be > 0")
    if mineral_dose_mg <= 0:
        raise ValueError("mineral_dose_mg must be > 0")
    if mineral not in _VALID_MINERALS:
        raise ValueError(
            f"mineral must be one of {sorted(_VALID_MINERALS)}, got '{mineral}'"
        )

    # Normalize drug_class to known keys
    drug_class_key = drug_class.lower() if drug_class.lower() in _K_CHELATION else "other"
    k_chel = _K_CHELATION[drug_class_key][mineral]

    mw_mineral = _MINERAL_MW[mineral]
    mineral_moles = (mineral_dose_mg / 1000.0) / mw_mineral  # mol

    fraction_chelated = k_chel * mineral_moles / (1.0 + k_chel * mineral_moles)
    fraction_chelated = max(0.0, min(1.0, fraction_chelated))
    effective_dose_mg = drug_dose_mg * (1.0 - fraction_chelated)

    severity = (
        "high" if fraction_chelated > 0.5
        else "moderate" if fraction_chelated > 0.2
        else "low"
    )

    notes = (
        f"{drug_class_key.capitalize()} class drug with pKa={drug_pka:.1f}. "
        f"Chelation with {mineral}: K={k_chel:.0f} L/mol, "
        f"mineral_moles={mineral_moles:.4f} mol. "
        f"Fraction chelated={fraction_chelated:.3f} ({severity} severity). "
        f"Effective dose={effective_dose_mg:.1f} mg."
    )

    return ChelationResult(
        drug_name=drug_name,
        mineral=mineral,
        drug_dose_mg=drug_dose_mg,
        mineral_dose_mg=mineral_dose_mg,
        fraction_chelated=fraction_chelated,
        effective_dose_mg=effective_dose_mg,
        notes=notes,
    )


def classify_nutra_interaction_risk(aucr: float, direction: str = "increase") -> str:
    """Classify nutraceutical interaction risk based on AUCR.

    Parameters
    ----------
    aucr : float
        AUC ratio (affected / control).
    direction : str
        Expected direction: 'increase' (inhibition) or 'decrease' (induction).

    Returns
    -------
    str
        'significant_increase', 'significant_decrease', 'moderate', or 'negligible'.
    """
    if direction not in ("increase", "decrease"):
        raise ValueError("direction must be 'increase' or 'decrease'")

    if direction == "increase":
        if aucr >= 2.0:
            return "significant_increase"
        if aucr >= 1.25:
            return "moderate"
        return "negligible"
    else:  # decrease
        if aucr <= 0.5:
            return "significant_decrease"
        if aucr <= 0.8:
            return "moderate"
        return "negligible"
