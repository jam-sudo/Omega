"""Chemical degradation kinetics prediction for drug stability assessment.

Predicts first-order degradation rate constants for hydrolysis, oxidation,
and photolysis pathways. Follows ICH Q1A guidance for shelf-life estimation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = [
    "DegradationResult",
    "predict_degradation",
    "screen_storage_conditions",
]

# Physical constants
_EA_KJ_PER_MOL = 80.0  # Activation energy (kJ/mol)
_R_KJ_PER_MOL_K = 8.314e-3  # Gas constant (kJ/mol/K)
_T_REF_K = 298.15  # Reference temperature (25°C in K)
_HOURS_PER_YEAR = 8760.0
_HOURS_PER_MONTH = 24.0 * 30.0


@dataclass(frozen=True)
class DegradationResult:
    """Result of drug degradation kinetics prediction."""

    drug_name: str
    degradation_pathway: str  # "hydrolysis", "oxidation", "photolysis", "combined"
    rate_constant_per_h: float  # k_deg (first-order, /h)
    t_half_h: float  # ln2 / k_deg
    t90_h: float  # time to 90% remaining = ln(10/9) / k_deg
    shelf_life_months: float  # time to 95% remaining / (24*30)
    activation_energy_kJ_per_mol: float
    q10_factor: float  # rate increase per 10 C
    concentration_at_1yr: float  # fraction remaining after 1 year
    degradation_products: str  # comma-separated predicted products
    risk_level: str  # "low", "moderate", "high"
    ich_q1a_compliant: bool  # True if shelf_life_months >= 24
    notes: str


def _base_rate_hydrolysis(ph: float, pka: float, relative_humidity_pct: float) -> float:
    """First-order hydrolysis rate constant at 25°C (1/h)."""
    k_base = 0.001 * (1.0 + abs(ph - 7.0) / 7.0) * (1.0 + relative_humidity_pct / 100.0)

    # Acidic drugs (pKa < 5): faster at low pH
    if pka < 5.0:
        k_base *= 1.0 + 2.0 * max(0.0, 7.0 - ph) / 7.0

    # Basic drugs (pKa > 9): faster at high pH
    if pka > 9.0:
        k_base *= 1.0 + 2.0 * max(0.0, ph - 7.0) / 7.0

    return k_base


def _base_rate_oxidation(logp: float) -> float:
    """First-order oxidation rate constant at 25°C (1/h)."""
    if logp > 0:
        k_base = 0.0005 * (1.0 - logp / 10.0)
    else:
        k_base = 0.0005 * 1.0
    # Clamp to [1e-6, 0.01]
    return max(1e-6, min(0.01, k_base))


def _base_rate_photolysis(logp: float) -> float:
    """First-order photolysis rate constant at 25°C (1/h)."""
    if logp > 0:
        k_base = 0.002 * (1.0 - logp / 8.0)
    else:
        k_base = 0.002 * 1.0
    # Clamp to [1e-6, 0.005]
    return max(1e-6, min(0.005, k_base))


def _temperature_factor(storage_temp_celsius: float) -> float:
    """Arrhenius temperature correction factor relative to 25°C."""
    t_k = storage_temp_celsius + 273.15
    return math.exp(-_EA_KJ_PER_MOL / _R_KJ_PER_MOL_K * (1.0 / t_k - 1.0 / _T_REF_K))


def _q10_factor(storage_temp_celsius: float) -> float:
    """Q10 factor: rate increase per 10°C above reference (approximate Arrhenius)."""
    t_k = storage_temp_celsius + 273.15
    return math.exp(_EA_KJ_PER_MOL / _R_KJ_PER_MOL_K * (10.0 / (t_k * (t_k - 10.0))))


def _degradation_products(pathway: str) -> str:
    """Return predicted degradation products string."""
    if pathway == "hydrolysis":
        return "carboxylic acid, alcohol"
    elif pathway == "oxidation":
        return "N-oxide, sulfoxide"
    elif pathway == "photolysis":
        return "radical fragments"
    else:  # combined
        return "carboxylic acid, alcohol, N-oxide, sulfoxide, radical fragments"


def predict_degradation(
    drug_name: str,
    logp: float = 2.0,
    pka: float = 7.0,
    mw_da: float = 300.0,
    pathway: str = "hydrolysis",
    storage_temp_celsius: float = 25.0,
    ph: float = 7.0,
    relative_humidity_pct: float = 60.0,
) -> DegradationResult:
    """Predict chemical degradation kinetics for a drug under given storage conditions.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    logp:
        Octanol-water partition coefficient.
    pka:
        Ionisation constant of the drug.
    mw_da:
        Molecular weight (Da).
    pathway:
        Degradation mechanism: "hydrolysis", "oxidation", "photolysis", or "combined".
    storage_temp_celsius:
        Storage temperature in Celsius.
    ph:
        pH of the medium/solution.
    relative_humidity_pct:
        Relative humidity percentage (0-100).

    Returns
    -------
    DegradationResult with first-order kinetics and ICH Q1A compliance flag.
    """
    # --- Validation ---
    if not (0.0 <= relative_humidity_pct <= 100.0):
        raise ValueError(f"relative_humidity_pct must be in [0, 100], got {relative_humidity_pct}")
    if storage_temp_celsius < -80.0 or storage_temp_celsius > 80.0:
        raise ValueError(f"storage_temp_celsius must be in [-80, 80], got {storage_temp_celsius}")
    if pathway not in ("hydrolysis", "oxidation", "photolysis", "combined"):
        raise ValueError(
            f"pathway must be one of hydrolysis/oxidation/photolysis/combined, got {pathway!r}"
        )

    # --- Base rate constants at 25°C ---
    k_hyd = _base_rate_hydrolysis(ph, pka, relative_humidity_pct)
    k_ox = _base_rate_oxidation(logp)
    k_photo = _base_rate_photolysis(logp)

    if pathway == "hydrolysis":
        k_base = k_hyd
    elif pathway == "oxidation":
        k_base = k_ox
    elif pathway == "photolysis":
        k_base = k_photo
    else:  # combined
        k_base = k_hyd + k_ox + k_photo

    # --- Temperature adjustment (Arrhenius) ---
    temp_factor = _temperature_factor(storage_temp_celsius)
    k_deg = k_base * temp_factor

    # Ensure k_deg is positive (safety clamp)
    k_deg = max(k_deg, 1e-12)

    # --- Derived metrics ---
    t_half = math.log(2.0) / k_deg
    t90 = math.log(10.0 / 9.0) / k_deg  # time to 90% remaining
    shelf_life_months = math.log(1.0 / 0.95) / k_deg / _HOURS_PER_MONTH
    concentration_at_1yr = math.exp(-k_deg * _HOURS_PER_YEAR)
    q10 = _q10_factor(storage_temp_celsius)

    # --- Risk classification ---
    if shelf_life_months < 6.0:
        risk_level = "high"
    elif shelf_life_months < 24.0:
        risk_level = "moderate"
    else:
        risk_level = "low"

    ich_q1a_compliant = shelf_life_months >= 24.0

    products = _degradation_products(pathway)

    notes_parts = [
        f"k_deg={k_deg:.4e}/h at {storage_temp_celsius:.0f}°C",
        f"t1/2={t_half:.1f}h",
        f"shelf_life={shelf_life_months:.1f} months",
        f"pathway={pathway}",
    ]

    return DegradationResult(
        drug_name=drug_name,
        degradation_pathway=pathway,
        rate_constant_per_h=k_deg,
        t_half_h=t_half,
        t90_h=t90,
        shelf_life_months=shelf_life_months,
        activation_energy_kJ_per_mol=_EA_KJ_PER_MOL,
        q10_factor=q10,
        concentration_at_1yr=concentration_at_1yr,
        degradation_products=products,
        risk_level=risk_level,
        ich_q1a_compliant=ich_q1a_compliant,
        notes="; ".join(notes_parts),
    )


def screen_storage_conditions(
    drug_name: str,
    logp: float,
    pka: float,
    mw_da: float,
    pathway: str,
    conditions: list[dict],
) -> list[DegradationResult]:
    """Screen multiple storage conditions and rank by stability.

    Parameters
    ----------
    conditions:
        List of dicts, each with keys: "temp_celsius", "ph", "rh_pct".

    Returns
    -------
    List of DegradationResult sorted by shelf_life_months descending (most stable first).
    """
    results: list[DegradationResult] = []
    for cond in conditions:
        res = predict_degradation(
            drug_name=drug_name,
            logp=logp,
            pka=pka,
            mw_da=mw_da,
            pathway=pathway,
            storage_temp_celsius=cond["temp_celsius"],
            ph=cond["ph"],
            relative_humidity_pct=cond["rh_pct"],
        )
        results.append(res)

    results.sort(key=lambda r: r.shelf_life_months, reverse=True)
    return results
