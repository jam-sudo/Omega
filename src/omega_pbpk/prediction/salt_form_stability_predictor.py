"""Salt form stability predictor for drug compounds.

Predicts chemical stability of drug salt forms under different storage
conditions using an Arrhenius-based degradation model.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# Salt form lookup table: (hygroscopicity_factor, ph_stability_min, ph_stability_max)
_SALT_FORMS: dict = {
    "HCl": (0.8, 1.0, 7.0),
    "mesylate": (0.6, 2.0, 8.0),
    "besylate": (0.4, 2.0, 9.0),
    "tosylate": (0.5, 2.0, 8.5),
    "phosphate": (0.9, 1.0, 8.0),
    "sulfate": (0.7, 1.5, 7.5),
}

# Storage condition lookup table: (temp_C, rh_pct)
_STORAGE_CONDITIONS: dict = {
    "ICH_25C_60RH": (25.0, 60.0),
    "ICH_40C_75RH": (40.0, 75.0),
    "refrigerated_5C": (5.0, 30.0),
    "frozen_minus20C": (-20.0, 10.0),
}

# Arrhenius constants
_K_REF = 0.0001  # degradation rate at 25 C (1/day)
_EA = 80.0  # activation energy kJ/mol
_R = 8.314e-3  # gas constant kJ/mol/K
_T_REF = 298.15  # reference temperature K


@dataclass(frozen=True)
class SaltStabilityResult:
    """Result of salt form stability prediction."""

    drug_name: str
    salt_form: str
    storage_condition: str
    temp_C: float
    rh_pct: float
    degradation_rate_per_day: float
    predicted_purity_pct_12months: float
    predicted_purity_pct_24months: float
    shelf_life_days: float
    hygroscopic_risk: str
    stability_class: str
    notes: str


def _hygroscopic_risk(factor: float) -> str:
    if factor < 0.5:
        return "low"
    if factor <= 0.7:
        return "moderate"
    return "high"


def _stability_class(purity_12m: float) -> str:
    if purity_12m >= 98.0:
        return "stable"
    if purity_12m >= 95.0:
        return "marginal"
    return "unstable"


def predict_salt_stability(
    drug_name: str,
    salt_form: str,
    storage_condition: str,
    drug_pka: float = None,
) -> SaltStabilityResult:
    """Predict stability of a drug salt form under a storage condition.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    salt_form:
        Salt form identifier (e.g., "HCl", "mesylate").
    storage_condition:
        Storage condition key (e.g., "ICH_25C_60RH").
    drug_pka:
        Optional pKa of the drug (not used in current model, reserved).

    Returns
    -------
    SaltStabilityResult
    """
    if salt_form not in _SALT_FORMS:
        raise ValueError(
            f"Unknown salt_form '{salt_form}'. Valid options: {sorted(_SALT_FORMS.keys())}"
        )
    if storage_condition not in _STORAGE_CONDITIONS:
        raise ValueError(
            f"Unknown storage_condition '{storage_condition}'. "
            f"Valid options: {sorted(_STORAGE_CONDITIONS.keys())}"
        )

    hygro_factor, ph_min, ph_max = _SALT_FORMS[salt_form]
    temp_C, rh_pct = _STORAGE_CONDITIONS[storage_condition]

    temp_K = temp_C + 273.15

    # Arrhenius-based degradation rate
    k_deg = _K_REF * math.exp(_EA / _R * (1.0 / _T_REF - 1.0 / temp_K))

    # Moisture correction
    k_deg *= 1.0 + 0.01 * rh_pct * hygro_factor

    # Frozen: near-zero degradation
    if storage_condition == "frozen_minus20C":
        k_deg *= 0.001

    # Purity at 12 and 24 months (360 and 720 days)
    purity_12m = 100.0 * math.exp(-k_deg * 360.0)
    purity_24m = 100.0 * math.exp(-k_deg * 720.0)

    # Shelf life: time to reach 90% purity
    if k_deg > 0.0:
        shelf_life_days = -math.log(0.9) / k_deg
    else:
        shelf_life_days = float("inf")

    hygro_risk = _hygroscopic_risk(hygro_factor)
    stab_class = _stability_class(purity_12m)

    notes = (
        f"k_deg={k_deg:.6f}/day; "
        f"pH stability range [{ph_min:.1f},{ph_max:.1f}]; "
        f"hygroscopicity_factor={hygro_factor:.1f}; "
        f"shelf_life={shelf_life_days:.0f} days"
    )

    return SaltStabilityResult(
        drug_name=drug_name,
        salt_form=salt_form,
        storage_condition=storage_condition,
        temp_C=temp_C,
        rh_pct=rh_pct,
        degradation_rate_per_day=k_deg,
        predicted_purity_pct_12months=purity_12m,
        predicted_purity_pct_24months=purity_24m,
        shelf_life_days=shelf_life_days,
        hygroscopic_risk=hygro_risk,
        stability_class=stab_class,
        notes=notes,
    )


def screen_salt_stability(
    drug_name: str,
    drug_pka: float = None,
) -> list:
    """Screen all salt forms under ICH stress condition (40C/75RH).

    Parameters
    ----------
    drug_name:
        Name of the drug.
    drug_pka:
        Optional pKa (passed through to predict_salt_stability).

    Returns
    -------
    list of SaltStabilityResult sorted by predicted_purity_pct_12months
    descending (most stable first).
    """
    results = []
    for salt_form in _SALT_FORMS:
        result = predict_salt_stability(
            drug_name=drug_name,
            salt_form=salt_form,
            storage_condition="ICH_40C_75RH",
            drug_pka=drug_pka,
        )
        results.append(result)

    results.sort(key=lambda r: r.predicted_purity_pct_12months, reverse=True)
    return results
