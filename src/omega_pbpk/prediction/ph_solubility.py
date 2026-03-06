"""pH-solubility profile prediction — Henderson-Hasselbalch for acids, bases, amphoteric."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class pHSolubilityResult:
    drug_name: str
    drug_type: str           # "acid", "base", "amphoteric", "neutral"
    pka_values: list[float]
    intrinsic_solubility_mg_mL: float
    ph_values: list[float]
    solubility_mg_mL: list[float]
    ph_max_solubility: float   # pH of maximum solubility
    max_solubility_mg_mL: float
    ph_min_solubility: float   # pH of minimum solubility
    min_solubility_mg_mL: float
    solubility_at_ph_7_4: float
    solubility_at_ph_6_8: float   # small intestine
    solubility_at_ph_1_2: float   # stomach


def _ionization_factor_acid(ph: float, pka: float) -> float:
    """Ionization factor for monoprotic acid: 1 + 10^(pH-pKa)."""
    return 1.0 + 10.0 ** (ph - pka)


def _ionization_factor_base(ph: float, pka: float) -> float:
    """Ionization factor for monoprotic base: 1 + 10^(pKa-pH)."""
    return 1.0 + 10.0 ** (pka - ph)


def predict_ph_solubility(
    drug_name: str,
    intrinsic_solubility_mg_mL: float,
    drug_type: str = "acid",
    pka1: float | None = None,
    pka2: float | None = None,
    ph_range: tuple[float, float] = (1.0, 9.0),
    n_points: int = 81,
) -> pHSolubilityResult:
    """Predict solubility vs pH using Henderson-Hasselbalch equation.

    For acids:   S(pH) = S0 * (1 + 10^(pH-pKa))
    For bases:   S(pH) = S0 * (1 + 10^(pKa-pH))
    For neutral: S(pH) = S0 (constant)
    For amphoteric (has both pKa1 acid + pKa2 base):
        S(pH) = S0 * (1 + 10^(pH-pKa_acid) + 10^(pKa_base-pH))

    Parameters
    ----------
    intrinsic_solubility_mg_mL : float
        S0 — solubility of the unionized form.
    pka1, pka2 : float or None
        pKa values. For acids: pka1 is acidic pKa. For bases: pka1 is basic pKa.
        For amphoteric: pka1=acidic, pka2=basic.
    """
    if intrinsic_solubility_mg_mL <= 0:
        raise ValueError("intrinsic_solubility_mg_mL must be > 0")

    drug_type = drug_type.lower()
    if drug_type not in ("acid", "base", "amphoteric", "neutral"):
        raise ValueError("drug_type must be 'acid', 'base', 'amphoteric', or 'neutral'")

    ph_vals = [ph_range[0] + i * (ph_range[1] - ph_range[0]) / (n_points - 1) for i in range(n_points)]
    S0 = intrinsic_solubility_mg_mL

    pka_list = []
    if pka1 is not None:
        pka_list.append(pka1)
    if pka2 is not None:
        pka_list.append(pka2)

    sol_vals = []
    for ph in ph_vals:
        if drug_type == "neutral":
            s = S0
        elif drug_type == "acid":
            pka = pka1 if pka1 is not None else 4.0
            s = S0 * _ionization_factor_acid(ph, pka)
        elif drug_type == "base":
            pka = pka1 if pka1 is not None else 9.0
            s = S0 * _ionization_factor_base(ph, pka)
        else:  # amphoteric
            pka_acid = pka1 if pka1 is not None else 4.0
            pka_base = pka2 if pka2 is not None else 9.0
            s = S0 * (1.0 + 10.0 ** (ph - pka_acid) + 10.0 ** (pka_base - ph))
        sol_vals.append(max(s, 1e-9))

    max_s = max(sol_vals)
    min_s = min(sol_vals)
    ph_max = ph_vals[sol_vals.index(max_s)]
    ph_min = ph_vals[sol_vals.index(min_s)]

    def _sol_at(target_ph: float) -> float:
        # Find closest pH point
        idx = min(range(len(ph_vals)), key=lambda i: abs(ph_vals[i] - target_ph))
        return sol_vals[idx]

    return pHSolubilityResult(
        drug_name=drug_name,
        drug_type=drug_type,
        pka_values=pka_list,
        intrinsic_solubility_mg_mL=S0,
        ph_values=ph_vals,
        solubility_mg_mL=sol_vals,
        ph_max_solubility=ph_max,
        max_solubility_mg_mL=max_s,
        ph_min_solubility=ph_min,
        min_solubility_mg_mL=min_s,
        solubility_at_ph_7_4=_sol_at(7.4),
        solubility_at_ph_6_8=_sol_at(6.8),
        solubility_at_ph_1_2=_sol_at(1.2),
    )


def screen_ph_solubility(compounds: list[dict]) -> list[pHSolubilityResult]:
    """Screen multiple compounds for pH-solubility profiles."""
    results = []
    for c in compounds:
        r = predict_ph_solubility(
            drug_name=c.get("drug_name", "unknown"),
            intrinsic_solubility_mg_mL=c.get("intrinsic_solubility_mg_mL", 0.1),
            drug_type=c.get("drug_type", "neutral"),
            pka1=c.get("pka1"),
            pka2=c.get("pka2"),
        )
        results.append(r)
    return results


__all__ = [
    "pHSolubilityResult",
    "predict_ph_solubility",
    "screen_ph_solubility",
]
