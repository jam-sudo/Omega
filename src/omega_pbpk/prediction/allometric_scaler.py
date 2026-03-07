"""Allometric scaling predictor for human PK from preclinical species data.

Uses power-law allometric relationships: PK_parameter = a * BW^b
  CL exponent b ~ 0.75 (3/4 power)
  Vd exponent b ~ 1.0 (linear)

OLS regression performed on log-log scale when >= 2 species available.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Species body weights (kg)
# ---------------------------------------------------------------------------

SPECIES_BW_KG: dict[str, float] = {
    "mouse": 0.025,
    "rat": 0.25,
    "rabbit": 2.0,
    "dog": 10.0,
    "monkey": 5.0,
    "human": 70.0,
}

# Default allometric exponents (when using single-species scaling)
DEFAULT_CL_EXPONENT = 0.75
DEFAULT_VD_EXPONENT = 1.0


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AllometricScalingResult:
    """Result of allometric scaling to predict human PK."""

    drug_name: str
    bw_human_kg: float
    cl_human_mL_per_min_per_kg: float
    cl_human_L_per_h: float
    vd_human_L_per_kg: float
    vd_human_L: float
    t_half_human_h: float
    cl_exponent: float
    vd_exponent: float
    n_species_used: int
    confidence: str  # "high" (n>=3), "moderate" (n=2), "low" (n=1)
    notes: str


# ---------------------------------------------------------------------------
# OLS log-log regression helper
# ---------------------------------------------------------------------------


def _ols_log_log(bw_vals: list[float], pk_vals: list[float]) -> tuple[float, float]:
    """Fit log(PK) = log(a) + b*log(BW) via OLS.

    Returns:
        (a, b) — intercept coefficient and exponent
    """
    n = len(bw_vals)
    x = [math.log(bw) for bw in bw_vals]
    y = [math.log(pk) for pk in pk_vals]

    x_mean = sum(x) / n
    y_mean = sum(y) / n

    num = sum((x[i] - x_mean) * (y[i] - y_mean) for i in range(n))
    den = sum((x[i] - x_mean) ** 2 for i in range(n))

    if abs(den) < 1e-15:
        # All x values identical — use mean y and exponent 0
        b = 0.0
    else:
        b = num / den

    log_a = y_mean - b * x_mean
    a = math.exp(log_a)
    return a, b


# ---------------------------------------------------------------------------
# Validation helper
# ---------------------------------------------------------------------------


def _validate_preclinical_data(preclinical_data: list[dict]) -> None:
    if not preclinical_data:
        raise ValueError("preclinical_data must contain at least one species entry")
    for entry in preclinical_data:
        if entry.get("cl_mL_per_min_per_kg", 0) <= 0:
            raise ValueError(
                f"cl_mL_per_min_per_kg must be > 0 for species {entry.get('species', 'unknown')}"
            )
        if entry.get("vd_L_per_kg", 0) <= 0:
            raise ValueError(
                f"vd_L_per_kg must be > 0 for species {entry.get('species', 'unknown')}"
            )


# ---------------------------------------------------------------------------
# Core prediction function
# ---------------------------------------------------------------------------


def predict_human_pk_allometric(
    drug_name: str,
    preclinical_data: list[dict],
    bw_human_kg: float = 70.0,
) -> AllometricScalingResult:
    """Predict human PK parameters from preclinical allometric scaling.

    Args:
        drug_name: Name of the drug.
        preclinical_data: List of dicts with keys:
            species (str), cl_mL_per_min_per_kg (float), vd_L_per_kg (float)
        bw_human_kg: Human body weight in kg (default 70.0).

    Returns:
        AllometricScalingResult with predicted human PK.
    """
    if bw_human_kg <= 0:
        raise ValueError("bw_human_kg must be > 0")
    _validate_preclinical_data(preclinical_data)

    n = len(preclinical_data)

    if n == 1:
        # Single species: use fixed exponents
        entry = preclinical_data[0]
        species_name = entry["species"].lower()
        bw_animal = SPECIES_BW_KG.get(species_name, entry.get("bw_kg", None))
        if bw_animal is None:
            raise ValueError(
                f"Unknown species '{species_name}'. Provide 'bw_kg' key or use a known species."
            )

        cl_animal = entry["cl_mL_per_min_per_kg"]
        vd_animal = entry["vd_L_per_kg"]

        ratio = bw_human_kg / bw_animal
        # CL_human_mL_per_min_per_kg = CL_human_mL_per_min / BW_human
        cl_human_mL_per_min = cl_animal * bw_animal * (ratio**DEFAULT_CL_EXPONENT)
        cl_human_mL_per_min_per_kg_val = cl_human_mL_per_min / bw_human_kg

        vd_human_mL = vd_animal * bw_animal * 1000.0 * (ratio**DEFAULT_VD_EXPONENT)
        vd_human_L = vd_human_mL / 1000.0
        vd_human_L_per_kg = vd_human_L / bw_human_kg

        cl_exp = DEFAULT_CL_EXPONENT
        vd_exp = DEFAULT_VD_EXPONENT
        confidence = "low"
        method_note = "Single-species scaling with fixed exponents (CL^0.75, Vd^1.0)"

    else:
        # Multi-species OLS regression
        bw_list: list[float] = []
        cl_list: list[float] = []  # mL/min (total, not per kg)
        vd_list: list[float] = []  # L (total, not per kg)

        for entry in preclinical_data:
            species_name = entry["species"].lower()
            bw_animal = SPECIES_BW_KG.get(species_name, entry.get("bw_kg", None))
            if bw_animal is None:
                raise ValueError(
                    f"Unknown species '{species_name}'. Provide 'bw_kg' key or use a known species."
                )

            # Convert per-kg to total values for regression
            cl_total = entry["cl_mL_per_min_per_kg"] * bw_animal  # mL/min
            vd_total = entry["vd_L_per_kg"] * bw_animal  # L

            bw_list.append(bw_animal)
            cl_list.append(cl_total)
            vd_list.append(vd_total)

        # OLS regressions
        a_cl, cl_exp = _ols_log_log(bw_list, cl_list)
        a_vd, vd_exp = _ols_log_log(bw_list, vd_list)

        # Predict human values (total)
        cl_human_total_mL_per_min = a_cl * (bw_human_kg**cl_exp)
        vd_human_L = a_vd * (bw_human_kg**vd_exp)

        cl_human_mL_per_min_per_kg_val = cl_human_total_mL_per_min / bw_human_kg
        vd_human_L_per_kg = vd_human_L / bw_human_kg

        confidence = "high" if n >= 3 else "moderate"
        method_note = f"OLS log-log regression across {n} species"

    # Derived metrics
    # cl_human_L_per_h: CL (mL/min/kg) * BW_human_kg * 60 / 1000
    cl_human_L_per_h = cl_human_mL_per_min_per_kg_val * bw_human_kg * 60.0 / 1000.0

    if cl_human_L_per_h > 0:
        t_half_human_h = math.log(2) * vd_human_L / cl_human_L_per_h
    else:
        t_half_human_h = float("inf")

    notes = (
        f"{method_note}. "
        f"Predicted human CL: {cl_human_L_per_h:.2f} L/h, "
        f"Vd: {vd_human_L:.1f} L, "
        f"t½: {t_half_human_h:.1f} h. "
        f"CL exponent b={cl_exp:.3f}, Vd exponent b={vd_exp:.3f}. "
        f"Confidence: {confidence} ({n} species)."
    )

    return AllometricScalingResult(
        drug_name=drug_name,
        bw_human_kg=bw_human_kg,
        cl_human_mL_per_min_per_kg=cl_human_mL_per_min_per_kg_val,
        cl_human_L_per_h=cl_human_L_per_h,
        vd_human_L_per_kg=vd_human_L_per_kg,
        vd_human_L=vd_human_L,
        t_half_human_h=t_half_human_h,
        cl_exponent=cl_exp,
        vd_exponent=vd_exp,
        n_species_used=n,
        confidence=confidence,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------


def multi_species_scaling(
    drug_name: str,
    preclinical_data: list[dict],
    bw_human_kg: float = 70.0,
) -> AllometricScalingResult:
    """Alias for predict_human_pk_allometric using all provided species.

    Identical behaviour — always uses all entries in preclinical_data.
    """
    return predict_human_pk_allometric(
        drug_name=drug_name,
        preclinical_data=preclinical_data,
        bw_human_kg=bw_human_kg,
    )
