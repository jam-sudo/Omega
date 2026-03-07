"""Predict DDI severity from inhibition/induction constants using FDA static models."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DDIPredictionResult:
    """Result of DDI severity prediction."""

    perpetrator_name: str
    victim_name: str
    ki_uM: float
    c_plasma_uM: float
    fm_cyp: float
    inhibition_type: str
    r_value: float
    aucr_estimate: float
    ddi_risk_category: str
    fold_change_cl: float
    notes: str


_VALID_INHIBITION_TYPES = {"competitive", "mechanism_based", "induction", "mixed"}


def _risk_category(aucr: float) -> str:
    """Classify DDI risk based on AUCR."""
    if aucr > 5.0:
        return "severe"
    elif aucr >= 2.0:
        return "moderate"
    elif aucr >= 1.25:
        return "mild"
    else:
        return "negligible"


def predict_ddi_severity(
    perpetrator_name: str,
    victim_name: str,
    ki_uM: float,
    c_plasma_uM: float,
    fm_cyp: float,
    inhibition_type: str = "competitive",
    induction_emax: float = 0.0,
    induction_ec50_uM: float = 1.0,
) -> DDIPredictionResult:
    """Predict DDI severity using FDA static models.

    Parameters
    ----------
    perpetrator_name:
        Name of the perpetrator drug (inhibitor/inducer).
    victim_name:
        Name of the victim drug (substrate).
    ki_uM:
        Inhibition constant Ki in micromolar.
    c_plasma_uM:
        Perpetrator unbound plasma concentration in micromolar.
    fm_cyp:
        Fraction of victim drug metabolized by the affected CYP enzyme (0–1).
    inhibition_type:
        Mechanism: "competitive", "mechanism_based", "induction", or "mixed".
    induction_emax:
        Maximum fold-induction for induction models (dimensionless).
    induction_ec50_uM:
        EC50 for induction in micromolar.

    Returns
    -------
    DDIPredictionResult
    """
    if not perpetrator_name or not perpetrator_name.strip():
        raise ValueError("perpetrator_name must be a non-empty string")
    if not victim_name or not victim_name.strip():
        raise ValueError("victim_name must be a non-empty string")
    if ki_uM <= 0:
        raise ValueError("ki_uM must be > 0")
    if c_plasma_uM < 0:
        raise ValueError("c_plasma_uM must be >= 0")
    if not 0.0 <= fm_cyp <= 1.0:
        raise ValueError("fm_cyp must be between 0 and 1")
    if inhibition_type not in _VALID_INHIBITION_TYPES:
        raise ValueError(
            f"inhibition_type must be one of {_VALID_INHIBITION_TYPES}, "
            f"got '{inhibition_type}'"
        )
    if induction_emax < 0:
        raise ValueError("induction_emax must be >= 0")
    if induction_ec50_uM <= 0:
        raise ValueError("induction_ec50_uM must be > 0")

    notes_parts: list[str] = []

    if inhibition_type == "competitive":
        r_value = 1.0 + c_plasma_uM / ki_uM
        aucr = fm_cyp * r_value + (1.0 - fm_cyp)
        notes_parts.append(f"Competitive inhibition: R1={r_value:.3f}")

    elif inhibition_type == "mechanism_based":
        # Simplified MBI: R2 = 1 + 5*C/Ki
        r_value = 1.0 + 5.0 * c_plasma_uM / ki_uM
        aucr = fm_cyp * r_value + (1.0 - fm_cyp)
        notes_parts.append(f"Mechanism-based inhibition: R2={r_value:.3f}")

    elif inhibition_type == "induction":
        # Fold induction = Emax * C / (EC50 + C)
        ind_factor = induction_emax * c_plasma_uM / (induction_ec50_uM + c_plasma_uM)
        # Induction increases CL → decreases AUCR below 1
        r_value = 1.0 + ind_factor
        # For induction, AUCR < 1 (victim AUC decreases)
        aucr = 1.0 / (fm_cyp * r_value + (1.0 - fm_cyp))
        notes_parts.append(
            f"Induction: Emax={induction_emax:.2f}, EC50={induction_ec50_uM:.2f} uM, "
            f"fold-induction={ind_factor:.3f}"
        )

    else:  # mixed
        r1 = 1.0 + c_plasma_uM / ki_uM
        r2 = 1.0 + 5.0 * c_plasma_uM / ki_uM
        r_value = max(r1, r2)
        aucr = fm_cyp * r_value + (1.0 - fm_cyp)
        notes_parts.append(f"Mixed inhibition: max(R1={r1:.3f}, R2={r2:.3f})={r_value:.3f}")

    risk = _risk_category(aucr)
    fold_change_cl = 1.0 / aucr if aucr > 0 else float("inf")

    notes_parts.append(
        f"AUCR={aucr:.3f}, DDI risk={risk}, fm_cyp={fm_cyp:.2f}"
    )

    return DDIPredictionResult(
        perpetrator_name=perpetrator_name,
        victim_name=victim_name,
        ki_uM=ki_uM,
        c_plasma_uM=c_plasma_uM,
        fm_cyp=fm_cyp,
        inhibition_type=inhibition_type,
        r_value=r_value,
        aucr_estimate=aucr,
        ddi_risk_category=risk,
        fold_change_cl=fold_change_cl,
        notes="; ".join(notes_parts),
    )


def ddi_matrix(compounds: list[dict]) -> list[list[str]]:
    """Compute pairwise DDI risk matrix for a list of compounds.

    Each compound dict must contain:
      - "name": str
      - "ki_uM": float
      - "c_plasma_uM": float
      - "fm_cyp": float
      - "inhibition_type": str (optional, default "competitive")
      - "induction_emax": float (optional, default 0.0)
      - "induction_ec50_uM": float (optional, default 1.0)

    Parameters
    ----------
    compounds:
        List of compound parameter dicts.

    Returns
    -------
    list[list[str]]
        n×n matrix of DDI risk categories. Entry [i][j] is the risk when
        compound i is the perpetrator and compound j is the victim.
        Diagonal entries are "self" (no DDI).
    """
    if not compounds:
        raise ValueError("compounds must be a non-empty list")
    if len(compounds) < 2:
        raise ValueError("compounds must contain at least 2 entries for DDI matrix")

    n = len(compounds)
    matrix: list[list[str]] = [["" for _ in range(n)] for _ in range(n)]

    for i, perp in enumerate(compounds):
        if "name" not in perp:
            raise ValueError(f"Compound at index {i} is missing 'name'")
        for j, victim in enumerate(compounds):
            if "name" not in victim:
                raise ValueError(f"Compound at index {j} is missing 'name'")
            if i == j:
                matrix[i][j] = "self"
                continue
            try:
                result = predict_ddi_severity(
                    perpetrator_name=perp["name"],
                    victim_name=victim["name"],
                    ki_uM=float(perp.get("ki_uM", 1.0)),
                    c_plasma_uM=float(perp.get("c_plasma_uM", 0.1)),
                    fm_cyp=float(victim.get("fm_cyp", 0.5)),
                    inhibition_type=perp.get("inhibition_type", "competitive"),
                    induction_emax=float(perp.get("induction_emax", 0.0)),
                    induction_ec50_uM=float(perp.get("induction_ec50_uM", 1.0)),
                )
                matrix[i][j] = result.ddi_risk_category
            except ValueError:
                matrix[i][j] = "unknown"

    return matrix
