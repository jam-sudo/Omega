"""
quantitative_ddi_network.py — Quantitative DDI Network with AUCR Propagation

Computes pairwise and polypharmacy DDI AUCR values for multi-drug combinations
using CYP inhibition constants. Distinct from ddi_network.py which uses a
graph/severity-based qualitative approach; this module is fully quantitative.

Reference: FDA 2020 DDI Guidance — static R1 model: R = 1 + I/Ki
AUCR = 1 / (fm_cyp * (1 - 1/R) + (1 - fm_cyp))
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "DDINetwork",
    "PolypharmacyRiskResult",
    "SequentialDDIResult",
    "build_ddi_network",
    "evaluate_polypharmacy_risk",
    "simulate_sequential_ddi",
]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_VD_PLASMA_L = 70.0  # simplified distribution volume for free Cmax estimate
_HOURS_PER_DAY = 24.0


# ---------------------------------------------------------------------------
# Result dataclasses (frozen)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DDINetwork:
    """Pairwise quantitative DDI network for a set of co-administered drugs."""

    n_drugs: int
    n_pairs: int
    all_aucrs: dict[str, float]  # key = "substrate_perpetrator"
    notes: str


@dataclass(frozen=True)
class PolypharmacyRiskResult:
    """Aggregate quantitative risk summary for a polypharmacy regimen."""

    n_drugs: int
    max_aucr: float
    highest_risk_pair: str  # "substrate_perpetrator"
    n_pairs_gt2: int
    n_pairs_gt5: int
    risk_level: str  # "low", "moderate", "high"
    all_aucrs: dict[str, float]
    notes: str


@dataclass(frozen=True)
class SequentialDDIResult:
    """AUCR progression as perpetrators are added one-by-one to a substrate."""

    substrate_name: str
    perpetrator_order: list[str]
    aucr_progression: list[float]
    final_aucr: float
    notes: str


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _validate_drug(drug: dict) -> None:
    """Raise ValueError if a drug dict is missing required fields or invalid."""
    name = drug.get("name")
    if not name:
        raise ValueError("Each drug must have a non-empty 'name' field.")

    cl = drug.get("cl_intrinsic")
    if cl is None or cl <= 0:
        raise ValueError(f"Drug '{name}': cl_intrinsic must be > 0.")

    dose = drug.get("daily_dose_mg")
    if dose is None or dose <= 0:
        raise ValueError(f"Drug '{name}': daily_dose_mg must be > 0.")

    for fm_key in ("fm_cyp3a4", "fm_cyp2d6", "fm_cyp2c9"):
        fm = drug.get(fm_key, 0.0)
        if not (0.0 <= fm <= 1.0):
            raise ValueError(f"Drug '{name}': {fm_key} must be in [0, 1], got {fm}.")


def _free_cmax(drug: dict) -> float:
    """Simplified free Cmax (mg/L) = daily_dose_mg / (Vd_plasma * 24 h)."""
    return drug["daily_dose_mg"] / (_VD_PLASMA_L * _HOURS_PER_DAY)


def _r_value(inhibitor: dict, cyp: str) -> float:
    """Compute R = 1 + I/Ki for a given CYP enzyme.

    cyp should be '3a4', '2d6', or '2c9'.
    Returns 1.0 if no inhibition constant is provided.
    """
    ki_key = f"ki_{cyp}"
    ki = inhibitor.get(ki_key)
    if ki is None or ki <= 0:
        return 1.0  # no inhibition
    i_free = _free_cmax(inhibitor)
    return 1.0 + i_free / ki


def _aucr_pair(substrate: dict, perpetrator: dict) -> float:
    """
    AUCR of substrate when perpetrator is co-administered (single perpetrator).

    AUCR = prod over CYPs of [1 / (fm * (1 - 1/R) + (1 - fm))]
    which simplifies to: 1 / (fm/R + 1 - fm)

    When fm = 0 for a CYP, that CYP contributes an AUCR factor of 1.
    """
    cyps = ("3a4", "2d6", "2c9")
    fm_keys = ("fm_cyp3a4", "fm_cyp2d6", "fm_cyp2c9")

    total_fm = sum(substrate.get(k, 0.0) for k in fm_keys)
    if total_fm == 0.0:
        return 1.0  # substrate has no defined CYP pathways

    aucr = 1.0
    for cyp, fm_key in zip(cyps, fm_keys, strict=True):
        fm = substrate.get(fm_key, 0.0)
        if fm == 0.0:
            continue
        r = _r_value(perpetrator, cyp)
        # denom = fm/R + (1 - fm)
        denom = fm / r + (1.0 - fm)
        if denom <= 0.0:
            denom = 1e-9
        aucr *= 1.0 / denom

    return aucr


def _pair_key(substrate_name: str, perpetrator_name: str) -> str:
    return f"{substrate_name}_{perpetrator_name}"


def _combined_aucr(substrate: dict, perpetrators: list[dict]) -> float:
    """
    AUCR when multiple perpetrators inhibit simultaneously.

    For each CYP, combined inhibition uses additive I/Ki:
    R_combined = 1 + sum(I_j / Ki_j) = sum(R_j - 1) + 1
    (conservative additive model).
    """
    cyps = ("3a4", "2d6", "2c9")
    fm_keys = ("fm_cyp3a4", "fm_cyp2d6", "fm_cyp2c9")

    total_fm = sum(substrate.get(k, 0.0) for k in fm_keys)
    if total_fm == 0.0:
        return 1.0

    aucr = 1.0
    for cyp, fm_key in zip(cyps, fm_keys, strict=True):
        fm = substrate.get(fm_key, 0.0)
        if fm == 0.0:
            continue
        # Combined R via additive inhibition: R_combined = 1 + sum(R_j - 1)
        combined_r = 1.0
        for perp in perpetrators:
            combined_r += _r_value(perp, cyp) - 1.0

        denom = fm / combined_r + (1.0 - fm)
        if denom <= 0.0:
            denom = 1e-9
        aucr *= 1.0 / denom

    return aucr


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_ddi_network(drugs: list[dict]) -> DDINetwork:
    """
    Build a quantitative pairwise DDI AUCR network for all co-administered drugs.

    Parameters
    ----------
    drugs:
        List of drug dicts. Each must contain:
        name, cl_intrinsic (>0), fm_cyp3a4, fm_cyp2d6, fm_cyp2c9 (each in [0,1]),
        ki_3a4, ki_2d6, ki_2c9 (None or IC50 in mg/L; None = no inhibition),
        daily_dose_mg (>0).

    Returns
    -------
    DDINetwork with all substrate-perpetrator AUCR pairs.
    """
    if len(drugs) < 2:
        raise ValueError("At least 2 drugs are required to build a DDI network.")

    for drug in drugs:
        _validate_drug(drug)

    all_aucrs: dict[str, float] = {}
    n = len(drugs)

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            substrate = drugs[i]
            perpetrator = drugs[j]
            aucr = _aucr_pair(substrate, perpetrator)
            key = _pair_key(substrate["name"], perpetrator["name"])
            all_aucrs[key] = round(aucr, 4)

    n_pairs = n * (n - 1)
    notes = (
        f"Quantitative AUCR network for {n} drugs. "
        "R = 1 + I/Ki (FDA static model). "
        "I = daily_dose_mg / (70 L * 24 h). "
        "AUCR = 1/(fm/R + 1 - fm) per CYP pathway."
    )
    return DDINetwork(
        n_drugs=n,
        n_pairs=n_pairs,
        all_aucrs=all_aucrs,
        notes=notes,
    )


def evaluate_polypharmacy_risk(drugs: list[dict]) -> PolypharmacyRiskResult:
    """
    Evaluate overall polypharmacy DDI risk for a drug combination.

    Risk levels:
    - low: max AUCR < 2.0
    - moderate: max AUCR 2.0–5.0
    - high: max AUCR > 5.0

    Parameters
    ----------
    drugs:
        Same format as build_ddi_network.

    Returns
    -------
    PolypharmacyRiskResult
    """
    network = build_ddi_network(drugs)

    all_aucrs = network.all_aucrs
    if not all_aucrs:
        raise ValueError("No AUCR pairs computed — check drug list.")

    max_aucr = max(all_aucrs.values())
    highest_risk_pair = max(all_aucrs, key=lambda k: all_aucrs[k])
    n_pairs_gt2 = sum(1 for v in all_aucrs.values() if v > 2.0)
    n_pairs_gt5 = sum(1 for v in all_aucrs.values() if v > 5.0)

    if max_aucr < 2.0:
        risk_level = "low"
    elif max_aucr < 5.0:
        risk_level = "moderate"
    else:
        risk_level = "high"

    notes = (
        f"Polypharmacy risk: {risk_level.upper()}. "
        f"Highest AUCR = {max_aucr:.2f} for pair '{highest_risk_pair}'. "
        f"{n_pairs_gt2} pair(s) with AUCR > 2.0; {n_pairs_gt5} with AUCR > 5.0."
    )

    return PolypharmacyRiskResult(
        n_drugs=network.n_drugs,
        max_aucr=round(max_aucr, 4),
        highest_risk_pair=highest_risk_pair,
        n_pairs_gt2=n_pairs_gt2,
        n_pairs_gt5=n_pairs_gt5,
        risk_level=risk_level,
        all_aucrs=dict(all_aucrs),
        notes=notes,
    )


def simulate_sequential_ddi(
    substrate: dict,
    perpetrators: list[dict],
    order: list[str],
) -> SequentialDDIResult:
    """
    Track how the substrate AUCR changes as perpetrators are added in sequence.

    Parameters
    ----------
    substrate:
        Drug dict for the victim drug.
    perpetrators:
        List of drug dicts for all potential perpetrators.
    order:
        Names of perpetrators in the sequence they are added.

    Returns
    -------
    SequentialDDIResult
    """
    _validate_drug(substrate)
    for p in perpetrators:
        _validate_drug(p)

    perp_by_name: dict[str, dict] = {p["name"]: p for p in perpetrators}

    for name in order:
        if name not in perp_by_name:
            raise ValueError(f"Perpetrator '{name}' in order not found in perpetrators list.")

    aucr_progression: list[float] = []
    current_perpetrators: list[dict] = []

    for name in order:
        current_perpetrators.append(perp_by_name[name])
        combined_aucr = _combined_aucr(substrate, current_perpetrators)
        aucr_progression.append(round(combined_aucr, 4))

    final_aucr = aucr_progression[-1] if aucr_progression else 1.0

    notes = (
        f"Sequential DDI for substrate '{substrate['name']}'. "
        f"{len(order)} perpetrator(s) added: {' -> '.join(order)}. "
        f"Final AUCR = {final_aucr:.2f}. "
        "Combined inhibition modelled as additive I/Ki."
    )

    return SequentialDDIResult(
        substrate_name=substrate["name"],
        perpetrator_order=list(order),
        aucr_progression=aucr_progression,
        final_aucr=final_aucr,
        notes=notes,
    )
