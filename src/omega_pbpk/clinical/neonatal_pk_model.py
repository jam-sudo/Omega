"""Phase 520 — Neonatal PK Model.

PK model for neonates (0-28 days) with age-specific organ maturation.
Pure Python (stdlib only), forward Euler not needed — all calculations are
analytical/multiplicative scaling.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

GFR_ADULT_REF_L_PER_H = 7.2  # adult reference GFR (L/h)
ADULT_WEIGHT_REF_KG = 70.0  # reference adult body weight (kg)

VALID_CYPS = {"CYP3A4", "CYP2D6", "CYP1A2", "renal"}


# ---------------------------------------------------------------------------
# Result dataclass (frozen — no list/dict fields)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NeonatalPKResult:
    """Result of neonatal PK prediction."""

    drug_name: str
    postnatal_age_days: float
    weight_kg: float
    dominant_cyp: str

    f_gfr: float
    f_cyp: float
    f_alb: float
    f_tbw: float

    gfr_neonatal_L_per_h: float

    cl_renal_neonatal_L_per_h: float
    cl_metabolic_neonatal_L_per_h: float
    cl_total_neonatal_L_per_h: float

    vd_neonatal_L: float
    t_half_neonatal_h: float

    dose_adjustment_factor: float  # CL_neo / CL_adult
    monitoring_required: bool  # True if dose_adjustment_factor < 0.3
    notes: str


# ---------------------------------------------------------------------------
# Maturation factor helpers
# ---------------------------------------------------------------------------


def _f_gfr(postnatal_age_days: float) -> float:
    """GFR maturation: 10%–100% over first 28 postnatal days."""
    return 0.1 + 0.9 * (postnatal_age_days / 28.0)


def _f_cyp(dominant_cyp: str, postnatal_age_days: float) -> float:
    """CYP maturation factor at a given postnatal age."""
    day_frac = postnatal_age_days / 28.0
    if dominant_cyp == "CYP3A4":
        return 0.05 + 0.95 * day_frac
    elif dominant_cyp == "CYP2D6":
        return 1.0 * day_frac  # absent at birth
    elif dominant_cyp == "CYP1A2":
        return 0.05  # barely present, not age-dependent
    else:  # renal — no metabolic CYP
        return 1.0  # CL_metabolic will be scaled by this; can be 0 separately


def _f_alb(postnatal_age_days: float) -> float:
    """Albumin maturation: 60%–100% over first 28 days."""
    return 0.6 + 0.4 * (postnatal_age_days / 28.0)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def predict_neonatal_pk(
    drug_name: str,
    postnatal_age_days: float,
    weight_kg: float,
    dominant_cyp: str,
    cl_adult_metabolic_L_per_h: float,
    vd_adult_L: float,
    fup_adult: float = 0.1,
    adult_weight_kg: float = ADULT_WEIGHT_REF_KG,
) -> NeonatalPKResult:
    """Predict neonatal PK parameters from adult reference values.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    postnatal_age_days:
        Postnatal age in days (0–28).
    weight_kg:
        Neonate body weight in kg.
    dominant_cyp:
        Dominant elimination pathway: "CYP3A4", "CYP2D6", "CYP1A2", or "renal".
    cl_adult_metabolic_L_per_h:
        Adult hepatic/metabolic clearance (L/h); 0 for renally cleared drugs.
    vd_adult_L:
        Adult volume of distribution (L).
    fup_adult:
        Adult unbound fraction in plasma (0–1).
    adult_weight_kg:
        Reference adult weight for allometric scaling (default 70 kg).

    Returns
    -------
    NeonatalPKResult
    """
    # --- Validation ---
    if not (0 <= postnatal_age_days <= 28):
        raise ValueError(f"postnatal_age_days must be in [0, 28], got {postnatal_age_days}")
    if weight_kg <= 0:
        raise ValueError(f"weight_kg must be > 0, got {weight_kg}")
    if dominant_cyp not in VALID_CYPS:
        raise ValueError(f"dominant_cyp must be one of {VALID_CYPS}, got '{dominant_cyp}'")

    # --- Maturation factors ---
    fgfr = _f_gfr(postnatal_age_days)
    fcyp = _f_cyp(dominant_cyp, postnatal_age_days)
    falb = _f_alb(postnatal_age_days)
    ftbw = 1.3  # fixed constant: neonates have ~30% more TBW

    # --- Neonatal GFR (allometric, weight-scaled) ---
    gfr_neo = fgfr * GFR_ADULT_REF_L_PER_H * (weight_kg / adult_weight_kg) ** 0.75

    # --- Unbound fraction adjusted for neonatal albumin ---
    # fup_neo = fup_adult / f_alb  (lower albumin → higher fup)
    fup_neo = fup_adult / falb

    # --- Renal clearance ---
    cl_renal = gfr_neo * fup_neo

    # --- Metabolic clearance ---
    if dominant_cyp == "renal":
        cl_metabolic = 0.0
    else:
        cl_metabolic = cl_adult_metabolic_L_per_h * fcyp

    # --- Total clearance ---
    cl_total = cl_renal + cl_metabolic

    # --- Volume of distribution ---
    vd_neo = vd_adult_L * ftbw

    # --- Half-life ---
    if cl_total > 0:
        t_half = math.log(2) * vd_neo / cl_total
    else:
        t_half = float("inf")

    # --- Dose adjustment factor ---
    # CL_adult_total = cl_adult_metabolic + GFR_adult * fup_adult (simplified)
    gfr_adult = GFR_ADULT_REF_L_PER_H  # reference adult (1.0 maturation, full weight)
    cl_renal_adult = gfr_adult * fup_adult
    cl_metabolic_adult = cl_adult_metabolic_L_per_h  # full adult metabolic CL
    cl_adult_total = cl_renal_adult + cl_metabolic_adult

    if cl_adult_total > 0:
        dose_adj_factor = cl_total / cl_adult_total
    else:
        dose_adj_factor = 0.0

    monitoring_required = dose_adj_factor < 0.3

    notes = (
        f"Neonatal PK prediction for {drug_name} at {postnatal_age_days:.1f} days. "
        f"f_GFR={fgfr:.3f}, f_CYP={fcyp:.3f}, f_Alb={falb:.3f}. "
        f"CL_total={cl_total:.4f} L/h, Vd={vd_neo:.3f} L, t1/2={t_half:.2f} h."
    )

    return NeonatalPKResult(
        drug_name=drug_name,
        postnatal_age_days=postnatal_age_days,
        weight_kg=weight_kg,
        dominant_cyp=dominant_cyp,
        f_gfr=fgfr,
        f_cyp=fcyp,
        f_alb=falb,
        f_tbw=ftbw,
        gfr_neonatal_L_per_h=gfr_neo,
        cl_renal_neonatal_L_per_h=cl_renal,
        cl_metabolic_neonatal_L_per_h=cl_metabolic,
        cl_total_neonatal_L_per_h=cl_total,
        vd_neonatal_L=vd_neo,
        t_half_neonatal_h=t_half,
        dose_adjustment_factor=dose_adj_factor,
        monitoring_required=monitoring_required,
        notes=notes,
    )


def neonatal_age_progression(
    drug_name: str,
    weight_kg: float,
    dominant_cyp: str,
    cl_adult_L_per_h: float,
    vd_adult_L: float,
) -> list:
    """Return neonatal PK predictions at days 0, 7, 14, 21, 28.

    Results are sorted by postnatal_age_days ascending.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    weight_kg:
        Neonate body weight in kg (assumed constant over 28 days for simplicity).
    dominant_cyp:
        Dominant elimination pathway.
    cl_adult_L_per_h:
        Adult total metabolic clearance (L/h).
    vd_adult_L:
        Adult volume of distribution (L).

    Returns
    -------
    list of NeonatalPKResult sorted by postnatal_age_days ascending.
    """
    days = [0.0, 7.0, 14.0, 21.0, 28.0]
    results = []
    for d in days:
        r = predict_neonatal_pk(
            drug_name=drug_name,
            postnatal_age_days=d,
            weight_kg=weight_kg,
            dominant_cyp=dominant_cyp,
            cl_adult_metabolic_L_per_h=cl_adult_L_per_h,
            vd_adult_L=vd_adult_L,
        )
        results.append(r)
    results.sort(key=lambda x: x.postnatal_age_days)
    return results
