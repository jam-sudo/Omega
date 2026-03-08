"""Phase 1085 — Drug Absorption Window and Colon Absorption.

Multi-segment gut model to determine drug absorption window and quantify
colon contribution to overall bioavailability. Useful for extended-release
formulation design.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import exp

# ---------------------------------------------------------------------------
# Segment definitions
# ---------------------------------------------------------------------------

_SEGMENTS = [
    {"name": "duodenum", "pH": 5.5, "t_transit_h": 0.5, "length_m": 0.25},
    {"name": "jejunum_proximal", "pH": 6.2, "t_transit_h": 1.5, "length_m": 1.2},
    {"name": "jejunum_distal", "pH": 6.5, "t_transit_h": 1.5, "length_m": 1.2},
    {"name": "ileum_proximal", "pH": 7.0, "t_transit_h": 1.5, "length_m": 1.2},
    {"name": "ileum_distal", "pH": 7.2, "t_transit_h": 1.5, "length_m": 1.2},
    {"name": "colon", "pH": 6.8, "t_transit_h": 20.0, "length_m": 1.5},
]

_SI_SEGMENTS = [s["name"] for s in _SEGMENTS if s["name"] != "colon"]
_COLON_PEFF_FACTOR = 0.1  # Colon is 10x less permeable than SI

# Reference segment for pH factor normalisation (duodenum, pH 5.5)
_REF_PH = 5.5


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _f_unionized(pH: float, pka: float, ionization_type: str) -> float:
    """Henderson-Hasselbalch fraction of unionized drug at given pH."""
    if ionization_type == "neutral":
        return 1.0
    if ionization_type == "acid":
        # HA ⇌ H⁺ + A⁻  →  f_un = 1 / (1 + 10^(pH - pKa))
        return 1.0 / (1.0 + 10.0 ** (pH - pka))
    if ionization_type == "base":
        # BH⁺ ⇌ B + H⁺  →  f_un = 1 / (1 + 10^(pKa - pH))
        return 1.0 / (1.0 + 10.0 ** (pka - pH))
    raise ValueError(f"Unknown ionization_type: {ionization_type!r}")


def _ph_factor(pH: float, pka: float, ionization_type: str) -> float:
    """Peff multiplier relative to duodenum reference."""
    if ionization_type == "neutral":
        return 1.0
    f_ref = _f_unionized(_REF_PH, pka, ionization_type)
    if f_ref <= 0.0:
        return 0.0
    return _f_unionized(pH, pka, ionization_type) / f_ref


def _bioavailability_class(f: float) -> str:
    if f >= 0.8:
        return "high"
    if f >= 0.4:
        return "moderate"
    return "low"


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ColonAbsorptionResult:
    drug_name: str
    pka: float
    logP: float
    ionization_type: str
    fraction_absorbed_duodenum: float
    fraction_absorbed_jejunum: float
    fraction_absorbed_ileum: float
    fraction_absorbed_colon: float
    f_total_absorbed: float
    absorption_window_segments: tuple
    has_colon_absorption: bool
    er_design_feasible: bool
    bioavailability_class: str
    notes: str


# ---------------------------------------------------------------------------
# Core prediction function
# ---------------------------------------------------------------------------


def predict_absorption_window(
    drug_name: str,
    pka: float,
    logP: float,
    ionization_type: str,
    peff_base_cm_per_s: float = 1e-4,
) -> ColonAbsorptionResult:
    """Predict absorption window and colon contribution.

    Parameters
    ----------
    drug_name:
        Name of the drug (used only for labelling).
    pka:
        Acid/base pKa value (0–14).
    logP:
        Octanol-water log partition coefficient (-5 to 10).
    ionization_type:
        One of "acid", "base", "neutral".
    peff_base_cm_per_s:
        Basal effective permeability in the reference segment (cm/s).

    Returns
    -------
    ColonAbsorptionResult
    """
    # --- Validation -------------------------------------------------------
    if ionization_type not in {"acid", "base", "neutral"}:
        raise ValueError(
            f"ionization_type must be 'acid', 'base', or 'neutral', got {ionization_type!r}"
        )
    if not (0.0 <= pka <= 14.0):
        raise ValueError(f"pKa must be in [0, 14], got {pka}")
    if not (-5.0 <= logP <= 10.0):
        raise ValueError(f"logP must be in [-5, 10], got {logP}")
    if peff_base_cm_per_s <= 0.0:
        raise ValueError(f"peff_base_cm_per_s must be > 0, got {peff_base_cm_per_s}")

    # --- Per-segment computation ------------------------------------------
    seg_fractions: dict[str, float] = {}
    remaining = 1.0  # fraction of dose not yet absorbed

    for seg in _SEGMENTS:
        seg_name = seg["name"]
        pH = seg["pH"]
        t_transit = seg["t_transit_h"]

        # Peff adjusted for ionization and colon
        factor = _ph_factor(pH, pka, ionization_type)
        if seg_name == "colon":
            factor *= _COLON_PEFF_FACTOR

        # k_abs in 1/h: Peff (cm/s) * 3600 s/h / 0.175 cm (intestinal radius)
        k_abs_per_h = peff_base_cm_per_s * factor * 3600.0 / 0.175

        # Fraction absorbed in this segment from drug arriving here
        f_seg = 1.0 - exp(-k_abs_per_h * t_transit)
        f_seg = max(0.0, min(1.0, f_seg))

        # Absolute fraction of dose absorbed here
        absorbed_here = remaining * f_seg
        seg_fractions[seg_name] = absorbed_here
        remaining -= absorbed_here

    # --- Aggregate into region fractions ----------------------------------
    frac_duodenum = seg_fractions["duodenum"]
    frac_jejunum = seg_fractions["jejunum_proximal"] + seg_fractions["jejunum_distal"]
    frac_ileum = seg_fractions["ileum_proximal"] + seg_fractions["ileum_distal"]
    frac_colon = seg_fractions["colon"]

    f_total = frac_duodenum + frac_jejunum + frac_ileum + frac_colon
    f_total = min(1.0, max(0.0, f_total))

    # --- Absorption window (segments where >5 % of total dose absorbed) ---
    threshold = 0.05 * f_total if f_total > 0.0 else 0.0
    window_segs = tuple(s["name"] for s in _SEGMENTS if seg_fractions[s["name"]] > threshold)

    # --- Flags ------------------------------------------------------------
    has_colon_abs = frac_colon > 0.05 * f_total if f_total > 0.0 else False
    er_feasible = has_colon_abs

    # --- Notes ------------------------------------------------------------
    bcs_cls = _bioavailability_class(f_total)
    note_parts = [f"BCS-like class: {bcs_cls}"]
    if er_feasible:
        note_parts.append("Colon absorption supports ER formulation design.")
    else:
        note_parts.append("Minimal colon absorption; ER design may reduce bioavailability.")
    notes = " ".join(note_parts)

    return ColonAbsorptionResult(
        drug_name=drug_name,
        pka=pka,
        logP=logP,
        ionization_type=ionization_type,
        fraction_absorbed_duodenum=round(frac_duodenum, 6),
        fraction_absorbed_jejunum=round(frac_jejunum, 6),
        fraction_absorbed_ileum=round(frac_ileum, 6),
        fraction_absorbed_colon=round(frac_colon, 6),
        f_total_absorbed=round(f_total, 6),
        absorption_window_segments=window_segs,
        has_colon_absorption=has_colon_abs,
        er_design_feasible=er_feasible,
        bioavailability_class=bcs_cls,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Comparison utility
# ---------------------------------------------------------------------------


def compare_ionization_types(
    drug_name: str,
    pka: float = 5.0,
    logP: float = 2.0,
) -> list[ColonAbsorptionResult]:
    """Run predict_absorption_window for all three ionization types.

    Returns results sorted by f_total_absorbed descending.
    """
    results = [
        predict_absorption_window(drug_name, pka=pka, logP=logP, ionization_type=t)
        for t in ("acid", "base", "neutral")
    ]
    results.sort(key=lambda r: r.f_total_absorbed, reverse=True)
    return results
