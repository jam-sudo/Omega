"""Phase 493 — Dialysis Drug Removal Kinetics.

Models drug removal during hemodialysis sessions based on MW,
protein binding, and dialyzer characteristics.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Dialyzer parameters
# ---------------------------------------------------------------------------

_DIALYZERS: dict[str, dict[str, float]] = {
    "high_flux": {"mw_cutoff": 50_000.0, "qd_mL_per_min": 500.0, "efficiency": 0.85},
    "low_flux": {"mw_cutoff": 15_000.0, "qd_mL_per_min": 500.0, "efficiency": 0.60},
    "high_efficiency": {"mw_cutoff": 25_000.0, "qd_mL_per_min": 800.0, "efficiency": 0.90},
    "peritoneal": {"mw_cutoff": 10_000.0, "qd_mL_per_min": 20.0, "efficiency": 0.40},
}

_VALID_DIALYZERS = frozenset(_DIALYZERS.keys())

# Default blood flow: 300 mL/min = 18 L/h
_QB_DEFAULT_L_PER_H: float = 300.0 * 60.0 / 1000.0  # 18.0 L/h


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DialysisRemovalResult:
    """Results from dialysis drug removal kinetics prediction."""

    drug_name: str
    dialyzer_type: str
    mw_Da: float
    fu_plasma: float
    vd_L: float
    sieving_coefficient: float
    cl_dialysis_L_per_h: float
    fraction_removed_per_session: float
    post_dialysis_rebound_pct: float
    supplemental_dose_pct: float
    dialysis_class: str
    notes: str


# ---------------------------------------------------------------------------
# Core prediction function
# ---------------------------------------------------------------------------


def predict_dialysis_removal(
    drug_name: str,
    dialyzer_type: str,
    mw_Da: float,
    fu_plasma: float,
    vd_L: float = 50.0,
    cl_body_L_per_h: float = 1.0,
    session_h: float = 4.0,
) -> DialysisRemovalResult:
    """Predict drug removal during a hemodialysis session.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    dialyzer_type:
        One of "high_flux", "low_flux", "high_efficiency", "peritoneal".
    mw_Da:
        Molecular weight in Daltons.
    fu_plasma:
        Unbound fraction in plasma (0 < fu_plasma <= 1).
    vd_L:
        Volume of distribution in litres.
    cl_body_L_per_h:
        Endogenous body clearance in L/h.
    session_h:
        Duration of dialysis session in hours.

    Returns
    -------
    DialysisRemovalResult
    """
    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    if dialyzer_type not in _VALID_DIALYZERS:
        raise ValueError(
            f"dialyzer_type must be one of {sorted(_VALID_DIALYZERS)}, got {dialyzer_type!r}"
        )
    if mw_Da <= 0:
        raise ValueError(f"mw_Da must be > 0, got {mw_Da}")
    if not (0 < fu_plasma <= 1):
        raise ValueError(f"fu_plasma must be in (0, 1], got {fu_plasma}")
    if vd_L <= 0:
        raise ValueError(f"vd_L must be > 0, got {vd_L}")
    if session_h <= 0:
        raise ValueError(f"session_h must be > 0, got {session_h}")

    # ------------------------------------------------------------------
    # Dialyzer parameters
    # ------------------------------------------------------------------
    params = _DIALYZERS[dialyzer_type]
    mw_cutoff = params["mw_cutoff"]
    efficiency = params["efficiency"]

    # Sieving coefficient: SC = fu * exp(-MW / MW_cutoff)
    sc = fu_plasma * math.exp(-mw_Da / mw_cutoff)
    # Clamp to [0, 1]
    sc = max(0.0, min(1.0, sc))

    # Dialysis clearance (L/h)
    cl_dialysis = sc * _QB_DEFAULT_L_PER_H * efficiency

    # Effective total clearance
    cl_eff = cl_dialysis + cl_body_L_per_h

    # Fraction removed during session using first-order kinetics
    # C(t) = C0 * exp(-CL_eff / Vd * t)
    k_elim = cl_eff / vd_L
    fraction_remaining = math.exp(-k_elim * session_h)
    fraction_removed = 1.0 - fraction_remaining
    fraction_removed = max(0.0, min(1.0, fraction_removed))

    # Post-dialysis rebound (20% for Vd > 50 L)
    rebound_pct = 20.0 if vd_L > 50.0 else 0.0

    # Supplemental dose percentage
    supplemental_dose_pct = fraction_removed * 100.0

    # Dialysis classification
    if sc < 0.1:
        dialysis_class = "not_dialyzable"
    elif sc <= 0.5:
        dialysis_class = "partially"
    else:
        dialysis_class = "well_dialyzable"

    # Notes
    notes_parts: list[str] = [
        f"SC={sc:.3f}",
        f"CL_dialysis={cl_dialysis:.2f} L/h",
        f"CL_eff={cl_eff:.2f} L/h",
        f"session={session_h}h",
        f"class={dialysis_class}",
    ]
    if rebound_pct > 0:
        notes_parts.append(f"rebound={rebound_pct:.0f}% (large Vd)")
    notes = "; ".join(notes_parts)

    return DialysisRemovalResult(
        drug_name=drug_name,
        dialyzer_type=dialyzer_type,
        mw_Da=mw_Da,
        fu_plasma=fu_plasma,
        vd_L=vd_L,
        sieving_coefficient=sc,
        cl_dialysis_L_per_h=cl_dialysis,
        fraction_removed_per_session=fraction_removed,
        post_dialysis_rebound_pct=rebound_pct,
        supplemental_dose_pct=supplemental_dose_pct,
        dialysis_class=dialysis_class,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Comparison across all dialyzers
# ---------------------------------------------------------------------------


def compare_dialyzers(
    drug_name: str,
    mw_Da: float,
    fu_plasma: float,
    vd_L: float = 50.0,
) -> list[DialysisRemovalResult]:
    """Compare drug removal across all four dialyzer types.

    Returns results sorted by fraction_removed_per_session descending.
    """
    results = [
        predict_dialysis_removal(
            drug_name=drug_name,
            dialyzer_type=dt,
            mw_Da=mw_Da,
            fu_plasma=fu_plasma,
            vd_L=vd_L,
        )
        for dt in _VALID_DIALYZERS
    ]
    results.sort(key=lambda r: r.fraction_removed_per_session, reverse=True)
    return results
