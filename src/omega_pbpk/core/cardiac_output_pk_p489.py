"""Phase 489 — Cardiac Output Effect on PK.

Models the effect of cardiac output on drug distribution and clearance.
CO affects organ blood flows → distribution kinetics.
"""

from __future__ import annotations

from dataclasses import dataclass

# ---------------------------------------------------------------------------
# CO Conditions database
# ---------------------------------------------------------------------------

_CO_CONDITIONS: dict[str, dict[str, float]] = {
    "normal": {
        "co_L_per_min": 5.0,
        "Q_hepatic_fraction": 0.27,
        "Q_renal_fraction": 0.20,
    },
    "heart_failure_mild": {
        "co_L_per_min": 3.5,
        "Q_hepatic_fraction": 0.25,
        "Q_renal_fraction": 0.18,
    },
    "heart_failure_severe": {
        "co_L_per_min": 2.0,
        "Q_hepatic_fraction": 0.20,
        "Q_renal_fraction": 0.15,
    },
    "exercise_moderate": {
        "co_L_per_min": 12.0,
        "Q_hepatic_fraction": 0.10,
        "Q_renal_fraction": 0.15,
    },
    "exercise_intense": {
        "co_L_per_min": 20.0,
        "Q_hepatic_fraction": 0.05,
        "Q_renal_fraction": 0.10,
    },
}

_VALID_CONDITIONS = set(_CO_CONDITIONS.keys())

# Normal reference values for AUC ratio computation
_NORMAL_CO = 5.0
_NORMAL_Q_RENAL_FRAC = 0.20
_BASE_GFR_ML_PER_MIN = 120.0


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CardiacOutputPKResult:
    """Result of cardiac output PK prediction."""

    drug_name: str
    co_condition: str
    cardiac_output_L_per_min: float
    Q_hepatic_L_per_h: float
    Q_renal_L_per_h: float
    GFR_mL_per_min: float
    cl_hepatic_L_per_h: float
    cl_renal_L_per_h: float
    cl_total_L_per_h: float
    vd_L: float
    t_half_h: float
    auc_ratio_vs_normal: float
    pk_change: str
    notes: str


# ---------------------------------------------------------------------------
# Core computation helpers
# ---------------------------------------------------------------------------


def _compute_result(
    drug_name: str,
    co_condition: str,
    fu_plasma: float,
    cl_int_hepatic_L_per_h: float,
    vd_L: float,
    cl_total_normal: float,
) -> CardiacOutputPKResult:
    params = _CO_CONDITIONS[co_condition]
    co = params["co_L_per_min"]
    q_hep_frac = params["Q_hepatic_fraction"]
    q_ren_frac = params["Q_renal_fraction"]

    # Convert flows to L/h
    q_h = co * 60.0 * q_hep_frac  # L/h
    q_r = co * 60.0 * q_ren_frac  # L/h

    # GFR scales with CO and renal fraction
    gfr = _BASE_GFR_ML_PER_MIN * (co / _NORMAL_CO) * (q_ren_frac / _NORMAL_Q_RENAL_FRAC)

    # Hepatic clearance (well-stirred model)
    fu_clint = fu_plasma * cl_int_hepatic_L_per_h
    cl_h = q_h * fu_clint / (q_h + fu_clint)

    # Renal clearance
    cl_r = gfr * fu_plasma * 60.0 / 1000.0  # mL/min → L/h

    cl_total = cl_h + cl_r

    # Half-life
    t_half = 0.693 * vd_L / cl_total

    # AUC ratio vs normal (AUC ∝ 1/CL)
    auc_ratio = cl_total_normal / cl_total if cl_total > 0.0 else 1.0

    # PK change classification (based on % change from normal)
    pct_change = abs(auc_ratio - 1.0) * 100.0
    if pct_change < 10.0:
        pk_change = "no_change"
    elif pct_change < 25.0:
        pk_change = "minor"
    elif pct_change < 50.0:
        pk_change = "moderate"
    else:
        pk_change = "major"

    # Notes
    if co_condition == "normal":
        notes = "Reference condition."
    elif "heart_failure" in co_condition:
        notes = (
            f"Reduced CO ({co} L/min) decreases hepatic/renal flow → lower CL → "
            f"higher AUC (ratio {auc_ratio:.2f}). Consider dose reduction."
        )
    else:
        notes = (
            f"Elevated CO ({co} L/min) due to exercise. Hepatic fraction reduced; "
            f"net PK effect depends on extraction ratio."
        )

    return CardiacOutputPKResult(
        drug_name=drug_name,
        co_condition=co_condition,
        cardiac_output_L_per_min=co,
        Q_hepatic_L_per_h=round(q_h, 4),
        Q_renal_L_per_h=round(q_r, 4),
        GFR_mL_per_min=round(gfr, 4),
        cl_hepatic_L_per_h=round(cl_h, 6),
        cl_renal_L_per_h=round(cl_r, 6),
        cl_total_L_per_h=round(cl_total, 6),
        vd_L=vd_L,
        t_half_h=round(t_half, 6),
        auc_ratio_vs_normal=round(auc_ratio, 6),
        pk_change=pk_change,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def predict_cardiac_output_pk(
    drug_name: str,
    co_condition: str,
    fu_plasma: float,
    cl_int_hepatic_L_per_h: float,
    vd_L: float = 50.0,
) -> CardiacOutputPKResult:
    """Predict PK parameters under a given cardiac output condition.

    Parameters
    ----------
    drug_name:
        Identifier for the drug.
    co_condition:
        One of 'normal', 'heart_failure_mild', 'heart_failure_severe',
        'exercise_moderate', 'exercise_intense'.
    fu_plasma:
        Unbound fraction in plasma (0 < fu <= 1).
    cl_int_hepatic_L_per_h:
        Intrinsic hepatic clearance (L/h).
    vd_L:
        Volume of distribution (L). Default 50.0.

    Returns
    -------
    CardiacOutputPKResult
    """
    if co_condition not in _VALID_CONDITIONS:
        raise ValueError(
            f"co_condition must be one of {sorted(_VALID_CONDITIONS)}, got '{co_condition}'"
        )
    if not (0 < fu_plasma <= 1.0):
        raise ValueError(f"fu_plasma must be in (0, 1], got {fu_plasma}")
    if cl_int_hepatic_L_per_h <= 0:
        raise ValueError(f"cl_int_hepatic_L_per_h must be > 0, got {cl_int_hepatic_L_per_h}")
    if vd_L <= 0:
        raise ValueError(f"vd_L must be > 0, got {vd_L}")

    # Compute normal CL first (for AUC ratio denominator)
    normal_params = _CO_CONDITIONS["normal"]
    co_n = normal_params["co_L_per_min"]
    q_h_n = co_n * 60.0 * normal_params["Q_hepatic_fraction"]
    gfr_n = _BASE_GFR_ML_PER_MIN  # normal GFR = 120 mL/min
    fu_clint = fu_plasma * cl_int_hepatic_L_per_h
    cl_h_n = q_h_n * fu_clint / (q_h_n + fu_clint)
    cl_r_n = gfr_n * fu_plasma * 60.0 / 1000.0
    cl_total_normal = cl_h_n + cl_r_n

    return _compute_result(
        drug_name=drug_name,
        co_condition=co_condition,
        fu_plasma=fu_plasma,
        cl_int_hepatic_L_per_h=cl_int_hepatic_L_per_h,
        vd_L=vd_L,
        cl_total_normal=cl_total_normal,
    )


def compare_co_conditions(
    drug_name: str,
    fu_plasma: float,
    cl_int_hepatic_L_per_h: float,
    vd_L: float = 50.0,
) -> list[CardiacOutputPKResult]:
    """Compare PK across all 5 CO conditions.

    Returns list sorted by cl_total_L_per_h descending.
    """
    if not (0 < fu_plasma <= 1.0):
        raise ValueError(f"fu_plasma must be in (0, 1], got {fu_plasma}")
    if cl_int_hepatic_L_per_h <= 0:
        raise ValueError(f"cl_int_hepatic_L_per_h must be > 0, got {cl_int_hepatic_L_per_h}")
    if vd_L <= 0:
        raise ValueError(f"vd_L must be > 0, got {vd_L}")

    # Normal CL for ratio computation
    normal_params = _CO_CONDITIONS["normal"]
    co_n = normal_params["co_L_per_min"]
    q_h_n = co_n * 60.0 * normal_params["Q_hepatic_fraction"]
    gfr_n = _BASE_GFR_ML_PER_MIN
    fu_clint = fu_plasma * cl_int_hepatic_L_per_h
    cl_h_n = q_h_n * fu_clint / (q_h_n + fu_clint)
    cl_r_n = gfr_n * fu_plasma * 60.0 / 1000.0
    cl_total_normal = cl_h_n + cl_r_n

    results = [
        _compute_result(
            drug_name=drug_name,
            co_condition=cond,
            fu_plasma=fu_plasma,
            cl_int_hepatic_L_per_h=cl_int_hepatic_L_per_h,
            vd_L=vd_L,
            cl_total_normal=cl_total_normal,
        )
        for cond in _VALID_CONDITIONS
    ]

    return sorted(results, key=lambda r: r.cl_total_L_per_h, reverse=True)
