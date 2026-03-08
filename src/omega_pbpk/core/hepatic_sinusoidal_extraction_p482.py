"""
Phase 482 — Hepatic Sinusoidal Extraction

Zonal model of hepatic extraction along the sinusoid.
Three zones (periportal, midzonal, centrilobular) with decreasing CYP activity.
Drug passes through zones sequentially.
"""

from __future__ import annotations

from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Zone definitions
# ---------------------------------------------------------------------------

_ZONES = [
    {"name": "periportal", "cyp_factor": 1.0, "fraction": 0.40},
    {"name": "midzonal", "cyp_factor": 0.8, "fraction": 0.35},
    {"name": "centrilobular", "cyp_factor": 0.5, "fraction": 0.25},
]


# ---------------------------------------------------------------------------
# Result dataclass (frozen=True — no list fields)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HepaticSinusoidalResult:
    """Result of hepatic sinusoidal extraction simulation."""

    drug_name: str
    fu_plasma: float
    cl_int_total_L_per_h: float
    Q_h_L_per_h: float
    # Per-zone extraction ratios
    e_zone1: float
    e_zone2: float
    e_zone3: float
    # Per-zone availability
    f_h_zone1: float
    f_h_zone2: float
    f_h_zone3: float
    # Cascade total
    f_h_total: float
    er_total: float
    cl_hepatic_L_per_h: float
    # Comparison with well-stirred model
    well_stirred_er: float
    zonal_vs_wellstirred_ratio: float
    notes: str


# ---------------------------------------------------------------------------
# Core function
# ---------------------------------------------------------------------------


def simulate_hepatic_sinusoidal_extraction(
    drug_name: str,
    fu_plasma: float,
    cl_int_total_L_per_h: float,
    Q_h_L_per_h: float = 90.0,
) -> HepaticSinusoidalResult:
    """Simulate hepatic sinusoidal extraction using a 3-zone model.

    Parameters
    ----------
    drug_name : str
        Name of the drug.
    fu_plasma : float
        Fraction unbound in plasma (0 < fu_plasma <= 1).
    cl_int_total_L_per_h : float
        Total intrinsic clearance (L/h).
    Q_h_L_per_h : float
        Hepatic blood flow (L/h). Default 90 L/h.

    Returns
    -------
    HepaticSinusoidalResult
    """
    # --- Validation ---
    if not (0 < fu_plasma <= 1):
        raise ValueError("fu_plasma must be in (0, 1]")
    if cl_int_total_L_per_h <= 0:
        raise ValueError("cl_int_total_L_per_h must be > 0")
    if Q_h_L_per_h <= 0:
        raise ValueError("Q_h_L_per_h must be > 0")

    # --- Per-zone extraction ---
    zone_e = []
    for zone in _ZONES:
        cyp_f = zone["cyp_factor"]
        frac = zone["fraction"]
        cl_int_zone = cl_int_total_L_per_h * cyp_f * frac
        Q_zone = Q_h_L_per_h * frac
        denom = Q_zone + fu_plasma * cl_int_zone
        e_zone = fu_plasma * cl_int_zone / denom if denom > 0 else 0.0
        zone_e.append(e_zone)

    e_zone1, e_zone2, e_zone3 = zone_e
    f_h_zone1 = 1.0 - e_zone1
    f_h_zone2 = 1.0 - e_zone2
    f_h_zone3 = 1.0 - e_zone3

    f_h_total = f_h_zone1 * f_h_zone2 * f_h_zone3
    er_total = 1.0 - f_h_total
    cl_hepatic = Q_h_L_per_h * er_total

    # --- Well-stirred model for comparison ---
    ws_denom = Q_h_L_per_h + fu_plasma * cl_int_total_L_per_h
    well_stirred_er = fu_plasma * cl_int_total_L_per_h / ws_denom if ws_denom > 0 else 0.0

    if well_stirred_er > 0:
        ratio = er_total / well_stirred_er
    else:
        ratio = 1.0

    # --- Notes ---
    if er_total < 0.3:
        extraction_class = "low extraction"
    elif er_total < 0.7:
        extraction_class = "intermediate extraction"
    else:
        extraction_class = "high extraction"

    notes = (
        f"Zonal model: {extraction_class} drug (ER={er_total:.3f}). "
        f"Well-stirred ER={well_stirred_er:.3f}, zonal/WS ratio={ratio:.3f}."
    )

    return HepaticSinusoidalResult(
        drug_name=drug_name,
        fu_plasma=fu_plasma,
        cl_int_total_L_per_h=cl_int_total_L_per_h,
        Q_h_L_per_h=Q_h_L_per_h,
        e_zone1=e_zone1,
        e_zone2=e_zone2,
        e_zone3=e_zone3,
        f_h_zone1=f_h_zone1,
        f_h_zone2=f_h_zone2,
        f_h_zone3=f_h_zone3,
        f_h_total=f_h_total,
        er_total=er_total,
        cl_hepatic_L_per_h=cl_hepatic,
        well_stirred_er=well_stirred_er,
        zonal_vs_wellstirred_ratio=ratio,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Screen multiple drugs
# ---------------------------------------------------------------------------


def screen_drugs_sinusoidal(drugs: list) -> list:
    """Screen multiple drugs; sorted by f_h_total descending (best F first).

    Parameters
    ----------
    drugs : list[dict]
        Each dict must have keys: "drug_name", "fu_plasma", "cl_int_total_L_per_h".
        Optional: "Q_h_L_per_h".

    Returns
    -------
    list[HepaticSinusoidalResult]
        Sorted by f_h_total descending.
    """
    results = []
    for d in drugs:
        result = simulate_hepatic_sinusoidal_extraction(
            drug_name=d["drug_name"],
            fu_plasma=d["fu_plasma"],
            cl_int_total_L_per_h=d["cl_int_total_L_per_h"],
            Q_h_L_per_h=d.get("Q_h_L_per_h", 90.0),
        )
        results.append(result)

    results.sort(key=lambda r: r.f_h_total, reverse=True)
    return results
