"""
Phase 513 — Hepatic Zonation CYP Activity Model

Models CYP activity variation across hepatic zones (periportal vs. centrilobular).
Hepatic lobule zones differ in O2 tension and dominant CYP enzymes.
"""

from __future__ import annotations

from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

V_LIVER_L = 1.5  # total liver volume (L)

# Zone fractions of total liver volume
ZONE_FRACTIONS = {
    "zone1": 0.30,  # periportal
    "zone2": 0.40,  # midzonal
    "zone3": 0.30,  # centrilobular
}

# Activity multipliers per zone
ZONE_ACTIVITY = {
    "zone1": 1.5,
    "zone2": 1.0,
    "zone3": 0.7,
}

# Dominant CYP mapping by zone
ZONE_DOMINANT_CYP = {
    "zone1": {"CYP1A2", "CYP2E1"},
    "zone2": {"CYP3A4"},
    "zone3": {"CYP2D6", "CYP2C9"},
}

# All valid CYP enzymes
VALID_CYPS = {"CYP1A2", "CYP2E1", "CYP3A4", "CYP2D6", "CYP2C9"}

# CYP dominant zone lookup (which zone multiplier applies to a given CYP)
CYP_ZONE_MULTIPLIERS = {
    "CYP1A2": {
        "zone1": ZONE_ACTIVITY["zone1"],
        "zone2": ZONE_ACTIVITY["zone2"],
        "zone3": ZONE_ACTIVITY["zone3"],
    },
    "CYP2E1": {
        "zone1": ZONE_ACTIVITY["zone1"],
        "zone2": ZONE_ACTIVITY["zone2"],
        "zone3": ZONE_ACTIVITY["zone3"],
    },
    "CYP3A4": {
        "zone1": ZONE_ACTIVITY["zone1"],
        "zone2": ZONE_ACTIVITY["zone2"],
        "zone3": ZONE_ACTIVITY["zone3"],
    },
    "CYP2D6": {
        "zone1": ZONE_ACTIVITY["zone1"],
        "zone2": ZONE_ACTIVITY["zone2"],
        "zone3": ZONE_ACTIVITY["zone3"],
    },
    "CYP2C9": {
        "zone1": ZONE_ACTIVITY["zone1"],
        "zone2": ZONE_ACTIVITY["zone2"],
        "zone3": ZONE_ACTIVITY["zone3"],
    },
}

ZONATION_INDEX = ZONE_ACTIVITY["zone3"] / ZONE_ACTIVITY["zone1"]  # 0.7 / 1.5 ≈ 0.4667


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HepaticZonationResult:
    """Result of hepatic zonation CYP activity simulation."""

    drug_name: str
    fup: float
    cl_int_total_L_per_h: float
    dominant_cyp: str
    cl_int_zone1_L_per_h: float
    cl_int_zone2_L_per_h: float
    cl_int_zone3_L_per_h: float
    cl_int_effective_L_per_h: float
    eh_hepatic: float
    fh_hepatic: float
    cl_hepatic_L_per_h: float
    zonation_index: float
    metabolic_risk: str
    notes: str


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------


def _compute_zone_clints(cl_int_total: float) -> tuple[float, float, float]:
    """Compute CLint for each zone.

    CLint_zone_i = CLint_total * zone_activity_i * zone_fraction_i
    """
    cl_z1 = cl_int_total * ZONE_ACTIVITY["zone1"] * ZONE_FRACTIONS["zone1"]
    cl_z2 = cl_int_total * ZONE_ACTIVITY["zone2"] * ZONE_FRACTIONS["zone2"]
    cl_z3 = cl_int_total * ZONE_ACTIVITY["zone3"] * ZONE_FRACTIONS["zone3"]
    return cl_z1, cl_z2, cl_z3


def simulate_hepatic_zonation(
    drug_name: str,
    fup: float,
    cl_int_total_L_per_h: float,
    dominant_cyp: str,
    q_hepatic_L_per_h: float = 90.0,
) -> HepaticZonationResult:
    """Simulate hepatic zonation CYP activity model.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    fup:
        Fraction unbound in plasma (0 < fup <= 1).
    cl_int_total_L_per_h:
        Total intrinsic clearance (L/h), > 0.
    dominant_cyp:
        Dominant CYP enzyme; one of CYP1A2, CYP2E1, CYP3A4, CYP2D6, CYP2C9.
    q_hepatic_L_per_h:
        Hepatic blood flow (L/h), default 90 L/h.

    Returns
    -------
    HepaticZonationResult
    """
    # Validation
    if not (0.0 < fup <= 1.0):
        raise ValueError(f"fup must be in (0, 1], got {fup}")
    if cl_int_total_L_per_h <= 0.0:
        raise ValueError(f"cl_int_total_L_per_h must be > 0, got {cl_int_total_L_per_h}")
    if q_hepatic_L_per_h <= 0.0:
        raise ValueError(f"q_hepatic_L_per_h must be > 0, got {q_hepatic_L_per_h}")
    if dominant_cyp not in VALID_CYPS:
        raise ValueError(f"dominant_cyp must be one of {VALID_CYPS}, got {dominant_cyp!r}")

    # Zone CLints
    cl_z1, cl_z2, cl_z3 = _compute_zone_clints(cl_int_total_L_per_h)

    # Effective CLint = sum of all zone CLints
    cl_eff = cl_z1 + cl_z2 + cl_z3

    # Well-stirred hepatic model
    numerator = fup * cl_eff
    eh = numerator / (q_hepatic_L_per_h + numerator)
    fh = 1.0 - eh
    cl_hepatic = q_hepatic_L_per_h * eh

    # Metabolic risk: high if dominant CYP is CYP2D6 or CYP2C9 (zone3, centrilobular toxic metabolites)
    metabolic_risk = "high" if dominant_cyp in {"CYP2D6", "CYP2C9"} else "normal"

    notes = (
        f"Zonation index (Z3/Z1) = {ZONATION_INDEX:.3f}. "
        f"Dominant CYP {dominant_cyp} metabolic risk: {metabolic_risk}. "
        f"Effective CLint aggregated over 3 hepatic zones."
    )

    return HepaticZonationResult(
        drug_name=drug_name,
        fup=fup,
        cl_int_total_L_per_h=cl_int_total_L_per_h,
        dominant_cyp=dominant_cyp,
        cl_int_zone1_L_per_h=cl_z1,
        cl_int_zone2_L_per_h=cl_z2,
        cl_int_zone3_L_per_h=cl_z3,
        cl_int_effective_L_per_h=cl_eff,
        eh_hepatic=eh,
        fh_hepatic=fh,
        cl_hepatic_L_per_h=cl_hepatic,
        zonation_index=ZONATION_INDEX,
        metabolic_risk=metabolic_risk,
        notes=notes,
    )


def compare_cyp_enzymes(
    drug_name: str,
    fup: float,
    cl_int_total_L_per_h: float,
    q_hepatic_L_per_h: float = 90.0,
) -> list[HepaticZonationResult]:
    """Run simulation for all 5 CYP enzymes and return results sorted by EH ascending.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    fup:
        Fraction unbound in plasma.
    cl_int_total_L_per_h:
        Total intrinsic clearance (L/h).
    q_hepatic_L_per_h:
        Hepatic blood flow (L/h).

    Returns
    -------
    list of HepaticZonationResult sorted by eh_hepatic ascending
    """
    results = []
    for cyp in sorted(VALID_CYPS):
        res = simulate_hepatic_zonation(
            drug_name=drug_name,
            fup=fup,
            cl_int_total_L_per_h=cl_int_total_L_per_h,
            dominant_cyp=cyp,
            q_hepatic_L_per_h=q_hepatic_L_per_h,
        )
        results.append(res)
    # Sort by eh_hepatic ascending
    results.sort(key=lambda r: r.eh_hepatic)
    return results
