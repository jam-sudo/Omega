"""Phase 865 — Hepatic Zonation Model.

Models drug metabolism across hepatic zones (periportal, midzonal,
centrilobular) with oxygen and enzyme gradients using a well-stirred
approximation per zone in series.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class HepaticZonationResult:
    """Result of hepatic zonation simulation."""

    drug_name: str
    cyp_enzyme: str
    disease_state: str
    clint_total_mL_per_min: float
    q_hepatic_L_per_h: float
    fup: float
    c_out_zone1_fraction: float
    c_out_zone2_fraction: float
    c_out_zone3_fraction: float
    extraction_ratio: float
    cl_hepatic_L_per_h: float
    fh_bioavailability: float
    high_extraction: bool
    zone_contributing_most: str
    notes: str


def _apply_disease_clint(clint: float, disease_state: str) -> float:
    """Scale CLint based on disease state."""
    if disease_state == "normal":
        return clint
    elif disease_state == "fibrosis":
        return clint * 0.7
    elif disease_state == "cirrhosis":
        return clint * 0.4
    else:
        raise ValueError(
            f"Unknown disease_state: '{disease_state}'. "
            "Must be 'normal', 'fibrosis', or 'cirrhosis'."
        )


def simulate_hepatic_zonation(
    drug_name: str,
    clint_total_mL_per_min: float = 50.0,
    q_hepatic_L_per_h: float = 90.0,
    fup: float = 0.1,
    zone1_fraction: float = 0.4,
    zone2_fraction: float = 0.3,
    zone3_fraction: float = 0.3,
    cyp_enzyme: str = "mixed",
    disease_state: str = "normal",
) -> HepaticZonationResult:
    """Simulate hepatic zonation PK using well-stirred model per zone.

    Blood flows zone1 -> zone2 -> zone3 in series.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    clint_total_mL_per_min:
        Total intrinsic hepatic clearance (mL/min).
    q_hepatic_L_per_h:
        Hepatic blood flow (L/h). Default 90.0 L/h (~1.5 L/min).
    fup:
        Unbound fraction in plasma (0, 1].
    zone1_fraction:
        Fraction of CLint in periportal zone.
    zone2_fraction:
        Fraction of CLint in midzonal zone.
    zone3_fraction:
        Fraction of CLint in centrilobular zone.
    cyp_enzyme:
        Dominant enzyme: 'CYP1A2', 'CYP2E1', 'CYP3A4', or 'mixed'.
    disease_state:
        Liver disease state: 'normal', 'fibrosis', or 'cirrhosis'.

    Returns
    -------
    HepaticZonationResult
    """
    # Validation
    frac_sum = zone1_fraction + zone2_fraction + zone3_fraction
    if abs(frac_sum - 1.0) > 0.01:
        raise ValueError(f"zone fractions must sum to 1.0, got {frac_sum:.4f}")
    if clint_total_mL_per_min <= 0:
        raise ValueError("clint_total_mL_per_min must be > 0")
    if q_hepatic_L_per_h <= 0:
        raise ValueError("q_hepatic_L_per_h must be > 0")
    if not (0 < fup <= 1.0):
        raise ValueError("fup must be in (0, 1]")

    # Apply disease state scaling
    clint_effective = _apply_disease_clint(clint_total_mL_per_min, disease_state)

    # Convert CLint from mL/min to L/h: 1 mL/min = 0.06 L/h
    clint_L_per_h = clint_effective * 0.06

    # Per-zone CLint and flow (equal flow split across 3 zones)
    q_zone = q_hepatic_L_per_h / 3.0

    clint_z1 = clint_L_per_h * zone1_fraction
    clint_z2 = clint_L_per_h * zone2_fraction
    clint_z3 = clint_L_per_h * zone3_fraction

    # Well-stirred model per zone: C_out = C_in / (1 + CLint_z * fup / Q_z)
    # c_out_fraction = C_out / C_in for each zone
    c_out_z1 = 1.0 / (1.0 + clint_z1 * fup / q_zone)
    c_out_z2 = 1.0 / (1.0 + clint_z2 * fup / q_zone)
    c_out_z3 = 1.0 / (1.0 + clint_z3 * fup / q_zone)

    # Overall: blood passes through zones in series
    # C_out_total / C_in = product of individual fractions
    c_out_total_fraction = c_out_z1 * c_out_z2 * c_out_z3

    extraction_ratio = 1.0 - c_out_total_fraction
    # Clamp to [0, 1] for numerical safety
    extraction_ratio = max(0.0, min(1.0, extraction_ratio))

    cl_hepatic_L_per_h = q_hepatic_L_per_h * extraction_ratio
    fh_bioavailability = 1.0 - extraction_ratio

    high_extraction = extraction_ratio > 0.7

    # Determine zone contributing most (highest extraction fraction)
    ext_z1 = 1.0 - c_out_z1
    ext_z2 = 1.0 - c_out_z2
    ext_z3 = 1.0 - c_out_z3

    if ext_z1 >= ext_z2 and ext_z1 >= ext_z3:
        zone_contributing_most = "zone1"
    elif ext_z2 >= ext_z3:
        zone_contributing_most = "zone2"
    else:
        zone_contributing_most = "zone3"

    # Build notes
    notes_parts = []
    if disease_state != "normal":
        factor = clint_effective / clint_total_mL_per_min
        notes_parts.append(f"{disease_state} reduces CLint by {(1 - factor) * 100:.0f}%")
    if high_extraction:
        notes_parts.append("High extraction ratio: first-pass effect is significant")
    else:
        notes_parts.append("Low-to-moderate extraction ratio")

    notes = "; ".join(notes_parts) if notes_parts else "Normal hepatic zonation"

    return HepaticZonationResult(
        drug_name=drug_name,
        cyp_enzyme=cyp_enzyme,
        disease_state=disease_state,
        clint_total_mL_per_min=clint_total_mL_per_min,
        q_hepatic_L_per_h=q_hepatic_L_per_h,
        fup=fup,
        c_out_zone1_fraction=c_out_z1,
        c_out_zone2_fraction=c_out_z2,
        c_out_zone3_fraction=c_out_z3,
        extraction_ratio=extraction_ratio,
        cl_hepatic_L_per_h=cl_hepatic_L_per_h,
        fh_bioavailability=fh_bioavailability,
        high_extraction=high_extraction,
        zone_contributing_most=zone_contributing_most,
        notes=notes,
    )


def compare_disease_states(
    drug_name: str,
    clint_total_mL_per_min: float,
    q_hepatic_L_per_h: float = 90.0,
    fup: float = 0.1,
) -> list[HepaticZonationResult]:
    """Compare hepatic zonation across normal, fibrosis, and cirrhosis.

    Returns results sorted by cl_hepatic_L_per_h descending (highest first).
    """
    results = []
    for ds in ("normal", "fibrosis", "cirrhosis"):
        r = simulate_hepatic_zonation(
            drug_name=drug_name,
            clint_total_mL_per_min=clint_total_mL_per_min,
            q_hepatic_L_per_h=q_hepatic_L_per_h,
            fup=fup,
            disease_state=ds,
        )
        results.append(r)

    results.sort(key=lambda r: r.cl_hepatic_L_per_h, reverse=True)
    return results
