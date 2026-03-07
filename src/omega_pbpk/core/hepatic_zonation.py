"""
Phase 1077 -- Hepatic Zone-Specific Metabolism

Models zone-specific hepatic metabolism accounting for periportal (Zone 1)
vs. pericentral (Zone 3) enzyme expression gradients.
"""

from __future__ import annotations

from dataclasses import dataclass

# Zone enzyme multipliers: (zone1, zone2, zone3)
_CYP_ZONE_MULTIPLIERS: dict[str, tuple[float, float, float]] = {
    "CYP3A4": (0.5, 1.0, 2.0),
    "CYP2E1": (0.3, 0.8, 2.5),
    "CYP2D6": (2.0, 1.0, 0.5),
    "CYP2C9": (1.0, 1.0, 1.0),
    "CYP1A2": (0.8, 1.0, 1.5),
}

_SUPPORTED_CYPS = list(_CYP_ZONE_MULTIPLIERS.keys())


@dataclass(frozen=True)
class HepaticZonationResult:
    drug_name: str
    cyp_enzyme: str
    clint_total_mL_per_min_per_g: float
    q_portal_L_per_h: float
    fu_plasma: float
    e_zone1: float
    e_zone2: float
    e_zone3: float
    e_total: float
    cl_hepatic_L_per_h: float
    fu_mic: float
    first_pass_effect: str
    zone_sensitivity: str
    notes: str


def _validate_inputs(
    clint_total: float,
    q_portal: float,
    fu_plasma: float,
    fu_mic: float,
    liver_weight: float,
) -> None:
    if clint_total <= 0:
        raise ValueError(f"clint_total_mL_per_min_per_g must be > 0, got {clint_total}")
    if q_portal <= 0:
        raise ValueError(f"q_portal_L_per_h must be > 0, got {q_portal}")
    if not (0 < fu_plasma <= 1):
        raise ValueError(f"fu_plasma must be in (0, 1], got {fu_plasma}")
    if not (0 < fu_mic <= 1):
        raise ValueError(f"fu_mic must be in (0, 1], got {fu_mic}")
    if liver_weight <= 0:
        raise ValueError(f"liver_weight_g must be > 0, got {liver_weight}")


def simulate_hepatic_zonation(
    drug_name: str,
    cyp_enzyme: str,
    clint_total_mL_per_min_per_g: float = 10.0,
    q_portal_L_per_h: float = 60.0,
    fu_plasma: float = 0.1,
    fu_mic: float = 0.5,
    liver_weight_g: float = 1500.0,
) -> HepaticZonationResult:
    """Simulate hepatic zonation-dependent metabolism using a 3-zone series model."""
    cyp_upper = cyp_enzyme.upper()
    if cyp_upper not in _CYP_ZONE_MULTIPLIERS:
        raise ValueError(f"Unsupported CYP enzyme '{cyp_enzyme}'. Supported: {_SUPPORTED_CYPS}")

    _validate_inputs(
        clint_total_mL_per_min_per_g,
        q_portal_L_per_h,
        fu_plasma,
        fu_mic,
        liver_weight_g,
    )

    multipliers = _CYP_ZONE_MULTIPLIERS[cyp_upper]
    # Normalize multipliers so their sum equals 3 (preserving total CLint scale)
    mult_sum = sum(multipliers)
    norm_multipliers = tuple(m * 3.0 / mult_sum for m in multipliers)

    # Convert CLint from mL/min/g to L/h for whole liver
    clint_total_L_per_h = clint_total_mL_per_min_per_g * liver_weight_g * 60.0 / 1000.0

    # Split CLint equally across 3 zones (base), then apply zone multipliers
    clint_per_zone_base = clint_total_L_per_h / 3.0
    q_zone = q_portal_L_per_h / 3.0  # each zone gets 1/3 of portal flow

    # Extraction ratio per zone using well-stirred model
    # E_z = (fu * CLint_z) / (Q_zone + fu * CLint_z)
    def zone_extraction(mult: float) -> float:
        clint_z = clint_per_zone_base * mult
        numerator = fu_plasma * clint_z
        denominator = q_zone + fu_plasma * clint_z
        return numerator / denominator

    e1 = zone_extraction(norm_multipliers[0])
    e2 = zone_extraction(norm_multipliers[1])
    e3 = zone_extraction(norm_multipliers[2])

    e_total = 1.0 - (1.0 - e1) * (1.0 - e2) * (1.0 - e3)
    cl_hepatic = q_portal_L_per_h * e_total

    # First-pass effect classification
    if e_total < 0.3:
        first_pass_effect = "low"
    elif e_total <= 0.7:
        first_pass_effect = "intermediate"
    else:
        first_pass_effect = "high"

    # Zone sensitivity: periportal if zone1 raw multiplier > zone3 raw multiplier
    raw_z1 = multipliers[0]
    raw_z3 = multipliers[2]
    if raw_z1 > raw_z3:
        zone_sensitivity = "periportal"
    else:
        zone_sensitivity = "pericentral"

    notes = (
        f"{cyp_upper} shows {zone_sensitivity} dominance. "
        f"E_total={e_total:.3f} ({first_pass_effect}). "
        f"CL_hep={cl_hepatic:.2f} L/h."
    )

    return HepaticZonationResult(
        drug_name=drug_name,
        cyp_enzyme=cyp_upper,
        clint_total_mL_per_min_per_g=clint_total_mL_per_min_per_g,
        q_portal_L_per_h=q_portal_L_per_h,
        fu_plasma=fu_plasma,
        e_zone1=e1,
        e_zone2=e2,
        e_zone3=e3,
        e_total=e_total,
        cl_hepatic_L_per_h=cl_hepatic,
        fu_mic=fu_mic,
        first_pass_effect=first_pass_effect,
        zone_sensitivity=zone_sensitivity,
        notes=notes,
    )


def compare_cyp_enzymes(
    drug_name: str,
    clint_total: float = 10.0,
    **kwargs,
) -> list[HepaticZonationResult]:
    """Run simulation for all 5 supported CYP enzymes, sorted by cl_hepatic descending."""
    results = []
    for cyp in _SUPPORTED_CYPS:
        result = simulate_hepatic_zonation(
            drug_name=drug_name,
            cyp_enzyme=cyp,
            clint_total_mL_per_min_per_g=clint_total,
            **kwargs,
        )
        results.append(result)
    results.sort(key=lambda r: r.cl_hepatic_L_per_h, reverse=True)
    return results
