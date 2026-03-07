"""Hepatic zonal drug metabolism model.

Models zone-dependent hepatic drug metabolism across three hepatic zones:
  Zone 1 (periportal):     High O2, receives portal blood first
  Zone 2 (midzonal):       Intermediate conditions
  Zone 3 (centrilobular):  Low O2, CYP2E1/1A2/3A4 predominant, toxicity prone

References:
  - Jungermann & Kietzmann (1996). Zonation of parenchymal and nonparenchymal
    metabolism in liver. Annu Rev Nutr.
  - Lindros (1997). Zonation of cytochrome P450 expression, drug metabolism
    and toxicity in liver. Gen Pharmacol.
"""

from __future__ import annotations

from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_ZONE1_MASS_FRACTION = 0.35  # periportal zone fraction of liver mass
_ZONE2_MASS_FRACTION = 0.20  # midzonal zone fraction
_ZONE3_MASS_FRACTION = 0.45  # centrilobular zone fraction

_DEFAULT_HEPATIC_BLOOD_FLOW = 90.0  # L/h
_DEFAULT_FU_PLASMA = 0.5
_DEFAULT_CLINT_ZONE1 = 0.5   # mL/min/g liver
_DEFAULT_CLINT_ZONE3 = 2.0   # mL/min/g liver
_DEFAULT_LIVER_MASS_G = 1500.0

_REACTIVE_METABOLITE_ZONE3_FRACTION_HIGH = 0.7
_REACTIVE_METABOLITE_ZONE3_FRACTION_LOW = 0.3


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HepaticZonationResult:
    """Results of hepatic zonal metabolism simulation."""

    drug_name: str
    dose_mg: float

    # Zonal distribution
    f_metabolized_zone1: float  # fraction metabolized in Zone 1 (periportal)
    f_metabolized_zone2: float  # fraction in Zone 2 (midzonal)
    f_metabolized_zone3: float  # fraction in Zone 3 (centrilobular)

    # Toxicity-relevant metrics
    zone3_burden_ratio: float  # Zone 3 metabolic burden relative to Zone 1
    reactive_metabolite_zone3_fraction: float  # RM fraction in Zone 3

    # PK metrics
    cl_total_L_per_h: float
    cl_intrinsic_zone1_L_per_h: float
    cl_intrinsic_zone3_L_per_h: float
    extraction_ratio: float

    hepatotoxicity_risk: str  # "low", "moderate", "high"
    notes: str


# ---------------------------------------------------------------------------
# Core simulation
# ---------------------------------------------------------------------------


def simulate_hepatic_zonation(
    drug_name: str,
    dose_mg: float,
    hepatic_blood_flow_L_per_h: float = _DEFAULT_HEPATIC_BLOOD_FLOW,
    fu_plasma: float = _DEFAULT_FU_PLASMA,
    clint_zone1_mL_per_min_per_g: float = _DEFAULT_CLINT_ZONE1,
    clint_zone3_mL_per_min_per_g: float = _DEFAULT_CLINT_ZONE3,
    liver_mass_g: float = _DEFAULT_LIVER_MASS_G,
    cyp_zone1_fraction: float = 0.35,
    cyp_zone3_fraction: float = 0.50,
    reactive_metabolite: bool = False,
) -> HepaticZonationResult:
    """Simulate zone-dependent hepatic drug metabolism.

    Uses a three-zone well-stirred liver model where each zone receives
    equal blood flow (1/3 of total hepatic flow).

    Parameters
    ----------
    drug_name:
        Identifier for the drug.
    dose_mg:
        Administered dose in mg.
    hepatic_blood_flow_L_per_h:
        Total hepatic blood flow (L/h). Default 90 L/h.
    fu_plasma:
        Unbound fraction in plasma (0 < fu <= 1).
    clint_zone1_mL_per_min_per_g:
        Intrinsic clearance per gram of liver in Zone 1 (mL/min/g).
    clint_zone3_mL_per_min_per_g:
        Intrinsic clearance per gram of liver in Zone 3 (mL/min/g).
    liver_mass_g:
        Total liver mass in grams.
    cyp_zone1_fraction:
        Fraction of CYP enzymes in Zone 1 (informational).
    cyp_zone3_fraction:
        Fraction of CYP enzymes in Zone 3 (informational).
    reactive_metabolite:
        Whether the drug forms reactive metabolites predominantly via Zone 3 CYPs.

    Returns
    -------
    HepaticZonationResult
    """
    # Input validation
    if hepatic_blood_flow_L_per_h <= 0:
        raise ValueError(
            f"hepatic_blood_flow_L_per_h must be > 0, got {hepatic_blood_flow_L_per_h}"
        )
    if clint_zone1_mL_per_min_per_g < 0:
        raise ValueError(
            f"clint_zone1_mL_per_min_per_g must be >= 0, got {clint_zone1_mL_per_min_per_g}"
        )
    if clint_zone3_mL_per_min_per_g < 0:
        raise ValueError(
            f"clint_zone3_mL_per_min_per_g must be >= 0, got {clint_zone3_mL_per_min_per_g}"
        )
    if liver_mass_g <= 0:
        raise ValueError(f"liver_mass_g must be > 0, got {liver_mass_g}")
    if not (0 < fu_plasma <= 1):
        raise ValueError(f"fu_plasma must be in (0, 1], got {fu_plasma}")

    # Zonal liver mass (g)
    mass_z1 = liver_mass_g * _ZONE1_MASS_FRACTION  # 35%
    mass_z2 = liver_mass_g * _ZONE2_MASS_FRACTION  # 20%
    mass_z3 = liver_mass_g * _ZONE3_MASS_FRACTION  # 45%

    # Midzone CLint: average of zone1 and zone3 rates
    clint_zone2_mL_per_min_per_g = (
        clint_zone1_mL_per_min_per_g + clint_zone3_mL_per_min_per_g
    ) / 2.0

    # Convert CLint from mL/min/g to L/h per zone
    # CLint_zone [L/h] = clint [mL/min/g] * mass [g] * 60 [min/h] / 1000 [mL/L]
    cl_int_z1 = clint_zone1_mL_per_min_per_g * mass_z1 * 60.0 / 1000.0
    cl_int_z2 = clint_zone2_mL_per_min_per_g * mass_z2 * 60.0 / 1000.0
    cl_int_z3 = clint_zone3_mL_per_min_per_g * mass_z3 * 60.0 / 1000.0

    # Blood flow per zone: equal thirds
    q_zone = hepatic_blood_flow_L_per_h / 3.0

    # Well-stirred extraction ratio per zone: E = fu * CLint / (Q + fu * CLint)
    def _well_stirred_extraction(q: float, fu: float, cl_int: float) -> float:
        denom = q + fu * cl_int
        if denom <= 0:
            return 0.0
        return (fu * cl_int) / denom

    e_z1 = _well_stirred_extraction(q_zone, fu_plasma, cl_int_z1)
    e_z2 = _well_stirred_extraction(q_zone, fu_plasma, cl_int_z2)
    e_z3 = _well_stirred_extraction(q_zone, fu_plasma, cl_int_z3)

    # Total hepatic extraction (simplified averaging)
    e_total = (e_z1 + e_z2 + e_z3) / 3.0
    e_total = min(1.0, max(0.0, e_total))

    cl_total = hepatic_blood_flow_L_per_h * e_total

    # Fraction metabolized per zone (proportional to fu * CLint)
    fu_clint_z1 = fu_plasma * cl_int_z1
    fu_clint_z2 = fu_plasma * cl_int_z2
    fu_clint_z3 = fu_plasma * cl_int_z3
    total_fu_clint = fu_clint_z1 + fu_clint_z2 + fu_clint_z3

    if total_fu_clint <= 0:
        # No metabolism: distribute uniformly
        f_z1 = _ZONE1_MASS_FRACTION
        f_z2 = _ZONE2_MASS_FRACTION
        f_z3 = _ZONE3_MASS_FRACTION
    else:
        f_z1 = fu_clint_z1 / total_fu_clint
        f_z2 = fu_clint_z2 / total_fu_clint
        f_z3 = fu_clint_z3 / total_fu_clint

    # Zone 3 burden relative to Zone 1
    if f_z1 > 0:
        zone3_burden_ratio = f_z3 / f_z1
    else:
        zone3_burden_ratio = 0.0

    # Reactive metabolite fraction in Zone 3
    rm_z3_fraction = (
        _REACTIVE_METABOLITE_ZONE3_FRACTION_HIGH
        if reactive_metabolite
        else _REACTIVE_METABOLITE_ZONE3_FRACTION_LOW
    )

    # Hepatotoxicity risk classification
    if zone3_burden_ratio > 3.0 and reactive_metabolite:
        hepatotoxicity_risk = "high"
    elif zone3_burden_ratio > 2.0 or (
        reactive_metabolite and zone3_burden_ratio > 1.0
    ):
        hepatotoxicity_risk = "moderate"
    else:
        hepatotoxicity_risk = "low"

    notes = (
        f"{drug_name}: Zone 3 burden ratio {zone3_burden_ratio:.2f}; "
        f"extraction ratio {e_total:.3f}; "
        f"reactive_metabolite={reactive_metabolite}; "
        f"CYP zone1_frac={cyp_zone1_fraction:.2f}, zone3_frac={cyp_zone3_fraction:.2f}"
    )

    return HepaticZonationResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        f_metabolized_zone1=f_z1,
        f_metabolized_zone2=f_z2,
        f_metabolized_zone3=f_z3,
        zone3_burden_ratio=zone3_burden_ratio,
        reactive_metabolite_zone3_fraction=rm_z3_fraction,
        cl_total_L_per_h=cl_total,
        cl_intrinsic_zone1_L_per_h=cl_int_z1,
        cl_intrinsic_zone3_L_per_h=cl_int_z3,
        extraction_ratio=e_total,
        hepatotoxicity_risk=hepatotoxicity_risk,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Enzyme induction comparison
# ---------------------------------------------------------------------------


def compare_enzyme_inducers(
    drug_name: str,
    dose_mg: float,
    fu_plasma: float,
    clint_z1_baseline: float,
    fold_induction_z3_values: list[float] | None = None,
) -> list[HepaticZonationResult]:
    """Compare effect of Zone 3 CYP induction on hepatic zonal metabolism.

    CYP induction (e.g., by rifampin, phenobarbital) preferentially upregulates
    CYP3A4/2E1/1A2 in Zone 3. This function evaluates a range of induction
    fold-changes applied to the Zone 3 intrinsic clearance.

    Parameters
    ----------
    drug_name:
        Drug identifier.
    dose_mg:
        Administered dose in mg.
    fu_plasma:
        Unbound fraction in plasma.
    clint_z1_baseline:
        Zone 1 baseline intrinsic clearance (mL/min/g).
    fold_induction_z3_values:
        List of fold-induction values to simulate for Zone 3.
        Default: [1, 2, 5, 10].

    Returns
    -------
    list[HepaticZonationResult]
        Results sorted by zone3_burden_ratio descending.
    """
    if fold_induction_z3_values is None:
        fold_induction_z3_values = [1, 2, 5, 10]

    results = []
    for fold in fold_induction_z3_values:
        clint_z3 = clint_z1_baseline * fold
        result = simulate_hepatic_zonation(
            drug_name=drug_name,
            dose_mg=dose_mg,
            fu_plasma=fu_plasma,
            clint_zone1_mL_per_min_per_g=clint_z1_baseline,
            clint_zone3_mL_per_min_per_g=clint_z3,
        )
        results.append(result)

    # Sort by zone3_burden_ratio descending (highest induction first)
    results.sort(key=lambda r: r.zone3_burden_ratio, reverse=True)
    return results
