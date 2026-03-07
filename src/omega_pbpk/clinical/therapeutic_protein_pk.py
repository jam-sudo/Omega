"""Phase 246: PK model for therapeutic proteins and biologics."""

import math
from dataclasses import dataclass

# Protein type lookup: name -> (mw_kDa, cl_mL_per_h_per_kg, vd_mL_per_kg, t_half_days, fcrn_recycling)
_PROTEIN_TABLE = {
    "mab_igg1": (150.0, 3.5, 70.0, 21.0, True),
    "mab_igg4": (150.0, 2.5, 60.0, 28.0, True),
    "fab_fragment": (50.0, 20.0, 200.0, 1.5, False),
    "nanobody": (15.0, 100.0, 400.0, 0.5, False),
    "scfv": (27.0, 50.0, 300.0, 1.0, False),
}

_VALID_ROUTES = {"iv_bolus", "sc"}

# SC absorption rate constant (h^-1) typical for mAbs
_KA_SC = 0.015


@dataclass(frozen=True)
class TherapeuticProteinPKResult:
    drug_name: str
    protein_type: str
    dose_mg: float
    weight_kg: float
    route: str
    mw_kDa: float
    t_half_h: float
    t_half_days_result: float
    cl_L_per_h: float
    vd_L: float
    cmax_mg_L: float
    auc_mg_h_per_L: float
    tmax_h: float
    bioavailability_SC: float
    dosing_frequency_recommendation: str
    notes: str


def simulate_therapeutic_protein_pk(
    drug_name: str,
    dose_mg: float,
    protein_type: str,
    weight_kg: float = 70.0,
    route: str = "iv_bolus",
) -> TherapeuticProteinPKResult:
    """Simulate PK for a therapeutic protein."""
    if dose_mg <= 0:
        raise ValueError(f"dose_mg must be > 0, got {dose_mg}")
    if weight_kg <= 0:
        raise ValueError(f"weight_kg must be > 0, got {weight_kg}")
    if protein_type not in _PROTEIN_TABLE:
        raise ValueError(
            f"Unknown protein_type '{protein_type}'. Valid: {list(_PROTEIN_TABLE.keys())}"
        )
    if route not in _VALID_ROUTES:
        raise ValueError(f"Unknown route '{route}'. Valid: {_VALID_ROUTES}")

    mw_kDa, cl_per_kg, vd_per_kg, _t_half_days_nominal, fcrn_recycling = _PROTEIN_TABLE[
        protein_type
    ]

    # Scale per-kg values to absolute (convert mL to L)
    cl_L_per_h = cl_per_kg * weight_kg / 1000.0
    vd_L = vd_per_kg * weight_kg / 1000.0

    t_half_h = 0.693 * vd_L / cl_L_per_h
    t_half_days_result = t_half_h / 24.0

    if route == "iv_bolus":
        bioavailability_SC = 1.0
        ke = cl_L_per_h / vd_L  # elimination rate constant (h^-1)
        cmax_mg_L = dose_mg / vd_L
        tmax_h = 0.0
        auc_mg_h_per_L = dose_mg * bioavailability_SC / cl_L_per_h
    else:
        # SC route
        bioavailability_SC = 0.65 if fcrn_recycling else 0.5
        ke = cl_L_per_h / vd_L
        ka = _KA_SC
        # tmax = ln(ka/ke) / (ka - ke)
        if ka != ke:
            tmax_h = math.log(ka / ke) / (ka - ke)
        else:
            tmax_h = 1.0 / ke  # degenerate case
        # Cmax for 1-cpt SC: F * dose / Vd * ka/(ka-ke) * [exp(-ke*tmax) - exp(-ka*tmax)]
        # Alternatively use the standard formula:
        # Cmax = F * dose * ka / (Vd * (ka - ke)) * (exp(-ke*tmax) - exp(-ka*tmax))
        if ka != ke:
            cmax_mg_L = (
                bioavailability_SC
                * dose_mg
                * ka
                / (vd_L * (ka - ke))
                * (math.exp(-ke * tmax_h) - math.exp(-ka * tmax_h))
            )
        else:
            cmax_mg_L = bioavailability_SC * dose_mg / vd_L * math.exp(-ke * tmax_h)
        auc_mg_h_per_L = dose_mg * bioavailability_SC / cl_L_per_h

    # Dosing frequency: every int(t_half_days * 3) days, minimum 1
    freq_days = max(1, int(round(t_half_days_result * 3.0)))
    dosing_frequency_recommendation = f"Every {freq_days} days"

    notes = (
        f"Protein type: {protein_type}, MW={mw_kDa} kDa, route={route}, "
        f"FcRn recycling={fcrn_recycling}, t1/2={t_half_h:.1f} h"
    )

    return TherapeuticProteinPKResult(
        drug_name=drug_name,
        protein_type=protein_type,
        dose_mg=dose_mg,
        weight_kg=weight_kg,
        route=route,
        mw_kDa=mw_kDa,
        t_half_h=t_half_h,
        t_half_days_result=t_half_days_result,
        cl_L_per_h=cl_L_per_h,
        vd_L=vd_L,
        cmax_mg_L=cmax_mg_L,
        auc_mg_h_per_L=auc_mg_h_per_L,
        tmax_h=tmax_h,
        bioavailability_SC=bioavailability_SC,
        dosing_frequency_recommendation=dosing_frequency_recommendation,
        notes=notes,
    )


def compare_protein_types(
    drug_name: str,
    dose_mg: float,
    weight_kg: float = 70.0,
) -> list[TherapeuticProteinPKResult]:
    """Compare all 5 protein types via IV bolus, sorted by t_half_h descending."""
    results = []
    for protein_type in _PROTEIN_TABLE:
        result = simulate_therapeutic_protein_pk(
            drug_name=drug_name,
            dose_mg=dose_mg,
            protein_type=protein_type,
            weight_kg=weight_kg,
            route="iv_bolus",
        )
        results.append(result)
    results.sort(key=lambda r: r.t_half_h, reverse=True)
    return results
