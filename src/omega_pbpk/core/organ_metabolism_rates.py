"""Phase 553 — Organ-Specific Drug Metabolism Rates.

Models relative metabolic rates across organs (liver, intestine, kidney, lung)
using the well-stirred model for organ clearance.
"""

from dataclasses import dataclass

ORGAN_PARAMS = {
    "liver": {"weight_g": 1500.0, "blood_flow_L_per_h": 90.0},
    "intestine": {"weight_g": 1000.0, "blood_flow_L_per_h": 36.0},
    "kidney": {"weight_g": 300.0, "blood_flow_L_per_h": 72.0},
    "lung": {"weight_g": 500.0, "blood_flow_L_per_h": 400.0},
}


@dataclass(frozen=True)
class OrganMetabolismResult:
    drug_name: str
    organ: str
    cl_intrinsic_mL_per_min_per_g: float
    organ_weight_g: float
    blood_flow_L_per_h: float
    cl_organ_L_per_h: float
    extraction_ratio: float
    contribution_pct: float
    notes: str


def predict_organ_metabolism(
    drug_name: str,
    organ: str,
    cl_intrinsic_mL_per_min_per_g: float,
    fup: float = 0.1,
) -> OrganMetabolismResult:
    """Predict organ-specific metabolism using the well-stirred model."""
    if not drug_name or not isinstance(drug_name, str):
        raise ValueError("drug_name must be a non-empty string")
    organ_lower = organ.lower()
    if organ_lower not in ORGAN_PARAMS:
        raise ValueError(f"Unknown organ '{organ}'. Valid: {list(ORGAN_PARAMS.keys())}")
    if cl_intrinsic_mL_per_min_per_g < 0:
        raise ValueError("cl_intrinsic_mL_per_min_per_g must be >= 0")
    if not (0 < fup <= 1):
        raise ValueError("fup must be in (0, 1]")

    params = ORGAN_PARAMS[organ_lower]
    weight_g = params["weight_g"]
    q = params["blood_flow_L_per_h"]

    # CLint total in L/h
    cl_int_total = cl_intrinsic_mL_per_min_per_g * weight_g * 60.0 / 1000.0

    # Well-stirred model: CL_organ = Q * fu * CLint / (Q + fu * CLint)
    fu_clint = fup * cl_int_total
    cl_organ = q * fu_clint / (q + fu_clint) if (q + fu_clint) > 0 else 0.0

    # Extraction ratio
    e = cl_organ / q if q > 0 else 0.0
    e = max(0.0, min(1.0, e))

    return OrganMetabolismResult(
        drug_name=drug_name,
        organ=organ_lower,
        cl_intrinsic_mL_per_min_per_g=cl_intrinsic_mL_per_min_per_g,
        organ_weight_g=weight_g,
        blood_flow_L_per_h=q,
        cl_organ_L_per_h=cl_organ,
        extraction_ratio=e,
        contribution_pct=100.0,
        notes=f"Well-stirred model; fup={fup}",
    )


def screen_organ_metabolism(
    drug_name: str,
    cl_intrinsic_mL_per_min_per_g: float,
    fup: float = 0.1,
) -> list:
    """Screen metabolism across all 4 organs and compute contributions."""
    results = []
    for organ in ORGAN_PARAMS:
        r = predict_organ_metabolism(drug_name, organ, cl_intrinsic_mL_per_min_per_g, fup)
        results.append(r)

    total_cl = sum(r.cl_organ_L_per_h for r in results)

    updated = []
    for r in results:
        pct = (r.cl_organ_L_per_h / total_cl * 100.0) if total_cl > 0 else 0.0
        updated.append(
            OrganMetabolismResult(
                drug_name=r.drug_name,
                organ=r.organ,
                cl_intrinsic_mL_per_min_per_g=r.cl_intrinsic_mL_per_min_per_g,
                organ_weight_g=r.organ_weight_g,
                blood_flow_L_per_h=r.blood_flow_L_per_h,
                cl_organ_L_per_h=r.cl_organ_L_per_h,
                extraction_ratio=r.extraction_ratio,
                contribution_pct=pct,
                notes=r.notes,
            )
        )

    updated.sort(key=lambda r: r.cl_organ_L_per_h, reverse=True)
    return updated
