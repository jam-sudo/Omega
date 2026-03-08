"""Phase 444 — Metabolite PK prediction from parent drug properties.

Predicts PK parameters of drug metabolites based on transformation type
(hydroxylation, glucuronidation, sulfation, N-demethylation, O-demethylation,
reduction, hydrolysis).
"""

from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Transformation database
# ---------------------------------------------------------------------------

TRANSFORMATIONS = {
    "hydroxylation": {
        "mw_delta": 16,
        "logp_delta": -0.5,
        "fu_factor": 1.3,
        "cl_factor": 1.5,
        "activity": "active",
    },
    "glucuronidation": {
        "mw_delta": 176,
        "logp_delta": -3.0,
        "fu_factor": 2.0,
        "cl_factor": 3.0,
        "activity": "inactive",
    },
    "sulfation": {
        "mw_delta": 80,
        "logp_delta": -2.5,
        "fu_factor": 1.8,
        "cl_factor": 2.5,
        "activity": "inactive",
    },
    "N_demethylation": {
        "mw_delta": -14,
        "logp_delta": -0.3,
        "fu_factor": 1.1,
        "cl_factor": 0.8,
        "activity": "active",
    },
    "O_demethylation": {
        "mw_delta": -14,
        "logp_delta": -0.5,
        "fu_factor": 1.2,
        "cl_factor": 0.9,
        "activity": "active",
    },
    "reduction": {
        "mw_delta": 2,
        "logp_delta": 0.1,
        "fu_factor": 1.0,
        "cl_factor": 1.2,
        "activity": "active",
    },
    "hydrolysis": {
        "mw_delta": 18,
        "logp_delta": -1.0,
        "fu_factor": 1.5,
        "cl_factor": 2.0,
        "activity": "inactive",
    },
}


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MetabolitePKResult:
    """Predicted PK parameters for a drug metabolite."""

    drug_name: str
    metabolite_name: str
    metabolite_type: str
    parent_mw_Da: float
    metabolite_mw_Da: float
    parent_logp: float
    metabolite_logp: float
    parent_fu_plasma: float
    metabolite_fu_plasma: float
    parent_cl_L_per_h: float
    metabolite_cl_L_per_h: float
    metabolite_vd_L: float
    metabolite_t_half_h: float
    auc_ratio_metabolite_to_parent: float
    activity_prediction: str
    excretion_route: str
    notes: str


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _clamp(value: float, lo: float, hi: float) -> float:
    """Clamp value to [lo, hi]."""
    if value < lo:
        return lo
    if value > hi:
        return hi
    return value


def _excretion_route(metabolite_mw: float, metabolite_logp: float) -> str:
    """Determine primary excretion route."""
    if metabolite_mw > 500:
        return "biliary"
    if metabolite_logp < 0:
        return "renal"
    return "both"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def predict_metabolite_pk(
    drug_name: str,
    metabolite_name: str,
    metabolite_type: str,
    parent_mw_Da: float,
    parent_logp: float,
    parent_fu_plasma: float,
    parent_cl_L_per_h: float,
    parent_vd_L: float,
    fm: float = 0.5,
) -> MetabolitePKResult:
    """Predict metabolite PK from parent drug properties and metabolite type.

    Parameters
    ----------
    drug_name : str
        Parent drug name.
    metabolite_name : str
        Name of the metabolite.
    metabolite_type : str
        Transformation type. Must be one of the keys in TRANSFORMATIONS.
    parent_mw_Da : float
        Parent molecular weight (Da).
    parent_logp : float
        Parent logP.
    parent_fu_plasma : float
        Parent fraction unbound in plasma (0–1].
    parent_cl_L_per_h : float
        Parent total clearance (L/h).
    parent_vd_L : float
        Parent volume of distribution (L).
    fm : float
        Fraction of parent dose converted to this metabolite (0–1].

    Returns
    -------
    MetabolitePKResult
    """
    # Validation
    if metabolite_type not in TRANSFORMATIONS:
        raise ValueError(
            f"Unknown metabolite_type '{metabolite_type}'. "
            f"Valid options: {list(TRANSFORMATIONS.keys())}"
        )
    if parent_mw_Da <= 0:
        raise ValueError(f"parent_mw_Da must be > 0, got {parent_mw_Da}")
    if parent_cl_L_per_h <= 0:
        raise ValueError(f"parent_cl_L_per_h must be > 0, got {parent_cl_L_per_h}")
    if parent_vd_L <= 0:
        raise ValueError(f"parent_vd_L must be > 0, got {parent_vd_L}")
    if not (0 < fm <= 1):
        raise ValueError(f"fm must be in (0, 1], got {fm}")
    if not (0 < parent_fu_plasma <= 1):
        raise ValueError(f"parent_fu_plasma must be in (0, 1], got {parent_fu_plasma}")

    t = TRANSFORMATIONS[metabolite_type]
    mw_delta = t["mw_delta"]
    logp_delta = t["logp_delta"]
    fu_factor = t["fu_factor"]
    cl_factor = t["cl_factor"]
    activity = t["activity"]

    # Derived metabolite properties
    metabolite_mw = parent_mw_Da + mw_delta
    metabolite_logp = _clamp(parent_logp + logp_delta, -5.0, 10.0)
    metabolite_fu = min(1.0, parent_fu_plasma * fu_factor)
    metabolite_cl = parent_cl_L_per_h * cl_factor

    # Volume of distribution: scale by fu ratio and MW-dependent factor
    fu_ratio = metabolite_fu / parent_fu_plasma
    mw_correction = 1.0 + 0.1 * mw_delta / parent_mw_Da
    metabolite_vd = parent_vd_L * fu_ratio * mw_correction
    if metabolite_vd <= 0:
        metabolite_vd = parent_vd_L * fu_ratio * 0.1  # fallback

    metabolite_t_half = 0.693 * metabolite_vd / metabolite_cl

    # AUC ratio (steady-state M/P): fm * CL_parent / CL_metabolite
    auc_ratio = (fm * parent_cl_L_per_h) / metabolite_cl

    excretion = _excretion_route(metabolite_mw, metabolite_logp)

    notes = (
        f"{metabolite_name} is a {metabolite_type} metabolite of {drug_name}. "
        f"Activity: {activity}. "
        f"Primary excretion: {excretion}. "
        f"MW: {metabolite_mw:.1f} Da, logP: {metabolite_logp:.2f}, "
        f"t1/2: {metabolite_t_half:.2f} h, AUC ratio M/P: {auc_ratio:.3f}."
    )

    return MetabolitePKResult(
        drug_name=drug_name,
        metabolite_name=metabolite_name,
        metabolite_type=metabolite_type,
        parent_mw_Da=parent_mw_Da,
        metabolite_mw_Da=metabolite_mw,
        parent_logp=parent_logp,
        metabolite_logp=metabolite_logp,
        parent_fu_plasma=parent_fu_plasma,
        metabolite_fu_plasma=metabolite_fu,
        parent_cl_L_per_h=parent_cl_L_per_h,
        metabolite_cl_L_per_h=metabolite_cl,
        metabolite_vd_L=metabolite_vd,
        metabolite_t_half_h=metabolite_t_half,
        auc_ratio_metabolite_to_parent=auc_ratio,
        activity_prediction=activity,
        excretion_route=excretion,
        notes=notes,
    )


def predict_metabolite_panel(
    drug_name: str,
    parent_mw_Da: float,
    parent_logp: float,
    parent_fu_plasma: float,
    parent_cl_L_per_h: float,
    parent_vd_L: float,
) -> list[MetabolitePKResult]:
    """Predict PK for all 7 metabolite transformation types.

    Returns list sorted by auc_ratio_metabolite_to_parent descending.
    """
    results = []
    for met_type in TRANSFORMATIONS:
        metabolite_name = f"{drug_name}_{met_type}_metabolite"
        result = predict_metabolite_pk(
            drug_name=drug_name,
            metabolite_name=metabolite_name,
            metabolite_type=met_type,
            parent_mw_Da=parent_mw_Da,
            parent_logp=parent_logp,
            parent_fu_plasma=parent_fu_plasma,
            parent_cl_L_per_h=parent_cl_L_per_h,
            parent_vd_L=parent_vd_L,
            fm=0.5,
        )
        results.append(result)
    results.sort(key=lambda r: r.auc_ratio_metabolite_to_parent, reverse=True)
    return results
