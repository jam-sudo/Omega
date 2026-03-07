"""Active metabolite formation and contribution prediction.

Includes Phase 692 (MetaboliteFormationResult / predict_metabolite_formation) and
Phase 778 (ActiveMetaboliteResult / predict_active_metabolite / predict_metabolite_cascade).
"""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = [
    # Phase 692 API
    "MetaboliteFormationResult",
    "predict_metabolite_formation",
    "screen_metabolites",
    "estimate_metabolite_clearance",
    # Phase 778 API
    "ActiveMetaboliteResult",
    "predict_active_metabolite",
    "predict_metabolite_cascade",
]


@dataclass
class MetaboliteFormationResult:
    """Result of active metabolite formation prediction."""

    compound_name: str
    fm_metabolite: float
    potency_ratio: float
    dose_mg: float
    vd_parent_L: float
    times_h: list
    c_parent_mg_L: list
    c_metabolite_mg_L: list
    auc_parent: float
    auc_metabolite: float
    metabolite_auc_ratio: float
    weighted_auc_ratio: float
    cmax_parent: float
    cmax_metabolite: float
    tmax_metabolite_h: float
    metabolite_accumulation: float
    safety_flag: bool
    notes: str = field(default="")


def predict_metabolite_formation(
    compound_name: str,
    mw_parent: float,
    logP_parent: float,
    cl_parent_L_per_h: float = 5.0,
    fm_metabolite: float = 0.3,
    cl_metabolite_L_per_h: float = 2.0,
    vd_metabolite_L: float = 30.0,
    potency_ratio: float = 0.5,
    t_end_h: float = 24.0,
    dt_h: float = 0.05,
    dose_mg: float = 100.0,
    vd_parent_L: float = 50.0,
) -> MetaboliteFormationResult:
    """Predict active metabolite formation using a 2-compartment parent-metabolite model.

    Uses Forward Euler integration with an IV bolus initial condition.

    Parameters
    ----------
    compound_name:
        Name of the parent compound.
    mw_parent:
        Molecular weight of the parent compound (g/mol).
    logP_parent:
        LogP of the parent compound.
    cl_parent_L_per_h:
        Total clearance of parent in L/h.
    fm_metabolite:
        Fraction of parent metabolized to the active metabolite (0-1).
    cl_metabolite_L_per_h:
        Clearance of metabolite in L/h.
    vd_metabolite_L:
        Volume of distribution of metabolite in L.
    potency_ratio:
        Potency of metabolite relative to parent (>1 = more potent).
    t_end_h:
        Simulation end time in hours.
    dt_h:
        Integration time step in hours.
    dose_mg:
        IV bolus dose in mg (default 100).
    vd_parent_L:
        Volume of distribution of parent in L (default 50).

    Returns
    -------
    MetaboliteFormationResult
    """
    # Validate inputs
    if not (0.0 <= fm_metabolite <= 1.0):
        raise ValueError(f"fm_metabolite must be in [0, 1], got {fm_metabolite}")
    if cl_parent_L_per_h <= 0:
        raise ValueError("cl_parent_L_per_h must be > 0")
    if cl_metabolite_L_per_h <= 0:
        raise ValueError("cl_metabolite_L_per_h must be > 0")
    if vd_metabolite_L <= 0:
        raise ValueError("vd_metabolite_L must be > 0")
    if vd_parent_L <= 0:
        raise ValueError("vd_parent_L must be > 0")
    if potency_ratio < 0:
        raise ValueError("potency_ratio must be >= 0")

    ke_parent = cl_parent_L_per_h / vd_parent_L
    ke_metabolite = cl_metabolite_L_per_h / vd_metabolite_L

    # Initial conditions: IV bolus
    C_parent = dose_mg / vd_parent_L
    C_metabolite = 0.0

    n_steps = int(round(t_end_h / dt_h)) + 1
    times_out = []
    c_parent_out = []
    c_metabolite_out = []

    t = 0.0
    for _ in range(n_steps):
        if t > t_end_h + 1e-9:
            break
        times_out.append(t)
        c_parent_out.append(C_parent)
        c_metabolite_out.append(C_metabolite)

        # Formation rate of metabolite (mg/h)
        formation_rate = fm_metabolite * ke_parent * C_parent * vd_parent_L

        dC_parent = -ke_parent * C_parent
        dC_metabolite = formation_rate / vd_metabolite_L - ke_metabolite * C_metabolite

        C_parent += dC_parent * dt_h
        C_metabolite += dC_metabolite * dt_h

        if C_parent < 0.0:
            C_parent = 0.0
        if C_metabolite < 0.0:
            C_metabolite = 0.0

        t += dt_h

    # AUC by trapezoidal rule
    auc_parent = _trapz(times_out, c_parent_out)
    auc_metabolite = _trapz(times_out, c_metabolite_out)

    metabolite_auc_ratio = auc_metabolite / auc_parent if auc_parent > 0 else 0.0
    weighted_auc_ratio = metabolite_auc_ratio * potency_ratio

    cmax_parent = max(c_parent_out) if c_parent_out else 0.0
    cmax_metabolite = max(c_metabolite_out) if c_metabolite_out else 0.0
    tmax_metabolite_h = (
        times_out[c_metabolite_out.index(cmax_metabolite)] if c_metabolite_out else 0.0
    )

    metabolite_accumulation = cmax_metabolite / cmax_parent if cmax_parent > 0 else 0.0

    safety_flag = weighted_auc_ratio > 1.0

    notes = (
        f"ke_parent={ke_parent:.4f}/h; ke_metabolite={ke_metabolite:.4f}/h; "
        f"mw_parent={mw_parent:.1f}; logP={logP_parent:.2f}"
    )

    return MetaboliteFormationResult(
        compound_name=compound_name,
        fm_metabolite=fm_metabolite,
        potency_ratio=potency_ratio,
        dose_mg=dose_mg,
        vd_parent_L=vd_parent_L,
        times_h=times_out,
        c_parent_mg_L=c_parent_out,
        c_metabolite_mg_L=c_metabolite_out,
        auc_parent=auc_parent,
        auc_metabolite=auc_metabolite,
        metabolite_auc_ratio=metabolite_auc_ratio,
        weighted_auc_ratio=weighted_auc_ratio,
        cmax_parent=cmax_parent,
        cmax_metabolite=cmax_metabolite,
        tmax_metabolite_h=tmax_metabolite_h,
        metabolite_accumulation=metabolite_accumulation,
        safety_flag=safety_flag,
        notes=notes,
    )


def _trapz(xs: list, ys: list) -> float:
    """Compute trapezoidal AUC."""
    if len(xs) < 2:
        return 0.0
    auc = 0.0
    for i in range(1, len(xs)):
        auc += 0.5 * (ys[i] + ys[i - 1]) * (xs[i] - xs[i - 1])
    return auc


def screen_metabolites(compounds: list) -> list:
    """Screen a list of compound dicts, returning results sorted by weighted_auc_ratio descending.

    Each dict should contain keyword arguments for predict_metabolite_formation.
    Required keys: 'compound_name', 'mw_parent', 'logP_parent'.
    Optional keys mirror predict_metabolite_formation parameters.

    Parameters
    ----------
    compounds:
        List of compound parameter dicts.

    Returns
    -------
    List of MetaboliteFormationResult sorted by weighted_auc_ratio descending.
    """
    if not compounds:
        return []

    results = []
    for comp in compounds:
        result = predict_metabolite_formation(**comp)
        results.append(result)

    results.sort(key=lambda r: r.weighted_auc_ratio, reverse=True)
    return results


def estimate_metabolite_clearance(
    parent_cl: float,
    fm: float,
    mw_metabolite: float,
) -> float:
    """Estimate metabolite clearance from parent CL, formation fraction, and metabolite MW.

    Uses the relationship:
    CL_metabolite ≈ parent_cl × (1/fm) × (300/mw_metabolite)^0.25

    Heavier metabolites tend to clear more slowly.

    Parameters
    ----------
    parent_cl:
        Parent clearance in L/h.
    fm:
        Fraction of parent dose forming the metabolite.
    mw_metabolite:
        Molecular weight of the metabolite in g/mol.

    Returns
    -------
    float
        Estimated metabolite clearance in L/h.
    """
    if fm <= 0:
        fm = 1e-9  # avoid division by zero
    cl_metabolite = parent_cl * (1.0 / fm) * ((300.0 / mw_metabolite) ** 0.25)
    return float(cl_metabolite)


# ─────────────────────────────────────────────────────────────────────────────
# Phase 778: Active metabolite PK consequence assessment
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ActiveMetaboliteResult:
    """Result of active metabolite prediction and PK consequence assessment."""

    parent_name: str
    metabolite_name: str
    pathway: str  # e.g., "CYP3A4 N-dealkylation"
    formation_fraction: float  # fraction of parent → metabolite
    metabolite_cl_L_per_h: float
    metabolite_vd_L: float
    metabolite_t_half_h: float
    auc_ratio_met_parent: float  # metabolite AUC / parent AUC at steady state
    pharmacological_activity: str  # "active" / "inactive" / "toxic"
    mist_relevant: bool  # >10% of parent AUC → FDA MIST guidance
    notes: str


def predict_active_metabolite(
    parent_name: str,
    parent_dose_mg: float,
    parent_cl_L_per_h: float,
    parent_vd_L: float,
    metabolite_name: str = "M1",
    pathway: str = "CYP3A4",
    fm: float = 0.3,
    met_cl_L_per_h: float = 10.0,
    met_vd_L: float = 30.0,
    pharmacological_activity: str = "active",
) -> ActiveMetaboliteResult:
    """Predict active metabolite PK and safety relevance.

    Computes:
    - metabolite_t_half = 0.693 * met_vd_L / met_cl_L_per_h
    - auc_ratio = fm * parent_CL / met_CL  (steady-state approximation)
    - mist_relevant = auc_ratio > 0.1

    Parameters
    ----------
    parent_name:
        Name of the parent drug.
    parent_dose_mg:
        Dose of parent drug in mg.
    parent_cl_L_per_h:
        Total clearance of parent in L/h.
    parent_vd_L:
        Volume of distribution of parent in L.
    metabolite_name:
        Identifier for the metabolite (default "M1").
    pathway:
        Metabolic pathway label (e.g., "CYP3A4").
    fm:
        Fraction of parent metabolized via this pathway (0–1).
    met_cl_L_per_h:
        Clearance of metabolite in L/h (must be > 0).
    met_vd_L:
        Volume of distribution of metabolite in L (must be > 0).
    pharmacological_activity:
        Activity classification: "active", "inactive", or "toxic".

    Returns
    -------
    ActiveMetaboliteResult
    """
    valid_activities = {"active", "inactive", "toxic"}
    if pharmacological_activity not in valid_activities:
        raise ValueError(
            f"pharmacological_activity must be one of {valid_activities}, "
            f"got {pharmacological_activity!r}"
        )
    if not (0.0 <= fm <= 1.0):
        raise ValueError(f"fm must be in [0, 1], got {fm}")
    if met_cl_L_per_h <= 0:
        raise ValueError(f"met_cl_L_per_h must be > 0, got {met_cl_L_per_h}")
    if met_vd_L <= 0:
        raise ValueError(f"met_vd_L must be > 0, got {met_vd_L}")
    if parent_cl_L_per_h <= 0:
        raise ValueError(f"parent_cl_L_per_h must be > 0, got {parent_cl_L_per_h}")
    if parent_vd_L <= 0:
        raise ValueError(f"parent_vd_L must be > 0, got {parent_vd_L}")

    metabolite_t_half_h = 0.693 * met_vd_L / met_cl_L_per_h

    # Steady-state AUC ratio: AUC_met/AUC_parent = fm * CL_parent / CL_met
    auc_ratio_met_parent = fm * parent_cl_L_per_h / met_cl_L_per_h

    mist_relevant = auc_ratio_met_parent > 0.1

    notes = (
        f"t_half={metabolite_t_half_h:.2f}h; "
        f"AUC_ratio={auc_ratio_met_parent:.3f}; "
        f"mist={'yes' if mist_relevant else 'no'}; "
        f"dose={parent_dose_mg:.1f}mg"
    )

    return ActiveMetaboliteResult(
        parent_name=parent_name,
        metabolite_name=metabolite_name,
        pathway=pathway,
        formation_fraction=fm,
        metabolite_cl_L_per_h=met_cl_L_per_h,
        metabolite_vd_L=met_vd_L,
        metabolite_t_half_h=metabolite_t_half_h,
        auc_ratio_met_parent=auc_ratio_met_parent,
        pharmacological_activity=pharmacological_activity,
        mist_relevant=mist_relevant,
        notes=notes,
    )


def predict_metabolite_cascade(
    parent_name: str,
    parent_dose_mg: float,
    parent_cl_L_per_h: float,
    metabolites: list[dict],
) -> list[ActiveMetaboliteResult]:
    """Predict PK consequences for a cascade of metabolites.

    Parameters
    ----------
    parent_name:
        Name of the parent drug.
    parent_dose_mg:
        Dose of parent drug in mg.
    parent_cl_L_per_h:
        Total clearance of parent in L/h.
    metabolites:
        List of dicts, each with keys:
        - metabolite_name: str
        - pathway: str
        - fm: float
        - met_cl_L_per_h: float
        - met_vd_L: float
        - pharmacological_activity: str  (optional, default "active")

    Returns
    -------
    List of ActiveMetaboliteResult, one per metabolite entry.
    """
    results = []
    for met in metabolites:
        result = predict_active_metabolite(
            parent_name=parent_name,
            parent_dose_mg=parent_dose_mg,
            parent_cl_L_per_h=parent_cl_L_per_h,
            parent_vd_L=met.get("parent_vd_L", 50.0),
            metabolite_name=met.get("metabolite_name", "M1"),
            pathway=met.get("pathway", "CYP3A4"),
            fm=met.get("fm", 0.3),
            met_cl_L_per_h=met.get("met_cl_L_per_h", 10.0),
            met_vd_L=met.get("met_vd_L", 30.0),
            pharmacological_activity=met.get("pharmacological_activity", "active"),
        )
        results.append(result)
    return results
