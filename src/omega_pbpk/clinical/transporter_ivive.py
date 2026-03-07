"""Drug-transporter interaction IVIVE.

Scale in vitro transporter inhibition (IC50) to predict in vivo DDI risk
following FDA 2020 transporter DDI guidance.

Supported transporters: P-gp, BCRP, OATP1B1, OATP1B3, OAT1, OAT3, OCT2
"""

from __future__ import annotations

from dataclasses import dataclass

VALID_TRANSPORTERS = frozenset({"P-gp", "BCRP", "OATP1B1", "OATP1B3", "OAT1", "OAT3", "OCT2"})

_INTESTINAL_TRANSPORTERS = frozenset({"P-gp", "BCRP"})
_HEPATIC_TRANSPORTERS = frozenset({"OATP1B1", "OATP1B3"})
_RENAL_TRANSPORTERS = frozenset({"OAT1", "OAT3", "OCT2"})


@dataclass(frozen=True)
class TransporterIVIVEResult:
    perpetrator_name: str
    victim_drug: str
    transporter: str

    ic50_vitro_uM: float  # In vitro IC50 (µM)
    c_inlet_uM: float  # Intestinal or hepatic inlet concentration (µM)
    c_systemic_uM: float  # Systemic plasma concentration (µM)

    # Static IVIVE predictions
    r_value: float  # R = 1 + C_inlet / IC50
    r2_value: float  # R2 for gut: 1 + (Dose/250mL) / IC50 (in µM)
    predicted_auc_ratio: float  # Predicted victim AUC ratio (capped at 5)

    # Clinical relevance
    ddi_category: str  # "no_ddi", "weak", "moderate", "strong"
    clinical_significance: str
    requires_clinical_study: bool  # predicted AUC ratio > 1.25

    notes: str


def _ddi_category(r: float) -> str:
    if r < 1.1:
        return "no_ddi"
    if r < 2.0:
        return "weak"
    if r <= 5.0:
        return "moderate"
    return "strong"


def _clinical_significance(category: str) -> str:
    return {
        "no_ddi": "No clinically relevant transporter-mediated DDI expected.",
        "weak": "Weak transporter inhibition; monitor victim drug exposure.",
        "moderate": "Moderate DDI risk; dose adjustment of victim drug may be required.",
        "strong": "Strong transporter inhibition; clinical DDI study required.",
    }[category]


def predict_transporter_ddi(
    perpetrator_name: str,
    victim_drug: str,
    transporter: str,
    ic50_vitro_uM: float,
    dose_mg_perpetrator: float,
    mw_perpetrator: float,
    f_oral: float = 0.9,
    cl_L_per_h: float = 10.0,
    vd_L: float = 50.0,
    fu_plasma: float = 0.5,
    bioavailability_vitro_correction: float = 1.0,
) -> TransporterIVIVEResult:
    """Predict transporter-mediated DDI risk using static FDA IVIVE approach.

    Parameters
    ----------
    perpetrator_name : str
        Name of the inhibiting drug (perpetrator).
    victim_drug : str
        Name of the substrate drug (victim).
    transporter : str
        Transporter name; must be one of VALID_TRANSPORTERS.
    ic50_vitro_uM : float
        In vitro inhibitory concentration 50% (µM).
    dose_mg_perpetrator : float
        Dose of the perpetrator drug (mg).
    mw_perpetrator : float
        Molecular weight of perpetrator (g/mol).
    f_oral : float
        Oral bioavailability of perpetrator (0-1).
    cl_L_per_h : float
        Systemic clearance of perpetrator (L/h).
    vd_L : float
        Volume of distribution of perpetrator (L).
    fu_plasma : float
        Unbound fraction in plasma (0-1].
    bioavailability_vitro_correction : float
        Correction factor for in vitro system differences (default 1.0).
    """
    # Validation
    if ic50_vitro_uM <= 0:
        raise ValueError("ic50_vitro_uM must be > 0")
    if dose_mg_perpetrator <= 0:
        raise ValueError("dose_mg_perpetrator must be > 0")
    if mw_perpetrator <= 0:
        raise ValueError("mw_perpetrator must be > 0")
    if cl_L_per_h <= 0:
        raise ValueError("cl_L_per_h must be > 0")
    if vd_L <= 0:
        raise ValueError("vd_L must be > 0")
    if not (0 < fu_plasma <= 1):
        raise ValueError("fu_plasma must be in (0, 1]")
    if transporter not in VALID_TRANSPORTERS:
        raise ValueError(
            f"transporter must be one of {sorted(VALID_TRANSPORTERS)}, got {transporter!r}"
        )

    # Systemic unbound Cmax approximation: dose / Vd * fu_plasma (µM)
    c_systemic_mgL = dose_mg_perpetrator / vd_L * fu_plasma
    c_systemic_uM = c_systemic_mgL * 1000.0 / mw_perpetrator

    r_value = 1.0
    r2_value = 1.0
    c_inlet_uM = 0.0

    if transporter in _INTESTINAL_TRANSPORTERS:
        # FDA gut inhibition: R2 = 1 + Cgut / IC50
        # Cgut (µM) = (dose_mg / V_lumen_mL) * 1000 / mw
        # V_lumen = 250 mL per FDA guidance
        v_lumen_mL = 250.0
        c_gut_mgL = dose_mg_perpetrator / (v_lumen_mL * 1e-3)  # mg/L
        c_gut_uM = c_gut_mgL * 1000.0 / mw_perpetrator / bioavailability_vitro_correction
        c_inlet_uM = c_gut_uM
        r2_value = 1.0 + c_gut_uM / ic50_vitro_uM
        # For intestinal transporters, use R2 as primary metric
        r_value = r2_value

    elif transporter in _HEPATIC_TRANSPORTERS:
        # Portal vein: FDA R = 1 + C_portal,unbound / IC50
        # C_portal,unbound = (dose_mg / vd_L) * fu_plasma * 1.5 (portal enrichment)
        c_portal_mgL = (dose_mg_perpetrator / vd_L) * fu_plasma * 1.5
        c_portal_uM = c_portal_mgL * 1000.0 / mw_perpetrator
        c_inlet_uM = c_portal_uM
        r_value = 1.0 + c_portal_uM / ic50_vitro_uM
        r2_value = r_value  # same for hepatic

    elif transporter in _RENAL_TRANSPORTERS:
        # Renal: use systemic unbound concentration
        c_inlet_uM = c_systemic_uM
        r_value = 1.0 + c_systemic_uM / ic50_vitro_uM
        r2_value = r_value

    # Predicted AUC ratio (simplified: R as proxy, capped at 5)
    predicted_auc_ratio = min(r_value, 5.0)

    category = _ddi_category(r_value)
    significance = _clinical_significance(category)
    requires_clinical_study = predicted_auc_ratio > 1.25

    notes = (
        f"FDA static IVIVE; transporter={transporter}; "
        f"IC50={ic50_vitro_uM:.2f} µM; dose={dose_mg_perpetrator} mg"
    )

    return TransporterIVIVEResult(
        perpetrator_name=perpetrator_name,
        victim_drug=victim_drug,
        transporter=transporter,
        ic50_vitro_uM=ic50_vitro_uM,
        c_inlet_uM=c_inlet_uM,
        c_systemic_uM=c_systemic_uM,
        r_value=r_value,
        r2_value=r2_value,
        predicted_auc_ratio=predicted_auc_ratio,
        ddi_category=category,
        clinical_significance=significance,
        requires_clinical_study=requires_clinical_study,
        notes=notes,
    )


def screen_transporter_inhibitors(
    compounds: list[dict],
    victim_drug: str,
    transporter: str,
) -> list[TransporterIVIVEResult]:
    """Screen a list of compounds as transporter inhibitors.

    Parameters
    ----------
    compounds : list of dict
        Each entry: {"name": str, "dose_mg": float, "mw": float, "ic50_uM": float}
        Optional keys: fu_plasma, cl_L_per_h, vd_L, f_oral.
    victim_drug : str
        Name of the victim drug.
    transporter : str
        Transporter to screen against.

    Returns
    -------
    list[TransporterIVIVEResult]
        Sorted by r_value descending (strongest inhibitors first).
    """
    results: list[TransporterIVIVEResult] = []
    for compound in compounds:
        result = predict_transporter_ddi(
            perpetrator_name=compound["name"],
            victim_drug=victim_drug,
            transporter=transporter,
            ic50_vitro_uM=compound["ic50_uM"],
            dose_mg_perpetrator=compound["dose_mg"],
            mw_perpetrator=compound["mw"],
            fu_plasma=compound.get("fu_plasma", 0.5),
            cl_L_per_h=compound.get("cl_L_per_h", 10.0),
            vd_L=compound.get("vd_L", 50.0),
            f_oral=compound.get("f_oral", 0.9),
        )
        results.append(result)

    results.sort(key=lambda r: r.r_value, reverse=True)
    return results


def summarize_transporter_risk(results: list[TransporterIVIVEResult]) -> dict:
    """Summarize transporter DDI risk across multiple predictions.

    Parameters
    ----------
    results : list[TransporterIVIVEResult]
        List of IVIVE results to summarize.

    Returns
    -------
    dict
        Keys: "high_risk_count" (int), "transporter_summary" (dict transporter→max R).
    """
    high_risk_count = sum(1 for r in results if r.ddi_category in ("moderate", "strong"))

    transporter_summary: dict[str, float] = {}
    for r in results:
        current_max = transporter_summary.get(r.transporter, 0.0)
        transporter_summary[r.transporter] = max(current_max, r.r_value)

    return {
        "high_risk_count": high_risk_count,
        "transporter_summary": transporter_summary,
    }
