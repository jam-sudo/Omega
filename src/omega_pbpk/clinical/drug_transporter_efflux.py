"""Phase 425 — P-glycoprotein and BCRP efflux transporter impact on drug
distribution and CNS penetration.
"""

from __future__ import annotations

from dataclasses import dataclass

VALID_TRANSPORTERS = {"P-gp", "BCRP", "MRP2", "combined"}


@dataclass(frozen=True)
class TransporterEffluxResult:
    drug_name: str
    transporter: str
    substrate_class: str
    inhibitor_class: str
    efflux_ratio: float
    cns_kp_uu: float
    cns_penetration_class: str
    gut_extraction_pct: float
    f_oral_reduction_factor: float
    ddi_risk_as_inhibitor: str
    inhibition_notes: str
    notes: str


def _classify_substrate(efflux_ratio: float) -> str:
    if efflux_ratio > 10:
        return "high"
    elif efflux_ratio >= 2:
        return "moderate"
    elif efflux_ratio >= 1.5:
        return "low"
    else:
        return "non_substrate"


def _classify_inhibitor(ic50_uM: float) -> str:
    if ic50_uM < 0.1:
        return "strong"
    elif ic50_uM < 1.0:
        return "moderate"
    elif ic50_uM <= 10.0:
        return "weak"
    else:
        return "non_inhibitor"


def _classify_cns_penetration(kp_uu: float) -> str:
    if kp_uu > 0.3:
        return "high"
    elif kp_uu >= 0.1:
        return "moderate"
    elif kp_uu >= 0.01:
        return "low"
    else:
        return "minimal"


def _ddi_risk(substrate_class: str, inhibitor_class: str) -> str:
    if inhibitor_class == "strong":
        return "high"
    elif inhibitor_class == "moderate":
        if substrate_class in ("high", "moderate"):
            return "high"
        return "moderate"
    elif inhibitor_class == "weak":
        if substrate_class == "high":
            return "moderate"
        return "low"
    else:
        return "low"


def assess_transporter_efflux(
    drug_name: str,
    transporter: str,
    logp: float,
    mw_Da: float,
    in_vitro_efflux_ratio: float,
    fu_brain: float = 0.05,
    ic50_uM: float = 100.0,
    ionization_type: str = "neutral",
) -> TransporterEffluxResult:
    """Assess efflux transporter impact on a drug.

    Parameters
    ----------
    drug_name:
        Name of the compound.
    transporter:
        One of "P-gp", "BCRP", "MRP2", "combined".
    logp:
        Lipophilicity.
    mw_Da:
        Molecular weight in Da.
    in_vitro_efflux_ratio:
        Caco-2 / MDR1-MDCK B-to-A / A-to-B ratio (must be > 0).
    fu_brain:
        Unbound fraction in brain (0 < fu_brain <= 1).
    ic50_uM:
        In vitro IC50 for transporter inhibition (uM, must be > 0).
    ionization_type:
        Ionization state: "neutral", "acid", "base", "zwitterion".

    Returns
    -------
    TransporterEffluxResult
    """
    if transporter not in VALID_TRANSPORTERS:
        raise ValueError(f"transporter must be one of {VALID_TRANSPORTERS}, got {transporter!r}")
    if in_vitro_efflux_ratio <= 0:
        raise ValueError(f"in_vitro_efflux_ratio must be > 0, got {in_vitro_efflux_ratio}")
    if not (0 < fu_brain <= 1):
        raise ValueError(f"fu_brain must be in (0, 1], got {fu_brain}")
    if ic50_uM <= 0:
        raise ValueError(f"ic50_uM must be > 0, got {ic50_uM}")

    # For combined transporters the effective efflux ratio is scaled up.
    efflux_ratio = in_vitro_efflux_ratio
    if transporter == "combined":
        efflux_ratio = in_vitro_efflux_ratio * 1.2

    substrate_class = _classify_substrate(efflux_ratio)
    inhibitor_class = _classify_inhibitor(ic50_uM)

    # Kp,uu,brain — simplified active efflux model
    denominator = 1.0 + (efflux_ratio - 1.0) * 0.7
    cns_kp_uu_raw = fu_brain / denominator
    cns_kp_uu = max(0.001, min(1.0, cns_kp_uu_raw))

    cns_penetration_class = _classify_cns_penetration(cns_kp_uu)

    # Gut extraction (%)
    gut_extraction_pct = min(80.0, max(0.0, (efflux_ratio - 1.0) * 5.0))
    f_oral_reduction_factor = 1.0 - gut_extraction_pct / 100.0

    ddi_risk = _ddi_risk(substrate_class, inhibitor_class)

    # Inhibition notes
    inh_parts = [
        f"IC50 = {ic50_uM} uM ({inhibitor_class} inhibitor).",
        f"DDI risk as {transporter} inhibitor: {ddi_risk}.",
    ]
    if inhibitor_class in ("strong", "moderate"):
        inh_parts.append("Clinical DDI studies with sensitive substrates are recommended.")
    inhibition_notes = " ".join(inh_parts)

    # General notes
    note_parts = [
        f"{drug_name} is a {substrate_class} {transporter} substrate "
        f"(efflux ratio = {efflux_ratio:.2f}).",
        f"Kp,uu,brain = {cns_kp_uu:.4f} → CNS penetration: {cns_penetration_class}.",
        f"Gut extraction: {gut_extraction_pct:.1f}%.",
        f"Oral bioavailability reduction factor: {f_oral_reduction_factor:.3f}.",
    ]
    notes = " ".join(note_parts)

    return TransporterEffluxResult(
        drug_name=drug_name,
        transporter=transporter,
        substrate_class=substrate_class,
        inhibitor_class=inhibitor_class,
        efflux_ratio=efflux_ratio,
        cns_kp_uu=cns_kp_uu,
        cns_penetration_class=cns_penetration_class,
        gut_extraction_pct=gut_extraction_pct,
        f_oral_reduction_factor=f_oral_reduction_factor,
        ddi_risk_as_inhibitor=ddi_risk,
        inhibition_notes=inhibition_notes,
        notes=notes,
    )


def compare_efflux_scenarios(
    drug_name: str,
    logp: float,
    mw_Da: float,
    efflux_ratios: list,
    fu_brain: float = 0.05,
) -> list:
    """Compare multiple efflux scenarios for the same drug.

    Parameters
    ----------
    drug_name:
        Name of the compound.
    logp:
        Lipophilicity.
    mw_Da:
        Molecular weight in Da.
    efflux_ratios:
        List of (transporter, in_vitro_efflux_ratio) tuples.
    fu_brain:
        Unbound fraction in brain.

    Returns
    -------
    List of TransporterEffluxResult sorted by cns_kp_uu descending.
    """
    results = []
    for transporter, ratio in efflux_ratios:
        result = assess_transporter_efflux(
            drug_name=drug_name,
            transporter=transporter,
            logp=logp,
            mw_Da=mw_Da,
            in_vitro_efflux_ratio=ratio,
            fu_brain=fu_brain,
        )
        results.append(result)
    results.sort(key=lambda r: r.cns_kp_uu, reverse=True)
    return results
