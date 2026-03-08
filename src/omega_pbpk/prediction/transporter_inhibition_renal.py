"""Phase 1083 — Drug-Induced Transporter Inhibition (OAT/OCT).

Predicts DDI caused by inhibition of renal organic anion/cation transporters
(OAT1, OAT3, OCT2, MATE1, P_glycoprotein_renal). Inhibition reduces renal
secretion, increasing systemic exposure of victim drugs.
"""

from __future__ import annotations

from dataclasses import dataclass

# Baseline secretion clearances (L/h) from literature
_TRANSPORTER_CL_SEC: dict[str, float] = {
    "OAT1": 2.0,
    "OAT3": 1.5,
    "OCT2": 3.0,
    "MATE1": 1.0,
    "P_glycoprotein_renal": 0.5,
}

_VALID_TRANSPORTERS = frozenset(_TRANSPORTER_CL_SEC.keys())


@dataclass(frozen=True)
class RenalTransporterInhibitionResult:
    """Result of a renal transporter inhibition DDI prediction."""

    victim_drug: str
    perpetrator_drug: str
    transporter: str
    ki_mg_per_L: float
    perpetrator_cmax_mg_per_L: float
    cl_sec_baseline_L_per_h: float
    cl_sec_inhibited_L_per_h: float
    cl_fil_L_per_h: float
    aucr: float
    ddi_risk: str
    regulatory_concern: bool
    recommendation: str
    notes: str


def _classify_ddi_risk(aucr: float) -> str:
    if aucr < 1.25:
        return "none"
    elif aucr < 2.0:
        return "mild"
    elif aucr < 5.0:
        return "moderate"
    else:
        return "severe"


def _build_recommendation(
    aucr: float,
    ddi_risk: str,
    regulatory_concern: bool,
    victim_drug: str,
    perpetrator_drug: str,
    transporter: str,
) -> str:
    if ddi_risk == "none":
        return (
            f"No clinically significant DDI expected between {perpetrator_drug} "
            f"and {victim_drug} via {transporter} inhibition."
        )
    base = (
        f"DDI risk ({ddi_risk}) detected: {perpetrator_drug} inhibits {transporter}, "
        f"increasing {victim_drug} AUC ~{aucr:.2f}-fold."
    )
    if regulatory_concern:
        base += " AUCR exceeds FDA 1.5-fold threshold — a clinical DDI study is recommended."
    if ddi_risk == "severe":
        base += " Consider dose reduction of victim drug or avoid co-administration."
    elif ddi_risk == "moderate":
        base += " Monitor victim drug levels and adjust dose as needed."
    return base


def predict_renal_transporter_inhibition(
    victim_drug: str,
    perpetrator_drug: str,
    transporter: str,
    ki_mg_per_L: float,
    perpetrator_cmax_mg_per_L: float,
    gfr_L_per_h: float = 7.2,
    fu_plasma: float = 0.5,
    cl_reabsorption_L_per_h: float = 0.0,
) -> RenalTransporterInhibitionResult:
    """Predict renal transporter-mediated DDI for a single transporter.

    Parameters
    ----------
    victim_drug:
        Name of the victim (substrate) drug.
    perpetrator_drug:
        Name of the perpetrator (inhibitor) drug.
    transporter:
        Renal transporter being inhibited. Must be one of:
        OAT1, OAT3, OCT2, MATE1, P_glycoprotein_renal.
    ki_mg_per_L:
        Inhibition constant (Ki = IC50/2 approximation) in mg/L. Must be > 0.
    perpetrator_cmax_mg_per_L:
        Perpetrator plasma Cmax in mg/L. Must be >= 0.
    gfr_L_per_h:
        Glomerular filtration rate in L/h. Default 7.2 L/h (~120 mL/min).
    fu_plasma:
        Fraction unbound in plasma for victim drug. Must be in (0, 1].
    cl_reabsorption_L_per_h:
        Passive tubular reabsorption clearance in L/h. Default 0.
    """
    # --- Validation ---
    if transporter not in _VALID_TRANSPORTERS:
        raise ValueError(
            f"Invalid transporter '{transporter}'. Must be one of: {sorted(_VALID_TRANSPORTERS)}"
        )
    if ki_mg_per_L <= 0:
        raise ValueError(f"ki_mg_per_L must be > 0, got {ki_mg_per_L}")
    if perpetrator_cmax_mg_per_L < 0:
        raise ValueError(f"perpetrator_cmax_mg_per_L must be >= 0, got {perpetrator_cmax_mg_per_L}")
    if gfr_L_per_h <= 0:
        raise ValueError(f"gfr_L_per_h must be > 0, got {gfr_L_per_h}")
    if not (0 < fu_plasma <= 1.0):
        raise ValueError(f"fu_plasma must be in (0, 1], got {fu_plasma}")
    if cl_reabsorption_L_per_h < 0:
        raise ValueError(f"cl_reabsorption_L_per_h must be >= 0, got {cl_reabsorption_L_per_h}")

    # --- Clearance components ---
    cl_fil = gfr_L_per_h * fu_plasma  # filtration clearance
    cl_sec_baseline = _TRANSPORTER_CL_SEC[transporter]

    # Competitive inhibition: CL_sec_inhibited = CL_sec * 1/(1 + [I]/Ki)
    inhibition_factor = 1.0 / (1.0 + perpetrator_cmax_mg_per_L / ki_mg_per_L)
    cl_sec_inhibited = cl_sec_baseline * inhibition_factor

    # AUCR = (cl_fil + cl_sec_baseline) / (cl_fil + cl_sec_inhibited)
    # Reabsorption reduces overall renal CL but acts equally in both states
    numerator = cl_fil + cl_sec_baseline - cl_reabsorption_L_per_h
    denominator = cl_fil + cl_sec_inhibited - cl_reabsorption_L_per_h

    # Guard against zero/negative denominator (edge case)
    if denominator <= 0:
        denominator = 1e-9

    aucr = numerator / denominator
    # AUCR must be >= 1.0 (inhibition reduces CL, increasing AUC)
    if aucr < 1.0:
        aucr = 1.0

    ddi_risk = _classify_ddi_risk(aucr)
    regulatory_concern = aucr > 1.5

    recommendation = _build_recommendation(
        aucr,
        ddi_risk,
        regulatory_concern,
        victim_drug,
        perpetrator_drug,
        transporter,
    )
    notes = (
        f"Inhibition factor: {inhibition_factor:.3f} "
        f"(Cmax/Ki = {perpetrator_cmax_mg_per_L / ki_mg_per_L:.3f}). "
        f"CL_fil = {cl_fil:.3f} L/h. "
        f"Reabsorption CL = {cl_reabsorption_L_per_h:.3f} L/h."
    )

    return RenalTransporterInhibitionResult(
        victim_drug=victim_drug,
        perpetrator_drug=perpetrator_drug,
        transporter=transporter,
        ki_mg_per_L=ki_mg_per_L,
        perpetrator_cmax_mg_per_L=perpetrator_cmax_mg_per_L,
        cl_sec_baseline_L_per_h=cl_sec_baseline,
        cl_sec_inhibited_L_per_h=cl_sec_inhibited,
        cl_fil_L_per_h=cl_fil,
        aucr=aucr,
        ddi_risk=ddi_risk,
        regulatory_concern=regulatory_concern,
        recommendation=recommendation,
        notes=notes,
    )


def screen_transporters(
    victim_drug: str,
    perpetrator_drug: str,
    ki_dict: dict[str, float],
    perpetrator_cmax: float,
    **kwargs,
) -> list[RenalTransporterInhibitionResult]:
    """Screen a perpetrator against multiple transporters.

    Parameters
    ----------
    victim_drug:
        Name of the victim drug.
    perpetrator_drug:
        Name of the perpetrator drug.
    ki_dict:
        Mapping of {transporter_name: ki_value_mg_per_L}.
    perpetrator_cmax:
        Perpetrator plasma Cmax in mg/L.
    **kwargs:
        Additional keyword arguments forwarded to
        ``predict_renal_transporter_inhibition`` (e.g. gfr_L_per_h, fu_plasma).
    """
    results: list[RenalTransporterInhibitionResult] = []
    for transporter, ki in ki_dict.items():
        result = predict_renal_transporter_inhibition(
            victim_drug=victim_drug,
            perpetrator_drug=perpetrator_drug,
            transporter=transporter,
            ki_mg_per_L=ki,
            perpetrator_cmax_mg_per_L=perpetrator_cmax,
            **kwargs,
        )
        results.append(result)
    results.sort(key=lambda r: r.aucr, reverse=True)
    return results
