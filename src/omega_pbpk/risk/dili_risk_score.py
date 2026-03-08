"""Phase 392 — Drug-Induced Hepatotoxicity (DILI) Risk Score.

Predict drug-induced liver injury risk using physicochemical and structural
features based on the rule-of-two and dose-dependent thresholds.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class DILIRiskResult:
    """Result of DILI risk prediction."""

    drug_name: str
    dili_score: float  # 0-100
    dili_risk_category: str  # "low", "moderate", "high", "very_high"
    reactive_metabolite_risk: bool
    mitochondrial_liability: bool
    bile_salt_export_pump_inhibition: bool
    cholestatic_risk: bool
    hepatocellular_risk: bool
    daily_dose_contribution: float
    physicochemical_contribution: float
    structural_contribution: float
    max_daily_dose_mg: float
    idiosyncratic_risk: bool
    notes: str


def _risk_category(score: float) -> str:
    if score < 25.0:
        return "low"
    elif score < 50.0:
        return "moderate"
    elif score < 75.0:
        return "high"
    else:
        return "very_high"


def predict_dili_risk(
    drug_name: str,
    max_daily_dose_mg: float,
    logp: float,
    mw_da: float,
    n_reactive_groups: int,
    n_aromatic_rings: int,
    cLogP: float | None = None,
) -> DILIRiskResult:
    """Predict DILI risk from physicochemical and structural properties.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    max_daily_dose_mg:
        Maximum daily dose in milligrams.
    logp:
        Measured or calculated log P.
    mw_da:
        Molecular weight in Daltons.
    n_reactive_groups:
        Number of structural reactive metabolite alerts (e.g., quinones,
        Michael acceptors, epoxides).
    n_aromatic_rings:
        Number of aromatic rings in the molecule.
    cLogP:
        Calculated log P; if None, uses logp.

    Returns
    -------
    DILIRiskResult
    """
    # --- validation ---
    if max_daily_dose_mg <= 0:
        raise ValueError("max_daily_dose_mg must be > 0")
    if mw_da <= 0:
        raise ValueError("mw_da must be > 0")
    if n_reactive_groups < 0:
        raise ValueError("n_reactive_groups must be >= 0")
    if n_aromatic_rings < 0:
        raise ValueError("n_aromatic_rings must be >= 0")

    if cLogP is None:
        cLogP = logp

    # --- daily dose contribution (0-40 points) ---
    daily_dose_contribution = min(40.0, max_daily_dose_mg / 10.0)

    # --- physicochemical contribution (0-30 points) ---
    logp_points = max(0.0, (cLogP - 2.0) * 3.0)
    mw_points = max(0.0, (mw_da - 300.0) / 50.0)
    physicochemical_contribution = min(30.0, logp_points + mw_points)

    # --- structural contribution (0-30 points) ---
    reactive_points = n_reactive_groups * 8.0
    ring_points = max(0.0, (n_aromatic_rings - 2) * 3.0)
    structural_contribution = min(30.0, reactive_points + ring_points)

    # --- total score ---
    dili_score = min(
        100.0, daily_dose_contribution + physicochemical_contribution + structural_contribution
    )

    # --- risk flags ---
    reactive_metabolite_risk = n_reactive_groups >= 1
    mitochondrial_liability = cLogP > 3.0 and mw_da > 400.0
    bile_salt_export_pump_inhibition = cLogP > 2.0 and max_daily_dose_mg > 100.0
    cholestatic_risk = bile_salt_export_pump_inhibition
    hepatocellular_risk = reactive_metabolite_risk or (cLogP > 3.0 and max_daily_dose_mg > 200.0)
    idiosyncratic_risk = n_reactive_groups >= 1

    category = _risk_category(dili_score)

    # --- build notes ---
    flags = []
    if reactive_metabolite_risk:
        flags.append("reactive metabolite risk")
    if mitochondrial_liability:
        flags.append("mitochondrial liability")
    if bile_salt_export_pump_inhibition:
        flags.append("BSEP inhibition risk")
    if cholestatic_risk:
        flags.append("cholestatic risk")
    if idiosyncratic_risk:
        flags.append("idiosyncratic DILI risk")

    if flags:
        notes = f"DILI concerns: {', '.join(flags)}."
    else:
        notes = "No major DILI flags identified."

    return DILIRiskResult(
        drug_name=drug_name,
        dili_score=dili_score,
        dili_risk_category=category,
        reactive_metabolite_risk=reactive_metabolite_risk,
        mitochondrial_liability=mitochondrial_liability,
        bile_salt_export_pump_inhibition=bile_salt_export_pump_inhibition,
        cholestatic_risk=cholestatic_risk,
        hepatocellular_risk=hepatocellular_risk,
        daily_dose_contribution=daily_dose_contribution,
        physicochemical_contribution=physicochemical_contribution,
        structural_contribution=structural_contribution,
        max_daily_dose_mg=max_daily_dose_mg,
        idiosyncratic_risk=idiosyncratic_risk,
        notes=notes,
    )


def screen_dili_risk(drugs_list: list[dict]) -> list[DILIRiskResult]:
    """Screen a list of drugs for DILI risk.

    Parameters
    ----------
    drugs_list:
        List of dicts, each with keys:
        name, max_daily_dose_mg, logp, mw_da, n_reactive_groups, n_aromatic_rings.
        Optional key: cLogP.

    Returns
    -------
    List of DILIRiskResult sorted by dili_score descending (highest risk first).
    """
    results = []
    for drug in drugs_list:
        r = predict_dili_risk(
            drug_name=drug["name"],
            max_daily_dose_mg=drug["max_daily_dose_mg"],
            logp=drug["logp"],
            mw_da=drug["mw_da"],
            n_reactive_groups=drug["n_reactive_groups"],
            n_aromatic_rings=drug["n_aromatic_rings"],
            cLogP=drug.get("cLogP"),
        )
        results.append(r)

    results.sort(key=lambda r: r.dili_score, reverse=True)
    return results
