"""Phase 302: Drug precipitation risk predictor for IV administration.

Predicts precipitation risk when a drug concentrate (e.g., IV push) is diluted
into physiological fluids. Based on solubility, pH shift, and supersaturation.

Physics
-------
- Diluted concentration: C_diluted = drug_conc / (1 + dilution_ratio)
- pH after dilution:
    ph_diluted = (vehicle_ph + dilution_ratio * dilution_ph) / (1 + dilution_ratio)
- Solubility at diluted pH (Henderson-Hasselbalch):
    acid:    sol_ph = sol_ph74 * (1 + 10^(ph-pka)) / (1 + 10^(7.4-pka))
    base:    sol_ph = sol_ph74 * (1 + 10^(pka-ph)) / (1 + 10^(pka-7.4))
    neutral: sol_ph = sol_ph74
- Supersaturation ratio: SR = C_diluted / sol_ph
- Risk score: min(100, max(0, (SR-1)*100)) if SR>1
              else max(0, (1 - sol_ph74)*10)
- Time to precipitate: 10 / (SR-1) min if SR>1, else inf

References
----------
- Yalkowsky SH et al., J Pharm Sci. 2010;99(11):4568-90 (IV compatibility)
- Serajuddin ATM, Adv Drug Deliv Rev. 2007;59(7):603-16 (solubility)
- Lindenberg M et al., Eur J Pharm Biopharm. 2004;58(2):265-78 (HH ionization)
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "PrecipitationRiskResult",
    "predict_precipitation_risk",
    "assess_iv_compatibility",
]

_VALID_IONIZATION_TYPES = {"acid", "base", "neutral"}


@dataclass(frozen=True)
class PrecipitationRiskResult:
    """Results from precipitation risk prediction (Phase 302).

    Attributes
    ----------
    drug_name : Drug identifier.
    drug_concentration_mg_mL : Original drug concentrate concentration (mg/mL).
    solubility_at_ph74_mg_mL : Drug solubility at pH 7.4 (mg/mL).
    pka : Drug pKa value.
    ionization_type : 'acid', 'base', or 'neutral'.
    vehicle_ph : pH of the drug vehicle/concentrate.
    dilution_ph : pH of the diluent fluid.
    dilution_ratio : Volume ratio diluent / drug (e.g., 10 means 10:1).
    c_diluted_mg_mL : Drug concentration after dilution (mg/mL).
    ph_diluted : pH of the mixture after dilution.
    solubility_at_diluted_ph_mg_mL : Drug solubility at the diluted pH (mg/mL).
    supersaturation_ratio : SR = C_diluted / solubility_at_diluted_ph.
    precipitation_risk_score : Risk score 0-100 (higher = more risk).
    risk_class : 'high' (SR>2), 'moderate' (1<SR<=2), or 'low' (SR<=1).
    time_to_precipitate_min : Estimated induction time (min); inf if no risk.
    notes : Summary of the prediction.
    """

    drug_name: str
    drug_concentration_mg_mL: float
    solubility_at_ph74_mg_mL: float
    pka: float
    ionization_type: str
    vehicle_ph: float
    dilution_ph: float
    dilution_ratio: float
    c_diluted_mg_mL: float
    ph_diluted: float
    solubility_at_diluted_ph_mg_mL: float
    supersaturation_ratio: float
    precipitation_risk_score: float
    risk_class: str
    time_to_precipitate_min: float
    notes: str


def _solubility_at_ph(
    sol_ph74: float,
    pka: float,
    ionization_type: str,
    ph: float,
) -> float:
    """Compute drug solubility at a given pH via Henderson-Hasselbalch.

    Parameters
    ----------
    sol_ph74 : Solubility at pH 7.4 (mg/mL).
    pka : Drug pKa.
    ionization_type : 'acid', 'base', or 'neutral'.
    ph : Target pH.

    Returns
    -------
    float
        Solubility at the target pH (mg/mL). Minimum 1e-9 to avoid division errors.
    """
    if ionization_type == "neutral":
        return sol_ph74

    # Henderson-Hasselbalch ratio relative to pH 7.4
    if ionization_type == "acid":
        # Acidic drug: ionized at high pH (more soluble at high pH)
        numerator = 1.0 + 10.0 ** (ph - pka)
        denominator = 1.0 + 10.0 ** (7.4 - pka)
    else:
        # Basic drug: ionized at low pH (more soluble at low pH)
        numerator = 1.0 + 10.0 ** (pka - ph)
        denominator = 1.0 + 10.0 ** (pka - 7.4)

    if denominator <= 0:
        return sol_ph74

    ratio = numerator / denominator
    return max(1e-9, sol_ph74 * ratio)


def predict_precipitation_risk(
    drug_name: str,
    drug_concentration_mg_mL: float,
    solubility_at_ph74_mg_mL: float,
    pka: float,
    ionization_type: str,
    vehicle_ph: float,
    dilution_ph: float,
    dilution_ratio: float,
) -> PrecipitationRiskResult:
    """Predict precipitation risk upon dilution of IV drug concentrate.

    Parameters
    ----------
    drug_name : Drug name.
    drug_concentration_mg_mL : Drug concentration in concentrate (mg/mL, must be > 0).
    solubility_at_ph74_mg_mL : Aqueous solubility at pH 7.4 (mg/mL, must be > 0).
    pka : Drug pKa (must be in (0, 14]).
    ionization_type : Ionization type: 'acid', 'base', or 'neutral'.
    vehicle_ph : pH of the drug vehicle concentrate (must be in (0, 14)).
    dilution_ph : pH of the diluent (must be in (0, 14)).
    dilution_ratio : Volume ratio diluent/drug (must be > 0).

    Returns
    -------
    PrecipitationRiskResult

    Raises
    ------
    ValueError : If any parameter is invalid.
    """
    # --- Validation ---
    if drug_concentration_mg_mL <= 0:
        raise ValueError(f"drug_concentration_mg_mL must be > 0, got {drug_concentration_mg_mL}")
    if solubility_at_ph74_mg_mL <= 0:
        raise ValueError(f"solubility_at_ph74_mg_mL must be > 0, got {solubility_at_ph74_mg_mL}")
    if not (0 < pka <= 14):
        raise ValueError(f"pka must be in (0, 14], got {pka}")
    if ionization_type not in _VALID_IONIZATION_TYPES:
        raise ValueError(
            f"ionization_type must be one of {_VALID_IONIZATION_TYPES}, got {ionization_type!r}"
        )
    if not (0 < vehicle_ph < 14):
        raise ValueError(f"vehicle_ph must be in (0, 14), got {vehicle_ph}")
    if not (0 < dilution_ph < 14):
        raise ValueError(f"dilution_ph must be in (0, 14), got {dilution_ph}")
    if dilution_ratio <= 0:
        raise ValueError(f"dilution_ratio must be > 0, got {dilution_ratio}")

    # --- Physics ---
    c_diluted = drug_concentration_mg_mL / (1.0 + dilution_ratio)
    ph_diluted = (vehicle_ph + dilution_ratio * dilution_ph) / (1.0 + dilution_ratio)

    sol_diluted = _solubility_at_ph(solubility_at_ph74_mg_mL, pka, ionization_type, ph_diluted)

    sr = c_diluted / sol_diluted if sol_diluted > 0 else float("inf")

    # Risk score
    if sr > 1.0:
        risk_score = min(100.0, max(0.0, (sr - 1.0) * 100.0))
        time_to_ppt = 10.0 / (sr - 1.0)
        risk_class = "high" if sr > 2.0 else "moderate"
    else:
        risk_score = max(0.0, (1.0 - solubility_at_ph74_mg_mL) * 10.0)
        time_to_ppt = float("inf")
        risk_class = "low"

    notes = (
        f"Precipitation risk for {drug_name}: "
        f"C_diluted={c_diluted:.4g} mg/mL, pH_mix={ph_diluted:.2f}, "
        f"Sol@pH_mix={sol_diluted:.4g} mg/mL. "
        f"SR={sr:.3f}, Risk={risk_class} ({risk_score:.1f}/100). "
        f"Time to ppt={time_to_ppt:.1f} min."
    )

    return PrecipitationRiskResult(
        drug_name=drug_name,
        drug_concentration_mg_mL=drug_concentration_mg_mL,
        solubility_at_ph74_mg_mL=solubility_at_ph74_mg_mL,
        pka=pka,
        ionization_type=ionization_type,
        vehicle_ph=vehicle_ph,
        dilution_ph=dilution_ph,
        dilution_ratio=dilution_ratio,
        c_diluted_mg_mL=c_diluted,
        ph_diluted=ph_diluted,
        solubility_at_diluted_ph_mg_mL=sol_diluted,
        supersaturation_ratio=sr,
        precipitation_risk_score=risk_score,
        risk_class=risk_class,
        time_to_precipitate_min=time_to_ppt,
        notes=notes,
    )


def assess_iv_compatibility(
    drug_name: str,
    drug_concentration_mg_mL: float,
    solubility_at_ph74_mg_mL: float,
    pka: float,
    ionization_type: str,
    common_iv_fluids: list[dict],
) -> list[PrecipitationRiskResult]:
    """Assess IV compatibility with multiple common diluent fluids.

    Parameters
    ----------
    drug_name : Drug name.
    drug_concentration_mg_mL : Drug concentration in the IV concentrate (mg/mL).
    solubility_at_ph74_mg_mL : Drug solubility at pH 7.4 (mg/mL).
    pka : Drug pKa.
    ionization_type : 'acid', 'base', or 'neutral'.
    common_iv_fluids : List of dicts, each with keys:
        - fluid_name (str): Name of the IV fluid.
        - ph (float): pH of the IV fluid.
        The function uses dilution_ratio=10 and vehicle_ph=5.0 as defaults.

    Returns
    -------
    list[PrecipitationRiskResult]
        Results sorted by precipitation_risk_score ascending (safest first).

    Raises
    ------
    ValueError : If common_iv_fluids is empty or missing required keys.
    """
    if not common_iv_fluids:
        raise ValueError("common_iv_fluids must not be empty")

    results: list[PrecipitationRiskResult] = []
    for fluid in common_iv_fluids:
        if "fluid_name" not in fluid or "ph" not in fluid:
            raise ValueError("Each fluid dict must have 'fluid_name' and 'ph' keys")
        fluid_name = str(fluid["fluid_name"])
        fluid_ph = float(fluid["ph"])
        labeled_name = f"{drug_name} in {fluid_name}"

        r = predict_precipitation_risk(
            drug_name=labeled_name,
            drug_concentration_mg_mL=drug_concentration_mg_mL,
            solubility_at_ph74_mg_mL=solubility_at_ph74_mg_mL,
            pka=pka,
            ionization_type=ionization_type,
            vehicle_ph=5.0,  # typical drug vehicle pH
            dilution_ph=fluid_ph,
            dilution_ratio=10.0,  # standard 10:1 dilution
        )
        results.append(r)

    results.sort(key=lambda r: r.precipitation_risk_score)
    return results
