"""Phase 902 — Drug Pancreatic Secretion Model.

Models drug distribution into pancreatic juice — relevant for drugs used
in pancreatitis, pancreatic cancer treatment, and exocrine pancreatic
insufficiency.

Pancreatic juice (pH 7.8-8.3, ~1.5 L/day = ~0.063 mL/min) concentrates
basic drugs via ion trapping (Henderson-Hasselbalch).
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class PancreaticSecretionResult:
    """Result of pancreatic drug secretion prediction."""

    drug_name: str
    pka: float
    logp: float
    ionization_type: str
    pancreatic_ph: float
    pancreatic_plasma_ratio: float
    predicted_pancreatic_conc_mg_L: float
    plasma_conc_input_mg_L: float
    daily_pancreatic_excretion_mg: float
    pancreatic_t_to_duodenum_h: float
    local_enzyme_inhibition_risk: bool
    notes: str


def predict_pancreatic_secretion(
    drug_name: str,
    pka: float,
    logp: float,
    ionization_type: str = "base",
    plasma_conc_mg_L: float = 1.0,
    pancreatic_ph: float = 8.0,
    fu_plasma: float = None,
) -> PancreaticSecretionResult:
    """Predict drug concentration and secretion in pancreatic juice.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    pka:
        Acid dissociation constant (0–14).
    logp:
        Octanol-water partition coefficient.
    ionization_type:
        "acid", "base", or "neutral".
    plasma_conc_mg_L:
        Plasma drug concentration in mg/L.
    pancreatic_ph:
        pH of pancreatic juice (default 8.0).
    fu_plasma:
        Unbound fraction in plasma. If None, estimated from logP.

    Returns
    -------
    PancreaticSecretionResult
        Predicted pancreatic pharmacokinetic parameters.

    Raises
    ------
    ValueError
        If pka not in [0, 14], plasma_conc_mg_L <= 0, or ionization_type invalid.
    """
    if not (0 <= pka <= 14):
        raise ValueError(f"pka must be in [0, 14], got {pka}")
    if plasma_conc_mg_L <= 0:
        raise ValueError(f"plasma_conc_mg_L must be > 0, got {plasma_conc_mg_L}")
    valid_types = {"acid", "base", "neutral"}
    if ionization_type not in valid_types:
        raise ValueError(f"ionization_type must be one of {valid_types}, got '{ionization_type}'")

    plasma_ph = 7.4

    # Estimate fu_plasma if not provided
    if fu_plasma is None:
        fu_plasma_estimate = max(0.01, min(1.0, 1 / (1 + 10 ** (logp - 2.5))))
    else:
        fu_plasma_estimate = fu_plasma

    # Henderson-Hasselbalch P/P ratio (ion trapping)
    if ionization_type == "base":
        numerator = 1 + 10 ** (pka - pancreatic_ph)
        denominator = 1 + 10 ** (pka - plasma_ph)
        pp_ratio = numerator / denominator
    elif ionization_type == "acid":
        numerator = 1 + 10 ** (pancreatic_ph - pka)
        denominator = 1 + 10 ** (plasma_ph - pka)
        pp_ratio = numerator / denominator
    else:  # neutral
        pp_ratio = 1.0

    # Clamp P/P ratio to [0.1, 50.0]
    pp_ratio = max(0.1, min(50.0, pp_ratio))

    # Predicted pancreatic concentration
    predicted_pancreatic_conc_mg_L = pp_ratio * plasma_conc_mg_L * fu_plasma_estimate

    # Daily pancreatic excretion (1.5 L/day × conc in mg/L = mg/day)
    daily_pancreatic_excretion_mg = predicted_pancreatic_conc_mg_L * 1.5

    # Fixed transit time
    pancreatic_t_to_duodenum_h = 0.5

    # Local enzyme inhibition risk (5x local concentration)
    local_enzyme_inhibition_risk = predicted_pancreatic_conc_mg_L > plasma_conc_mg_L * 5

    # Notes
    if local_enzyme_inhibition_risk:
        notes = "Local pancreatic enzyme inhibition risk — may impair digestion with chronic dosing"
    elif pp_ratio > 5:
        notes = "High pancreatic accumulation — consider pancreatitis monitoring"
    else:
        notes = "Normal pancreatic drug secretion"

    return PancreaticSecretionResult(
        drug_name=drug_name,
        pka=pka,
        logp=logp,
        ionization_type=ionization_type,
        pancreatic_ph=pancreatic_ph,
        pancreatic_plasma_ratio=pp_ratio,
        predicted_pancreatic_conc_mg_L=predicted_pancreatic_conc_mg_L,
        plasma_conc_input_mg_L=plasma_conc_mg_L,
        daily_pancreatic_excretion_mg=daily_pancreatic_excretion_mg,
        pancreatic_t_to_duodenum_h=pancreatic_t_to_duodenum_h,
        local_enzyme_inhibition_risk=local_enzyme_inhibition_risk,
        notes=notes,
    )


def screen_pancreatic_secretion(candidates: list) -> list:
    """Screen a list of drug candidates for pancreatic secretion.

    Parameters
    ----------
    candidates:
        List of dicts with keys: "name", "pka", "logp",
        optional "ionization_type" (default "neutral").

    Returns
    -------
    list
        List of PancreaticSecretionResult sorted by pancreatic_plasma_ratio descending.
    """
    results = []
    for cand in candidates:
        result = predict_pancreatic_secretion(
            drug_name=cand["name"],
            pka=cand["pka"],
            logp=cand["logp"],
            ionization_type=cand.get("ionization_type", "neutral"),
        )
        results.append(result)

    results.sort(key=lambda r: r.pancreatic_plasma_ratio, reverse=True)
    return results
