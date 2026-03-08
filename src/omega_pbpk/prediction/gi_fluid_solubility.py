"""Phase 316: Drug Solubility in Gastrointestinal Fluids.

Predict drug solubility in simulated gastrointestinal fluids (SGF, SIF,
FaSSIF, FeSSIF, FaSSGF) considering bile salt and lecithin effects.
"""

from dataclasses import dataclass

_FLUID_PARAMS: dict = {
    "sgf": {
        "ph": 1.6,
        "bile_salt_mM": 0.0,
        "lecithin_mM": 0.0,
        "name": "Simulated Gastric Fluid",
    },
    "sif": {
        "ph": 6.8,
        "bile_salt_mM": 0.0,
        "lecithin_mM": 0.0,
        "name": "Simulated Intestinal Fluid",
    },
    "fassif": {
        "ph": 6.5,
        "bile_salt_mM": 3.0,
        "lecithin_mM": 0.75,
        "name": "Fasted State SIF",
    },
    "fessif": {
        "ph": 5.0,
        "bile_salt_mM": 15.0,
        "lecithin_mM": 3.75,
        "name": "Fed State SIF",
    },
    "fassgf": {
        "ph": 1.6,
        "bile_salt_mM": 0.08,
        "lecithin_mM": 0.02,
        "name": "Fasted State SGF",
    },
}

_VALID_IONIZATION_TYPES = {"acid", "base", "neutral"}


@dataclass(frozen=True)
class GIFluidSolubilityResult:
    """Result of GI fluid solubility prediction."""

    drug_name: str
    fluid_type: str
    logp: float
    pka: float
    ionization_type: str
    fluid_ph: float
    bile_salt_mM: float
    lecithin_mM: float
    intrinsic_solubility_mg_mL: float
    predicted_solubility_mg_mL: float
    biorelevant_enhancement_fold: float
    bcs_solubility_class: str
    notes: str


def predict_gi_solubility(
    drug_name: str,
    logp: float,
    pka: float,
    ionization_type: str,
    mw: float,
    intrinsic_solubility_mg_mL: float,
    fluid_type: str,
    dose_mg_reference: float = 1.0,
) -> GIFluidSolubilityResult:
    """Predict drug solubility in a specific GI fluid.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    logp:
        Octanol-water partition coefficient (log scale).
    pka:
        Acid dissociation constant.
    ionization_type:
        Ionization type: "acid", "base", or "neutral".
    mw:
        Molecular weight (g/mol).
    intrinsic_solubility_mg_mL:
        Intrinsic (un-ionized) solubility in mg/mL.
    fluid_type:
        GI fluid type: "sgf", "sif", "fassif", "fessif", or "fassgf".
    dose_mg_reference:
        Reference dose for BCS dose number calculation (mg). Default 1.0.

    Returns
    -------
    GIFluidSolubilityResult
    """
    _validate_gi_inputs(logp, pka, ionization_type, intrinsic_solubility_mg_mL, fluid_type)

    params = _FLUID_PARAMS[fluid_type]
    ph = params["ph"]
    bile_salt_mM = params["bile_salt_mM"]
    lecithin_mM = params["lecithin_mM"]
    fluid_name = params["name"]

    # pH-dependent solubility (Henderson-Hasselbalch)
    sol_ph = _ph_solubility(intrinsic_solubility_mg_mL, ph, pka, ionization_type)

    # Bile salt and lecithin solubilization (only for lipophilic drugs)
    if logp > 1.0:
        k_m_bile = 10.0 / max(1.0, logp)
        bile_enhancement = 1.0 + k_m_bile * bile_salt_mM

        if lecithin_mM > 0.0:
            lec_factor = 1.0 + 0.5 * lecithin_mM * max(0.0, logp - 1.0)
        else:
            lec_factor = 1.0
    else:
        bile_enhancement = 1.0
        lec_factor = 1.0

    predicted_raw = sol_ph * bile_enhancement * lec_factor
    # Clamp to [1e-6, 1000.0]
    predicted_solubility = max(1e-6, min(1000.0, predicted_raw))

    # BCS dose number (250 mL reference volume)
    dose_number = dose_mg_reference / (predicted_solubility * 250.0)
    bcs_solubility_class = "high" if dose_number <= 1.0 else "low"

    intrinsic_denom = max(intrinsic_solubility_mg_mL, 1e-9)
    biorelevant_enhancement_fold = predicted_solubility / intrinsic_denom

    notes = (
        f"Solubility of {drug_name} in {fluid_name} (pH {ph:.1f}). "
        f"logP={logp}, pKa={pka}, {ionization_type}. "
        f"Bile salt={bile_salt_mM} mM, lecithin={lecithin_mM} mM. "
        f"BCS class: {bcs_solubility_class} (dose number={dose_number:.4f})."
    )

    return GIFluidSolubilityResult(
        drug_name=drug_name,
        fluid_type=fluid_type,
        logp=logp,
        pka=pka,
        ionization_type=ionization_type,
        fluid_ph=ph,
        bile_salt_mM=bile_salt_mM,
        lecithin_mM=lecithin_mM,
        intrinsic_solubility_mg_mL=intrinsic_solubility_mg_mL,
        predicted_solubility_mg_mL=predicted_solubility,
        biorelevant_enhancement_fold=biorelevant_enhancement_fold,
        bcs_solubility_class=bcs_solubility_class,
        notes=notes,
    )


def compare_fluids(
    drug_name: str,
    logp: float,
    pka: float,
    ionization_type: str,
    mw: float,
    intrinsic_solubility_mg_mL: float,
    dose_mg_reference: float = 1.0,
) -> list:
    """Compare drug solubility across all GI fluid types.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    logp:
        Octanol-water partition coefficient (log scale).
    pka:
        Acid dissociation constant.
    ionization_type:
        Ionization type: "acid", "base", or "neutral".
    mw:
        Molecular weight (g/mol).
    intrinsic_solubility_mg_mL:
        Intrinsic (un-ionized) solubility in mg/mL.
    dose_mg_reference:
        Reference dose for BCS dose number calculation (mg).

    Returns
    -------
    list of GIFluidSolubilityResult sorted by predicted_solubility_mg_mL descending.
    """
    results = []
    for fluid_type in _FLUID_PARAMS:
        result = predict_gi_solubility(
            drug_name=drug_name,
            logp=logp,
            pka=pka,
            ionization_type=ionization_type,
            mw=mw,
            intrinsic_solubility_mg_mL=intrinsic_solubility_mg_mL,
            fluid_type=fluid_type,
            dose_mg_reference=dose_mg_reference,
        )
        results.append(result)
    results.sort(key=lambda r: r.predicted_solubility_mg_mL, reverse=True)
    return results


def _ph_solubility(intrinsic: float, ph: float, pka: float, ionization_type: str) -> float:
    """Calculate pH-dependent solubility using Henderson-Hasselbalch."""
    if ionization_type == "acid":
        # More soluble at high pH (ionized form)
        sol_ph = intrinsic * (1.0 + 10.0 ** (ph - pka))
    elif ionization_type == "base":
        # More soluble at low pH (ionized form)
        sol_ph = intrinsic * (1.0 + 10.0 ** (pka - ph))
    else:
        # Neutral: no pH effect
        sol_ph = intrinsic

    # Clamp: [intrinsic * 0.01, intrinsic * 1000]
    low = intrinsic * 0.01
    high = intrinsic * 1000.0
    return max(low, min(high, sol_ph))


def _validate_gi_inputs(
    logp: float,
    pka: float,
    ionization_type: str,
    intrinsic_solubility_mg_mL: float,
    fluid_type: str,
) -> None:
    """Validate inputs for GI solubility prediction."""
    if not (-5.0 < logp < 10.0):
        raise ValueError(f"logp must be in (-5, 10), got {logp}")
    if not (0.0 < pka <= 14.0):
        raise ValueError(f"pka must be in (0, 14], got {pka}")
    if ionization_type not in _VALID_IONIZATION_TYPES:
        raise ValueError(
            f"ionization_type must be one of {_VALID_IONIZATION_TYPES}, got '{ionization_type}'"
        )
    if intrinsic_solubility_mg_mL <= 0:
        raise ValueError(
            f"intrinsic_solubility_mg_mL must be > 0, got {intrinsic_solubility_mg_mL}"
        )
    if fluid_type not in _FLUID_PARAMS:
        raise ValueError(
            f"fluid_type must be one of {set(_FLUID_PARAMS.keys())}, got '{fluid_type}'"
        )
