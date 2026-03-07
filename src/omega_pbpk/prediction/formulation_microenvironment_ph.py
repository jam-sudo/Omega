"""Formulation microenvironment pH prediction.

Predicts the microenvironment pH inside drug formulations (tablet cores, capsules,
microspheres) and quantifies its effect on drug solubility, dissolution rate,
stability, and release profile.

The microenvironment pH can differ significantly from the bulk medium pH due to
buffering excipients in the tablet/capsule core.  Acidic excipients (citric acid,
tartaric acid) depress the local pH; basic excipients (sodium bicarbonate,
magnesium carbonate) elevate it.  This pH gradient drives changes in ionisation,
solubility, and chemical stability for pH-sensitive drugs.

References
----------
- Kranz H & Wagner T. Eur J Pharm Biopharm. 2006;62(3):247-54
- Badawy SI & Hussain MA. AAPS PharmSciTech. 2007;8(4):E102
- pH-partition hypothesis: Shore PA et al., J Pharmacol Exp Ther. 1957;119(3):361-9
"""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = [
    "FormulationMicroenvironmentResult",
    "predict_microenvironment_ph",
    "screen_excipients",
]

# ---------------------------------------------------------------------------
# Valid option sets
# ---------------------------------------------------------------------------

_VALID_FORMULATION_TYPES = frozenset({"tablet", "capsule", "suspension", "solution", "microsphere"})
_VALID_IONIZATION_TYPES = frozenset({"acid", "base", "neutral"})
_VALID_BUFFER_SPECIES = frozenset({"citrate", "phosphate", "acetate", "none"})

# Bulk pH for each buffer species
_BUFFER_PH: dict[str, float] = {
    "citrate": 4.5,
    "phosphate": 7.0,
    "acetate": 4.8,
}

# Bulk pH when buffer_species == "none", keyed by formulation_type
_NO_BUFFER_PH: dict[str, float] = {
    "tablet": 5.5,
    "capsule": 5.5,
    "suspension": 5.0,
    "solution": 7.0,
    "microsphere": 6.0,
}

# Acidic and basic tablet-core excipients
_ACIDIC_EXCIPIENTS = frozenset({"citric_acid", "tartaric_acid"})
_BASIC_EXCIPIENTS = frozenset({"Na_bicarbonate", "Mg_carbonate"})

_MICROENV_ACID_OFFSET = -1.5
_MICROENV_BASE_OFFSET = +2.0
_MICROENV_PH_MIN = 2.0
_MICROENV_PH_MAX = 10.0

# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FormulationMicroenvironmentResult:
    """Microenvironment pH prediction for a drug formulation.

    Attributes
    ----------
    drug_name : Name of the drug or compound.
    formulation_type : One of ``tablet``, ``capsule``, ``suspension``,
        ``solution``, ``microsphere``.
    buffer_species : Buffer used in the formulation (``citrate``,
        ``phosphate``, ``acetate``, ``none``).
    buffer_conc_mM : Buffer concentration (mM).
    microenv_ph : Predicted microenvironment pH inside the formulation matrix.
    bulk_ph : Bulk dissolution-medium pH.
    ph_gradient : Absolute difference between microenv_ph and bulk_ph.
    solubility_at_microenv_mg_mL : Predicted solubility at microenv_ph (mg/mL).
    dissolution_rate_factor : Rate relative to bulk-pH dissolution (sqrt ratio).
    drug_stability_pct : % parent drug remaining after 1 h at microenv_ph.
    release_profile : One of ``immediate``, ``pH_dependent``, ``sustained``.
    compatibility_score : 0–100 composite (100 = excellent).
    notes : Human-readable summary.
    """

    drug_name: str
    formulation_type: str
    buffer_species: str
    buffer_conc_mM: float
    microenv_ph: float
    bulk_ph: float
    ph_gradient: float
    solubility_at_microenv_mg_mL: float
    dissolution_rate_factor: float
    drug_stability_pct: float
    release_profile: str
    compatibility_score: float
    notes: str


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _solubility(logp: float, pka: float, ionization_type: str, ph: float) -> float:
    """Henderson-Hasselbalch solubility (mg/mL).

    S0 = 10^(-logP + 1) mg/mL (intrinsic solubility, simplified).
    """
    s0 = 10.0 ** (-logp + 1.0)
    if ionization_type == "acid":
        s = s0 * (1.0 + 10.0 ** (ph - pka))
    elif ionization_type == "base":
        s = s0 * (1.0 + 10.0 ** (pka - ph))
    else:
        s = s0
    # Clamp to physical range
    s = max(0.001, min(1000.0, s))
    return s


def _bulk_ph(buffer_species: str, formulation_type: str) -> float:
    if buffer_species == "none":
        return _NO_BUFFER_PH[formulation_type]
    return _BUFFER_PH[buffer_species]


def _microenv_ph(
    bulk: float,
    formulation_type: str,
    tablet_core_excipient: str,
) -> float:
    if formulation_type != "tablet":
        return bulk
    if tablet_core_excipient in _ACIDIC_EXCIPIENTS:
        raw = bulk + _MICROENV_ACID_OFFSET
    elif tablet_core_excipient in _BASIC_EXCIPIENTS:
        raw = bulk + _MICROENV_BASE_OFFSET
    else:
        return bulk
    return max(_MICROENV_PH_MIN, min(_MICROENV_PH_MAX, raw))


def _stability_pct(microenv_ph: float) -> float:
    k_deg = 0.01 * (1.0 + abs(microenv_ph - 7.0) / 3.5)
    pct = 100.0 * math.exp(-k_deg * 1.0)
    return max(0.0, min(100.0, pct))


def _release_profile(dissolution_rate_factor: float) -> str:
    if dissolution_rate_factor > 2.0:
        return "pH_dependent"
    if dissolution_rate_factor < 0.5:
        return "sustained"
    return "immediate"


def _compatibility(stability_pct: float, dissolution_rate_factor: float) -> float:
    score = 80.0
    if stability_pct > 95.0:
        score += 10.0
    if 0.8 <= dissolution_rate_factor <= 2.0:
        score += 10.0
    if stability_pct < 70.0:
        score -= 20.0
    if dissolution_rate_factor < 0.3:
        score -= 10.0
    return max(0.0, min(100.0, score))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def predict_microenvironment_ph(
    drug_name: str,
    pka: float,
    logp: float,
    ionization_type: str,
    formulation_type: str,
    buffer_species: str = "none",
    buffer_conc_mM: float = 10.0,
    tablet_core_excipient: str = "MCC",
) -> FormulationMicroenvironmentResult:
    """Predict microenvironment pH and formulation compatibility.

    Parameters
    ----------
    drug_name : Name or identifier of the drug.
    pka : Drug pKa (0–14).
    logp : Octanol-water partition coefficient.
    ionization_type : ``acid``, ``base``, or ``neutral``.
    formulation_type : ``tablet``, ``capsule``, ``suspension``, ``solution``,
        or ``microsphere``.
    buffer_species : ``citrate``, ``phosphate``, ``acetate``, or ``none``.
    buffer_conc_mM : Buffer concentration (mM). Must be ≥ 0.
    tablet_core_excipient : Core excipient for tablets.  Special values
        ``citric_acid``, ``tartaric_acid`` (acidic), ``Na_bicarbonate``,
        ``Mg_carbonate`` (basic). Other strings (e.g. ``MCC``) → neutral.

    Returns
    -------
    FormulationMicroenvironmentResult
    """
    # --- Validation ---
    if not (0.0 <= pka <= 14.0):
        raise ValueError(f"pka must be in [0, 14]; got {pka}")
    if buffer_conc_mM < 0.0:
        raise ValueError(f"buffer_conc_mM must be >= 0; got {buffer_conc_mM}")
    if formulation_type not in _VALID_FORMULATION_TYPES:
        raise ValueError(
            f"Invalid formulation_type '{formulation_type}'. "
            f"Must be one of {sorted(_VALID_FORMULATION_TYPES)}."
        )
    if ionization_type not in _VALID_IONIZATION_TYPES:
        raise ValueError(
            f"Invalid ionization_type '{ionization_type}'. "
            f"Must be one of {sorted(_VALID_IONIZATION_TYPES)}."
        )
    if buffer_species not in _VALID_BUFFER_SPECIES:
        raise ValueError(
            f"Invalid buffer_species '{buffer_species}'. "
            f"Must be one of {sorted(_VALID_BUFFER_SPECIES)}."
        )

    # --- pH calculations ---
    bulk = _bulk_ph(buffer_species, formulation_type)
    micro = _microenv_ph(bulk, formulation_type, tablet_core_excipient)

    ph_gradient = abs(micro - bulk)

    # --- Solubility ---
    sol_micro = _solubility(logp, pka, ionization_type, micro)
    sol_bulk = _solubility(logp, pka, ionization_type, bulk)

    diss_factor = math.sqrt(sol_micro / max(sol_bulk, 0.001))

    # --- Stability ---
    stab_pct = _stability_pct(micro)

    # --- Release profile ---
    release = _release_profile(diss_factor)

    # --- Compatibility ---
    compat = _compatibility(stab_pct, diss_factor)

    # --- Notes ---
    notes = (
        f"Bulk pH={bulk:.2f}, microenv pH={micro:.2f} "
        f"(excipient='{tablet_core_excipient}'). "
        f"Solubility at microenv pH: {sol_micro:.3f} mg/mL "
        f"(bulk: {sol_bulk:.3f} mg/mL). "
        f"Dissolution rate factor={diss_factor:.3f}, "
        f"stability={stab_pct:.1f}%, "
        f"release='{release}', "
        f"compatibility={compat:.0f}/100."
    )

    return FormulationMicroenvironmentResult(
        drug_name=drug_name,
        formulation_type=formulation_type,
        buffer_species=buffer_species,
        buffer_conc_mM=buffer_conc_mM,
        microenv_ph=micro,
        bulk_ph=bulk,
        ph_gradient=ph_gradient,
        solubility_at_microenv_mg_mL=sol_micro,
        dissolution_rate_factor=diss_factor,
        drug_stability_pct=stab_pct,
        release_profile=release,
        compatibility_score=compat,
        notes=notes,
    )


def screen_excipients(
    drug_name: str,
    pka: float,
    logp: float,
    ionization_type: str,
    tablet_core_excipients: list[str],
) -> list[FormulationMicroenvironmentResult]:
    """Screen multiple tablet-core excipients and rank by compatibility.

    Parameters
    ----------
    drug_name : Name of the drug.
    pka : Drug pKa.
    logp : Drug logP.
    ionization_type : ``acid``, ``base``, or ``neutral``.
    tablet_core_excipients : List of excipient names to evaluate.

    Returns
    -------
    list[FormulationMicroenvironmentResult]
        Results sorted by compatibility_score descending.
    """
    results = [
        predict_microenvironment_ph(
            drug_name=drug_name,
            pka=pka,
            logp=logp,
            ionization_type=ionization_type,
            formulation_type="tablet",
            buffer_species="none",
            tablet_core_excipient=exc,
        )
        for exc in tablet_core_excipients
    ]
    results.sort(key=lambda r: r.compatibility_score, reverse=True)
    return results
