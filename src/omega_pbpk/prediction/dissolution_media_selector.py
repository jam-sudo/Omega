"""Phase 177 — Dissolution media selector for in vitro dissolution studies."""

import math
from dataclasses import dataclass

_MEDIA: dict[str, dict] = {
    "SGF_pH1.2": {
        "ph": 1.2,
        "volume_mL": 900,
        "buffer": "HCl",
        "enzymes": False,
        "fasted": True,
    },
    "acetate_pH4.5": {
        "ph": 4.5,
        "volume_mL": 900,
        "buffer": "acetate",
        "enzymes": False,
        "fasted": True,
    },
    "phosphate_pH6.8": {
        "ph": 6.8,
        "volume_mL": 900,
        "buffer": "phosphate",
        "enzymes": False,
        "fasted": True,
    },
    "FaSSIF": {
        "ph": 6.5,
        "volume_mL": 500,
        "buffer": "phosphate",
        "enzymes": False,
        "fasted": True,
        "bile_salts_mM": 3.0,
    },
    "FeSSIF": {
        "ph": 5.0,
        "volume_mL": 250,
        "buffer": "acetate",
        "enzymes": False,
        "fasted": False,
        "bile_salts_mM": 15.0,
    },
    "FaSSGF": {
        "ph": 1.6,
        "volume_mL": 250,
        "buffer": "HCl",
        "enzymes": True,
        "fasted": True,
    },
    "water": {
        "ph": 6.8,
        "volume_mL": 900,
        "buffer": "none",
        "enzymes": False,
        "fasted": True,
    },
}

_BIORELEVANT_MEDIA = {"FaSSIF", "FeSSIF", "FaSSGF"}

_FDA_PRIMARY: dict[str, str] = {
    "I": "phosphate_pH6.8",
    "II": "FaSSIF",
    "III": "phosphate_pH6.8",
    "IV": "FeSSIF",
}

_VALID_BCS = {"I", "II", "III", "IV"}
_VALID_ION = {"acid", "base", "neutral"}


@dataclass(frozen=True)
class DissolutionMediaResult:
    drug_name: str
    media_name: str
    media_ph: float
    bcs_class: str
    suitability_score: float
    solubility_at_media_ph_mg_mL: float
    fda_recommended: bool
    biorelevant: bool
    notes: str


@dataclass(frozen=True)
class DissolutionRateResult:
    drug_name: str
    media_ph: float
    solubility_at_ph_mg_mL: float
    diffusivity_cm2_per_s: float
    dissolution_rate_mg_per_min: float
    t90_min: float
    dissolution_class: str
    notes: str


def _solubility_at_ph(
    solubility_intrinsic_mg_mL: float,
    ionization_type: str,
    pka: float,
    ph: float,
) -> float:
    """Henderson-Hasselbalch solubility adjustment."""
    if ionization_type == "neutral":
        return solubility_intrinsic_mg_mL
    if ionization_type == "acid":
        # Sol = S0 * (1 + 10^(pH - pKa))
        factor = 1.0 + 10.0 ** (ph - pka)
    else:  # base
        # Sol = S0 * (1 + 10^(pKa - pH))
        factor = 1.0 + 10.0 ** (pka - ph)
    return solubility_intrinsic_mg_mL * factor


def _validate_inputs(
    bcs_class: str,
    ionization_type: str,
    pka: float,
    solubility_mg_mL: float,
) -> None:
    if bcs_class not in _VALID_BCS:
        raise ValueError(f"bcs_class must be one of {_VALID_BCS}, got '{bcs_class}'")
    if ionization_type not in _VALID_ION:
        raise ValueError(f"ionization_type must be one of {_VALID_ION}, got '{ionization_type}'")
    if not (0 <= pka <= 14):
        raise ValueError(f"pka must be in [0, 14], got {pka}")
    if solubility_mg_mL <= 0:
        raise ValueError(f"solubility_mg_mL must be > 0, got {solubility_mg_mL}")


def recommend_dissolution_media(
    drug_name: str,
    bcs_class: str,
    ionization_type: str,
    pka: float,
    solubility_mg_mL: float,
    dose_mg: float = 500.0,
) -> list[DissolutionMediaResult]:
    """Return list of DissolutionMediaResult ranked by suitability_score descending."""
    _validate_inputs(bcs_class, ionization_type, pka, solubility_mg_mL)
    if dose_mg <= 0:
        raise ValueError(f"dose_mg must be > 0, got {dose_mg}")

    primary = _FDA_PRIMARY[bcs_class]
    results = []

    for media_name, props in _MEDIA.items():
        score = 10.0  # base

        # FDA guidance match
        if media_name == primary:
            if bcs_class in ("I", "II", "IV"):
                score += 40.0
            else:  # BCS III
                score += 30.0

        # pH-solubility match
        media_ph = props["ph"]
        sol_at_ph = _solubility_at_ph(solubility_mg_mL, ionization_type, pka, media_ph)
        threshold = (dose_mg / 900.0) * 2.0
        if sol_at_ph > threshold:
            score += 30.0

        # Physiological relevance
        biorelevant = media_name in _BIORELEVANT_MEDIA
        score += 20.0 if biorelevant else 10.0

        # Cap at 100
        score = min(score, 100.0)

        fda_recommended = media_name == primary
        notes_parts = []
        if fda_recommended:
            notes_parts.append(f"FDA recommended for BCS {bcs_class}")
        if biorelevant:
            notes_parts.append("biorelevant medium")
        if sol_at_ph > threshold:
            notes_parts.append("adequate solubility at this pH")
        else:
            notes_parts.append("low solubility risk")
        notes = "; ".join(notes_parts) if notes_parts else ""

        results.append(
            DissolutionMediaResult(
                drug_name=drug_name,
                media_name=media_name,
                media_ph=media_ph,
                bcs_class=bcs_class,
                suitability_score=round(score, 2),
                solubility_at_media_ph_mg_mL=round(sol_at_ph, 4),
                fda_recommended=fda_recommended,
                biorelevant=biorelevant,
                notes=notes,
            )
        )

    results.sort(key=lambda r: r.suitability_score, reverse=True)
    return results


def calculate_dissolution_rate(
    drug_name: str,
    solubility_mg_mL: float,
    ionization_type: str,
    pka: float,
    media_ph: float,
    dose_mg: float,
    paddle_rpm: float = 50.0,
    mw: float = 300.0,
) -> DissolutionRateResult:
    """Noyes-Whitney dissolution rate calculation."""
    if solubility_mg_mL <= 0:
        raise ValueError(f"solubility_mg_mL must be > 0, got {solubility_mg_mL}")
    if ionization_type not in _VALID_ION:
        raise ValueError(f"ionization_type must be one of {_VALID_ION}, got '{ionization_type}'")
    if not (0 <= pka <= 14):
        raise ValueError(f"pka must be in [0, 14], got {pka}")
    if not (0 <= media_ph <= 14):
        raise ValueError(f"media_ph must be in [0, 14], got {media_ph}")
    if dose_mg <= 0:
        raise ValueError(f"dose_mg must be > 0, got {dose_mg}")
    if paddle_rpm <= 0:
        raise ValueError(f"paddle_rpm must be > 0, got {paddle_rpm}")

    # Diffusivity: D = 2.3e-6 / (MW^0.45) cm²/s
    effective_mw = max(mw, 1.0)
    D = 2.3e-6 / (effective_mw**0.45)

    # Diffusion layer thickness: h = 50/rpm * 0.01 cm
    h = (50.0 / paddle_rpm) * 0.01

    # Mass-transfer coefficient
    k = D / h  # cm/s

    # Surface area (fixed)
    A = 0.1  # cm²

    # Cs at media pH
    Cs = _solubility_at_ph(solubility_mg_mL, ionization_type, pka, media_ph)

    # Sink condition: C ≈ 0 → rate = k * A * Cs  (mg/cm³ * cm/s * cm²) = mg/s
    # Cs in mg/mL = mg/cm³
    rate_mg_per_s = k * A * Cs
    rate_mg_per_min = rate_mg_per_s * 60.0

    # t90: time for 90% of dose to dissolve
    if rate_mg_per_min > 0:
        t90_min = (dose_mg * 0.9) / rate_mg_per_min
    else:
        t90_min = float("inf")

    if t90_min < 15.0:
        dissolution_class = "rapid"
    elif t90_min <= 60.0:
        dissolution_class = "moderate"
    else:
        dissolution_class = "slow"

    notes = f"D={D:.3e} cm²/s, h={h:.4f} cm, k={k:.3e} cm/s, Cs={Cs:.4f} mg/mL at pH {media_ph:.1f}"

    return DissolutionRateResult(
        drug_name=drug_name,
        media_ph=media_ph,
        solubility_at_ph_mg_mL=round(Cs, 4),
        diffusivity_cm2_per_s=round(D, 8),
        dissolution_rate_mg_per_min=round(rate_mg_per_min, 6),
        t90_min=round(t90_min, 4) if math.isfinite(t90_min) else t90_min,
        dissolution_class=dissolution_class,
        notes=notes,
    )
