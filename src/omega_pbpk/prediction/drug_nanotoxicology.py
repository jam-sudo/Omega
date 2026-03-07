"""Drug nanotoxicology prediction based on nanoparticle physicochemical properties."""

from __future__ import annotations

from dataclasses import dataclass

_VALID_MATERIALS = {"gold", "silver", "silica", "iron_oxide", "carbon", "polymer"}
_VALID_COATINGS = {"none", "PEG", "amine", "carboxyl"}


@dataclass(frozen=True)
class NanotoxicologyResult:
    particle_name: str
    size_nm: float
    surface_charge_mV: float
    material: str
    cytotoxicity_score: float
    oxidative_stress_score: float
    inflammatory_score: float
    genotoxicity_score: float
    overall_risk_score: float
    risk_category: str
    notes: str


def _clamp(value: float, low: float = 0.0, high: float = 100.0) -> float:
    return max(low, min(high, value))


def _cytotoxicity(
    material: str,
    size_nm: float,
    concentration_ug_mL: float,
    surface_coating: str,
) -> float:
    score = 20.0
    material_bonus = {
        "silver": 40.0,
        "carbon": 25.0,
        "silica": 15.0,
        "iron_oxide": 10.0,
        "gold": 5.0,
        "polymer": 0.0,
    }
    score += material_bonus.get(material, 0.0)

    if size_nm < 10:
        score += 20.0
    elif size_nm < 50:
        score += 10.0

    conc_bonus = min(0.01 * concentration_ug_mL, 30.0)
    score += conc_bonus

    coating_bonus = {"PEG": -15.0, "amine": 10.0, "carboxyl": 5.0, "none": 0.0}
    score += coating_bonus.get(surface_coating, 0.0)

    return _clamp(score)


def _oxidative_stress(
    material: str,
    size_nm: float,
    surface_charge_mV: float,
) -> float:
    score = 15.0
    material_bonus = {
        "silver": 35.0,
        "iron_oxide": 30.0,
        "carbon": 20.0,
        "silica": 10.0,
        "gold": 5.0,
        "polymer": 0.0,
    }
    score += material_bonus.get(material, 0.0)

    if size_nm < 20:
        score += 15.0

    if abs(surface_charge_mV) > 30:
        score += 10.0

    return _clamp(score)


def _inflammatory(
    material: str,
    surface_charge_mV: float,
    concentration_ug_mL: float,
) -> float:
    score = 10.0
    material_bonus = {
        "carbon": 30.0,
        "silica": 25.0,
        "silver": 20.0,
        "iron_oxide": 15.0,
        "gold": 5.0,
        "polymer": 0.0,
    }
    score += material_bonus.get(material, 0.0)

    if surface_charge_mV > 10:
        score += 15.0
    elif surface_charge_mV < -30:
        score += 10.0

    if concentration_ug_mL > 500:
        score += 20.0

    return _clamp(score)


def _genotoxicity(
    material: str,
    size_nm: float,
    surface_coating: str,
) -> float:
    score = 5.0
    material_bonus = {
        "silver": 30.0,
        "carbon": 20.0,
        "silica": 10.0,
        "iron_oxide": 10.0,
        "gold": 5.0,
        "polymer": 0.0,
    }
    score += material_bonus.get(material, 0.0)

    if size_nm < 10:
        score += 20.0

    if surface_coating == "amine":
        score += 10.0

    return _clamp(score)


def predict_nanotoxicology(
    particle_name: str,
    size_nm: float,
    surface_charge_mV: float,
    material: str,
    surface_coating: str = "none",
    concentration_ug_mL: float = 100.0,
) -> NanotoxicologyResult:
    """Predict toxicological risk of nanoparticles from physicochemical properties."""
    if size_nm <= 0 or size_nm > 1000:
        raise ValueError("size_nm must be in (0, 1000]")
    if concentration_ug_mL <= 0:
        raise ValueError("concentration_ug_mL must be > 0")
    if material not in _VALID_MATERIALS:
        raise ValueError(f"material must be one of {_VALID_MATERIALS}, got '{material}'")
    if surface_coating not in _VALID_COATINGS:
        raise ValueError(
            f"surface_coating must be one of {_VALID_COATINGS}, got '{surface_coating}'"
        )

    cyto = _cytotoxicity(material, size_nm, concentration_ug_mL, surface_coating)
    ox = _oxidative_stress(material, size_nm, surface_charge_mV)
    inflam = _inflammatory(material, surface_charge_mV, concentration_ug_mL)
    geno = _genotoxicity(material, size_nm, surface_coating)

    overall = _clamp(0.35 * cyto + 0.25 * ox + 0.25 * inflam + 0.15 * geno)

    if overall < 25:
        risk_category = "low"
    elif overall < 50:
        risk_category = "moderate"
    elif overall < 75:
        risk_category = "high"
    else:
        risk_category = "very_high"

    notes = (
        f"material={material}; size={size_nm}nm; coating={surface_coating}; "
        f"overall_risk={overall:.1f} ({risk_category})"
    )

    return NanotoxicologyResult(
        particle_name=particle_name,
        size_nm=size_nm,
        surface_charge_mV=surface_charge_mV,
        material=material,
        cytotoxicity_score=cyto,
        oxidative_stress_score=ox,
        inflammatory_score=inflam,
        genotoxicity_score=geno,
        overall_risk_score=overall,
        risk_category=risk_category,
        notes=notes,
    )


def compare_nanoparticles(particles: list) -> list:
    """Compare multiple nanoparticles sorted by overall_risk_score ascending."""
    results = [predict_nanotoxicology(**p) for p in particles]
    return sorted(results, key=lambda r: r.overall_risk_score)
