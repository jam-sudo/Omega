"""First-in-human (FIH) dose prediction from preclinical NOAEL data."""

from __future__ import annotations

from dataclasses import dataclass

# Body surface area (m²) and body weight (kg) for standard species
_SPECIES_DATA = {
    "mouse": {"bw_kg": 0.02, "bsa_m2": 0.007, "km_factor": 3},
    "rat": {"bw_kg": 0.15, "bsa_m2": 0.025, "km_factor": 6},
    "rabbit": {"bw_kg": 1.8, "bsa_m2": 0.15, "km_factor": 12},
    "dog": {"bw_kg": 10.0, "bsa_m2": 0.50, "km_factor": 20},
    "monkey": {"bw_kg": 3.0, "bsa_m2": 0.24, "km_factor": 12},
    "human": {"bw_kg": 60.0, "bsa_m2": 1.62, "km_factor": 37},
}

# FDA-recommended safety factors by species (Guidance for Industry 2005)
_SAFETY_FACTORS = {
    "mouse": 10,
    "rat": 6,
    "rabbit": 12,
    "dog": 2,
    "monkey": 3,
}


@dataclass(frozen=True)
class FIHDoseResult:
    drug_name: str
    species: str
    noael_mg_kg: float  # NOAEL in the animal species
    noael_mg_m2: float  # NOAEL normalized to body surface area
    hed_mg_kg: float  # Human equivalent dose (mg/kg)
    hed_mg: float  # Absolute HED for 60 kg human
    mrsd_mg: float  # Maximum recommended starting dose (mg)
    mrsd_mg_kg: float  # MRSD per kg
    safety_factor: float  # Applied safety factor
    method: str  # "bsa_normalization" or "allometric"
    human_bw_kg: float
    notes: str


def predict_fih_dose(
    drug_name: str,
    noael_mg_kg: float,
    species: str = "rat",
    safety_factor: float | None = None,
    method: str = "bsa",
    human_bw_kg: float = 60.0,
    allometric_exponent: float = 0.75,
) -> FIHDoseResult:
    """Predict first-in-human starting dose from animal NOAEL.

    Two methods (FDA 2005 Guidance):
    1. **BSA normalization** (default): HED = NOAEL * (animal_Km / human_Km)
       where Km = BW(kg) / BSA(m²) is the body weight to BSA conversion factor.
    2. **Allometric scaling**: HED = NOAEL * (BW_human/BW_animal)^(1-exponent)

    MRSD = HED / safety_factor

    Parameters
    ----------
    noael_mg_kg : float
        No-observed-adverse-effect level in the test species (mg/kg).
    species : str
        Test species: 'mouse', 'rat', 'rabbit', 'dog', 'monkey'.
    safety_factor : float or None
        Override default FDA safety factor. None = use FDA species-specific default.
    method : str
        'bsa' (BSA normalization) or 'allometric'.
    allometric_exponent : float
        Exponent for allometric scaling (default 0.75 for CL, 1.0 for dose).
    """
    if noael_mg_kg <= 0:
        raise ValueError("noael_mg_kg must be > 0")
    species_lower = species.lower()
    if species_lower not in _SPECIES_DATA:
        raise ValueError(f"Unknown species '{species}'. Valid: {list(_SPECIES_DATA.keys())}")

    animal = _SPECIES_DATA[species_lower]
    human = _SPECIES_DATA["human"]

    # Adjust human data for actual body weight
    bw_ratio = human_bw_kg / human["bw_kg"]
    human_bsa_m2 = human["bsa_m2"] * (bw_ratio**0.667)  # BSA scales as BW^0.667
    human_km = human_bw_kg / human_bsa_m2

    # NOAEL in mg/m² (animal)
    animal_bsa = animal["bsa_m2"]
    animal_bw = animal["bw_kg"]
    noael_mg_m2 = noael_mg_kg * animal_bw / animal_bsa

    if method == "bsa":
        # HED (mg/kg) = NOAEL(mg/kg) * Km_animal / Km_human
        animal_km = animal["km_factor"]  # FDA tabulated Km values
        hed_mg_kg = noael_mg_kg * animal_km / human_km
    else:  # allometric
        # HED = NOAEL * (BW_human / BW_animal)^(1 - exponent)
        hed_mg_kg = noael_mg_kg * (human_bw_kg / animal_bw) ** (1.0 - allometric_exponent)

    hed_mg = hed_mg_kg * human_bw_kg

    # Safety factor
    sf = safety_factor if safety_factor is not None else _SAFETY_FACTORS.get(species_lower, 10)

    mrsd_mg = hed_mg / sf
    mrsd_mg_kg = mrsd_mg / human_bw_kg

    notes = (
        f"FDA 2005 Guidance: {method.upper()} method; "
        f"HED={hed_mg:.2f} mg ({hed_mg_kg:.3f} mg/kg); "
        f"Safety factor={sf}x; MRSD={mrsd_mg:.2f} mg"
    )

    return FIHDoseResult(
        drug_name=drug_name,
        species=species_lower,
        noael_mg_kg=noael_mg_kg,
        noael_mg_m2=noael_mg_m2,
        hed_mg_kg=hed_mg_kg,
        hed_mg=hed_mg,
        mrsd_mg=mrsd_mg,
        mrsd_mg_kg=mrsd_mg_kg,
        safety_factor=float(sf),
        method=f"{method}_normalization" if method == "bsa" else "allometric",
        human_bw_kg=human_bw_kg,
        notes=notes,
    )


def compare_species(
    drug_name: str,
    noael_by_species: dict[str, float],
    method: str = "bsa",
    human_bw_kg: float = 60.0,
) -> dict[str, FIHDoseResult]:
    """Compare FIH dose predictions across multiple species.

    The most conservative (lowest MRSD) is typically selected.

    Parameters
    ----------
    noael_by_species : dict
        {species: noael_mg_kg} e.g. {'rat': 100, 'dog': 30}
    """
    results = {}
    for species, noael in noael_by_species.items():
        results[species] = predict_fih_dose(
            drug_name, noael, species, method=method, human_bw_kg=human_bw_kg
        )
    return results


def most_conservative_mrsd(results: dict[str, FIHDoseResult]) -> FIHDoseResult:
    """Return the most conservative (lowest MRSD) result across species."""
    return min(results.values(), key=lambda r: r.mrsd_mg)


__all__ = [
    "FIHDoseResult",
    "predict_fih_dose",
    "compare_species",
    "most_conservative_mrsd",
]
