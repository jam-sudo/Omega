"""Phase 1060 — Drug Buccal Film Absorption Predictor.

Predict absorption from buccal film drug delivery systems
(mucoadhesive films, patches).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

_VALID_FILM_TYPES = {"HPMC", "PVA", "chitosan", "Eudragit"}
_VALID_ENHANCERS = {"none", "oleic_acid", "sodium_lauryl_sulfate", "menthol"}
_VALID_IONIZATION = {"acid", "base", "neutral", "zwitterion"}

# Film-type parameters: k_erode (/h), mucoadhesion (0-100), base_ka (arbitrary)
_FILM_PARAMS: dict[str, dict[str, float]] = {
    "HPMC": {"k_erode": 0.1, "mucoadhesion": 80.0, "base_ka": 0.8},
    "PVA": {"k_erode": 0.2, "mucoadhesion": 60.0, "base_ka": 1.0},
    "chitosan": {"k_erode": 0.15, "mucoadhesion": 90.0, "base_ka": 0.9},
    "Eudragit": {"k_erode": 0.05, "mucoadhesion": 70.0, "base_ka": 0.5},
}

_ENHANCER_FOLD: dict[str, float] = {
    "none": 1.0,
    "oleic_acid": 2.5,
    "sodium_lauryl_sulfate": 3.0,
    "menthol": 1.5,
}


@dataclass(frozen=True)
class BuccalFilmResult:
    """Result of buccal film absorption prediction."""

    drug_name: str
    film_type: str
    drug_loading_mg: float
    fa_buccal: float
    f_systemic: float
    erosion_time_h: float
    flux_ug_cm2_h: float
    tmax_h: float
    permeation_enhancer_effect: float
    mucoadhesion_score: float
    notes: str


def _validate_buccal_inputs(
    drug_loading_mg: float,
    film_area_cm2: float,
    film_thickness_um: float,
    film_type: str,
    permeation_enhancer: str,
    ionization_type: str,
    cl_hepatic: float,
    q_hepatic: float,
) -> None:
    if drug_loading_mg <= 0:
        raise ValueError(f"drug_loading_mg must be > 0, got {drug_loading_mg}")
    if film_area_cm2 <= 0:
        raise ValueError(f"film_area_cm2 must be > 0, got {film_area_cm2}")
    if film_thickness_um <= 0:
        raise ValueError(f"film_thickness_um must be > 0, got {film_thickness_um}")
    if film_type not in _VALID_FILM_TYPES:
        raise ValueError(f"film_type must be one of {sorted(_VALID_FILM_TYPES)}, got {film_type!r}")
    if permeation_enhancer not in _VALID_ENHANCERS:
        raise ValueError(
            f"permeation_enhancer must be one of {sorted(_VALID_ENHANCERS)}, got {permeation_enhancer!r}"
        )
    if ionization_type not in _VALID_IONIZATION:
        raise ValueError(
            f"ionization_type must be one of {sorted(_VALID_IONIZATION)}, got {ionization_type!r}"
        )
    if cl_hepatic <= 0:
        raise ValueError(f"cl_hepatic_L_per_h must be > 0, got {cl_hepatic}")
    if q_hepatic <= 0:
        raise ValueError(f"q_hepatic_L_per_h must be > 0, got {q_hepatic}")


def predict_buccal_film_absorption(
    drug_name: str,
    mw_Da: float,
    logp: float,
    drug_loading_mg: float,
    film_type: str = "HPMC",
    film_area_cm2: float = 2.0,
    film_thickness_um: float = 100.0,
    permeation_enhancer: str = "none",
    pka: float = 7.0,
    ionization_type: str = "neutral",
    cl_hepatic_L_per_h: float = 5.0,
    q_hepatic_L_per_h: float = 90.0,
) -> BuccalFilmResult:
    """Predict absorption from a buccal film delivery system.

    Parameters
    ----------
    drug_name:
        Identifier for the drug.
    mw_Da:
        Molecular weight in Daltons.
    logp:
        Octanol-water partition coefficient.
    drug_loading_mg:
        Total drug mass in film (mg).
    film_type:
        Polymer matrix: "HPMC", "PVA", "chitosan", or "Eudragit".
    film_area_cm2:
        Surface area of the buccal film (cm²).
    film_thickness_um:
        Film thickness (µm).
    permeation_enhancer:
        Chemical enhancer: "none", "oleic_acid", "sodium_lauryl_sulfate", "menthol".
    pka:
        Acid/base pKa value.
    ionization_type:
        One of "acid", "base", "neutral", "zwitterion".
    cl_hepatic_L_per_h:
        Hepatic clearance (L/h).
    q_hepatic_L_per_h:
        Hepatic blood flow (L/h).

    Returns
    -------
    BuccalFilmResult
    """
    _validate_buccal_inputs(
        drug_loading_mg,
        film_area_cm2,
        film_thickness_um,
        film_type,
        permeation_enhancer,
        ionization_type,
        cl_hepatic_L_per_h,
        q_hepatic_L_per_h,
    )

    params = _FILM_PARAMS[film_type]
    k_erode = params["k_erode"]
    mucoadhesion_score = params["mucoadhesion"]

    # Erosion time: time to 99% eroded = ln(0.01) / (-k_erode) = 4.6 / k_erode
    erosion_time_h = math.log(0.01) / (-k_erode)  # = 4.6/k_erode

    # Buccal pH = 7.0
    buccal_ph = 7.0
    if ionization_type == "acid":
        fi = 1.0 / (1.0 + 10 ** (pka - buccal_ph))
    elif ionization_type == "base":
        fi = 1.0 / (1.0 + 10 ** (buccal_ph - pka))
    else:
        fi = 0.0

    unionized = 1.0 - fi

    # Buccal permeability (slightly lower than SL)
    papp_log = 0.5 * logp - 0.01 * mw_Da - 2.5
    papp = 10**papp_log  # cm/s
    papp = max(1e-7, min(0.005, papp))
    peff_cm_h = papp * 3600.0  # cm/h

    # Ionization correction
    peff_cm_h *= unionized + fi * 0.01

    # Permeation enhancer effect
    enhancer_fold = _ENHANCER_FOLD[permeation_enhancer]

    # Surface drug concentration: mg/cm³
    # film_thickness_um → cm = film_thickness_um * 1e-4
    thickness_cm = film_thickness_um * 1e-4
    c_surface_mg_cm3 = drug_loading_mg / (film_area_cm2 * thickness_cm)

    # Steady-state flux (µg/cm²/h)
    flux = peff_cm_h * c_surface_mg_cm3 * enhancer_fold * 1e3  # µg/cm²/h
    flux = max(0.01, flux)

    # fa_buccal
    total_flux_ug = flux * film_area_cm2 * erosion_time_h
    total_flux_mg = total_flux_ug / 1000.0
    fa_buccal = min(0.9, total_flux_mg / drug_loading_mg)

    # f_systemic (buccal also bypasses portal, slightly less than SL)
    f_portal_bypass_b = 0.75
    e_h = min(0.99, cl_hepatic_L_per_h / q_hepatic_L_per_h)
    f_hepatic = 1.0 - e_h
    f_systemic = fa_buccal * (f_portal_bypass_b + (1.0 - f_portal_bypass_b) * f_hepatic)

    # tmax: rough midpoint of erosion
    tmax_h = erosion_time_h / 2.0

    # Notes
    note_parts = [
        f"erosion_time={erosion_time_h:.2f}h",
        f"Peff={peff_cm_h:.4f} cm/h",
        f"enhancer_fold={enhancer_fold:.1f}x",
        f"E_hepatic={e_h:.2f}",
        f"mucoadhesion={mucoadhesion_score:.0f}",
    ]
    notes = "; ".join(note_parts)

    return BuccalFilmResult(
        drug_name=drug_name,
        film_type=film_type,
        drug_loading_mg=drug_loading_mg,
        fa_buccal=fa_buccal,
        f_systemic=f_systemic,
        erosion_time_h=erosion_time_h,
        flux_ug_cm2_h=flux,
        tmax_h=tmax_h,
        permeation_enhancer_effect=enhancer_fold,
        mucoadhesion_score=mucoadhesion_score,
        notes=notes,
    )


def compare_film_types(
    drug_name: str,
    mw_Da: float,
    logp: float,
    drug_loading_mg: float,
) -> list[BuccalFilmResult]:
    """Compare all 4 film types for a given drug.

    Returns a list of BuccalFilmResult for all film types,
    sorted by f_systemic descending.
    """
    results = [
        predict_buccal_film_absorption(
            drug_name=drug_name,
            mw_Da=mw_Da,
            logp=logp,
            drug_loading_mg=drug_loading_mg,
            film_type=ft,
        )
        for ft in sorted(_VALID_FILM_TYPES)
    ]
    return sorted(results, key=lambda r: r.f_systemic, reverse=True)
