"""Phase 462 — Mucoadhesive Drug Delivery.

Models predicting mucoadhesive drug delivery performance, residence time,
and local concentration AUC for various polymers and mucosal routes.
"""

from __future__ import annotations

from dataclasses import dataclass

# Polymer database: name -> work_of_adhesion (mJ/m2), baseline residence time (h)
_POLYMER_DB: dict[str, dict[str, float]] = {
    "HPMC": {"woa": 15.0, "res_time": 4.0},
    "carbopol": {"woa": 35.0, "res_time": 8.0},
    "chitosan": {"woa": 25.0, "res_time": 6.0},
    "PVA": {"woa": 10.0, "res_time": 2.0},
    "alginate": {"woa": 20.0, "res_time": 5.0},
}

# Route mucus thickness in mm
_ROUTE_MUCUS_THICKNESS_MM: dict[str, float] = {
    "oral": 0.2,
    "nasal": 0.05,
    "vaginal": 0.3,
    "rectal": 0.1,
    "ophthalmic": 0.007,
}

_VALID_ROUTES = set(_ROUTE_MUCUS_THICKNESS_MM.keys())


@dataclass(frozen=True)
class MucoadhesiveResult:
    """Result of mucoadhesive delivery prediction."""

    drug_name: str
    polymer: str
    route: str
    mw_Da: float
    logp: float
    charge: float
    work_of_adhesion_mJ_per_m2: float
    residence_time_h: float
    mucus_thickness_mm: float
    drug_permeation_rate_mg_per_cm2_per_h: float
    local_auc_mg_h_per_mL: float
    bioavailability_enhancement_factor: float
    recommended: bool
    notes: str


def predict_mucoadhesive_delivery(
    drug_name: str,
    polymer: str,
    route: str,
    mw_Da: float,
    logp: float,
    charge: float = 0.0,
    dose_mg: float = 10.0,
) -> MucoadhesiveResult:
    """Predict mucoadhesive delivery performance.

    Parameters
    ----------
    drug_name : str
        Name of the drug.
    polymer : str
        Mucoadhesive polymer name (HPMC, carbopol, chitosan, PVA, alginate).
    route : str
        Administration route (oral, nasal, vaginal, rectal, ophthalmic).
    mw_Da : float
        Molecular weight in Daltons.
    logp : float
        Octanol-water partition coefficient.
    charge : float
        Drug charge (formal charge). Default 0.
    dose_mg : float
        Dose in mg. Default 10.0.

    Returns
    -------
    MucoadhesiveResult
    """
    if mw_Da <= 0:
        raise ValueError("mw_Da must be > 0")
    if route not in _VALID_ROUTES:
        raise ValueError(f"route must be one of {sorted(_VALID_ROUTES)}, got '{route}'")
    if polymer not in _POLYMER_DB:
        raise ValueError(f"polymer must be one of {sorted(_POLYMER_DB)}, got '{polymer}'")

    poly = _POLYMER_DB[polymer]
    woa_base = poly["woa"]
    res_time_base = poly["res_time"]
    mucus_thickness_mm = _ROUTE_MUCUS_THICKNESS_MM[route]

    # Work of adhesion: scale by charge and MW
    mw_factor = max(0.5, min(2.0, 500.0 / mw_Da))
    woa_effective = woa_base * (1.0 + 0.1 * abs(charge)) * mw_factor

    # Residence time: scale by charge and logP
    logp_factor = 0.5 if logp > 3.0 else 1.0
    residence_time_h = res_time_base * (1.0 + 0.05 * abs(charge)) * logp_factor

    # Permeation rate (mg/cm2/h), clamped to [0.001, 1.0]
    if mucus_thickness_mm > 0:
        raw_p = 0.01 * (1.0 + logp) / (mw_Da / 300.0) / mucus_thickness_mm
    else:
        raw_p = 1.0
    permeation_rate = max(0.001, min(1.0, raw_p))

    # Local AUC (simplified)
    denom = permeation_rate * mucus_thickness_mm * residence_time_h + 0.1
    local_auc = dose_mg / denom

    # Bioavailability enhancement factor
    bef = min(5.0, residence_time_h / 2.0)

    recommended = residence_time_h > 3.0 and permeation_rate > 0.0

    notes = (
        f"Polymer={polymer}, Route={route}, MW={mw_Da:.1f} Da, logP={logp:.2f}, "
        f"charge={charge}. WOA_eff={woa_effective:.2f} mJ/m2, "
        f"RT={residence_time_h:.2f} h, P={permeation_rate:.4f} mg/cm2/h."
    )

    return MucoadhesiveResult(
        drug_name=drug_name,
        polymer=polymer,
        route=route,
        mw_Da=mw_Da,
        logp=logp,
        charge=charge,
        work_of_adhesion_mJ_per_m2=woa_effective,
        residence_time_h=residence_time_h,
        mucus_thickness_mm=mucus_thickness_mm,
        drug_permeation_rate_mg_per_cm2_per_h=permeation_rate,
        local_auc_mg_h_per_mL=local_auc,
        bioavailability_enhancement_factor=bef,
        recommended=recommended,
        notes=notes,
    )


def screen_polymers(
    drug_name: str,
    route: str,
    mw_Da: float,
    logp: float,
    polymers: list | None = None,
) -> list:
    """Screen multiple polymers for a drug/route combination.

    Returns list of MucoadhesiveResult sorted by local_auc descending.
    """
    if polymers is None:
        polymers = list(_POLYMER_DB.keys())

    results = [
        predict_mucoadhesive_delivery(
            drug_name=drug_name,
            polymer=p,
            route=route,
            mw_Da=mw_Da,
            logp=logp,
        )
        for p in polymers
    ]
    return sorted(results, key=lambda r: r.local_auc_mg_h_per_mL, reverse=True)
