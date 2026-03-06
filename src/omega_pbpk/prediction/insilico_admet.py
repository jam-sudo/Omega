"""Integrated in silico ADMET profile — all-in-one composite predictor."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class ADMETProfile:
    compound_name: str
    smiles: str
    mw: float
    logP: float
    # Absorption
    hia_pct: float                    # Human intestinal absorption (%)
    oral_bioavailability_pct: float
    # Distribution
    vss_L_per_kg: float               # Steady-state Vd per kg
    fup: float                        # Unbound plasma fraction
    bbb_penetration: bool             # CNS penetration
    # Metabolism
    cyp3a4_substrate: bool
    cyp2d6_substrate: bool
    t_half_h: float                   # predicted terminal half-life
    # Excretion
    cl_total_L_per_h_per_kg: float
    fe_renal: float                   # renal excretion fraction
    # Toxicity flags
    herg_risk: bool                   # hERG inhibition risk
    ames_risk: bool                   # mutagenicity risk
    hepatotoxicity_risk: bool
    # Drug-likeness
    lipinski_pass: bool               # Rule of Five
    pains_flag: bool                  # PAINS alert (simplified)
    composite_score: float            # 0–100, higher = better ADMET
    notes: str


def predict_admet_profile(
    compound_name: str,
    smiles: str,
    mw: float,
    logP: float,
    psa: float = 90.0,
    n_hbd: int = 2,
    n_hba: int = 4,
    pka: float = 7.0,
) -> ADMETProfile:
    """Predict integrated ADMET profile from physicochemical descriptors.

    Parameters
    ----------
    compound_name : str
        Compound identifier.
    smiles : str
        SMILES string (used for structural alerts).
    mw : float
        Molecular weight (g/mol).
    logP : float
        Lipophilicity (log octanol/water partition coefficient).
    psa : float
        Polar surface area (Å²).
    n_hbd : int
        Number of hydrogen bond donors.
    n_hba : int
        Number of hydrogen bond acceptors.
    pka : float
        Dominant pKa (used for miscellaneous adjustments).
    """
    if mw <= 0:
        raise ValueError("mw must be > 0")
    if not smiles or not smiles.strip():
        raise ValueError("smiles must be a non-empty string")

    # ------------------------------------------------------------------ #
    # Absorption
    # ------------------------------------------------------------------ #
    hia_pct = 100.0 / (1.0 + math.exp(-2.0 * (logP - 1.0)))

    ba_pct = hia_pct * max(0.1, 1.0 - psa / 200.0) * max(0.1, 1.0 - (mw - 300.0) / 1000.0)
    ba_pct = max(0.0, min(100.0, ba_pct))

    # ------------------------------------------------------------------ #
    # Distribution
    # ------------------------------------------------------------------ #
    vss = max(0.1, 0.5 + logP * 0.3)           # L/kg rough estimate
    fup = max(0.01, min(1.0, 1.0 - 0.1 * logP))  # simplified PPB

    bbb = (1.0 <= logP <= 4.0) and (mw < 450.0) and (psa < 90.0)

    # ------------------------------------------------------------------ #
    # Metabolism
    # ------------------------------------------------------------------ #
    cyp3a4_sub = (logP > 2.0) and (mw > 300.0)
    cyp2d6_sub = ("N" in smiles) and (mw < 400.0)

    cl_est = max(0.01, logP * 0.5 + 0.5)       # L/h/kg rough
    t_half = 0.693 * vss / cl_est              # hours

    # ------------------------------------------------------------------ #
    # Excretion
    # ------------------------------------------------------------------ #
    fe_renal = max(0.0, 0.3 - logP * 0.05)

    # ------------------------------------------------------------------ #
    # Toxicity flags
    # ------------------------------------------------------------------ #
    herg_risk = (logP > 3.0) and (n_hba < 3) and (mw > 300.0)
    ames_risk = any(alert in smiles for alert in ["[N+](=O)[O-]", "NN", "C(=O)Cl"])
    hepatotox_risk = (logP > 3.0) and ("N" in smiles) and (mw > 250.0)

    # ------------------------------------------------------------------ #
    # Drug-likeness
    # ------------------------------------------------------------------ #
    lipinski = (mw <= 500.0) and (logP <= 5.0) and (n_hbd <= 5) and (n_hba <= 10)

    pains = (
        ("c1ccsc1" in smiles)
        or ("c1ccon1" in smiles)
        or ("[N+](=O)[O-]" in smiles and "c" in smiles)
    )

    # ------------------------------------------------------------------ #
    # Composite score 0–100
    # ------------------------------------------------------------------ #
    score = 0.0
    score += ba_pct * 0.3                       # max 30
    score += (0.0 if herg_risk else 1.0) * 20   # max 20
    score += (0.0 if ames_risk else 1.0) * 20   # max 20
    score += (15.0 if lipinski else 0.0)        # 15
    score += (0.0 if hepatotox_risk else 1.0) * 15  # max 15
    score = min(100.0, max(0.0, score))

    notes = (
        f"HIA={hia_pct:.1f}%, BA={ba_pct:.1f}%, "
        f"Vss={vss:.2f} L/kg, t½={t_half:.1f} h, "
        f"score={score:.1f}"
    )

    return ADMETProfile(
        compound_name=compound_name,
        smiles=smiles,
        mw=mw,
        logP=logP,
        hia_pct=hia_pct,
        oral_bioavailability_pct=ba_pct,
        vss_L_per_kg=vss,
        fup=fup,
        bbb_penetration=bbb,
        cyp3a4_substrate=cyp3a4_sub,
        cyp2d6_substrate=cyp2d6_sub,
        t_half_h=t_half,
        cl_total_L_per_h_per_kg=cl_est,
        fe_renal=fe_renal,
        herg_risk=herg_risk,
        ames_risk=ames_risk,
        hepatotoxicity_risk=hepatotox_risk,
        lipinski_pass=lipinski,
        pains_flag=pains,
        composite_score=score,
        notes=notes,
    )


def compare_admet_profiles(compounds: list[dict]) -> list[ADMETProfile]:
    """Predict ADMET profiles for multiple compounds and rank by composite score.

    Parameters
    ----------
    compounds : list[dict]
        Each dict should contain at minimum: "name", "smiles", "mw", "logP".
        Optional keys: "psa", "n_hbd", "n_hba", "pka".

    Returns
    -------
    list[ADMETProfile]
        Profiles sorted by composite_score descending.
    """
    profiles: list[ADMETProfile] = []
    for c in compounds:
        profile = predict_admet_profile(
            compound_name=c["name"],
            smiles=c["smiles"],
            mw=c["mw"],
            logP=c["logP"],
            psa=c.get("psa", 90.0),
            n_hbd=c.get("n_hbd", 2),
            n_hba=c.get("n_hba", 4),
            pka=c.get("pka", 7.0),
        )
        profiles.append(profile)
    return sorted(profiles, key=lambda p: p.composite_score, reverse=True)


__all__ = [
    "ADMETProfile",
    "predict_admet_profile",
    "compare_admet_profiles",
]
