"""
Drug-likeness scoring beyond Lipinski's Rule of Five.
Implements QED (Bickerton 2012, simplified) and MPO (multiparameter optimization) scoring.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class DrugLikenessResult:
    compound_name: str
    mw_da: float
    logp: float
    hbd: int  # H-bond donors
    hba: int  # H-bond acceptors
    psa_A2: float
    n_rotatable_bonds: int
    n_aromatic_rings: int
    lipinski_passes: bool
    lipinski_violations: int
    veber_passes: bool  # psa <= 140 and rotbonds <= 10
    qed_score: float  # 0-1 (Bickerton 2012 simplified)
    mpo_score: float  # multiparameter optimization score 0-6
    drug_likeness_class: str  # "excellent", "good", "moderate", "poor"
    fragment_complexity: float  # MW/n_heavy_atoms proxy
    notes: str


def _gaussian(x: float, mu: float, sigma: float) -> float:
    """Gaussian desirability function."""
    return math.exp(-(((x - mu) / sigma) ** 2))


def _qed_score(
    mw: float,
    logp: float,
    hbd: int,
    hba: int,
    psa: float,
    rotb: int,
    rings: int,
) -> float:
    """
    Simplified QED (Bickerton 2012): geometric mean of 8 desirability functions.
    """
    d_mw = _gaussian(mw, 339.0, 106.0)
    d_logp = _gaussian(logp, 2.4, 1.7)
    d_hbd = _gaussian(hbd, 0.5, 0.7)
    d_hba = _gaussian(hba, 2.7, 2.0)
    d_psa = _gaussian(psa, 57.0, 42.0)
    d_rotb = _gaussian(rotb, 3.2, 2.5)
    d_rings = _gaussian(rings, 1.5, 1.5)
    d_alerts = 0.9  # no structural alerts assumed

    product = d_mw * d_logp * d_hbd * d_hba * d_psa * d_rotb * d_rings * d_alerts
    if product <= 0.0:
        return 0.0
    return product ** (1.0 / 8.0)


def _mpo_score(
    mw: float,
    logp: float,
    hbd: int,
    psa: float,
    pka_basic: float | None,
    rings: int,
) -> float:
    """
    MPO score: 6 parameters each scored 0-1, summed to give 0-6.
    """
    # MW: optimal 360-500 → 1, 500-550 → 0.5, >550 → 0; <360 partial credit
    if 360.0 <= mw <= 500.0:
        s_mw = 1.0
    elif 500.0 < mw <= 550.0:
        s_mw = 0.5
    elif mw > 550.0:
        s_mw = 0.0
    elif 200.0 <= mw < 360.0:
        s_mw = 0.5 + 0.5 * (mw - 200.0) / (360.0 - 200.0)
    else:
        s_mw = 0.0

    # LogP: 1-3 → 1, 0-1 or 3-5 → 0.5, <0 or >5 → 0
    if 1.0 <= logp <= 3.0:
        s_logp = 1.0
    elif (0.0 <= logp < 1.0) or (3.0 < logp <= 5.0):
        s_logp = 0.5
    else:
        s_logp = 0.0

    # HBD: 0 → 1, 1-2 → 0.8, >2 → 0.3
    if hbd == 0:
        s_hbd = 1.0
    elif 1 <= hbd <= 2:
        s_hbd = 0.8
    else:
        s_hbd = 0.3

    # PSA: 40-90 → 1, 90-120 → 0.5, 120-140 → 0.25, >140 → 0; <40 → 0.5
    if 40.0 <= psa <= 90.0:
        s_psa = 1.0
    elif 90.0 < psa <= 120.0:
        s_psa = 0.5
    elif 120.0 < psa <= 140.0:
        s_psa = 0.25
    elif psa > 140.0:
        s_psa = 0.0
    else:
        s_psa = 0.5

    # pKa (basic): 7-10 → 1; not provided or outside → 0.5
    if pka_basic is not None:
        if 7.0 <= pka_basic <= 10.0:
            s_pka = 1.0
        elif 5.0 <= pka_basic < 7.0 or 10.0 < pka_basic <= 12.0:
            s_pka = 0.5
        else:
            s_pka = 0.0
    else:
        s_pka = 0.5  # unknown; neutral assumption

    # Aromatic rings: <=3 → 1, 4 → 0.5, >4 → 0
    if rings <= 3:
        s_rings = 1.0
    elif rings == 4:
        s_rings = 0.5
    else:
        s_rings = 0.0

    return s_mw + s_logp + s_hbd + s_psa + s_pka + s_rings


def score_drug_likeness(
    compound_name: str,
    mw_da: float,
    logp: float,
    hbd: int,
    hba: int,
    psa_A2: float,
    n_rotatable_bonds: int = 5,
    n_aromatic_rings: int = 2,
    n_heavy_atoms: int = 25,
    pka_basic: float | None = None,
) -> DrugLikenessResult:
    """
    Score drug-likeness using Lipinski RO5, Veber, QED, and MPO criteria.

    Parameters
    ----------
    compound_name : str
    mw_da : float
        Molecular weight in Da. Must be > 0.
    logp : float
        Calculated logP.
    hbd : int
        Number of H-bond donors. Must be >= 0.
    hba : int
        Number of H-bond acceptors. Must be >= 0.
    psa_A2 : float
        Polar surface area in Angstrom^2. Must be >= 0.
    n_rotatable_bonds : int
        Number of rotatable bonds.
    n_aromatic_rings : int
        Number of aromatic rings.
    n_heavy_atoms : int
        Number of heavy atoms (for complexity proxy).
    pka_basic : float or None
        Basic pKa value (optional; used in MPO scoring).

    Returns
    -------
    DrugLikenessResult
    """
    # --- Validation ---
    if mw_da <= 0:
        raise ValueError(f"mw_da must be > 0, got {mw_da}")
    if hbd < 0:
        raise ValueError(f"hbd must be >= 0, got {hbd}")
    if hba < 0:
        raise ValueError(f"hba must be >= 0, got {hba}")
    if psa_A2 < 0:
        raise ValueError(f"psa_A2 must be >= 0, got {psa_A2}")
    if n_rotatable_bonds < 0:
        raise ValueError(f"n_rotatable_bonds must be >= 0, got {n_rotatable_bonds}")
    if n_aromatic_rings < 0:
        raise ValueError(f"n_aromatic_rings must be >= 0, got {n_aromatic_rings}")
    if n_heavy_atoms <= 0:
        raise ValueError(f"n_heavy_atoms must be > 0, got {n_heavy_atoms}")

    # --- Lipinski RO5 ---
    violations = 0
    if mw_da > 500.0:
        violations += 1
    if logp > 5.0:
        violations += 1
    if hbd > 5:
        violations += 1
    if hba > 10:
        violations += 1
    lipinski_passes = violations == 0

    # --- Veber ---
    veber_passes = psa_A2 <= 140.0 and n_rotatable_bonds <= 10

    # --- QED ---
    qed = _qed_score(mw_da, logp, hbd, hba, psa_A2, n_rotatable_bonds, n_aromatic_rings)
    # Clamp to [0, 1]
    qed = max(0.0, min(1.0, qed))

    # --- MPO ---
    mpo = _mpo_score(mw_da, logp, hbd, psa_A2, pka_basic, n_aromatic_rings)
    # Clamp to [0, 6]
    mpo = max(0.0, min(6.0, mpo))

    # --- Drug-likeness class ---
    if qed > 0.7:
        dl_class = "excellent"
    elif qed >= 0.5:
        dl_class = "good"
    elif qed >= 0.3:
        dl_class = "moderate"
    else:
        dl_class = "poor"

    # --- Fragment complexity ---
    fragment_complexity = mw_da / n_heavy_atoms

    # --- Notes ---
    notes_parts = []
    if lipinski_passes:
        notes_parts.append("passes Lipinski RO5")
    else:
        notes_parts.append(f"Lipinski violations={violations}")
    if veber_passes:
        notes_parts.append("passes Veber")
    else:
        notes_parts.append("fails Veber")
    notes_parts.append(f"QED={qed:.3f}")
    notes_parts.append(f"MPO={mpo:.2f}")
    notes = "; ".join(notes_parts)

    return DrugLikenessResult(
        compound_name=compound_name,
        mw_da=mw_da,
        logp=logp,
        hbd=hbd,
        hba=hba,
        psa_A2=psa_A2,
        n_rotatable_bonds=n_rotatable_bonds,
        n_aromatic_rings=n_aromatic_rings,
        lipinski_passes=lipinski_passes,
        lipinski_violations=violations,
        veber_passes=veber_passes,
        qed_score=qed,
        mpo_score=mpo,
        drug_likeness_class=dl_class,
        fragment_complexity=fragment_complexity,
        notes=notes,
    )


def screen_drug_likeness(compounds: list[dict]) -> list[DrugLikenessResult]:
    """
    Score a list of compounds and return results sorted by QED score descending.

    Each dict must contain keys: compound_name, mw_da, logp, hbd, hba, psa_A2.
    Optional keys: n_rotatable_bonds, n_aromatic_rings, n_heavy_atoms, pka_basic.

    Parameters
    ----------
    compounds : list of dict

    Returns
    -------
    list of DrugLikenessResult sorted by qed_score descending
    """
    results = []
    for c in compounds:
        result = score_drug_likeness(
            compound_name=c["compound_name"],
            mw_da=c["mw_da"],
            logp=c["logp"],
            hbd=c["hbd"],
            hba=c["hba"],
            psa_A2=c["psa_A2"],
            n_rotatable_bonds=c.get("n_rotatable_bonds", 5),
            n_aromatic_rings=c.get("n_aromatic_rings", 2),
            n_heavy_atoms=c.get("n_heavy_atoms", 25),
            pka_basic=c.get("pka_basic", None),
        )
        results.append(result)
    results.sort(key=lambda r: r.qed_score, reverse=True)
    return results
