"""
Phase 418 — TPSA Calculator: Topological Polar Surface Area prediction.
"""

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TPSAResult:
    drug_name: str
    n_h_donors: int
    n_h_acceptors: int
    n_aromatic_rings: int
    mw_Da: float
    tpsa_angstrom2: float
    logp_estimated: float
    oral_absorption_class: str
    cns_penetration_class: str
    bbb_score: float
    gi_absorption_pct: float
    ro5_violations: int
    drug_likeness_score: float
    notes: str


def calculate_tpsa(
    drug_name: str,
    n_h_donors: int,
    n_h_acceptors: int,
    n_aromatic_rings: int,
    mw_Da: float,
    has_amide_bonds: int = 0,
    has_phosphate: bool = False,
) -> TPSAResult:
    """Calculate TPSA and related druglikeness descriptors.

    Parameters
    ----------
    drug_name : str
        Name of the compound.
    n_h_donors : int
        Number of H-bond donors (NH, OH groups).
    n_h_acceptors : int
        Number of H-bond acceptors (N, O atoms).
    n_aromatic_rings : int
        Number of aromatic rings.
    mw_Da : float
        Molecular weight in Daltons.
    has_amide_bonds : int
        Number of amide bonds (default 0).
    has_phosphate : bool
        Whether a phosphate group is present (default False).

    Returns
    -------
    TPSAResult
    """
    # Validation
    if mw_Da <= 0:
        raise ValueError(f"mw_Da must be > 0, got {mw_Da}")
    if n_h_donors < 0:
        raise ValueError(f"n_h_donors must be >= 0, got {n_h_donors}")
    if n_h_acceptors < 0:
        raise ValueError(f"n_h_acceptors must be >= 0, got {n_h_acceptors}")
    if n_aromatic_rings < 0:
        raise ValueError(f"n_aromatic_rings must be >= 0, got {n_aromatic_rings}")

    # TPSA (simplified Ertl 2000)
    tpsa = 20.23 * n_h_donors + 26.02 * n_h_acceptors + 12.36 * has_amide_bonds
    if has_phosphate:
        tpsa += 40.5
    tpsa = max(0.0, tpsa)

    # logP estimation
    logp = 0.4 * n_aromatic_rings - 0.1 * n_h_donors - 0.1 * n_h_acceptors + mw_Da * 0.01 - 1.0
    logp = max(-5.0, min(10.0, logp))

    # Oral absorption class
    if tpsa < 60.0:
        oral_absorption_class = "excellent"
    elif tpsa < 90.0:
        oral_absorption_class = "good"
    elif tpsa < 140.0:
        oral_absorption_class = "moderate"
    else:
        oral_absorption_class = "poor"

    # CNS penetration class
    if tpsa < 60.0:
        cns_penetration_class = "likely"
    elif tpsa < 90.0:
        cns_penetration_class = "possible"
    else:
        cns_penetration_class = "unlikely"

    # BBB score
    if tpsa < 140.0:
        bbb_score = max(0.0, min(1.0, 1.0 - tpsa / 140.0))
    else:
        bbb_score = 0.0

    # GI absorption
    if tpsa < 200.0:
        gi_absorption_pct = max(0.0, min(100.0, 100.0 - tpsa * 0.5))
    else:
        gi_absorption_pct = 0.0

    # Lipinski RO5 violations
    ro5_violations = 0
    if mw_Da > 500:
        ro5_violations += 1
    if logp > 5:
        ro5_violations += 1
    if n_h_donors > 5:
        ro5_violations += 1
    if n_h_acceptors > 10:
        ro5_violations += 1

    # Drug likeness score
    drug_likeness_score = (
        100.0
        - 15.0 * ro5_violations
        - max(0.0, (tpsa - 130.0) * 0.3)
        - max(0.0, (mw_Da - 500.0) * 0.05)
    )
    drug_likeness_score = max(0.0, min(100.0, drug_likeness_score))

    # Notes
    notes = (
        f"TPSA = {tpsa:.1f} A2 ({oral_absorption_class} oral absorption). "
        f"CNS penetration: {cns_penetration_class}. "
        f"RO5 violations: {ro5_violations}."
    )

    return TPSAResult(
        drug_name=drug_name,
        n_h_donors=n_h_donors,
        n_h_acceptors=n_h_acceptors,
        n_aromatic_rings=n_aromatic_rings,
        mw_Da=mw_Da,
        tpsa_angstrom2=tpsa,
        logp_estimated=logp,
        oral_absorption_class=oral_absorption_class,
        cns_penetration_class=cns_penetration_class,
        bbb_score=bbb_score,
        gi_absorption_pct=gi_absorption_pct,
        ro5_violations=ro5_violations,
        drug_likeness_score=drug_likeness_score,
        notes=notes,
    )


def screen_compounds(compounds_list: list[dict[str, Any]]) -> list[TPSAResult]:
    """Screen a list of compounds and rank by drug likeness.

    Parameters
    ----------
    compounds_list : list of dict
        Each dict has keys matching calculate_tpsa params.

    Returns
    -------
    list of TPSAResult sorted by drug_likeness_score descending.
    """
    results = []
    for compound in compounds_list:
        result = calculate_tpsa(
            drug_name=compound["drug_name"],
            n_h_donors=compound["n_h_donors"],
            n_h_acceptors=compound["n_h_acceptors"],
            n_aromatic_rings=compound["n_aromatic_rings"],
            mw_Da=compound["mw_Da"],
            has_amide_bonds=compound.get("has_amide_bonds", 0),
            has_phosphate=compound.get("has_phosphate", False),
        )
        results.append(result)

    results.sort(key=lambda r: r.drug_likeness_score, reverse=True)
    return results
