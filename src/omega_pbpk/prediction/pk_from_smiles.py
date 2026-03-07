"""Plasma concentration prediction from SMILES structure.

Predict key PK parameters directly from SMILES using structure-property
relationships, then simulate a PK profile using a 1-compartment model.

No RDKit required — uses simple string-based SMILES parsing.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PKFromSMILESResult:
    """Complete PK prediction result from a SMILES string."""

    smiles: str
    compound_name: str

    # Predicted physicochemical properties
    pred_logP: float
    pred_mw: float
    pred_hbd: int
    pred_hba: int
    pred_psa: float
    pred_rotatable_bonds: int

    # Predicted ADME parameters
    pred_fu_plasma: float
    pred_vd_L: float
    pred_cl_L_per_h: float
    pred_f_oral: float
    pred_t_half_h: float
    pred_bcs_class: str

    # Simulated PK (single oral dose)
    dose_mg: float
    times_h: list
    c_plasma_mg_L: list
    auc_mg_h_per_L: float
    cmax_mg_L: float
    tmax_h: float

    # Drug-likeness
    lipinski_passes: bool
    notes: str


# ── SMILES parsing helpers ────────────────────────────────────────────────────

_ATOM_MW = {
    "C": 12.0,
    "N": 14.0,
    "O": 16.0,
    "S": 32.0,
    "F": 19.0,
    "P": 31.0,
    # Two-character atoms handled separately
}
_HALOGEN_MW = {"Cl": 35.5, "Br": 80.0, "I": 127.0}

# Average H count added per heavy atom (rough valence estimate)
_ATOM_H_EST = {
    "C": 2.0,  # average sp3 CH2
    "N": 1.0,
    "O": 0.5,
    "S": 0.0,
    "F": 0.0,
    "P": 0.0,
}

# logP fragment contributions (simplified Wildman-Crippen)
_LOGP_CONTRIB = {
    "C": 0.5,
    "N": -1.0,
    "O": -0.5,
    "S": 1.0,
    "F": 0.25,
    "Cl": 0.75,
    "Br": 1.5,
    "I": 1.8,
    "P": -0.5,
}


def parse_smiles_properties(smiles: str) -> dict:
    """Parse physicochemical properties from a SMILES string.

    Uses simple string analysis — no RDKit required.

    Parameters
    ----------
    smiles : str
        SMILES string for the compound.

    Returns
    -------
    dict
        Predicted physicochemical properties including mw, logP, hbd, hba,
        psa, rotatable_bonds, n_rings.
    """
    if not smiles or not smiles.strip():
        raise ValueError("SMILES string must be non-empty")

    s = smiles.strip()

    # ── Count atoms ──────────────────────────────────────────────────────────
    # Handle two-character atoms first (Cl, Br, handled as "[Cl]" or bare "Cl")
    s_work = s

    # Count Cl, Br, I (two-char atoms)
    n_cl = len(re.findall(r"Cl", s_work))
    n_br = len(re.findall(r"Br", s_work))
    n_i_bracket = len(re.findall(r"\[I\]", s_work))

    # Remove two-char atoms and brackets to count single-char atoms
    stripped = re.sub(r"Cl|Br", "X", s_work)  # replace with placeholder
    stripped = re.sub(r"\[.*?\]", lambda m: _expand_bracket(m.group(0)), stripped)

    # Count single-character heavy atoms (uppercase letters)
    n_c = stripped.count("C") + stripped.count("c")  # aromatic c counts
    n_n = stripped.count("N") + stripped.count("n")
    n_o = stripped.count("O") + stripped.count("o")
    n_s = stripped.count("S") + stripped.count("s")
    n_f = stripped.count("F")
    n_p = stripped.count("P")

    # ── Molecular weight estimate ─────────────────────────────────────────────
    mw = (
        n_c * 12.0
        + n_n * 14.0
        + n_o * 16.0
        + n_s * 32.0
        + n_f * 19.0
        + n_p * 31.0
        + n_cl * 35.5
        + n_br * 80.0
        + n_i_bracket * 127.0
    )
    # Add approximate H mass
    n_heavy = n_c + n_n + n_o + n_s + n_f + n_p + n_cl + n_br + n_i_bracket
    # Rough H count based on typical valences minus bonds
    n_h_est = max(0, n_c * 2 + n_n * 1 + n_o * 0 - n_c * 0.5)
    # Simple estimate: add ~1 H per 4 heavy atoms
    n_h_est = n_heavy * 0.6
    mw += n_h_est * 1.008
    mw = max(mw, 18.0)

    # ── H-bond donors ─────────────────────────────────────────────────────────
    # [OH], [NH], [NH2], O with explicit H, N with explicit H
    hbd = 0
    hbd += len(re.findall(r"\[OH\]|\[NH\]|\[NH2\]|\[NH3\]", s))
    # Bare OH patterns in SMILES: O followed by H (not in brackets)
    hbd += len(re.findall(r"(?<!\[)O(?!H)(?=\)|\[|$|[A-Z]|[0-9]|\()", s))
    # Count capital N not in brackets (aliphatic NH)
    # A simpler heuristic: count N and O atoms / 3
    hbd = max(0, hbd)
    # Fallback: rough count from N+O
    if hbd == 0:
        hbd = max(0, n_n // 2 + (1 if n_o > 0 else 0))

    # Ensure reasonable bounds
    hbd = min(hbd, n_n + n_o)

    # ── H-bond acceptors ──────────────────────────────────────────────────────
    hba = n_n + n_o  # rough: all N and O atoms

    # ── PSA estimate ──────────────────────────────────────────────────────────
    psa = 25.0 * hba  # rough correlation

    # ── Rotatable bonds estimate ──────────────────────────────────────────────
    # Count single bonds between non-ring heavy atoms
    # Rough proxy: count uppercase C-C single bond patterns
    # Use chain length minus ring count as approximation
    n_rings = _count_rings(s)
    rotatable_bonds = max(0, n_c - n_rings * 3 - 1)
    rotatable_bonds = min(rotatable_bonds, n_c)

    # ── logP estimate ─────────────────────────────────────────────────────────
    logP = (
        n_c * _LOGP_CONTRIB["C"]
        + n_n * _LOGP_CONTRIB["N"]
        + n_o * _LOGP_CONTRIB["O"]
        + n_s * _LOGP_CONTRIB["S"]
        + n_f * _LOGP_CONTRIB["F"]
        + n_cl * _LOGP_CONTRIB["Cl"]
        + n_br * _LOGP_CONTRIB["Br"]
        + n_i_bracket * _LOGP_CONTRIB["I"]
        + n_p * _LOGP_CONTRIB["P"]
    )
    # Aromatic ring correction
    logP -= 0.5 * n_rings

    return {
        "mw": round(mw, 1),
        "logP": round(logP, 2),
        "hbd": hbd,
        "hba": hba,
        "psa": round(psa, 1),
        "rotatable_bonds": rotatable_bonds,
        "n_rings": n_rings,
        "n_c": n_c,
        "n_n": n_n,
        "n_o": n_o,
        "n_s": n_s,
        "n_f": n_f,
        "n_cl": n_cl,
        "n_br": n_br,
    }


def _expand_bracket(bracket_atom: str) -> str:
    """Extract the element symbol from a bracket atom like [NH2] → N."""
    inner = bracket_atom[1:-1]
    # Remove charge, isotope, H count
    inner = re.sub(r"[0-9@+\-Hh]", "", inner)
    return inner


def _count_rings(smiles: str) -> int:
    """Count number of ring closure digits in SMILES (rough ring count)."""
    # Ring closures: digits that appear exactly twice
    # Find all ring closure digits/labels
    digits = re.findall(r"(?<![%])\d|%\d{2}", smiles)
    # Each ring closure digit appears twice (open + close) → count unique ones
    from collections import Counter

    cnt = Counter(digits)
    # Number of rings = number of digits that appear >=2 times
    return sum(1 for v in cnt.values() if v >= 2)


def _predict_adme(props: dict) -> dict:
    """Predict ADME parameters from physicochemical properties."""
    logP = props["logP"]
    mw = props["mw"]
    hbd = props["hbd"]
    hba = props["hba"]
    psa = props["psa"]

    # Lipinski violations
    violations = 0
    if mw > 500:
        violations += 1
    if logP > 5:
        violations += 1
    if hbd > 5:
        violations += 1
    if hba > 10:
        violations += 1

    # fu_plasma (Lombardo 2001 correlation)
    fu_plasma = 1.0 / (1.0 + 10 ** (0.9 * logP - 1.5))
    fu_plasma = max(0.001, min(1.0, fu_plasma))

    # Vd_ss (L) rough QSPR
    vd_L = 0.5 + 0.4 * logP + 0.05 * mw / 100.0
    vd_L = max(0.1, vd_L)

    # CLsys (L/h) rough logP-based estimate
    cl_L_per_h = min(300.0, 0.5 * max(logP, 0.0) * 10.0 + 2.0)
    cl_L_per_h = max(0.1, cl_L_per_h)

    # Oral bioavailability
    if violations > 1:
        f_oral = 0.2
    elif psa > 140:
        f_oral = 0.3
    elif logP > 5:
        f_oral = 0.4
    else:
        f_oral = 0.7

    # BCS class: based on logP (solubility) and PSA/HBD (permeability)
    high_solubility = logP < 2.0  # log-based proxy
    high_permeability = psa < 90.0 and hbd <= 3
    if high_solubility and high_permeability:
        bcs_class = "I"
    elif not high_solubility and high_permeability:
        bcs_class = "II"
    elif high_solubility and not high_permeability:
        bcs_class = "III"
    else:
        bcs_class = "IV"

    # Half-life
    ke = cl_L_per_h / vd_L
    t_half_h = 0.693 / ke

    lipinski_passes = violations == 0

    return {
        "fu_plasma": fu_plasma,
        "vd_L": vd_L,
        "cl_L_per_h": cl_L_per_h,
        "f_oral": f_oral,
        "t_half_h": t_half_h,
        "bcs_class": bcs_class,
        "lipinski_passes": lipinski_passes,
        "violations": violations,
        "ke": ke,
    }


def _simulate_oral_pk(
    dose_mg: float,
    vd_L: float,
    cl_L_per_h: float,
    f_oral: float,
    ka: float = 1.5,
    t_end_h: float = 24.0,
    dt: float = 0.1,
) -> tuple[list, list, float, float, float]:
    """Simulate 1-compartment oral PK by forward Euler.

    Returns
    -------
    tuple
        (times_h, c_plasma_mg_L, auc, cmax, tmax)
    """
    ke = cl_L_per_h / vd_L
    dose_absorbed = dose_mg * f_oral

    n_steps = int(t_end_h / dt) + 1
    times = [i * dt for i in range(n_steps)]

    # State: [Agut (mg), Acentral (mg)]
    agut = dose_absorbed
    acentral = 0.0

    concs = []
    for _ in times:
        c = acentral / vd_L
        concs.append(max(c, 0.0))
        # Forward Euler
        d_agut = -ka * agut
        d_acentral = ka * agut - ke * acentral
        agut += d_agut * dt
        acentral += d_acentral * dt
        agut = max(agut, 0.0)
        acentral = max(acentral, 0.0)

    # NCA metrics
    auc = float(np.trapz(concs, times))
    cmax = float(max(concs))
    tmax_idx = concs.index(cmax)
    tmax = times[tmax_idx]

    return times, concs, auc, cmax, tmax


def predict_pk_from_smiles(
    smiles: str,
    compound_name: str = "Compound",
    dose_mg: float = 100.0,
    t_end_h: float = 24.0,
    route: str = "oral",
) -> PKFromSMILESResult:
    """Predict PK parameters and simulate a PK profile from a SMILES string.

    Parameters
    ----------
    smiles : str
        SMILES string for the compound.
    compound_name : str
        Name for the compound.
    dose_mg : float
        Dose in mg.
    t_end_h : float
        Simulation end time in hours.
    route : str
        Route of administration (currently only "oral" implemented).

    Returns
    -------
    PKFromSMILESResult
        Frozen dataclass with predicted properties, ADME parameters, and
        simulated PK profile.

    Raises
    ------
    ValueError
        If smiles is empty or dose_mg is non-positive.
    """
    if not smiles or not smiles.strip():
        raise ValueError("smiles must be a non-empty string")
    if dose_mg <= 0:
        raise ValueError(f"dose_mg must be positive, got {dose_mg}")

    # Parse physicochemical properties
    props = parse_smiles_properties(smiles)

    # Predict ADME
    adme = _predict_adme(props)

    # Simulate PK
    times, concs, auc, cmax, tmax = _simulate_oral_pk(
        dose_mg=dose_mg,
        vd_L=adme["vd_L"],
        cl_L_per_h=adme["cl_L_per_h"],
        f_oral=adme["f_oral"],
        t_end_h=t_end_h,
    )

    notes = (
        f"Lipinski violations: {adme['violations']}; "
        f"BCS class: {adme['bcs_class']}; "
        f"logP: {props['logP']:.2f}; "
        f"MW: {props['mw']:.1f}"
    )

    return PKFromSMILESResult(
        smiles=smiles,
        compound_name=compound_name,
        pred_logP=props["logP"],
        pred_mw=props["mw"],
        pred_hbd=props["hbd"],
        pred_hba=props["hba"],
        pred_psa=props["psa"],
        pred_rotatable_bonds=props["rotatable_bonds"],
        pred_fu_plasma=adme["fu_plasma"],
        pred_vd_L=adme["vd_L"],
        pred_cl_L_per_h=adme["cl_L_per_h"],
        pred_f_oral=adme["f_oral"],
        pred_t_half_h=adme["t_half_h"],
        pred_bcs_class=adme["bcs_class"],
        dose_mg=dose_mg,
        times_h=times,
        c_plasma_mg_L=concs,
        auc_mg_h_per_L=auc,
        cmax_mg_L=cmax,
        tmax_h=tmax,
        lipinski_passes=adme["lipinski_passes"],
        notes=notes,
    )


def batch_predict_pk(
    compounds_list: list[dict],
) -> list[PKFromSMILESResult]:
    """Predict PK for a list of compounds.

    Parameters
    ----------
    compounds_list : list[dict]
        Each dict must have "smiles" (str) and optionally "name" (str)
        and "dose_mg" (float).

    Returns
    -------
    list[PKFromSMILESResult]
        Results sorted by pred_cl_L_per_h ascending (slowest CL first,
        i.e., longest half-life first).
    """
    results = []
    for c in compounds_list:
        smiles = c.get("smiles", "")
        name = c.get("name", c.get("compound_name", "Compound"))
        dose = float(c.get("dose_mg", 100.0))
        r = predict_pk_from_smiles(smiles=smiles, compound_name=name, dose_mg=dose)
        results.append(r)

    # Sort by CL ascending (slowest CL = longest t_half first)
    results.sort(key=lambda x: x.pred_cl_L_per_h)
    return results


__all__ = [
    "PKFromSMILESResult",
    "parse_smiles_properties",
    "predict_pk_from_smiles",
    "batch_predict_pk",
]
