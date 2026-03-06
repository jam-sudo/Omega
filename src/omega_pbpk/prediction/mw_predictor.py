"""Molecular weight and formula estimation from atom counts."""

from __future__ import annotations

from dataclasses import dataclass

# Atomic weights (g/mol)
_ATOMIC_WEIGHTS = {
    "C": 12.011,
    "H": 1.008,
    "N": 14.007,
    "O": 15.999,
    "S": 32.06,
    "P": 30.974,
    "F": 18.998,
    "Cl": 35.45,
    "Br": 79.904,
    "I": 126.904,
}

# Average H count added per heavy atom (rule of thumb for organic molecules)
_H_PER_HEAVY = {
    "C_sp3": 2.0,
    "C_sp2": 1.0,
    "C_sp": 0.0,
    "N_amine": 1.0,
    "N_amide": 1.0,
    "N_aromatic": 0.0,
    "O_hydroxyl": 1.0,
    "O_ether": 0.0,
    "S_thiol": 1.0,
    "S_thioether": 0.0,
}


@dataclass(frozen=True)
class MWResult:
    drug_name: str
    molecular_formula: str
    MW_g_mol: float
    n_heavy_atoms: int
    n_H: int
    n_C: int
    n_N: int
    n_O: int
    n_S: int
    n_halogens: int
    monoisotopic_MW: float
    lipinski_mw_ok: bool  # MW < 500
    drug_like: bool  # Lipinski-like check


def predict_mw(
    drug_name: str,
    n_C: int,
    n_N: int = 0,
    n_O: int = 0,
    n_S: int = 0,
    n_P: int = 0,
    n_F: int = 0,
    n_Cl: int = 0,
    n_Br: int = 0,
    n_I: int = 0,
    n_aromatic_rings: int = 0,
    n_sp3_carbons: int | None = None,
    explicit_H: int | None = None,
) -> MWResult:
    """Estimate molecular weight and formula from atom counts.

    Hydrogen count is estimated from saturation rules if not provided.

    Parameters
    ----------
    n_sp3_carbons : int or None
        Number of sp3 carbons. If None, estimate from ring count.
    explicit_H : int or None
        Override H count if known.
    """
    # Estimate H count using valence rules
    if explicit_H is not None:
        n_H = explicit_H
    else:
        # Formula for acyclic: H = 2C + 2 + N - (halogens)
        n_halogens = n_F + n_Cl + n_Br + n_I
        # Rings reduce H by 2 each
        h_base = 2 * n_C + 2 + n_N - n_halogens - 2 * n_aromatic_rings
        # O and S don't change H count in H formula
        # Amines add H: already counted via N
        n_H = max(h_base, 0)

    n_halogens = n_F + n_Cl + n_Br + n_I
    n_heavy = n_C + n_N + n_O + n_S + n_P + n_halogens

    # Average MW calculation
    mw = (
        n_C * _ATOMIC_WEIGHTS["C"]
        + n_H * _ATOMIC_WEIGHTS["H"]
        + n_N * _ATOMIC_WEIGHTS["N"]
        + n_O * _ATOMIC_WEIGHTS["O"]
        + n_S * _ATOMIC_WEIGHTS["S"]
        + n_P * _ATOMIC_WEIGHTS["P"]
        + n_F * _ATOMIC_WEIGHTS["F"]
        + n_Cl * _ATOMIC_WEIGHTS["Cl"]
        + n_Br * _ATOMIC_WEIGHTS["Br"]
        + n_I * _ATOMIC_WEIGHTS["I"]
    )

    # Monoisotopic MW (use most abundant isotopes, already close to standard)
    mono_mw = mw  # simplified (would differ if heavy atoms)

    # Molecular formula string
    parts = []
    for sym, count in [
        ("C", n_C),
        ("H", n_H),
        ("N", n_N),
        ("O", n_O),
        ("S", n_S),
        ("P", n_P),
        ("F", n_F),
        ("Cl", n_Cl),
        ("Br", n_Br),
        ("I", n_I),
    ]:
        if count > 0:
            parts.append(f"{sym}{count}" if count > 1 else sym)
    formula = "".join(parts)

    lipinski_ok = mw < 500.0
    # Drug-like: MW 100-700, has C, has at least some heteroatoms
    drug_like = (100 <= mw <= 700) and (n_C >= 3) and ((n_N + n_O) >= 1)

    return MWResult(
        drug_name=drug_name,
        molecular_formula=formula,
        MW_g_mol=float(mw),
        n_heavy_atoms=n_heavy,
        n_H=n_H,
        n_C=n_C,
        n_N=n_N,
        n_O=n_O,
        n_S=n_S,
        n_halogens=n_halogens,
        monoisotopic_MW=float(mono_mw),
        lipinski_mw_ok=lipinski_ok,
        drug_like=drug_like,
    )


def screen_mw(compounds: list[dict]) -> list[MWResult]:
    """Calculate MW for a list of compounds.

    Each dict: {drug_name, n_C, n_N, n_O, ...}
    """
    results = []
    for c in compounds:
        r = predict_mw(
            drug_name=c.get("drug_name", "unknown"),
            n_C=c.get("n_C", 10),
            n_N=c.get("n_N", 0),
            n_O=c.get("n_O", 0),
            n_S=c.get("n_S", 0),
            n_P=c.get("n_P", 0),
            n_F=c.get("n_F", 0),
            n_Cl=c.get("n_Cl", 0),
            n_Br=c.get("n_Br", 0),
            n_I=c.get("n_I", 0),
            n_aromatic_rings=c.get("n_aromatic_rings", 0),
            explicit_H=c.get("explicit_H"),
        )
        results.append(r)
    return results


__all__ = [
    "MWResult",
    "predict_mw",
    "screen_mw",
]
