"""Pure-Python SMILES featurizer — no external dependencies required.

Parses a SMILES string to extract 20 molecular descriptors for ADME
prediction. Works as a drop-in replacement for RDKitFeaturizer when
RDKit is not installed.

Features (index, name, description):
  0  mw            Molecular weight (g/mol, ~8 % H correction)
  1  heavy_atoms   Number of non-hydrogen atoms
  2  n_C           Aliphatic carbon count
  3  n_C_arom      Aromatic carbon count
  4  n_N           Aliphatic nitrogen count
  5  n_N_arom      Aromatic nitrogen count
  6  n_O           Oxygen count (all)
  7  n_S           Sulfur count (all)
  8  n_F           Fluorine count
  9  n_Cl          Chlorine count
 10  n_Br          Bromine count
 11  n_hbd         H-bond donor estimate
 12  n_hba         H-bond acceptor count (N + O + F)
 13  n_rings       Ring count (from ring-closure labels)
 14  n_aromatic    Total aromatic atom count
 15  n_carbonyl    C=O group count
 16  logp_contrib  Atom-contribution logP (Crippen-inspired)
 17  tpsa_approx   Topological PSA approximation (Å²)
 18  frac_sp3      Fraction sp3 carbons (aliphatic / total C)
 19  halogen_score Weighted halogen lipophilicity score
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

# ---------------------------------------------------------------------------
# Per-atom constants
# ---------------------------------------------------------------------------

_ATOM_MW: dict[str, float] = {
    "C": 12.011, "N": 14.007, "O": 15.999, "S": 32.06,
    "F": 18.998, "Cl": 35.45, "Br": 79.904, "I": 126.904,
    "P": 30.974, "Si": 28.086, "B": 10.811, "Se": 78.96,
    "Na": 22.990, "K": 39.098, "Ca": 40.078, "Fe": 55.845,
}

# Simplified Crippen-like atom-type logP increments.
# Keys are (upper_symbol, is_aromatic).
_LOGP_CONTRIB: dict[tuple[str, bool], float] = {
    ("C",  False): 0.20,   # aliphatic C
    ("C",  True):  0.13,   # aromatic C
    ("N",  False): -0.62,  # aliphatic N (amine)
    ("N",  True):  -0.18,  # aromatic N (pyridine-like)
    ("O",  False): -0.44,  # aliphatic O (ether/alcohol)
    ("O",  True):  -0.20,  # aromatic O (furan)
    ("S",  False): 0.19,   # aliphatic S
    ("S",  True):  0.11,   # aromatic S
    ("F",  False): 0.14,
    ("Cl", False): 0.60,
    ("Br", False): 0.82,
    ("I",  False): 1.10,
    ("P",  False): -0.15,
    ("Si", False): 0.20,
    ("B",  False): 0.10,
    ("Se", False): 0.25,
}

# TPSA contribution per polar atom type (Å²); values from Ertl 2000.
_TPSA_CONTRIB: dict[tuple[str, bool], float] = {
    ("N",  False): 26.0,
    ("N",  True):  12.9,
    ("O",  False): 20.2,
    ("O",  True):  13.1,
    ("S",  False): 25.3,
    ("S",  True):  28.2,
    ("P",  False): 40.0,
}

# Single-char organic-subset atoms (uppercase = aliphatic in SMILES).
_ORGANIC_UPPER = frozenset("BCNOPSFIK")
# Aromatic-subset (lowercase).
_ORGANIC_LOWER = frozenset("bcnosp")
# Two-character atoms handled explicitly.
_TWO_CHAR = frozenset({"Cl", "Br", "Si", "Se", "Na", "Mg", "Al",
                        "Ca", "Fe", "Cu", "Zn", "Ge", "As", "Sn"})

# ---------------------------------------------------------------------------
# Public data types
# ---------------------------------------------------------------------------

FEATURE_NAMES: list[str] = [
    "mw", "heavy_atoms", "n_C", "n_C_arom",
    "n_N", "n_N_arom", "n_O", "n_S",
    "n_F", "n_Cl", "n_Br", "n_hbd",
    "n_hba", "n_rings", "n_aromatic", "n_carbonyl",
    "logp_contrib", "tpsa_approx", "frac_sp3", "halogen_score",
]
N_FEATURES = len(FEATURE_NAMES)  # 20


@dataclass(frozen=True)
class SmilesFeatureVector:
    """Container for a 20-descriptor SMILES feature vector."""

    descriptors: NDArray[np.float64]
    feature_names: list[str]


# ---------------------------------------------------------------------------
# Featurizer class
# ---------------------------------------------------------------------------

class SmilesFeaturizer:
    """Extract 20 molecular descriptors from a SMILES string.

    No external libraries required. Handles aromatic atoms, bracket
    atoms (explicit H, charge, isotope), and ring-closure detection.
    """

    FEATURE_NAMES = FEATURE_NAMES
    N_FEATURES = N_FEATURES

    def featurize(self, smiles: str) -> SmilesFeatureVector:
        """Compute 20-feature descriptor vector for a SMILES string.

        Args:
            smiles: A valid SMILES string.

        Returns:
            SmilesFeatureVector with descriptors of shape (20,).
        """
        smiles = smiles.strip()
        atoms = _parse_atoms(smiles)

        # ---- aggregate atom-type counts ----
        by_type: dict[tuple[str, bool], int] = {}
        for sym, is_arom in atoms:
            key = (sym, is_arom)
            by_type[key] = by_type.get(key, 0) + 1

        n_C      = by_type.get(("C", False), 0)
        n_C_arom = by_type.get(("C", True),  0)
        n_N      = by_type.get(("N", False), 0)
        n_N_arom = by_type.get(("N", True),  0)
        n_O      = (by_type.get(("O", False), 0)
                    + by_type.get(("O", True),  0))
        n_S      = (by_type.get(("S", False), 0)
                    + by_type.get(("S", True),  0))
        n_F      = by_type.get(("F", False), 0)
        n_Cl     = by_type.get(("Cl", False), 0)
        n_Br     = by_type.get(("Br", False), 0)
        n_I      = by_type.get(("I",  False), 0)

        heavy_atoms = len(atoms)
        n_aromatic  = sum(cnt for (_, arom), cnt in by_type.items() if arom)

        # ---- derived features ----
        n_hba = n_N + n_N_arom + n_O + n_F
        n_hbd = _count_hbd(smiles)
        n_rings    = _count_rings(smiles)
        n_carbonyl = _count_carbonyl(smiles)

        # Molecular weight (sum of heavy-atom MW × 1.08 for implicit H)
        mw = sum(_ATOM_MW.get(sym, 12.011) for sym, _ in atoms) * 1.08

        # logP via atom-contribution model
        logp_contrib = sum(
            _LOGP_CONTRIB.get((sym, arom), 0.0)
            for sym, arom in atoms
        )

        # TPSA approximation + 10 Å² per HBD for N-H / O-H surface
        tpsa = sum(
            _TPSA_CONTRIB.get((sym, arom), 0.0)
            for sym, arom in atoms
        ) + 10.0 * n_hbd

        # Fraction sp3 C
        total_C = n_C + n_C_arom
        frac_sp3 = n_C / total_C if total_C > 0 else 0.0

        # Weighted halogen score (lipophilicity contribution)
        halogen_score = 0.14 * n_F + 0.60 * n_Cl + 0.82 * n_Br + 1.10 * n_I

        descriptors = np.array([
            mw, heavy_atoms, n_C, n_C_arom,
            n_N, n_N_arom, n_O, n_S,
            n_F, n_Cl, n_Br, n_hbd,
            n_hba, n_rings, n_aromatic, n_carbonyl,
            logp_contrib, tpsa, frac_sp3, halogen_score,
        ], dtype=np.float64)

        return SmilesFeatureVector(
            descriptors=descriptors,
            feature_names=list(FEATURE_NAMES),
        )

    def featurize_batch(self, smiles_list: list[str]) -> NDArray[np.float64]:
        """Featurize a list of SMILES strings.

        Returns:
            Array of shape (N, 20).
        """
        out = np.zeros((len(smiles_list), N_FEATURES), dtype=np.float64)
        for i, smi in enumerate(smiles_list):
            out[i] = self.featurize(smi).descriptors
        return out


# ---------------------------------------------------------------------------
# Internal SMILES parsing helpers
# ---------------------------------------------------------------------------

def _parse_atoms(smiles: str) -> list[tuple[str, bool]]:
    """Return (symbol, is_aromatic) for each heavy atom in SMILES.

    Handles:
    - Bracket atoms: [NH], [OH], [nH], [C@@H], [13C], [N+], …
    - Two-char atoms: Cl, Br, Si, Se, Na, …
    - Organic subset: B, C, N, O, P, S, F, I, K
    - Aromatic subset: b, c, n, o, p, s
    """
    atoms: list[tuple[str, bool]] = []
    i = 0
    n = len(smiles)

    while i < n:
        c = smiles[i]

        if c == "[":
            j = smiles.find("]", i + 1)
            if j == -1:
                i += 1
                continue
            content = smiles[i + 1 : j]
            sym, is_arom = _parse_bracket_atom(content)
            if sym and sym != "H":
                atoms.append((sym, is_arom))
            i = j + 1

        elif i + 1 < n and smiles[i : i + 2] in _TWO_CHAR:
            atoms.append((smiles[i : i + 2], False))
            i += 2

        elif c in _ORGANIC_UPPER:
            atoms.append((c, False))
            i += 1

        elif c in _ORGANIC_LOWER:
            atoms.append((c.upper(), True))
            i += 1

        else:
            i += 1  # bond char, digit, branch, etc.

    return atoms


def _parse_bracket_atom(content: str) -> tuple[str, bool]:
    """Parse bracket atom content → (symbol, is_aromatic).

    Handles isotope prefix, stereochemistry markers, H count, and charge.
    """
    # Strip leading isotope digits
    s = re.sub(r"^\d+", "", content)
    if not s:
        return ("C", False)

    is_arom = s[0].islower() and s[0].isalpha()

    # Extract symbol (one or two characters)
    two = s[:2].title() if len(s) >= 2 else ""
    if two in _TWO_CHAR:
        sym = two
    elif s[0].isalpha():
        sym = s[0].upper()
    else:
        sym = "C"

    return (sym, is_arom)


def _count_rings(smiles: str) -> int:
    """Count the number of rings via ring-closure label counting."""
    # Remove bracket atoms to avoid counting digits inside them
    s = re.sub(r"\[.*?\]", "", smiles)
    # Two-digit ring closures (%nn)
    double = re.findall(r"%\d\d", s)
    s2 = re.sub(r"%\d\d", "", s)
    # Single-digit ring closures
    single = re.findall(r"\d", s2)
    # Each ring label appears twice (open + close) → n_rings = total / 2
    return (len(single) + len(double)) // 2


def _count_carbonyl(smiles: str) -> int:
    """Count C=O groups (ketone, aldehyde, ester, amide, acid)."""
    # Matches: C=O, c=O, or (=O) branches that belong to C
    return len(re.findall(r"[Cc]=O|\(=O\)", smiles))


def _count_hbd(smiles: str) -> int:
    """Estimate H-bond donor count from SMILES.

    Donors counted:
    - Bracket [NH], [NH2], [nH], [OH], [SH] (explicit H on N, O, S)
    - Non-bracket aliphatic N (typically primary/secondary amine, HBD)
    - Carboxylic acid O: pattern C(=O)O or C(O)=O (contributes 1 donor)
    """
    count = 0

    # ---- explicit bracket donors ----
    for m in re.finditer(r"\[([NOSnosb])H(\d?)", smiles):
        h_str = m.group(2)
        count += int(h_str) if h_str else 1

    # ---- non-bracket N (organic subset) ----
    # Remove bracket content to avoid double-counting
    s_nb = re.sub(r"\[.*?\]", "", smiles)
    n_N_nb = len(re.findall(r"N", s_nb))
    # Heuristic: ~65 % of non-bracket N bear at least one H
    count += round(n_N_nb * 0.65)

    # ---- carboxylic acid / sulfonic acid ----
    count += len(re.findall(r"C\(=O\)O[^C]|C\(=O\)O$|C\(O\)=O", smiles))

    return max(0, count)
