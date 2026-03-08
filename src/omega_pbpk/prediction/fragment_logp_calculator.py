"""
Phase 438: Fragment-based logP calculation using Crippen/Wildman fragment contributions.

Calculates logP from atom counts using fragment contribution model,
with drug-likeness and property predictions.
"""

from dataclasses import dataclass
from typing import Any

# Crippen 1999 simplified fragment contributions
FRAGMENT_CONTRIBUTIONS = {
    "C_aliphatic": 0.20,  # per aliphatic carbon
    "C_aromatic": 0.10,  # per aromatic carbon
    "N_aliphatic": -1.00,  # per aliphatic nitrogen
    "N_aromatic": -0.50,  # per aromatic nitrogen
    "O_aliphatic": -0.67,  # per aliphatic oxygen
    "O_aromatic": -0.30,  # per aromatic oxygen
    "F": 0.14,
    "Cl": 0.60,
    "Br": 0.94,
    "I": 1.35,
    "S": 0.03,
    "ring_bonus": 0.20,  # per aromatic ring
}

# Halogen molecular weights
_HALOGEN_MW = {
    "F": 19.0,
    "Cl": 35.0,
    "Br": 80.0,
    "I": 127.0,
}

_VALID_HALOGENS = {"F", "Cl", "Br", "I"}


@dataclass(frozen=True)
class FragmentLogPResult:
    """Result of fragment-based logP calculation."""

    drug_name: str
    smiles_like_input: str
    n_carbons: int
    n_nitrogens: int
    n_oxygens: int
    n_halogens: int
    n_sulfurs: int
    n_aromatic_rings: int
    n_h_bond_donors: int
    n_h_bond_acceptors: int
    mw_estimated: float
    logp_calculated: float
    logp_class: str
    bbb_penetration_likely: bool
    oral_absorption_likely: bool
    aqueous_solubility_estimate_mg_mL: float
    notes: str


def _classify_logp(logp: float) -> str:
    """Classify logP into descriptive category."""
    if logp > 5.0:
        return "highly_lipophilic"
    elif logp >= 2.0:
        return "lipophilic"
    elif logp >= 0.0:
        return "moderate"
    else:
        return "hydrophilic"


def calculate_fragment_logp(
    drug_name: str,
    n_carbons: int,
    n_nitrogens: int,
    n_oxygens: int,
    n_halogens: int,
    n_sulfurs: int,
    n_aromatic_rings: int,
    n_h_bond_donors: int,
    n_h_bond_acceptors: int,
    halogen_type: str = "Cl",
    aromatic_n_fraction: float = 0.5,
    aromatic_o_fraction: float = 0.3,
    mw_override: float | None = None,
) -> FragmentLogPResult:
    """
    Calculate fragment-based logP from molecular composition.

    Args:
        drug_name: Name of the drug/compound
        n_carbons: Number of carbon atoms
        n_nitrogens: Number of nitrogen atoms
        n_oxygens: Number of oxygen atoms
        n_halogens: Number of halogen atoms (F, Cl, Br, or I)
        n_sulfurs: Number of sulfur atoms
        n_aromatic_rings: Number of aromatic rings
        n_h_bond_donors: Number of H-bond donors (OH + NH groups)
        n_h_bond_acceptors: Number of H-bond acceptors (N + O atoms)
        halogen_type: Type of halogen ("F", "Cl", "Br", "I"); default "Cl"
        aromatic_n_fraction: Fraction of nitrogens that are aromatic (0-1)
        aromatic_o_fraction: Fraction of oxygens that are aromatic (0-1)
        mw_override: If provided, use this value instead of estimated MW

    Returns:
        FragmentLogPResult with calculated logP and drug-likeness properties
    """
    # Validation
    if n_carbons < 0:
        raise ValueError("n_carbons must be >= 0")
    if n_nitrogens < 0:
        raise ValueError("n_nitrogens must be >= 0")
    if n_oxygens < 0:
        raise ValueError("n_oxygens must be >= 0")
    if n_halogens < 0:
        raise ValueError("n_halogens must be >= 0")
    if n_sulfurs < 0:
        raise ValueError("n_sulfurs must be >= 0")
    if n_aromatic_rings < 0:
        raise ValueError("n_aromatic_rings must be >= 0")
    if n_h_bond_donors < 0:
        raise ValueError("n_h_bond_donors must be >= 0")
    if n_h_bond_acceptors < 0:
        raise ValueError("n_h_bond_acceptors must be >= 0")
    if halogen_type not in _VALID_HALOGENS:
        raise ValueError(f"halogen_type must be one of {_VALID_HALOGENS}, got '{halogen_type}'")

    # Clamp fractions to [0, 1]
    aromatic_n_fraction = max(0.0, min(1.0, aromatic_n_fraction))
    aromatic_o_fraction = max(0.0, min(1.0, aromatic_o_fraction))

    # Split carbons into aromatic and aliphatic
    # ~4 carbons per aromatic ring on average
    n_c_arom = min(n_carbons, round(n_aromatic_rings * 4))
    n_c_alip = max(0, n_carbons - n_c_arom)

    # Split nitrogens
    n_n_arom = round(n_nitrogens * aromatic_n_fraction)
    n_n_alip = n_nitrogens - n_n_arom

    # Split oxygens
    n_o_arom = round(n_oxygens * aromatic_o_fraction)
    n_o_alip = n_oxygens - n_o_arom

    # Calculate logP from fragments
    logp = (
        n_c_alip * FRAGMENT_CONTRIBUTIONS["C_aliphatic"]
        + n_c_arom * FRAGMENT_CONTRIBUTIONS["C_aromatic"]
        + n_n_alip * FRAGMENT_CONTRIBUTIONS["N_aliphatic"]
        + n_n_arom * FRAGMENT_CONTRIBUTIONS["N_aromatic"]
        + n_o_alip * FRAGMENT_CONTRIBUTIONS["O_aliphatic"]
        + n_o_arom * FRAGMENT_CONTRIBUTIONS["O_aromatic"]
        + n_halogens * FRAGMENT_CONTRIBUTIONS[halogen_type]
        + n_sulfurs * FRAGMENT_CONTRIBUTIONS["S"]
        + n_aromatic_rings * FRAGMENT_CONTRIBUTIONS["ring_bonus"]
    )

    # Clamp logP to [-5, 10]
    logp_calculated = max(-5.0, min(10.0, logp))

    # Estimate molecular weight
    halogen_mw = _HALOGEN_MW[halogen_type]
    mw_estimated = (
        n_carbons * 12.0
        + n_nitrogens * 14.0
        + n_oxygens * 16.0
        + n_halogens * halogen_mw
        + n_sulfurs * 32.0
        + n_aromatic_rings * 2.0
        + 18.0  # base (water equivalent)
    )

    if mw_override is not None:
        mw_estimated = float(mw_override)

    # Classify logP
    logp_class = _classify_logp(logp_calculated)

    # BBB penetration likelihood
    bbb_penetration_likely = (
        1.0 <= logp_calculated <= 4.0 and mw_estimated < 400.0 and n_h_bond_donors < 3
    )

    # Oral absorption (Lipinski rule of five)
    oral_absorption_likely = (
        mw_estimated < 500.0
        and logp_calculated < 5.0
        and n_h_bond_donors < 5
        and n_h_bond_acceptors < 10
    )

    # Aqueous solubility estimate (rough): 10^(0.5 - logP) * 100
    raw_solubility = (10 ** (0.5 - logp_calculated)) * 100.0
    aqueous_solubility = max(0.001, min(1000.0, raw_solubility))

    # Build simplified SMILES-like descriptor
    smiles_like = (
        f"C{n_carbons}N{n_nitrogens}O{n_oxygens}"
        f"{halogen_type}{n_halogens}S{n_sulfurs}"
        f"Ring{n_aromatic_rings}"
    )

    # Notes
    bbb_str = "BBB penetration likely" if bbb_penetration_likely else "BBB penetration unlikely"
    oral_str = (
        "oral absorption likely (Lipinski)"
        if oral_absorption_likely
        else "poor oral absorption predicted"
    )
    notes = (
        f"{drug_name}: logP={logp_calculated:.2f} ({logp_class}), "
        f"MW~{mw_estimated:.0f} Da, "
        f"solubility~{aqueous_solubility:.3f} mg/mL. "
        f"{bbb_str}. {oral_str}."
    )

    return FragmentLogPResult(
        drug_name=drug_name,
        smiles_like_input=smiles_like,
        n_carbons=n_carbons,
        n_nitrogens=n_nitrogens,
        n_oxygens=n_oxygens,
        n_halogens=n_halogens,
        n_sulfurs=n_sulfurs,
        n_aromatic_rings=n_aromatic_rings,
        n_h_bond_donors=n_h_bond_donors,
        n_h_bond_acceptors=n_h_bond_acceptors,
        mw_estimated=mw_estimated,
        logp_calculated=logp_calculated,
        logp_class=logp_class,
        bbb_penetration_likely=bbb_penetration_likely,
        oral_absorption_likely=oral_absorption_likely,
        aqueous_solubility_estimate_mg_mL=aqueous_solubility,
        notes=notes,
    )


def screen_fragments(compounds_list: list[dict[str, Any]]) -> list[FragmentLogPResult]:
    """
    Screen a list of compounds and return sorted by logP ascending (hydrophilic first).

    Args:
        compounds_list: List of dicts with keys matching calculate_fragment_logp params

    Returns:
        List of FragmentLogPResult sorted by logp_calculated ascending
    """
    results = []
    for compound in compounds_list:
        result = calculate_fragment_logp(**compound)
        results.append(result)

    # Sort by logP ascending (hydrophilic first)
    results.sort(key=lambda r: r.logp_calculated)
    return results
