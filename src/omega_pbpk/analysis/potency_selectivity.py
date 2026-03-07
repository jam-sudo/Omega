"""Drug potency, selectivity, and ligand efficiency analysis.

Provides metrics for evaluating drug potency and selectivity profiles
including ligand efficiency (LE), lipophilic efficiency (LipE), and
selectivity ratios across target panels.
"""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = [
    "LEResult",
    "SelectivityResult",
    "ligand_efficiency",
    "compute_selectivity_ratio",
    "selectivity_profile",
    "potency_optimization_score",
]


@dataclass(frozen=True)
class LEResult:
    """Ligand efficiency analysis result.

    Attributes
    ----------
    pIC50:
        Negative log10 of IC50 in molar units.
    mw:
        Molecular weight in g/mol.
    n_heavy_atoms:
        Number of non-hydrogen atoms used for LE calculation.
    logP:
        Octanol-water partition coefficient.
    le:
        Ligand efficiency = pIC50 / n_heavy_atoms.
    lipe:
        Lipophilic efficiency = pIC50 - logP.
    lelp:
        Ligand efficiency-dependent lipophilicity parameter = logP / LE.
    le_category:
        Classification: 'high' (>0.3), 'moderate' (0.2-0.3), 'low' (<0.2).
    notes:
        Human-readable summary of the result.
    """

    pIC50: float
    mw: float
    n_heavy_atoms: int
    logP: float
    le: float
    lipe: float
    lelp: float
    le_category: str
    notes: str


@dataclass(frozen=True)
class SelectivityResult:
    """Selectivity profile result for a compound across a target panel.

    Attributes
    ----------
    compound_name:
        Name of the compound.
    primary_target:
        Name of the primary (on-target) receptor.
    primary_ic50_nM:
        IC50 at the primary target in nM.
    selectivity_ratios:
        Dict mapping off-target name to selectivity ratio (IC50_off / IC50_primary).
    most_selective_off_target:
        Off-target with the highest selectivity ratio.
    worst_selectivity_ratio:
        Lowest selectivity ratio across all off-targets (worst selectivity concern).
    overall_selectivity_class:
        'selective' (worst_ratio > 100), 'moderate' (10-100), 'nonselective' (<10).
    notes:
        Human-readable summary.
    """

    compound_name: str
    primary_target: str
    primary_ic50_nM: float
    selectivity_ratios: dict[str, float] = field(default_factory=dict)
    most_selective_off_target: str = ""
    worst_selectivity_ratio: float = float("inf")
    overall_selectivity_class: str = "selective"
    notes: str = ""


def ligand_efficiency(
    pIC50: float,
    mw: float,
    logP: float,
    n_heavy_atoms: int | None = None,
) -> LEResult:
    """Compute ligand efficiency metrics for a compound.

    Parameters
    ----------
    pIC50:
        Negative log10 of IC50 (e.g., pIC50=9 means IC50=1 nM).
    mw:
        Molecular weight in g/mol.
    logP:
        Octanol-water partition coefficient.
    n_heavy_atoms:
        Number of non-hydrogen atoms. If None, estimated as mw / 13.

    Returns
    -------
    LEResult
        Ligand efficiency result with LE, LipE, LELP, and category.

    Raises
    ------
    ValueError
        If pIC50 <= 0 or mw <= 0.
    """
    if pIC50 <= 0:
        raise ValueError(f"pIC50 must be > 0, got {pIC50}")
    if mw <= 0:
        raise ValueError(f"mw must be > 0, got {mw}")

    if n_heavy_atoms is None:
        n_heavy_atoms = max(1, round(mw / 13.0))

    le = pIC50 / n_heavy_atoms
    lipe = pIC50 - logP
    lelp = logP / le if le > 0 else float("inf")

    if le > 0.3:
        le_category = "high"
    elif le >= 0.2:
        le_category = "moderate"
    else:
        le_category = "low"

    notes = (
        f"LE={le:.3f} ({le_category}), LipE={lipe:.2f}, LELP={lelp:.2f}. "
        f"n_heavy_atoms={n_heavy_atoms}, pIC50={pIC50:.2f}, MW={mw:.1f}, logP={logP:.2f}."
    )

    return LEResult(
        pIC50=pIC50,
        mw=mw,
        n_heavy_atoms=n_heavy_atoms,
        logP=logP,
        le=le,
        lipe=lipe,
        lelp=lelp,
        le_category=le_category,
        notes=notes,
    )


def compute_selectivity_ratio(
    ic50_target_nM: float,
    ic50_off_target_nM: float,
) -> float:
    """Compute selectivity ratio for a compound between target and off-target.

    Selectivity = IC50_off_target / IC50_target.
    Higher values indicate better selectivity for the primary target.
    - > 100: selective
    - 10-100: moderate selectivity
    - < 10: nonselective

    Parameters
    ----------
    ic50_target_nM:
        IC50 at the primary target in nM.
    ic50_off_target_nM:
        IC50 at the off-target receptor in nM.

    Returns
    -------
    float
        Selectivity ratio.

    Raises
    ------
    ValueError
        If either IC50 value is non-positive.
    """
    if ic50_target_nM <= 0:
        raise ValueError(f"ic50_target_nM must be > 0, got {ic50_target_nM}")
    if ic50_off_target_nM <= 0:
        raise ValueError(f"ic50_off_target_nM must be > 0, got {ic50_off_target_nM}")
    return ic50_off_target_nM / ic50_target_nM


def selectivity_profile(
    compound_name: str,
    target_ic50s: dict[str, float],
) -> SelectivityResult:
    """Compute selectivity profile for a compound across a target panel.

    The primary target is the first key in target_ic50s. All remaining
    targets are treated as off-targets.

    Parameters
    ----------
    compound_name:
        Identifier for the compound.
    target_ic50s:
        Ordered dict mapping target name to IC50 in nM. The first entry
        is the primary target.

    Returns
    -------
    SelectivityResult
        Selectivity ratios for all off-targets vs. the primary target.

    Raises
    ------
    ValueError
        If fewer than 2 targets provided or any IC50 <= 0.
    """
    if len(target_ic50s) < 1:
        raise ValueError("At least one target IC50 must be provided.")

    for name, val in target_ic50s.items():
        if val <= 0:
            raise ValueError(f"IC50 for '{name}' must be > 0, got {val}")

    primary_target = next(iter(target_ic50s))
    primary_ic50 = target_ic50s[primary_target]

    if len(target_ic50s) < 2:
        return SelectivityResult(
            compound_name=compound_name,
            primary_target=primary_target,
            primary_ic50_nM=primary_ic50,
            selectivity_ratios={},
            most_selective_off_target="",
            worst_selectivity_ratio=float("inf"),
            overall_selectivity_class="selective",
            notes="Only one target provided; no off-targets to compare.",
        )

    ratios: dict[str, float] = {}
    for target, ic50 in target_ic50s.items():
        if target == primary_target:
            continue
        ratios[target] = compute_selectivity_ratio(primary_ic50, ic50)

    most_selective = max(ratios, key=lambda k: ratios[k])
    worst_ratio = min(ratios.values())

    if worst_ratio > 100:
        overall_class = "selective"
    elif worst_ratio >= 10:
        overall_class = "moderate"
    else:
        overall_class = "nonselective"

    notes = (
        f"{compound_name}: primary target '{primary_target}' IC50={primary_ic50:.2f} nM. "
        f"Worst off-target selectivity ratio={worst_ratio:.1f}x ({overall_class}). "
        f"Most selective off-target: '{most_selective}' ({ratios[most_selective]:.1f}x)."
    )

    return SelectivityResult(
        compound_name=compound_name,
        primary_target=primary_target,
        primary_ic50_nM=primary_ic50,
        selectivity_ratios=ratios,
        most_selective_off_target=most_selective,
        worst_selectivity_ratio=worst_ratio,
        overall_selectivity_class=overall_class,
        notes=notes,
    )


def _desirability_linear(value: float, low: float, high: float) -> float:
    """Compute a linear desirability score (0-1) between low and high thresholds."""
    if value >= high:
        return 1.0
    if value <= low:
        return 0.0
    return (value - low) / (high - low)


def potency_optimization_score(
    pIC50: float,
    logP: float,
    mw: float,
    hbd: int,
    psa: float,
    n_heavy_atoms: int | None = None,
) -> float:
    """Compute composite potency optimization score (0-1).

    Integrates ligand efficiency, lipophilic efficiency, molecular weight,
    hydrogen bond donor count, and polar surface area into a single
    desirability-weighted composite score.

    Weights:
    - 0.30 * LE score
    - 0.20 * LipE score
    - 0.20 * MW score
    - 0.15 * HBD score
    - 0.15 * PSA score

    Parameters
    ----------
    pIC50:
        Negative log10 of IC50 in molar units.
    logP:
        Octanol-water partition coefficient.
    mw:
        Molecular weight in g/mol.
    hbd:
        Number of hydrogen bond donors.
    psa:
        Polar surface area in Angstrom^2.
    n_heavy_atoms:
        Number of heavy atoms (estimated from MW if None).

    Returns
    -------
    float
        Composite score in [0, 1]. Higher is better.

    Raises
    ------
    ValueError
        If pIC50 <= 0 or mw <= 0.
    """
    if pIC50 <= 0:
        raise ValueError(f"pIC50 must be > 0, got {pIC50}")
    if mw <= 0:
        raise ValueError(f"mw must be > 0, got {mw}")

    if n_heavy_atoms is None:
        n_heavy_atoms = max(1, round(mw / 13.0))

    le = pIC50 / n_heavy_atoms
    lipe = pIC50 - logP

    # LE: 0.1 -> 0.0, 0.3 -> 1.0
    le_score = _desirability_linear(le, low=0.1, high=0.3)

    # LipE: 3 -> 0.0, 6 -> 1.0
    lipe_score = _desirability_linear(lipe, low=3.0, high=6.0)

    # MW: drug-like range. 500 -> 0.0, 300 -> 1.0 (lower is better)
    # Invert: high MW = low score
    mw_score = _desirability_linear(mw, low=300.0, high=500.0)
    mw_score = 1.0 - mw_score  # invert so low MW scores high

    # HBD: <=3 is ideal, >5 is bad
    hbd_score = _desirability_linear(float(hbd), low=0.0, high=5.0)
    hbd_score = 1.0 - hbd_score  # invert so low HBD scores high

    # PSA: 60-140 Angstrom^2 is good oral absorption range
    # Optimal ~90; score based on being in the good range
    # <60 -> 0.0 (too low), 60-140 -> scales up, >140 -> 0.0
    if psa < 60.0:
        psa_score = _desirability_linear(psa, low=20.0, high=60.0)
    elif psa <= 140.0:
        # Peak at midpoint 100 Angstrom^2
        mid = 100.0
        width = 40.0
        psa_score = max(0.0, 1.0 - abs(psa - mid) / width)
    else:
        psa_score = _desirability_linear(psa, low=200.0, high=140.0)
        psa_score = max(0.0, psa_score)

    composite = (
        0.30 * le_score + 0.20 * lipe_score + 0.20 * mw_score + 0.15 * hbd_score + 0.15 * psa_score
    )

    return float(min(1.0, max(0.0, composite)))
