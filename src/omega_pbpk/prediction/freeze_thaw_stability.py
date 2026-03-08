"""Phase 341 — Drug Stability in Freeze-Thaw Cycles.

Models protein/drug degradation during freeze-thaw cycling for
biopharmaceutical formulations using physicochemical stress parameters.
"""

from __future__ import annotations

from dataclasses import dataclass

_VALID_PROTEIN_TYPES = {"mAb", "peptide", "small_molecule"}


def _validate_inputs(
    n_cycles: int,
    k_aggregation_per_cycle: float,
    k_denaturation_per_cycle: float,
    excipient_protection_factor: float,
    protein_type: str,
) -> None:
    if not (1 <= n_cycles <= 20):
        raise ValueError(f"n_cycles must be between 1 and 20, got {n_cycles}")
    if not (0.0 <= k_aggregation_per_cycle <= 1.0):
        raise ValueError(
            f"k_aggregation_per_cycle must be in [0, 1], got {k_aggregation_per_cycle}"
        )
    if not (0.0 <= k_denaturation_per_cycle <= 1.0):
        raise ValueError(
            f"k_denaturation_per_cycle must be in [0, 1], got {k_denaturation_per_cycle}"
        )
    if not (0.0 <= excipient_protection_factor <= 1.0):
        raise ValueError(
            f"excipient_protection_factor must be in [0, 1], got {excipient_protection_factor}"
        )
    if protein_type not in _VALID_PROTEIN_TYPES:
        raise ValueError(
            f"protein_type must be one of {_VALID_PROTEIN_TYPES}, got '{protein_type}'"
        )


@dataclass
class FreezeTHawStabilityResult:
    """Result of freeze-thaw stability analysis."""

    drug_name: str
    n_cycles: int
    initial_concentration_mg_mL: float
    cycle_numbers: list
    purity_per_cycle: list
    aggregation_per_cycle: list
    final_purity_pct: float
    final_concentration_mg_mL: float
    aggregation_fraction: float
    stability_class: str
    recommended_max_cycles: int
    excipient_effectiveness_pct: float
    notes: str


def simulate_freeze_thaw_stability(
    drug_name: str,
    n_cycles: int,
    initial_concentration_mg_mL: float,
    k_aggregation_per_cycle: float,
    k_denaturation_per_cycle: float,
    excipient_protection_factor: float,
    protein_type: str,
    freeze_rate_C_per_min: float = 1.0,
    thaw_rate_C_per_min: float = 2.0,
) -> FreezeTHawStabilityResult:
    """Simulate drug stability over multiple freeze-thaw cycles.

    Parameters
    ----------
    drug_name:
        Name of the drug/protein.
    n_cycles:
        Number of freeze-thaw cycles (1-20).
    initial_concentration_mg_mL:
        Initial drug concentration (mg/mL).
    k_aggregation_per_cycle:
        Aggregation rate constant per cycle (0-1).
    k_denaturation_per_cycle:
        Denaturation rate constant per cycle (0-1).
    excipient_protection_factor:
        Excipient protection factor (0=no protection, 1=full protection).
    protein_type:
        Type of protein/drug: 'mAb', 'peptide', or 'small_molecule'.
    freeze_rate_C_per_min:
        Freezing rate in degrees C per minute (default 1.0).
    thaw_rate_C_per_min:
        Thawing rate in degrees C per minute (default 2.0).

    Returns
    -------
    FreezeTHawStabilityResult
        Comprehensive stability analysis result.
    """
    _validate_inputs(
        n_cycles,
        k_aggregation_per_cycle,
        k_denaturation_per_cycle,
        excipient_protection_factor,
        protein_type,
    )

    # Compute net rate constants (protected by excipient)
    net_k_agg = k_aggregation_per_cycle * (1.0 - excipient_protection_factor)
    net_k_den = k_denaturation_per_cycle * (1.0 - excipient_protection_factor)

    # Stress from freeze and thaw rates (slower = more stress)
    freeze_stress = (1.0 / freeze_rate_C_per_min) * 0.01
    thaw_stress = (1.0 / thaw_rate_C_per_min) * 0.005

    # Per-cycle fractional loss
    per_cycle_loss = net_k_agg + net_k_den + freeze_stress + thaw_stress
    # Clamp to [0, 1] so (1 - per_cycle_loss) is always non-negative
    per_cycle_loss = max(0.0, min(1.0, per_cycle_loss))

    cycle_numbers: list = []
    purity_per_cycle: list = []
    aggregation_per_cycle: list = []

    for cycle in range(1, n_cycles + 1):
        remaining_fraction = max(0.0, min(1.0, (1.0 - per_cycle_loss) ** cycle))
        purity_pct = remaining_fraction * 100.0
        agg_frac = min(1.0, net_k_agg * cycle * remaining_fraction)

        cycle_numbers.append(cycle)
        purity_per_cycle.append(purity_pct)
        aggregation_per_cycle.append(agg_frac)

    final_remaining = max(0.0, min(1.0, (1.0 - per_cycle_loss) ** n_cycles))
    final_purity_pct = final_remaining * 100.0
    final_concentration_mg_mL = initial_concentration_mg_mL * final_remaining
    aggregation_fraction = min(1.0, net_k_agg * n_cycles * final_remaining)

    # Stability class
    if final_purity_pct > 95.0:
        stability_class = "stable"
    elif final_purity_pct >= 90.0:
        stability_class = "borderline"
    else:
        stability_class = "unstable"

    # Recommended max cycles: iterate until purity < 90%, cap at 20
    recommended_max_cycles = 1
    for c in range(1, 21):
        rf = max(0.0, min(1.0, (1.0 - per_cycle_loss) ** c))
        if rf * 100.0 >= 90.0:
            recommended_max_cycles = c
        else:
            break

    excipient_effectiveness_pct = excipient_protection_factor * 100.0

    # Build notes
    notes_parts = [
        f"Per-cycle loss fraction: {per_cycle_loss:.4f}.",
        f"Protein type: {protein_type}.",
    ]
    if excipient_protection_factor > 0:
        notes_parts.append(f"Excipient provides {excipient_effectiveness_pct:.1f}% protection.")
    if stability_class == "unstable":
        notes_parts.append(f"Purity below 90% after {n_cycles} cycles — reformulation recommended.")
    elif stability_class == "borderline":
        notes_parts.append("Purity borderline — minimize freeze-thaw cycles.")
    else:
        notes_parts.append("Product stable under tested conditions.")

    notes = " ".join(notes_parts)

    return FreezeTHawStabilityResult(
        drug_name=drug_name,
        n_cycles=n_cycles,
        initial_concentration_mg_mL=initial_concentration_mg_mL,
        cycle_numbers=cycle_numbers,
        purity_per_cycle=purity_per_cycle,
        aggregation_per_cycle=aggregation_per_cycle,
        final_purity_pct=final_purity_pct,
        final_concentration_mg_mL=final_concentration_mg_mL,
        aggregation_fraction=aggregation_fraction,
        stability_class=stability_class,
        recommended_max_cycles=recommended_max_cycles,
        excipient_effectiveness_pct=excipient_effectiveness_pct,
        notes=notes,
    )
