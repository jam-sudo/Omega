"""Gut microbiome drug interaction model — Phase 1099.

Models the effect of gut microbiome on drug metabolism and bioavailability,
covering microbial reduction of prodrugs, azo bond cleavage, sulfatase activity,
and beta-glucuronidase deconjugation.
"""

from __future__ import annotations

from dataclasses import dataclass

# Base first-order rate constants (per hour) for each enzymatic reaction
_BASE_K: dict[str, float] = {
    "beta_glucuronidase": 0.05,
    "azoreductase": 0.1,
    "sulfatase": 0.03,
    "nitroreductase": 0.08,
    "none": 0.0,
}

_COLONIZATION_FACTOR: dict[str, float] = {
    "dysbiotic": 0.2,
    "normal": 1.0,
    "enriched": 2.0,
}

_VALID_REACTIONS = frozenset(_BASE_K.keys())
_VALID_COLONIZATION = frozenset(_COLONIZATION_FACTOR.keys())


def _trapz(times: list[float], values: list[float]) -> float:
    """Trapezoidal integration."""
    total = 0.0
    for i in range(1, len(times)):
        dt = times[i] - times[i - 1]
        total += 0.5 * (values[i] + values[i - 1]) * dt
    return total


@dataclass
class GutMicrobiomePKResult:
    """Result from gut microbiome PK simulation — Phase 1099."""

    drug_name: str
    enzymatic_reaction: str
    colonization_level: str
    transit_h: float
    times_h: list
    a_prodrug_mg: list
    a_active_mg: list
    c_plasma_mg_L: list
    fraction_activated: float
    auc_plasma: float
    cmax_plasma: float
    microbiome_contribution_pct: float
    dysbiosis_risk: str
    notes: str


def _compute_dysbiosis_risk(
    enzymatic_reaction: str,
    colonization_level: str,
) -> str:
    """Determine dysbiosis risk level."""
    if enzymatic_reaction == "none":
        return "low"
    potent_enzymes = {"beta_glucuronidase", "azoreductase", "nitroreductase"}
    if colonization_level == "enriched" and enzymatic_reaction in potent_enzymes:
        return "high"
    return "moderate"


def simulate_microbiome_pk(
    drug_name: str,
    dose_mg: float,
    enzymatic_reaction: str,
    colonization_level: str = "normal",
    colon_transit_h: float = 24.0,
    ka_absorption: float = 0.3,
    cl_L_per_h: float = 5.0,
    vd_L: float = 50.0,
    t_end_h: float = 48.0,
    dt_h: float = 0.1,
) -> GutMicrobiomePKResult:
    """Simulate drug gut microbiome interaction PK.

    Args:
        drug_name: Name of the drug.
        dose_mg: Initial dose in mg.
        enzymatic_reaction: Type of microbial enzymatic reaction.
        colonization_level: Microbiome colonization level.
        colon_transit_h: Colon transit time in hours.
        ka_absorption: First-order absorption rate of active drug (1/h).
        cl_L_per_h: Systemic clearance (L/h).
        vd_L: Volume of distribution (L).
        t_end_h: Simulation end time (h).
        dt_h: Integration time step (h).

    Returns:
        GutMicrobiomePKResult with full simulation output.
    """
    if enzymatic_reaction not in _VALID_REACTIONS:
        raise ValueError(
            f"enzymatic_reaction must be one of {sorted(_VALID_REACTIONS)}, "
            f"got '{enzymatic_reaction}'"
        )
    if colonization_level not in _VALID_COLONIZATION:
        raise ValueError(
            f"colonization_level must be one of {sorted(_VALID_COLONIZATION)}, "
            f"got '{colonization_level}'"
        )
    if not (4.0 <= colon_transit_h <= 96.0):
        raise ValueError(f"colon_transit_h must be in [4, 96], got {colon_transit_h}")
    if dose_mg <= 0:
        raise ValueError(f"dose_mg must be > 0, got {dose_mg}")

    # Compute effective microbial rate constant
    base_k = _BASE_K[enzymatic_reaction]
    col_factor = _COLONIZATION_FACTOR[colonization_level]
    transit_factor = colon_transit_h / 24.0
    k_microbial = base_k * col_factor * transit_factor

    ke = cl_L_per_h / vd_L  # elimination rate constant

    # Initial conditions
    a_prodrug = dose_mg
    a_active = 0.0
    c_plasma = 0.0

    times_h: list[float] = []
    a_prodrug_list: list[float] = []
    a_active_list: list[float] = []
    c_plasma_list: list[float] = []

    t = 0.0
    n_steps = int(t_end_h / dt_h) + 1

    for _ in range(n_steps):
        times_h.append(t)
        a_prodrug_list.append(a_prodrug)
        a_active_list.append(a_active)
        c_plasma_list.append(c_plasma)

        # Forward Euler derivatives
        d_prodrug = -k_microbial * a_prodrug
        d_active = k_microbial * a_prodrug - ka_absorption * a_active
        d_plasma = (ka_absorption * a_active / vd_L) - ke * c_plasma

        a_prodrug = max(0.0, a_prodrug + d_prodrug * dt_h)
        a_active = max(0.0, a_active + d_active * dt_h)
        c_plasma = max(0.0, c_plasma + d_plasma * dt_h)

        t = round(t + dt_h, 10)

    auc_plasma = _trapz(times_h, c_plasma_list)
    cmax_plasma = max(c_plasma_list)

    # Fraction activated: amount converted from prodrug = dose - remaining prodrug
    final_prodrug = a_prodrug_list[-1]
    amount_converted = dose_mg - final_prodrug
    fraction_activated = max(0.0, min(1.0, amount_converted / dose_mg)) if dose_mg > 0 else 0.0

    # Microbiome contribution: if no microbial activity, plasma AUC ~ 0
    # We compare actual AUC to a hypothetical with no microbiome (k_microbial=0)
    if enzymatic_reaction == "none" or k_microbial == 0.0:
        microbiome_contribution_pct = 0.0
    else:
        # With no microbiome, active drug released is 0
        # AUC attributed to microbiome = fraction_activated of total AUC
        microbiome_contribution_pct = min(100.0, fraction_activated * 100.0)

    dysbiosis_risk = _compute_dysbiosis_risk(enzymatic_reaction, colonization_level)

    notes_parts = [
        f"k_microbial={k_microbial:.4f}/h",
        f"colonization_factor={col_factor}",
        f"transit_factor={transit_factor:.2f}",
    ]
    notes = "; ".join(notes_parts)

    return GutMicrobiomePKResult(
        drug_name=drug_name,
        enzymatic_reaction=enzymatic_reaction,
        colonization_level=colonization_level,
        transit_h=colon_transit_h,
        times_h=times_h,
        a_prodrug_mg=a_prodrug_list,
        a_active_mg=a_active_list,
        c_plasma_mg_L=c_plasma_list,
        fraction_activated=fraction_activated,
        auc_plasma=auc_plasma,
        cmax_plasma=cmax_plasma,
        microbiome_contribution_pct=microbiome_contribution_pct,
        dysbiosis_risk=dysbiosis_risk,
        notes=notes,
    )


def compare_colonization_levels(
    drug_name: str,
    dose_mg: float,
    enzymatic_reaction: str,
) -> list[GutMicrobiomePKResult]:
    """Simulate microbiome PK for all 3 colonization levels.

    Args:
        drug_name: Name of the drug.
        dose_mg: Initial dose in mg.
        enzymatic_reaction: Type of microbial enzymatic reaction.

    Returns:
        List of 3 GutMicrobiomePKResult objects sorted by fraction_activated descending.
    """
    levels = ["dysbiotic", "normal", "enriched"]
    results = [
        simulate_microbiome_pk(
            drug_name=drug_name,
            dose_mg=dose_mg,
            enzymatic_reaction=enzymatic_reaction,
            colonization_level=level,
        )
        for level in levels
    ]
    results.sort(key=lambda r: r.fraction_activated, reverse=True)
    return results
