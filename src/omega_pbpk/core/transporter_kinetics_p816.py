"""Membrane transporter kinetics — saturable Michaelis-Menten with competitive inhibition (Phase 816).

Models uptake and efflux transporter flux with optional competitive inhibition.
Flux = Vmax * C / (Km * (1 + I/Ki) + C)
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "TransporterKineticsResult",
    "calculate_transporter_kinetics",
    "screen_inhibitors",
]


@dataclass(frozen=True)
class TransporterKineticsResult:
    """Result of transporter kinetics calculation."""

    drug_name: str
    transporter_name: str
    direction: str  # "uptake" or "efflux"
    km_uM: float
    vmax_pmol_per_min_per_mg: float
    inhibitor_name: str  # "" if no inhibitor
    ki_uM: float  # 0.0 if no inhibitor
    inhibitor_conc_uM: float  # 0.0 if no inhibitor
    net_flux_pmol_per_min_per_mg: float  # flux at given substrate_conc
    substrate_conc_uM: float
    transport_fraction: float  # net_flux / uninhibited_flux; 1.0 if no inhibitor
    efflux_ratio: float  # 2.0 for efflux direction, 1.0 for uptake (simplified)
    ddi_risk: str  # "low", "moderate", "high"
    notes: str


def _mm_flux(
    substrate_conc_uM: float,
    km_uM: float,
    vmax: float,
    inhibitor_conc_uM: float,
    ki_uM: float,
) -> float:
    """Michaelis-Menten flux with competitive inhibition."""
    if ki_uM > 0.0 and inhibitor_conc_uM > 0.0:
        km_app = km_uM * (1.0 + inhibitor_conc_uM / ki_uM)
    else:
        km_app = km_uM
    return vmax * substrate_conc_uM / (km_app + substrate_conc_uM)


def calculate_transporter_kinetics(
    drug_name: str,
    transporter_name: str,
    substrate_conc_uM: float,
    km_uM: float,
    vmax_pmol_per_min_per_mg: float,
    direction: str = "uptake",
    inhibitor_name: str = "",
    inhibitor_conc_uM: float = 0.0,
    ki_uM: float = 0.0,
) -> TransporterKineticsResult:
    """Calculate transporter-mediated flux with optional competitive inhibition.

    Parameters
    ----------
    drug_name:
        Name of the substrate drug.
    transporter_name:
        Name of the transporter (e.g., "OATP1B1", "P-gp").
    substrate_conc_uM:
        Substrate concentration (uM, >= 0).
    km_uM:
        Michaelis constant (uM, must be > 0).
    vmax_pmol_per_min_per_mg:
        Maximum transport velocity (pmol/min/mg protein, must be > 0).
    direction:
        "uptake" or "efflux".
    inhibitor_name:
        Name of the inhibitor; "" if none.
    inhibitor_conc_uM:
        Inhibitor concentration (uM); 0.0 if none.
    ki_uM:
        Inhibitor Ki (uM); 0.0 if none.

    Returns
    -------
    TransporterKineticsResult
    """
    if km_uM <= 0.0:
        raise ValueError(f"km_uM must be > 0, got {km_uM}")
    if vmax_pmol_per_min_per_mg <= 0.0:
        raise ValueError(f"vmax_pmol_per_min_per_mg must be > 0, got {vmax_pmol_per_min_per_mg}")
    if substrate_conc_uM < 0.0:
        raise ValueError(f"substrate_conc_uM must be >= 0, got {substrate_conc_uM}")

    vmax = vmax_pmol_per_min_per_mg

    # Uninhibited flux
    uninhibited_flux = _mm_flux(substrate_conc_uM, km_uM, vmax, 0.0, 0.0)

    # Determine if inhibitor is active
    has_inhibitor = inhibitor_name != "" and inhibitor_conc_uM > 0.0 and ki_uM > 0.0

    if has_inhibitor:
        net_flux = _mm_flux(substrate_conc_uM, km_uM, vmax, inhibitor_conc_uM, ki_uM)
    else:
        net_flux = uninhibited_flux

    # transport_fraction: ratio of inhibited to uninhibited flux
    if uninhibited_flux > 0.0:
        transport_fraction = net_flux / uninhibited_flux
    else:
        transport_fraction = 1.0  # no flux in either case → no inhibition to speak of

    # DDI risk based on transport_fraction reduction
    if transport_fraction < 0.5:
        ddi_risk = "high"
    elif transport_fraction < 0.8:
        ddi_risk = "moderate"
    else:
        ddi_risk = "low"

    # Efflux ratio: simplified
    efflux_ratio = 2.0 if direction == "efflux" else 1.0

    # Notes
    notes_parts = []
    if not has_inhibitor:
        notes_parts.append("no inhibitor; baseline kinetics")
    else:
        ki_str = f"Ki={ki_uM:.2f} uM"
        notes_parts.append(
            f"{inhibitor_name} competitive inhibition ({ki_str}); DDI risk: {ddi_risk}"
        )
    if substrate_conc_uM > 10.0 * km_uM:
        notes_parts.append("substrate >> Km; near saturation")
    notes = "; ".join(notes_parts)

    return TransporterKineticsResult(
        drug_name=drug_name,
        transporter_name=transporter_name,
        direction=direction,
        km_uM=km_uM,
        vmax_pmol_per_min_per_mg=vmax,
        inhibitor_name=inhibitor_name if has_inhibitor else "",
        ki_uM=ki_uM if has_inhibitor else 0.0,
        inhibitor_conc_uM=inhibitor_conc_uM if has_inhibitor else 0.0,
        net_flux_pmol_per_min_per_mg=net_flux,
        substrate_conc_uM=substrate_conc_uM,
        transport_fraction=transport_fraction,
        efflux_ratio=efflux_ratio,
        ddi_risk=ddi_risk,
        notes=notes,
    )


def screen_inhibitors(
    drug_name: str,
    transporter_name: str,
    substrate_conc_uM: float,
    km_uM: float,
    vmax_pmol_per_min_per_mg: float,
    inhibitors: list[dict],
) -> list[TransporterKineticsResult]:
    """Screen a list of inhibitors and return sorted by transport_fraction ascending.

    Most inhibitory (lowest transport_fraction) first.

    Parameters
    ----------
    inhibitors:
        List of dicts with keys: "name", "ki_uM", "conc_uM".
    """
    results = []
    for inh in inhibitors:
        result = calculate_transporter_kinetics(
            drug_name=drug_name,
            transporter_name=transporter_name,
            substrate_conc_uM=substrate_conc_uM,
            km_uM=km_uM,
            vmax_pmol_per_min_per_mg=vmax_pmol_per_min_per_mg,
            inhibitor_name=inh["name"],
            inhibitor_conc_uM=inh["conc_uM"],
            ki_uM=inh["ki_uM"],
        )
        results.append(result)
    results.sort(key=lambda r: r.transport_fraction)
    return results
