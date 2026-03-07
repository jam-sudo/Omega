"""Biphasic (hormetic) pharmacodynamic model — Phase 255.

Models drugs exhibiting stimulation at low concentrations and inhibition
at high concentrations (or vice versa). Used for hormesis, inverted-U
dose-response, and beta-adrenergic effects.

References
----------
- Calabrese EJ & Baldwin LA, Nature. 2003;421(6924):691-2
- Chou TC, Pharmacol Rev. 2006;58(3):621-81
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class BiphasicPDResult:
    """Result from biphasic PD model evaluation."""

    drug_name: str
    concentrations: list[float]  # mg/L
    effects: list[float]  # Effect values
    effect_at_cmax: float
    peak_stimulation: float  # Maximum stimulatory effect
    peak_stimulation_conc: float  # Conc at peak stimulation
    tipping_point_conc: float  # Where effect crosses baseline (E=1 or E=0)
    model_type: str  # "bell" or "inverted_bell"
    notes: str


def biphasic_effect(
    concentration: float,
    e_stim: float = 1.5,
    ec50_stim: float = 0.5,
    e_inhib: float = 0.2,
    ec50_inhib: float = 5.0,
    hill_stim: float = 1.0,
    hill_inhib: float = 2.0,
) -> float:
    """Compute biphasic (bell-shaped) drug effect at a given concentration.

    Effect = E_stim_component - E_inhib_component + baseline(1.0)

    Parameters
    ----------
    concentration : Drug concentration (mg/L). Must be >= 0.
    e_stim : Maximum stimulatory effect above baseline.
    ec50_stim : EC50 for stimulatory component (mg/L). Must be > 0.
    e_inhib : Maximum inhibitory effect below baseline.
    ec50_inhib : EC50 for inhibitory component (mg/L). Must be > 0.
    hill_stim : Hill coefficient for stimulation.
    hill_inhib : Hill coefficient for inhibition.

    Returns
    -------
    float : Effect value. 1.0 = baseline, >1 = stimulation, <1 = inhibition.
    """
    if ec50_stim <= 0:
        raise ValueError("ec50_stim must be > 0")
    if ec50_inhib <= 0:
        raise ValueError("ec50_inhib must be > 0")
    if concentration < 0:
        raise ValueError("concentration must be >= 0")

    if concentration == 0:
        return 1.0

    stim = (
        e_stim
        * concentration**hill_stim
        / (ec50_stim**hill_stim + concentration**hill_stim)
    )
    inhib = (
        e_inhib
        * concentration**hill_inhib
        / (ec50_inhib**hill_inhib + concentration**hill_inhib)
    )

    return 1.0 + stim - inhib


def evaluate_biphasic_pd(
    drug_name: str,
    c_max: float,
    n_points: int = 50,
    e_stim: float = 1.5,
    ec50_stim: float = 0.5,
    e_inhib: float = 0.2,
    ec50_inhib: float = 5.0,
    hill_stim: float = 1.0,
    hill_inhib: float = 2.0,
) -> BiphasicPDResult:
    """Evaluate biphasic PD model over a concentration range.

    Parameters
    ----------
    drug_name : Drug name.
    c_max : Maximum concentration for evaluation (mg/L). Must be > 0.
    n_points : Number of concentration points. Must be >= 2.
    e_stim : Maximum stimulatory effect above baseline.
    ec50_stim : EC50 for stimulatory component (mg/L).
    e_inhib : Maximum inhibitory effect below baseline.
    ec50_inhib : EC50 for inhibitory component (mg/L).
    hill_stim : Hill coefficient for stimulation.
    hill_inhib : Hill coefficient for inhibition.

    Returns
    -------
    BiphasicPDResult
    """
    if c_max <= 0:
        raise ValueError("c_max must be > 0")
    if n_points < 2:
        raise ValueError("n_points must be >= 2")

    # Logarithmically spaced concentrations from c_max/1000 to c_max
    log_min = math.log10(c_max / 1000.0) if c_max > 0.001 else -3
    log_max = math.log10(c_max)
    step = (log_max - log_min) / (n_points - 1)
    concs = [10 ** (log_min + i * step) for i in range(n_points)]

    effects = [
        biphasic_effect(c, e_stim, ec50_stim, e_inhib, ec50_inhib, hill_stim, hill_inhib)
        for c in concs
    ]

    effect_at_cmax = effects[-1]
    peak_stim = max(effects)
    peak_stim_conc = concs[effects.index(peak_stim)]

    # Determine model type from whether peak effect > 1 at low conc
    if e_stim > 0 and e_stim >= e_inhib:
        model_type = "bell"  # stimulate then inhibit
    else:
        model_type = "inverted_bell"

    # Tipping point: where effect crosses back to 1.0 (from stimulation to inhibition)
    tipping = c_max  # default
    for i in range(1, len(concs)):
        if effects[i - 1] >= 1.0 > effects[i]:
            # Linear interpolation
            frac = (1.0 - effects[i - 1]) / (effects[i] - effects[i - 1])
            tipping = concs[i - 1] + frac * (concs[i] - concs[i - 1])
            break

    notes = (
        f"{drug_name}: bell model. Peak stimulation {peak_stim:.3f}x at "
        f"C={peak_stim_conc:.3f} mg/L. Tipping to inhibition at ~{tipping:.3f} mg/L. "
        f"Effect at Cmax={effect_at_cmax:.3f}x."
    )

    return BiphasicPDResult(
        drug_name=drug_name,
        concentrations=concs,
        effects=effects,
        effect_at_cmax=effect_at_cmax,
        peak_stimulation=peak_stim,
        peak_stimulation_conc=peak_stim_conc,
        tipping_point_conc=tipping,
        model_type=model_type,
        notes=notes,
    )


__all__ = ["BiphasicPDResult", "biphasic_effect", "evaluate_biphasic_pd"]
