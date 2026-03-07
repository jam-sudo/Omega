"""Drug Microencapsulation PK — Phase 1026.

Predicts drug release kinetics from microencapsulated formulations (polymer shell
microspheres) using empirical/semi-mechanistic release models.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# Valid polymers
_VALID_POLYMERS = {"PLGA", "ethylcellulose", "Eudragit", "gelatin", "chitosan"}

# Polymer property table: (base_t50_h, mechanism_raw, erosion_contrib, burst_base_pct)
_POLYMER_PROPS: dict[str, tuple[float, float, float]] = {
    #                   base_t50_h  erosion_contrib  burst_base_pct
    "PLGA": (168.0, 0.5, 5.0),
    "ethylcellulose": (48.0, 0.1, 10.0),
    "Eudragit": (24.0, 0.2, 8.0),
    "gelatin": (12.0, 0.8, 15.0),
    "chitosan": (36.0, 0.6, 12.0),
}


@dataclass(frozen=True)
class MicroencapsulationResult:
    drug_name: str
    polymer: str
    particle_size_um: float
    drug_loading_pct: float
    burst_release_pct: float
    t50_h: float
    t90_h: float
    release_mechanism: str
    release_rate_constant_per_h: float
    predicted_bioavailability: float
    notes: str


def predict_microencapsulation_pk(
    drug_name: str,
    mw_Da: float,
    logp: float,
    polymer: str = "PLGA",
    particle_size_um: float = 50.0,
    drug_loading_pct: float = 20.0,
    shell_thickness_um: float = 5.0,
    dissolution_medium_ph: float = 7.4,
) -> MicroencapsulationResult:
    """Predict release kinetics of a drug from a microencapsulated formulation.

    Parameters
    ----------
    drug_name : str
        Name of the drug.
    mw_Da : float
        Molecular weight (Da).
    logp : float
        Octanol-water partition coefficient (logP).
    polymer : str
        Polymer type: one of PLGA, ethylcellulose, Eudragit, gelatin, chitosan.
    particle_size_um : float
        Microsphere diameter (µm).
    drug_loading_pct : float
        Drug loading (wt%, 0–100 exclusive).
    shell_thickness_um : float
        Polymer shell thickness (µm).
    dissolution_medium_ph : float
        pH of dissolution medium.

    Returns
    -------
    MicroencapsulationResult
    """
    # Validation
    if mw_Da <= 0:
        raise ValueError("mw_Da must be > 0")
    if particle_size_um <= 0:
        raise ValueError("particle_size_um must be > 0")
    if drug_loading_pct <= 0 or drug_loading_pct >= 100:
        raise ValueError("drug_loading_pct must be in (0, 100)")
    if shell_thickness_um <= 0:
        raise ValueError("shell_thickness_um must be > 0")
    if polymer not in _VALID_POLYMERS:
        raise ValueError(f"polymer must be one of {sorted(_VALID_POLYMERS)}, got '{polymer}'")

    base_t50_h, erosion_contrib, burst_base_pct = _POLYMER_PROPS[polymer]

    # ---- Adjustments to t50 ----

    # Particle size: larger → slower release
    size_factor = math.sqrt(particle_size_um / 50.0)

    # Drug loading: higher loading → slightly faster (pores form)
    loading_factor = max(0.5, 1.0 - (drug_loading_pct - 20.0) * 0.01)

    # logP: lipophilic drugs partition into polymer → slower release
    logp_factor = 1.0 + 0.05 * max(0.0, logp)

    # MW: larger MW → slower diffusion, clamped [0.5, 3.0]
    mw_raw = 1.0 + 0.001 * (mw_Da - 300.0)
    mw_factor = max(0.5, min(3.0, mw_raw))

    t50_final = base_t50_h * size_factor * loading_factor * logp_factor * mw_factor

    # ---- Burst release ----
    burst_pct = burst_base_pct + shell_thickness_um * 0.1
    burst_pct = max(0.0, min(30.0, burst_pct))

    # ---- t90 from first-order kinetics ----
    # t50 = ln(2)/k, t90 = ln(10)/k → t90 = t50 * ln(10)/ln(2)
    t90_final = t50_final * math.log(10.0) / math.log(2.0)

    # ---- Release mechanism ----
    if erosion_contrib >= 0.6:
        release_mechanism = "erosion"
    elif erosion_contrib <= 0.2:
        release_mechanism = "diffusion"
    else:
        release_mechanism = "combined"

    # ---- Rate constant ----
    k_release = math.log(2.0) / t50_final

    # ---- Predicted bioavailability ----
    f_base = 0.8
    if polymer == "PLGA":
        f_base = 0.85
    f_val = f_base + 0.02 * max(0.0, logp) - drug_loading_pct * 0.002
    predicted_f = max(0.3, min(0.95, f_val))

    notes = (
        f"Microencapsulation PK: {drug_name} in {polymer}. "
        f"Particle size {particle_size_um} µm, loading {drug_loading_pct}%, "
        f"shell {shell_thickness_um} µm. "
        f"t50={t50_final:.1f} h, t90={t90_final:.1f} h, "
        f"burst={burst_pct:.1f}%, mechanism={release_mechanism}."
    )

    return MicroencapsulationResult(
        drug_name=drug_name,
        polymer=polymer,
        particle_size_um=particle_size_um,
        drug_loading_pct=drug_loading_pct,
        burst_release_pct=burst_pct,
        t50_h=t50_final,
        t90_h=t90_final,
        release_mechanism=release_mechanism,
        release_rate_constant_per_h=k_release,
        predicted_bioavailability=predicted_f,
        notes=notes,
    )


def compare_polymers(
    drug_name: str,
    mw_Da: float,
    logp: float,
    particle_size_um: float = 50.0,
    drug_loading_pct: float = 20.0,
) -> list:
    """Compare all 5 polymers for a given drug, sorted by t50 ascending.

    Parameters
    ----------
    drug_name : str
        Name of the drug.
    mw_Da : float
        Molecular weight (Da).
    logp : float
        logP value.
    particle_size_um : float
        Microsphere diameter (µm).
    drug_loading_pct : float
        Drug loading (wt%).

    Returns
    -------
    list[MicroencapsulationResult]
        Results for all 5 polymers, sorted by t50_h ascending.
    """
    results = []
    for polymer in _VALID_POLYMERS:
        result = predict_microencapsulation_pk(
            drug_name=drug_name,
            mw_Da=mw_Da,
            logp=logp,
            polymer=polymer,
            particle_size_um=particle_size_um,
            drug_loading_pct=drug_loading_pct,
        )
        results.append(result)
    results.sort(key=lambda r: r.t50_h)
    return results
