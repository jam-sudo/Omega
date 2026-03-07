"""Amorphous solid dispersion (ASD) drug release and supersaturation kinetics — Phase 1136.

Models spring-and-parachute supersaturation: rapid dissolution above
crystalline solubility followed by eventual crystallization (parachute phase).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# Polymer parameters: name -> {msr_factor, k_diss (1/h), k_cryst (1/h)}
_POLYMER_PARAMS: dict[str, dict[str, float]] = {
    "HPMC": {"msr_factor": 3.0, "k_diss": 0.5, "k_cryst": 0.1},
    "PVP": {"msr_factor": 5.0, "k_diss": 1.0, "k_cryst": 0.2},
    "HPMC-AS": {"msr_factor": 8.0, "k_diss": 0.8, "k_cryst": 0.05},
    "EUDRAGIT": {"msr_factor": 4.0, "k_diss": 0.3, "k_cryst": 0.15},
}

_VALID_POLYMERS = frozenset(_POLYMER_PARAMS.keys())


@dataclass(frozen=True)
class ASDReleaseResult:
    """Result of ASD drug release prediction."""

    drug_name: str
    intrinsic_solubility_mg_mL: float
    polymer_type: str
    logP: float
    drug_loading_pct: float
    max_supersaturation_mg_mL: float
    supersaturation_ratio: float
    times_h: tuple
    concentration_mg_mL: tuple
    tmax_supersaturation_h: float
    auc_supersaturation_mg_h_per_mL: float
    parachute_duration_h: float
    polymer_recommendation: str
    notes: str


def _trapz(times: tuple, concs: tuple) -> float:
    """Manual trapezoidal integration."""
    auc = 0.0
    for i in range(1, len(times)):
        dt = times[i] - times[i - 1]
        auc += 0.5 * (concs[i - 1] + concs[i]) * dt
    return auc


def predict_asd_release(
    drug_name: str,
    intrinsic_solubility_mg_mL: float,
    polymer_type: str,
    logP: float,
    drug_loading_pct: float,
    t_end_h: float = 4.0,
    n_points: int = 40,
) -> ASDReleaseResult:
    """Predict ASD drug release and supersaturation profile.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    intrinsic_solubility_mg_mL:
        Crystalline solubility in mg/mL.
    polymer_type:
        One of: HPMC, PVP, HPMC-AS, EUDRAGIT.
    logP:
        Octanol-water partition coefficient (must be in [-5, 10]).
    drug_loading_pct:
        Drug loading in the ASD formulation (0 < x <= 100).
    t_end_h:
        Time horizon for dissolution profile in hours.
    n_points:
        Number of time points in the output profile.

    Returns
    -------
    ASDReleaseResult
    """
    # Validation
    if intrinsic_solubility_mg_mL <= 0:
        raise ValueError(
            f"intrinsic_solubility_mg_mL must be > 0, got {intrinsic_solubility_mg_mL}"
        )
    if polymer_type not in _VALID_POLYMERS:
        raise ValueError(
            f"polymer_type must be one of {sorted(_VALID_POLYMERS)}, got {polymer_type!r}"
        )
    if not (0 < drug_loading_pct <= 100):
        raise ValueError(f"drug_loading_pct must be in (0, 100], got {drug_loading_pct}")
    if not (-5 <= logP <= 10):
        raise ValueError(f"logP must be in [-5, 10], got {logP}")

    params = _POLYMER_PARAMS[polymer_type]
    msr_factor = params["msr_factor"]
    k_diss = params["k_diss"]
    k_cryst = params["k_cryst"]

    # Drug loading penalty
    if drug_loading_pct > 30.0:
        msr_factor = msr_factor * 0.8

    # Maximum supersaturation concentration
    css = msr_factor * intrinsic_solubility_mg_mL

    # t_peak from crystallization rate
    # C(t) = Css * (1 - exp(-k_diss * t)) * exp(-k_cryst * max(0, t - t_peak))
    # t_peak = 1 / k_cryst
    t_peak = 1.0 / k_cryst

    # Build time profile
    if n_points < 2:
        n_points = 2
    dt = t_end_h / (n_points - 1)
    times_list: list[float] = []
    conc_list: list[float] = []

    for i in range(n_points):
        t = i * dt
        spring = 1.0 - math.exp(-k_diss * t)
        decay = math.exp(-k_cryst * max(0.0, t - t_peak))
        c = css * spring * decay
        times_list.append(t)
        conc_list.append(c)

    times_h = tuple(times_list)
    concentration_mg_mL = tuple(conc_list)

    # Peak supersaturation
    max_conc = max(concentration_mg_mL)
    tmax_idx = conc_list.index(max_conc)
    tmax_h = times_list[tmax_idx]

    supersaturation_ratio = max_conc / intrinsic_solubility_mg_mL

    # AUC
    auc = _trapz(times_h, concentration_mg_mL)

    # Parachute duration: time until C falls to 2x intrinsic solubility after peak
    threshold = 2.0 * intrinsic_solubility_mg_mL
    parachute_duration = 0.0
    for i in range(tmax_idx, len(times_list)):
        if conc_list[i] < threshold:
            # Interpolate
            if i > tmax_idx:
                t_prev = times_list[i - 1]
                c_prev = conc_list[i - 1]
                t_curr = times_list[i]
                c_curr = conc_list[i]
                if c_curr < c_prev:
                    frac = (c_prev - threshold) / (c_prev - c_curr)
                    parachute_duration = (t_prev + frac * (t_curr - t_prev)) - tmax_h
                else:
                    parachute_duration = t_curr - tmax_h
            break
        if i == len(times_list) - 1:
            # Never fell below threshold in simulation window
            parachute_duration = times_list[-1] - tmax_h

    parachute_duration = max(0.0, parachute_duration)

    # Polymer recommendation: highest auc_supersaturation → HPMC-AS typically
    # Simple rule: prefer lowest k_cryst polymer
    best_polymer = min(_POLYMER_PARAMS.keys(), key=lambda p: _POLYMER_PARAMS[p]["k_cryst"])
    polymer_recommendation = best_polymer

    notes = (
        f"polymer={polymer_type}, MSR={msr_factor:.1f}x, "
        f"Css={css:.3f} mg/mL, k_diss={k_diss}/h, k_cryst={k_cryst}/h. "
        f"{'Loading penalty applied (>30%).' if drug_loading_pct > 30.0 else ''}"
    ).strip()

    return ASDReleaseResult(
        drug_name=drug_name,
        intrinsic_solubility_mg_mL=intrinsic_solubility_mg_mL,
        polymer_type=polymer_type,
        logP=logP,
        drug_loading_pct=drug_loading_pct,
        max_supersaturation_mg_mL=max_conc,
        supersaturation_ratio=supersaturation_ratio,
        times_h=times_h,
        concentration_mg_mL=concentration_mg_mL,
        tmax_supersaturation_h=tmax_h,
        auc_supersaturation_mg_h_per_mL=auc,
        parachute_duration_h=parachute_duration,
        polymer_recommendation=polymer_recommendation,
        notes=notes,
    )


def compare_polymers(
    drug_name: str,
    intrinsic_solubility_mg_mL: float,
    logP: float,
    drug_loading_pct: float,
) -> list[ASDReleaseResult]:
    """Compare all 4 polymer types, sorted by AUC supersaturation descending.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    intrinsic_solubility_mg_mL:
        Crystalline solubility in mg/mL.
    logP:
        Octanol-water partition coefficient.
    drug_loading_pct:
        Drug loading percentage.

    Returns
    -------
    list of 4 ASDReleaseResult sorted by auc_supersaturation_mg_h_per_mL descending.
    """
    results = []
    for polymer in sorted(_VALID_POLYMERS):
        result = predict_asd_release(
            drug_name=drug_name,
            intrinsic_solubility_mg_mL=intrinsic_solubility_mg_mL,
            polymer_type=polymer,
            logP=logP,
            drug_loading_pct=drug_loading_pct,
        )
        results.append(result)

    results.sort(key=lambda r: r.auc_supersaturation_mg_h_per_mL, reverse=True)
    return results
