"""CNS PBPK / Blood-Brain Barrier model.

Predicts CNS penetration based on physicochemical properties and simulates
brain compartment pharmacokinetics using a simple forward-Euler ODE approach.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from omega_pbpk._compat import np_trapz


@dataclass(frozen=True)
class CNSPenetrationResult:
    """Result of BBB penetration prediction."""

    drug_name: str
    logP: float
    MW: float
    psa: float
    is_pgp_substrate: bool
    kp_uu_brain: float
    efflux_ratio: float
    cns_classification: str
    risk_factors: list[str]


def predict_bbb_penetration(
    drug_name: str,
    logP: float,
    MW: float,
    psa: float,
    is_pgp_substrate: bool = False,
    efflux_ratio: float = 3.0,
) -> CNSPenetrationResult:
    """Predict blood-brain barrier penetration from physicochemical properties."""
    logP_factor = 1.0 / (1.0 + math.exp(-(logP - 1.5) * 0.8))
    mw_factor = math.exp(-max(MW - 400, 0) / 300)
    psa_factor = math.exp(-max(psa - 60, 0) / 60)

    eff = max(efflux_ratio, 1.0) if is_pgp_substrate else 1.0
    kp_uu_brain = float(np.clip(logP_factor * mw_factor * psa_factor / eff, 0.001, 1.0))

    if kp_uu_brain > 0.3:
        cns_classification = "CNS+"
    elif kp_uu_brain >= 0.1:
        cns_classification = "CNS+/-"
    else:
        cns_classification = "CNS-"

    risk_factors: list[str] = []
    if MW > 400:
        risk_factors.append(f"High MW (>{MW:.0f} Da)")
    if psa > 60:
        risk_factors.append(f"High PSA ({psa:.0f} \u00c5\u00b2)")
    if is_pgp_substrate:
        risk_factors.append(f"P-gp substrate (efflux ratio {efflux_ratio:.1f}x)")
    if logP < 1.0:
        risk_factors.append(f"Low logP ({logP:.1f})")

    return CNSPenetrationResult(
        drug_name=drug_name,
        logP=logP,
        MW=MW,
        psa=psa,
        is_pgp_substrate=is_pgp_substrate,
        kp_uu_brain=kp_uu_brain,
        efflux_ratio=eff,
        cns_classification=cns_classification,
        risk_factors=risk_factors,
    )


def simulate_cns_pk(
    plasma_times: list[float],
    plasma_conc: list[float],
    kp_uu_brain: float,
    fup: float = 1.0,
    ke_brain: float = 0.1,
    dt_h: float = 0.05,
) -> dict:
    """Simulate brain compartment PK using forward Euler integration."""
    t_start = plasma_times[0]
    t_end = plasma_times[-1]
    n_steps = int(math.ceil((t_end - t_start) / dt_h))
    k_in = ke_brain * kp_uu_brain

    times = np.linspace(t_start, t_end, n_steps + 1)
    cp_interp = np.interp(times, plasma_times, plasma_conc)

    c_brain = np.zeros(len(times))
    for i in range(len(times) - 1):
        dt = times[i + 1] - times[i]
        c_plasma_u = fup * cp_interp[i]
        c_brain[i + 1] = c_brain[i] + dt * (k_in * c_plasma_u - ke_brain * c_brain[i])

    auc_brain = float(np_trapz(c_brain, times))
    auc_plasma = float(np_trapz(cp_interp, times))
    ratio = auc_brain / auc_plasma if auc_plasma > 0 else 0.0

    return {
        "times_h": times.tolist(),
        "brain_conc": c_brain.tolist(),
        "plasma_conc": cp_interp.tolist(),
        "kp_uu_ss": kp_uu_brain,
        "auc_brain": auc_brain,
        "auc_plasma": auc_plasma,
        "brain_plasma_auc_ratio": ratio,
    }


def assess_cns_exposure(
    drug_name: str,
    logP: float,
    MW: float,
    psa: float,
    plasma_times: list[float],
    plasma_conc: list[float],
    fup: float = 1.0,
    is_pgp_substrate: bool = False,
    efflux_ratio: float = 3.0,
) -> dict:
    """Combined BBB penetration prediction and CNS PK simulation."""
    penetration = predict_bbb_penetration(
        drug_name=drug_name,
        logP=logP,
        MW=MW,
        psa=psa,
        is_pgp_substrate=is_pgp_substrate,
        efflux_ratio=efflux_ratio,
    )
    pk = simulate_cns_pk(
        plasma_times=plasma_times,
        plasma_conc=plasma_conc,
        kp_uu_brain=penetration.kp_uu_brain,
        fup=fup,
    )
    return {"penetration": penetration, "pk": pk}


__all__ = [
    "CNSPenetrationResult",
    "predict_bbb_penetration",
    "simulate_cns_pk",
    "assess_cns_exposure",
]
