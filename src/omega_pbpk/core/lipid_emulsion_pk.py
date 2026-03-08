"""Phase 470 — Intravenous Lipid Emulsion (ILE) PK Model.

Models drug rescue therapy with IV lipid emulsion (lipid sink effect).
Used for lipophilic drug toxicity management.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class LipidEmulsionPKResult:
    """Result of IV lipid emulsion drug rescue simulation."""

    drug_name: str
    dose_mg: float
    logP: float
    ile_dose_mL: float
    times_h: list
    c_total_mg_L: list
    c_free_mg_L: list
    cmax_free_mg_L: float
    auc_free_mg_h_per_L: float
    kp_ile: float
    f_free: float
    sequestration_pct: float
    vd_effective_L: float
    t_half_effective_h: float
    rescue_efficacy: str
    notes: str


def _rescue_efficacy(sequestration_pct: float) -> str:
    """Classify rescue efficacy based on sequestration percentage."""
    if sequestration_pct < 20.0:
        return "low"
    if sequestration_pct <= 60.0:
        return "moderate"
    return "high"


def simulate_lipid_emulsion_rescue(
    drug_name: str,
    dose_mg: float,
    logP: float,
    cl_sys_L_per_h: float,
    vd_sys_L: float = 50.0,
    ile_dose_mL: float = 100.0,
    ile_lipid_fraction: float = 0.20,
    t_end_h: float = 6.0,
    dt_h: float = 0.05,
) -> LipidEmulsionPKResult:
    """Simulate IV lipid emulsion drug rescue PK.

    Parameters
    ----------
    drug_name : str
        Name of the drug.
    dose_mg : float
        Toxic drug dose already in body (mg).
    logP : float
        Log octanol-water partition coefficient.
    cl_sys_L_per_h : float
        Systemic clearance of the drug (L/h).
    vd_sys_L : float
        Systemic volume of distribution (L).
    ile_dose_mL : float
        Volume of ILE administered IV (mL).
    ile_lipid_fraction : float
        Lipid content fraction of ILE (e.g., 0.20 for 20%).
    t_end_h : float
        Simulation duration (h).
    dt_h : float
        Time step (h).

    Returns
    -------
    LipidEmulsionPKResult
    """
    if dose_mg <= 0.0:
        raise ValueError(f"dose_mg must be > 0, got {dose_mg}")
    if cl_sys_L_per_h <= 0.0:
        raise ValueError(f"cl_sys_L_per_h must be > 0, got {cl_sys_L_per_h}")
    if vd_sys_L <= 0.0:
        raise ValueError(f"vd_sys_L must be > 0, got {vd_sys_L}")
    if ile_dose_mL <= 0.0:
        raise ValueError(f"ile_dose_mL must be > 0, got {ile_dose_mL}")
    if not (0.0 < ile_lipid_fraction <= 1.0):
        raise ValueError(f"ile_lipid_fraction must be in (0, 1], got {ile_lipid_fraction}")

    # Volume of lipid in plasma (L)
    v_lipid_L = ile_dose_mL * ile_lipid_fraction / 1000.0

    # Partition coefficient ILE/water: Kp_ile = 10^(0.7 * logP), clamped [1, 1000]
    kp_ile = max(1.0, min(1000.0, 10.0 ** (0.7 * logP)))

    # Effective volume of distribution with ILE
    vd_effective_L = vd_sys_L + v_lipid_L * kp_ile

    # Free fraction in presence of ILE
    f_free = vd_sys_L / vd_effective_L

    # Sequestration fraction
    sequestration_pct = (1.0 - f_free) * 100.0

    # Effective half-life (clearance only eliminates free drug)
    # Rate constant k = CL_eff / Vd_effective
    k_elim = cl_sys_L_per_h / vd_effective_L
    t_half_effective_h = math.log(2.0) / k_elim if k_elim > 0.0 else float("inf")

    # Build time course using analytical solution:
    # C_total(t) = (dose / Vd_eff) * exp(-k_elim * t)
    # C_free(t)  = C_total(t) * f_free
    times_h: list[float] = []
    c_total_mg_L: list[float] = []
    c_free_mg_L: list[float] = []

    n_steps = max(1, round(t_end_h / dt_h)) + 1
    c0_total = dose_mg / vd_effective_L

    for i in range(n_steps):
        t_i = i * dt_h
        if t_i > t_end_h + 1e-9:
            break
        c_tot = c0_total * math.exp(-k_elim * t_i)
        c_free = c_tot * f_free
        times_h.append(t_i)
        c_total_mg_L.append(c_tot)
        c_free_mg_L.append(c_free)

    cmax_free_mg_L = c_free_mg_L[0] if c_free_mg_L else 0.0

    # Trapezoidal AUC for free drug
    auc_free = sum(
        0.5 * (c_free_mg_L[i] + c_free_mg_L[i - 1]) * (times_h[i] - times_h[i - 1])
        for i in range(1, len(times_h))
    )

    efficacy = _rescue_efficacy(sequestration_pct)

    notes_parts = [
        f"ILE lipid volume = {v_lipid_L:.4g} L.",
        f"Kp_ILE = {kp_ile:.4g} (logP={logP:.2f}).",
        f"Vd_eff = {vd_effective_L:.4g} L vs. baseline {vd_sys_L:.4g} L.",
        f"Free fraction = {f_free:.4f} ({sequestration_pct:.1f}% sequestered).",
        f"Rescue efficacy: {efficacy}.",
    ]

    return LipidEmulsionPKResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        logP=logP,
        ile_dose_mL=ile_dose_mL,
        times_h=times_h,
        c_total_mg_L=c_total_mg_L,
        c_free_mg_L=c_free_mg_L,
        cmax_free_mg_L=cmax_free_mg_L,
        auc_free_mg_h_per_L=auc_free,
        kp_ile=kp_ile,
        f_free=f_free,
        sequestration_pct=sequestration_pct,
        vd_effective_L=vd_effective_L,
        t_half_effective_h=t_half_effective_h,
        rescue_efficacy=efficacy,
        notes=" ".join(notes_parts),
    )


def screen_ile_doses(
    drug_name: str,
    dose_mg: float,
    logP: float,
    cl_sys_L_per_h: float,
    ile_doses_mL: list | None = None,
) -> list[LipidEmulsionPKResult]:
    """Screen multiple ILE doses sorted by sequestration_pct descending.

    Parameters
    ----------
    drug_name : str
        Name of the drug.
    dose_mg : float
        Toxic drug dose already in body (mg).
    logP : float
        Log octanol-water partition coefficient.
    cl_sys_L_per_h : float
        Systemic clearance of the drug (L/h).
    ile_doses_mL : list or None
        ILE volumes to screen (mL). Defaults to [50, 100, 200, 500].

    Returns
    -------
    list[LipidEmulsionPKResult]
        Results sorted by sequestration_pct descending (highest first).
    """
    if ile_doses_mL is None:
        ile_doses_mL = [50.0, 100.0, 200.0, 500.0]

    results = [
        simulate_lipid_emulsion_rescue(
            drug_name=drug_name,
            dose_mg=dose_mg,
            logP=logP,
            cl_sys_L_per_h=cl_sys_L_per_h,
            ile_dose_mL=float(d),
        )
        for d in ile_doses_mL
    ]
    return sorted(results, key=lambda r: r.sequestration_pct, reverse=True)


__all__ = [
    "LipidEmulsionPKResult",
    "simulate_lipid_emulsion_rescue",
    "screen_ile_doses",
]


if __name__ == "__main__":  # pragma: no cover
    result = simulate_lipid_emulsion_rescue(
        drug_name="TestDrug",
        dose_mg=100.0,
        logP=3.0,
        cl_sys_L_per_h=5.0,
        vd_sys_L=50.0,
        ile_dose_mL=100.0,
    )
    print(result.notes)
    print(f"f_free={result.f_free:.4f}, seq%={result.sequestration_pct:.1f}%")
    print(f"t_half_eff={result.t_half_effective_h:.2f} h")
    print("Smoke test passed.")
