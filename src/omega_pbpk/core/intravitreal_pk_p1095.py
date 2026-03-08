"""Intravitreal injection PK — Phase 1095.

Detailed 3-compartment intravitreal PK for anti-VEGF biologics and small molecules.
  Vitreous → Aqueous humor → Systemic plasma

MW-dependent elimination:
  k_vit_aq = 0.003 + 0.05 * exp(-mw_kDa / 50.0)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class IntravitreалPKResult:
    """Result of intravitreal PK simulation (Phase 1095)."""

    drug_name: str
    dose_mg: float
    mw_kDa: float
    times_h: list = field(default_factory=list)
    a_vitreous_mg: list = field(default_factory=list)
    a_aqueous_mg: list = field(default_factory=list)
    c_plasma_mg_L: list = field(default_factory=list)
    cmax_vitreous_mg: float = 0.0
    t_half_vitreous_h: float = 0.0
    auc_vitreous: float = 0.0
    cmax_aqueous_mg: float = 0.0
    tmax_aqueous_h: float = 0.0
    cmax_plasma_mg_L: float = 0.0
    systemic_exposure_ratio: float = 0.0
    dosing_interval_recommended_days: float = 0.0
    notes: str = ""


def _trapz(times: list[float], values: list[float]) -> float:
    """Trapezoidal integration."""
    auc = 0.0
    for i in range(len(times) - 1):
        auc += 0.5 * (values[i] + values[i + 1]) * (times[i + 1] - times[i])
    return auc


def _k_vit_aq(mw_kDa: float) -> float:
    """MW-dependent vitreous→aqueous transfer rate (h⁻¹)."""
    return 0.003 + 0.05 * math.exp(-mw_kDa / 50.0)


def simulate_intravitreal_pk(
    drug_name: str,
    dose_mg: float,
    mw_kDa: float = 150.0,
    vd_sys_L: float = 3.5,
    cl_sys_L_per_h: float = 0.5,
    t_end_h: float = 720.0,
    dt_h: float = 1.0,
) -> IntravitreалPKResult:
    """Simulate intravitreal injection PK.

    Parameters
    ----------
    drug_name : str
        Name of the drug.
    dose_mg : float
        Dose injected directly into the vitreous (mg).
    mw_kDa : float
        Molecular weight in kDa. Controls vitreous drainage rate.
    vd_sys_L : float
        Systemic volume of distribution (L).
    cl_sys_L_per_h : float
        Systemic clearance (L/h).
    t_end_h : float
        Simulation duration (h). Default 720 h (30 days).
    dt_h : float
        Forward-Euler time step (h).

    Returns
    -------
    IntravitreалPKResult
    """
    if dose_mg <= 0:
        raise ValueError(f"dose_mg must be > 0, got {dose_mg}")
    if mw_kDa <= 0:
        raise ValueError(f"mw_kDa must be > 0, got {mw_kDa}")
    if vd_sys_L <= 0:
        raise ValueError(f"vd_sys_L must be > 0, got {vd_sys_L}")
    if cl_sys_L_per_h <= 0:
        raise ValueError(f"cl_sys_L_per_h must be > 0, got {cl_sys_L_per_h}")

    # Rate constants (h⁻¹)
    kvaq = _k_vit_aq(mw_kDa)  # vitreous → aqueous
    k_vit_elim = 0.001  # non-specific elimination from vitreous
    k_aq_sys = 0.3  # aqueous → systemic plasma
    k_aq_elim = 0.1  # aqueous elimination
    ke_sys = cl_sys_L_per_h / vd_sys_L  # systemic elimination rate

    t_half_vit = math.log(2.0) / (kvaq + k_vit_elim)

    # Initial amounts (mg)
    A_vit = dose_mg
    A_aq = 0.0
    C_plasma = 0.0  # mg/L

    n_steps = max(1, int(round(t_end_h / dt_h)))

    times: list[float] = []
    a_vit_list: list[float] = []
    a_aq_list: list[float] = []
    c_pl_list: list[float] = []

    cmax_aq = 0.0
    tmax_aq = 0.0
    cmax_pl = 0.0

    for step in range(n_steps + 1):
        t = step * dt_h
        times.append(t)
        a_vit_list.append(A_vit)
        a_aq_list.append(A_aq)
        c_pl_list.append(C_plasma)

        if A_aq > cmax_aq:
            cmax_aq = A_aq
            tmax_aq = t

        if C_plasma > cmax_pl:
            cmax_pl = C_plasma

        if step < n_steps:
            dA_vit = -(kvaq + k_vit_elim) * A_vit
            dA_aq = kvaq * A_vit - (k_aq_sys + k_aq_elim) * A_aq
            dC_pl = k_aq_sys * A_aq / vd_sys_L - ke_sys * C_plasma

            A_vit = max(0.0, A_vit + dA_vit * dt_h)
            A_aq = max(0.0, A_aq + dA_aq * dt_h)
            C_plasma = max(0.0, C_plasma + dC_pl * dt_h)

    auc_vit = _trapz(times, a_vit_list)

    # systemic_exposure_ratio: cmax_plasma relative to dose/Vd (normalised plasma peak)
    dose_over_vd = dose_mg / vd_sys_L
    systemic_exposure_ratio = cmax_pl / dose_over_vd if dose_over_vd > 0 else 0.0

    dosing_interval_days = 3.0 * t_half_vit / 24.0

    is_large = mw_kDa > 5.0
    mol_type = (
        "mAb/biologic" if mw_kDa >= 100 else ("large molecule" if is_large else "small molecule")
    )
    notes = (
        f"{mol_type} (MW={mw_kDa} kDa). "
        f"k_vit_aq={kvaq:.5f}/h, k_vit_elim={k_vit_elim:.4f}/h. "
        f"Vitreous t½={t_half_vit:.1f} h ({t_half_vit / 24:.1f} days). "
        f"Recommended dosing interval ≈{dosing_interval_days:.1f} days."
    )

    return IntravitreалPKResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        mw_kDa=mw_kDa,
        times_h=times,
        a_vitreous_mg=a_vit_list,
        a_aqueous_mg=a_aq_list,
        c_plasma_mg_L=c_pl_list,
        cmax_vitreous_mg=dose_mg,
        t_half_vitreous_h=t_half_vit,
        auc_vitreous=auc_vit,
        cmax_aqueous_mg=cmax_aq,
        tmax_aqueous_h=tmax_aq,
        cmax_plasma_mg_L=cmax_pl,
        systemic_exposure_ratio=systemic_exposure_ratio,
        dosing_interval_recommended_days=dosing_interval_days,
        notes=notes,
    )


def compare_molecule_sizes(
    drug_name: str,
    dose_mg: float,
    mw_list: list[float],
) -> list[IntravitreалPKResult]:
    """Simulate IVT PK for multiple MW values, sorted by t_half_vitreous_h descending.

    Parameters
    ----------
    drug_name : str
        Drug name prefix (MW will be appended).
    dose_mg : float
        Dose in mg (same for all).
    mw_list : list[float]
        List of molecular weights (kDa) to compare.

    Returns
    -------
    list[IntravitreалPKResult]
        Results sorted by t_half_vitreous_h descending (large MW → longer t½).
    """
    results = []
    for mw in mw_list:
        result = simulate_intravitreal_pk(
            drug_name=f"{drug_name}_{mw:.0f}kDa",
            dose_mg=dose_mg,
            mw_kDa=mw,
        )
        results.append(result)
    results.sort(key=lambda r: r.t_half_vitreous_h, reverse=True)
    return results
