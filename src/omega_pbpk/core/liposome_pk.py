"""Liposome PK model for IV-administered phospholipid vesicles.

Models drug encapsulation, controlled release, and PEGylation effects on
mononuclear phagocyte system (MPS) clearance. Distinct from LNP and nano_pbpk
modules: liposomes are phospholipid bilayer vesicles with different kinetics.

Key features
------------
- PEGylation-extended circulation half-life (stealth liposomes)
- Size-dependent MPS/liver uptake
- Dual-compartment tracking: liposome carrier + free drug
- Forward Euler ODE integration

References
----------
- Allen TM & Cullis PR. Science 2004 (liposome drug delivery)
- Drummond DC et al., Pharmacol Rev 1999 (liposome pharmacokinetics)
- Gabizon A et al., Clin Cancer Res 1999 (Doxil PEGylated liposome PK)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from omega_pbpk._compat import np_trapz

__all__ = ["LiposomePKResult", "simulate_liposome_pk", "compare_liposome_sizes"]


@dataclass(frozen=True)
class LiposomePKResult:
    """Results from a liposome PK simulation."""

    formulation_name: str
    dose_mg_lipid: float
    drug_loading_pct: float
    size_nm: float
    peg_fraction: float
    times_h: list[float]
    c_liposome_mg_L: list[float]
    c_free_drug_mg_L: list[float]
    c_encapsulated_mg_L: list[float]
    cmax_free_drug: float
    cmax_liposome: float
    auc_free_drug: float
    auc_liposome: float
    t_half_liposome_h: float
    release_t_half_h: float
    liver_uptake_pct: float
    notes: str


def simulate_liposome_pk(
    formulation_name: str,
    dose_mg_lipid: float,
    size_nm: float,
    drug_loading_pct: float = 10.0,
    peg_fraction: float = 0.05,
    k_release_per_h: float = 0.05,
    cl_free_drug_L_per_h: float = 5.0,
    vd_free_drug_L: float = 20.0,
    t_end_h: float = 72.0,
    dt_h: float = 0.1,
) -> LiposomePKResult:
    """Simulate liposome PK after IV administration.

    Parameters
    ----------
    formulation_name:
        Label for the formulation.
    dose_mg_lipid:
        Total lipid dose in mg.
    size_nm:
        Liposome diameter in nm.
    drug_loading_pct:
        Drug mass as percent of lipid mass (0-100).
    peg_fraction:
        Mole fraction of PEG lipid (0-1); higher = stealth effect.
    k_release_per_h:
        First-order drug release rate from liposome (h^-1).
    cl_free_drug_L_per_h:
        Clearance of free drug (L/h).
    vd_free_drug_L:
        Volume of distribution for free drug (L).
    t_end_h:
        Simulation duration (h).
    dt_h:
        ODE step size (h).

    Returns
    -------
    LiposomePKResult
    """
    # --- Validation ---
    if dose_mg_lipid <= 0:
        raise ValueError("dose_mg_lipid must be > 0")
    if size_nm <= 0:
        raise ValueError("size_nm must be > 0")
    if not (0 < drug_loading_pct < 100):
        raise ValueError("drug_loading_pct must be in (0, 100)")
    if not (0.0 <= peg_fraction <= 1.0):
        raise ValueError("peg_fraction must be in [0, 1]")
    if k_release_per_h < 0:
        raise ValueError("k_release_per_h must be >= 0")
    if cl_free_drug_L_per_h <= 0:
        raise ValueError("cl_free_drug_L_per_h must be > 0")
    if vd_free_drug_L <= 0:
        raise ValueError("vd_free_drug_L must be > 0")

    # --- PK parameters ---
    base_k_cl = 0.1  # h^-1 for 100 nm non-PEGylated
    size_factor = 1.0 + max(0.0, (size_nm - 100.0) / 200.0)
    peg_factor = max(0.1, 1.0 - peg_fraction * 10.0)
    k_cl_lipo = base_k_cl * size_factor * peg_factor

    k_liver = k_cl_lipo * 0.7  # 70% liver MPS uptake
    # k_other  = k_cl_lipo * 0.3  # remaining clearance (not tracked separately)

    drug_dose = dose_mg_lipid * drug_loading_pct / 100.0

    V_lipo = 3.0  # plasma volume for liposome (L)
    ke_free = cl_free_drug_L_per_h / vd_free_drug_L

    # --- Initial conditions ---
    a_lipo = dose_mg_lipid  # total liposome mass in plasma (mg)
    a_enc = drug_dose        # encapsulated drug mass (mg)
    c_free = 0.0             # free drug plasma concentration (mg/L)

    n_steps = int(round(t_end_h / dt_h))

    times: list[float] = [0.0]
    c_lipo_list: list[float] = [a_lipo / V_lipo]
    c_free_list: list[float] = [c_free]
    c_enc_list: list[float] = [a_enc / V_lipo]

    for _ in range(n_steps):
        da_lipo = -(k_cl_lipo + k_release_per_h) * a_lipo * dt_h
        da_enc = -k_release_per_h * a_enc * dt_h
        dc_free = (k_release_per_h * a_enc / vd_free_drug_L - ke_free * c_free) * dt_h

        a_lipo = max(0.0, a_lipo + da_lipo)
        a_enc = max(0.0, a_enc + da_enc)
        c_free = max(0.0, c_free + dc_free)

        times.append(times[-1] + dt_h)
        c_lipo_list.append(a_lipo / V_lipo)
        c_free_list.append(c_free)
        c_enc_list.append(a_enc / V_lipo)

    cmax_free = max(c_free_list)
    cmax_lipo = max(c_lipo_list)
    auc_free = float(np_trapz(c_free_list, times))
    auc_lipo = float(np_trapz(c_lipo_list, times))

    t_half_lipo = 0.693 / k_cl_lipo
    release_t_half = 0.693 / k_release_per_h if k_release_per_h > 0 else float("inf")

    # Liver uptake: fraction cleared via liver × fraction that decays × 100
    liver_uptake_pct = (
        k_liver / k_cl_lipo * (1.0 - math.exp(-k_cl_lipo * t_end_h)) * 100.0
    )

    notes = (
        f"k_cl_lipo={k_cl_lipo:.4f} h^-1 (size_factor={size_factor:.2f}, "
        f"peg_factor={peg_factor:.2f}); t½_lipo={t_half_lipo:.1f} h; "
        f"release t½={release_t_half:.1f} h"
    )

    return LiposomePKResult(
        formulation_name=formulation_name,
        dose_mg_lipid=dose_mg_lipid,
        drug_loading_pct=drug_loading_pct,
        size_nm=size_nm,
        peg_fraction=peg_fraction,
        times_h=times,
        c_liposome_mg_L=c_lipo_list,
        c_free_drug_mg_L=c_free_list,
        c_encapsulated_mg_L=c_enc_list,
        cmax_free_drug=cmax_free,
        cmax_liposome=cmax_lipo,
        auc_free_drug=auc_free,
        auc_liposome=auc_lipo,
        t_half_liposome_h=t_half_lipo,
        release_t_half_h=release_t_half,
        liver_uptake_pct=liver_uptake_pct,
        notes=notes,
    )


def compare_liposome_sizes(
    formulation_name: str,
    dose_mg_lipid: float,
    sizes_nm: list[float],
    **kwargs,
) -> list[LiposomePKResult]:
    """Compare liposome formulations across different sizes.

    Parameters
    ----------
    formulation_name:
        Base label; each result appended with size.
    dose_mg_lipid:
        Total lipid dose (mg).
    sizes_nm:
        List of diameters to evaluate.
    **kwargs:
        Passed to :func:`simulate_liposome_pk`.

    Returns
    -------
    list[LiposomePKResult]
        Sorted by ``auc_free_drug`` descending.
    """
    results = [
        simulate_liposome_pk(
            formulation_name=f"{formulation_name}_{int(sz)}nm",
            dose_mg_lipid=dose_mg_lipid,
            size_nm=sz,
            **kwargs,
        )
        for sz in sizes_nm
    ]
    return sorted(results, key=lambda r: r.auc_free_drug, reverse=True)
