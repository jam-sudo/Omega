"""Nanoparticle/nanomedicine PBPK model for IV-administered nanoparticles.

Models size-dependent clearance, protein corona formation, MPS uptake,
and tumor EPR (Enhanced Permeability and Retention) effect.

Compartments
------------
1. Plasma -- nanoparticle circulates, subject to MPS clearance
2. Liver -- MPS: Kupffer cells uptake, size- and PEG-dependent
3. Spleen -- MPS uptake
4. Lung -- trapping of large particles >200 nm
5. Other tissues -- non-specific distribution
6. Tumor -- EPR effect (optional, if tumor_weight_g > 0)

References
----------
- Wilhelm S et al., Nat Rev Mater. 2016;1:16014 (NP delivery analysis)
- Alexis F et al., Mol Pharm. 2008;5(4):505-15 (NP parameters)
- Li S-D & Huang L, ACS Nano. 2017 (PEGylation and stealth)
- Anchordoquy TJ et al., ACS Nano. 2017;11:12-18 (NP challenges)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from omega_pbpk._compat import np_trapz


@dataclass(frozen=True)
class NanoPBPKResult:
    """Results from nanoparticle PBPK simulation.

    Attributes
    ----------
    nanoparticle_name : Name of the nanoparticle formulation.
    size_nm : Nanoparticle diameter (nm).
    peg_density : PEGylation density (chains/nm^2).
    zeta_potential_mV : Surface charge (mV).
    dose_mg : IV dose (mg).
    times_h : Simulation time points (h).
    c_plasma_mg_L : NP concentration in plasma (mg/L).
    a_liver_pct : Percent dose in liver over time.
    a_spleen_pct : Percent dose in spleen over time.
    a_tumor_pct : Percent dose in tumor over time (EPR).
    c_drug_plasma_mg_L : Released drug plasma concentration (mg/L).
    cmax_plasma_mg_L : NP peak plasma concentration (mg/L).
    auc_plasma : NP AUC 0->t (mg*h/L).
    tumor_auc_pct_h : Percent dose*h in tumor (EPR efficacy metric).
    liver_pct_at_24h : Percent dose in liver at t=24 h.
    tumor_pct_at_24h : Percent dose in tumor at t=24 h (0 if no tumor).
    t_half_h : NP plasma half-life (h).
    epr_efficiency : tumor_auc / (dose * t_end) * 100 (%).
    corona_factor : Protein corona formation factor.
    notes : Summary text.
    """

    nanoparticle_name: str
    size_nm: float
    peg_density: float
    zeta_potential_mV: float
    dose_mg: float
    times_h: list[float]
    c_plasma_mg_L: list[float]
    a_liver_pct: list[float]
    a_spleen_pct: list[float]
    a_tumor_pct: list[float]
    c_drug_plasma_mg_L: list[float]
    cmax_plasma_mg_L: float
    auc_plasma: float
    tumor_auc_pct_h: float
    liver_pct_at_24h: float
    tumor_pct_at_24h: float
    t_half_h: float
    epr_efficiency: float
    corona_factor: float
    notes: str


# ---------------------------------------------------------------------------
# Compartment volumes (L)
# ---------------------------------------------------------------------------
_V_PLASMA = 3.0
_V_LIVER = 1.7
_V_SPLEEN = 0.15
_V_LUNG = 0.5
_V_OTHER = 40.0


def _compute_corona_factor(peg_density: float) -> float:
    """Protein corona factor: PEGylation reduces corona formation."""
    return 1.0 + 0.5 * max(0.0, 1.0 - peg_density * 5.0)


def _compute_mps_rates(
    size_nm: float, peg_density: float, corona_factor: float
) -> tuple[float, float]:
    """Return (k_liver, k_spleen) MPS uptake rates (h^-1)."""
    k_mps_liver = 0.3
    k_mps_spleen = 0.15

    # Size factor: optimal 50-150 nm for long circulation
    if 50.0 < size_nm < 150.0:
        f_size = 1.0
    else:
        f_size = 1.5

    # PEG factor: PEGylation reduces MPS uptake
    f_peg = max(0.2, 1.0 - 2.0 * peg_density)

    k_liver = k_mps_liver * f_size * f_peg * corona_factor
    k_spleen = k_mps_spleen * f_size * f_peg * corona_factor
    return k_liver, k_spleen


def _compute_lung_trapping(size_nm: float) -> float:
    """Lung trapping rate for large particles (>200 nm)."""
    return 0.05 * max(0.0, (size_nm - 200.0) / 200.0)


def _compute_epr_rate(peg_density: float, zeta_potential_mV: float) -> float:
    """EPR permeation rate (h^-1)."""
    k_epr = 0.01
    k_epr *= 1.0 + peg_density * 2.0
    k_epr *= 1.0 + max(0.0, -zeta_potential_mV / 100.0)
    return k_epr


def simulate_nano_pk(
    nanoparticle_name: str,
    dose_mg: float,
    size_nm: float,
    zeta_potential_mV: float = 0.0,
    peg_density_chains_per_nm2: float = 0.1,
    drug_loading_pct: float = 10.0,
    drug_release_rate_per_h: float = 0.1,
    cl_drug_L_per_h: float = 1.0,
    vd_drug_L: float = 20.0,
    tumor_weight_g: float = 1.0,
    t_end_h: float = 72.0,
    dt_h: float = 0.1,
) -> NanoPBPKResult:
    """Simulate nanoparticle PBPK after IV administration.

    Parameters
    ----------
    nanoparticle_name : Formulation name.
    dose_mg : IV dose of nanoparticle (mg, including drug payload).
    size_nm : Nanoparticle diameter (nm).
    zeta_potential_mV : Surface charge (mV).
    peg_density_chains_per_nm2 : PEGylation density (chains/nm^2).
    drug_loading_pct : Drug loading (%, 0-100 exclusive).
    drug_release_rate_per_h : Drug release rate from NP (h^-1).
    cl_drug_L_per_h : Clearance of released drug (L/h).
    vd_drug_L : Volume of distribution of released drug (L).
    tumor_weight_g : Tumor weight (g). Set 0 to disable tumor compartment.
    t_end_h : Simulation duration (h).
    dt_h : Time step (h).

    Returns
    -------
    NanoPBPKResult with full time-course and PK metrics.

    Raises
    ------
    ValueError : If any parameter is invalid.
    """
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if size_nm <= 0:
        raise ValueError("size_nm must be > 0")
    if drug_loading_pct <= 0 or drug_loading_pct >= 100:
        raise ValueError("drug_loading_pct must be in (0, 100)")

    peg_density = peg_density_chains_per_nm2
    tumor_present = tumor_weight_g > 0
    v_tumor = tumor_weight_g * 0.001 if tumor_present else 0.0

    # Biophysical parameters
    corona = _compute_corona_factor(peg_density)
    k_liver, k_spleen = _compute_mps_rates(size_nm, peg_density, corona)
    k_lung = _compute_lung_trapping(size_nm)
    k_epr = _compute_epr_rate(peg_density, zeta_potential_mV) if tumor_present else 0.0

    # Renal clearance: only for very small NPs (<8 nm)
    k_renal = 0.5 if size_nm < 8.0 else 0.0
    k_fecal = 0.02

    # ODE integration (forward Euler)
    n_steps = max(int(t_end_h / dt_h), 1)
    t = np.linspace(0.0, t_end_h, n_steps + 1)

    a_plasma = np.zeros(n_steps + 1)
    a_liver = np.zeros(n_steps + 1)
    a_spleen = np.zeros(n_steps + 1)
    a_lung = np.zeros(n_steps + 1)
    a_other = np.zeros(n_steps + 1)
    a_tumor = np.zeros(n_steps + 1)
    c_drug = np.zeros(n_steps + 1)

    # Initial condition: entire dose in plasma
    a_plasma[0] = dose_mg

    for i in range(n_steps):
        # EPR tumor uptake
        epr_flux = k_epr * a_plasma[i] * (v_tumor / _V_PLASMA) if tumor_present else 0.0
        k_tumor_drain = 0.001

        # Plasma: losses to MPS organs + renal/fecal + EPR
        da_plasma = (
            -(k_liver + k_spleen + k_lung + k_fecal + k_renal) * a_plasma[i]
            - epr_flux
            + k_tumor_drain * a_tumor[i]
        ) * dt_h

        da_liver = (k_liver * a_plasma[i]) * dt_h
        da_spleen = (k_spleen * a_plasma[i]) * dt_h
        da_lung = (k_lung * a_plasma[i] - 0.001 * a_lung[i]) * dt_h
        da_other = (0.05 * a_plasma[i] - 0.02 * a_other[i]) * dt_h
        da_tumor = (epr_flux - 0.001 * a_tumor[i]) * dt_h if tumor_present else 0.0

        # Released drug PK (1-compartment)
        dc_drug = (
            drug_release_rate_per_h * a_plasma[i] / _V_PLASMA
            - (cl_drug_L_per_h / vd_drug_L) * c_drug[i]
        ) * dt_h

        a_plasma[i + 1] = max(0.0, a_plasma[i] + da_plasma)
        a_liver[i + 1] = max(0.0, a_liver[i] + da_liver)
        a_spleen[i + 1] = max(0.0, a_spleen[i] + da_spleen)
        a_lung[i + 1] = max(0.0, a_lung[i] + da_lung)
        a_other[i + 1] = max(0.0, a_other[i] + da_other)
        a_tumor[i + 1] = max(0.0, a_tumor[i] + da_tumor)
        c_drug[i + 1] = max(0.0, c_drug[i] + dc_drug)

    # Concentrations and percentages
    c_plasma = a_plasma / _V_PLASMA
    liver_pct = a_liver / dose_mg * 100.0
    spleen_pct = a_spleen / dose_mg * 100.0
    tumor_pct = a_tumor / dose_mg * 100.0

    # Summary metrics
    cmax = float(np.max(c_plasma))
    auc = float(np_trapz(c_plasma, t))
    tumor_auc = float(np_trapz(tumor_pct, t))

    # Find index closest to 24 h
    idx_24 = int(np.argmin(np.abs(t - 24.0)))
    liver_24 = float(liver_pct[idx_24])
    tumor_24 = float(tumor_pct[idx_24]) if tumor_present else 0.0

    # Plasma half-life: time for c_plasma to drop below 50% of initial
    c0 = c_plasma[0]
    half_indices = np.where(c_plasma <= c0 * 0.5)[0]
    if len(half_indices) > 0:
        t_half = float(t[half_indices[0]])
    else:
        t_half = float(t_end_h)

    epr_eff = tumor_auc / (dose_mg * t_end_h) * 100.0 if t_end_h > 0 else 0.0

    notes = (
        f"Nano PBPK: {nanoparticle_name}, size={size_nm:.0f} nm, "
        f"dose={dose_mg:.1f} mg. "
        f"Corona factor={corona:.2f}, t1/2={t_half:.1f} h. "
        f"Liver={liver_24:.1f}% at 24h, Tumor={tumor_24:.2f}% at 24h."
    )

    return NanoPBPKResult(
        nanoparticle_name=nanoparticle_name,
        size_nm=size_nm,
        peg_density=peg_density,
        zeta_potential_mV=zeta_potential_mV,
        dose_mg=dose_mg,
        times_h=t.tolist(),
        c_plasma_mg_L=c_plasma.tolist(),
        a_liver_pct=liver_pct.tolist(),
        a_spleen_pct=spleen_pct.tolist(),
        a_tumor_pct=tumor_pct.tolist(),
        c_drug_plasma_mg_L=c_drug.tolist(),
        cmax_plasma_mg_L=cmax,
        auc_plasma=auc,
        tumor_auc_pct_h=tumor_auc,
        liver_pct_at_24h=liver_24,
        tumor_pct_at_24h=tumor_24,
        t_half_h=t_half,
        epr_efficiency=epr_eff,
        corona_factor=corona,
        notes=notes,
    )


def compare_formulations(
    nanoparticle_name: str,
    dose_mg: float,
    formulations: list[dict],
    **kwargs,
) -> list[NanoPBPKResult]:
    """Compare multiple nanoparticle formulations.

    Parameters
    ----------
    nanoparticle_name : Base name for the formulations.
    dose_mg : IV dose (mg).
    formulations : List of dicts with keys: size_nm, peg_density, zeta_potential_mV.
    **kwargs : Additional keyword arguments passed to simulate_nano_pk.

    Returns
    -------
    List of NanoPBPKResult, one per formulation.
    """
    results = []
    for f in formulations:
        r = simulate_nano_pk(
            nanoparticle_name=nanoparticle_name,
            dose_mg=dose_mg,
            size_nm=f["size_nm"],
            zeta_potential_mV=f.get("zeta_potential_mV", 0.0),
            peg_density_chains_per_nm2=f.get("peg_density", 0.1),
            **kwargs,
        )
        results.append(r)
    return results


__all__ = [
    "NanoPBPKResult",
    "simulate_nano_pk",
    "compare_formulations",
]
