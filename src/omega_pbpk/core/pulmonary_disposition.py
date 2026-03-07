"""Pulmonary Drug Disposition — Phase 324.

Models drug disposition in the lung for inhaled and systemically delivered drugs,
focusing on pulmonary targeting and local tissue distribution.

Model structure
---------------
Two lung regions:
  1. Bronchial — subject to mucociliary clearance + epithelial absorption
  2. Alveolar — deep lung, slower mucociliary; efficient epithelial absorption

Systemic compartment:
  dC_sys/dt = k_abs_total * (A_bronch + A_alv) / Vd_sys - (CL_sys / Vd_sys) * C_sys

All ODEs integrated by Forward Euler.

References
----------
- Patton JS & Byron PR. Nat Rev Drug Discov. 2007;6(1):67-74.
- Borghardt JM et al. CPT Pharmacometrics Syst Pharmacol. 2018;7:281-294.
- Newman SP. Expert Opin Drug Deliv. 2017;14(3):395-406.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from omega_pbpk._compat import np_trapz

# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PulmonaryPKResult:
    """Results from pulmonary disposition simulation.

    Attributes
    ----------
    drug_name : Drug name.
    dose_mg : Administered dose (mg).
    particle_size_um : Aerosol particle size (µm).
    lung_deposition_fraction : Fraction of dose depositing in the lung.
    times_h : Simulation time points (h).
    c_systemic : Systemic plasma concentration (mg/L) at each time point.
    cmax_systemic : Peak systemic concentration (mg/L).
    tmax_systemic_h : Time of peak systemic concentration (h).
    auc_systemic : AUC 0→t of systemic concentration (mg·h/L).
    auc_lung_exposure : Simplified lung exposure integral (mg·h/L).
    pulmonary_targeting_index : AUC_lung / AUC_systemic (>1 = local advantage).
    notes : Summary description.
    """

    drug_name: str
    dose_mg: float
    particle_size_um: float
    lung_deposition_fraction: float
    times_h: list[float]
    c_systemic: list[float]
    cmax_systemic: float
    tmax_systemic_h: float
    auc_systemic: float
    auc_lung_exposure: float
    pulmonary_targeting_index: float
    notes: str


# ---------------------------------------------------------------------------
# Helper: deposition fractions by particle size
# ---------------------------------------------------------------------------


def _deposition_fractions(particle_size_um: float) -> tuple[float, float, float]:
    """Return (total_lung_fraction, bronchial_fraction, alveolar_fraction).

    Based on particle size (µm):
      < 1 µm : total 0.4, bronch 0.2, alv 0.2 (mostly exhaled)
      1–5 µm : total 0.6, bronch 0.4, alv 0.6 (optimal)  [normalised below]
      > 5 µm : total 0.3, bronch 0.2, alv 0.2 (oropharyngeal)

    The spec states bronchial_fraction=0.4 and alveolar_fraction=0.6 for 1<size<5.
    These are applied to the deposited fraction, not the total dose.
    """
    if particle_size_um < 1.0:
        total = 0.4
        bronch_frac = 0.2  # of deposited
        alv_frac = 0.2  # of deposited
    elif particle_size_um <= 5.0:
        total = 0.6
        bronch_frac = 0.4  # of deposited
        alv_frac = 0.6  # of deposited
    else:
        total = 0.3
        bronch_frac = 0.2  # of deposited
        alv_frac = 0.2  # of deposited
    return total, bronch_frac, alv_frac


# ---------------------------------------------------------------------------
# Core simulation
# ---------------------------------------------------------------------------


def simulate_inhaled_pk(
    drug_name: str,
    dose_mg: float,
    particle_size_um: float = 2.0,
    cl_mucociliary_per_h: float = 10.0,
    cl_alveolar_per_h: float = 0.5,
    papp_epithelium_cm_s: float = 1e-5,
    vd_lung_L: float = 1.0,
    cl_systemic_L_per_h: float = 10.0,
    vd_systemic_L: float = 50.0,
    t_end_h: float = 12.0,
    dt_h: float = 0.05,
) -> PulmonaryPKResult:
    """Simulate inhaled drug pharmacokinetics in the lung and systemic circulation.

    Lung deposition by particle size:
      < 1 µm  → 40% of dose deposited (mostly exhaled/alveolar)
      1–5 µm  → 60% of dose deposited (bronchial/alveolar optimal)
      > 5 µm  → 30% of dose deposited (oropharyngeal → swallowed)

    ODE system (Forward Euler):
      dA_bronch/dt = -(k_mc + k_abs_b) * A_bronch
      dA_alv/dt   = -k_abs_a * A_alv
      dC_sys/dt   = k_abs_total * (A_b + A_a) / Vd_sys - (CL_sys / Vd_sys) * C_sys

    where k_abs_b and k_abs_a are derived from Papp and a surface area proxy.

    Args:
        drug_name: Drug identifier.
        dose_mg: Nominal inhaled dose (mg, > 0).
        particle_size_um: Mass median aerodynamic diameter (µm, > 0).
        cl_mucociliary_per_h: Bronchial mucociliary clearance rate (h⁻¹, > 0).
        cl_alveolar_per_h: Alveolar clearance rate constant (h⁻¹, > 0).
        papp_epithelium_cm_s: Apparent permeability across lung epithelium (cm/s, > 0).
        vd_lung_L: Lung tissue volume used for lung AUC (L, > 0).
        cl_systemic_L_per_h: Systemic clearance (L/h, > 0).
        vd_systemic_L: Systemic volume of distribution (L, > 0).
        t_end_h: Simulation end time (h).
        dt_h: Forward Euler time step (h).

    Returns:
        PulmonaryPKResult with full time-course and PK metrics.

    Raises:
        ValueError: If any required parameter is invalid.
    """
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if particle_size_um <= 0:
        raise ValueError("particle_size_um must be > 0")
    if cl_mucociliary_per_h <= 0:
        raise ValueError("cl_mucociliary_per_h must be > 0")
    if cl_alveolar_per_h <= 0:
        raise ValueError("cl_alveolar_per_h must be > 0")
    if papp_epithelium_cm_s <= 0:
        raise ValueError("papp_epithelium_cm_s must be > 0")
    if vd_lung_L <= 0:
        raise ValueError("vd_lung_L must be > 0")
    if cl_systemic_L_per_h <= 0:
        raise ValueError("cl_systemic_L_per_h must be > 0")
    if vd_systemic_L <= 0:
        raise ValueError("vd_systemic_L must be > 0")

    # Deposition
    total_frac, bronch_frac, alv_frac = _deposition_fractions(particle_size_um)
    deposited_mg = dose_mg * total_frac

    # Initial masses in each region
    a_bronch0 = deposited_mg * bronch_frac
    a_alv0 = deposited_mg * alv_frac

    # Absorption rate constants from Papp (cm/s → h⁻¹).
    # Surface area proxy: bronchial ~200 cm², alveolar ~700 cm².
    # k_abs (h⁻¹) = Papp * SA_cm2 / V_mg_proxy; simplified to Papp * 3600 * scaling.
    # We use Papp * 3600 [cm/h] * SA_cm2 / Vd_lung_cm3 with Vd_lung_cm3 = vd_lung_L*1000.
    sa_bronch_cm2 = 200.0
    sa_alv_cm2 = 700.0
    vd_lung_cm3 = vd_lung_L * 1000.0  # cm³

    papp_per_h = papp_epithelium_cm_s * 3600.0  # cm/h
    k_abs_b = papp_per_h * sa_bronch_cm2 / vd_lung_cm3  # h⁻¹
    k_abs_a = papp_per_h * sa_alv_cm2 / vd_lung_cm3  # h⁻¹

    k_mc = cl_mucociliary_per_h  # mucociliary clearance from bronchial (h⁻¹)
    k_sys = cl_systemic_L_per_h / vd_systemic_L  # elimination rate (h⁻¹)

    n_steps = max(int(t_end_h / dt_h), 1)
    times = np.linspace(0.0, t_end_h, n_steps + 1)

    a_bronch = np.zeros(n_steps + 1)
    a_alv = np.zeros(n_steps + 1)
    c_sys = np.zeros(n_steps + 1)
    c_lung = np.zeros(n_steps + 1)  # (A_b + A_a) / Vd_lung_L for AUC

    a_bronch[0] = a_bronch0
    a_alv[0] = a_alv0
    c_lung[0] = (a_bronch0 + a_alv0) / vd_lung_L

    for i in range(n_steps):
        ab = a_bronch[i]
        aa = a_alv[i]
        cs = c_sys[i]

        # Mass-transfer rates (mg/h)
        rate_abs_b = k_abs_b * ab  # bronchial absorption to systemic
        rate_mc = k_mc * ab  # mucociliary removal (cleared, not absorbed)
        rate_abs_a = k_abs_a * aa  # alveolar absorption

        # Forward Euler updates
        a_bronch[i + 1] = max(0.0, ab + dt_h * (-(rate_abs_b + rate_mc)))
        a_alv[i + 1] = max(0.0, aa + dt_h * (-rate_abs_a))

        # Systemic absorption input = absorbed from both regions (mg/h)
        input_sys = rate_abs_b + rate_abs_a  # mg/h
        dcs = (input_sys / vd_systemic_L) - k_sys * cs
        c_sys[i + 1] = max(0.0, cs + dt_h * dcs)

        c_lung[i + 1] = (a_bronch[i + 1] + a_alv[i + 1]) / vd_lung_L

    # PK metrics
    cmax_sys = float(np.max(c_sys))
    tmax_sys = float(times[int(np.argmax(c_sys))])
    auc_sys = float(np_trapz(c_sys, times))
    auc_lung = float(np_trapz(c_lung, times))

    pti = pulmonary_targeting_index(auc_lung, auc_sys) if auc_sys > 0 else float("inf")

    notes = (
        f"Pulmonary PK: {drug_name}, {dose_mg} mg, particle={particle_size_um} µm. "
        f"Lung deposition={total_frac:.0%}. "
        f"Cmax_sys={cmax_sys:.4f} mg/L at {tmax_sys:.2f} h. "
        f"AUC_sys={auc_sys:.3f} mg·h/L. PTI={pti:.2f}."
    )

    return PulmonaryPKResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        particle_size_um=particle_size_um,
        lung_deposition_fraction=total_frac,
        times_h=times.tolist(),
        c_systemic=c_sys.tolist(),
        cmax_systemic=cmax_sys,
        tmax_systemic_h=tmax_sys,
        auc_systemic=auc_sys,
        auc_lung_exposure=auc_lung,
        pulmonary_targeting_index=pti,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Targeting index
# ---------------------------------------------------------------------------


def pulmonary_targeting_index(auc_lung: float, auc_systemic: float) -> float:
    """Compute pulmonary targeting index (PTI).

    PTI = AUC_lung / AUC_systemic.
    PTI > 1 indicates local lung targeting advantage over systemic exposure.

    Args:
        auc_lung: Lung AUC (any consistent unit, > 0).
        auc_systemic: Systemic AUC (same unit, > 0).

    Returns:
        PTI as a float.

    Raises:
        ValueError: If either AUC value is not positive.
    """
    if auc_lung <= 0:
        raise ValueError("auc_lung must be > 0")
    if auc_systemic <= 0:
        raise ValueError("auc_systemic must be > 0")
    return auc_lung / auc_systemic


# ---------------------------------------------------------------------------
# Particle size comparison
# ---------------------------------------------------------------------------


def compare_particle_sizes(
    drug_name: str,
    dose_mg: float,
    cl_systemic: float,
    vd_systemic: float,
) -> dict[str, PulmonaryPKResult]:
    """Compare pulmonary PK for four representative particle sizes.

    Particle sizes compared: 0.5, 2.0, 5.0, 10.0 µm.

    Args:
        drug_name: Drug identifier.
        dose_mg: Nominal inhaled dose (mg, > 0).
        cl_systemic: Systemic clearance (L/h, > 0).
        vd_systemic: Systemic volume of distribution (L, > 0).

    Returns:
        Dict mapping particle size string (e.g. "0.5um") → PulmonaryPKResult.

    Raises:
        ValueError: On invalid inputs (propagated from simulate_inhaled_pk).
    """
    sizes = [0.5, 2.0, 5.0, 10.0]
    results: dict[str, PulmonaryPKResult] = {}
    for size in sizes:
        result = simulate_inhaled_pk(
            drug_name=drug_name,
            dose_mg=dose_mg,
            particle_size_um=size,
            cl_systemic_L_per_h=cl_systemic,
            vd_systemic_L=vd_systemic,
        )
        results[f"{size}um"] = result
    return results


__all__ = [
    "PulmonaryPKResult",
    "simulate_inhaled_pk",
    "pulmonary_targeting_index",
    "compare_particle_sizes",
]
