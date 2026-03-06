"""CNS target engagement model linking brain PK to receptor occupancy.

Models blood-brain barrier transport and receptor binding dynamics
for CNS-targeted drugs, predicting receptor occupancy over time.

Model structure
---------------
Two compartments: plasma + brain (ECF).
BBB transport with asymmetric influx/efflux (P-gp effect).
Target-mediated drug disposition: kon/koff binding to receptor.

ODEs (Forward Euler):
  dC_plasma/dt = -ke*Cp - k_bbb_in*Cp + k_bbb_out*Cb*(Vb/Vp) [+ oral input]
  dC_brain/dt  = k_bbb_in*Cp*(Vp/Vb) - k_bbb_out*Cb - kon*Cb*R + koff*RC
  dRC/dt       = kon*Cb*R - (koff + kint)*RC
  R            = Bmax - RC  (conservation)

References
----------
- Mager DE & Jusko WJ, J Pharmacokinet Pharmacodyn. 2001;28:507–32
- Hammarlund-Udenaes M, Fluids Barriers CNS. 2010;7:5
- Kielbasa W et al., J Pharmacokinet Pharmacodyn. 2009;36:245–63
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from omega_pbpk._compat import np_trapz


@dataclass(frozen=True)
class CNSTargetEngagementResult:
    """Results from CNS target engagement simulation.

    Attributes
    ----------
    drug_name : Drug identifier.
    route : Administration route ('iv' or 'oral').
    dose_mg : Administered dose (mg).
    mw_g_mol : Molecular weight (g/mol).
    times_h : Simulation time points (h).
    c_plasma_nM : Plasma concentration over time (nM).
    c_brain_nM : Brain ECF concentration over time (nM).
    rc_nM : Receptor-drug complex concentration over time (nM).
    receptor_occupancy : Receptor occupancy over time (%).
    free_receptor_nM : Free receptor concentration over time (nM).
    peak_occupancy_pct : Maximum receptor occupancy (%).
    tmax_occupancy_h : Time of peak occupancy (h).
    auc_occupancy : Cumulative occupancy (% * h).
    time_above_50pct_h : Duration of >50% occupancy (h).
    kd_nM : Equilibrium dissociation constant (nM).
    brain_plasma_ratio : Max brain / max plasma concentration ratio.
    notes : Summary text.
    """

    drug_name: str
    route: str
    dose_mg: float
    mw_g_mol: float
    times_h: list[float]
    c_plasma_nM: list[float]
    c_brain_nM: list[float]
    rc_nM: list[float]
    receptor_occupancy: list[float]
    free_receptor_nM: list[float]
    peak_occupancy_pct: float
    tmax_occupancy_h: float
    auc_occupancy: float
    time_above_50pct_h: float
    kd_nM: float
    brain_plasma_ratio: float
    notes: str


def simulate_cns_target_engagement(
    drug_name: str,
    dose_mg: float,
    mw_g_mol: float = 300.0,
    route: str = "iv",
    ka_per_h: float = 1.0,
    f_oral: float = 1.0,
    cl_plasma_L_per_h: float = 1.0,
    vd_plasma_L: float = 5.0,
    k_bbb_in_per_h: float = 0.1,
    k_bbb_out_per_h: float = 0.5,
    vbrain_L: float = 1.4e-3,
    bmax_nM: float = 100.0,
    kon_per_nM_per_h: float = 0.01,
    koff_per_h: float = 1.0,
    kint_per_h: float = 0.1,
    t_end_h: float = 24.0,
    dt_h: float = 0.02,
) -> CNSTargetEngagementResult:
    """Simulate CNS target engagement with BBB transport and receptor binding.

    Parameters
    ----------
    drug_name : Drug identifier.
    dose_mg : Dose in mg.
    mw_g_mol : Molecular weight (g/mol) for mg-to-nM conversion.
    route : 'iv' (bolus) or 'oral'.
    ka_per_h : Oral absorption rate constant (h-1).
    f_oral : Oral bioavailability fraction.
    cl_plasma_L_per_h : Plasma clearance (L/h).
    vd_plasma_L : Plasma volume of distribution (L).
    k_bbb_in_per_h : BBB influx rate (h-1).
    k_bbb_out_per_h : BBB efflux rate (h-1, includes P-gp).
    vbrain_L : Brain ECF volume (L).
    bmax_nM : Total receptor density (nM).
    kon_per_nM_per_h : Association rate constant (nM-1 h-1).
    koff_per_h : Dissociation rate constant (h-1).
    kint_per_h : Receptor internalization rate (h-1).
    t_end_h : Simulation duration (h).
    dt_h : Time step (h).

    Returns
    -------
    CNSTargetEngagementResult

    Raises
    ------
    ValueError : If any parameter is invalid.
    """
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if cl_plasma_L_per_h <= 0:
        raise ValueError("cl_plasma_L_per_h must be > 0")
    if vd_plasma_L <= 0:
        raise ValueError("vd_plasma_L must be > 0")
    if bmax_nM <= 0:
        raise ValueError("bmax_nM must be > 0")
    if kon_per_nM_per_h <= 0:
        raise ValueError("kon_per_nM_per_h must be > 0")
    if koff_per_h <= 0:
        raise ValueError("koff_per_h must be > 0")

    ke = cl_plasma_L_per_h / vd_plasma_L
    kd = koff_per_h / kon_per_nM_per_h

    # Dose in nM: (mg * 1e6 ng/mg) / (g/mol * L) = ng / (ng/nmol * L) = nmol/L = nM
    dose_nM_in_vd = (dose_mg * 1e6) / (mw_g_mol * vd_plasma_L)

    n_steps = max(int(t_end_h / dt_h), 1)
    t = np.linspace(0.0, t_end_h, n_steps + 1)

    c_plasma = np.zeros(n_steps + 1)
    c_brain = np.zeros(n_steps + 1)
    rc = np.zeros(n_steps + 1)

    # For oral route, use a gut compartment
    a_gut = 0.0

    if route.lower() == "iv":
        c_plasma[0] = dose_nM_in_vd
    else:
        a_gut = dose_nM_in_vd * f_oral  # effective amount in gut (nM equiv)

    vp = vd_plasma_L
    vb = vbrain_L

    for i in range(n_steps):
        r_free = max(0.0, bmax_nM - rc[i])

        # Oral absorption input
        oral_input = 0.0
        if route.lower() == "oral" and a_gut > 0:
            absorbed = ka_per_h * a_gut * dt_h
            a_gut -= absorbed
            a_gut = max(0.0, a_gut)
            oral_input = absorbed / dt_h  # rate in nM/h

        dcp = (
            -ke * c_plasma[i]
            - k_bbb_in_per_h * c_plasma[i]
            + k_bbb_out_per_h * c_brain[i] * (vb / vp)
            + oral_input
        )
        dcb = (
            k_bbb_in_per_h * c_plasma[i] * (vp / vb)
            - k_bbb_out_per_h * c_brain[i]
            - kon_per_nM_per_h * c_brain[i] * r_free
            + koff_per_h * rc[i]
        )
        drc = kon_per_nM_per_h * c_brain[i] * r_free - (koff_per_h + kint_per_h) * rc[i]

        c_plasma[i + 1] = max(0.0, c_plasma[i] + dcp * dt_h)
        c_brain[i + 1] = max(0.0, c_brain[i] + dcb * dt_h)
        rc[i + 1] = max(0.0, min(bmax_nM, rc[i] + drc * dt_h))

    # Derived arrays
    occupancy = rc / bmax_nM * 100.0
    free_rec = bmax_nM - rc

    # Metrics
    peak_occ = float(np.max(occupancy))
    tmax_occ = float(t[int(np.argmax(occupancy))])
    auc_occ = float(np_trapz(occupancy, t))

    above_50 = occupancy > 50.0
    time_above_50 = float(np.sum(above_50) * dt_h) if np.any(above_50) else 0.0

    max_brain = float(np.max(c_brain))
    max_plasma = float(np.max(c_plasma))
    bp_ratio = max_brain / max_plasma if max_plasma > 0 else 0.0

    notes = (
        f"CNS target engagement: {drug_name}, {dose_mg}mg {route}. "
        f"Peak occupancy={peak_occ:.1f}% at {tmax_occ:.2f}h. "
        f"Kd={kd:.1f}nM, brain/plasma ratio={bp_ratio:.4f}."
    )

    return CNSTargetEngagementResult(
        drug_name=drug_name,
        route=route,
        dose_mg=dose_mg,
        mw_g_mol=mw_g_mol,
        times_h=t.tolist(),
        c_plasma_nM=c_plasma.tolist(),
        c_brain_nM=c_brain.tolist(),
        rc_nM=rc.tolist(),
        receptor_occupancy=occupancy.tolist(),
        free_receptor_nM=free_rec.tolist(),
        peak_occupancy_pct=peak_occ,
        tmax_occupancy_h=tmax_occ,
        auc_occupancy=auc_occ,
        time_above_50pct_h=time_above_50,
        kd_nM=kd,
        brain_plasma_ratio=bp_ratio,
        notes=notes,
    )


def occupancy_dose_response(
    drug_name: str,
    doses_mg: list[float],
    **kwargs,
) -> list[CNSTargetEngagementResult]:
    """Run simulations across multiple doses for dose-occupancy analysis.

    Parameters
    ----------
    drug_name : Drug identifier.
    doses_mg : List of doses to simulate.
    **kwargs : Passed to simulate_cns_target_engagement.

    Returns
    -------
    List of CNSTargetEngagementResult, one per dose.
    """
    return [simulate_cns_target_engagement(drug_name, dose, **kwargs) for dose in doses_mg]


__all__ = [
    "CNSTargetEngagementResult",
    "simulate_cns_target_engagement",
    "occupancy_dose_response",
]
