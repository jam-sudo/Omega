"""Lung PBPK model for inhaled drug pharmacokinetics.

Models pulmonary drug delivery and absorption for inhaled formulations
(nebulizers, dry powder inhalers, metered-dose inhalers).

Model structure
---------------
Three lung regions based on ICRP 66 respiratory tract model:

  1. Oropharyngeal (OP) — upper airway, fast mucociliary clearance
  2. Tracheobronchial (TB) — conducting airways, slower clearance
  3. Pulmonary (PUL) — alveoli, absorption to systemic circulation

Drug deposited in each region is absorbed into blood (with rate ka_region)
or cleared by mucociliary transport (kmc, swallowed → GI).

Systemic compartment:
  dCp/dt = sum(ka * A_region / Vd) - CL/Vd * Cp

References
----------
- Borghardt JM et al., CPT Pharmacometrics Syst Pharmacol. 2018;7:281–94
- Weers JG et al., J Aerosol Med. 2010;23 Suppl 2:S7–30 (deposition fractions)
- Patton JS & Byron PR, Nat Rev Drug Discov. 2007;6(1):67–74 (pulmonary PK)
- ICRP Publication 66, Ann ICRP. 1994;24:1–482 (respiratory tract model)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from omega_pbpk._compat import np_trapz


@dataclass(frozen=True)
class InhaledDeposition:
    """Regional deposition fractions for an inhaled dose.

    Fractions sum to ≤ 1.0 (remainder exhaled).

    Attributes
    ----------
    oropharyngeal : Fraction deposited in OP region (0–1).
    tracheobronchial : Fraction deposited in TB region (0–1).
    pulmonary : Fraction deposited in alveolar region (0–1).
    exhaled : Fraction exhaled and not deposited.
    """

    oropharyngeal: float
    tracheobronchial: float
    pulmonary: float
    exhaled: float


# ---------------------------------------------------------------------------
# Default deposition fractions by device type (Weers 2010, Borghardt 2018)
# ---------------------------------------------------------------------------

_DEVICE_DEPOSITION: dict[str, InhaledDeposition] = {
    "mdi": InhaledDeposition(
        oropharyngeal=0.70,
        tracheobronchial=0.10,
        pulmonary=0.10,
        exhaled=0.10,
    ),
    "dpi": InhaledDeposition(
        oropharyngeal=0.25,
        tracheobronchial=0.20,
        pulmonary=0.45,
        exhaled=0.10,
    ),
    "nebulizer": InhaledDeposition(
        oropharyngeal=0.10,
        tracheobronchial=0.20,
        pulmonary=0.60,
        exhaled=0.10,
    ),
    "soft_mist": InhaledDeposition(
        oropharyngeal=0.15,
        tracheobronchial=0.25,
        pulmonary=0.55,
        exhaled=0.05,
    ),
}


@dataclass(frozen=True)
class LungPBPKResult:
    """Results from inhaled lung PBPK simulation.

    Attributes
    ----------
    drug_name : Name of the drug.
    device : Delivery device type.
    dose_mg : Nominal inhaled dose (mg).
    times_h : Simulation time points (h).
    cp_mg_L : Systemic plasma concentration (mg/L).
    a_op_mg : Mass in oropharyngeal region (mg).
    a_tb_mg : Mass in tracheobronchial region (mg).
    a_pul_mg : Mass in pulmonary region (mg).
    a_gi_mg : Mass swallowed (GI absorption from OP mucociliary) (mg).
    cmax_mg_L : Peak systemic concentration (mg/L).
    tmax_h : Time of peak systemic concentration (h).
    auc_0_t : AUC 0→t (mg·h/L).
    lung_bioavailability : Fraction of nominal dose absorbed from lung (0–1).
    total_bioavailability : Total F (lung + GI swallowed) (0–1).
    deposition : InhaledDeposition used.
    notes : Summary text.
    """

    drug_name: str
    device: str
    dose_mg: float
    times_h: list[float]
    cp_mg_L: list[float]
    a_op_mg: list[float]
    a_tb_mg: list[float]
    a_pul_mg: list[float]
    a_gi_mg: list[float]
    cmax_mg_L: float
    tmax_h: float
    auc_0_t: float
    lung_bioavailability: float
    total_bioavailability: float
    deposition: InhaledDeposition
    notes: str


def simulate_lung_pk(
    drug_name: str,
    dose_mg: float,
    cl_L_per_h: float,
    vd_L: float,
    device: str = "dpi",
    ka_pulmonary_per_h: float = 4.0,
    ka_tb_per_h: float = 1.0,
    ka_op_per_h: float = 0.5,
    kmc_op_per_h: float = 8.0,
    kmc_tb_per_h: float = 2.0,
    f_gi: float = 0.5,
    custom_deposition: InhaledDeposition | None = None,
    t_end_h: float = 24.0,
    dt_h: float = 0.02,
) -> LungPBPKResult:
    """Simulate inhaled drug PK with regional lung deposition and absorption.

    Parameters
    ----------
    drug_name : Drug name.
    dose_mg : Nominal inhaled dose (mg).
    cl_L_per_h : Systemic clearance (L/h).
    vd_L : Systemic volume of distribution (L).
    device : Delivery device ('mdi', 'dpi', 'nebulizer', 'soft_mist').
    ka_pulmonary_per_h : Absorption rate from alveoli to blood (h⁻¹).
        High for lipophilic drugs (~4–10/h), low for hydrophilic (~0.5/h).
    ka_tb_per_h : Absorption rate from TB airway to blood (h⁻¹).
    ka_op_per_h : Absorption rate from OP to blood/GI (h⁻¹).
    kmc_op_per_h : Mucociliary clearance from OP (swallowed, h⁻¹).
    kmc_tb_per_h : Mucociliary clearance from TB to OP then GI (h⁻¹).
    f_gi : Oral bioavailability of swallowed drug from GI tract.
    custom_deposition : Override device-based deposition fractions.
    t_end_h : Simulation duration (h).
    dt_h : Time step (h).

    Returns
    -------
    LungPBPKResult with full time-course and PK metrics.

    Raises
    ------
    ValueError : If any parameter is invalid.
    """
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if cl_L_per_h <= 0:
        raise ValueError("cl_L_per_h must be > 0")
    if vd_L <= 0:
        raise ValueError("vd_L must be > 0")

    if custom_deposition is not None:
        dep = custom_deposition
    elif device.lower() in _DEVICE_DEPOSITION:
        dep = _DEVICE_DEPOSITION[device.lower()]
    else:
        raise ValueError(f"Unknown device '{device}'. Choose from: {list(_DEVICE_DEPOSITION)}")

    # Initial regional masses (mg) from deposition fractions
    a_op0 = dose_mg * dep.oropharyngeal
    a_tb0 = dose_mg * dep.tracheobronchial
    a_pul0 = dose_mg * dep.pulmonary

    ke = cl_L_per_h / vd_L

    n_steps = max(int(t_end_h / dt_h), 1)
    t = np.linspace(0.0, t_end_h, n_steps + 1)

    # State variables
    cp = np.zeros(n_steps + 1)  # systemic plasma conc (mg/L)
    a_op = np.zeros(n_steps + 1)  # OP mass (mg)
    a_tb = np.zeros(n_steps + 1)  # TB mass (mg)
    a_pul = np.zeros(n_steps + 1)  # pulmonary mass (mg)
    a_gi = np.zeros(n_steps + 1)  # cumulative GI absorbed (mg)

    # Initial conditions
    a_op[0] = a_op0
    a_tb[0] = a_tb0
    a_pul[0] = a_pul0

    # Track total absorbed from lung and swallowed
    total_lung_absorbed = 0.0
    total_gi_input = 0.0

    for i in range(n_steps):
        # Mucociliary clearance: OP → swallowed (GI), TB → OP
        mcc_op = kmc_op_per_h * a_op[i]  # mg/h to GI
        mcc_tb = kmc_tb_per_h * a_tb[i]  # mg/h to OP

        # Lung absorption to blood
        abs_pul = ka_pulmonary_per_h * a_pul[i]  # mg/h
        abs_tb = ka_tb_per_h * a_tb[i]  # mg/h (via capillary)
        abs_op = ka_op_per_h * a_op[i]  # mg/h (buccal/submucosal)

        # GI bioavailability of swallowed drug
        gi_rate = f_gi * mcc_op  # effectively absorbed from GI (mg/h)

        # ODE steps
        da_op = (mcc_tb - mcc_op - abs_op) * dt_h
        da_tb = (-mcc_tb - abs_tb) * dt_h
        da_pul = -abs_pul * dt_h
        dcp = ((abs_pul + abs_tb + abs_op + gi_rate) / vd_L - ke * cp[i]) * dt_h

        a_op[i + 1] = max(0.0, a_op[i] + da_op)
        a_tb[i + 1] = max(0.0, a_tb[i] + da_tb)
        a_pul[i + 1] = max(0.0, a_pul[i] + da_pul)
        cp[i + 1] = max(0.0, cp[i] + dcp)
        a_gi[i + 1] = a_gi[i] + mcc_op * dt_h

        total_lung_absorbed += (abs_pul + abs_tb + abs_op) * dt_h
        total_gi_input += mcc_op * dt_h

    # Summary metrics
    cmax = float(np.max(cp))
    tmax = float(t[int(np.argmax(cp))])
    auc = float(np_trapz(cp, t))
    lung_f = total_lung_absorbed / dose_mg
    total_f = (total_lung_absorbed + f_gi * total_gi_input) / dose_mg
    total_f = min(total_f, 1.0)

    notes = (
        f"Lung PBPK: {drug_name}, {dose_mg}mg via {device}. "
        f"Lung bioavailability={lung_f:.1%}, total F={total_f:.1%}. "
        f"Cmax={cmax:.4f} mg/L at {tmax:.2f} h. AUC={auc:.3f} mg·h/L."
    )

    return LungPBPKResult(
        drug_name=drug_name,
        device=device,
        dose_mg=dose_mg,
        times_h=t.tolist(),
        cp_mg_L=cp.tolist(),
        a_op_mg=a_op.tolist(),
        a_tb_mg=a_tb.tolist(),
        a_pul_mg=a_pul.tolist(),
        a_gi_mg=a_gi.tolist(),
        cmax_mg_L=cmax,
        tmax_h=tmax,
        auc_0_t=auc,
        lung_bioavailability=float(np.clip(lung_f, 0.0, 1.0)),
        total_bioavailability=float(np.clip(total_f, 0.0, 1.0)),
        deposition=dep,
        notes=notes,
    )


def compare_devices(
    drug_name: str,
    dose_mg: float,
    cl_L_per_h: float,
    vd_L: float,
    devices: list[str] | None = None,
    **kwargs,
) -> list[LungPBPKResult]:
    """Compare lung PK for different inhaler devices.

    Parameters
    ----------
    devices : List of device names (default: all four built-in devices).

    Returns
    -------
    List of LungPBPKResult, one per device.
    """
    if devices is None:
        devices = list(_DEVICE_DEPOSITION.keys())
    return [
        simulate_lung_pk(drug_name, dose_mg, cl_L_per_h, vd_L, device=dev, **kwargs)
        for dev in devices
    ]


__all__ = [
    "InhaledDeposition",
    "LungPBPKResult",
    "simulate_lung_pk",
    "compare_devices",
]
