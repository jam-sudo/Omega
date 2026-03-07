"""Combined oral contraceptive (OC) pharmacokinetics.

Models the PK of combined oral contraceptives (ethinylestradiol + progestin).
Important for DDI assessment — enzyme inducers (e.g., rifampicin, carbamazepine)
can substantially reduce contraceptive steroid exposure and reduce efficacy.

Default parameters reflect published population PK data:
- Ethinylestradiol (EE): MW=296 g/mol, highly protein-bound (fup=0.02),
  CYP3A4-mediated hepatic metabolism + enterohepatic circulation.
- Levonorgestrel (LNG): MW=312 g/mol, fup=0.015, CYP3A4/2C9 metabolism.

References (simplified model):
    Back DJ et al. Clin Pharmacokinet 1987; 12: 227.
    Stanczyk FZ et al. Contraception 2013; 87: 706.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class OCPKResult:
    """Result from combined OC pharmacokinetic simulation.

    Concentrations are in ng/mL; doses in ug.

    Attributes:
        dose_ee_ug: EE dose (ug).
        dose_progestin_ug: Progestin dose (ug).
        progestin_type: Progestin name (e.g., "LNG").
        times_h: Simulation time points (h).
        c_ee_ng_mL: EE plasma concentrations (ng/mL).
        c_progestin_ng_mL: Progestin plasma concentrations (ng/mL).
        cmax_ee: EE Cmax (ng/mL).
        tmax_ee_h: Time to EE Cmax (h).
        auc_ee: EE AUC (ng*h/mL), linear trapezoidal.
        cmax_progestin: Progestin Cmax (ng/mL).
        tmax_progestin_h: Time to progestin Cmax (h).
        auc_progestin: Progestin AUC (ng*h/mL).
        ee_secondary_peak_present: True if ERC secondary EE peak detected.
        notes: Descriptive simulation notes.
    """

    dose_ee_ug: float
    dose_progestin_ug: float
    progestin_type: str
    times_h: tuple[float, ...]
    c_ee_ng_mL: tuple[float, ...]
    c_progestin_ng_mL: tuple[float, ...]
    cmax_ee: float
    tmax_ee_h: float
    auc_ee: float
    cmax_progestin: float
    tmax_progestin_h: float
    auc_progestin: float
    ee_secondary_peak_present: bool
    notes: str


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _oral_1cpt_conc(
    dose_ug: float,
    f_oral: float,
    vd_L: float,
    ka_per_h: float,
    ke_per_h: float,
    t: float,
) -> float:
    """Return oral 1-compartment concentration at time t (ng/mL).

    Uses absorbed dose = F * dose_ug (ug = 1000 ng) scaled by Vd.
    C(t) = (F*D*ka / (Vd*(ka-ke))) * (exp(-ke*t) - exp(-ka*t))

    Units:
        dose_ug: ug → converted to ng inside (×1000).
        vd_L: L.
        result: ng/mL  (= ug/L when 1 ug = 1000 ng and 1 L = 1000 mL, so
            ng/mL = ug/L → the dose in ug / Vd in L gives ug/L = ng/mL directly).

    Note: 1 ug/L = 1 ng/mL.  So dose_ug / vd_L → ug/L = ng/mL.
    """
    if t < 0:
        return 0.0
    # Avoid singularity when ka ≈ ke
    if abs(ka_per_h - ke_per_h) < 1e-8:
        ke_per_h = ke_per_h * (1.0 + 1e-6)

    absorbed_dose = f_oral * dose_ug  # ug
    # C = F*D*ka / (Vd*(ka-ke)) * (e^{-ke*t} - e^{-ka*t})
    # Units: ug * (1/h) / (L * (1/h)) = ug/L = ng/mL  ✓
    factor = absorbed_dose * ka_per_h / (vd_L * (ka_per_h - ke_per_h))
    c = factor * (math.exp(-ke_per_h * t) - math.exp(-ka_per_h * t))
    return max(c, 0.0)


def _iv_1cpt_conc(
    dose_ug: float,
    vd_L: float,
    ke_per_h: float,
    t: float,
) -> float:
    """IV bolus 1-compartment concentration (ng/mL = ug/L)."""
    if t < 0:
        return 0.0
    c0 = dose_ug / vd_L  # ug/L = ng/mL
    return c0 * math.exp(-ke_per_h * t)


def _trapezoidal_auc(times: list[float], concs: list[float]) -> float:
    """Linear trapezoidal AUC."""
    auc = 0.0
    for i in range(1, len(times)):
        auc += 0.5 * (concs[i] + concs[i - 1]) * (times[i] - times[i - 1])
    return auc


def _detect_secondary_peak(times: list[float], concs: list[float], tmax_h: float) -> bool:
    """Detect a secondary peak after the primary tmax."""
    if not concs or tmax_h is None:
        return False

    primary_idx = concs.index(max(concs))
    if primary_idx + 2 >= len(concs):
        return False

    cmax = concs[primary_idx]
    threshold = cmax * 0.05  # secondary peak > 5% Cmax

    for i in range(primary_idx + 1, len(concs) - 1):
        if (
            concs[i] > concs[i - 1]
            and concs[i] > concs[i + 1]
            and concs[i] > threshold
            and times[i] > tmax_h + 0.5
        ):
            return True
    return False


def _simulate_ee_with_erc(
    dose_ee_ug: float,
    f_ee: float,
    cl_ee_L_per_h: float,
    vd_ee_L: float,
    ka_ee_per_h: float,
    erc_ee_fraction: float,
    k_reabs_per_h: float,
    times: list[float],
    dt_h: float,
) -> list[float]:
    """Forward Euler ODE for EE with enterohepatic recirculation.

    State:
        C: plasma concentration (ng/mL = ug/L)
        A_gut: oral dose remaining in gut (ug)
        A_bile: accumulated biliary EE (ug)
        A_reabs_gut: EE in gut for ERC reabsorption (ug)

    Returns: list of concentrations (ng/mL) at each time step.
    """
    ke = cl_ee_L_per_h / vd_ee_L  # 1/h

    C = 0.0  # start at 0 for oral
    A_gut = dose_ee_ug  # all dose in gut at t=0
    A_bile = 0.0
    A_reabs_gut = 0.0

    # Bile secretion fraction of CL (assumed 20% of EE CL via bile)
    f_bile_ee = 0.2
    ke_bile = ke * f_bile_ee  # portion of elimination that goes to bile

    bile_release_start_h = 4.0  # gallbladder contraction ~4h post-dose
    bile_release_duration_h = 2.0
    bile_released = False

    concs = []

    for _i, t in enumerate(times):
        concs.append(C)

        # Oral absorption from gut
        oral_rate = ka_ee_per_h * A_gut  # ug/h

        # ERC reabsorption
        reabs_rate = k_reabs_per_h * A_reabs_gut  # ug/h

        # Bile secretion (remove from plasma, add to bile pool)
        bile_sec_rate = ke_bile * C * vd_ee_L  # ug/h (ke_bile * C * Vd)

        # Net plasma dC/dt = absorption - elimination + ERC
        dC = (
            (oral_rate + reabs_rate) / vd_ee_L  # into plasma (ng/mL per h)
            - ke * C  # elimination
        )
        dA_gut = -oral_rate

        # Bile pool: accumulate secreted drug, then release
        if (
            not bile_released
            and t >= bile_release_start_h
            and t < bile_release_start_h + bile_release_duration_h
        ):
            release_rate = A_bile / bile_release_duration_h  # ug/h
            dA_bile = bile_sec_rate - release_rate
            dA_reabs_gut = release_rate * erc_ee_fraction - reabs_rate
        else:
            dA_bile = bile_sec_rate
            dA_reabs_gut = -reabs_rate
            if t >= bile_release_start_h + bile_release_duration_h:
                bile_released = True

        # Euler update
        C = max(C + dC * dt_h, 0.0)
        A_gut = max(A_gut + dA_gut * dt_h, 0.0)
        A_bile = max(A_bile + dA_bile * dt_h, 0.0)
        A_reabs_gut = max(A_reabs_gut + dA_reabs_gut * dt_h, 0.0)

    return concs


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def simulate_oc_pk(
    dose_ee_ug: float = 30.0,
    dose_progestin_ug: float = 150.0,
    progestin_type: str = "LNG",
    cl_ee_L_per_h: float = 27.0,
    vd_ee_L: float = 211.0,
    ka_ee_per_h: float = 2.0,
    f_ee: float = 0.45,
    cl_progestin_L_per_h: float = 4.5,
    vd_progestin_L: float = 130.0,
    ka_progestin_per_h: float = 1.5,
    f_progestin: float = 0.88,
    erc_ee_fraction: float = 0.3,
    t_end_h: float = 72.0,
    dt_h: float = 0.1,
) -> OCPKResult:
    """Simulate combined oral contraceptive pharmacokinetics.

    Models EE and progestin as independent 1-compartment systems with oral
    first-order absorption. EE additionally has enterohepatic recirculation.

    Default parameters represent a 30 ug EE / 150 ug LNG monophasic pill
    based on published population PK values.

    Args:
        dose_ee_ug: EE dose per tablet (ug). Must be > 0.
        dose_progestin_ug: Progestin dose per tablet (ug). Must be > 0.
        progestin_type: Progestin name for labeling (e.g., "LNG").
        cl_ee_L_per_h: EE apparent oral clearance CL/F (L/h). Must be > 0.
        vd_ee_L: EE volume of distribution Vd/F (L). Must be > 0.
        ka_ee_per_h: EE absorption rate constant (1/h). Must be > 0.
        f_ee: EE absolute oral bioavailability [0, 1].
        cl_progestin_L_per_h: Progestin CL/F (L/h). Must be > 0.
        vd_progestin_L: Progestin Vd/F (L). Must be > 0.
        ka_progestin_per_h: Progestin absorption rate (1/h). Must be > 0.
        f_progestin: Progestin absolute bioavailability [0, 1].
        erc_ee_fraction: Fraction of bile EE undergoing enterohepatic recirculation.
        t_end_h: Simulation end time (h). Must be > 0.
        dt_h: ODE time step (h). Must be > 0.

    Returns:
        OCPKResult with full PK profiles and summary metrics.

    Raises:
        ValueError: If any parameter fails validation.
    """
    # --- Validation ---
    if dose_ee_ug <= 0:
        raise ValueError(f"dose_ee_ug must be > 0, got {dose_ee_ug}")
    if dose_progestin_ug <= 0:
        raise ValueError(f"dose_progestin_ug must be > 0, got {dose_progestin_ug}")
    if cl_ee_L_per_h <= 0:
        raise ValueError(f"cl_ee_L_per_h must be > 0, got {cl_ee_L_per_h}")
    if vd_ee_L <= 0:
        raise ValueError(f"vd_ee_L must be > 0, got {vd_ee_L}")
    if ka_ee_per_h <= 0:
        raise ValueError(f"ka_ee_per_h must be > 0, got {ka_ee_per_h}")
    if cl_progestin_L_per_h <= 0:
        raise ValueError(f"cl_progestin_L_per_h must be > 0, got {cl_progestin_L_per_h}")
    if vd_progestin_L <= 0:
        raise ValueError(f"vd_progestin_L must be > 0, got {vd_progestin_L}")
    if ka_progestin_per_h <= 0:
        raise ValueError(f"ka_progestin_per_h must be > 0, got {ka_progestin_per_h}")
    if not (0.0 <= f_ee <= 1.0):
        raise ValueError(f"f_ee must be in [0, 1], got {f_ee}")
    if not (0.0 <= f_progestin <= 1.0):
        raise ValueError(f"f_progestin must be in [0, 1], got {f_progestin}")
    if not (0.0 <= erc_ee_fraction <= 1.0):
        raise ValueError(f"erc_ee_fraction must be in [0, 1], got {erc_ee_fraction}")
    if t_end_h <= 0:
        raise ValueError(f"t_end_h must be > 0, got {t_end_h}")
    if dt_h <= 0:
        raise ValueError(f"dt_h must be > 0, got {dt_h}")

    # --- Time grid ---
    n_steps = int(round(t_end_h / dt_h)) + 1
    times = [i * dt_h for i in range(n_steps)]

    # --- EE simulation with ERC ---
    k_reabs = 0.2  # bile reabsorption rate (1/h)
    c_ee = _simulate_ee_with_erc(
        dose_ee_ug=dose_ee_ug,
        f_ee=f_ee,
        cl_ee_L_per_h=cl_ee_L_per_h,
        vd_ee_L=vd_ee_L,
        ka_ee_per_h=ka_ee_per_h,
        erc_ee_fraction=erc_ee_fraction,
        k_reabs_per_h=k_reabs,
        times=times,
        dt_h=dt_h,
    )

    # --- Progestin simulation (simple 1-cpt oral) ---
    ke_p = cl_progestin_L_per_h / vd_progestin_L
    c_progestin = [
        _oral_1cpt_conc(
            dose_ug=dose_progestin_ug,
            f_oral=f_progestin,
            vd_L=vd_progestin_L,
            ka_per_h=ka_progestin_per_h,
            ke_per_h=ke_p,
            t=t,
        )
        for t in times
    ]

    # --- PK metrics ---
    cmax_ee = max(c_ee)
    tmax_ee_h = times[c_ee.index(cmax_ee)]
    auc_ee = _trapezoidal_auc(times, c_ee)

    cmax_progestin = max(c_progestin)
    tmax_progestin_h = times[c_progestin.index(cmax_progestin)]
    auc_progestin = _trapezoidal_auc(times, c_progestin)

    ee_secondary = _detect_secondary_peak(times, c_ee, tmax_ee_h)

    notes_parts = [
        f"EE: CL={cl_ee_L_per_h} L/h, Vd={vd_ee_L} L, F={f_ee * 100:.0f}%",
        f"{progestin_type}: CL={cl_progestin_L_per_h} L/h, "
        f"Vd={vd_progestin_L} L, F={f_progestin * 100:.0f}%",
        f"EE ERC fraction={erc_ee_fraction:.2f}",
    ]
    if ee_secondary:
        notes_parts.append("EE secondary peak (ERC) detected")

    return OCPKResult(
        dose_ee_ug=dose_ee_ug,
        dose_progestin_ug=dose_progestin_ug,
        progestin_type=progestin_type,
        times_h=tuple(times),
        c_ee_ng_mL=tuple(c_ee),
        c_progestin_ng_mL=tuple(c_progestin),
        cmax_ee=cmax_ee,
        tmax_ee_h=tmax_ee_h,
        auc_ee=auc_ee,
        cmax_progestin=cmax_progestin,
        tmax_progestin_h=tmax_progestin_h,
        auc_progestin=auc_progestin,
        ee_secondary_peak_present=ee_secondary,
        notes="; ".join(notes_parts),
    )


def inducer_impact(
    induction_factor_ee: float,
    induction_factor_progestin: float | None = None,
    dose_ee_ug: float = 30.0,
    dose_progestin_ug: float = 150.0,
    progestin_type: str = "LNG",
    cl_ee_L_per_h: float = 27.0,
    vd_ee_L: float = 211.0,
    ka_ee_per_h: float = 2.0,
    f_ee: float = 0.45,
    cl_progestin_L_per_h: float = 4.5,
    vd_progestin_L: float = 130.0,
    ka_progestin_per_h: float = 1.5,
    f_progestin: float = 0.88,
    erc_ee_fraction: float = 0.3,
    t_end_h: float = 72.0,
    dt_h: float = 0.1,
) -> OCPKResult:
    """Simulate OC PK in the presence of an enzyme inducer.

    Scales the EE and progestin clearances by the induction factor. An induction
    factor of 3.5 represents strong inducers like rifampicin (CL × 3.5 → AUC / 3.5).

    Args:
        induction_factor_ee: Factor to multiply EE clearance by (> 0). Typical
            values: rifampicin=3.5, carbamazepine=2.0, phenytoin=2.5.
        induction_factor_progestin: Factor to multiply progestin CL by. If None,
            uses the same value as induction_factor_ee.
        **kwargs: All other arguments match simulate_oc_pk().

    Returns:
        OCPKResult reflecting induced clearance.

    Raises:
        ValueError: If induction_factor_ee <= 0.
    """
    if induction_factor_ee <= 0:
        raise ValueError(f"induction_factor_ee must be > 0, got {induction_factor_ee}")

    if induction_factor_progestin is None:
        induction_factor_progestin = induction_factor_ee

    if induction_factor_progestin <= 0:
        raise ValueError(
            f"induction_factor_progestin must be > 0, got {induction_factor_progestin}"
        )

    induced_cl_ee = cl_ee_L_per_h * induction_factor_ee
    induced_cl_progestin = cl_progestin_L_per_h * induction_factor_progestin

    result = simulate_oc_pk(
        dose_ee_ug=dose_ee_ug,
        dose_progestin_ug=dose_progestin_ug,
        progestin_type=progestin_type,
        cl_ee_L_per_h=induced_cl_ee,
        vd_ee_L=vd_ee_L,
        ka_ee_per_h=ka_ee_per_h,
        f_ee=f_ee,
        cl_progestin_L_per_h=induced_cl_progestin,
        vd_progestin_L=vd_progestin_L,
        ka_progestin_per_h=ka_progestin_per_h,
        f_progestin=f_progestin,
        erc_ee_fraction=erc_ee_fraction,
        t_end_h=t_end_h,
        dt_h=dt_h,
    )

    return result
