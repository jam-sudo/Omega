"""
Phase 638 — Drug Muscle Tissue Distribution
Models drug distribution into skeletal muscle tissue, important for
muscle toxicity and local drug action.
"""

import math
from dataclasses import dataclass


@dataclass
class MuscleTissuePKResult:
    drug_name: str
    dose_mg: float
    muscle_mass_kg: float
    times_h: list
    c_plasma_mg_L: list
    c_muscle_mg_g: list
    kp_muscle: float
    cmax_plasma_mg_L: float
    cmax_muscle_mg_g: float
    auc_plasma_mg_h_per_L: float
    auc_muscle_mg_h_per_g: float
    muscle_to_plasma_ratio: float
    myotoxicity_index: float
    t_equilibration_h: float
    notes: str


def simulate_muscle_pk(
    drug_name: str,
    dose_mg: float,
    logP: float,
    mw_Da: float,
    cl_sys_L_per_h: float,
    vd_sys_L: float,
    muscle_mass_kg: float = 28.0,
    sex: str = "male",
    t_end_h: float = 48.0,
    dt_h: float = 0.1,
) -> MuscleTissuePKResult:
    """
    Simulate drug PK in plasma and skeletal muscle tissue.

    Parameters
    ----------
    drug_name : str
    dose_mg : float          oral dose (mg)
    logP : float             octanol-water partition coefficient
    mw_Da : float            molecular weight (Da)
    cl_sys_L_per_h : float   systemic clearance (L/h)
    vd_sys_L : float         systemic volume of distribution (L)
    muscle_mass_kg : float   skeletal muscle mass (kg), default 28
    sex : str                "male" or "female"
    t_end_h : float          simulation end time (h)
    dt_h : float             time step (h)
    """
    # --- Validation ---
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if cl_sys_L_per_h <= 0:
        raise ValueError("cl_sys_L_per_h must be > 0")
    if vd_sys_L <= 0:
        raise ValueError("vd_sys_L must be > 0")
    if muscle_mass_kg <= 0:
        raise ValueError("muscle_mass_kg must be > 0")
    if sex not in ("male", "female"):
        raise ValueError("sex must be 'male' or 'female'")

    # --- Sex-based effective muscle mass ---
    if sex == "female":
        muscle_mass_kg_eff = muscle_mass_kg * 0.85
    else:
        muscle_mass_kg_eff = muscle_mass_kg

    # --- Muscle partition coefficient ---
    kp_muscle = max(0.3, (logP + 0.5) * 0.4) / max(1.0, mw_Da / 300.0)
    kp_muscle = max(0.1, min(8.0, kp_muscle))

    # --- Volumes and flows ---
    Vd_muscle_L = muscle_mass_kg_eff * kp_muscle  # L (density ~1 kg/L)
    Qm = 0.15 * muscle_mass_kg_eff  # L/h (resting muscle blood flow)

    # Transfer rate constants
    k_in = Qm / vd_sys_L  # /h  plasma → muscle
    k_out = Qm / Vd_muscle_L  # /h  muscle → plasma

    # Systemic PK rate constants
    k_el = cl_sys_L_per_h / vd_sys_L  # /h  elimination from plasma
    ka = 1.2  # /h  oral absorption

    # --- Initial conditions ---
    A_gut = dose_mg * 0.85  # mg  (85% oral bioavailability)
    C_plasma = 0.0  # mg/L
    C_muscle = 0.0  # mg/L (in muscle volume terms)

    # --- Storage ---
    times_h = []
    c_plasma_list = []
    c_muscle_list = []

    t = 0.0
    n_steps = int(round(t_end_h / dt_h)) + 1

    for step in range(n_steps):
        # Convert C_muscle (mg/L in muscle volume) to mg/g tissue
        c_muscle_mg_g = C_muscle * kp_muscle / 1000.0

        times_h.append(t)
        c_plasma_list.append(C_plasma)
        c_muscle_list.append(c_muscle_mg_g)

        if step == n_steps - 1:
            break

        # --- ODEs (Forward Euler) ---
        dA_gut = -ka * A_gut

        dC_plasma = (
            ka * A_gut / vd_sys_L
            - k_el * C_plasma
            - k_in * C_plasma
            + k_out * C_muscle * (Vd_muscle_L / vd_sys_L)
        )

        dC_muscle = k_in * C_plasma * (vd_sys_L / Vd_muscle_L) - k_out * C_muscle

        A_gut = max(0.0, A_gut + dA_gut * dt_h)
        C_plasma = max(0.0, C_plasma + dC_plasma * dt_h)
        C_muscle = max(0.0, C_muscle + dC_muscle * dt_h)

        t += dt_h

    # --- Derived PK metrics ---
    cmax_plasma = max(c_plasma_list)
    cmax_muscle = max(c_muscle_list)

    # Trapezoidal AUC
    auc_plasma = sum(
        0.5 * (c_plasma_list[i] + c_plasma_list[i - 1]) * (times_h[i] - times_h[i - 1])
        for i in range(1, len(times_h))
    )
    auc_muscle = sum(
        0.5 * (c_muscle_list[i] + c_muscle_list[i - 1]) * (times_h[i] - times_h[i - 1])
        for i in range(1, len(times_h))
    )

    muscle_to_plasma_ratio = auc_muscle / auc_plasma if auc_plasma > 0 else 0.0

    # Myotoxicity index: higher lipophilicity → more mitochondrial toxicity
    myotoxicity_index = cmax_muscle * max(1.0, logP) / 2.0

    # Time to 90% equilibration (≈ 3 elimination half-lives of muscle compartment)
    t_equilibration_h = 3.0 * math.log(2.0) / k_out

    notes = (
        f"kp_muscle={kp_muscle:.3f}, Vd_muscle={Vd_muscle_L:.2f} L, "
        f"Qm={Qm:.2f} L/h, k_in={k_in:.3f} /h, k_out={k_out:.3f} /h, "
        f"muscle_mass_eff={muscle_mass_kg_eff:.1f} kg"
    )

    return MuscleTissuePKResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        muscle_mass_kg=muscle_mass_kg,
        times_h=times_h,
        c_plasma_mg_L=c_plasma_list,
        c_muscle_mg_g=c_muscle_list,
        kp_muscle=kp_muscle,
        cmax_plasma_mg_L=cmax_plasma,
        cmax_muscle_mg_g=cmax_muscle,
        auc_plasma_mg_h_per_L=auc_plasma,
        auc_muscle_mg_h_per_g=auc_muscle,
        muscle_to_plasma_ratio=muscle_to_plasma_ratio,
        myotoxicity_index=myotoxicity_index,
        t_equilibration_h=t_equilibration_h,
        notes=notes,
    )
