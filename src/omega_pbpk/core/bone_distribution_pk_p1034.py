"""Phase 1034 — Drug Bone Distribution PK.

Models drug distribution into cortical and trabecular bone compartments.
Relevant for bisphosphonates, fluoroquinolones, antibiotics.

3-compartment model: plasma + cortical bone + trabecular bone.
Uses forward Euler ODE integration.
"""

from __future__ import annotations

from dataclasses import dataclass, field


def _trapz(times: list, values: list) -> float:
    """Manual trapezoidal integration."""
    auc = 0.0
    for i in range(1, len(times)):
        dt = times[i] - times[i - 1]
        auc += 0.5 * (values[i] + values[i - 1]) * dt
    return auc


_VALID_DRUG_CLASSES = {"antibiotic", "bisphosphonate", "fluoroquinolone", "general"}

_KP_MULTIPLIERS = {
    "bisphosphonate": (10.0, 5.0),
    "fluoroquinolone": (2.0, 2.0),
    "antibiotic": (1.0, 1.0),
    "general": (0.5, 0.5),
}


@dataclass
class BoneDistributionResult:
    drug_name: str
    dose_mg: float
    times_h: list = field(default_factory=list)
    c_plasma_mg_L: list = field(default_factory=list)
    c_cortical_mg_L: list = field(default_factory=list)
    c_trabecular_mg_L: list = field(default_factory=list)
    auc_plasma: float = 0.0
    auc_cortical: float = 0.0
    auc_trabecular: float = 0.0
    cmax_cortical: float = 0.0
    tmax_cortical_h: float = 0.0
    cortical_to_plasma_auc_ratio: float = 0.0
    trabecular_to_plasma_auc_ratio: float = 0.0
    bone_affinity: str = "low"
    notes: str = ""


def simulate_bone_distribution(
    drug_name: str,
    dose_mg: float,
    drug_class: str = "antibiotic",
    cl_sys_L_per_h: float = 5.0,
    vd_sys_L: float = 30.0,
    q_cortical_L_h: float = 0.3,
    q_trabecular_L_h: float = 0.6,
    v_cortical_L: float = 1.0,
    v_trabecular_L: float = 0.5,
    kp_cortical: float = 0.3,
    kp_trabecular: float = 0.8,
    t_end_h: float = 72.0,
    dt_h: float = 0.1,
) -> BoneDistributionResult:
    """Simulate drug distribution into cortical and trabecular bone."""
    # Validation
    if dose_mg <= 0:
        raise ValueError(f"dose_mg must be > 0, got {dose_mg}")
    if cl_sys_L_per_h <= 0:
        raise ValueError(f"cl_sys_L_per_h must be > 0, got {cl_sys_L_per_h}")
    if vd_sys_L <= 0:
        raise ValueError(f"vd_sys_L must be > 0, got {vd_sys_L}")
    if q_cortical_L_h <= 0:
        raise ValueError(f"q_cortical_L_h must be > 0, got {q_cortical_L_h}")
    if q_trabecular_L_h <= 0:
        raise ValueError(f"q_trabecular_L_h must be > 0, got {q_trabecular_L_h}")
    if v_cortical_L <= 0:
        raise ValueError(f"v_cortical_L must be > 0, got {v_cortical_L}")
    if v_trabecular_L <= 0:
        raise ValueError(f"v_trabecular_L must be > 0, got {v_trabecular_L}")
    if kp_cortical <= 0:
        raise ValueError(f"kp_cortical must be > 0, got {kp_cortical}")
    if kp_trabecular <= 0:
        raise ValueError(f"kp_trabecular must be > 0, got {kp_trabecular}")
    if drug_class not in _VALID_DRUG_CLASSES:
        raise ValueError(f"drug_class must be one of {_VALID_DRUG_CLASSES}, got '{drug_class}'")

    # Apply drug class multipliers to Kp values
    mult_c, mult_t = _KP_MULTIPLIERS[drug_class]
    kp_cortical_eff = kp_cortical * mult_c
    kp_trabecular_eff = kp_trabecular * mult_t

    # Rate constants
    ke = cl_sys_L_per_h / vd_sys_L
    k_c_in = q_cortical_L_h / vd_sys_L
    k_c_out = q_cortical_L_h / (v_cortical_L * kp_cortical_eff)
    k_t_in = q_trabecular_L_h / vd_sys_L
    k_t_out = q_trabecular_L_h / (v_trabecular_L * kp_trabecular_eff)

    # Initial conditions
    c_plasma = dose_mg / vd_sys_L
    c_cortical = 0.0
    c_trabecular = 0.0

    # Storage
    times = []
    c_plasma_list = []
    c_cortical_list = []
    c_trabecular_list = []

    t = 0.0
    n_steps = int(t_end_h / dt_h) + 1

    for _i in range(n_steps):
        times.append(t)
        c_plasma_list.append(max(0.0, c_plasma))
        c_cortical_list.append(max(0.0, c_cortical))
        c_trabecular_list.append(max(0.0, c_trabecular))

        # ODEs (forward Euler)
        dc_plasma = (
            -ke * c_plasma
            - k_c_in * c_plasma
            + k_c_out * c_cortical
            - k_t_in * c_plasma
            + k_t_out * c_trabecular
        )
        dc_cortical = k_c_in * (vd_sys_L / v_cortical_L) * c_plasma - k_c_out * c_cortical
        dc_trabecular = k_t_in * (vd_sys_L / v_trabecular_L) * c_plasma - k_t_out * c_trabecular

        c_plasma = max(0.0, c_plasma + dc_plasma * dt_h)
        c_cortical = max(0.0, c_cortical + dc_cortical * dt_h)
        c_trabecular = max(0.0, c_trabecular + dc_trabecular * dt_h)

        t += dt_h

    # AUCs
    auc_plasma = _trapz(times, c_plasma_list)
    auc_cortical = _trapz(times, c_cortical_list)
    auc_trabecular = _trapz(times, c_trabecular_list)

    # Cmax cortical and tmax
    cmax_cortical = max(c_cortical_list)
    idx_max = c_cortical_list.index(cmax_cortical)
    tmax_cortical_h = times[idx_max]

    # Ratios
    if auc_plasma > 0:
        cortical_ratio = auc_cortical / auc_plasma
        trabecular_ratio = auc_trabecular / auc_plasma
    else:
        cortical_ratio = 0.0
        trabecular_ratio = 0.0

    # Bone affinity based on average ratio
    avg_ratio = (cortical_ratio + trabecular_ratio) / 2.0
    if avg_ratio > 2.0:
        bone_affinity = "high"
    elif avg_ratio >= 0.5:
        bone_affinity = "moderate"
    else:
        bone_affinity = "low"

    # Notes
    notes_parts = [
        f"Drug class: {drug_class}",
        f"kp_cortical_eff={kp_cortical_eff:.2f}, kp_trabecular_eff={kp_trabecular_eff:.2f}",
        f"Bone affinity: {bone_affinity} (avg AUC ratio={avg_ratio:.2f})",
    ]
    if drug_class == "bisphosphonate":
        notes_parts.append(
            "Bisphosphonates have very high bone affinity due to hydroxyapatite binding"
        )
    elif drug_class == "fluoroquinolone":
        notes_parts.append("Fluoroquinolones chelate calcium in bone, increasing distribution")

    return BoneDistributionResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        times_h=times,
        c_plasma_mg_L=c_plasma_list,
        c_cortical_mg_L=c_cortical_list,
        c_trabecular_mg_L=c_trabecular_list,
        auc_plasma=auc_plasma,
        auc_cortical=auc_cortical,
        auc_trabecular=auc_trabecular,
        cmax_cortical=cmax_cortical,
        tmax_cortical_h=tmax_cortical_h,
        cortical_to_plasma_auc_ratio=cortical_ratio,
        trabecular_to_plasma_auc_ratio=trabecular_ratio,
        bone_affinity=bone_affinity,
        notes="; ".join(notes_parts),
    )
