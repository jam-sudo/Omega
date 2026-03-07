"""Phase 1065 — Drug Tissue Perfusion Sensitivity Analysis.

Local sensitivity analysis of tissue concentrations to organ blood flow changes.
"""

from __future__ import annotations

from dataclasses import dataclass

_TISSUE_FLOW_L_H: dict[str, float] = {
    "liver": 90.0,
    "kidney": 72.0,
    "muscle": 75.0,
    "fat": 18.0,
    "brain": 42.0,
    "lung": 300.0,
}

_TISSUE_VOLUME_L: dict[str, float] = {
    "liver": 1.5,
    "kidney": 0.3,
    "muscle": 28.0,
    "fat": 14.0,
    "brain": 1.4,
    "lung": 0.5,
}

_VALID_TISSUES = frozenset(_TISSUE_FLOW_L_H.keys())


@dataclass(frozen=True)
class PerfusionSensitivityResult:
    drug_name: str
    tissue: str
    base_perfusion_L_h: float
    base_auc_tissue: float
    base_auc_plasma: float
    sensitivity_coefficient: float
    flow_limited: bool
    capacity_limited: bool
    perfusion_class: str
    critical_flow_reduction_pct: float
    notes: str


def _trapz(times: list[float], values: list[float]) -> float:
    """Manual trapezoidal AUC."""
    auc = 0.0
    for i in range(1, len(times)):
        auc += 0.5 * (values[i] + values[i - 1]) * (times[i] - times[i - 1])
    return auc


def _simulate_two_cpt(
    q_L_h: float,
    cl_sys_L_per_h: float,
    vd_sys_L: float,
    dose_mg: float,
    kp_tissue: float,
    v_tissue: float,
    t_end_h: float,
    dt_h: float,
) -> tuple[float, float]:
    """Run 2-compartment simulation, return (auc_tissue, auc_plasma)."""
    ke = cl_sys_L_per_h / vd_sys_L
    k_in = q_L_h / vd_sys_L
    k_out = q_L_h / (v_tissue * kp_tissue)

    c_plasma = dose_mg / vd_sys_L
    c_tissue = 0.0

    times = [0.0]
    plasma_vals = [c_plasma]
    tissue_vals = [c_tissue]

    t = 0.0
    while t < t_end_h - 1e-9:
        dc_plasma = -ke * c_plasma - k_in * c_plasma + k_out * c_tissue
        dc_tissue = k_in * (vd_sys_L / v_tissue) * c_plasma - k_out * c_tissue
        c_plasma = max(0.0, c_plasma + dc_plasma * dt_h)
        c_tissue = max(0.0, c_tissue + dc_tissue * dt_h)
        t += dt_h
        times.append(t)
        plasma_vals.append(c_plasma)
        tissue_vals.append(c_tissue)

    auc_plasma = _trapz(times, plasma_vals)
    auc_tissue = _trapz(times, tissue_vals)
    return auc_tissue, auc_plasma


def analyze_perfusion_sensitivity(
    drug_name: str,
    tissue: str,
    cl_sys_L_per_h: float = 5.0,
    vd_sys_L: float = 30.0,
    dose_mg: float = 10.0,
    kp_tissue: float = 2.0,
    delta_q_pct: float = 20.0,
    t_end_h: float = 24.0,
    dt_h: float = 0.1,
) -> PerfusionSensitivityResult:
    """Perform local sensitivity analysis of tissue AUC to organ blood flow changes."""
    if cl_sys_L_per_h <= 0:
        raise ValueError("cl_sys_L_per_h must be > 0")
    if vd_sys_L <= 0:
        raise ValueError("vd_sys_L must be > 0")
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if kp_tissue <= 0:
        raise ValueError("kp_tissue must be > 0")
    if tissue not in _VALID_TISSUES:
        raise ValueError(f"tissue must be one of {sorted(_VALID_TISSUES)}, got '{tissue}'")
    if not (0 < delta_q_pct < 100):
        raise ValueError("delta_q_pct must be in (0, 100)")

    base_q = _TISSUE_FLOW_L_H[tissue]
    v_tissue = _TISSUE_VOLUME_L[tissue]

    # Base simulation
    base_auc_tissue, base_auc_plasma = _simulate_two_cpt(
        base_q, cl_sys_L_per_h, vd_sys_L, dose_mg, kp_tissue, v_tissue, t_end_h, dt_h
    )

    # High-flow simulation
    high_q = base_q * (1.0 + delta_q_pct / 100.0)
    high_auc_tissue, _ = _simulate_two_cpt(
        high_q, cl_sys_L_per_h, vd_sys_L, dose_mg, kp_tissue, v_tissue, t_end_h, dt_h
    )

    # Normalized sensitivity coefficient
    if base_auc_tissue > 0:
        sc = (high_auc_tissue - base_auc_tissue) / base_auc_tissue / (delta_q_pct / 100.0)
    else:
        sc = 0.0

    flow_limited = abs(sc) > 0.5
    capacity_limited = abs(sc) < 0.1

    if flow_limited:
        perfusion_class = "flow_limited"
    elif capacity_limited:
        perfusion_class = "capacity_limited"
    else:
        perfusion_class = "intermediate"

    # Critical flow reduction: find % Q reduction that halves AUC_tissue
    critical_reduction = 100.0
    for pct in range(10, 100, 10):
        reduced_q = base_q * (1.0 - pct / 100.0)
        red_auc_tissue, _ = _simulate_two_cpt(
            reduced_q, cl_sys_L_per_h, vd_sys_L, dose_mg, kp_tissue, v_tissue, t_end_h, dt_h
        )
        if red_auc_tissue < 0.5 * base_auc_tissue:
            critical_reduction = float(pct)
            break

    notes = (
        f"{drug_name} in {tissue}: SC={sc:.3f}, class={perfusion_class}, "
        f"base_Q={base_q} L/h, kp={kp_tissue}"
    )

    return PerfusionSensitivityResult(
        drug_name=drug_name,
        tissue=tissue,
        base_perfusion_L_h=base_q,
        base_auc_tissue=base_auc_tissue,
        base_auc_plasma=base_auc_plasma,
        sensitivity_coefficient=sc,
        flow_limited=flow_limited,
        capacity_limited=capacity_limited,
        perfusion_class=perfusion_class,
        critical_flow_reduction_pct=critical_reduction,
        notes=notes,
    )
