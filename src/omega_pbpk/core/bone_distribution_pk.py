"""Phase 920 — Drug Bone/Skeletal Distribution.

Models drug distribution into cortical and trabecular bone tissue.
Relevant for bisphosphonates, fluoroquinolones, and tetracyclines.

Note: This is distinct from bone_marrow_pk.py — it models cortical/trabecular
bone compartments, not bone marrow.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class BoneDistributionPKResult:
    """Result of bone distribution PK simulation."""

    drug_name: str
    dose_mg: float
    times_h: list[float] = field(default_factory=list)
    c_plasma_mg_L: list[float] = field(default_factory=list)
    c_cortical_mg_L: list[float] = field(default_factory=list)
    c_trabecular_mg_L: list[float] = field(default_factory=list)
    cmax_plasma: float = 0.0
    auc_plasma: float = 0.0
    cmax_cortical: float = 0.0
    auc_cortical: float = 0.0
    cmax_trabecular: float = 0.0
    auc_trabecular: float = 0.0
    bone_plasma_kp_cortical: float = 0.0
    bone_plasma_kp_trabecular: float = 0.0
    bone_sequestration_fraction: float = 0.0
    notes: str = ""


def _trapz(times: list[float], values: list[float]) -> float:
    """Manual trapezoidal integration."""
    auc = 0.0
    for i in range(1, len(times)):
        auc += 0.5 * (values[i] + values[i - 1]) * (times[i] - times[i - 1])
    return auc


def simulate_bone_distribution(
    drug_name: str,
    dose_mg: float,
    k_abs_per_h: float = 1.0,
    k_cort_uptake: float = 0.02,
    k_cort_release: float = 0.001,
    k_trab_uptake: float = 0.1,
    k_trab_release: float = 0.05,
    cl_L_per_h: float = 10.0,
    vd_L: float = 50.0,
    v_cort_L: float = 4.0,
    v_trab_L: float = 2.0,
    t_end_h: float = 168.0,
    dt_h: float = 0.5,
) -> BoneDistributionPKResult:
    """Simulate drug distribution into cortical and trabecular bone.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    dose_mg:
        Oral dose in milligrams.
    k_abs_per_h:
        First-order absorption rate constant (1/h).
    k_cort_uptake:
        First-order uptake rate into cortical bone from plasma (1/h).
    k_cort_release:
        First-order release rate from cortical bone to plasma (1/h).
    k_trab_uptake:
        First-order uptake rate into trabecular bone from plasma (1/h).
    k_trab_release:
        First-order release rate from trabecular bone to plasma (1/h).
    cl_L_per_h:
        Systemic clearance (L/h).
    vd_L:
        Volume of distribution (L).
    v_cort_L:
        Volume of cortical bone (L).
    v_trab_L:
        Volume of trabecular bone (L).
    t_end_h:
        Simulation end time (h); default 168 h (1 week).
    dt_h:
        Integration step size (h).

    Returns
    -------
    BoneDistributionPKResult
    """
    if dose_mg <= 0:
        raise ValueError(f"dose_mg must be positive, got {dose_mg}")
    if cl_L_per_h <= 0:
        raise ValueError(f"cl_L_per_h must be positive, got {cl_L_per_h}")
    if v_cort_L <= 0:
        raise ValueError(f"v_cort_L must be positive, got {v_cort_L}")

    ke = cl_L_per_h / vd_L  # elimination rate constant (1/h)

    # State variables (amounts in mg)
    a_gut = dose_mg
    a_plasma = 0.0
    a_cort = 0.0
    a_trab = 0.0

    times: list[float] = []
    c_plasma_list: list[float] = []
    c_cort_list: list[float] = []
    c_trab_list: list[float] = []

    t = 0.0
    n_steps = int(round(t_end_h / dt_h)) + 1

    for _ in range(n_steps):
        c_plasma = a_plasma / vd_L
        c_cort = a_cort / v_cort_L
        c_trab = a_trab / v_trab_L

        times.append(t)
        c_plasma_list.append(c_plasma)
        c_cort_list.append(c_cort)
        c_trab_list.append(c_trab)

        if t >= t_end_h:
            break

        # ODEs — Forward Euler
        da_gut = -k_abs_per_h * a_gut
        da_plasma = (
            k_abs_per_h * a_gut
            - ke * a_plasma
            - k_cort_uptake * a_plasma
            + k_cort_release * a_cort
            - k_trab_uptake * a_plasma
            + k_trab_release * a_trab
        )
        da_cort = k_cort_uptake * a_plasma - k_cort_release * a_cort
        da_trab = k_trab_uptake * a_plasma - k_trab_release * a_trab

        a_gut = max(0.0, a_gut + da_gut * dt_h)
        a_plasma = max(0.0, a_plasma + da_plasma * dt_h)
        a_cort = max(0.0, a_cort + da_cort * dt_h)
        a_trab = max(0.0, a_trab + da_trab * dt_h)

        t = round(t + dt_h, 10)

    # Final amounts for sequestration fraction
    a_cort_final = a_cort
    a_trab_final = a_trab

    # PK metrics
    cmax_plasma = max(c_plasma_list)
    auc_plasma = _trapz(times, c_plasma_list)

    cmax_cortical = max(c_cort_list)
    auc_cortical = _trapz(times, c_cort_list)

    cmax_trabecular = max(c_trab_list)
    auc_trabecular = _trapz(times, c_trab_list)

    bone_plasma_kp_cortical = auc_cortical / auc_plasma if auc_plasma > 0.0 else 0.0
    bone_plasma_kp_trabecular = auc_trabecular / auc_plasma if auc_plasma > 0.0 else 0.0

    raw_seq = (a_cort_final + a_trab_final) / dose_mg
    bone_sequestration_fraction = min(1.0, max(0.0, raw_seq))

    # Notes
    if bone_sequestration_fraction > 0.5:
        notes = (
            "High bone sequestration — prolonged tissue half-life."
            " Drug may persist weeks-months after stopping treatment."
        )
    elif bone_sequestration_fraction > 0.2:
        notes = "Moderate bone sequestration — consider loading dose strategy"
    else:
        notes = "Limited bone sequestration"

    return BoneDistributionPKResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        times_h=times,
        c_plasma_mg_L=c_plasma_list,
        c_cortical_mg_L=c_cort_list,
        c_trabecular_mg_L=c_trab_list,
        cmax_plasma=cmax_plasma,
        auc_plasma=auc_plasma,
        cmax_cortical=cmax_cortical,
        auc_cortical=auc_cortical,
        cmax_trabecular=cmax_trabecular,
        auc_trabecular=auc_trabecular,
        bone_plasma_kp_cortical=bone_plasma_kp_cortical,
        bone_plasma_kp_trabecular=bone_plasma_kp_trabecular,
        bone_sequestration_fraction=bone_sequestration_fraction,
        notes=notes,
    )


def compare_bone_binding(
    drug_name: str,
    dose_mg: float,
    uptake_rates: list[dict],
) -> list[BoneDistributionPKResult]:
    """Compare bone binding across different uptake rate scenarios.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    dose_mg:
        Dose in milligrams.
    uptake_rates:
        List of dicts, each providing keyword arguments to override defaults
        in ``simulate_bone_distribution``.

    Returns
    -------
    List of BoneDistributionPKResult sorted by bone_plasma_kp_cortical descending.
    """
    results: list[BoneDistributionPKResult] = []
    for scenario in uptake_rates:
        result = simulate_bone_distribution(drug_name, dose_mg, **scenario)
        results.append(result)

    results.sort(key=lambda r: r.bone_plasma_kp_cortical, reverse=True)
    return results
