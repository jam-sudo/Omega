"""COPD inhalation PK — lung deposition and plasma PK for inhaled drugs in COPD vs normal."""

from __future__ import annotations

from dataclasses import dataclass

# Fine-particle fraction (FPF) defaults by device type (fraction reaching lower airways)
_DEVICE_K_ABS: dict[str, float] = {
    "MDI": 3.0,  # /h
    "DPI": 2.0,  # /h
    "nebulizer": 1.5,  # /h
}

# GOLD stage deposition factors (fraction of normal deposition)
_GOLD_FACTORS: dict[str, float] = {
    "I": 0.85,
    "II": 0.70,
    "III": 0.55,
    "IV": 0.40,
}

_VALID_DEVICES = frozenset(_DEVICE_K_ABS.keys())


@dataclass
class COPDInhalationResult:
    """Result of COPD inhalation PK simulation."""

    drug_name: str
    dose_mg: float
    device_type: str
    copd_factor: float
    times_h: list[float]
    c_plasma_normal: list[float]
    c_plasma_copd: list[float]
    cmax_normal: float
    cmax_copd: float
    auc_normal: float
    auc_copd: float
    lung_dose_normal_mg: float
    lung_dose_copd_mg: float
    auc_ratio_copd_normal: float
    notes: str


def _trapezoidal_auc(times: list[float], concs: list[float]) -> float:
    """Compute AUC using the trapezoidal rule."""
    auc = 0.0
    for i in range(1, len(times)):
        dt = times[i] - times[i - 1]
        auc += 0.5 * (concs[i] + concs[i - 1]) * dt
    return auc


def _simulate_lung_plasma(
    lung_dose_mg: float,
    k_abs: float,
    cl_L_per_h: float,
    vd_L: float,
    t_end_h: float,
    dt_h: float,
) -> tuple[list[float], list[float]]:
    """Forward Euler simulation: lung depot → plasma → elimination.

    State vector: [A_lung (mg), C_plasma (mg/L)]
    dA_lung/dt = -k_abs * A_lung
    dC_plasma/dt = k_abs * A_lung / Vd - (CL/Vd) * C_plasma
    """
    n_steps = max(2, int(round(t_end_h / dt_h)) + 1)
    times: list[float] = []
    c_plasma: list[float] = []

    a_lung = lung_dose_mg
    c = 0.0
    ke = cl_L_per_h / vd_L  # elimination rate constant /h

    for i in range(n_steps):
        t = i * dt_h
        times.append(t)
        c_plasma.append(c)

        # Forward Euler update
        da_lung = -k_abs * a_lung * dt_h
        dc = (k_abs * a_lung / vd_L - ke * c) * dt_h

        a_lung = max(0.0, a_lung + da_lung)
        c = max(0.0, c + dc)

    return times, c_plasma


def simulate_copd_inhalation_pk(
    drug_name: str,
    dose_mg: float,
    device_type: str,
    cl_L_per_h: float,
    vd_L: float,
    fpf_normal: float,
    t_end_h: float = 24.0,
    dt_h: float = 0.05,
    copd_factor: float = 0.55,
) -> COPDInhalationResult:
    """Simulate inhaled drug PK in normal vs COPD subjects.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    dose_mg:
        Nominal (metered) dose in mg.
    device_type:
        Inhaler type: "MDI", "DPI", or "nebulizer".
    cl_L_per_h:
        Systemic clearance (L/h).
    vd_L:
        Volume of distribution (L).
    fpf_normal:
        Fine-particle fraction in a normal subject (0–1). Determines the
        fraction of the dose that reaches the lower airways.
    t_end_h:
        Simulation duration (h).
    dt_h:
        Integration time step (h).
    copd_factor:
        Fraction of normal lung deposition achieved in COPD (default 0.55,
        representing ~45% reduction in lung deposition).

    Returns
    -------
    COPDInhalationResult
    """
    # --- Input validation ---
    if not drug_name or not drug_name.strip():
        raise ValueError("drug_name must be a non-empty string.")
    if dose_mg <= 0:
        raise ValueError(f"dose_mg must be positive; got {dose_mg}.")
    if device_type not in _VALID_DEVICES:
        raise ValueError(
            f"device_type must be one of {sorted(_VALID_DEVICES)}; got '{device_type}'."
        )
    if cl_L_per_h <= 0:
        raise ValueError(f"cl_L_per_h must be positive; got {cl_L_per_h}.")
    if vd_L <= 0:
        raise ValueError(f"vd_L must be positive; got {vd_L}.")
    if not (0.0 < fpf_normal <= 1.0):
        raise ValueError(f"fpf_normal must be in (0, 1]; got {fpf_normal}.")
    if t_end_h <= 0:
        raise ValueError(f"t_end_h must be positive; got {t_end_h}.")
    if dt_h <= 0 or dt_h >= t_end_h:
        raise ValueError(
            f"dt_h must be positive and less than t_end_h; got dt_h={dt_h}, t_end_h={t_end_h}."
        )
    if not (0.0 < copd_factor <= 1.0):
        raise ValueError(f"copd_factor must be in (0, 1]; got {copd_factor}.")

    k_abs = _DEVICE_K_ABS[device_type]

    # Lung doses
    lung_dose_normal = dose_mg * fpf_normal
    lung_dose_copd = lung_dose_normal * copd_factor

    # Simulate both profiles
    times, c_normal = _simulate_lung_plasma(
        lung_dose_normal, k_abs, cl_L_per_h, vd_L, t_end_h, dt_h
    )
    _, c_copd = _simulate_lung_plasma(lung_dose_copd, k_abs, cl_L_per_h, vd_L, t_end_h, dt_h)

    cmax_normal = max(c_normal) if c_normal else 0.0
    cmax_copd = max(c_copd) if c_copd else 0.0
    auc_normal = _trapezoidal_auc(times, c_normal)
    auc_copd = _trapezoidal_auc(times, c_copd)
    auc_ratio = auc_copd / auc_normal if auc_normal > 0 else 0.0

    notes = (
        f"COPD factor {copd_factor:.2f} applied to normal lung dose "
        f"({lung_dose_normal:.3f} mg → {lung_dose_copd:.3f} mg). "
        f"k_abs={k_abs}/h ({device_type})."
    )

    return COPDInhalationResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        device_type=device_type,
        copd_factor=copd_factor,
        times_h=times,
        c_plasma_normal=c_normal,
        c_plasma_copd=c_copd,
        cmax_normal=cmax_normal,
        cmax_copd=cmax_copd,
        auc_normal=auc_normal,
        auc_copd=auc_copd,
        lung_dose_normal_mg=lung_dose_normal,
        lung_dose_copd_mg=lung_dose_copd,
        auc_ratio_copd_normal=auc_ratio,
        notes=notes,
    )


def compare_copd_severity(
    drug_name: str,
    dose_mg: float,
    device_type: str,
    cl_L_per_h: float,
    vd_L: float,
    fpf_normal: float,
    t_end_h: float = 24.0,
    dt_h: float = 0.05,
) -> dict:
    """Compare PK across GOLD COPD severity stages I–IV.

    Parameters
    ----------
    drug_name, dose_mg, device_type, cl_L_per_h, vd_L, fpf_normal:
        Passed through to :func:`simulate_copd_inhalation_pk`.
    t_end_h:
        Simulation duration (h).
    dt_h:
        Integration step (h).

    Returns
    -------
    dict with keys:
        - "stages": list of GOLD stage labels (["I", "II", "III", "IV"])
        - "results": list of :class:`COPDInhalationResult` (one per stage)
        - "auc_by_stage": dict mapping stage label → AUC (mg·h/L)
        - "cmax_by_stage": dict mapping stage label → Cmax (mg/L)
        - "auc_ratios": dict mapping stage label → AUC ratio vs normal
    """
    # Validate shared parameters early (simulate_copd_inhalation_pk will also validate)
    if not drug_name or not drug_name.strip():
        raise ValueError("drug_name must be a non-empty string.")
    if dose_mg <= 0:
        raise ValueError(f"dose_mg must be positive; got {dose_mg}.")
    if device_type not in _VALID_DEVICES:
        raise ValueError(
            f"device_type must be one of {sorted(_VALID_DEVICES)}; got '{device_type}'."
        )
    if cl_L_per_h <= 0:
        raise ValueError(f"cl_L_per_h must be positive; got {cl_L_per_h}.")
    if vd_L <= 0:
        raise ValueError(f"vd_L must be positive; got {vd_L}.")
    if not (0.0 < fpf_normal <= 1.0):
        raise ValueError(f"fpf_normal must be in (0, 1]; got {fpf_normal}.")

    stages = list(_GOLD_FACTORS.keys())
    results: list[COPDInhalationResult] = []
    auc_by_stage: dict[str, float] = {}
    cmax_by_stage: dict[str, float] = {}
    auc_ratios: dict[str, float] = {}

    for stage in stages:
        factor = _GOLD_FACTORS[stage]
        res = simulate_copd_inhalation_pk(
            drug_name=drug_name,
            dose_mg=dose_mg,
            device_type=device_type,
            cl_L_per_h=cl_L_per_h,
            vd_L=vd_L,
            fpf_normal=fpf_normal,
            t_end_h=t_end_h,
            dt_h=dt_h,
            copd_factor=factor,
        )
        results.append(res)
        auc_by_stage[stage] = res.auc_copd
        cmax_by_stage[stage] = res.cmax_copd
        auc_ratios[stage] = res.auc_ratio_copd_normal

    return {
        "stages": stages,
        "results": results,
        "auc_by_stage": auc_by_stage,
        "cmax_by_stage": cmax_by_stage,
        "auc_ratios": auc_ratios,
    }


__all__ = [
    "COPDInhalationResult",
    "simulate_copd_inhalation_pk",
    "compare_copd_severity",
]
