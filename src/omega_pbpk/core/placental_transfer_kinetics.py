"""Phase 1139: Placental transfer kinetics model.

Models dynamic drug transfer across placenta during pregnancy:
maternal plasma -> placental tissue -> fetal plasma (with reverse transfer).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PlacentalTransferResult:
    """Result of placental transfer kinetics simulation."""

    drug_name: str
    dose_mg: float
    logP: float
    ionization_type: str
    gestational_age_weeks: int
    times_h: list
    c_maternal_mg_L: list
    c_placenta_mg_L: list
    c_fetal_mg_L: list
    cmax_fetal_mg_L: float
    tmax_fetal_h: float
    auc_maternal_mg_h_per_L: float
    auc_fetal_mg_h_per_L: float
    fetal_to_maternal_auc_ratio: float
    fetal_exposure_class: str
    teratogenic_concern: bool
    notes: str


def _validate_inputs(
    dose_mg: float,
    logP: float,
    ionization_type: str,
    gestational_age_weeks: int,
    cl_maternal_L_per_h: float,
    vd_maternal_L: float,
) -> None:
    if dose_mg <= 0:
        raise ValueError(f"dose_mg must be > 0, got {dose_mg}")
    if cl_maternal_L_per_h <= 0:
        raise ValueError(f"cl_maternal_L_per_h must be > 0, got {cl_maternal_L_per_h}")
    if vd_maternal_L <= 0:
        raise ValueError(f"vd_maternal_L must be > 0, got {vd_maternal_L}")
    if not (-5.0 <= logP <= 10.0):
        raise ValueError(f"logP must be in [-5, 10], got {logP}")
    if ionization_type not in {"acid", "base", "neutral"}:
        raise ValueError(
            f"ionization_type must be 'acid', 'base', or 'neutral', got {ionization_type!r}"
        )
    if not (8 <= gestational_age_weeks <= 42):
        raise ValueError(f"gestational_age_weeks must be in [8, 42], got {gestational_age_weeks}")


def _trapz_auc(times: list, concentrations: list) -> float:
    """Compute AUC using trapezoidal rule."""
    auc = 0.0
    for i in range(1, len(times)):
        dt = times[i] - times[i - 1]
        auc += 0.5 * (concentrations[i] + concentrations[i - 1]) * dt
    return auc


def _classify_fetal_exposure(ratio: float) -> str:
    if ratio < 0.1:
        return "minimal"
    elif ratio < 0.3:
        return "low"
    elif ratio < 0.6:
        return "moderate"
    else:
        return "high"


def simulate_placental_transfer(
    drug_name: str,
    dose_mg: float,
    logP: float,
    ionization_type: str = "neutral",
    gestational_age_weeks: int = 32,
    cl_maternal_L_per_h: float = 8.0,
    vd_maternal_L: float = 60.0,
    t_end_h: float = 24.0,
    dt_h: float = 0.1,
) -> PlacentalTransferResult:
    """Simulate dynamic placental drug transfer using Forward Euler ODE.

    Compartments: maternal plasma, placental tissue, fetal plasma.

    Args:
        drug_name: Name of the drug.
        dose_mg: IV bolus dose in mg.
        logP: Octanol-water partition coefficient.
        ionization_type: One of 'acid', 'base', 'neutral'.
        gestational_age_weeks: Gestational age (8-42 weeks).
        cl_maternal_L_per_h: Maternal plasma clearance (L/h).
        vd_maternal_L: Maternal volume of distribution (L).
        t_end_h: Simulation end time (h).
        dt_h: Time step for Forward Euler (h).

    Returns:
        PlacentalTransferResult with time-course and summary metrics.
    """
    _validate_inputs(
        dose_mg, logP, ionization_type, gestational_age_weeks, cl_maternal_L_per_h, vd_maternal_L
    )

    # Rate constants
    kmt = max(0.01, min(0.5, 0.1 * (1.0 + logP / 5.0)))  # maternal -> placenta
    ktm = 0.1  # placenta -> maternal (h^-1)
    kpf_base = 0.05  # placenta -> fetal (h^-1)
    kpf = kpf_base * (gestational_age_weeks / 40.0)  # gestational age scaling
    kfp = 0.02  # fetal -> placenta (h^-1)
    ke_maternal = cl_maternal_L_per_h / vd_maternal_L  # maternal elimination (h^-1)
    ke_fetal = 0.01  # fetal elimination (h^-1)

    # Initial conditions
    c_mat = dose_mg / vd_maternal_L
    c_plac = 0.0
    c_fet = 0.0

    times: list = []
    c_maternal_list: list = []
    c_placenta_list: list = []
    c_fetal_list: list = []

    t = 0.0
    n_steps = int(round(t_end_h / dt_h)) + 1

    for _ in range(n_steps):
        times.append(t)
        c_maternal_list.append(c_mat)
        c_placenta_list.append(c_plac)
        c_fetal_list.append(c_fet)

        # Forward Euler derivatives (per spec ODEs, concentrations in mg/L)
        dc_mat = -ke_maternal * c_mat - kmt * c_mat + ktm * c_plac
        dc_plac = kmt * c_mat - ktm * c_plac - kpf * c_plac + kfp * c_fet
        dc_fet = kpf * c_plac - kfp * c_fet - ke_fetal * c_fet

        c_mat = max(0.0, c_mat + dc_mat * dt_h)
        c_plac = max(0.0, c_plac + dc_plac * dt_h)
        c_fet = max(0.0, c_fet + dc_fet * dt_h)

        t = round(t + dt_h, 10)

    # Metrics
    cmax_fetal = max(c_fetal_list)
    tmax_fetal = times[c_fetal_list.index(cmax_fetal)]

    auc_maternal = _trapz_auc(times, c_maternal_list)
    auc_fetal = _trapz_auc(times, c_fetal_list)

    if auc_maternal > 0.0:
        ratio = auc_fetal / auc_maternal
    else:
        ratio = 0.0

    exposure_class = _classify_fetal_exposure(ratio)
    teratogenic_concern = ratio > 0.5

    # Build notes
    notes_parts = [
        f"kmt={kmt:.3f}/h (logP-dependent)",
        f"kpf={kpf:.4f}/h (gestational age {gestational_age_weeks}w)",
        f"Fetal/maternal AUC ratio={ratio:.3f}",
        f"Exposure class: {exposure_class}",
    ]
    if teratogenic_concern:
        notes_parts.append("Teratogenic concern flagged (ratio > 0.5)")

    return PlacentalTransferResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        logP=logP,
        ionization_type=ionization_type,
        gestational_age_weeks=gestational_age_weeks,
        times_h=times,
        c_maternal_mg_L=c_maternal_list,
        c_placenta_mg_L=c_placenta_list,
        c_fetal_mg_L=c_fetal_list,
        cmax_fetal_mg_L=cmax_fetal,
        tmax_fetal_h=tmax_fetal,
        auc_maternal_mg_h_per_L=auc_maternal,
        auc_fetal_mg_h_per_L=auc_fetal,
        fetal_to_maternal_auc_ratio=ratio,
        fetal_exposure_class=exposure_class,
        teratogenic_concern=teratogenic_concern,
        notes="; ".join(notes_parts),
    )
