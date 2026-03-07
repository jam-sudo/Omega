"""Lung Tissue Partitioning (Phase 554).

Predict lung tissue-to-plasma partition coefficients for inhaled and
systemic drugs using a Rodgers-Rowland approach.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class InhaledLungPKResult:
    """PK result from inhaled lung deposition simulation."""

    times_h: list
    a_lung_mg: list
    c_plasma_mg_L: list
    cmax_plasma: float
    tmax_plasma_h: float
    auc_plasma: float
    f_absorbed: float
    notes: str


def predict_kp_lung(
    logP: float,
    pka: float,
    drug_type: str,
    fu_plasma: float,
    neutral_lipid_fraction: float = 0.003,
    phospholipid_fraction: float = 0.009,
    water_fraction: float = 0.811,
) -> dict:
    """Predict lung tissue-to-plasma partition coefficient (Kp_lung).

    Uses Rodgers-Rowland approach adapted for lung tissue composition.

    Parameters
    ----------
    logP:
        Octanol-water partition coefficient (log units).
    pka:
        pKa of the drug.
    drug_type:
        Drug type: 'neutral', 'acid', 'weak_base', 'strong_base', 'zwitterion'.
    fu_plasma:
        Unbound fraction in plasma (0-1).
    neutral_lipid_fraction:
        Neutral lipid volume fraction in lung tissue.
    phospholipid_fraction:
        Phospholipid volume fraction in lung tissue.
    water_fraction:
        Water volume fraction in lung tissue.

    Returns
    -------
    dict with keys: kp_lung, kp_lung_to_plasma, distribution_category
    """
    if fu_plasma <= 0 or fu_plasma > 1.0:
        raise ValueError("fu_plasma must be in (0, 1]")

    p = 10.0**logP

    # Neutral lipid partitioning
    k_neutral_lipid = p * neutral_lipid_fraction

    # Phospholipid partitioning (slightly higher logP for phospholipids)
    k_phospholipid = 10.0 ** (logP + 0.6) * phospholipid_fraction

    # Weak bases and strong bases: enhanced phospholipid binding (lysosomotropism)
    if drug_type in ("weak_base", "strong_base"):
        k_phospholipid *= 3.0

    # Water partitioning (driven by plasma protein binding)
    k_water = water_fraction / fu_plasma

    kp_lung = k_neutral_lipid + k_phospholipid + k_water

    # kp_lung_to_plasma: Kp relative to unbound plasma (same as Kp here for simplicity)
    kp_lung_to_plasma = kp_lung

    # Categorise
    if kp_lung < 1.0:
        distribution_category = "low"
    elif kp_lung <= 10.0:
        distribution_category = "moderate"
    else:
        distribution_category = "high"

    return {
        "kp_lung": kp_lung,
        "kp_lung_to_plasma": kp_lung_to_plasma,
        "distribution_category": distribution_category,
    }


def predict_kp_lung_multiple(drugs: list[dict]) -> list[dict]:
    """Predict Kp_lung for multiple drugs, ranked descending.

    Parameters
    ----------
    drugs:
        List of dicts, each with keys: name, logP, pka, drug_type, fu_plasma.
        Optional tissue fraction keys are passed through to predict_kp_lung.

    Returns
    -------
    List of dicts with input data plus kp_lung, kp_lung_to_plasma,
    distribution_category; sorted by kp_lung descending.
    """
    results = []
    for drug in drugs:
        kp_result = predict_kp_lung(
            logP=drug["logP"],
            pka=drug["pka"],
            drug_type=drug["drug_type"],
            fu_plasma=drug["fu_plasma"],
        )
        entry = dict(drug)
        entry.update(kp_result)
        results.append(entry)

    results.sort(key=lambda x: x["kp_lung"], reverse=True)
    return results


def lung_concentration_after_iv(
    dose_mg: float,
    kp_lung: float,
    vd_plasma_L: float = 50.0,
    lung_volume_L: float = 1.0,
) -> dict:
    """Estimate lung concentration at distribution equilibrium after IV dose.

    Parameters
    ----------
    dose_mg:
        IV dose in milligrams.
    kp_lung:
        Lung-to-plasma partition coefficient.
    vd_plasma_L:
        Plasma volume of distribution (L).
    lung_volume_L:
        Lung tissue volume (L).

    Returns
    -------
    dict with keys: c_plasma_mg_L, c_lung_mg_L, lung_dose_fraction
    """
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if kp_lung <= 0:
        raise ValueError("kp_lung must be > 0")
    if vd_plasma_L <= 0:
        raise ValueError("vd_plasma_L must be > 0")

    c_plasma_mg_L = dose_mg / vd_plasma_L
    c_lung_mg_L = kp_lung * c_plasma_mg_L
    lung_dose_fraction = c_lung_mg_L * lung_volume_L / dose_mg

    return {
        "c_plasma_mg_L": c_plasma_mg_L,
        "c_lung_mg_L": c_lung_mg_L,
        "lung_dose_fraction": lung_dose_fraction,
    }


def inhaled_lung_deposit_pk(
    drug_name: str,
    deposited_dose_mg: float,
    kp_lung: float,
    cl_mucus_L_per_h: float = 0.5,
    cl_absorption_L_per_h: float = 2.0,
    vd_L: float = 50.0,
    cl_systemic_L_per_h: float = 10.0,
    t_end_h: float = 24.0,
    dt_h: float = 0.1,
) -> InhaledLungPKResult:
    """Simulate inhaled drug PK with lung depot and systemic compartment.

    Uses forward Euler integration of a 2-compartment model:
      - Lung amount depot (A_lung) loses drug via mucociliary clearance and absorption.
      - Systemic compartment (Cp) gains from absorption, loses via systemic clearance.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    deposited_dose_mg:
        Dose deposited in lung (mg).
    kp_lung:
        Lung-to-plasma partition coefficient (for annotation only in this ODE).
    cl_mucus_L_per_h:
        Mucociliary clearance rate (L/h, first-order approximation: rate = cl_mucus).
    cl_absorption_L_per_h:
        Absorption clearance from lung to systemic (L/h).
    vd_L:
        Systemic volume of distribution (L).
    cl_systemic_L_per_h:
        Systemic elimination clearance (L/h).
    t_end_h:
        Simulation end time (h).
    dt_h:
        Time step for forward Euler (h).

    Returns
    -------
    InhaledLungPKResult with time courses and summary PK metrics.
    """
    if deposited_dose_mg <= 0:
        raise ValueError("deposited_dose_mg must be > 0")
    if t_end_h <= 0:
        raise ValueError("t_end_h must be > 0")
    if dt_h <= 0:
        raise ValueError("dt_h must be > 0")

    # Initial conditions
    a_lung = deposited_dose_mg  # mg
    cp = 0.0  # mg/L

    times_h: list[float] = []
    a_lung_mg: list[float] = []
    c_plasma_mg_L: list[float] = []

    t = 0.0
    n_steps = int(t_end_h / dt_h) + 1

    for _ in range(n_steps):
        times_h.append(t)
        a_lung_mg.append(a_lung)
        c_plasma_mg_L.append(cp)

        # ODEs (first-order rates, no volume needed for lung — treat amount directly)
        da_lung_dt = -(cl_absorption_L_per_h + cl_mucus_L_per_h) * a_lung
        dcp_dt = cl_absorption_L_per_h * a_lung / vd_L - cl_systemic_L_per_h * cp / vd_L

        # Forward Euler
        a_lung = max(0.0, a_lung + da_lung_dt * dt_h)
        cp = max(0.0, cp + dcp_dt * dt_h)

        t += dt_h

    # Summary metrics
    cmax_plasma = max(c_plasma_mg_L)
    tmax_plasma_h = times_h[c_plasma_mg_L.index(cmax_plasma)]

    # Trapezoidal AUC for plasma
    auc_plasma = 0.0
    for i in range(1, len(times_h)):
        dt = times_h[i] - times_h[i - 1]
        auc_plasma += 0.5 * (c_plasma_mg_L[i] + c_plasma_mg_L[i - 1]) * dt

    # Fraction absorbed = absorbed / total deposited
    # Absorbed amount = deposited - remaining - mucus-cleared
    # mucus-cleared is hard to track separately; use:
    # f_absorbed = cl_absorption / (cl_absorption + cl_mucus)
    total_elimination = cl_absorption_L_per_h + cl_mucus_L_per_h
    f_absorbed = cl_absorption_L_per_h / total_elimination if total_elimination > 0 else 0.0

    notes = (
        f"Drug: {drug_name}; Kp_lung={kp_lung:.2f}; "
        f"deposited={deposited_dose_mg}mg; f_absorbed={f_absorbed:.3f}"
    )

    return InhaledLungPKResult(
        times_h=times_h,
        a_lung_mg=a_lung_mg,
        c_plasma_mg_L=c_plasma_mg_L,
        cmax_plasma=cmax_plasma,
        tmax_plasma_h=tmax_plasma_h,
        auc_plasma=auc_plasma,
        f_absorbed=f_absorbed,
        notes=notes,
    )
