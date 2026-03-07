"""Phase 1008 — Nasal Drug Absorption Prediction.

Predicts nasal bioavailability for intranasal drug delivery, accounting
for nasal membrane permeability, mucociliary clearance, and physico-
chemical properties.  Nasal route bypasses hepatic first-pass metabolism.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class NasalAbsorptionResult:
    """Predicted nasal absorption parameters for a drug."""

    drug_name: str
    logp: float
    mw: float
    psa: float
    dose_mg: float
    f_nasal: float
    tmax_nasal_min: float
    ka_nasal_per_min: float
    mucociliary_clearance_fraction: float
    nasal_vs_oral_f_ratio: float
    recommended_formulation: str
    notes: str


def predict_nasal_absorption(
    drug_name: str,
    logp: float,
    mw: float,
    psa: float = 80.0,
    dose_mg: float = 1.0,
    oral_f: float = 0.3,
) -> NasalAbsorptionResult:
    """Predict nasal absorption and bioavailability.

    Parameters
    ----------
    drug_name:
        Name of the drug compound.
    logp:
        Octanol-water log partition coefficient.
    mw:
        Molecular weight (g/mol).  Must be > 0.
    psa:
        Polar surface area (Å²).  Must be >= 0.
    dose_mg:
        Intranasal dose in mg.  Must be > 0.
    oral_f:
        Oral bioavailability (0–1) for comparison.  Must be in [0, 1].

    Returns
    -------
    NasalAbsorptionResult
    """
    # --- Validation ---
    if mw <= 0:
        raise ValueError(f"mw must be > 0, got {mw}")
    if psa < 0:
        raise ValueError(f"psa must be >= 0, got {psa}")
    if dose_mg <= 0:
        raise ValueError(f"dose_mg must be > 0, got {dose_mg}")
    if not (0.0 <= oral_f <= 1.0):
        raise ValueError(f"oral_f must be in [0, 1], got {oral_f}")

    # --- Nasal membrane permeability ---
    # log_pnasal in log(cm/s)
    log_pnasal = 0.6 * logp - 0.005 * psa - 0.003 * mw - 1.5
    p_nasal_cm_s = max(1e-6, 10.0**log_pnasal)

    # Surface-area / volume factor for nasal cavity (1/cm)
    sa_vol = 600.0  # cm-1

    # --- Mucociliary clearance ---
    k_mcc = 0.05  # /min
    # Effective absorption rate (1/min) competing with MCC
    k_abs_eff = (
        p_nasal_cm_s * sa_vol
    )  # per min (units: cm/s * cm-1 * 60 s/min ~60, but keep consistent)
    f_mcc_raw = k_mcc / (k_mcc + k_abs_eff)
    f_mcc = min(0.8, max(0.05, f_mcc_raw))

    # --- Nasal bioavailability ---
    # PSA penalty: high PSA reduces paracellular/transcellular nasal absorption.
    # Scaled linearly beyond PSA=90 Å²: ~0.3% reduction per Å².
    psa_penalty = max(0.0, (psa - 90.0) * 0.003)
    large_mw_penalty = 0.1 if mw > 500 else 0.0
    f_nasal = max(0.05, 1.0 - f_mcc - large_mw_penalty - psa_penalty)
    f_nasal = min(0.95, f_nasal)

    # --- Absorption rate constant (/min) ---
    ka_nasal_per_min = min(0.5, max(0.01, k_abs_eff))

    # --- Tmax (min) ---
    tmax_nasal_min = min(60.0, 0.693 / ka_nasal_per_min * 2.0)

    # --- Nasal vs oral ratio ---
    nasal_vs_oral_f_ratio = f_nasal / max(0.01, oral_f)

    # --- Recommended formulation ---
    if mw > 500:
        recommended_formulation = "powder"
    elif logp < 0.0:
        recommended_formulation = "gel"
    elif mw < 200 and logp > 1.0:
        recommended_formulation = "spray"
    else:
        recommended_formulation = "solution"

    notes_parts = [
        f"p_nasal={p_nasal_cm_s:.2e} cm/s",
        f"f_mcc={f_mcc:.3f}",
        f"ka={ka_nasal_per_min:.4f}/min",
        f"formulation={recommended_formulation}",
    ]
    if mw > 500:
        notes_parts.append("Large MW reduces nasal absorption")
    if psa > 120:
        notes_parts.append("High PSA limits nasal permeability")

    return NasalAbsorptionResult(
        drug_name=drug_name,
        logp=logp,
        mw=mw,
        psa=psa,
        dose_mg=dose_mg,
        f_nasal=f_nasal,
        tmax_nasal_min=tmax_nasal_min,
        ka_nasal_per_min=ka_nasal_per_min,
        mucociliary_clearance_fraction=f_mcc,
        nasal_vs_oral_f_ratio=nasal_vs_oral_f_ratio,
        recommended_formulation=recommended_formulation,
        notes="; ".join(notes_parts),
    )


def compare_nasal_compounds(compounds: list) -> list:
    """Compare nasal absorption across multiple compounds.

    Parameters
    ----------
    compounds:
        List of dicts, each with keys: drug_name, logp, mw.
        Optional keys: psa, dose_mg, oral_f.

    Returns
    -------
    list of NasalAbsorptionResult sorted by f_nasal descending.
    """
    results = []
    for c in compounds:
        result = predict_nasal_absorption(
            drug_name=c["drug_name"],
            logp=c["logp"],
            mw=c["mw"],
            psa=c.get("psa", 80.0),
            dose_mg=c.get("dose_mg", 1.0),
            oral_f=c.get("oral_f", 0.3),
        )
        results.append(result)
    results.sort(key=lambda r: r.f_nasal, reverse=True)
    return results
