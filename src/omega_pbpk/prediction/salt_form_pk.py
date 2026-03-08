"""
Phase 954 — Salt Form PK Prediction
Predict PK impact of different drug salt forms — solubility, dissolution rate,
and absorption differences.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = ["SaltFormPKResult", "predict_salt_form_pk", "compare_salt_forms"]

SALT_FORMS = {
    "free_base": {"solubility_factor": 1.0, "dissolution_factor": 1.0},
    "HCl_salt": {"solubility_factor": 20.0, "dissolution_factor": 10.0},
    "mesylate": {"solubility_factor": 15.0, "dissolution_factor": 8.0},
    "tosylate": {"solubility_factor": 12.0, "dissolution_factor": 6.0},
    "free_acid": {"solubility_factor": 1.0, "dissolution_factor": 1.0},
    "Na_salt": {"solubility_factor": 50.0, "dissolution_factor": 25.0},
    "K_salt": {"solubility_factor": 40.0, "dissolution_factor": 20.0},
    "Ca_salt": {"solubility_factor": 5.0, "dissolution_factor": 3.0},
    "phosphate_salt": {"solubility_factor": 30.0, "dissolution_factor": 15.0},
    "maleate": {"solubility_factor": 10.0, "dissolution_factor": 5.0},
}


@dataclass(frozen=True)
class SaltFormPKResult:
    drug_name: str
    salt_form: str
    logp: float
    dose_mg: float
    s0_free_form_mg_mL: float
    s_salt_mg_mL: float
    solubility_factor: float
    fa_base: float
    fa_predicted: float
    auc_predicted_mg_h_per_L: float
    bioavailability_pct: float
    dissolution_advantage: str
    notes: str


def _compute_s0(logp: float) -> float:
    """Intrinsic solubility of free form (Yalkowsky): S0 = 10^(-logP + 0.5) mg/mL."""
    return 10.0 ** (-logp + 0.5)


def _compute_fa_base(dose_mg: float, s0: float) -> float:
    """BCS-based absorption fraction for free form."""
    # Dose number Dn = dose / (S0 * 250 mL)
    dn = dose_mg / (s0 * 250.0)
    if dn <= 1.0:
        fa_base = 0.9
    else:
        fa_base = dn ** (-0.5)
    return min(1.0, max(0.0, fa_base))


def _compute_fa_predicted(fa_base: float, solubility_factor: float) -> float:
    """Absorption with salt effect."""
    fa = fa_base * (1.0 + 0.1 * math.log10(solubility_factor))
    return min(1.0, max(0.0, fa))


def _dissolution_advantage(dissolution_factor: float) -> str:
    if dissolution_factor > 10.0:
        return "high"
    elif dissolution_factor > 3.0:
        return "moderate"
    else:
        return "low"


def predict_salt_form_pk(
    drug_name: str,
    logp: float,
    dose_mg: float,
    salt_form: str,
    cl_L_per_h: float = 10.0,
    q_liver_L_per_h: float = 90.0,
) -> SaltFormPKResult:
    """
    Predict PK parameters for a specific salt form.
    """
    # Validation
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if cl_L_per_h <= 0:
        raise ValueError("cl_L_per_h must be > 0")
    if q_liver_L_per_h <= 0:
        raise ValueError("q_liver_L_per_h must be > 0")
    if salt_form not in SALT_FORMS:
        raise ValueError(f"salt_form '{salt_form}' not in library")

    form_data = SALT_FORMS[salt_form]
    solubility_factor = form_data["solubility_factor"]
    dissolution_factor = form_data["dissolution_factor"]

    # Solubility
    s0 = _compute_s0(logp)
    s_salt = min(1000.0, s0 * solubility_factor)

    # Absorption
    fa_base = _compute_fa_base(dose_mg, s0)
    fa_predicted = _compute_fa_predicted(fa_base, solubility_factor)

    # Hepatic extraction ratio and first-pass
    e_h = cl_L_per_h / (cl_L_per_h + q_liver_L_per_h)
    f_hepatic = 1.0 - e_h

    # Bioavailability
    bioavailability = fa_predicted * f_hepatic
    bioavailability_pct = bioavailability * 100.0

    # Single-dose AUC = dose * F / CL
    auc = dose_mg * bioavailability / cl_L_per_h

    # Dissolution advantage
    dis_adv = _dissolution_advantage(dissolution_factor)

    notes = (
        f"Salt form '{salt_form}': solubility {s_salt:.2f} mg/mL "
        f"({solubility_factor:.0f}x free form S0={s0:.4f} mg/mL). "
        f"fa={fa_predicted:.3f}, F={bioavailability_pct:.1f}%, "
        f"AUC={auc:.3f} mg*h/L. Dissolution advantage: {dis_adv}."
    )

    return SaltFormPKResult(
        drug_name=drug_name,
        salt_form=salt_form,
        logp=logp,
        dose_mg=dose_mg,
        s0_free_form_mg_mL=s0,
        s_salt_mg_mL=s_salt,
        solubility_factor=solubility_factor,
        fa_base=fa_base,
        fa_predicted=fa_predicted,
        auc_predicted_mg_h_per_L=auc,
        bioavailability_pct=bioavailability_pct,
        dissolution_advantage=dis_adv,
        notes=notes,
    )


def compare_salt_forms(
    drug_name: str,
    logp: float,
    dose_mg: float,
    cl_L_per_h: float = 10.0,
    q_liver_L_per_h: float = 90.0,
) -> list[SaltFormPKResult]:
    """
    Compare all 10 salt forms, sorted by auc_predicted_mg_h_per_L descending.
    """
    results = []
    for salt_form in SALT_FORMS:
        result = predict_salt_form_pk(
            drug_name=drug_name,
            logp=logp,
            dose_mg=dose_mg,
            salt_form=salt_form,
            cl_L_per_h=cl_L_per_h,
            q_liver_L_per_h=q_liver_L_per_h,
        )
        results.append(result)

    results.sort(key=lambda r: r.auc_predicted_mg_h_per_L, reverse=True)
    return results
