"""Phase 459 — Dose-Dependent Bioavailability.

Models non-linear oral bioavailability due to saturable absorption
or saturable first-pass hepatic metabolism.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DoseBioavailabilityResult:
    """Result of dose-dependent bioavailability prediction."""

    drug_name: str
    dose_mg: float
    fa: float  # fraction absorbed
    f_hepatic: float  # hepatic first-pass availability
    f_oral: float  # combined oral bioavailability
    mechanism: str  # "linear", "saturable_absorption", "saturable_firstpass", "both"
    dose_proportionality_index: float  # AUC ratio / dose ratio
    notes: str


def _compute_fa(
    dose_mg: float,
    fa_max: float,
    Ka_mg: float,
    saturable: bool,
) -> float:
    """Compute fraction absorbed."""
    if saturable:
        return fa_max * dose_mg / (Ka_mg + dose_mg)
    return min(fa_max, 1.0)


def _compute_f_hepatic(
    dose_mg: float,
    fa: float,
    fu_plasma: float,
    cl_int_L_per_h: float,
    Q_h_L_per_h: float,
    saturable: bool,
) -> float:
    """Compute hepatic first-pass availability (F_h).

    For saturable first-pass, use Michaelis-Menten approximation where
    portal concentration is estimated from absorbed dose assuming rapid
    distribution into portal flow (typical portal blood flow ~ 1.5 L/min
    but we keep units consistent with the inputs).

    Portal concentration proxy: C_portal = fa * dose_mg / (Q_h_L_per_h * 1.0)
    Then: E_h = fu * Vmax / (Km + C_portal) approximated via:
        CLint_eff = cl_int_L_per_h / (1 + C_portal / Km)
    where Km is set to a reference concentration (10 mg/L here).
    """
    if saturable:
        # Portal concentration (mg/L) — rough estimate
        C_portal = fa * dose_mg / max(Q_h_L_per_h, 1e-9)
        Km_ref = 10.0  # reference saturation concentration (mg/L)
        cl_int_eff = cl_int_L_per_h / (1.0 + C_portal / Km_ref)
        # Well-stirred liver model: F_h = Q_h / (Q_h + fu * CLint_eff)
        f_h = Q_h_L_per_h / (Q_h_L_per_h + fu_plasma * cl_int_eff)
    else:
        # Linear: fixed extraction ratio
        f_h = Q_h_L_per_h / (Q_h_L_per_h + fu_plasma * cl_int_L_per_h)
    return max(0.0, min(1.0, f_h))


def _determine_mechanism(
    saturable_absorption: bool,
    saturable_firstpass: bool,
) -> str:
    if saturable_absorption and saturable_firstpass:
        return "both"
    if saturable_absorption:
        return "saturable_absorption"
    if saturable_firstpass:
        return "saturable_firstpass"
    return "linear"


def _compute_dose_proportionality_index(
    dose_mg: float,
    fa_max: float,
    Ka_mg: float,
    fu_plasma: float,
    cl_int_L_per_h: float,
    Q_h_L_per_h: float,
    saturable_absorption: bool,
    saturable_firstpass: bool,
) -> float:
    """Compute DPI = (AUC2 / AUC1) / (dose2 / dose1).

    We use AUC proportional to F * dose (1-cpt IV-equivalent).
    AUC ∝ F_oral * dose / CL_sys, but since CL_sys is constant,
    DPI = (F2 * dose2) / (F1 * dose1) / (dose2 / dose1) = F2 / F1.
    """
    dose1 = dose_mg
    dose2 = dose_mg * 2.0

    fa1 = _compute_fa(dose1, fa_max, Ka_mg, saturable_absorption)
    f_h1 = _compute_f_hepatic(
        dose1, fa1, fu_plasma, cl_int_L_per_h, Q_h_L_per_h, saturable_firstpass
    )
    f1 = fa1 * f_h1

    fa2 = _compute_fa(dose2, fa_max, Ka_mg, saturable_absorption)
    f_h2 = _compute_f_hepatic(
        dose2, fa2, fu_plasma, cl_int_L_per_h, Q_h_L_per_h, saturable_firstpass
    )
    f2 = fa2 * f_h2

    auc1 = f1 * dose1
    auc2 = f2 * dose2
    dose_ratio = dose2 / dose1  # always 2.0

    if auc1 <= 0.0:
        return 1.0
    dpi = (auc2 / auc1) / dose_ratio
    return dpi


def predict_dose_bioavailability(
    drug_name: str,
    dose_mg: float,
    fa_max: float = 0.95,
    Ka_mg: float = 50.0,
    fu_plasma: float = 0.1,
    cl_int_L_per_h: float = 20.0,
    Q_h_L_per_h: float = 90.0,
    saturable_absorption: bool = True,
    saturable_firstpass: bool = False,
) -> DoseBioavailabilityResult:
    """Predict dose-dependent oral bioavailability.

    Parameters
    ----------
    drug_name:
        Identifier for the compound.
    dose_mg:
        Oral dose in milligrams (> 0).
    fa_max:
        Maximum fraction absorbed (0, 1].
    Ka_mg:
        Michaelis constant for absorption saturation (mg).
    fu_plasma:
        Unbound fraction in plasma (0, 1].
    cl_int_L_per_h:
        Intrinsic hepatic clearance (L/h).
    Q_h_L_per_h:
        Hepatic blood flow (L/h).
    saturable_absorption:
        If True, apply Michaelis-Menten absorption saturation.
    saturable_firstpass:
        If True, apply saturable first-pass hepatic metabolism.

    Returns
    -------
    DoseBioavailabilityResult
    """
    # Validation
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if not (0 < fa_max <= 1.0):
        raise ValueError("fa_max must be in (0, 1]")
    if Ka_mg <= 0:
        raise ValueError("Ka_mg must be > 0")
    if not (0 < fu_plasma <= 1.0):
        raise ValueError("fu_plasma must be in (0, 1]")
    if cl_int_L_per_h <= 0:
        raise ValueError("cl_int_L_per_h must be > 0")
    if Q_h_L_per_h <= 0:
        raise ValueError("Q_h_L_per_h must be > 0")

    fa = _compute_fa(dose_mg, fa_max, Ka_mg, saturable_absorption)
    f_h = _compute_f_hepatic(
        dose_mg, fa, fu_plasma, cl_int_L_per_h, Q_h_L_per_h, saturable_firstpass
    )
    f_oral = fa * f_h

    mechanism = _determine_mechanism(saturable_absorption, saturable_firstpass)

    dpi = _compute_dose_proportionality_index(
        dose_mg,
        fa_max,
        Ka_mg,
        fu_plasma,
        cl_int_L_per_h,
        Q_h_L_per_h,
        saturable_absorption,
        saturable_firstpass,
    )

    notes_parts = []
    if saturable_absorption:
        saturation_pct = dose_mg / (Ka_mg + dose_mg) * 100.0
        notes_parts.append(f"Absorption {saturation_pct:.1f}% of saturation")
    if saturable_firstpass:
        notes_parts.append("Saturable first-pass applied (well-stirred model)")
    if dpi < 0.9:
        notes_parts.append("Less-than-dose-proportional (DPI < 0.9)")
    elif dpi > 1.1:
        notes_parts.append("More-than-dose-proportional (DPI > 1.1)")
    else:
        notes_parts.append("Approximately dose-proportional")
    notes = "; ".join(notes_parts) if notes_parts else "Linear bioavailability"

    return DoseBioavailabilityResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        fa=fa,
        f_hepatic=f_h,
        f_oral=f_oral,
        mechanism=mechanism,
        dose_proportionality_index=dpi,
        notes=notes,
    )


def screen_doses(
    drug_name: str,
    doses_mg: list,
    **kwargs,
) -> list:
    """Screen multiple doses and return results sorted by dose ascending.

    Parameters
    ----------
    drug_name:
        Compound identifier.
    doses_mg:
        List of doses (mg) to evaluate.
    **kwargs:
        Forwarded to predict_dose_bioavailability.

    Returns
    -------
    list[DoseBioavailabilityResult] sorted by dose_mg ascending.
    """
    results = [predict_dose_bioavailability(drug_name, d, **kwargs) for d in doses_mg]
    return sorted(results, key=lambda r: r.dose_mg)


__all__ = [
    "DoseBioavailabilityResult",
    "predict_dose_bioavailability",
    "screen_doses",
]
