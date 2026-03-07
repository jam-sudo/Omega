"""Phase 791 — PK/PD Target Calculator.

Reverse calculation: given a desired pharmacodynamic effect,
compute the required exposure and dose.

Reference: Meibohm B, Derendorf H (1997). Basic concepts of PK/PD modelling.
Drug Metab Pharmacokinet.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PKPDTargetResult:
    """Result of PK/PD target calculation."""

    drug_name: str
    pd_model: str  # "emax", "sigmoid_emax"
    target_effect_pct: float
    ec50: float
    hill_coef: float
    emax_pct: float
    required_exposure: float  # concentration needed for target effect
    required_dose_mg: float  # back-calculated dose
    therapeutic_index: float  # EC90 / EC10 (width of therapeutic window)
    pd_margin: float  # (EC_toxic - required) / required, if mtc given
    target_met_at_trough: bool  # if trough_exposure given
    trough_exposure: float
    notes: str


def _sigmoid_emax_ec(
    effect_pct: float,
    ec50: float,
    emax_pct: float,
    hill_coef: float,
) -> float:
    """Reverse sigmoid Emax: compute EC for a given effect percentage.

    E = Emax * C^hill / (EC50^hill + C^hill)
    Solving for C:
    C = EC50 * (E / (Emax - E))^(1/hill)
    """
    if effect_pct <= 0.0:
        return 0.0
    if effect_pct >= emax_pct:
        raise ValueError(f"target_effect_pct ({effect_pct}) must be < emax_pct ({emax_pct})")
    ratio = effect_pct / (emax_pct - effect_pct)
    return ec50 * (ratio ** (1.0 / hill_coef))


def calculate_pk_pd_target(
    drug_name: str,
    target_effect_pct: float,
    ec50: float,
    emax_pct: float = 100.0,
    hill_coef: float = 1.0,
    pd_model: str = "sigmoid_emax",
    cl_L_per_h: float = 5.0,
    vd_L: float = 50.0,
    f_oral: float = 0.8,
    tau_h: float = 24.0,
    mtc_effect_pct: float = 90.0,
    trough_exposure: float = 0.0,
) -> PKPDTargetResult:
    """Compute the exposure and dose required to achieve a target PD effect.

    Args:
        drug_name: Name of the drug.
        target_effect_pct: Desired pharmacodynamic effect (% of maximum).
        ec50: Concentration producing 50% of Emax (same units as exposure).
        emax_pct: Maximum achievable effect as a percentage (default 100).
        hill_coef: Hill coefficient / sigmoidicity factor (default 1).
        pd_model: PD model type: "emax" or "sigmoid_emax" (default "sigmoid_emax").
        cl_L_per_h: Clearance in L/h (default 5).
        vd_L: Volume of distribution in L (default 50).
        f_oral: Oral bioavailability fraction (default 0.8).
        tau_h: Dosing interval in hours (default 24).
        mtc_effect_pct: Minimum toxic concentration effect threshold % (default 90).
        trough_exposure: Measured or predicted trough concentration (default 0).

    Returns:
        PKPDTargetResult dataclass.

    Raises:
        ValueError: If target_effect_pct >= emax_pct, ec50 <= 0, or hill_coef <= 0.
    """
    # Validation
    if ec50 <= 0.0:
        raise ValueError(f"ec50 must be > 0, got {ec50}")
    if hill_coef <= 0.0:
        raise ValueError(f"hill_coef must be > 0, got {hill_coef}")
    if target_effect_pct >= emax_pct:
        raise ValueError(f"target_effect_pct ({target_effect_pct}) must be < emax_pct ({emax_pct})")
    if target_effect_pct < 0.0:
        raise ValueError(f"target_effect_pct must be >= 0, got {target_effect_pct}")

    # Both "emax" and "sigmoid_emax" use the same Emax equation;
    # "emax" is the linear (hill=1) special case.
    effective_hill = 1.0 if pd_model == "emax" else hill_coef

    # Required exposure for target effect
    required_exposure = _sigmoid_emax_ec(target_effect_pct, ec50, emax_pct, effective_hill)

    # Back-calculate dose: at steady state Cmin ~ Css_avg for simplicity
    # Using Css_avg = F * Dose / (CL * tau) → Dose = Cmin * CL * tau / F
    required_dose_mg = required_exposure * cl_L_per_h * tau_h / f_oral

    # EC10 and EC90 for therapeutic index
    ec10 = _sigmoid_emax_ec(10.0, ec50, emax_pct, effective_hill)
    ec90 = _sigmoid_emax_ec(90.0, ec50, emax_pct, effective_hill)
    if ec10 > 0.0:
        therapeutic_index = ec90 / ec10
    else:
        therapeutic_index = float("inf")

    # PD margin: distance from required_exposure to toxic threshold
    if mtc_effect_pct < emax_pct:
        ec_toxic = _sigmoid_emax_ec(mtc_effect_pct, ec50, emax_pct, effective_hill)
        if required_exposure > 0.0:
            pd_margin = (ec_toxic - required_exposure) / required_exposure
        else:
            pd_margin = float("inf")
    else:
        pd_margin = float("inf")

    # Trough check
    target_met_at_trough = trough_exposure >= required_exposure

    # Notes
    notes_parts = [
        f"model={pd_model}",
        f"EC_target={required_exposure:.4f}",
        f"dose={required_dose_mg:.2f} mg",
        f"TI(EC90/EC10)={therapeutic_index:.2f}",
        f"pd_margin={pd_margin:.2f}",
    ]
    if trough_exposure > 0.0:
        notes_parts.append(
            f"trough={trough_exposure:.4f} ({'ok' if target_met_at_trough else 'subtherapeutic'})"
        )
    notes = "; ".join(notes_parts)

    return PKPDTargetResult(
        drug_name=drug_name,
        pd_model=pd_model,
        target_effect_pct=target_effect_pct,
        ec50=ec50,
        hill_coef=hill_coef,
        emax_pct=emax_pct,
        required_exposure=required_exposure,
        required_dose_mg=required_dose_mg,
        therapeutic_index=therapeutic_index,
        pd_margin=pd_margin,
        target_met_at_trough=target_met_at_trough,
        trough_exposure=trough_exposure,
        notes=notes,
    )


def scan_pd_targets(
    drug_name: str,
    target_range: list[float],
    ec50: float,
    emax_pct: float = 100.0,
    hill_coef: float = 1.0,
    pd_model: str = "sigmoid_emax",
    cl_L_per_h: float = 5.0,
    vd_L: float = 50.0,
    f_oral: float = 0.8,
    tau_h: float = 24.0,
    mtc_effect_pct: float = 90.0,
    trough_exposure: float = 0.0,
) -> list[PKPDTargetResult]:
    """Scan a range of PD target effects and return results for each.

    Args:
        drug_name: Name of the drug.
        target_range: List of target effect percentages to evaluate.
        ec50: EC50 concentration.
        emax_pct: Maximum achievable effect percentage.
        hill_coef: Hill coefficient.
        pd_model: PD model type.
        cl_L_per_h: Clearance in L/h.
        vd_L: Volume of distribution in L.
        f_oral: Oral bioavailability fraction.
        tau_h: Dosing interval in hours.
        mtc_effect_pct: Toxic threshold effect percentage.
        trough_exposure: Trough concentration for threshold check.

    Returns:
        List of PKPDTargetResult, one per target_range value.
    """
    results: list[PKPDTargetResult] = []
    for target_pct in target_range:
        result = calculate_pk_pd_target(
            drug_name=drug_name,
            target_effect_pct=target_pct,
            ec50=ec50,
            emax_pct=emax_pct,
            hill_coef=hill_coef,
            pd_model=pd_model,
            cl_L_per_h=cl_L_per_h,
            vd_L=vd_L,
            f_oral=f_oral,
            tau_h=tau_h,
            mtc_effect_pct=mtc_effect_pct,
            trough_exposure=trough_exposure,
        )
        results.append(result)
    return results
