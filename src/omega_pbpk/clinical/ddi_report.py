"""FDA-guidance DDI risk assessment report generator.

Implements victim drug assessment per FDA 2020 Drug Interaction Guidance.

Static inhibition:
  R1     = 1 + [I]/Ki             (plasma — conservative Cmax-based)
  R1gut  = 1 + fa·ka·Dose_mol / (Qgut·Ki)  (intestinal CYP3A4)

Mechanism-based inactivation (MBI / time-dependent):
  R2     = 1 / (1 − kinact·[I] / ((Ki_MBI + [I]) · (kinact + kdeg)))
  (per FDA 2020 Section III.A.2)

CYP induction (Emax model, Fahmi et al. 2008):
  Net induction ratio = 1 + d · IndMax · [I] / (IndC50 + [I])

AUC ratio prediction (static):
  AUCR = 1 / (fm · 1/(R1·R2·induction_R) + (1 − fm))
  where induction_R = 1/net_induction_ratio

Transporter DDI (FDA 2012/2020):
  R_pgp  = 1 + Igut / IC50_pgp          (intestinal P-gp, use dose-based [I])
  R_oatp = 1 + Cmax_inlet / IC50_oatp   (OATP1B1/1B3 hepatic uptake)

Risk thresholds:
  R1 ≥ 2.0  → reversible inhibition study recommended
  R1gut ≥ 11 → gut CYP3A4 inhibition study recommended
  R2 ≥ 1.25 → MBI (time-dependent) study recommended
  induction ≥ 2× → induction study recommended
  R_pgp ≥ 1.25 → P-gp DDI study recommended
  R_oatp ≥ 1.25 → OATP DDI study recommended
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# FDA physiological constants
# ---------------------------------------------------------------------------

# CYP degradation rates (h⁻¹) — FDA 2020 Table 2
KDEG: dict[str, float] = {
    "CYP3A4": 0.03,  # 0.0005 /min × 60
    "CYP2D6": 0.018,  # 0.0003 /min × 60
    "CYP2C9": 0.018,
    "CYP1A2": 0.024,  # 0.0004 /min × 60
    "CYP2C19": 0.018,
}

QGUT_L_PER_H = 18.0  # portal/intestinal blood flow (L/h)
Q_PORTAL_L_PER_H = 72.0  # total portal vein flow (L/h) for inlet Cmax

# OATP inlet concentration scaling factor (FDA 2012): Cin = Cmax + Fa·Dose/(Qh/BPR)
Q_LIVER_L_PER_H = 96.6  # hepatic blood flow
BPR_DEFAULT = 1.0  # blood-to-plasma ratio


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DDIInhibitor:
    """Perpetrator drug properties for comprehensive DDI risk assessment.

    Supports static/reversible inhibition, MBI, induction, and transporter DDI.
    """

    name: str
    cmax_uM: float  # Maximum plasma concentration (µM)

    # --- Reversible (competitive) CYP inhibition Ki values (µM) ---
    ki_3a4_uM: float = float("inf")
    ki_2d6_uM: float = float("inf")
    ki_2c9_uM: float = float("inf")
    ki_1a2_uM: float = float("inf")
    ki_2c19_uM: float = float("inf")

    # --- Mechanism-based (time-dependent) inactivation — CYP3A4 ---
    kinact_3a4_per_h: float = 0.0  # maximum inactivation rate (h⁻¹)
    ki_mbi_3a4_uM: float = float("inf")  # concentration at half-max inactivation (µM)

    # --- MBI for other CYPs (optional) ---
    kinact_2d6_per_h: float = 0.0
    ki_mbi_2d6_uM: float = float("inf")
    kinact_2c9_per_h: float = 0.0
    ki_mbi_2c9_uM: float = float("inf")
    kinact_1a2_per_h: float = 0.0
    ki_mbi_1a2_uM: float = float("inf")

    # --- CYP induction (Emax/EC50 model) ---
    ind_max_3a4: float = 0.0  # maximum fold induction (Emax, dimensionless)
    ind_c50_3a4_uM: float = 1.0  # EC50 for induction (µM)
    ind_max_1a2: float = 0.0
    ind_c50_1a2_uM: float = 1.0

    # --- Transporter inhibition ---
    ic50_pgp_uM: float = float("inf")  # intestinal P-gp IC50 (µM)
    ic50_oatp1b1_uM: float = float("inf")  # hepatic OATP1B1 IC50 (µM)
    ic50_oatp1b3_uM: float = float("inf")  # hepatic OATP1B3 IC50 (µM)
    ic50_oct2_uM: float = float("inf")  # renal OCT2 IC50 (µM)

    # --- Dosing parameters (for gut concentration estimate) ---
    dose_mg: float = 100.0
    mw: float = 300.0
    fa: float = 0.85  # fraction absorbed
    fg: float = 0.6  # gut availability
    ka_per_h: float = 1.0
    fup: float = 0.1  # fraction unbound in plasma (for inlet Cmax)

    # Legacy field for backwards compatibility
    induction_fold_3a4: float = 1.0


@dataclass
class DDIRiskReport:
    """Full FDA DDI risk assessment results."""

    inhibitor_name: str

    # Static R1 (≥ 2.0 → study recommended)
    R1_3a4: float = 1.0
    R1_2d6: float = 1.0
    R1_2c9: float = 1.0
    R1_1a2: float = 1.0
    R1_2c19: float = 1.0

    # Gut R1 for CYP3A4 (≥ 11 → intestinal study)
    R1gut_3a4: float = 1.0

    # MBI R2 per enzyme (≥ 1.25 → time-dependent study)
    R2_3a4: float = 1.0
    R2_2d6: float = 1.0
    R2_2c9: float = 1.0
    R2_1a2: float = 1.0

    # Induction net ratio (> 1.0 → increased CLint)
    induction_ratio_3a4: float = 1.0
    induction_ratio_1a2: float = 1.0

    # Transporter R values (≥ 1.25 → transporter DDI study)
    R_pgp: float = 1.0
    R_oatp1b1: float = 1.0
    R_oatp1b3: float = 1.0
    R_oct2: float = 1.0

    # Net AUCR prediction (victim AUC ratio) for CYP3A4 victim
    aucr_3a4: float = 1.0

    flags: list[str] = field(default_factory=list)
    recommendation: str = "No clinical DDI study warranted"

    # Legacy field
    @property
    def induction_fold_3a4(self) -> float:
        return self.induction_ratio_3a4

    @property
    def R2_3a4_value(self) -> float:
        return self.R2_3a4


def _r1(cmax: float, ki: float) -> float:
    """Static reversible inhibition ratio: R1 = 1 + [I]/Ki."""
    return 1.0 + cmax / max(ki, 1e-12)


def _r2_mbi(cmax: float, kinact: float, ki_mbi: float, kdeg: float) -> float:
    """Mechanism-based inactivation ratio per FDA 2020 Eq. III.A.2.

    R2 = 1 / (1 − kinact·[I] / ((Ki_MBI + [I]) · (kinact + kdeg)))

    Args:
        cmax: Perpetrator plasma Cmax (µM).
        kinact: Maximum inactivation rate constant (h⁻¹).
        ki_mbi: Concentration at half-max inactivation (µM).
        kdeg: Enzyme degradation rate (h⁻¹).

    Returns:
        R2 ≥ 1.0. Returns 1.0 if inputs are zero/inf.
    """
    if kinact <= 0 or ki_mbi >= float("inf") or cmax <= 0:
        return 1.0
    denom = (ki_mbi + cmax) * (kinact + kdeg)
    frac = kinact * cmax / max(denom, 1e-30)
    frac = min(frac, 0.9999)  # cap at 99.99% inactivation (numerical safety)
    return 1.0 / max(1.0 - frac, 1e-6)


def _induction_ratio(cmax: float, ind_max: float, ind_c50: float) -> float:
    """Net induction ratio using Emax model (Fahmi et al. 2008).

    Net ratio = 1 + d × IndMax × [I] / (IndC50 + [I])
    where d = 1 (calibration factor, default 1).

    Returns:
        Ratio ≥ 1.0 representing fold-increase in CYP expression/CLint.
    """
    if ind_max <= 0 or cmax <= 0:
        return 1.0
    return 1.0 + ind_max * cmax / max(ind_c50 + cmax, 1e-12)


def _aucr(fm: float, R1: float, R2: float, induction_ratio: float) -> float:
    """Predict victim AUC ratio (AUCR) from static DDI model.

    AUCR = 1 / (fm × (1 / (R1 × R2 × (1/induction_ratio))) + (1 − fm))

    where:
      fm           = fraction of victim metabolized by the affected CYP
      R1           = reversible inhibition ratio (≥ 1)
      R2           = MBI ratio (≥ 1)
      induction_ratio = net fold-increase in CLint (≥ 1; divides effective CLint)

    Note: induction reduces victim AUC (increased CLint); inhibition/MBI increase it.

    Returns:
        AUCR (> 0). Values > 1 indicate increased victim exposure.
    """
    fm = max(0.0, min(fm, 1.0))
    combined_inhibition = R1 * R2  # combined increase in victim AUC
    effective_clint_ratio = induction_ratio  # fold-increase in CLint from induction
    # Net CLint scalar on victim: inhibition increases AUC, induction decreases
    clint_scalar = effective_clint_ratio / combined_inhibition
    # AUCR = 1 / (fm × clint_scalar + (1 − fm))
    denominator = fm * clint_scalar + (1.0 - fm)
    return 1.0 / max(denominator, 1e-6)


def assess_ddi_risk(
    inhibitor: DDIInhibitor,
    victim_fm_3a4: float = 0.5,
) -> DDIRiskReport:
    """Run comprehensive FDA 2020 DDI risk assessment for a perpetrator compound.

    Covers: static competitive inhibition (R1), mechanism-based inactivation (R2),
    CYP induction (Emax model), gut R1gut, transporter DDI, and net AUCR prediction.

    Args:
        inhibitor: Perpetrator drug properties.
        victim_fm_3a4: Fraction of victim drug metabolized by CYP3A4 (for AUCR).

    Returns:
        DDIRiskReport with all R-values, flags, and AUCR prediction.
    """
    report = DDIRiskReport(inhibitor_name=inhibitor.name)
    flags: list[str] = []
    cmax = inhibitor.cmax_uM

    # ------------------------------------------------------------------
    # 1. Static reversible inhibition R1 (plasma Cmax-based, conservative)
    # ------------------------------------------------------------------
    report.R1_3a4 = _r1(cmax, inhibitor.ki_3a4_uM)
    report.R1_2d6 = _r1(cmax, inhibitor.ki_2d6_uM)
    report.R1_2c9 = _r1(cmax, inhibitor.ki_2c9_uM)
    report.R1_1a2 = _r1(cmax, inhibitor.ki_1a2_uM)
    report.R1_2c19 = _r1(cmax, inhibitor.ki_2c19_uM)

    for r1_val, enzyme in [
        (report.R1_3a4, "CYP3A4"),
        (report.R1_2d6, "CYP2D6"),
        (report.R1_2c9, "CYP2C9"),
        (report.R1_1a2, "CYP1A2"),
        (report.R1_2c19, "CYP2C19"),
    ]:
        if r1_val >= 2.0:
            flags.append(
                f"{enzyme} static inhibition R1={r1_val:.2f} ≥ 2.0 → clinical study recommended"
            )

    # ------------------------------------------------------------------
    # 2. Gut R1gut for intestinal CYP3A4
    #    R1gut = 1 + fa·ka·Dose_µmol / (Qgut·Ki)
    # ------------------------------------------------------------------
    dose_umol = (inhibitor.dose_mg / max(inhibitor.mw, 1.0)) * 1e6
    R1gut_num = inhibitor.fa * inhibitor.ka_per_h * dose_umol
    report.R1gut_3a4 = 1.0 + R1gut_num / (QGUT_L_PER_H * max(inhibitor.ki_3a4_uM, 1e-12))
    if report.R1gut_3a4 >= 11.0:
        flags.append(
            f"CYP3A4 gut R1gut={report.R1gut_3a4:.1f} ≥ 11.0 → intestinal CYP3A4 DDI study"
        )

    # ------------------------------------------------------------------
    # 3. Mechanism-based inactivation R2 (multi-enzyme)
    # ------------------------------------------------------------------
    report.R2_3a4 = _r2_mbi(
        cmax, inhibitor.kinact_3a4_per_h, inhibitor.ki_mbi_3a4_uM, KDEG["CYP3A4"]
    )
    report.R2_2d6 = _r2_mbi(
        cmax, inhibitor.kinact_2d6_per_h, inhibitor.ki_mbi_2d6_uM, KDEG["CYP2D6"]
    )
    report.R2_2c9 = _r2_mbi(
        cmax, inhibitor.kinact_2c9_per_h, inhibitor.ki_mbi_2c9_uM, KDEG["CYP2C9"]
    )
    report.R2_1a2 = _r2_mbi(
        cmax, inhibitor.kinact_1a2_per_h, inhibitor.ki_mbi_1a2_uM, KDEG["CYP1A2"]
    )

    for r2_val, enzyme in [
        (report.R2_3a4, "CYP3A4"),
        (report.R2_2d6, "CYP2D6"),
        (report.R2_2c9, "CYP2C9"),
        (report.R2_1a2, "CYP1A2"),
    ]:
        if r2_val >= 1.25:
            flags.append(f"{enzyme} MBI R2={r2_val:.3f} ≥ 1.25 → time-dependent inhibition study")

    # ------------------------------------------------------------------
    # 4. CYP induction (Emax/EC50 model, Fahmi 2008)
    # ------------------------------------------------------------------
    report.induction_ratio_3a4 = _induction_ratio(
        cmax, inhibitor.ind_max_3a4, inhibitor.ind_c50_3a4_uM
    )
    report.induction_ratio_1a2 = _induction_ratio(
        cmax, inhibitor.ind_max_1a2, inhibitor.ind_c50_1a2_uM
    )

    for ind_ratio, enzyme in [
        (report.induction_ratio_3a4, "CYP3A4"),
        (report.induction_ratio_1a2, "CYP1A2"),
    ]:
        if ind_ratio >= 2.0:
            flags.append(
                f"{enzyme} induction ratio={ind_ratio:.1f}× ≥ 2.0 → induction study recommended"
            )

    # Backwards-compatible legacy field
    if inhibitor.induction_fold_3a4 > 1.0 and report.induction_ratio_3a4 <= 1.0:
        report.induction_ratio_3a4 = inhibitor.induction_fold_3a4

    # ------------------------------------------------------------------
    # 5. Transporter DDI
    #    Intestinal P-gp: R = 1 + [Igut] / IC50  ([Igut] = dose-based)
    #    Hepatic OATP: R = 1 + [Cinlet] / IC50  (Cinlet = Cmax + portal)
    #    Renal OCT2: R = 1 + Cmax / IC50
    # ------------------------------------------------------------------
    # Intestinal P-gp — high luminal concentration: use dose-based [I]
    dose_uM_gut = dose_umol / 0.25  # ~250 mL intestinal fluid volume
    report.R_pgp = 1.0 + dose_uM_gut / max(inhibitor.ic50_pgp_uM, 1e-12)

    # Hepatic OATP — inlet concentration = Cmax + fa·ka·dose / (Qh/BPR)
    c_inlet = cmax + inhibitor.fa * inhibitor.ka_per_h * dose_umol / max(Q_LIVER_L_PER_H, 1e-12)
    report.R_oatp1b1 = 1.0 + c_inlet / max(inhibitor.ic50_oatp1b1_uM, 1e-12)
    report.R_oatp1b3 = 1.0 + c_inlet / max(inhibitor.ic50_oatp1b3_uM, 1e-12)

    # Renal OCT2 — use systemic Cmax
    report.R_oct2 = 1.0 + cmax / max(inhibitor.ic50_oct2_uM, 1e-12)

    for r_val, transporter in [
        (report.R_pgp, "P-gp (intestinal)"),
        (report.R_oatp1b1, "OATP1B1 (hepatic)"),
        (report.R_oatp1b3, "OATP1B3 (hepatic)"),
        (report.R_oct2, "OCT2 (renal)"),
    ]:
        if r_val >= 1.25 and not math.isinf(r_val):
            flags.append(f"{transporter} DDI R={r_val:.2f} ≥ 1.25 → transporter clinical study")

    # ------------------------------------------------------------------
    # 6. Net AUCR prediction for CYP3A4 victim
    # ------------------------------------------------------------------
    report.aucr_3a4 = _aucr(
        fm=victim_fm_3a4,
        R1=report.R1_3a4,
        R2=report.R2_3a4,
        induction_ratio=report.induction_ratio_3a4,
    )

    # ------------------------------------------------------------------
    # Finalise report
    # ------------------------------------------------------------------
    report.flags = flags
    if flags:
        report.recommendation = "Clinical DDI study recommended. See flags."

    return report


def format_report(report: DDIRiskReport) -> str:
    """Format DDI risk report as human-readable string."""

    def _flag(val: float, threshold: float) -> str:
        return " ⚠" if val >= threshold else ""

    lines = [
        f"DDI Risk Assessment: {report.inhibitor_name}",
        "=" * 60,
        "",
        "── Static Reversible Inhibition (R1, threshold ≥ 2.0) ──────",
        f"  CYP3A4  R1 = {report.R1_3a4:.3f}{_flag(report.R1_3a4, 2.0)}",
        f"  CYP2D6  R1 = {report.R1_2d6:.3f}{_flag(report.R1_2d6, 2.0)}",
        f"  CYP2C9  R1 = {report.R1_2c9:.3f}{_flag(report.R1_2c9, 2.0)}",
        f"  CYP1A2  R1 = {report.R1_1a2:.3f}{_flag(report.R1_1a2, 2.0)}",
        f"  CYP2C19 R1 = {report.R1_2c19:.3f}{_flag(report.R1_2c19, 2.0)}",
        "",
        "── Intestinal CYP3A4 (R1gut, threshold ≥ 11.0) ─────────────",
        f"  R1gut   = {report.R1gut_3a4:.3f}{_flag(report.R1gut_3a4, 11.0)}",
        "",
        "── Mechanism-Based Inactivation (R2, threshold ≥ 1.25) ──────",
        f"  CYP3A4  R2 = {report.R2_3a4:.3f}{_flag(report.R2_3a4, 1.25)}",
        f"  CYP2D6  R2 = {report.R2_2d6:.3f}{_flag(report.R2_2d6, 1.25)}",
        f"  CYP2C9  R2 = {report.R2_2c9:.3f}{_flag(report.R2_2c9, 1.25)}",
        f"  CYP1A2  R2 = {report.R2_1a2:.3f}{_flag(report.R2_1a2, 1.25)}",
        "",
        "── CYP Induction (fold-increase CLint, threshold ≥ 2.0×) ───",
        f"  CYP3A4  induction = {report.induction_ratio_3a4:.2f}×"
        f"{_flag(report.induction_ratio_3a4, 2.0)}",
        f"  CYP1A2  induction = {report.induction_ratio_1a2:.2f}×"
        f"{_flag(report.induction_ratio_1a2, 2.0)}",
        "",
        "── Transporter DDI (R, threshold ≥ 1.25) ───────────────────",
        f"  P-gp  (intestinal) R = {report.R_pgp:.3f}{_flag(report.R_pgp, 1.25)}",
        f"  OATP1B1 (hepatic)  R = {report.R_oatp1b1:.3f}{_flag(report.R_oatp1b1, 1.25)}",
        f"  OATP1B3 (hepatic)  R = {report.R_oatp1b3:.3f}{_flag(report.R_oatp1b3, 1.25)}",
        f"  OCT2    (renal)    R = {report.R_oct2:.3f}{_flag(report.R_oct2, 1.25)}",
        "",
        "── Net CYP3A4 Victim AUCR Prediction ───────────────────────",
        f"  AUCR (CYP3A4) = {report.aucr_3a4:.3f}",
        "",
        "── Risk Flags ───────────────────────────────────────────────",
    ]

    if report.flags:
        for flag in report.flags:
            lines.append(f"  ⚠  {flag}")
    else:
        lines.append("  No risk flags raised.")

    lines.append("")
    lines.append(f"Recommendation: {report.recommendation}")
    return "\n".join(lines)
