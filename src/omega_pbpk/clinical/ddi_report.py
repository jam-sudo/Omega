"""FDA-guidance DDI risk assessment report generator.

Implements victim drug assessment per FDA 2020 Drug Interaction Guidance.
Static inhibition: R1 = 1 + [I]/Ki  (plasma)
             R1gut = 1 + Fg·ka·Dose / (Q_gut · Ki_gut)
MBI (time-dependent): R2 = kinact/(kinact + kdeg) × 1/(1 + kdeg·t½_drug)

Risk thresholds:
  R1 >= 2.0  → clinical DDI study recommended
  R1gut >= 11 → gut CYP3A4 inhibition study recommended
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class DDIInhibitor:
    """Perpetrator drug properties for DDI risk assessment."""
    name: str
    cmax_uM: float          # Maximum plasma concentration (µM)
    ki_3a4_uM: float = float("inf")   # CYP3A4 Ki (µM), inf = no inhibition
    ki_2d6_uM: float = float("inf")
    ki_2c9_uM: float = float("inf")
    ki_1a2_uM: float = float("inf")
    kinact_3a4_per_h: float = 0.0  # MBI kinact (/h)
    ki_mbi_3a4_uM: float = float("inf")  # MBI Ki (µM)
    dose_mg: float = 100.0
    mw: float = 300.0
    fa: float = 0.85
    fg: float = 0.6
    ka_per_h: float = 1.0
    induction_fold_3a4: float = 1.0   # max fold induction


@dataclass
class DDIRiskReport:
    """FDA DDI risk assessment results."""
    inhibitor_name: str
    # Static inhibition R1 values (>= 2.0 triggers clinical study)
    R1_3a4: float = 1.0
    R1_2d6: float = 1.0
    R1_2c9: float = 1.0
    R1_1a2: float = 1.0
    # Gut R1 for CYP3A4 (>= 11 triggers gut study)
    R1gut_3a4: float = 1.0
    # MBI-based R2 (time-dependent)
    R2_3a4: float = 1.0
    # Induction
    induction_fold_3a4: float = 1.0
    # Risk flags
    flags: list[str] = field(default_factory=list)
    recommendation: str = "No clinical DDI study warranted"


# FDA guidance constants
KDEG_3A4_PER_H = 0.0005 * 60  # 0.0005 /min × 60 = 0.03 /h
QGUT_L_PER_H = 18.0            # gut blood flow
FG_DEFAULT = 0.6


def assess_ddi_risk(
    inhibitor: DDIInhibitor,
    victim_fm_3a4: float = 0.5,  # fraction metabolized by CYP3A4
) -> DDIRiskReport:
    """Run FDA 2020 static DDI risk assessment for a perpetrator compound.

    Args:
        inhibitor: Perpetrator drug properties
        victim_fm_3a4: Fraction of victim drug metabolized by CYP3A4
    """
    report = DDIRiskReport(inhibitor_name=inhibitor.name)
    flags: list[str] = []

    # Static R1 = 1 + [I]/Ki  (use Cmax,unbound ≈ Cmax for plasma)
    # Using total Cmax as conservative estimate
    cmax = inhibitor.cmax_uM

    def r1(ki: float) -> float:
        return 1.0 + cmax / max(ki, 1e-9)

    report.R1_3a4 = r1(inhibitor.ki_3a4_uM)
    report.R1_2d6 = r1(inhibitor.ki_2d6_uM)
    report.R1_2c9 = r1(inhibitor.ki_2c9_uM)
    report.R1_1a2 = r1(inhibitor.ki_1a2_uM)

    # Gut R1gut = 1 + Fg*ka*Dose_mol / (Qgut * Ki_gut)
    dose_mol = (inhibitor.dose_mg / inhibitor.mw) * 1e6  # µmol
    R1gut_numerator = inhibitor.fa * inhibitor.ka_per_h * dose_mol
    report.R1gut_3a4 = 1.0 + R1gut_numerator / (QGUT_L_PER_H * max(inhibitor.ki_3a4_uM, 1e-9))

    # MBI R2 = kinact/(kinact + kdeg) × correction
    kinact = inhibitor.kinact_3a4_per_h
    ki_mbi = inhibitor.ki_mbi_3a4_uM
    if kinact > 0 and ki_mbi < float("inf"):
        report.R2_3a4 = 1.0 / (1.0 - kinact / (kinact + KDEG_3A4_PER_H) * cmax / (cmax + ki_mbi))

    # Induction
    report.induction_fold_3a4 = inhibitor.induction_fold_3a4

    # Risk flagging
    if report.R1_3a4 >= 2.0:
        flags.append(
            f"CYP3A4 static inhibition R1={report.R1_3a4:.1f} ≥ 2.0 → clinical study recommended"
        )
    if report.R1_2d6 >= 2.0:
        flags.append(
            f"CYP2D6 static inhibition R1={report.R1_2d6:.1f} ≥ 2.0 → clinical study recommended"
        )
    if report.R1_2c9 >= 2.0:
        flags.append(
            f"CYP2C9 static inhibition R1={report.R1_2c9:.1f} ≥ 2.0 → clinical study recommended"
        )
    if report.R1_1a2 >= 2.0:
        flags.append(
            f"CYP1A2 static inhibition R1={report.R1_1a2:.1f} ≥ 2.0 → clinical study recommended"
        )
    if report.R1gut_3a4 >= 11.0:
        flags.append(
            f"CYP3A4 gut inhibition R1gut={report.R1gut_3a4:.1f} ≥ 11.0 → intestinal DDI study"
        )
    if report.R2_3a4 >= 1.25:
        flags.append(f"CYP3A4 MBI R2={report.R2_3a4:.2f} ≥ 1.25 → time-dependent study")
    if report.induction_fold_3a4 >= 2.0:
        flags.append(f"CYP3A4 induction {report.induction_fold_3a4:.1f}× → induction study")

    report.flags = flags
    if flags:
        report.recommendation = "Clinical DDI study recommended. See flags."

    return report


def format_report(report: DDIRiskReport) -> str:
    """Format DDI risk report as human-readable string."""
    lines = [
        f"DDI Risk Assessment: {report.inhibitor_name}",
        "=" * 50,
        "Static R1 values (threshold ≥ 2.0):",
        f"  CYP3A4: {report.R1_3a4:.2f}",
        f"  CYP2D6: {report.R1_2d6:.2f}",
        f"  CYP2C9: {report.R1_2c9:.2f}",
        f"  CYP1A2: {report.R1_1a2:.2f}",
        f"Gut R1gut CYP3A4: {report.R1gut_3a4:.2f} (threshold ≥ 11.0)",
        f"MBI R2  CYP3A4:  {report.R2_3a4:.2f} (threshold ≥ 1.25)",
        f"Induction CYP3A4: {report.induction_fold_3a4:.1f}× (threshold ≥ 2.0)",
        "",
        "Flags:" if report.flags else "No risk flags.",
    ]
    for flag in report.flags:
        lines.append(f"  ⚠  {flag}")
    lines.append(f"\nRecommendation: {report.recommendation}")
    return "\n".join(lines)
