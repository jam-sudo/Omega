"""Phase 611 — ADME Liability Flagging.

Flag potential ADME liabilities based on physicochemical and structural properties.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ADMEFlag:
    flag_name: str
    category: str  # "absorption","distribution","metabolism","excretion","toxicity"
    severity: str  # "info","warning","concern","critical"
    description: str
    recommendation: str


@dataclass(frozen=True)
class ADMEFlagResult:
    drug_name: str
    flags: tuple  # tuple of ADMEFlag
    n_total_flags: int
    n_critical: int
    n_concerns: int
    n_warnings: int
    overall_risk: str  # "low","moderate","high","very_high"
    developability_score: float  # 0-100 (100=best)
    pass_absorption: bool
    pass_distribution: bool
    pass_metabolism: bool
    pass_excretion: bool
    notes: str


def flag_adme_liabilities(
    drug_name: str,
    logP: float,
    mw: float,
    psa: float,
    n_hbd: int,
    n_hba: int,
    solubility_mg_mL: float,
    n_aromatic_rings: int = 0,
    n_rotatable_bonds: int = 5,
    pka_acid: float | None = None,
    pka_base: float | None = None,
    fu_plasma: float = 0.5,
    vd_L: float = 50.0,
) -> ADMEFlagResult:
    """Flag ADME liabilities based on physicochemical and structural properties."""
    flags: list[ADMEFlag] = []

    # ---- Absorption flags ----
    if solubility_mg_mL < 0.1:
        flags.append(
            ADMEFlag(
                flag_name="low_solubility",
                category="absorption",
                severity="critical",
                description=f"Very low aqueous solubility ({solubility_mg_mL:.3f} mg/mL < 0.1 mg/mL). "
                "Dissolution-limited absorption expected.",
                recommendation="Consider salt formation, amorphous solid dispersion, or nano-milling to improve dissolution.",
            )
        )
    elif solubility_mg_mL < 1.0:
        flags.append(
            ADMEFlag(
                flag_name="low_solubility",
                category="absorption",
                severity="warning",
                description=f"Low aqueous solubility ({solubility_mg_mL:.3f} mg/mL < 1.0 mg/mL). "
                "May cause variable oral absorption.",
                recommendation="Evaluate formulation strategies to enhance solubility/dissolution.",
            )
        )

    if mw > 700:
        flags.append(
            ADMEFlag(
                flag_name="high_mw",
                category="absorption",
                severity="critical",
                description=f"Molecular weight ({mw:.1f} Da) exceeds 700 Da. "
                "Severely impaired passive permeability expected.",
                recommendation="Reduce molecular size through scaffold optimization or consider alternative routes.",
            )
        )
    elif mw > 500:
        flags.append(
            ADMEFlag(
                flag_name="high_mw",
                category="absorption",
                severity="concern",
                description=f"Molecular weight ({mw:.1f} Da) exceeds Lipinski 500 Da threshold. "
                "Reduced oral absorption likely.",
                recommendation="Consider structural simplification to reduce MW below 500 Da.",
            )
        )

    if psa > 140:
        flags.append(
            ADMEFlag(
                flag_name="high_psa",
                category="absorption",
                severity="critical",
                description=f"Polar surface area ({psa:.1f} Å²) exceeds 140 Å². "
                "Oral bioavailability expected to be <10%.",
                recommendation="Reduce hydrogen bond donors/acceptors or consider prodrug strategies.",
            )
        )
    elif psa > 90:
        flags.append(
            ADMEFlag(
                flag_name="high_psa",
                category="absorption",
                severity="concern",
                description=f"Polar surface area ({psa:.1f} Å²) exceeds 90 Å². "
                "Membrane permeability may be compromised.",
                recommendation="Evaluate permeability in Caco-2 or PAMPA assay; consider bioisostere replacement.",
            )
        )

    if n_hbd > 5:
        flags.append(
            ADMEFlag(
                flag_name="excess_hbd",
                category="absorption",
                severity="concern",
                description=f"Number of hydrogen bond donors ({n_hbd}) exceeds Lipinski limit of 5. "
                "Passive permeability likely reduced.",
                recommendation="Reduce HBD count through N-methylation, cyclization, or prodrug design.",
            )
        )

    if n_hba > 10:
        flags.append(
            ADMEFlag(
                flag_name="excess_hba",
                category="absorption",
                severity="concern",
                description=f"Number of hydrogen bond acceptors ({n_hba}) exceeds Lipinski limit of 10. "
                "Absorption may be impaired.",
                recommendation="Simplify the molecule or consider bioisostere replacement of acceptors.",
            )
        )

    if logP > 5:
        flags.append(
            ADMEFlag(
                flag_name="high_logP_poor_dissolution",
                category="absorption",
                severity="warning",
                description=f"High lipophilicity (logP={logP:.1f} > 5) may result in poor aqueous dissolution "
                "despite good membrane permeability.",
                recommendation="Assess dissolution in biorelevant media; consider lipophilicity reduction.",
            )
        )

    # ---- Distribution flags ----
    if fu_plasma < 0.05:
        flags.append(
            ADMEFlag(
                flag_name="high_protein_binding",
                category="distribution",
                severity="concern",
                description=f"High plasma protein binding (fu={fu_plasma:.3f}, >95% bound). "
                "Free drug concentration severely limited.",
                recommendation="Evaluate impact on efficacy; consider measuring unbound tissue concentrations.",
            )
        )

    if vd_L < 5.0:
        flags.append(
            ADMEFlag(
                flag_name="low_volume_distribution",
                category="distribution",
                severity="info",
                description=f"Low volume of distribution ({vd_L:.1f} L). "
                "Drug largely confined to plasma/blood.",
                recommendation="May limit tissue penetration; assess target tissue exposure.",
            )
        )

    if vd_L > 500:
        flags.append(
            ADMEFlag(
                flag_name="high_volume_distribution",
                category="distribution",
                severity="warning",
                description=f"Very high volume of distribution ({vd_L:.1f} L). "
                "Extensive tissue distribution with potential accumulation.",
                recommendation="Evaluate tissue accumulation risk, especially in adipose and highly perfused organs.",
            )
        )

    if psa > 90 or mw > 450:
        flags.append(
            ADMEFlag(
                flag_name="cns_penetration_concern",
                category="distribution",
                severity="warning",
                description=f"PSA={psa:.1f} Å² or MW={mw:.1f} Da suggest poor blood-brain barrier penetration.",
                recommendation="If CNS activity required, optimize for PSA<90 Å² and MW<450 Da.",
            )
        )

    if logP > 5:
        flags.append(
            ADMEFlag(
                flag_name="high_logP_cns",
                category="distribution",
                severity="concern",
                description=f"High logP ({logP:.1f} > 5) associated with non-specific CNS binding "
                "and promiscuous protein binding.",
                recommendation="Reduce lipophilicity; target logP 1-3 for CNS candidates.",
            )
        )

    # ---- Metabolism flags ----
    if n_aromatic_rings >= 4:
        flags.append(
            ADMEFlag(
                flag_name="pan_assay_interference",
                category="metabolism",
                severity="warning",
                description=f"High aromatic ring count ({n_aromatic_rings} ≥ 4) raises PAINS concerns. "
                "May exhibit pan-assay interference and promiscuous binding.",
                recommendation="Filter against PAINS databases; consider reducing aromatic ring count.",
            )
        )

    if n_aromatic_rings >= 3:
        flags.append(
            ADMEFlag(
                flag_name="high_aromatic_density",
                category="metabolism",
                severity="info",
                description=f"Elevated aromatic ring count ({n_aromatic_rings} ≥ 3). "
                "Increased risk of CYP inhibition and metabolic soft spots.",
                recommendation="Profile against major CYPs; consider aliphatic ring replacements.",
            )
        )

    if logP > 4 and mw < 400:
        flags.append(
            ADMEFlag(
                flag_name="metabolic_soft_spot",
                category="metabolism",
                severity="info",
                description=f"logP={logP:.1f} > 4 combined with small MW ({mw:.1f} < 400) suggests "
                "high hepatic first-pass metabolism.",
                recommendation="Conduct microsomal stability assay; consider metabolic blocking strategies.",
            )
        )

    if logP > 5 and fu_plasma < 0.1:
        flags.append(
            ADMEFlag(
                flag_name="poor_metabolic_stability",
                category="metabolism",
                severity="concern",
                description=f"High lipophilicity (logP={logP:.1f}) with high protein binding (fu={fu_plasma:.2f}) "
                "predicts poor metabolic stability.",
                recommendation="Reduce lipophilicity; measure hepatic intrinsic clearance.",
            )
        )

    # ---- Excretion flags ----
    if logP > 3 and mw > 300:
        flags.append(
            ADMEFlag(
                flag_name="low_renal_clearance_expected",
                category="excretion",
                severity="info",
                description=f"logP={logP:.1f} > 3 and MW={mw:.1f} > 300 suggest limited renal excretion. "
                "Hepatic metabolism likely primary elimination route.",
                recommendation="Assess renal vs hepatic clearance split; adjust dose in hepatic impairment.",
            )
        )

    if mw > 500:
        flags.append(
            ADMEFlag(
                flag_name="bile_excretion_risk",
                category="excretion",
                severity="info",
                description=f"High MW ({mw:.1f} > 500) raises possibility of biliary excretion "
                "and enterohepatic recycling.",
                recommendation="Evaluate biliary excretion in bile-duct cannulated animal models.",
            )
        )

    # ---- Toxicity flags ----
    if n_rotatable_bonds > 10:
        flags.append(
            ADMEFlag(
                flag_name="excess_rotatable_bonds",
                category="toxicity",
                severity="warning",
                description=f"High rotatable bond count ({n_rotatable_bonds} > 10) associated with "
                "poor oral absorption and potential toxicity.",
                recommendation="Reduce conformational flexibility through ring formation or rigidification.",
            )
        )

    if pka_acid is not None and pka_acid < 2:
        flags.append(
            ADMEFlag(
                flag_name="pka_extremes",
                category="toxicity",
                severity="concern",
                description=f"Very low pKa_acid ({pka_acid:.1f} < 2). "
                "Strong acid may cause GI irritation and altered distribution.",
                recommendation="Evaluate GI tolerability; consider prodrug or salt to moderate acidity.",
            )
        )

    if pka_base is not None and pka_base > 10:
        flags.append(
            ADMEFlag(
                flag_name="pka_extremes",
                category="toxicity",
                severity="concern",
                description=f"Very high pKa_base ({pka_base:.1f} > 10). "
                "Strong base may accumulate in lysosomes (phospholipidosis risk).",
                recommendation="Evaluate lysosomal trapping; assess phospholipidosis biomarkers.",
            )
        )

    # ---- Score computation ----
    n_critical = sum(1 for f in flags if f.severity == "critical")
    n_concerns = sum(1 for f in flags if f.severity == "concern")
    n_warnings = sum(1 for f in flags if f.severity == "warning")
    n_info = sum(1 for f in flags if f.severity == "info")

    score = 100.0
    score -= n_critical * 20
    score -= n_concerns * 10
    score -= n_warnings * 5
    score -= n_info * 1
    score = max(0.0, min(100.0, score))

    # ---- Overall risk ----
    if n_critical > 0 or score < 40:
        overall_risk = "very_high"
    elif n_concerns >= 3 or score < 60:
        overall_risk = "high"
    elif n_warnings >= 3 or score < 80:
        overall_risk = "moderate"
    else:
        overall_risk = "low"

    # ---- Pass/fail per category ----
    blocking = {"critical", "concern"}

    def _passes(category: str) -> bool:
        return not any(f.category == category and f.severity in blocking for f in flags)

    pass_absorption = _passes("absorption")
    pass_distribution = _passes("distribution")
    pass_metabolism = _passes("metabolism")
    pass_excretion = _passes("excretion")

    # ---- Notes ----
    note_parts = [f"Score: {score:.0f}/100, Risk: {overall_risk}"]
    if n_critical:
        note_parts.append(f"{n_critical} critical flag(s)")
    if n_concerns:
        note_parts.append(f"{n_concerns} concern(s)")
    notes = "; ".join(note_parts) + "."

    return ADMEFlagResult(
        drug_name=drug_name,
        flags=tuple(flags),
        n_total_flags=len(flags),
        n_critical=n_critical,
        n_concerns=n_concerns,
        n_warnings=n_warnings,
        overall_risk=overall_risk,
        developability_score=score,
        pass_absorption=pass_absorption,
        pass_distribution=pass_distribution,
        pass_metabolism=pass_metabolism,
        pass_excretion=pass_excretion,
        notes=notes,
    )


__all__ = ["ADMEFlag", "ADMEFlagResult", "flag_adme_liabilities"]
