"""Drug-likeness scoring using multiple rule sets: Lipinski Ro5, Veber, Ghose, Egan, Muegge."""

from dataclasses import dataclass


@dataclass(frozen=True)
class RuleSetResult:
    rule_set: str
    passed: bool
    violations: tuple  # tuple of violation strings
    score: float  # 0-1 fraction of rules passed


@dataclass(frozen=True)
class DrugLikenessResult:
    drug_name: str
    logP: float
    mw: float
    psa: float
    n_hbd: int
    n_hba: int
    n_rotatable_bonds: int
    n_aromatic_rings: int
    logD: float
    lipinski: RuleSetResult
    veber: RuleSetResult
    ghose: RuleSetResult
    egan: RuleSetResult
    muegge: RuleSetResult
    overall_score: float  # weighted average of rule set scores (0-100)
    drug_likeness_class: str  # "drug_like","borderline","non_drug_like"
    lead_like: bool  # MW<350, logP<3, HBD<3, HBA<6
    fragment_like: bool  # MW<300, logP<3, HBD<=3, HBA<=3
    notes: str


def apply_lipinski(mw: float, logP: float, n_hbd: int, n_hba: int) -> RuleSetResult:
    """MW<=500, logP<=5, HBD<=5, HBA<=10 (rule of 5 with max 1 violation)."""
    violations = []
    total_rules = 4

    if mw > 500:
        violations.append(f"MW={mw:.1f} > 500")
    if logP > 5:
        violations.append(f"logP={logP:.2f} > 5")
    if n_hbd > 5:
        violations.append(f"HBD={n_hbd} > 5")
    if n_hba > 10:
        violations.append(f"HBA={n_hba} > 10")

    n_violations = len(violations)
    passed = n_violations <= 1  # Lipinski allows max 1 violation
    score = (total_rules - n_violations) / total_rules
    score = max(0.0, score)

    return RuleSetResult(
        rule_set="Lipinski",
        passed=passed,
        violations=tuple(violations),
        score=score,
    )


def apply_veber(n_rotatable: int, psa: float) -> RuleSetResult:
    """Rotatable bonds <= 10, PSA <= 140 A^2."""
    violations = []
    total_rules = 2

    if n_rotatable > 10:
        violations.append(f"RotBonds={n_rotatable} > 10")
    if psa > 140:
        violations.append(f"PSA={psa:.1f} > 140")

    n_violations = len(violations)
    passed = n_violations == 0
    score = (total_rules - n_violations) / total_rules

    return RuleSetResult(
        rule_set="Veber",
        passed=passed,
        violations=tuple(violations),
        score=score,
    )


def apply_ghose(
    logP: float,
    mw: float,
    n_atoms: int = 30,
    molar_refractivity: float = 80.0,
) -> RuleSetResult:
    """logP in [-0.4, 5.6], MW in [160, 480], atoms in [20, 70], MR in [40, 130]."""
    violations = []
    total_rules = 4

    if logP < -0.4 or logP > 5.6:
        violations.append(f"logP={logP:.2f} not in [-0.4, 5.6]")
    if mw < 160 or mw > 480:
        violations.append(f"MW={mw:.1f} not in [160, 480]")
    if n_atoms < 20 or n_atoms > 70:
        violations.append(f"atoms={n_atoms} not in [20, 70]")
    if molar_refractivity < 40 or molar_refractivity > 130:
        violations.append(f"MR={molar_refractivity:.1f} not in [40, 130]")

    n_violations = len(violations)
    passed = n_violations == 0
    score = (total_rules - n_violations) / total_rules

    return RuleSetResult(
        rule_set="Ghose",
        passed=passed,
        violations=tuple(violations),
        score=score,
    )


def apply_egan(logP: float, psa: float) -> RuleSetResult:
    """logP <= 5.88, PSA <= 131.6 (absorption filter)."""
    violations = []
    total_rules = 2

    if logP > 5.88:
        violations.append(f"logP={logP:.2f} > 5.88")
    if psa > 131.6:
        violations.append(f"PSA={psa:.1f} > 131.6")

    n_violations = len(violations)
    passed = n_violations == 0
    score = (total_rules - n_violations) / total_rules

    return RuleSetResult(
        rule_set="Egan",
        passed=passed,
        violations=tuple(violations),
        score=score,
    )


def apply_muegge(
    mw: float,
    logP: float,
    psa: float,
    n_hbd: int,
    n_hba: int,
    n_rotatable: int,
    n_rings: int,
) -> RuleSetResult:
    """MW 200-600, logP -2 to 5, PSA<150, HBD<5, HBA<10, RotBonds<15, rings<7."""
    violations = []
    total_rules = 7

    if mw < 200 or mw > 600:
        violations.append(f"MW={mw:.1f} not in [200, 600]")
    if logP < -2 or logP > 5:
        violations.append(f"logP={logP:.2f} not in [-2, 5]")
    if psa >= 150:
        violations.append(f"PSA={psa:.1f} >= 150")
    if n_hbd >= 5:
        violations.append(f"HBD={n_hbd} >= 5")
    if n_hba >= 10:
        violations.append(f"HBA={n_hba} >= 10")
    if n_rotatable >= 15:
        violations.append(f"RotBonds={n_rotatable} >= 15")
    if n_rings >= 7:
        violations.append(f"rings={n_rings} >= 7")

    n_violations = len(violations)
    passed = n_violations == 0
    score = (total_rules - n_violations) / total_rules

    return RuleSetResult(
        rule_set="Muegge",
        passed=passed,
        violations=tuple(violations),
        score=score,
    )


def score_drug_likeness(
    drug_name: str,
    logP: float,
    mw: float,
    psa: float,
    n_hbd: int,
    n_hba: int,
    n_rotatable_bonds: int = 5,
    n_aromatic_rings: int = 1,
    logD: float | None = None,
) -> DrugLikenessResult:
    """Score drug-likeness using multiple rule sets."""
    if logD is None:
        logD = logP

    lipinski = apply_lipinski(mw, logP, n_hbd, n_hba)
    veber = apply_veber(n_rotatable_bonds, psa)
    ghose = apply_ghose(logP, mw)
    egan = apply_egan(logP, psa)
    muegge = apply_muegge(mw, logP, psa, n_hbd, n_hba, n_rotatable_bonds, n_aromatic_rings)

    # Overall score: equal-weighted average of all rule set scores x 100
    rule_scores = [lipinski.score, veber.score, ghose.score, egan.score, muegge.score]
    overall_score = (sum(rule_scores) / len(rule_scores)) * 100.0

    if overall_score >= 70:
        drug_likeness_class = "drug_like"
    elif overall_score >= 40:
        drug_likeness_class = "borderline"
    else:
        drug_likeness_class = "non_drug_like"

    # Lead-like: MW<350, logP<3, HBD<3, HBA<6
    lead_like = mw < 350 and logP < 3 and n_hbd < 3 and n_hba < 6

    # Fragment-like: MW<300, logP<3, HBD<=3, HBA<=3
    fragment_like = mw < 300 and logP < 3 and n_hbd <= 3 and n_hba <= 3

    # Build notes
    notes_parts = []
    passed_rules = [r.rule_set for r in [lipinski, veber, ghose, egan, muegge] if r.passed]
    failed_rules = [r.rule_set for r in [lipinski, veber, ghose, egan, muegge] if not r.passed]

    if passed_rules:
        notes_parts.append(f"Passed: {', '.join(passed_rules)}")
    if failed_rules:
        notes_parts.append(f"Failed: {', '.join(failed_rules)}")
    if lead_like:
        notes_parts.append("Lead-like")
    if fragment_like:
        notes_parts.append("Fragment-like")

    notes = "; ".join(notes_parts) if notes_parts else "No notes"

    return DrugLikenessResult(
        drug_name=drug_name,
        logP=logP,
        mw=mw,
        psa=psa,
        n_hbd=n_hbd,
        n_hba=n_hba,
        n_rotatable_bonds=n_rotatable_bonds,
        n_aromatic_rings=n_aromatic_rings,
        logD=logD,
        lipinski=lipinski,
        veber=veber,
        ghose=ghose,
        egan=egan,
        muegge=muegge,
        overall_score=overall_score,
        drug_likeness_class=drug_likeness_class,
        lead_like=lead_like,
        fragment_like=fragment_like,
        notes=notes,
    )
