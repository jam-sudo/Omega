"""Expanded Drug Metabolite Safety Assessment (MIST).

Implements FDA MIST guidance (2020) and ICH M3(R2) for identifying
metabolites requiring safety qualification based on exposure thresholds.

FDA MIST thresholds:
  - Metabolite requiring qualification: >10% of total drug-related material
    AUC in humans.
  - High concern (disproportionate): present in humans at >10% parent
    exposure but <10% in any animal species used in tox studies.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
FDA_MIST_THRESHOLD_PCT: float = 10.0  # percent
_THRESHOLD_FRAC: float = FDA_MIST_THRESHOLD_PCT / 100.0  # 0.10


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MISTResult:
    """MIST assessment result for one metabolite."""

    parent_name: str
    metabolite_name: str
    auc_parent_human: float
    auc_metabolite_human: float
    metabolite_to_parent_ratio: float  # percent
    auc_rat: float | None
    auc_dog: float | None
    rat_metabolite_ratio: float | None  # percent, None if not provided
    dog_metabolite_ratio: float | None  # percent, None if not provided
    disproportionate_in_humans: bool
    requires_qualification: bool
    recommendation: str
    fda_mist_threshold_pct: float
    notes: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------


def _compute_ratio_pct(auc_metabolite: float, auc_parent: float) -> float:
    """Return metabolite-to-parent AUC ratio as a percentage."""
    return (auc_metabolite / auc_parent) * 100.0


def _build_recommendation(
    metabolite_name: str,
    ratio_pct: float,
    requires_qualification: bool,
    disproportionate_in_humans: bool,
    rat_ratio: float | None,
    dog_ratio: float | None,
) -> tuple[str, list[str]]:
    """Build regulatory recommendation string and notes list."""
    notes: list[str] = []

    if not requires_qualification:
        recommendation = (
            f"{metabolite_name}: Metabolite-to-parent ratio "
            f"({ratio_pct:.1f}%) is below the FDA MIST 10% threshold. "
            "No additional safety qualification required under standard FDA MIST guidance."
        )
        return recommendation, notes

    if disproportionate_in_humans:
        animal_info_parts: list[str] = []
        if rat_ratio is not None:
            animal_info_parts.append(f"rat {rat_ratio:.1f}%")
            if rat_ratio < FDA_MIST_THRESHOLD_PCT:
                notes.append(
                    f"Rat metabolite ratio ({rat_ratio:.1f}%) is below 10% — "
                    "disproportionate exposure in humans relative to rat."
                )
        if dog_ratio is not None:
            animal_info_parts.append(f"dog {dog_ratio:.1f}%")
            if dog_ratio < FDA_MIST_THRESHOLD_PCT:
                notes.append(
                    f"Dog metabolite ratio ({dog_ratio:.1f}%) is below 10% — "
                    "disproportionate exposure in humans relative to dog."
                )
        animal_str = ", ".join(animal_info_parts) if animal_info_parts else "no animal data"
        recommendation = (
            f"{metabolite_name}: DISPROPORTIONATE human metabolite. "
            f"Human ratio: {ratio_pct:.1f}% (exceeds 10% threshold). "
            f"Animal species ratios: {animal_str}. "
            "Dedicated non-clinical safety studies are required per FDA MIST 2020 "
            "and ICH M3(R2) guidance. Consider separate toxicology studies in an "
            "appropriate species where metabolite exposure can be achieved."
        )
    else:
        animal_parts: list[str] = []
        if rat_ratio is not None:
            animal_parts.append(f"rat {rat_ratio:.1f}%")
        if dog_ratio is not None:
            animal_parts.append(f"dog {dog_ratio:.1f}%")
        animal_str = ", ".join(animal_parts) if animal_parts else "no animal data provided"
        recommendation = (
            f"{metabolite_name}: Human metabolite ratio {ratio_pct:.1f}% exceeds "
            f"the 10% FDA MIST threshold. Animal coverage: {animal_str}. "
            "Verify that existing non-clinical tox studies provide adequate coverage. "
            "If animal exposures are sufficient (>10%), dedicated studies may not be required."
        )

    return recommendation, notes


def assess_metabolite_mist(
    parent_name: str,
    metabolite_name: str,
    auc_parent_human: float,
    auc_metabolite_human: float,
    auc_metabolite_rat: float | None = None,
    auc_metabolite_dog: float | None = None,
    daily_dose_mg: float | None = None,
) -> MISTResult:
    """Assess a single metabolite against FDA MIST 2020 thresholds.

    Computes human metabolite-to-parent ratio, assesses disproportionality
    vs animal species, and gives a regulatory recommendation.

    Args:
        parent_name: Name of the parent drug.
        metabolite_name: Name of the metabolite.
        auc_parent_human: Parent AUC in humans (any consistent unit, e.g. ng·h/mL).
        auc_metabolite_human: Metabolite AUC in humans (same unit as parent).
        auc_metabolite_rat: Metabolite AUC in rat (same unit). Optional.
        auc_metabolite_dog: Metabolite AUC in dog (same unit). Optional.
        daily_dose_mg: Daily dose in mg; used for burden note. Optional.

    Returns:
        MISTResult with qualification flag and recommendation.

    Raises:
        ValueError: If AUC values are not positive.
    """
    if auc_parent_human <= 0:
        raise ValueError("auc_parent_human must be > 0")
    if auc_metabolite_human <= 0:
        raise ValueError("auc_metabolite_human must be > 0")
    if auc_metabolite_rat is not None and auc_metabolite_rat <= 0:
        raise ValueError("auc_metabolite_rat must be > 0 when provided")
    if auc_metabolite_dog is not None and auc_metabolite_dog <= 0:
        raise ValueError("auc_metabolite_dog must be > 0 when provided")
    if daily_dose_mg is not None and daily_dose_mg <= 0:
        raise ValueError("daily_dose_mg must be > 0 when provided")

    ratio_pct = _compute_ratio_pct(auc_metabolite_human, auc_parent_human)
    requires_qualification = ratio_pct > FDA_MIST_THRESHOLD_PCT

    # Compute animal ratios (metabolite / parent in each species)
    # We use human parent AUC as denominator per FDA MIST guidance which
    # compares metabolite-to-parent ratio in each species separately.
    # For animal disproportionality: compare metabolite AUC_animal to
    # the animal's own parent AUC — but since animal parent AUC is not
    # provided, the standard approach is to compare animal metabolite
    # ratio (met/parent) in each species. Here we compute
    # met_animal/parent_human as a proxy exposure ratio.
    rat_ratio: float | None = None
    dog_ratio: float | None = None

    if auc_metabolite_rat is not None:
        rat_ratio = _compute_ratio_pct(auc_metabolite_rat, auc_parent_human)
    if auc_metabolite_dog is not None:
        dog_ratio = _compute_ratio_pct(auc_metabolite_dog, auc_parent_human)

    # Disproportionate: human >10% AND any provided animal species <10%
    disproportionate = False
    if requires_qualification:
        if rat_ratio is not None and rat_ratio < FDA_MIST_THRESHOLD_PCT:
            disproportionate = True
        if dog_ratio is not None and dog_ratio < FDA_MIST_THRESHOLD_PCT:
            disproportionate = True

    recommendation, notes = _build_recommendation(
        metabolite_name=metabolite_name,
        ratio_pct=ratio_pct,
        requires_qualification=requires_qualification,
        disproportionate_in_humans=disproportionate,
        rat_ratio=rat_ratio,
        dog_ratio=dog_ratio,
    )

    if daily_dose_mg is not None:
        burden = calculate_metabolite_burden(auc_parent_human, auc_metabolite_human, daily_dose_mg)
        notes.append(
            f"Estimated daily metabolite burden: {burden:.2f} mg-equivalents/day "
            f"(dose: {daily_dose_mg:.1f} mg/day)"
        )

    return MISTResult(
        parent_name=parent_name,
        metabolite_name=metabolite_name,
        auc_parent_human=auc_parent_human,
        auc_metabolite_human=auc_metabolite_human,
        metabolite_to_parent_ratio=round(ratio_pct, 4),
        auc_rat=auc_metabolite_rat,
        auc_dog=auc_metabolite_dog,
        rat_metabolite_ratio=round(rat_ratio, 4) if rat_ratio is not None else None,
        dog_metabolite_ratio=round(dog_ratio, 4) if dog_ratio is not None else None,
        disproportionate_in_humans=disproportionate,
        requires_qualification=requires_qualification,
        recommendation=recommendation,
        fda_mist_threshold_pct=FDA_MIST_THRESHOLD_PCT,
        notes=notes,
    )


def screen_metabolites(
    parent_name: str,
    metabolites: list[dict],
    auc_parent_human: float,
) -> list[MISTResult]:
    """Screen multiple metabolites against FDA MIST thresholds.

    Args:
        parent_name: Name of the parent drug.
        metabolites: List of dicts, each with keys:
            - name (str): metabolite name
            - auc_human (float): metabolite AUC in humans
            - auc_rat (float, optional): metabolite AUC in rat
            - auc_dog (float, optional): metabolite AUC in dog
        auc_parent_human: Parent AUC in humans.

    Returns:
        List of MISTResult sorted by metabolite_to_parent_ratio descending.

    Raises:
        ValueError: If auc_parent_human <= 0 or required metabolite keys missing.
    """
    if auc_parent_human <= 0:
        raise ValueError("auc_parent_human must be > 0")

    results: list[MISTResult] = []
    for met in metabolites:
        result = assess_metabolite_mist(
            parent_name=parent_name,
            metabolite_name=met["name"],
            auc_parent_human=auc_parent_human,
            auc_metabolite_human=met["auc_human"],
            auc_metabolite_rat=met.get("auc_rat"),
            auc_metabolite_dog=met.get("auc_dog"),
        )
        results.append(result)

    return sorted(results, key=lambda r: r.metabolite_to_parent_ratio, reverse=True)


def calculate_metabolite_burden(
    auc_parent_human: float,
    auc_metabolite_human: float,
    dose_mg: float,
) -> float:
    """Calculate daily metabolite burden in mg-equivalents/day.

    Formula: burden = (AUC_met / AUC_parent) * dose_mg

    Args:
        auc_parent_human: Parent AUC in humans (any consistent unit).
        auc_metabolite_human: Metabolite AUC in humans (same unit).
        dose_mg: Daily dose in mg.

    Returns:
        Daily metabolite burden in mg-equivalents/day.

    Raises:
        ValueError: If any input is not positive.
    """
    if auc_parent_human <= 0:
        raise ValueError("auc_parent_human must be > 0")
    if auc_metabolite_human <= 0:
        raise ValueError("auc_metabolite_human must be > 0")
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")

    return (auc_metabolite_human / auc_parent_human) * dose_mg


# ---------------------------------------------------------------------------
# Phase 323 — Human-specific metabolite safety (FDA MIST / ICH M3(R2))
# ---------------------------------------------------------------------------

_MIST_FRAC: float = 0.10  # 10% threshold as fraction


@dataclass(frozen=True)
class MetaboliteSafetyResult:
    """MIST-based safety result for one metabolite (Phase 323)."""

    compound_name: str
    metabolite_name: str
    auc_metabolite: float  # ng·h/mL
    auc_parent: float  # ng·h/mL
    exposure_fraction: float  # auc_metabolite / auc_parent
    mist_flag: bool  # True if exposure_fraction > 10%
    risk_category: str  # "low", "moderate", "high"
    requires_qualification: bool
    notes: str


@dataclass(frozen=True)
class MetaboliteToxResult:
    """Toxicity risk result for a metabolite (Phase 323)."""

    metabolite_name: str
    smiles: str
    mw: float
    daily_dose_mg: float
    auc_fraction: float  # 0–1
    n_alerts: int
    toxic_burden: float
    risk_category: str  # "low", "moderate", "high"
    notes: str


# ---------------------------------------------------------------------------
# Core functions (Phase 323)
# ---------------------------------------------------------------------------


def metabolite_human_exposure_fraction(auc_metabolite: float, auc_parent: float) -> float:
    """Return metabolite exposure fraction relative to total drug-related AUC.

    FDA MIST: auc_metabolite / (auc_metabolite + auc_parent).

    Args:
        auc_metabolite: Metabolite AUC (any consistent unit, > 0).
        auc_parent: Parent drug AUC (same unit, > 0).

    Returns:
        Fraction in [0, 1].

    Raises:
        ValueError: If either AUC value is not positive.
    """
    if auc_metabolite <= 0:
        raise ValueError("auc_metabolite must be > 0")
    if auc_parent <= 0:
        raise ValueError("auc_parent must be > 0")
    return auc_metabolite / (auc_metabolite + auc_parent)


def classify_metabolite_safety(
    compound_name: str,
    metabolite_name: str,
    auc_metabolite_ng_h_mL: float,
    auc_parent_ng_h_mL: float,
    fm_formation: float,
    structural_alert: bool = False,
    daily_dose_mg: float = 100.0,
) -> MetaboliteSafetyResult:
    """Classify metabolite safety risk under FDA MIST and ICH M3(R2).

    Exposure fraction = auc_metabolite / auc_parent (direct parent ratio).
    MIST flag: exposure_fraction > 10% of parent AUC.
    Risk:
      - low: < 10% exposure AND no structural alert
      - moderate: 10–25% exposure OR structural alert present
      - high: > 25% exposure OR (structural alert AND high daily dose)

    Args:
        compound_name: Parent drug name.
        metabolite_name: Metabolite identifier.
        auc_metabolite_ng_h_mL: Metabolite AUC in humans (ng·h/mL, > 0).
        auc_parent_ng_h_mL: Parent AUC in humans (ng·h/mL, > 0).
        fm_formation: Fraction of parent converted to this metabolite (0–1).
        structural_alert: Whether the metabolite has a structural alert.
        daily_dose_mg: Daily administered dose of the parent drug (mg, > 0).

    Returns:
        MetaboliteSafetyResult with risk classification.

    Raises:
        ValueError: On invalid inputs.
    """
    if auc_metabolite_ng_h_mL <= 0:
        raise ValueError("auc_metabolite_ng_h_mL must be > 0")
    if auc_parent_ng_h_mL <= 0:
        raise ValueError("auc_parent_ng_h_mL must be > 0")
    if not (0.0 <= fm_formation <= 1.0):
        raise ValueError("fm_formation must be in [0, 1]")
    if daily_dose_mg <= 0:
        raise ValueError("daily_dose_mg must be > 0")

    exposure_fraction = auc_metabolite_ng_h_mL / auc_parent_ng_h_mL
    mist_flag = exposure_fraction > _MIST_FRAC

    note_parts: list[str] = []

    # Risk classification
    if exposure_fraction > 0.25 or (structural_alert and daily_dose_mg >= 100.0):
        risk_category = "high"
        requires_qualification = True
        note_parts.append(
            "High-risk metabolite: exposure > 25% parent AUC or structural alert with "
            "high dose. Standalone toxicology studies required (FDA MIST / ICH M3(R2))."
        )
    elif exposure_fraction > _MIST_FRAC or structural_alert:
        risk_category = "moderate"
        requires_qualification = True
        note_parts.append(
            "Moderate-risk metabolite: exposure 10–25% parent AUC or structural alert. "
            "Ensure adequate non-clinical tox coverage."
        )
    else:
        risk_category = "low"
        requires_qualification = False
        note_parts.append(
            "Low-risk metabolite: exposure < 10% parent AUC and no structural alert. "
            "Covered under standard non-clinical studies."
        )

    if mist_flag:
        note_parts.append(
            f"FDA MIST flag: metabolite/parent ratio = {exposure_fraction:.1%} > 10%."
        )

    # ICH M3(R2): unique-to-human metabolites need standalone tox regardless
    # fm_formation used as proxy — high fm (>0.5) with high exposure is a signal
    if fm_formation > 0.5 and exposure_fraction > _MIST_FRAC:
        note_parts.append(
            f"ICH M3(R2) note: high formation fraction (fm={fm_formation:.2f}) "
            "suggests metabolite may be human-specific; verify presence in tox species."
        )

    return MetaboliteSafetyResult(
        compound_name=compound_name,
        metabolite_name=metabolite_name,
        auc_metabolite=round(auc_metabolite_ng_h_mL, 6),
        auc_parent=round(auc_parent_ng_h_mL, 6),
        exposure_fraction=round(exposure_fraction, 6),
        mist_flag=mist_flag,
        risk_category=risk_category,
        requires_qualification=requires_qualification,
        notes=" | ".join(note_parts),
    )


def predict_metabolite_toxicity_risk(
    metabolite_name: str,
    smiles: str,
    mw: float,
    daily_dose_mg: float,
    auc_fraction: float,
    structural_alerts: list[str] | None = None,
) -> MetaboliteToxResult:
    """Predict metabolite toxicity risk combining exposure, dose, and alerts.

    toxic_burden = auc_fraction * daily_dose_mg / 100 * (1 + 0.5 * n_alerts)
    Risk: low (< 0.5), moderate (0.5–2), high (> 2).

    Args:
        metabolite_name: Metabolite identifier.
        smiles: SMILES string (used for context only).
        mw: Molecular weight (g/mol).
        daily_dose_mg: Daily administered dose (mg, > 0).
        auc_fraction: Metabolite AUC / parent AUC (0–1).
        structural_alerts: List of structural alert names (optional).

    Returns:
        MetaboliteToxResult with toxic burden and risk category.

    Raises:
        ValueError: On invalid inputs.
    """
    if daily_dose_mg <= 0:
        raise ValueError("daily_dose_mg must be > 0")
    if not (0.0 <= auc_fraction <= 1.0):
        raise ValueError("auc_fraction must be in [0, 1]")

    alerts: list[str] = structural_alerts if structural_alerts is not None else []
    n_alerts = len(alerts)

    toxic_burden = auc_fraction * daily_dose_mg / 100.0 * (1.0 + 0.5 * n_alerts)

    if toxic_burden < 0.5:
        risk_category = "low"
        note = f"Low toxic burden ({toxic_burden:.3f}); acceptable metabolite risk profile."
    elif toxic_burden <= 2.0:
        risk_category = "moderate"
        note = (
            f"Moderate toxic burden ({toxic_burden:.3f}); monitor for exposure-related toxicity."
        )
    else:
        risk_category = "high"
        note = (
            f"High toxic burden ({toxic_burden:.3f}); dedicated metabolite safety studies "
            "strongly recommended."
        )

    if alerts:
        note += f" Structural alerts: {', '.join(alerts)}."

    return MetaboliteToxResult(
        metabolite_name=metabolite_name,
        smiles=smiles,
        mw=mw,
        daily_dose_mg=daily_dose_mg,
        auc_fraction=round(auc_fraction, 6),
        n_alerts=n_alerts,
        toxic_burden=round(toxic_burden, 6),
        risk_category=risk_category,
        notes=note,
    )


def metabolite_battery_screen(
    metabolites: list[dict],
) -> list[MetaboliteSafetyResult]:
    """Screen a list of metabolites, ranking by exposure fraction.

    Each dict must have keys:
      - compound_name (str): parent drug name
      - metabolite_name (str): metabolite identifier
      - auc_metabolite (float): metabolite AUC in ng·h/mL
      - auc_parent (float): parent AUC in ng·h/mL
      - fm_formation (float): fraction of parent formed as this metabolite
      - structural_alert (bool, optional): structural alert flag
      - daily_dose_mg (float, optional): daily dose in mg

    Returns:
        List of MetaboliteSafetyResult sorted by exposure_fraction descending.

    Raises:
        ValueError: If required keys are missing or values are invalid.
    """
    results: list[MetaboliteSafetyResult] = []
    for met in metabolites:
        result = classify_metabolite_safety(
            compound_name=met["compound_name"],
            metabolite_name=met["metabolite_name"],
            auc_metabolite_ng_h_mL=met["auc_metabolite"],
            auc_parent_ng_h_mL=met["auc_parent"],
            fm_formation=met["fm_formation"],
            structural_alert=met.get("structural_alert", False),
            daily_dose_mg=met.get("daily_dose_mg", 100.0),
        )
        results.append(result)

    return sorted(results, key=lambda r: r.exposure_fraction, reverse=True)


__all__ = [
    "FDA_MIST_THRESHOLD_PCT",
    "MISTResult",
    "assess_metabolite_mist",
    "screen_metabolites",
    "calculate_metabolite_burden",
    # Phase 323
    "MetaboliteSafetyResult",
    "MetaboliteToxResult",
    "metabolite_human_exposure_fraction",
    "classify_metabolite_safety",
    "predict_metabolite_toxicity_risk",
    "metabolite_battery_screen",
]
