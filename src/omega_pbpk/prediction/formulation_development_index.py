"""Phase 1110 — Drug Formulation Development Index (FDI).

Composite formulation developability index (0-100) based on solubility,
permeability, chemical stability, physical stability, and dose number.
"""

from __future__ import annotations

from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FormulationDevelopmentResult:
    """Result of formulation development index calculation."""

    drug_name: str
    solubility_mg_mL: float
    logP: float
    dose_mg: float
    melting_point_C: float
    hygroscopic: bool
    has_ester: bool
    has_aldehyde: bool
    has_michael_acceptor: bool
    fdi_score: float
    solubility_score: float
    permeability_score: float
    chemical_stability_score: float
    physical_stability_score: float
    dose_number_score: float
    fdi_class: str
    recommended_formulation: str
    development_risk: str
    notes: str


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _validate_inputs(
    solubility_mg_mL: float,
    logP: float,
    dose_mg: float,
    melting_point_C: float,
) -> None:
    if solubility_mg_mL <= 0:
        raise ValueError(f"solubility_mg_mL must be > 0, got {solubility_mg_mL}")
    if dose_mg <= 0:
        raise ValueError(f"dose_mg must be > 0, got {dose_mg}")
    if not (-5 <= logP <= 10):
        raise ValueError(f"logP must be in [-5, 10], got {logP}")
    if not (20 <= melting_point_C <= 400):
        raise ValueError(f"melting_point_C must be in [20, 400], got {melting_point_C}")


# ---------------------------------------------------------------------------
# Score components
# ---------------------------------------------------------------------------


def _solubility_score(solubility_mg_mL: float) -> float:
    """Score 0-20 based on aqueous solubility."""
    if solubility_mg_mL > 100:
        return 20.0
    if solubility_mg_mL >= 10:
        return 15.0
    if solubility_mg_mL >= 1:
        return 10.0
    if solubility_mg_mL >= 0.1:
        return 5.0
    return 0.0


def _permeability_score(logP: float) -> float:
    """Score 0-20 based on logP as a permeability surrogate."""
    if 1 <= logP <= 3:
        return 20.0
    if (0 <= logP < 1) or (3 < logP <= 5):
        return 15.0
    if (logP < 0 and logP >= -5) or (5 < logP <= 7):
        return 8.0
    # logP > 7
    return 3.0


def _chemical_stability_score(
    has_ester: bool,
    has_aldehyde: bool,
    has_michael_acceptor: bool,
) -> float:
    """Score 0-20 based on reactive functional group presence."""
    if has_michael_acceptor:
        return 3.0
    if has_ester and has_aldehyde:
        return 5.0
    if has_ester or has_aldehyde:
        return 10.0
    return 20.0


def _physical_stability_score(melting_point_C: float, hygroscopic: bool) -> float:
    """Score 0-20 based on melting point and hygroscopicity."""
    if melting_point_C < 100:
        return 5.0
    if melting_point_C > 200 and not hygroscopic:
        return 20.0
    if 100 <= melting_point_C <= 200 and not hygroscopic:
        return 15.0
    if melting_point_C > 200 and hygroscopic:
        return 12.0
    # 100-200 and hygroscopic
    return 8.0


def _dose_number_score(dose_mg: float, solubility_mg_mL: float) -> float:
    """Score 0-20 based on dose number Do = dose/(solubility*250 mL)."""
    Do = dose_mg / (solubility_mg_mL * 250.0)
    if Do < 1:
        return 20.0
    if Do <= 5:
        return 15.0
    if Do <= 20:
        return 8.0
    return 3.0


# ---------------------------------------------------------------------------
# Classification helpers
# ---------------------------------------------------------------------------


def _fdi_class(fdi_score: float) -> str:
    if fdi_score >= 80:
        return "excellent"
    if fdi_score >= 60:
        return "good"
    if fdi_score >= 40:
        return "moderate"
    if fdi_score >= 20:
        return "challenging"
    return "very_challenging"


def _recommended_formulation(
    fdi_score: float,
    solubility_mg_mL: float,
    logP: float,
    has_michael_acceptor: bool,
) -> str:
    if has_michael_acceptor:
        return "parenteral"
    if fdi_score >= 80:
        return "standard_oral"
    if solubility_mg_mL < 0.1 and logP > 3:
        return "solid_dispersion"
    if solubility_mg_mL < 0.1 and logP <= 3:
        return "nanosuspension"
    if logP > 5:
        return "lipid_formulation"
    return "standard_oral"


def _development_risk(fdi_score: float) -> str:
    if fdi_score >= 60:
        return "low"
    if fdi_score >= 35:
        return "moderate"
    return "high"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def calculate_fdi(
    drug_name: str,
    solubility_mg_mL: float,
    logP: float,
    dose_mg: float,
    melting_point_C: float = 180.0,
    hygroscopic: bool = False,
    has_ester: bool = False,
    has_aldehyde: bool = False,
    has_michael_acceptor: bool = False,
) -> FormulationDevelopmentResult:
    """Calculate the Formulation Development Index (FDI) for a drug.

    Parameters
    ----------
    drug_name:
        Identifier for the drug candidate.
    solubility_mg_mL:
        Aqueous solubility in mg/mL (must be > 0).
    logP:
        Octanol-water partition coefficient (must be in [-5, 10]).
    dose_mg:
        Oral dose in milligrams (must be > 0).
    melting_point_C:
        Melting point in Celsius (must be in [20, 400]; default 180).
    hygroscopic:
        Whether the drug is hygroscopic (default False).
    has_ester:
        Whether an ester group is present (default False).
    has_aldehyde:
        Whether an aldehyde group is present (default False).
    has_michael_acceptor:
        Whether a Michael acceptor is present (default False).

    Returns
    -------
    FormulationDevelopmentResult
    """
    _validate_inputs(solubility_mg_mL, logP, dose_mg, melting_point_C)

    sol_score = _solubility_score(solubility_mg_mL)
    perm_score = _permeability_score(logP)
    chem_score = _chemical_stability_score(has_ester, has_aldehyde, has_michael_acceptor)
    phys_score = _physical_stability_score(melting_point_C, hygroscopic)
    dn_score = _dose_number_score(dose_mg, solubility_mg_mL)

    fdi = sol_score + perm_score + chem_score + phys_score + dn_score

    fdi_cls = _fdi_class(fdi)
    rec_form = _recommended_formulation(fdi, solubility_mg_mL, logP, has_michael_acceptor)
    dev_risk = _development_risk(fdi)

    # Build notes
    note_parts: list[str] = []
    if solubility_mg_mL < 0.1:
        note_parts.append("very low solubility is a major development challenge")
    if has_michael_acceptor:
        note_parts.append("Michael acceptor: reactive — parenteral preferred")
    if has_ester and has_aldehyde:
        note_parts.append("both ester and aldehyde groups reduce chemical stability")
    if melting_point_C < 100:
        note_parts.append("low melting point indicates waxy/oily solid — difficult to process")
    if hygroscopic and melting_point_C <= 200:
        note_parts.append("hygroscopic with moderate mp — storage and handling challenges")
    notes = "; ".join(note_parts) if note_parts else "acceptable physicochemical profile"

    return FormulationDevelopmentResult(
        drug_name=drug_name,
        solubility_mg_mL=solubility_mg_mL,
        logP=logP,
        dose_mg=dose_mg,
        melting_point_C=melting_point_C,
        hygroscopic=hygroscopic,
        has_ester=has_ester,
        has_aldehyde=has_aldehyde,
        has_michael_acceptor=has_michael_acceptor,
        fdi_score=fdi,
        solubility_score=sol_score,
        permeability_score=perm_score,
        chemical_stability_score=chem_score,
        physical_stability_score=phys_score,
        dose_number_score=dn_score,
        fdi_class=fdi_cls,
        recommended_formulation=rec_form,
        development_risk=dev_risk,
        notes=notes,
    )


def rank_candidates(candidates: list[dict]) -> list[FormulationDevelopmentResult]:
    """Calculate FDI for each candidate and return sorted by fdi_score descending.

    Each dict in ``candidates`` must contain keyword arguments accepted by
    :func:`calculate_fdi`.

    Returns
    -------
    list[FormulationDevelopmentResult]
        Sorted from most developable (highest FDI) to least.
    """
    results = [calculate_fdi(**c) for c in candidates]
    results.sort(key=lambda r: r.fdi_score, reverse=True)
    return results
