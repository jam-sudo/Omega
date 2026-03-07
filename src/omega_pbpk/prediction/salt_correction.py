"""Phase 786 — Salt form correction and free-base equivalent dose calculation.

Provides tools to convert between salt form doses and free-base equivalent
doses using molecular weight-based salt factors.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "SaltCorrectionResult",
    "calculate_salt_factor",
    "correct_dose_for_salt",
    "correct_dose_by_name",
    "screen_salt_forms",
    "_COUNTERION_MW",
]

# Prebuilt counterion molecular weight table (g/mol)
_COUNTERION_MW: dict[str, float] = {
    "HCl": 36.46,
    "HBr": 80.91,
    "Na": 22.99,
    "K": 39.10,
    "Ca": 40.08,
    "Mg": 24.31,
    "Mesylate": 96.11,
    "Tosylate": 172.20,
    "Maleate": 116.07,
    "Fumarate": 116.07,
    "Tartrate": 150.09,
    "Citrate": 192.12,
    "Phosphate": 97.99,
    "Succinate": 118.09,
    "Acetate": 59.04,
    "Sulfate": 96.06,
}

# Map salt forms to human-readable ionization state labels
_IONIZATION_LABELS: dict[str, str] = {
    "HCl": "HCl salt",
    "HBr": "HBr salt",
    "Na": "Na salt",
    "K": "K salt",
    "Ca": "Ca salt",
    "Mg": "Mg salt",
    "Mesylate": "Mesylate salt",
    "Tosylate": "Tosylate salt",
    "Maleate": "Maleate salt",
    "Fumarate": "Fumarate salt",
    "Tartrate": "Tartrate salt",
    "Citrate": "Citrate salt",
    "Phosphate": "Phosphate salt",
    "Succinate": "Succinate salt",
    "Acetate": "Acetate salt",
    "Sulfate": "Sulfate salt",
    "free base": "free base",
}


@dataclass(frozen=True)
class SaltCorrectionResult:
    """Result of a salt form dose correction calculation."""

    drug_name: str
    salt_form: str
    salt_dose_mg: float
    salt_factor: float  # fraction of salt that is free base (0 < sf < 1)
    free_base_dose_mg: float
    mw_salt: float
    mw_base: float
    ionization_state: str  # e.g. "HCl salt", "Na salt", "free base"
    bioequivalent_dose_mg: float  # same as free_base_dose_mg for labeling
    notes: str


def calculate_salt_factor(
    mw_base: float,
    mw_counterion: float,
    n_counterions: int = 1,
) -> float:
    """Calculate the salt factor (mass fraction of free base in the salt).

    salt_factor = mw_base / (mw_base + n_counterions * mw_counterion)

    Parameters
    ----------
    mw_base:
        Molecular weight of the free base (g/mol). Must be > 0.
    mw_counterion:
        Molecular weight of the counterion (g/mol). Must be > 0.
    n_counterions:
        Number of counterion molecules per drug molecule. Default 1.

    Returns
    -------
    float
        Salt factor in (0, 1).

    Raises
    ------
    ValueError
        If mw_base <= 0, mw_counterion <= 0, or n_counterions < 1.
    """
    if mw_base <= 0:
        raise ValueError(f"mw_base must be > 0, got {mw_base}.")
    if mw_counterion <= 0:
        raise ValueError(f"mw_counterion must be > 0, got {mw_counterion}.")
    if n_counterions < 1:
        raise ValueError(f"n_counterions must be >= 1, got {n_counterions}.")

    return mw_base / (mw_base + n_counterions * mw_counterion)


def correct_dose_for_salt(
    drug_name: str,
    salt_dose_mg: float,
    mw_base: float,
    mw_counterion: float,
    n_counterions: int = 1,
    salt_form: str = "HCl",
) -> SaltCorrectionResult:
    """Convert a salt form dose to the free-base equivalent dose.

    Parameters
    ----------
    drug_name:
        Name of the drug compound.
    salt_dose_mg:
        Administered dose as the salt form (mg).
    mw_base:
        Molecular weight of the free base (g/mol). Must be > 0.
    mw_counterion:
        Molecular weight of the counterion (g/mol). Must be > 0.
    n_counterions:
        Number of counterion molecules per drug molecule. Default 1.
    salt_form:
        Salt form label (e.g. "HCl", "Na", "Mesylate").

    Returns
    -------
    SaltCorrectionResult
        Dataclass containing free-base dose and related metadata.

    Raises
    ------
    ValueError
        If mw_base <= 0, mw_counterion <= 0, n_counterions < 1,
        or salt_dose_mg < 0.
    """
    if salt_dose_mg < 0:
        raise ValueError(f"salt_dose_mg must be >= 0, got {salt_dose_mg}.")

    sf = calculate_salt_factor(mw_base, mw_counterion, n_counterions)
    free_base_dose = salt_dose_mg * sf
    mw_salt = mw_base + n_counterions * mw_counterion
    ionization_state = _IONIZATION_LABELS.get(salt_form, f"{salt_form} salt")

    notes = (
        f"Salt factor: {sf:.4f}; "
        f"MW base: {mw_base:.2f} g/mol; "
        f"MW salt: {mw_salt:.2f} g/mol; "
        f"n counterions: {n_counterions}"
    )

    return SaltCorrectionResult(
        drug_name=drug_name,
        salt_form=salt_form,
        salt_dose_mg=salt_dose_mg,
        salt_factor=sf,
        free_base_dose_mg=free_base_dose,
        mw_salt=mw_salt,
        mw_base=mw_base,
        ionization_state=ionization_state,
        bioequivalent_dose_mg=free_base_dose,
        notes=notes,
    )


def correct_dose_by_name(
    drug_name: str,
    salt_dose_mg: float,
    mw_base: float,
    salt_form: str,
    n_counterions: int = 1,
) -> SaltCorrectionResult:
    """Convert a salt dose to free-base equivalent using named salt form.

    Looks up the counterion MW from the built-in _COUNTERION_MW table.

    Parameters
    ----------
    drug_name:
        Name of the drug compound.
    salt_dose_mg:
        Administered dose as the salt form (mg).
    mw_base:
        Molecular weight of the free base (g/mol). Must be > 0.
    salt_form:
        Salt form name that must exist in _COUNTERION_MW.
    n_counterions:
        Number of counterion molecules per drug molecule. Default 1.

    Returns
    -------
    SaltCorrectionResult
        Dataclass containing free-base dose and related metadata.

    Raises
    ------
    ValueError
        If mw_base <= 0, salt_form not in _COUNTERION_MW,
        or salt_dose_mg < 0.
    """
    if mw_base <= 0:
        raise ValueError(f"mw_base must be > 0, got {mw_base}.")
    if salt_form not in _COUNTERION_MW:
        raise ValueError(
            f"Unknown salt_form '{salt_form}'. Must be one of {sorted(_COUNTERION_MW.keys())}."
        )

    mw_counterion = _COUNTERION_MW[salt_form]
    return correct_dose_for_salt(
        drug_name=drug_name,
        salt_dose_mg=salt_dose_mg,
        mw_base=mw_base,
        mw_counterion=mw_counterion,
        n_counterions=n_counterions,
        salt_form=salt_form,
    )


def screen_salt_forms(
    drug_name: str,
    target_free_base_dose_mg: float,
    mw_base: float,
    salt_forms: list[str] | None = None,
) -> list[SaltCorrectionResult]:
    """Screen multiple salt forms to achieve a target free-base dose.

    For each salt form, calculates the salt dose required to deliver
    target_free_base_dose_mg of free base, then returns results sorted
    by salt_dose_mg ascending (lowest salt mass first).

    Parameters
    ----------
    drug_name:
        Name of the drug compound.
    target_free_base_dose_mg:
        Desired free-base equivalent dose (mg). Must be > 0.
    mw_base:
        Molecular weight of the free base (g/mol). Must be > 0.
    salt_forms:
        List of salt form names to evaluate. If None, uses all keys in
        _COUNTERION_MW.

    Returns
    -------
    list[SaltCorrectionResult]
        One result per salt form, sorted by salt_dose_mg ascending.

    Raises
    ------
    ValueError
        If mw_base <= 0, target_free_base_dose_mg <= 0,
        or any salt_form not in _COUNTERION_MW.
    """
    if mw_base <= 0:
        raise ValueError(f"mw_base must be > 0, got {mw_base}.")
    if target_free_base_dose_mg <= 0:
        raise ValueError(f"target_free_base_dose_mg must be > 0, got {target_free_base_dose_mg}.")

    forms_to_screen = salt_forms if salt_forms is not None else list(_COUNTERION_MW.keys())

    results: list[SaltCorrectionResult] = []
    for form in forms_to_screen:
        if form not in _COUNTERION_MW:
            raise ValueError(
                f"Unknown salt_form '{form}'. Must be one of {sorted(_COUNTERION_MW.keys())}."
            )
        mw_counterion = _COUNTERION_MW[form]
        sf = calculate_salt_factor(mw_base, mw_counterion)
        # Required salt dose to achieve target free-base dose
        required_salt_dose = target_free_base_dose_mg / sf
        result = correct_dose_for_salt(
            drug_name=drug_name,
            salt_dose_mg=required_salt_dose,
            mw_base=mw_base,
            mw_counterion=mw_counterion,
            n_counterions=1,
            salt_form=form,
        )
        results.append(result)

    # Sort by salt_dose_mg ascending
    results.sort(key=lambda r: r.salt_dose_mg)
    return results
