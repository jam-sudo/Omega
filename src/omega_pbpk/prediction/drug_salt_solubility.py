"""Phase 990 — pH-dependent drug salt solubility prediction.

Uses Henderson-Hasselbalch equation to compute solubility across pH gradient
for acids, bases, and neutral compounds.
"""

from __future__ import annotations

from dataclasses import dataclass

_VALID_SALT_FORMS = {
    "HCl",
    "Na",
    "K",
    "Mg",
    "Ca",
    "free_acid",
    "free_base",
    "besylate",
    "mesylate",
    "maleate",
    "tartrate",
}

_VALID_IONIZATION_TYPES = {"acid", "base", "neutral"}


@dataclass(frozen=True)
class DrugSaltSolubilityResult:
    """Result of pH-dependent drug salt solubility prediction."""

    drug_name: str
    salt_form: str
    pka: float
    ionization_type: str
    intrinsic_solubility_mg_mL: float
    ph_values: tuple
    solubility_mg_mL: tuple
    max_solubility_mg_mL: float
    max_solubility_ph: float
    min_solubility_mg_mL: float
    gastric_solubility_mg_mL: float
    intestinal_solubility_mg_mL: float
    solubility_ratio_gastric_intestinal: float
    notes: str


def _hh_solubility(so: float, ph: float, pka: float, ionization_type: str) -> float:
    """Compute Henderson-Hasselbalch solubility capped at 1000 mg/mL."""
    if ionization_type == "acid":
        s = so * (1.0 + 10.0 ** (ph - pka))
    elif ionization_type == "base":
        s = so * (1.0 + 10.0 ** (pka - ph))
    else:  # neutral
        s = so
    return min(s, 1000.0)


def predict_salt_solubility(
    drug_name: str,
    pka: float,
    ionization_type: str,
    salt_form: str = "HCl",
    logp: float = 2.0,
    t_celsius: float = 25.0,
) -> DrugSaltSolubilityResult:
    """Predict pH-dependent solubility of a drug salt.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    pka:
        Acid dissociation constant (0-14).
    ionization_type:
        One of 'acid', 'base', 'neutral'.
    salt_form:
        Salt counter-ion form (e.g. 'HCl', 'Na', 'free_acid').
    logp:
        Lipophilicity (log P); higher -> lower intrinsic solubility.
    t_celsius:
        Temperature in Celsius (10-60).

    Returns
    -------
    DrugSaltSolubilityResult
    """
    # Validation
    if not (0.0 <= pka <= 14.0):
        raise ValueError(f"pka must be in [0, 14], got {pka}")
    if ionization_type not in _VALID_IONIZATION_TYPES:
        raise ValueError(
            f"ionization_type must be one of {sorted(_VALID_IONIZATION_TYPES)}, "
            f"got '{ionization_type}'"
        )
    if not (10.0 <= t_celsius <= 60.0):
        raise ValueError(f"t_celsius must be in [10, 60], got {t_celsius}")
    if salt_form not in _VALID_SALT_FORMS:
        raise ValueError(f"salt_form must be one of {sorted(_VALID_SALT_FORMS)}, got '{salt_form}'")

    # Intrinsic solubility (neutral form) — Yalkowsky approximation
    so_base = 10.0 ** (-0.7 * logp + 0.44)
    # Temperature correction
    so = so_base * 10.0 ** (0.02 * (t_celsius - 25.0))

    # pH grid: 1.0 to 9.0 in 0.5 steps = 17 values
    ph_grid = [1.0 + 0.5 * k for k in range(17)]

    sol_grid = [_hh_solubility(so, ph, pka, ionization_type) for ph in ph_grid]

    max_sol = max(sol_grid)
    max_ph = ph_grid[sol_grid.index(max_sol)]
    min_sol = min(sol_grid)

    # Gastric (pH 2.0) and intestinal (pH 6.8) computed directly
    gastric_sol = _hh_solubility(so, 2.0, pka, ionization_type)
    intestinal_sol = _hh_solubility(so, 6.8, pka, ionization_type)

    ratio = gastric_sol / intestinal_sol if intestinal_sol > 0.0 else 0.0

    notes = (
        f"logP={logp:.2f}, T={t_celsius:.1f}C; "
        f"So={so:.4f} mg/mL; "
        f"BCS solubility class determined by intestinal pH solubility"
    )

    return DrugSaltSolubilityResult(
        drug_name=drug_name,
        salt_form=salt_form,
        pka=pka,
        ionization_type=ionization_type,
        intrinsic_solubility_mg_mL=so,
        ph_values=tuple(ph_grid),
        solubility_mg_mL=tuple(sol_grid),
        max_solubility_mg_mL=max_sol,
        max_solubility_ph=max_ph,
        min_solubility_mg_mL=min_sol,
        gastric_solubility_mg_mL=gastric_sol,
        intestinal_solubility_mg_mL=intestinal_sol,
        solubility_ratio_gastric_intestinal=ratio,
        notes=notes,
    )


def compare_salt_forms(
    drug_name: str,
    pka: float,
    ionization_type: str,
    logp: float,
    salt_forms: list,
) -> list:
    """Compare solubility across multiple salt forms.

    Returns results sorted by intestinal_solubility_mg_mL descending.
    """
    results = []
    for sf in salt_forms:
        result = predict_salt_solubility(
            drug_name=drug_name,
            pka=pka,
            ionization_type=ionization_type,
            salt_form=sf,
            logp=logp,
        )
        results.append(result)
    results.sort(key=lambda r: r.intestinal_solubility_mg_mL, reverse=True)
    return results
