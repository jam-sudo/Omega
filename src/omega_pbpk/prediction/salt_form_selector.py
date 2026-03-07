"""Drug salt form selection.

Predicts physicochemical properties (solubility, stability, hygroscopicity,
manufacturing feasibility) of common pharmaceutical salt forms and selects
the optimal counterion for a given drug molecule.

Supports ionization types:
  * base  — acidic counterions (HCl, H2SO4, HBr, citrate, maleate, tartrate,
             mesylate, phosphate, free_base)
  * acid  — basic counterions (Na, K, Ca, Mg, lysine, meglumine, free_acid)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SaltFormResult:
    """Predicted properties for a drug salt form."""

    drug_name: str
    salt_form: str  # e.g., "HCl", "Na", "free_base", "free_acid"
    counterion: str
    drug_pka: float
    salt_ph: float  # predicted solution pH of 1 % w/v solution
    solubility_fold_increase: float  # vs free form
    dissolution_rate_factor: float
    hygroscopicity: str  # "low", "moderate", "high"
    stability_score: float  # 0–100 (100 = most stable)
    recommended: bool
    manufacturing_feasibility: str  # "easy", "moderate", "difficult"
    notes: str


# ---------------------------------------------------------------------------
# Salt lookup tables
# ---------------------------------------------------------------------------

# Each entry: (counterion, base_ph, sol_factor_fn, hygro, stability, mfg)
# sol_factor_fn is a callable (logp: float) -> float


def _base_salts(logp: float, drug_pka: float) -> dict[str, dict]:
    """Return property dict for all counterions appropriate for basic drugs."""
    return {
        "HCl": {
            "counterion": "Cl-",
            "salt_ph": 4.0,
            "solubility_factor": 2.0 + logp,
            "hygroscopicity": "moderate",
            "stability_score": 80.0,
            "manufacturing_feasibility": "easy",
        },
        "H2SO4": {
            "counterion": "SO4²-",
            "salt_ph": 3.0,
            "solubility_factor": 1.5 + logp * 0.8,
            "hygroscopicity": "high",
            "stability_score": 60.0,
            "manufacturing_feasibility": "moderate",
        },
        "HBr": {
            "counterion": "Br-",
            "salt_ph": 3.5,
            "solubility_factor": 2.0 + logp,
            "hygroscopicity": "moderate",
            "stability_score": 78.0,
            "manufacturing_feasibility": "easy",
        },
        "citrate": {
            "counterion": "citrate³-",
            "salt_ph": 4.5,
            "solubility_factor": 3.0,
            "hygroscopicity": "moderate",
            "stability_score": 70.0,
            "manufacturing_feasibility": "moderate",
        },
        "maleate": {
            "counterion": "maleate²-",
            "salt_ph": 4.0,
            "solubility_factor": 2.5,
            "hygroscopicity": "low",
            "stability_score": 75.0,
            "manufacturing_feasibility": "moderate",
        },
        "tartrate": {
            "counterion": "tartrate²-",
            "salt_ph": 5.0,
            "solubility_factor": 2.0,
            "hygroscopicity": "moderate",
            "stability_score": 72.0,
            "manufacturing_feasibility": "easy",
        },
        "mesylate": {
            "counterion": "methanesulfonate-",
            "salt_ph": 3.5,
            "solubility_factor": 3.0 + logp * 0.5,
            "hygroscopicity": "low",
            "stability_score": 85.0,
            "manufacturing_feasibility": "easy",
        },
        "phosphate": {
            "counterion": "HPO4²-",
            "salt_ph": 5.5,
            "solubility_factor": 2.0,
            "hygroscopicity": "high",
            "stability_score": 65.0,
            "manufacturing_feasibility": "moderate",
        },
        "free_base": {
            "counterion": "none",
            # pH above pKa → neutral / freebase form
            "salt_ph": 7.0 + drug_pka,
            "solubility_factor": 1.0,
            "hygroscopicity": "low",
            "stability_score": 90.0,
            "manufacturing_feasibility": "easy",
        },
    }


def _acid_salts(logp: float, drug_pka: float) -> dict[str, dict]:
    """Return property dict for all counterions appropriate for acidic drugs."""
    return {
        "Na": {
            "counterion": "Na+",
            "salt_ph": 9.0,
            "solubility_factor": 3.0 + logp * 0.5,
            "hygroscopicity": "high",
            "stability_score": 70.0,
            "manufacturing_feasibility": "easy",
        },
        "K": {
            "counterion": "K+",
            "salt_ph": 9.5,
            "solubility_factor": 3.0 + logp * 0.5,
            "hygroscopicity": "high",
            "stability_score": 68.0,
            "manufacturing_feasibility": "easy",
        },
        "Ca": {
            "counterion": "Ca²+",
            "salt_ph": 8.0,
            "solubility_factor": 1.5,
            "hygroscopicity": "low",
            "stability_score": 80.0,
            "manufacturing_feasibility": "moderate",
        },
        "Mg": {
            "counterion": "Mg²+",
            "salt_ph": 8.5,
            "solubility_factor": 1.5,
            "hygroscopicity": "low",
            "stability_score": 78.0,
            "manufacturing_feasibility": "moderate",
        },
        "lysine": {
            "counterion": "lysinium+",
            "salt_ph": 9.0,
            "solubility_factor": 2.5,
            "hygroscopicity": "moderate",
            "stability_score": 75.0,
            "manufacturing_feasibility": "moderate",
        },
        "meglumine": {
            "counterion": "meglumine+",
            "salt_ph": 9.5,
            "solubility_factor": 2.0,
            "hygroscopicity": "moderate",
            "stability_score": 78.0,
            "manufacturing_feasibility": "moderate",
        },
        "free_acid": {
            "counterion": "none",
            "salt_ph": drug_pka - 2.0,
            "solubility_factor": 1.0,
            "hygroscopicity": "low",
            "stability_score": 90.0,
            "manufacturing_feasibility": "easy",
        },
    }


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

_CLAMP_SOL_MIN = 1.0
_CLAMP_SOL_MAX = 20.0


def _clamp_sol(val: float) -> float:
    return max(_CLAMP_SOL_MIN, min(_CLAMP_SOL_MAX, val))


def _is_recommended(sol: float, hygro: str, stability: float) -> bool:
    return stability >= 70.0 and hygro != "high" and sol >= 2.0


def _build_result(
    drug_name: str,
    salt_form: str,
    props: dict,
    drug_pka: float,
) -> SaltFormResult:
    sol = _clamp_sol(props["solubility_factor"])
    diss = math.sqrt(sol)
    hygro = props["hygroscopicity"]
    stab = props["stability_score"]
    rec = _is_recommended(sol, hygro, stab)
    notes = (
        f"Salt: {salt_form}, counterion: {props['counterion']}, "
        f"sol_fold={sol:.2f}, stability={stab:.0f}"
    )
    return SaltFormResult(
        drug_name=drug_name,
        salt_form=salt_form,
        counterion=props["counterion"],
        drug_pka=drug_pka,
        salt_ph=props["salt_ph"],
        solubility_fold_increase=sol,
        dissolution_rate_factor=diss,
        hygroscopicity=hygro,
        stability_score=stab,
        recommended=rec,
        manufacturing_feasibility=props["manufacturing_feasibility"],
        notes=notes,
    )


def predict_salt_properties(
    drug_name: str,
    drug_pka: float,
    drug_logp: float,
    ionization_type: str,
    salt_form: str,
) -> SaltFormResult:
    """Predict physicochemical properties for a specific salt form.

    Parameters
    ----------
    drug_name:
        Identifier for the drug molecule.
    drug_pka:
        Ionization pKa.  Must be in [0, 14].
    drug_logp:
        Octanol-water partition coefficient (used in solubility corrections).
    ionization_type:
        ``"acid"`` or ``"base"``.
    salt_form:
        Desired salt counterion (see module docstring for valid names).

    Returns
    -------
    SaltFormResult
    """
    # --- validation ---------------------------------------------------------
    if not (0.0 <= drug_pka <= 14.0):
        raise ValueError(f"drug_pka must be in [0, 14]; got {drug_pka}")
    if ionization_type not in ("acid", "base"):
        raise ValueError(f"ionization_type must be 'acid' or 'base'; got '{ionization_type}'")

    if ionization_type == "base":
        table = _base_salts(drug_logp, drug_pka)
        fallback_key = "free_base"
    else:
        table = _acid_salts(drug_logp, drug_pka)
        fallback_key = "free_acid"

    props = table.get(salt_form, table[fallback_key])
    return _build_result(drug_name, salt_form, props, drug_pka)


def select_optimal_salt(
    drug_name: str,
    drug_pka: float,
    drug_logp: float,
    ionization_type: str,
) -> list[SaltFormResult]:
    """Screen all applicable salt forms and rank by suitability.

    Parameters
    ----------
    drug_name, drug_pka, drug_logp, ionization_type:
        Same as :func:`predict_salt_properties`.

    Returns
    -------
    list[SaltFormResult]
        Sorted: recommended salts first (by stability_score descending),
        then non-recommended (by stability_score descending).
    """
    # --- validation ---------------------------------------------------------
    if not (0.0 <= drug_pka <= 14.0):
        raise ValueError(f"drug_pka must be in [0, 14]; got {drug_pka}")
    if ionization_type not in ("acid", "base"):
        raise ValueError(f"ionization_type must be 'acid' or 'base'; got '{ionization_type}'")

    if ionization_type == "base":
        table = _base_salts(drug_logp, drug_pka)
    else:
        table = _acid_salts(drug_logp, drug_pka)

    results = [_build_result(drug_name, sf, props, drug_pka) for sf, props in table.items()]

    # Sort: recommended first, then by stability descending
    results.sort(key=lambda r: (-int(r.recommended), -r.stability_score))
    return results
