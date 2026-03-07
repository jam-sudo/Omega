"""Phase 1111 — Henderson-Hasselbalch pH-dependent solubility prediction."""

from dataclasses import dataclass


@dataclass(frozen=True)
class PHSolubilityResult:
    drug_name: str
    s0_mg_mL: float
    pka: float
    ionization_type: str
    ph_values: tuple
    solubility_mg_mL: tuple
    max_solubility_mg_mL: float
    min_solubility_mg_mL: float
    ph_at_max_solubility: float
    solubility_ratio_max_min: float
    notes: str


def predict_ph_solubility(
    drug_name: str,
    s0_mg_mL: float,
    pka: float,
    ionization_type: str,
    ph_range_start: float = 1.0,
    ph_range_end: float = 10.0,
    n_points: int = 19,
) -> PHSolubilityResult:
    """Predict pH-dependent solubility using Henderson-Hasselbalch equation.

    Args:
        drug_name: Name of the drug compound.
        s0_mg_mL: Intrinsic (neutral form) solubility in mg/mL.
        pka: Acid dissociation constant.
        ionization_type: One of "acid", "base", or "neutral".
        ph_range_start: Starting pH (default 1.0).
        ph_range_end: Ending pH (default 10.0).
        n_points: Number of pH points to evaluate (default 19).

    Returns:
        PHSolubilityResult with pH-solubility profile.
    """
    if s0_mg_mL <= 0:
        raise ValueError(f"s0_mg_mL must be > 0, got {s0_mg_mL}")
    if not (0 <= pka <= 14):
        raise ValueError(f"pKa must be in [0, 14], got {pka}")
    if ionization_type not in {"acid", "base", "neutral"}:
        raise ValueError(
            f"ionization_type must be 'acid', 'base', or 'neutral', got '{ionization_type}'"
        )
    if ph_range_start >= ph_range_end:
        raise ValueError(
            f"ph_range_start ({ph_range_start}) must be < ph_range_end ({ph_range_end})"
        )
    if not (0 <= ph_range_start <= 14):
        raise ValueError(f"ph_range_start must be in [0, 14], got {ph_range_start}")
    if not (0 <= ph_range_end <= 14):
        raise ValueError(f"ph_range_end must be in [0, 14], got {ph_range_end}")
    if n_points < 2:
        raise ValueError(f"n_points must be >= 2, got {n_points}")

    step = (ph_range_end - ph_range_start) / (n_points - 1)
    ph_values = tuple(ph_range_start + i * step for i in range(n_points))

    solubility_values = []
    for ph in ph_values:
        if ionization_type == "acid":
            s = s0_mg_mL * (1.0 + 10.0 ** (ph - pka))
        elif ionization_type == "base":
            s = s0_mg_mL * (1.0 + 10.0 ** (pka - ph))
        else:
            s = s0_mg_mL
        solubility_values.append(s)

    solubility_mg_mL = tuple(solubility_values)
    max_sol = max(solubility_values)
    min_sol = min(solubility_values)
    max_idx = solubility_values.index(max_sol)
    ph_at_max = ph_values[max_idx]
    ratio = max_sol / min_sol if min_sol > 0 else float("inf")

    if ionization_type == "acid":
        notes = (
            f"Acid: solubility increases with pH as drug ionizes to anionic form. pKa = {pka:.2f}."
        )
    elif ionization_type == "base":
        notes = (
            "Base: solubility increases as pH decreases below pKa due to protonation. "
            f"pKa = {pka:.2f}."
        )
    else:
        notes = "Neutral drug: solubility is pH-independent."

    return PHSolubilityResult(
        drug_name=drug_name,
        s0_mg_mL=s0_mg_mL,
        pka=pka,
        ionization_type=ionization_type,
        ph_values=ph_values,
        solubility_mg_mL=solubility_mg_mL,
        max_solubility_mg_mL=max_sol,
        min_solubility_mg_mL=min_sol,
        ph_at_max_solubility=ph_at_max,
        solubility_ratio_max_min=ratio,
        notes=notes,
    )


def find_max_solubility_ph(result: PHSolubilityResult) -> float:
    """Return the pH at which solubility is maximum.

    Args:
        result: PHSolubilityResult from predict_ph_solubility.

    Returns:
        pH value corresponding to maximum solubility.
    """
    return result.ph_at_max_solubility
