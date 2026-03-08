"""Phase 1009: Predict tissue-to-plasma partition coefficients (Kp) using
a mechanistic Rodgers-Rowland simplified model."""

from __future__ import annotations

from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Tissue composition table (Poulin & Theil 2000)
# Keys: tissue name -> (f_nl, f_pl, V_L_per_kg)
# ---------------------------------------------------------------------------
_TISSUE_COMPOSITION: dict[str, tuple[float, float, float]] = {
    "muscle": (0.016, 0.016, 0.40),
    "liver": (0.038, 0.025, 0.026),
    "kidney": (0.012, 0.016, 0.004),
    "lung": (0.022, 0.013, 0.005),
    "brain": (0.039, 0.040, 0.020),
    "fat": (0.853, 0.002, 0.190),
    "heart": (0.016, 0.016, 0.005),
    "skin": (0.030, 0.006, 0.110),
}

_VALID_IONIZATION = {"acid", "base", "neutral"}
_PLASMA_V_L_PER_KG = 0.043
_KP_MIN = 0.01
_KP_MAX = 1000.0


@dataclass(frozen=True)
class PartitionCoefficientResult:
    """Result of tissue-to-plasma partition coefficient prediction."""

    drug_name: str
    logp: float
    pka: float
    ionization_type: str
    fup: float
    kp_muscle: float
    kp_liver: float
    kp_kidney: float
    kp_lung: float
    kp_brain: float
    kp_fat: float
    kp_heart: float
    kp_skin: float
    vss_L_per_kg: float
    highest_kp_tissue: str
    notes: str


def predict_partition_coefficients(
    drug_name: str,
    logp: float,
    pka: float,
    ionization_type: str = "neutral",
    fup: float = 0.5,
) -> PartitionCoefficientResult:
    """Predict tissue-to-plasma Kp values using Rodgers-Rowland simplified model.

    Parameters
    ----------
    drug_name:
        Identifier for the compound.
    logp:
        Octanol-water partition coefficient (log scale).
    pka:
        Acid dissociation constant (0-14).
    ionization_type:
        One of "acid", "base", or "neutral".
    fup:
        Fraction unbound in plasma (0 < fup <= 1).

    Returns
    -------
    PartitionCoefficientResult
    """
    # --- Validation ---
    if not (0.0 < fup <= 1.0):
        raise ValueError(f"fup must be in (0, 1], got {fup}")
    if not (0.0 <= pka <= 14.0):
        raise ValueError(f"pka must be in [0, 14], got {pka}")
    if ionization_type not in _VALID_IONIZATION:
        raise ValueError(
            f"ionization_type must be one of {_VALID_IONIZATION}, got {ionization_type!r}"
        )

    # --- Lipid partition affinities ---
    ka_nl = 10.0 ** (0.9 * logp - 0.4)

    if ionization_type == "neutral":
        ka_pl = 10.0 ** (0.5 * logp)
    elif ionization_type == "base":
        ka_pl = 10.0 ** (0.5 * logp + 0.8)
    else:  # acid
        ka_pl = 10.0 ** (0.5 * logp - 0.3)

    # --- Per-tissue Kp calculation ---
    kp_values: dict[str, float] = {}
    for tissue, (f_nl, f_pl, _v) in _TISSUE_COMPOSITION.items():
        raw_kp = (f_nl * ka_nl + f_pl * ka_pl + 0.01) / fup
        kp_values[tissue] = max(_KP_MIN, min(_KP_MAX, raw_kp))

    # --- Vss ---
    vss = _PLASMA_V_L_PER_KG
    for tissue, (_f_nl, _f_pl, v_tissue) in _TISSUE_COMPOSITION.items():
        vss += v_tissue * kp_values[tissue]

    # --- Highest Kp tissue ---
    highest_kp_tissue = max(kp_values, key=lambda t: kp_values[t])

    notes = (
        f"Rodgers-Rowland simplified; ionization={ionization_type}; "
        f"ka_nl={ka_nl:.3f}; ka_pl={ka_pl:.3f}"
    )

    return PartitionCoefficientResult(
        drug_name=drug_name,
        logp=logp,
        pka=pka,
        ionization_type=ionization_type,
        fup=fup,
        kp_muscle=kp_values["muscle"],
        kp_liver=kp_values["liver"],
        kp_kidney=kp_values["kidney"],
        kp_lung=kp_values["lung"],
        kp_brain=kp_values["brain"],
        kp_fat=kp_values["fat"],
        kp_heart=kp_values["heart"],
        kp_skin=kp_values["skin"],
        vss_L_per_kg=vss,
        highest_kp_tissue=highest_kp_tissue,
        notes=notes,
    )


def screen_partition_coefficients(compounds: list) -> list:
    """Screen a list of compound dicts and return results sorted by Vss descending.

    Each element of *compounds* should be a dict with keys matching the
    keyword arguments of :func:`predict_partition_coefficients`.

    Parameters
    ----------
    compounds:
        List of dicts, e.g.::

            [{"drug_name": "A", "logp": 2.0, "pka": 7.4, "ionization_type": "base", "fup": 0.3}]

    Returns
    -------
    list[PartitionCoefficientResult] sorted by vss_L_per_kg descending.
    """
    results = []
    for compound in compounds:
        result = predict_partition_coefficients(**compound)
        results.append(result)
    results.sort(key=lambda r: r.vss_L_per_kg, reverse=True)
    return results


__all__ = [
    "PartitionCoefficientResult",
    "predict_partition_coefficients",
    "screen_partition_coefficients",
]
