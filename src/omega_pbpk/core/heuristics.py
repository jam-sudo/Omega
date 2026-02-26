"""Heuristic tissue:plasma partition coefficient (Kp) estimation.

Provides fallback Kp estimation from physicochemical properties (logP, pKa,
drug_type) when experimentally-determined or calibrated Kp values are not
available.  The method uses empirical scaling from octanol-water partition
and ionisation state, with tissue-specific adjustment factors.

References:
    Poulin & Theil (2002) — tissue composition-based approach (simplified).
    Rodgers & Rowland (2005, 2006) — mechanistic tissue binding model (stub).
"""

from __future__ import annotations

# Tissue-specific lipid/water composition factors (simplified Poulin & Theil)
_TISSUE_FACTORS: dict[str, float] = {
    # High-lipid tissues
    "adipose": 2.5,
    "fat": 2.5,
    # Moderate-lipid tissues
    "brain": 1.3,
    "skin": 1.15,
    "bone": 0.8,
    # Well-perfused / standard
    "liver": 1.1,
    "kidney": 1.05,
    "heart": 1.0,
    "lung": 0.95,
    "muscle": 0.95,
    "spleen": 1.0,
    "gut_wall": 1.0,
    "pancreas": 0.95,
    "thymus": 0.9,
    "reproductive": 0.9,
    "rest": 0.9,
    "portal_vein": 1.0,
}


def heuristic_kp(
    logP: float,
    pka: float | None = None,
    drug_type: str = "neutral",
    tissue_name: str = "rest",
    fup: float = 0.5,
) -> float:
    """Estimate Kp for a single tissue using physicochemical properties.

    Args:
        logP: Octanol-water log partition coefficient.
        pka: Primary pKa (strongest acidic or basic centre).
        drug_type: One of 'neutral', 'monoprotic_acid', 'monoprotic_base', 'diprotic'.
        tissue_name: Target tissue name matching organ names in WholeBodyPBPK.
        fup: Fraction unbound in plasma (0–1).

    Returns:
        Estimated Kp (always ≥ 0.1).
    """
    # Base partitioning from lipophilicity
    # Kp_base ≈ 1 + fup * 10^(logP * alpha) where alpha dampens extreme logP
    alpha = 0.5
    base = 1.0 + fup * (10.0 ** (logP * alpha) - 1.0)
    base = max(base, 0.3)

    # Ionisation correction at physiological pH 7.4
    if pka is not None:
        if drug_type == "monoprotic_base":
            # Bases are more ionised → higher tissue binding in acidic tissues
            fraction_ionised = 1.0 / (1.0 + 10.0 ** (7.4 - pka))
            base *= 1.0 + 0.3 * fraction_ionised
        elif drug_type == "monoprotic_acid":
            # Acids are less ionised at pH 7.4 for pKa < 7.4
            fraction_ionised = 1.0 / (1.0 + 10.0 ** (pka - 7.4))
            base *= 1.0 - 0.15 * fraction_ionised

    # Tissue-specific adjustment
    tissue_factor = _TISSUE_FACTORS.get(tissue_name, 1.0)
    kp = base * tissue_factor

    # Protein binding correction (high binding → lower tissue distribution)
    # Very highly bound drugs (fup < 0.05) distribute less to non-binding tissues
    if fup < 0.05 and tissue_name not in ("liver", "kidney", "lung"):
        kp *= max(fup / 0.05, 0.3)

    return float(max(round(kp, 4), 0.1))


def estimate_all_kp(
    logP: float,
    pka: float | None = None,
    drug_type: str = "neutral",
    fup: float = 0.5,
    tissues: list[str] | None = None,
) -> dict[str, float]:
    """Estimate Kp for all standard PBPK tissues.

    Args:
        logP: Octanol-water log partition coefficient.
        pka: Primary pKa value.
        drug_type: Ionisation class.
        fup: Fraction unbound in plasma.
        tissues: Optional list of tissue names. Defaults to all standard organs.

    Returns:
        Dict mapping tissue name → estimated Kp.
    """
    if tissues is None:
        tissues = list(_TISSUE_FACTORS.keys())

    return {t: heuristic_kp(logP, pka, drug_type, t, fup) for t in tissues}


def log_kp_summary(kp_values: dict[str, float]) -> str:
    """Format Kp estimates as a readable summary string."""
    lines = ["Heuristic Kp estimates:"]
    for tissue, kp in sorted(kp_values.items(), key=lambda x: -x[1]):
        lines.append(f"  {tissue:18s}  Kp = {kp:.3f}")
    return "\n".join(lines)
