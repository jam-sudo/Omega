"""Tissue:plasma partition coefficient (Kp) estimation methods.

Provides Kp estimation from physicochemical properties (logP, pKa, compound
type) when experimentally-determined or calibrated Kp values are not available.

Methods:
    heuristic  — Simplified Poulin & Theil (2002) empirical scaling.
    rodgers_rowland — Mechanistic tissue binding, Rodgers & Rowland (2006).

References:
    Poulin P, Theil FP. J Pharm Sci. 2002;91(6):1358-70.
    Rodgers T, Rowland M. J Pharm Sci. 2006;95(6):1238-57.
"""

from __future__ import annotations

from collections.abc import Callable

# ---------------------------------------------------------------------------
# Tissue composition data — Rodgers & Rowland (2006), Tables 1–2
# fn = fractional neutral lipid, fp = fractional phospholipid,
# fw = fractional water, pH = intracellular/interstitial pH
# ---------------------------------------------------------------------------

TISSUE_COMPOSITION: dict[str, dict[str, float]] = {
    "muscle": {"fn": 0.0238, "fp": 0.0072, "fw": 0.760, "pH": 7.00},
    "fat": {"fn": 0.7021, "fp": 0.0022, "fw": 0.150, "pH": 7.00},
    "adipose": {"fn": 0.7021, "fp": 0.0022, "fw": 0.150, "pH": 7.00},
    "brain": {"fn": 0.0391, "fp": 0.0550, "fw": 0.620, "pH": 7.00},
    "kidney": {"fn": 0.0121, "fp": 0.0242, "fw": 0.783, "pH": 7.00},
    "liver": {"fn": 0.0348, "fp": 0.0252, "fw": 0.751, "pH": 7.00},
    "gut_wall": {"fn": 0.0163, "fp": 0.0185, "fw": 0.718, "pH": 7.00},
    "lung": {"fn": 0.0030, "fp": 0.0128, "fw": 0.811, "pH": 7.00},
    "heart": {"fn": 0.0117, "fp": 0.0166, "fw": 0.758, "pH": 7.00},
    "spleen": {"fn": 0.0077, "fp": 0.0113, "fw": 0.788, "pH": 7.00},
    "skin": {"fn": 0.0284, "fp": 0.0111, "fw": 0.718, "pH": 7.00},
    "bone": {"fn": 0.0174, "fp": 0.0010, "fw": 0.439, "pH": 7.00},
    "pancreas": {"fn": 0.0348, "fp": 0.0252, "fw": 0.751, "pH": 7.00},
    "thymus": {"fn": 0.0132, "fp": 0.0100, "fw": 0.700, "pH": 7.00},
    "reproductive": {"fn": 0.0132, "fp": 0.0100, "fw": 0.700, "pH": 7.00},
    "rest": {"fn": 0.0132, "fp": 0.0100, "fw": 0.700, "pH": 7.00},
    "portal_vein": {"fn": 0.0030, "fp": 0.0030, "fw": 0.940, "pH": 7.40},
}

PLASMA_COMPOSITION: dict[str, float] = {
    "fn": 0.0023,
    "fp": 0.0199,
    "fw": 0.945,
    "pH": 7.40,
}

# ---------------------------------------------------------------------------
# Tissue-specific lipid/water factors — simplified Poulin & Theil (2002)
# ---------------------------------------------------------------------------

_TISSUE_FACTORS: dict[str, float] = {
    "adipose": 2.5,
    "fat": 2.5,
    "brain": 1.3,
    "skin": 1.15,
    "bone": 0.8,
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


# ---------------------------------------------------------------------------
# Ionisation helpers
# ---------------------------------------------------------------------------


def _ionization_ratio(
    pka: float | None, pH_tissue: float, pH_plasma: float, compound_type: str
) -> float:
    """Compute the tissue/plasma ionisation ratio.

    For acids:  X = 1 + 10^(pH − pKa)  (total / unionised)
    For bases:  X = 1 + 10^(pKa − pH)
    Ratio = X_tissue / X_plasma
    """
    if pka is None or compound_type == "neutral":
        return 1.0
    if compound_type == "acid":
        x_t = 1.0 + 10.0 ** (pH_tissue - pka)
        x_p = 1.0 + 10.0 ** (pH_plasma - pka)
    elif compound_type in ("base", "zwitterion"):
        x_t = 1.0 + 10.0 ** (pka - pH_tissue)
        x_p = 1.0 + 10.0 ** (pka - pH_plasma)
    else:
        return 1.0
    return x_t / max(x_p, 1e-12)


# ---------------------------------------------------------------------------
# Method 1 — Simplified Poulin & Theil heuristic
# ---------------------------------------------------------------------------

# Accepted drug_type aliases → canonical compound_type
_DRUG_TYPE_MAP: dict[str, str] = {
    "neutral": "neutral",
    "monoprotic_acid": "acid",
    "monoprotic_base": "base",
    "acid": "acid",
    "base": "base",
    "diprotic": "zwitterion",
    "zwitterion": "zwitterion",
}


def heuristic_kp(
    logP: float,
    pka: float | None = None,
    drug_type: str = "neutral",
    tissue_name: str = "rest",
    fup: float = 0.5,
    *,
    compound_type: str | None = None,
) -> float:
    """Estimate Kp for a single tissue (simplified Poulin & Theil).

    Args:
        logP: Octanol-water log partition coefficient.
        pka: Primary pKa (strongest acidic or basic centre).
        drug_type: Legacy ionisation class.
        tissue_name: Target tissue name.
        fup: Fraction unbound in plasma (0–1).
        compound_type: Canonical ionisation class (overrides drug_type if given).

    Returns:
        Estimated Kp (always ≥ 0.1).
    """
    ct = compound_type or _DRUG_TYPE_MAP.get(drug_type, "neutral")

    alpha = 0.5
    base = 1.0 + fup * (10.0 ** (logP * alpha) - 1.0)
    base = max(base, 0.3)

    if pka is not None:
        if ct == "base":
            fraction_ionised = 1.0 / (1.0 + 10.0 ** (7.4 - pka))
            base *= 1.0 + 0.3 * fraction_ionised
        elif ct == "acid":
            fraction_ionised = 1.0 / (1.0 + 10.0 ** (pka - 7.4))
            base *= 1.0 - 0.15 * fraction_ionised

    tissue_factor = _TISSUE_FACTORS.get(tissue_name, 1.0)
    kp = base * tissue_factor

    if fup < 0.05 and tissue_name not in ("liver", "kidney", "lung"):
        kp *= max(fup / 0.05, 0.3)

    return float(max(round(kp, 4), 0.1))


# ---------------------------------------------------------------------------
# Method 2 — Rodgers & Rowland (2006) mechanistic Kp
# ---------------------------------------------------------------------------


def rodgers_rowland_kp(
    logP: float,
    pka: float | None = None,
    compound_type: str = "neutral",
    tissue_name: str = "rest",
    fup: float = 0.5,
    *,
    drug_type: str | None = None,
) -> float:
    """Estimate Kp using the Rodgers & Rowland (2006) method.

    For neutrals and weak acids: Eq. 4 (neutral/acid method).
    For bases and zwitterions: Eq. 5 (base method, enhanced phospholipid binding).

    Args:
        logP: Octanol-water log partition coefficient.
        pka: Primary pKa value.
        compound_type: One of 'neutral', 'acid', 'base', 'zwitterion'.
        tissue_name: Target tissue name.
        fup: Fraction unbound in plasma (0–1).
        drug_type: Legacy ionisation class (used if compound_type not given).

    Returns:
        Estimated Kp (always ≥ 0.01).
    """
    ct = compound_type
    if drug_type is not None and compound_type == "neutral":
        ct = _DRUG_TYPE_MAP.get(drug_type, compound_type)

    tissue = TISSUE_COMPOSITION.get(tissue_name)
    if tissue is None:
        return 1.0

    plasma = PLASMA_COMPOSITION
    p_ow = 10.0**logP

    fw_t = tissue["fw"]
    fn_t = tissue["fn"]
    fp_t = tissue["fp"]
    pH_t = tissue["pH"]

    fw_p = plasma["fw"]
    fn_p = plasma["fn"]
    fp_p = plasma["fp"]
    pH_p = plasma["pH"]

    # Lipid partition terms
    lipid_t = fn_t * p_ow + fp_t * (0.3 * p_ow + 0.7)
    lipid_p = fn_p * p_ow + fp_p * (0.3 * p_ow + 0.7)

    ion = _ionization_ratio(pka, pH_t, pH_p, ct)

    if ct in ("neutral", "acid"):
        # Rodgers & Rowland Eq. 4 — neutral / acid
        # Water partitioning scales with ionisation; lipid partitioning uses
        # neutral species only (P, not D) but is attenuated by ionisation.
        kpu = (fw_t * ion + lipid_t) / max(fw_p + lipid_p, 1e-12)
    else:
        # Rodgers & Rowland Eq. 5 — base / zwitterion
        # Ionised basic species interact with acidic phospholipids in tissue,
        # giving enhanced tissue distribution.
        kpu = (fw_t * ion + lipid_t + fp_t * max(ion - 1.0, 0.0)) / max(fw_p + lipid_p, 1e-12)

    return float(max(round(kpu * fup, 4), 0.01))


# ---------------------------------------------------------------------------
# Method 3 — Poulin & Theil (2002) mechanistic Kp
# ---------------------------------------------------------------------------


def poulin_theil_kp(
    logP: float,
    pka: float | None = None,
    compound_type: str = "neutral",
    tissue_name: str = "rest",
    fup: float = 0.5,
    *,
    drug_type: str | None = None,
) -> float:
    """Estimate Kp using the Poulin & Theil (2002) method.

    Kp = Kpu × fup where Kpu is the unbound tissue:plasma partition.
    Uses Eq. 4 of Rodgers & Rowland (identical for neutral/acid compounds;
    no extra phospholipid term for bases).

    Args:
        logP: Octanol-water log partition coefficient.
        pka: Primary pKa value.
        compound_type: One of 'neutral', 'acid', 'base', 'zwitterion'.
        tissue_name: Target tissue name.
        fup: Fraction unbound in plasma (0–1).  Kp scales linearly with fup.
        drug_type: Legacy ionisation class (used if compound_type not given).

    Returns:
        Estimated Kp (always ≥ 0.01).
    """
    ct = compound_type
    if drug_type is not None and compound_type == "neutral":
        ct = _DRUG_TYPE_MAP.get(drug_type, compound_type)

    tissue = TISSUE_COMPOSITION.get(tissue_name)
    if tissue is None:
        return 1.0

    plasma = PLASMA_COMPOSITION
    p_ow = 10.0**logP

    fw_t = tissue["fw"]
    fn_t = tissue["fn"]
    fp_t = tissue["fp"]
    pH_t = tissue["pH"]

    fw_p = plasma["fw"]
    fn_p = plasma["fn"]
    fp_p = plasma["fp"]
    pH_p = plasma["pH"]

    lipid_t = fn_t * p_ow + fp_t * (0.3 * p_ow + 0.7)
    lipid_p = fn_p * p_ow + fp_p * (0.3 * p_ow + 0.7)

    ion = _ionization_ratio(pka, pH_t, pH_p, ct)
    kpu = (fw_t * ion + lipid_t) / max(fw_p + lipid_p, 1e-12)

    return float(max(round(kpu * fup, 4), 0.01))


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

_KP_METHODS: dict[str, Callable[..., float]] = {
    "heuristic": heuristic_kp,
    "poulin_theil": poulin_theil_kp,
    "rodgers_rowland": rodgers_rowland_kp,
}

#: Sorted tuple of available Kp method names.
KP_METHOD_NAMES: tuple[str, ...] = tuple(sorted(_KP_METHODS))


def get_partition_method(name: str) -> Callable[..., float]:
    """Return the Kp estimation function for the given method name.

    Available methods: 'heuristic' (default), 'rodgers_rowland'.
    """
    fn = _KP_METHODS.get(name)
    if fn is None:
        raise ValueError(f"Unknown partition method '{name}'. Available: {sorted(_KP_METHODS)}")
    return fn


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


def estimate_all_kp(
    logP: float,
    pka: float | None = None,
    drug_type: str = "neutral",
    fup: float = 0.5,
    tissues: list[str] | None = None,
    method: str = "heuristic",
    compound_type: str | None = None,
) -> dict[str, float]:
    """Estimate Kp for all standard PBPK tissues.

    Args:
        method: Kp method — 'heuristic' (default), 'poulin_theil', or
            'rodgers_rowland'.
        compound_type: Canonical ionisation class; overrides drug_type when
            given.
    """
    if tissues is None:
        tissues = list(_TISSUE_FACTORS.keys())
    ct = compound_type or drug_type
    kp_fn = get_partition_method(method)
    return {t: kp_fn(logP=logP, pka=pka, compound_type=ct, tissue_name=t, fup=fup) for t in tissues}


def log_kp_summary(kp_values: dict[str, float]) -> str:
    """Format Kp estimates as a readable summary string."""
    lines = ["Heuristic Kp estimates:"]
    for tissue, kp in sorted(kp_values.items(), key=lambda x: -x[1]):
        lines.append(f"  {tissue:18s}  Kp = {kp:.3f}")
    return "\n".join(lines)
