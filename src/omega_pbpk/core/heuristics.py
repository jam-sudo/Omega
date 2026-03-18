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

import math
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


def _logD_at_pH(logP: float, pka: float | None, pH: float, compound_type: str) -> float:
    """Distribution coefficient (logD) at a given pH.

    For ionized acids/bases, only the neutral fraction partitions into neutral
    lipids. Using D (not P) for lipid partition terms gives physically correct
    Kp for strong acids like ibuprofen (pKa 4.0) at physiological pH 7.0,
    where D_7.0 ≈ logP − 3.0 (1000-fold reduction vs P_ow).

    For acids:     logD = logP − log10(1 + 10^(pH − pKa))
    For bases:     logD = logP − log10(1 + 10^(pKa − pH))
    For neutral:   logD = logP
    """
    if pka is None or compound_type == "neutral":
        return logP
    if compound_type == "acid":
        return logP - math.log10(1.0 + 10.0 ** (pH - pka))
    elif compound_type in ("base", "zwitterion"):
        return logP - math.log10(1.0 + 10.0 ** (pka - pH))
    return logP


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

    Implements an empirical simplification of Poulin & Theil (2002, J Pharm Sci
    91:1358-70). The full P&T method uses tissue-specific lipid/water volume
    fractions and separate octanol/water partition terms; this method collapses
    them into a single logP-derived scaling with a per-tissue empirical factor
    (_TISSUE_FACTORS above) calibrated against P&T outputs for typical drugs.

    Formula (neutral species baseline):
        base = 1 + fup * (P^alpha − 1),  where P = 10^logP, alpha = 0.5
        Kp   = base * tissue_factor

    Coefficients:
        alpha = 0.5   — Square-root dampening of logP to avoid over-prediction
                        for highly lipophilic compounds; empirically derived to
                        give median 2-fold accuracy across 20 reference drugs.
                        Note: prediction quality degrades for logP > 5 (use
                        rodgers_rowland method instead).
        0.30 (base)   — Lysosomal trapping / acidic phospholipid binding term
                        for basic drugs (cationic amphiphiles); approximates the
                        pH-partitioning enhancement from Rodgers & Rowland (2006)
                        Eq. 5 without requiring full tissue composition data.
        0.15 (acid)   — Ionisation-mediated distribution reduction for acids;
                        ionised species are excluded from lipid phase.
        fup < 0.05    — Highly plasma-bound compounds (fup < 5 %) show restricted
                        tissue distribution in non-eliminating organs. Correction
                        factor fup/0.05 (floor 0.3) approximates published
                        observations (Obach, Drug Metab Dispos 1999;27:1350-9).

    Known limitations:
        - Accuracy degrades for logP > 5 (very lipophilic) and logP < −1
          (very hydrophilic); use rodgers_rowland method for these ranges.
        - Assumes single dominant pKa; zwitterions treated as bases.
        - Does not account for active transport or tissue-specific protein binding.

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

    # Square-root dampening: alpha=0.5 limits over-prediction for lipophilic
    # compounds (empirically calibrated; see docstring).
    alpha = 0.5
    base = 1.0 + fup * (10.0 ** (logP * alpha) - 1.0)
    base = max(base, 0.3)

    if pka is not None:
        if ct == "base":
            # Fraction ionised at plasma pH 7.4 (Henderson-Hasselbalch for bases)
            fraction_ionised = 1.0 / (1.0 + 10.0 ** (7.4 - pka))
            # +30% for lysosomal trapping / acidic phospholipid binding
            base *= 1.0 + 0.3 * fraction_ionised
        elif ct == "acid":
            # Fraction ionised at plasma pH 7.4 (Henderson-Hasselbalch for acids)
            fraction_ionised = 1.0 / (1.0 + 10.0 ** (pka - 7.4))
            # −15% reduction: ionised acid excluded from lipid phase
            base *= 1.0 - 0.15 * fraction_ionised

    tissue_factor = _TISSUE_FACTORS.get(tissue_name, 1.0)

    # For adipose/fat tissues, modulate the tissue factor by lipophilicity.
    # Hydrophilic drugs (logP < 1) distribute poorly into fat; the constant
    # factor of 2.5 over-predicts adipose Kp for these compounds.
    # Sigmoid scaling: factor → 0.3 at logP=-2, ~1.0 at logP=1, 2.5 at logP=4+
    if tissue_name in ("adipose", "fat"):
        # Logistic transition centred at logP=1 (BCS lipophilicity threshold)
        logP_scale = 1.0 / (1.0 + 10.0 ** (1.0 - logP))
        tissue_factor = 0.3 + (tissue_factor - 0.3) * logP_scale

    kp = base * tissue_factor

    # Restricted distribution correction for highly plasma-bound drugs
    # (non-eliminating organs only; liver/kidney/lung retain normal Kp)
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

    fw_t = tissue["fw"]
    fn_t = tissue["fn"]
    fp_t = tissue["fp"]
    pH_t = tissue["pH"]

    fw_p = plasma["fw"]
    fn_p = plasma["fn"]
    fp_p = plasma["fp"]
    pH_p = plasma["pH"]

    # For acids only: use D instead of P for lipid partition terms.
    # Same rationale as berezhkovskiy_kp — see comment there.
    p_ow = 10.0**logP
    if ct == "acid":
        d_ow_t = 10.0 ** _logD_at_pH(logP, pka, pH_t, ct)
        d_ow_p = 10.0 ** _logD_at_pH(logP, pka, pH_p, ct)
    else:
        d_ow_t = p_ow
        d_ow_p = p_ow

    # Lipid partition terms
    lipid_t = fn_t * d_ow_t + fp_t * (0.3 * d_ow_t + 0.7)
    lipid_p = fn_p * d_ow_p + fp_p * (0.3 * d_ow_p + 0.7)

    ion = _ionization_ratio(pka, pH_t, pH_p, ct)

    if ct in ("neutral", "acid"):
        # Rodgers & Rowland Eq. 4 — neutral / acid
        kp = (fw_t * ion + lipid_t) / max(fw_p + lipid_p, 1e-12)
    else:
        # Rodgers & Rowland Eq. 5 — base / zwitterion
        # Ionised basic species interact with acidic phospholipids in tissue,
        # giving enhanced tissue distribution.
        kp = (fw_t * ion + lipid_t + fp_t * max(ion - 1.0, 0.0)) / max(fw_p + lipid_p, 1e-12)

    return float(max(round(kp, 4), 0.01))


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------


def ml_kp(
    logP: float,
    pka: float | None = None,
    compound_type: str = "neutral",
    tissue_name: str = "rest",
    fup: float = 0.5,
    mw: float = 300.0,
    *,
    drug_type: str | None = None,
) -> float:
    """Estimate Kp for a single tissue using the ML ensemble predictor.

    Delegates to the KpEnsemble singleton (auto-trains on first call).
    Falls back to heuristic_kp for tissues outside the 13-tissue PBPK list.

    Args:
        logP: Octanol-water log partition coefficient.
        pka: Primary pKa value (defaults to 7.0 when None).
        compound_type: Ionization class ('neutral', 'acid', 'base', 'ampholyte').
        tissue_name: Target tissue name.
        fup: Fraction unbound in plasma (0-1).
        mw: Molecular weight (g/mol).
        drug_type: Legacy ionisation class (overrides compound_type if given).

    Returns:
        Estimated Kp (clamped to [0.1, 100] for ML tissues).
    """
    from omega_pbpk.ml_models.kp_predictor import KpInput, predict_kp_ml

    ct = compound_type
    if drug_type is not None and compound_type == "neutral":
        ct = _DRUG_TYPE_MAP.get(drug_type, compound_type)
    inp = KpInput(
        logP=logP,
        pKa=pka if pka is not None else 7.0,
        fup=fup,
        mw=mw,
        compound_type=ct,
    )
    out = predict_kp_ml(inp)
    if tissue_name not in out.tissue_kp:
        return heuristic_kp(logP, pka, compound_type=ct, tissue_name=tissue_name, fup=fup)
    return out.tissue_kp[tissue_name]


def ml_kp_from_drug(drug: object) -> dict[str, float]:
    """Compute ML Kp for all 13 PBPK tissues from a Drug object.

    Args:
        drug: Drug instance with logP, pka, fup, mw, drug_type attributes.

    Returns:
        Dict mapping tissue name → Kp value.
    """
    from omega_pbpk.ml_models.kp_predictor import KpInput, predict_kp_ml

    pka_val: float = 7.0
    pka_list = getattr(drug, "pka", None)
    if pka_list:
        pka_val = float(pka_list[0])
    ct = _DRUG_TYPE_MAP.get(getattr(drug, "drug_type", "neutral"), "neutral")
    inp = KpInput(
        logP=float(getattr(drug, "logP", 2.0)),
        pKa=pka_val,
        fup=float(getattr(drug, "fup", 0.5)),
        mw=float(getattr(drug, "mw", 300.0)),
        compound_type=ct,
    )
    out = predict_kp_ml(inp)
    return dict(out.tissue_kp)


def berezhkovskiy_kp(
    logP: float,
    pka: float | None = None,
    compound_type: str = "neutral",
    tissue_name: str = "rest",
    fup: float = 0.5,
    *,
    drug_type: str | None = None,
) -> float:
    """Berezhkovskiy steady-state corrected Kp (2004).

    Uses Rodgers & Rowland Kp_uu as the mechanistic base, then applies
    the Berezhkovskiy correction: Kp_ss = Kp_uu × fup (assuming fut ≈ 1).

    This properly accounts for plasma protein binding: highly bound drugs
    (low fup) have restricted tissue distribution, reducing Vd and
    increasing Cmax — matching clinical observations.

    Reference:
        Berezhkovskiy LM. J Pharm Sci. 2004;93(6):1628-40.
        Rodgers T, Rowland M. J Pharm Sci. 2006;95(6):1238-57.
    """
    ct = compound_type
    if drug_type is not None and compound_type == "neutral":
        ct = _DRUG_TYPE_MAP.get(drug_type, compound_type)

    tissue = TISSUE_COMPOSITION.get(tissue_name)
    if tissue is None:
        return max(fup, 0.01)

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

    # For acids only: use D (distribution coefficient at pH) for lipid partition
    # terms instead of P. Only the neutral fraction partitions into neutral lipids;
    # for strong acids (pKa~4) at tissue pH 7.0, only ~0.1% is neutral →
    # D_7.0 ≈ P/1000. This corrects Kp for ibuprofen, atorvastatin, warfarin, etc.
    #
    # For bases/zwitterions: keep P_ow. R&R Eq.5 for bases was calibrated
    # assuming P for neutral lipid terms while the extra fp_t×(ion−1) term
    # captures ionized-species phospholipid binding independently. Using D for
    # bases over-reduces Kp (metoprolol actual Vd ~238L would drop too far).
    if ct == "acid":
        d_ow_t = 10.0 ** _logD_at_pH(logP, pka, pH_t, ct)
        d_ow_p = 10.0 ** _logD_at_pH(logP, pka, pH_p, ct)
    else:
        d_ow_t = p_ow
        d_ow_p = p_ow

    # Lipid partition terms
    lipid_t = fn_t * d_ow_t + fp_t * (0.3 * d_ow_t + 0.7)
    lipid_p = fn_p * d_ow_p + fp_p * (0.3 * d_ow_p + 0.7)

    ion = _ionization_ratio(pka, pH_t, pH_p, ct)

    if ct in ("neutral", "acid"):
        kp_uu = (fw_t * ion + lipid_t) / max(fw_p + lipid_p, 1e-12)
    else:
        # Base/zwitterion: enhanced phospholipid binding
        kp_uu = (fw_t * ion + lipid_t + fp_t * max(ion - 1.0, 0.0)) / max(fw_p + lipid_p, 1e-12)

    # Berezhkovskiy correction: Kp = Kp_uu × (fup / fut)
    # When fut ≈ 1 (hydrophilic): Kp ≈ Kp_uu × fup
    # For lipophilic drugs (high logP), tissue binding is significant (fut << 1),
    # so using fup alone over-corrects → Kp too small → Cmax too high.
    # Approximation: Kp = Kp_uu × fup^alpha, where alpha decreases with logP.
    # For bases/zwitterions, tissue binding is especially strong (lysosomal
    # trapping), so alpha is even lower — fut << fup means fup/fut > 1.
    if ct in ("base", "zwitterion"):
        # Bases: aggressive correction — tissue binding very strong
        alpha = max(0.15, min(1.0, 1.0 - 0.2 * logP))
    else:
        # Neutral/acid: moderate correction
        alpha = max(0.5, min(1.0, 1.0 - 0.125 * logP))
    kp = kp_uu * max(fup, 0.001) ** alpha

    return float(max(round(kp, 4), 0.01))


# Standard tissue volumes as fraction of body weight (L/kg), ICRP reference
_TISSUE_VOLS_L_PER_KG: dict[str, float] = {
    "liver": 0.026,
    "kidney": 0.0044,
    "muscle": 0.40,
    "adipose": 0.21,
    "fat": 0.21,
    "skin": 0.047,
    "brain": 0.020,
    "heart": 0.0047,
    "lung": 0.0076,
    "spleen": 0.0026,
    "gut_wall": 0.017,
    "bone": 0.079,
    "pancreas": 0.0014,
    "thymus": 0.0003,
    "reproductive": 0.0003,
    "rest": 0.043,
}


def vdss_from_kp(kp_dict: dict[str, float], bw_kg: float = 70.0) -> float:
    """Compute VDss from tissue Kp values.

    VDss = Vp + Σ(Kp_i × V_tissue_i)
    where Vp = 0.043 L/kg (plasma volume).
    """
    vp = 0.043 * bw_kg
    vd = vp
    for tissue, kp in kp_dict.items():
        vol_frac = _TISSUE_VOLS_L_PER_KG.get(tissue, 0.0)
        vd += kp * vol_frac * bw_kg
    return max(vd, vp)


def estimate_kp_vdss_calibrated(
    logP: float,
    fup: float,
    vdss_target_L_per_kg: float,
    pka: float | None = None,
    drug_type: str = "neutral",
    bw_kg: float = 70.0,
) -> dict[str, float]:
    """Estimate Kp for all tissues, scaled to match a target VDss.

    Uses Berezhkovskiy-corrected Rodgers & Rowland as the mechanistic base,
    then uniformly scales all Kp values so that the resulting VDss matches
    the target (e.g. from XGBoost VDss prediction).

    This preserves tissue-specific distribution ratios from the mechanistic
    model while calibrating the total VDss to match data-driven predictions.

    Args:
        logP: Octanol-water log partition coefficient.
        fup: Fraction unbound in plasma.
        vdss_target_L_per_kg: Target VDss in L/kg (e.g. from XGBoost).
        pka: Primary pKa value.
        drug_type: Ionization class.
        bw_kg: Body weight in kg.

    Returns:
        Dict mapping tissue name → scaled Kp value.
    """
    ct = _DRUG_TYPE_MAP.get(drug_type, "neutral")
    tissues = list(TISSUE_COMPOSITION.keys())

    # Step 1: Compute Berezhkovskiy-corrected Kp for all tissues
    kp_base = {}
    for t in tissues:
        kp_base[t] = berezhkovskiy_kp(logP=logP, pka=pka, compound_type=ct, tissue_name=t, fup=fup)

    # Step 2: Compute the Vd from these base Kp values
    vd_base = vdss_from_kp(kp_base, bw_kg)
    vd_base_L_per_kg = vd_base / bw_kg

    # Step 3: Compute scaling factor
    # Target Vd = Vp + scale × Σ(Kp_i × V_i)
    # scale = (Vd_target - Vp) / (Vd_base - Vp)
    vp_L_per_kg = 0.043
    numerator = max(vdss_target_L_per_kg - vp_L_per_kg, 0.001)
    denominator = max(vd_base_L_per_kg - vp_L_per_kg, 0.001)
    scale = numerator / denominator

    # Clamp scale to prevent extreme values
    scale = max(0.1, min(scale, 10.0))

    # Step 4: Apply scaling
    kp_scaled = {}
    for t, kp in kp_base.items():
        kp_scaled[t] = float(max(round(kp * scale, 4), 0.01))

    return kp_scaled


def estimate_all_kp(
    logP: float,
    pka: float | None = None,
    drug_type: str = "neutral",
    fup: float = 0.5,
    tissues: list[str] | None = None,
) -> dict[str, float]:
    """Estimate Kp for all standard PBPK tissues (heuristic method)."""
    if tissues is None:
        tissues = list(_TISSUE_FACTORS.keys())
    return {t: heuristic_kp(logP, pka, drug_type, t, fup) for t in tissues}


def log_kp_summary(kp_values: dict[str, float]) -> str:
    """Format Kp estimates as a readable summary string."""
    lines = ["Heuristic Kp estimates:"]
    for tissue, kp in sorted(kp_values.items(), key=lambda x: -x[1]):
        lines.append(f"  {tissue:18s}  Kp = {kp:.3f}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Dispatcher (after all Kp functions are defined)
# ---------------------------------------------------------------------------

_KP_METHODS: dict[str, Callable[..., float]] = {
    "heuristic": heuristic_kp,
    "rodgers_rowland": rodgers_rowland_kp,
    "berezhkovskiy": berezhkovskiy_kp,
    "ml": ml_kp,
}


def get_partition_method(name: str) -> Callable[..., float]:
    """Return the Kp estimation function for the given method name.

    Available methods: 'heuristic', 'rodgers_rowland', 'berezhkovskiy', 'ml'.
    """
    fn = _KP_METHODS.get(name)
    if fn is None:
        raise ValueError(f"Unknown partition method '{name}'. Available: {sorted(_KP_METHODS)}")
    return fn
