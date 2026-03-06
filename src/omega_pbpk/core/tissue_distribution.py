"""Tissue distribution calculator using Kp values from QSPR models.

Computes steady-state tissue concentrations from pre-computed or estimated
Kp values, identifies tissues at accumulation risk, and provides Vss.

References
----------
- Berezhkovskiy LM, J Pharm Sci. 2004;93:1628–40
- Rodgers T & Rowland M, J Pharm Sci. 2006;95:1238–57
- Schmitt W, Toxicol In Vitro. 2008;22:457–67
"""

from __future__ import annotations

from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Physiological constants (70 kg reference man)
# ---------------------------------------------------------------------------
V_BLOOD = 5.0  # L

TISSUE_VOLUMES: dict[str, float] = {
    "adipose": 14.5,
    "bone": 10.5,
    "brain": 1.45,
    "gut": 1.65,
    "heart": 0.35,
    "kidney": 0.31,
    "liver": 1.80,
    "lung": 0.50,
    "muscle": 29.0,
    "skin": 3.71,
    "spleen": 0.15,
    "testes": 0.035,
    "thymus": 0.02,
}

KP_MIN = 0.05
KP_MAX = 100.0


@dataclass(frozen=True)
class TissueConcentration:
    """Steady-state concentration in a single tissue.

    Attributes
    ----------
    tissue : Tissue name.
    kp : Tissue:plasma partition coefficient.
    ct_total_mg_L : Total tissue concentration (mg/L).
    ct_unbound_mg_L : Unbound tissue concentration (mg/L).
    accumulation_flag : 'high' (Kp>5), 'moderate' (Kp>2), 'low' (Kp<=2).
    """

    tissue: str
    kp: float
    ct_total_mg_L: float
    ct_unbound_mg_L: float
    accumulation_flag: str


@dataclass(frozen=True)
class TissueDistributionResult:
    """Full tissue distribution profile at steady state.

    Attributes
    ----------
    drug_name : Name of the drug.
    c_plasma_mg_L : Plasma concentration used (mg/L).
    fup : Fraction unbound in plasma.
    logP : Octanol-water partition coefficient (log10).
    tissue_concentrations : Per-tissue concentration data.
    vss_L : Steady-state volume of distribution (L).
    highest_kp_tissue : Tissue with maximum Kp.
    highest_kp_value : Maximum Kp value.
    tissues_at_high_risk : Tissues with Kp > 5.
    total_drug_in_tissues_mg : Sum of ct_total * Vt across tissues.
    plasma_fraction_pct : Percentage of total drug in plasma.
    notes : Summary text.
    """

    drug_name: str
    c_plasma_mg_L: float
    fup: float
    logP: float
    tissue_concentrations: list[TissueConcentration]
    vss_L: float
    highest_kp_tissue: str
    highest_kp_value: float
    tissues_at_high_risk: list[str]
    total_drug_in_tissues_mg: float
    plasma_fraction_pct: float
    notes: str


def _estimate_kp(tissue: str, logP: float, fup: float) -> float:
    """Estimate Kp for a tissue from logP and fup (simplified Berezhkovskiy)."""
    fup_safe = max(fup, 0.01)

    if tissue == "adipose":
        raw = 0.5 * (10**logP) * (1 - fup) / fup_safe + fup
        raw = max(0.1, raw)
    elif tissue == "muscle":
        raw = max(0.5, 0.3 * logP + 0.5) if logP > 0 else max(0.3, 0.5 + 0.1 * logP)
    elif tissue == "liver":
        raw = max(0.5, 1.5 * logP + 1.0) if logP > 0 else 1.0
    elif tissue == "brain":
        raw = max(0.1, 0.2 * logP + 0.3)
    elif tissue == "kidney":
        raw = max(0.5, logP + 1.0)
    elif tissue == "heart":
        raw = max(0.5, 0.5 * logP + 0.8)
    elif tissue == "gut":
        raw = max(0.5, logP + 0.5)
    elif tissue == "lung":
        raw = max(0.5, 0.3 * logP + 1.0)
    elif tissue == "skin":
        raw = max(0.3, 0.8 * logP + 0.5)
    elif tissue == "spleen":
        raw = max(0.5, 0.5 * logP + 0.8)
    elif tissue == "bone":
        raw = max(0.2, 0.2 * logP + 0.3)
    elif tissue == "testes":
        raw = max(0.3, 0.3 * logP + 0.5)
    elif tissue == "thymus":
        raw = max(0.3, 0.3 * logP + 0.5)
    else:
        raw = max(0.5, 0.5 * logP + 0.5)

    # Scale by fup factor (unbound drives partition)
    kp = raw * (1.0 / fup_safe)
    return float(min(max(kp, KP_MIN), KP_MAX))


def _accumulation_flag(kp: float) -> str:
    if kp > 5:
        return "high"
    if kp > 2:
        return "moderate"
    return "low"


def vss_from_kp(kp_values: dict[str, float]) -> float:
    """Compute Vss from provided Kp dictionary.

    Vss = V_blood + sum(Kp_i * V_i) for tissues present in both
    kp_values and TISSUE_VOLUMES.

    Parameters
    ----------
    kp_values : Mapping of tissue name to Kp value.

    Returns
    -------
    Vss in litres.
    """
    vss = V_BLOOD
    for tissue, kp in kp_values.items():
        if tissue in TISSUE_VOLUMES:
            vss += kp * TISSUE_VOLUMES[tissue]
    return vss


def predict_tissue_distribution(
    drug_name: str,
    c_plasma_mg_L: float,
    logP: float,
    fup: float,
    kp_overrides: dict[str, float] | None = None,
) -> TissueDistributionResult:
    """Predict steady-state tissue distribution from plasma concentration.

    Parameters
    ----------
    drug_name : Drug name.
    c_plasma_mg_L : Plasma concentration at steady state (mg/L).
    logP : Octanol-water partition coefficient.
    fup : Fraction unbound in plasma (0, 1].
    kp_overrides : Optional overrides for specific tissue Kp values.

    Returns
    -------
    TissueDistributionResult

    Raises
    ------
    ValueError : If parameters are invalid.
    """
    if c_plasma_mg_L <= 0:
        raise ValueError("c_plasma_mg_L must be > 0")
    if fup <= 0 or fup > 1:
        raise ValueError("fup must be in (0, 1]")

    overrides = kp_overrides or {}
    kp_values: dict[str, float] = {}
    tissue_concs: list[TissueConcentration] = []

    for tissue in TISSUE_VOLUMES:
        if tissue in overrides:
            kp = float(min(max(overrides[tissue], KP_MIN), KP_MAX))
        else:
            kp = _estimate_kp(tissue, logP, fup)

        kp_values[tissue] = kp
        ct_total = kp * c_plasma_mg_L
        ct_unbound = c_plasma_mg_L * fup  # same as plasma unbound at SS

        tissue_concs.append(
            TissueConcentration(
                tissue=tissue,
                kp=kp,
                ct_total_mg_L=ct_total,
                ct_unbound_mg_L=ct_unbound,
                accumulation_flag=_accumulation_flag(kp),
            )
        )

    vss = vss_from_kp(kp_values)
    highest = max(tissue_concs, key=lambda tc: tc.kp)
    high_risk = [tc.tissue for tc in tissue_concs if tc.kp > 5]

    total_drug_tissues = sum(tc.ct_total_mg_L * TISSUE_VOLUMES[tc.tissue] for tc in tissue_concs)
    drug_in_plasma = c_plasma_mg_L * V_BLOOD
    total_drug = total_drug_tissues + drug_in_plasma
    plasma_frac = (drug_in_plasma / total_drug * 100.0) if total_drug > 0 else 0.0

    notes = (
        f"Tissue distribution: {drug_name}, logP={logP}, fup={fup}. "
        f"Vss={vss:.1f}L. Highest Kp: {highest.tissue}={highest.kp:.2f}. "
        f"{len(high_risk)} tissue(s) at high accumulation risk."
    )

    return TissueDistributionResult(
        drug_name=drug_name,
        c_plasma_mg_L=c_plasma_mg_L,
        fup=fup,
        logP=logP,
        tissue_concentrations=tissue_concs,
        vss_L=vss,
        highest_kp_tissue=highest.tissue,
        highest_kp_value=highest.kp,
        tissues_at_high_risk=high_risk,
        total_drug_in_tissues_mg=total_drug_tissues,
        plasma_fraction_pct=plasma_frac,
        notes=notes,
    )


def accumulation_risk_screen(
    drugs: list[dict],
) -> list[TissueDistributionResult]:
    """Screen multiple drugs for tissue accumulation risk.

    Parameters
    ----------
    drugs : List of dicts with keys 'name', 'c_plasma_mg_L', 'logP', 'fup'.

    Returns
    -------
    List of TissueDistributionResult, sorted by vss_L descending.
    """
    results = [
        predict_tissue_distribution(
            drug_name=d["name"],
            c_plasma_mg_L=d["c_plasma_mg_L"],
            logP=d["logP"],
            fup=d["fup"],
        )
        for d in drugs
    ]
    return sorted(results, key=lambda r: r.vss_L, reverse=True)


__all__ = [
    "TissueConcentration",
    "TissueDistributionResult",
    "TISSUE_VOLUMES",
    "predict_tissue_distribution",
    "vss_from_kp",
    "accumulation_risk_screen",
]
