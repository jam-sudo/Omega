"""Phase 903 — Drug CSF vs Brain Parenchyma Distribution Predictor.

Predicts drug distribution between cerebrospinal fluid (CSF) and brain
parenchyma. Key metric is Kp,uu,brain (unbound brain/unbound plasma),
which determines target site exposure for CNS drugs.

Pure Python only — stdlib math and dataclasses.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CSFBrainDistributionResult:
    """Result of CSF/brain parenchyma distribution prediction."""

    drug_name: str
    logp: float
    psa_A2: float
    mw_Da: float

    fu_brain: float
    """Unbound fraction in brain tissue (0-1)."""

    kp_uu_brain: float
    """Unbound brain / unbound plasma (key CNS exposure metric)."""

    kp_brain: float
    """Total brain / plasma concentration ratio."""

    csf_plasma_ratio: float
    """Kp,uu,CSF — unbound CSF relative to unbound plasma."""

    csf_brain_ratio: float
    """CSF relative to brain (how well CSF sampling predicts brain exposure)."""

    bbb_permeability_class: str
    """BBB passive permeability class: 'high', 'moderate', or 'low'."""

    efflux_flag: bool
    """True if P-gp and/or BCRP efflux suspected."""

    target_site_exposure_adequate: bool
    """True if kp_uu_brain > 0.3."""

    notes: str


def _clamp(value: float, lo: float, hi: float) -> float:
    """Return value clamped to [lo, hi]."""
    return max(lo, min(hi, value))


def predict_csf_brain_distribution(
    drug_name: str,
    logp: float = 2.0,
    psa_A2: float = 60.0,
    mw_Da: float = 350.0,
    fu_plasma: float = 0.1,
    has_pgp: bool = False,
    has_bcrp: bool = False,
) -> CSFBrainDistributionResult:
    """Predict CSF and brain parenchyma distribution for a drug.

    Parameters
    ----------
    drug_name:
        Identifier for the drug.
    logp:
        Lipophilicity (log P octanol/water).
    psa_A2:
        Polar surface area in square Angstroms.
    mw_Da:
        Molecular weight in Daltons.
    fu_plasma:
        Unbound fraction in plasma (0-1).
    has_pgp:
        True if the drug is a P-glycoprotein substrate.
    has_bcrp:
        True if the drug is a BCRP substrate.

    Returns
    -------
    CSFBrainDistributionResult
    """
    if psa_A2 < 0:
        raise ValueError(f"psa_A2 must be >= 0, got {psa_A2}")
    if mw_Da <= 0:
        raise ValueError(f"mw_Da must be > 0, got {mw_Da}")

    # --- BBB passive permeability score ---
    bbb_score = logp - psa_A2 / 100.0 - mw_Da / 500.0

    if bbb_score > 1.0:
        bbb_permeability_class = "high"
    elif bbb_score > -1.0:
        bbb_permeability_class = "moderate"
    else:
        bbb_permeability_class = "low"

    # --- Nonspecific brain tissue binding ---
    fu_brain_raw = 1.0 / (1.0 + (10.0 ** (logp - 1.0)) * 0.3)
    fu_brain = _clamp(fu_brain_raw, 0.01, 1.0)

    # --- Unbound brain / unbound plasma (Kp,uu,brain) ---
    base_kp_uu = {"high": 0.8, "moderate": 0.3, "low": 0.05}[bbb_permeability_class]

    if has_pgp:
        base_kp_uu *= 0.1
    if has_bcrp:
        base_kp_uu *= 0.2

    kp_uu_brain = _clamp(max(0.001, base_kp_uu), 0.001, 1.0)

    # --- Total brain/plasma ratio ---
    kp_brain_raw = kp_uu_brain / fu_brain * fu_plasma
    kp_brain = _clamp(kp_brain_raw, 0.001, 100.0)

    # --- CSF metrics ---
    csf_plasma_ratio = _clamp(kp_uu_brain * 0.7, 0.001, 1.0)

    if kp_uu_brain > 0:
        csf_brain_ratio = csf_plasma_ratio / kp_uu_brain
    else:
        csf_brain_ratio = 0.0

    # --- Flags ---
    efflux_flag = has_pgp or has_bcrp
    target_site_exposure_adequate = kp_uu_brain > 0.3

    # --- Notes ---
    if target_site_exposure_adequate:
        notes = "Adequate CNS target site exposure — brain Kp,uu > 0.3"
    elif efflux_flag:
        notes = (
            "Active efflux reduces CNS penetration — consider P-gp/BCRP inhibitor co-administration"
        )
    else:
        notes = "Limited CNS penetration — consider structural optimization to improve BBB crossing"

    return CSFBrainDistributionResult(
        drug_name=drug_name,
        logp=logp,
        psa_A2=psa_A2,
        mw_Da=mw_Da,
        fu_brain=fu_brain,
        kp_uu_brain=kp_uu_brain,
        kp_brain=kp_brain,
        csf_plasma_ratio=csf_plasma_ratio,
        csf_brain_ratio=csf_brain_ratio,
        bbb_permeability_class=bbb_permeability_class,
        efflux_flag=efflux_flag,
        target_site_exposure_adequate=target_site_exposure_adequate,
        notes=notes,
    )


def screen_cns_penetration(candidates: list[dict]) -> list[CSFBrainDistributionResult]:
    """Screen a list of candidate drugs for CNS penetration.

    Parameters
    ----------
    candidates:
        List of dicts with keys: "name" (required), "logp" (required),
        "psa_A2" (optional, default 60), "mw_Da" (optional, default 350),
        "fu_plasma" (optional, default 0.1), "has_pgp" (optional, bool),
        "has_bcrp" (optional, bool).

    Returns
    -------
    List of CSFBrainDistributionResult sorted by kp_uu_brain descending.
    """
    results = []
    for cand in candidates:
        result = predict_csf_brain_distribution(
            drug_name=cand["name"],
            logp=cand["logp"],
            psa_A2=float(cand.get("psa_A2", 60.0)),
            mw_Da=float(cand.get("mw_Da", 350.0)),
            fu_plasma=float(cand.get("fu_plasma", 0.1)),
            has_pgp=bool(cand.get("has_pgp", False)),
            has_bcrp=bool(cand.get("has_bcrp", False)),
        )
        results.append(result)

    results.sort(key=lambda r: r.kp_uu_brain, reverse=True)
    return results
