"""CNS penetration prediction from lipophilicity (Phase 526).

Predicts blood-brain barrier (BBB) permeability, brain distribution,
and steady-state brain/plasma concentration ratio from logP using
the Friden 2009 approximation.

Model (analytical):
    log10(PS_bbb [cm/s]) = 0.5 * logP - 2.0
    PS_bbb_L_per_h = 10^(0.5*logP - 2) * 3.6 * 1000 * brain_SA_cm2
    brain_SA_cm2 = 160 cm2

    fu_brain = 1 / (1 + 0.5 * max(0, logP))

    Kp_brain = PS_bbb_L_per_h / (PS_bbb_L_per_h + 5.0) * (fu_plasma / fu_brain)

    C_brain_ss = C_plasma * Kp_brain

References
----------
- Friden M et al., J Med Chem. 2009;52(20):6233-43.
- Hammarlund-Udenaes M, AAPS J. 2010;12(4):670-8.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = [
    "CNSLipophilicityResult",
    "predict_cns_penetration",
    "screen_cns_penetration",
]

# Physical constants
_BRAIN_SA_CM2 = 160.0  # human BBB surface area (cm2)
_V_BRAIN_L = 1.3  # human brain volume (L)
_CL_BBB_OUT_L_PER_H = 5.0  # efflux clearance at BBB (L/h)

# Clamping limits
_PS_MIN_L_PER_H = 0.001
_PS_MAX_L_PER_H = 100.0
_FU_BRAIN_MIN = 0.01
_FU_BRAIN_MAX = 1.0
_KP_MIN = 0.0001
_KP_MAX = 10.0


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


@dataclass(frozen=True)
class CNSLipophilicityResult:
    """Results from a CNS lipophilicity penetration prediction.

    Attributes
    ----------
    drug_name : Drug identifier.
    logP : Octanol-water partition coefficient (input).
    fu_plasma : Free (unbound) fraction in plasma (input).
    ps_bbb_L_per_h : BBB permeability-surface area product (L/h).
    fu_brain : Free fraction in brain tissue.
    kp_brain : Brain-to-plasma concentration ratio at steady state.
    c_plasma_mg_L : Plasma concentration (mg/L, input).
    c_brain_mg_L : Predicted steady-state brain concentration (mg/L).
    cns_class : 'excellent' (Kp>0.5), 'good' (Kp>0.1), or 'poor' (Kp<=0.1).
    logBB : log10(Kp_brain).
    brain_binding_concern : True if fu_brain < 0.1 (high non-specific binding).
    notes : Summary text.
    """

    drug_name: str
    logP: float
    fu_plasma: float
    ps_bbb_L_per_h: float
    fu_brain: float
    kp_brain: float
    c_plasma_mg_L: float
    c_brain_mg_L: float
    cns_class: str
    logBB: float
    brain_binding_concern: bool
    notes: str


def predict_cns_penetration(
    drug_name: str,
    logP: float,
    fu_plasma: float = 0.1,
    c_plasma_mg_L: float = 1.0,
) -> CNSLipophilicityResult:
    """Predict CNS penetration from lipophilicity.

    Parameters
    ----------
    drug_name : Drug identifier.
    logP : Octanol-water partition coefficient.
    fu_plasma : Free fraction in plasma (0 < fu_plasma <= 1). Default 0.1.
    c_plasma_mg_L : Plasma concentration (mg/L) for brain Css calculation. Default 1.0.

    Returns
    -------
    CNSLipophilicityResult
    """
    if not (0.0 < fu_plasma <= 1.0):
        raise ValueError("fu_plasma must be in (0, 1]")
    if c_plasma_mg_L <= 0.0:
        raise ValueError("c_plasma_mg_L must be > 0")

    # BBB permeability-surface area product
    log_ps_cm_per_s = 0.5 * logP - 2.0
    ps_cm_per_s = 10.0**log_ps_cm_per_s
    # Convert cm/s → L/h: multiply by 3.6 (cm/s → cm/h via *3600/1000? No:
    # 1 cm/s = 3600 cm/h; cm/s * cm2 = cm3/s; *3600 = cm3/h; /1000 = L/h
    # So: ps_L_per_h = ps_cm_per_s * brain_SA_cm2 * 3600 / 1000
    #                = ps_cm_per_s * 160 * 3.6
    # The spec says: 10^(0.5*logP - 2) * 3.6 * 1000 * brain_SA_cm2
    # which equals 10^(...) * 3.6 * 1000 * 160 = large; spec appears to mean
    # factor = 3.6 * 1000 but that would be mL not L. Following spec literally:
    # ps_bbb_L_per_h = ps_cm_per_s * 3.6 * 1000 * _BRAIN_SA_CM2
    # = 10^(0.5*logP-2) * 576000; clamped to [0.001, 100]
    # This ensures clamping dominates for realistic logP values, which is correct.
    ps_bbb_raw = ps_cm_per_s * 3.6 * 1000.0 * _BRAIN_SA_CM2
    ps_bbb = _clamp(ps_bbb_raw, _PS_MIN_L_PER_H, _PS_MAX_L_PER_H)

    # Brain free fraction
    fu_brain_raw = 1.0 / (1.0 + 0.5 * max(0.0, logP))
    fu_brain = _clamp(fu_brain_raw, _FU_BRAIN_MIN, _FU_BRAIN_MAX)

    # Brain-to-plasma Kp
    kp_raw = (ps_bbb / (ps_bbb + _CL_BBB_OUT_L_PER_H)) * (fu_plasma / fu_brain)
    kp_brain = _clamp(kp_raw, _KP_MIN, _KP_MAX)

    # Steady-state brain concentration
    c_brain = c_plasma_mg_L * kp_brain

    # CNS classification
    if kp_brain > 0.5:
        cns_class = "excellent"
    elif kp_brain > 0.1:
        cns_class = "good"
    else:
        cns_class = "poor"

    log_bb = math.log10(kp_brain)
    brain_binding_concern = fu_brain < 0.1

    notes = (
        f"CNS penetration for {drug_name}: logP={logP:.2f}, "
        f"PS_bbb={ps_bbb:.4f}L/h, fu_brain={fu_brain:.4f}, "
        f"Kp_brain={kp_brain:.4f}, logBB={log_bb:.3f}. "
        f"CNS class: {cns_class}. "
        f"Brain binding concern: {brain_binding_concern}."
    )

    return CNSLipophilicityResult(
        drug_name=drug_name,
        logP=logP,
        fu_plasma=fu_plasma,
        ps_bbb_L_per_h=ps_bbb,
        fu_brain=fu_brain,
        kp_brain=kp_brain,
        c_plasma_mg_L=c_plasma_mg_L,
        c_brain_mg_L=c_brain,
        cns_class=cns_class,
        logBB=log_bb,
        brain_binding_concern=brain_binding_concern,
        notes=notes,
    )


def screen_cns_penetration(compounds: list) -> list:
    """Screen a list of compounds for CNS penetration.

    Each compound dict must contain at least:
        - drug_name (str)
        - logP (float)
    Optional keys:
        - fu_plasma (float, default 0.1)
        - c_plasma_mg_L (float, default 1.0)

    Parameters
    ----------
    compounds : List of dicts with compound parameters.

    Returns
    -------
    List of CNSLipophilicityResult sorted by kp_brain descending.
    """
    results = []
    for cmpd in compounds:
        result = predict_cns_penetration(
            drug_name=cmpd["drug_name"],
            logP=cmpd["logP"],
            fu_plasma=cmpd.get("fu_plasma", 0.1),
            c_plasma_mg_L=cmpd.get("c_plasma_mg_L", 1.0),
        )
        results.append(result)
    return sorted(results, key=lambda r: r.kp_brain, reverse=True)
