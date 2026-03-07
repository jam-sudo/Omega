"""
Phase 949 — BBB Transport Prediction
Predicts passive permeability, P-gp/BCRP efflux, and net CNS penetration.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["BBBTransportResult", "predict_bbb_transport", "screen_cns_penetration"]


@dataclass(frozen=True)
class BBBTransportResult:
    drug_name: str
    logp: float
    mw: float
    pka: float
    ionization_type: str
    psa_estimated: float
    bbb_passive_perm_cm_s: float
    pgp_efflux_probability: float
    bcrp_efflux_probability: float
    net_efflux_ratio: float
    kp_uu_brain: float
    cns_penetration: str
    pgp_substrate_risk: str
    bbb_limited: bool
    notes: str


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def predict_bbb_transport(
    drug_name: str,
    logp: float,
    mw: float,
    pka: float,
    ionization_type: str,
    *,
    notes_extra: str = "",
) -> BBBTransportResult:
    """Predict BBB transport properties from physicochemical parameters."""

    if mw <= 0:
        raise ValueError("mw must be > 0")
    valid_ion = {"acid", "base", "neutral"}
    if ionization_type not in valid_ion:
        raise ValueError(f"ionization_type must be one of {valid_ion}, got '{ionization_type}'")

    # --- PSA estimation ---
    ion_contrib = {"acid": 20.0, "base": 15.0, "neutral": 5.0}[ionization_type]
    ion_contrib = _clamp(ion_contrib, 0.0, 200.0)
    psa = mw * 0.12 * (1.0 - 0.08 * logp) + ion_contrib
    psa = max(0.0, psa)

    # --- Passive permeability ---
    log_papp = 0.5 * logp - 0.01 * mw - 0.02 * psa + 0.3
    bbb_perm = 10.0**log_papp * 1e-6  # cm/s
    bbb_perm = _clamp(bbb_perm, 1e-8, 1e-3)

    # --- P-gp efflux probability ---
    pgp_prob = 0.2
    if pka > 7.0:  # basic drug
        pgp_prob += 0.25
    if 1.0 <= logp <= 5.0:
        pgp_prob += 0.15
    if 300.0 <= mw <= 700.0:
        pgp_prob += 0.1
    pgp_prob = _clamp(pgp_prob, 0.0, 1.0)

    # --- BCRP efflux probability ---
    bcrp_prob = 0.1
    if (ionization_type == "acid" or ionization_type == "neutral") and logp > 2.0:
        bcrp_prob += 0.2
    if mw < 500.0:
        bcrp_prob += 0.1
    bcrp_prob = _clamp(bcrp_prob, 0.0, 1.0)

    # --- Net efflux ratio and Kp_uu ---
    net_efflux_ratio = 1.0 + pgp_prob * 3.0 + bcrp_prob * 1.5
    kp_uu = 1.0 / net_efflux_ratio
    kp_uu = _clamp(kp_uu, 0.01, 1.0)

    # --- CNS penetration classification ---
    if kp_uu >= 0.3:
        cns_penetration = "high"
    elif kp_uu >= 0.1:
        cns_penetration = "moderate"
    else:
        cns_penetration = "low"

    # --- P-gp substrate risk ---
    if pgp_prob < 0.3:
        pgp_substrate_risk = "low"
    elif pgp_prob <= 0.6:
        pgp_substrate_risk = "moderate"
    else:
        pgp_substrate_risk = "high"

    bbb_limited = kp_uu < 0.3

    notes_parts = [
        f"PSA_est={psa:.1f} A2",
        f"logPapp={log_papp:.2f}",
        f"net_ER={net_efflux_ratio:.2f}",
        f"CNS={cns_penetration}",
    ]
    if notes_extra:
        notes_parts.append(notes_extra)
    notes = "; ".join(notes_parts)

    return BBBTransportResult(
        drug_name=drug_name,
        logp=logp,
        mw=mw,
        pka=pka,
        ionization_type=ionization_type,
        psa_estimated=psa,
        bbb_passive_perm_cm_s=bbb_perm,
        pgp_efflux_probability=pgp_prob,
        bcrp_efflux_probability=bcrp_prob,
        net_efflux_ratio=net_efflux_ratio,
        kp_uu_brain=kp_uu,
        cns_penetration=cns_penetration,
        pgp_substrate_risk=pgp_substrate_risk,
        bbb_limited=bbb_limited,
        notes=notes,
    )


def screen_cns_penetration(compounds: list[dict]) -> list[BBBTransportResult]:
    """Screen a list of compound dicts for CNS penetration, sorted by kp_uu descending."""
    results = []
    for cpd in compounds:
        result = predict_bbb_transport(
            drug_name=cpd.get("drug_name", "unknown"),
            logp=cpd["logp"],
            mw=cpd["mw"],
            pka=cpd["pka"],
            ionization_type=cpd["ionization_type"],
        )
        results.append(result)
    results.sort(key=lambda r: r.kp_uu_brain, reverse=True)
    return results
