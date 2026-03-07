"""Mutagenicity and DNA damage risk assessment based on structural alerts."""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "MutagenicityResult",
    "assess_mutagenicity_risk",
    "screen_mutagenicity",
    "calculate_td50_estimate",
]


@dataclass(frozen=True)
class MutagenicityResult:
    """Result from mutagenicity risk assessment."""

    compound_name: str
    mutagenicity_score: float
    risk_class: str
    n_structural_alerts: int
    ames_prediction: str
    primary_alert: str
    requires_ames_test: bool
    notes: str


def _classify_risk(score: float) -> str:
    """Classify mutagenicity risk from score."""
    if score <= 0:
        return "none"
    elif score <= 20:
        return "low"
    elif score <= 50:
        return "moderate"
    elif score <= 80:
        return "high"
    else:
        return "very_high"


def assess_mutagenicity_risk(
    compound_name: str,
    smiles_features: dict,
) -> MutagenicityResult:
    """Assess mutagenicity and DNA damage risk from structural features.

    Parameters
    ----------
    compound_name:
        Name of the compound (must be non-empty).
    smiles_features:
        Dictionary with structural alert counts and physicochemical properties.

    Returns
    -------
    MutagenicityResult
    """
    if not compound_name or not compound_name.strip():
        raise ValueError("compound_name must be non-empty.")

    # Extract features with defaults
    n_nitro = int(smiles_features.get("n_nitro_groups", 0))
    n_paa = int(smiles_features.get("n_primary_aromatic_amines", 0))
    n_epoxides = int(smiles_features.get("n_epoxides", 0))
    n_aziridines = int(smiles_features.get("n_aziridines", 0))
    n_aldehydes = int(smiles_features.get("n_aldehydes", 0))
    n_michael = int(smiles_features.get("n_michael_acceptors", 0))
    n_alkylating = int(smiles_features.get("n_alkylating_agents", 0))
    n_aromatic_nitro = int(smiles_features.get("n_aromatic_nitro", 0))
    has_aflatoxin = bool(smiles_features.get("has_aflatoxin_scaffold", False))
    mw = float(smiles_features.get("mw", 300.0))
    logp = float(smiles_features.get("logP", 0.0))

    # Validate counts
    for name, val in [
        ("n_nitro_groups", n_nitro),
        ("n_primary_aromatic_amines", n_paa),
        ("n_epoxides", n_epoxides),
        ("n_aziridines", n_aziridines),
        ("n_aldehydes", n_aldehydes),
        ("n_michael_acceptors", n_michael),
        ("n_alkylating_agents", n_alkylating),
        ("n_aromatic_nitro", n_aromatic_nitro),
    ]:
        if val < 0:
            raise ValueError(f"{name} must be >= 0, got {val}.")

    # Score contributions (name, count, weight)
    alert_contributions = [
        ("nitro_groups", n_nitro, 20),
        ("primary_aromatic_amines", n_paa, 25),
        ("epoxides", n_epoxides, 30),
        ("aziridines", n_aziridines, 35),
        ("aldehydes", n_aldehydes, 15),
        ("michael_acceptors", n_michael, 10),
        ("alkylating_agents", n_alkylating, 40),
        ("aromatic_nitro", n_aromatic_nitro, 15),
    ]

    # Compute raw score and track alert counts
    raw_score = 0.0
    n_total_alerts = 0
    best_alert = "none"
    best_contribution = 0.0

    for alert_name, count, weight in alert_contributions:
        contrib = count * weight
        raw_score += contrib
        n_total_alerts += count
        if contrib > best_contribution:
            best_contribution = contrib
            best_alert = alert_name

    # Aflatoxin scaffold
    if has_aflatoxin:
        raw_score += 50
        n_total_alerts += 1
        if 50 > best_contribution:
            best_contribution = 50
            best_alert = "aflatoxin_scaffold"

    # MW modifier: large molecules don't penetrate cells
    if mw > 1000:
        raw_score *= 0.5

    # logP modifier: very hydrophilic compounds have reduced membrane permeability
    if logp < -2:
        raw_score *= 0.7

    # Clip to [0, 100]
    score = max(0.0, min(100.0, raw_score))

    risk_class = _classify_risk(score)
    requires_ames_test = score > 20
    ames_prediction = "likely_positive" if score > 30 else "likely_negative"

    alert_parts = []
    if n_nitro > 0:
        alert_parts.append(f"{n_nitro} nitro")
    if n_paa > 0:
        alert_parts.append(f"{n_paa} prim_ar_amine")
    if n_epoxides > 0:
        alert_parts.append(f"{n_epoxides} epoxide")
    if n_aziridines > 0:
        alert_parts.append(f"{n_aziridines} aziridine")
    if n_aldehydes > 0:
        alert_parts.append(f"{n_aldehydes} aldehyde")
    if n_michael > 0:
        alert_parts.append(f"{n_michael} michael")
    if n_alkylating > 0:
        alert_parts.append(f"{n_alkylating} alkylating")
    if n_aromatic_nitro > 0:
        alert_parts.append(f"{n_aromatic_nitro} ar_nitro")
    if has_aflatoxin:
        alert_parts.append("aflatoxin")

    alert_str = "; ".join(alert_parts) if alert_parts else "none"
    notes = (
        f"ICH S2(R1) mutagenicity assessment; score={score:.1f}; "
        f"alerts=[{alert_str}]; MW={mw:.0f}; logP={logp:.1f}"
    )

    return MutagenicityResult(
        compound_name=compound_name,
        mutagenicity_score=score,
        risk_class=risk_class,
        n_structural_alerts=n_total_alerts,
        ames_prediction=ames_prediction,
        primary_alert=best_alert,
        requires_ames_test=requires_ames_test,
        notes=notes,
    )


def screen_mutagenicity(compounds: list[dict]) -> list[MutagenicityResult]:
    """Screen a list of compounds for mutagenicity risk.

    Parameters
    ----------
    compounds:
        List of dicts, each with "compound_name" key plus smiles_features keys.

    Returns
    -------
    List of MutagenicityResult sorted by mutagenicity_score descending.
    """
    if not compounds:
        return []

    results = []
    for comp in compounds:
        name = comp.get("compound_name", "Unknown")
        features = {k: v for k, v in comp.items() if k != "compound_name"}
        result = assess_mutagenicity_risk(name, features)
        results.append(result)

    results.sort(key=lambda r: r.mutagenicity_score, reverse=True)
    return results


def calculate_td50_estimate(mutagenicity_score: float, mw: float = 300.0) -> float:
    """Estimate TD50 (mg/kg/day) using Meselhy correlation (simplified).

    log(TD50) = 3.5 - 0.03 * mutagenicity_score - 0.001 * (mw - 300)

    Parameters
    ----------
    mutagenicity_score:
        Mutagenicity score in [0, 100].
    mw:
        Molecular weight in g/mol (default 300.0).

    Returns
    -------
    TD50 in mg/kg/day (minimum 0.001).
    """
    log_td50 = 3.5 - 0.03 * mutagenicity_score - 0.001 * (mw - 300.0)
    td50 = max(0.001, 10**log_td50)
    return td50
