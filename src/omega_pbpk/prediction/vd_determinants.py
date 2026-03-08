"""Volume of Distribution Determinants — Phase 544.

Mechanistic prediction of steady-state volume of distribution (Vd_ss) from
tissue distribution parameters using a simplified Rodgers-Rowland approach.
"""

from __future__ import annotations

__all__ = [
    "VdDeterminantsResult",
    "predict_vd_determinants",
    "screen_vd_prediction",
]

from dataclasses import dataclass

# Human physiological volumes
_VP_L = 3.0  # plasma volume (L)
_VT_L = 38.0  # lean tissue volume (L)

_VALID_DRUG_TYPES = {"acid", "base", "neutral"}


@dataclass(frozen=True)
class VdDeterminantsResult:
    """Result of mechanistic Vd prediction from physicochemical properties."""

    drug_name: str
    logP: float
    mw_Da: float
    pKa: float
    drug_type: str
    fup: float
    fut: float
    kpu_tissue: float
    vd_predicted_L: float
    vd_class: str
    distribution_pattern: str
    lipophilicity_contribution_pct: float
    notes: str


def _compute_fut(logP: float) -> float:
    """Estimate unbound fraction in tissue from logP.

    fut = 1 / (1 + 0.5 * max(0, logP))
    Clamped to [0.01, 1.0].
    """
    fut = 1.0 / (1.0 + 0.5 * max(0.0, logP))
    return max(0.01, min(1.0, fut))


def _classify_vd(vd_L: float) -> str:
    """Return vd_class string based on predicted volume."""
    if vd_L < 10.0:
        return "very_low"
    elif vd_L < 50.0:
        return "low"
    elif vd_L < 200.0:
        return "moderate"
    else:
        return "high"


def _classify_distribution(vd_L: float) -> str:
    """Return distribution_pattern string."""
    if vd_L < 10.0:
        return "plasma_confined"
    elif vd_L < 200.0:
        return "tissue_distributed"
    else:
        return "highly_tissue_distributed"


def predict_vd_determinants(
    drug_name: str,
    logP: float,
    mw_Da: float,
    pKa: float,
    drug_type: str,
    fup: float = 0.1,
) -> VdDeterminantsResult:
    """Predict steady-state volume of distribution from physicochemical parameters.

    Model
    -----
    fut  = 1 / (1 + 0.5 * max(0, logP)), clamped to [0.01, 1.0]
    Kpu  = fup / fut
    Vd_ss = Vp + Vt * Kpu
    Clamped to [Vp, 1000 L]

    Parameters
    ----------
    drug_name:
        Name of the compound.
    logP:
        Octanol-water partition coefficient (log scale).
    mw_Da:
        Molecular weight in Daltons (> 0).
    pKa:
        Acid dissociation constant.
    drug_type:
        Ionisation class: "acid", "base", or "neutral".
    fup:
        Unbound fraction in plasma (0 < fup <= 1, default 0.1).

    Returns
    -------
    VdDeterminantsResult
    """
    if drug_type not in _VALID_DRUG_TYPES:
        raise ValueError(f"drug_type must be one of {sorted(_VALID_DRUG_TYPES)}, got {drug_type!r}")
    if not (0.0 < fup <= 1.0):
        raise ValueError(f"fup must be in (0, 1], got {fup}")
    if mw_Da <= 0.0:
        raise ValueError(f"mw_Da must be > 0, got {mw_Da}")

    # Tissue unbound fraction
    fut = _compute_fut(logP)

    # Tissue-to-unbound-plasma partition coefficient
    kpu_tissue = fup / fut

    # Steady-state Vd
    vd_ss = _VP_L + _VT_L * kpu_tissue
    vd_predicted = max(_VP_L, min(1000.0, vd_ss))

    # Classification
    vd_class = _classify_vd(vd_predicted)
    distribution_pattern = _classify_distribution(vd_predicted)

    # Lipophilicity contribution: tissue volume contribution vs plasma alone
    tissue_contribution = _VT_L * kpu_tissue
    if vd_predicted > 0.0:
        lipophilicity_contribution_pct = (tissue_contribution / vd_predicted) * 100.0
    else:
        lipophilicity_contribution_pct = 0.0
    lipophilicity_contribution_pct = max(0.0, min(100.0, lipophilicity_contribution_pct))

    # Notes
    notes_parts = [
        f"logP={logP:.2f}; fut={fut:.4f}; Kpu={kpu_tissue:.3f}",
        f"Vd_ss = {_VP_L:.0f} + {_VT_L:.0f} * {kpu_tissue:.3f} = {vd_ss:.1f} L",
        f"drug_type={drug_type}; pKa={pKa:.1f}; MW={mw_Da:.0f} Da",
        f"Class={vd_class}; pattern={distribution_pattern}",
    ]
    notes = "; ".join(notes_parts)

    return VdDeterminantsResult(
        drug_name=drug_name,
        logP=logP,
        mw_Da=mw_Da,
        pKa=pKa,
        drug_type=drug_type,
        fup=fup,
        fut=fut,
        kpu_tissue=kpu_tissue,
        vd_predicted_L=vd_predicted,
        vd_class=vd_class,
        distribution_pattern=distribution_pattern,
        lipophilicity_contribution_pct=lipophilicity_contribution_pct,
        notes=notes,
    )


def screen_vd_prediction(compounds: list) -> list:
    """Screen a list of compound dicts and sort by Vd descending.

    Each compound dict must contain keys accepted by :func:`predict_vd_determinants`:
    ``drug_name``, ``logP``, ``mw_Da``, ``pKa``, ``drug_type``, and optionally ``fup``.

    Parameters
    ----------
    compounds:
        List of dicts, one per compound.

    Returns
    -------
    List of VdDeterminantsResult sorted by vd_predicted_L descending.
    """
    results = []
    for cmpd in compounds:
        result = predict_vd_determinants(
            drug_name=cmpd["drug_name"],
            logP=cmpd["logP"],
            mw_Da=cmpd["mw_Da"],
            pKa=cmpd["pKa"],
            drug_type=cmpd["drug_type"],
            fup=cmpd.get("fup", 0.1),
        )
        results.append(result)

    results.sort(key=lambda r: r.vd_predicted_L, reverse=True)
    return results
