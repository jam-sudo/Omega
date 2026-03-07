"""Volume of distribution prediction using the Oie-Tozer model."""

from __future__ import annotations

from dataclasses import dataclass

# Oie-Tozer physiological volumes (L/kg)
_VP = 0.0436   # plasma volume
_VE = 0.151    # extravascular interstitial fluid volume
_VR = 0.380    # remaining (intracellular + deep tissue) volume


def _compute_re_fp(logP: float) -> float:
    """Compute the erythrocyte-to-plasma ratio from logP.

    RE/FP = 1 + logP * 0.2, clamped to [0.2, 5.0].
    """
    re_fp = 1.0 + logP * 0.2
    return max(0.2, min(5.0, re_fp))


def _estimate_fu_tissue(fu_plasma: float, logP: float) -> float:
    """Estimate fu_tissue from fu_plasma and logP.

    fu_tissue = fu_plasma / (1 + 0.072 * logP^2)
    Clamp result to (0, 1].
    """
    denom = 1.0 + 0.072 * logP**2
    fu_t = fu_plasma / max(denom, 1e-9)
    return max(1e-6, min(1.0, fu_t))


def _classify_vd(vd_L_per_kg: float) -> str:
    """Return Vd classification string."""
    if vd_L_per_kg < 0.6:
        return "low"
    if vd_L_per_kg < 3.0:
        return "moderate"
    if vd_L_per_kg < 20.0:
        return "high"
    return "very_high"


@dataclass(frozen=True)
class VdPredictionResult:
    """Result of Oie-Tozer Vd prediction."""

    mw: float
    logP: float
    psa: float
    fu_plasma: float
    fu_tissue_used: float
    pka: float | None
    molecule_type: str
    body_weight_kg: float
    vd_L_per_kg: float
    vd_L: float
    tissue_binding_factor: float
    vd_class: str
    notes: str


def predict_vd(
    mw: float,
    logP: float,
    psa: float,
    fu_plasma: float,
    fu_tissue: float | None = None,
    pka: float | None = None,
    molecule_type: str = "small_molecule",
    body_weight_kg: float = 70.0,
) -> VdPredictionResult:
    """Predict volume of distribution using the Oie-Tozer model.

    Vd = Vp + Vp*(RE/FP) + (fu_plasma/fu_tissue)*VR

    where:
        Vp  = 0.0436 L/kg  (plasma volume)
        VE  = 0.151  L/kg  (interstitial fluid)
        VR  = 0.38   L/kg  (intracellular/deep tissue)
        RE/FP = erythrocyte-to-plasma concentration ratio (from logP)

    Parameters
    ----------
    mw : float
        Molecular weight (g/mol). Used for notes only.
    logP : float
        Octanol-water partition coefficient.
    psa : float
        Polar surface area (Å²). Used for notes only.
    fu_plasma : float
        Fraction unbound in plasma (0 < fu_plasma <= 1).
    fu_tissue : float or None
        Fraction unbound in tissue. If None, estimated from fu_plasma and logP.
    pka : float or None
        pKa of the molecule (informational).
    molecule_type : str
        Molecule type label (e.g., 'small_molecule', 'peptide', 'biologic').
    body_weight_kg : float
        Subject body weight (kg). Used to compute total Vd in litres.

    Returns
    -------
    VdPredictionResult
    """
    if mw <= 0:
        raise ValueError("mw must be > 0")
    if psa < 0:
        raise ValueError("psa must be >= 0")
    if not 0 < fu_plasma <= 1.0:
        raise ValueError("fu_plasma must be in (0, 1]")
    if fu_tissue is not None and not 0 < fu_tissue <= 1.0:
        raise ValueError("fu_tissue must be in (0, 1] when provided")
    if body_weight_kg <= 0:
        raise ValueError("body_weight_kg must be > 0")

    # Compute RE/FP from logP
    re_fp = _compute_re_fp(logP)

    # Determine fu_tissue
    if fu_tissue is None:
        fu_t = _estimate_fu_tissue(fu_plasma, logP)
        fu_tissue_note = "estimated"
    else:
        fu_t = fu_tissue
        fu_tissue_note = "provided"

    # Oie-Tozer equation (all volumes in L/kg)
    # Vd = Vp + Vp * RE/FP + (fu_plasma / fu_tissue) * VR
    tissue_binding_factor = fu_plasma / max(fu_t, 1e-9)
    vd_L_per_kg = _VP + _VP * re_fp + tissue_binding_factor * _VR

    vd_L = vd_L_per_kg * body_weight_kg
    vd_class = _classify_vd(vd_L_per_kg)

    notes = (
        f"Oie-Tozer model. RE/FP={re_fp:.3f} (logP={logP}). "
        f"fu_tissue={fu_t:.4f} ({fu_tissue_note}). "
        f"Vd={vd_L_per_kg:.3f} L/kg ({vd_class})."
    )

    return VdPredictionResult(
        mw=mw,
        logP=logP,
        psa=psa,
        fu_plasma=fu_plasma,
        fu_tissue_used=fu_t,
        pka=pka,
        molecule_type=molecule_type,
        body_weight_kg=body_weight_kg,
        vd_L_per_kg=vd_L_per_kg,
        vd_L=vd_L,
        tissue_binding_factor=tissue_binding_factor,
        vd_class=vd_class,
        notes=notes,
    )


def vd_sensitivity(
    mw: float,
    logP_values: list[float],
    psa: float,
    fu_plasma: float,
    pka: float | None = None,
    molecule_type: str = "small_molecule",
) -> list[dict]:
    """Sweep logP values and return Vd predictions for each.

    Parameters
    ----------
    mw : float
        Molecular weight (g/mol).
    logP_values : list[float]
        List of logP values to evaluate.
    psa : float
        Polar surface area (Å²).
    fu_plasma : float
        Fraction unbound in plasma.
    pka : float or None
        pKa of the molecule.
    molecule_type : str
        Molecule type label.

    Returns
    -------
    list[dict]
        Each dict has keys: logP, vd_L_per_kg, vd_L, vd_class, tissue_binding_factor.
    """
    if not logP_values:
        raise ValueError("logP_values must not be empty")
    if mw <= 0:
        raise ValueError("mw must be > 0")
    if psa < 0:
        raise ValueError("psa must be >= 0")
    if not 0 < fu_plasma <= 1.0:
        raise ValueError("fu_plasma must be in (0, 1]")

    results = []
    for lp in logP_values:
        r = predict_vd(
            mw=mw,
            logP=lp,
            psa=psa,
            fu_plasma=fu_plasma,
            pka=pka,
            molecule_type=molecule_type,
        )
        results.append(
            {
                "logP": lp,
                "vd_L_per_kg": r.vd_L_per_kg,
                "vd_L": r.vd_L,
                "vd_class": r.vd_class,
                "tissue_binding_factor": r.tissue_binding_factor,
            }
        )
    return results


__all__ = [
    "VdPredictionResult",
    "predict_vd",
    "vd_sensitivity",
]
