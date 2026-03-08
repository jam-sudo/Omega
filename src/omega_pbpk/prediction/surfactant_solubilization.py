"""Drug solubilization by surfactants above CMC.

Predicts micellar solubilization based on Yalkowsky & Roseman (1981).

Key equation:
    S_total = S_aqueous + K_mw * [surfactant - CMC]  if [surf] > CMC else S_aqueous

where K_mw is the micellar partition coefficient derived from logP:
    K_mw = 10^(0.7 * logP - 0.3)  [clamped to 1–100,000]

References
----------
- Yalkowsky SH & Roseman TJ. Solubilization of drugs by cosolvents.
  In: Techniques of Solubilization of Drugs. 1981.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "SurfactantSolubilizationResult",
    "predict_solubilization",
    "screen_surfactants",
]

# Known surfactants: name → (CMC_mM, micelle_molar_volume_L_mol)
_SURFACTANTS: dict[str, tuple[float, float]] = {
    "SDS": (8.0, 0.25),
    "Tween_80": (0.012, 0.95),
    "CTAB": (1.0, 0.30),
    "Pluronic_F127": (0.001, 2.0),
    "sodium_cholate": (5.0, 0.40),
}

_LOG_P_MIN = -5.0
_LOG_P_MAX = 10.0


@dataclass(frozen=True)
class SurfactantSolubilizationResult:
    """Result of surfactant solubilization prediction.

    Attributes
    ----------
    drug_name : str
    surfactant : str
    logP : float
    surfactant_conc_mM : float
    cmc_mM : float
    s_aqueous_mg_mL : float       Intrinsic aqueous solubility (mg/mL).
    s_total_mg_mL : float         Total solubility with surfactant (mg/mL).
    fold_enhancement : float       S_total / S_aqueous.
    k_mw : float                  Micellar partition coefficient (dimensionless).
    above_cmc : bool              True if surfactant_conc > CMC.
    solubilization_efficiency : str  "none" (<1.1×), "low" (<2×), "moderate" (<10×), "high" (≥10×).
    notes : str
    """

    drug_name: str
    surfactant: str
    logP: float
    surfactant_conc_mM: float
    cmc_mM: float
    s_aqueous_mg_mL: float
    s_total_mg_mL: float
    fold_enhancement: float
    k_mw: float
    above_cmc: bool
    solubilization_efficiency: str
    notes: str


def _compute_k_mw(logP: float) -> float:
    """Micellar partition coefficient from logP (clamped to [1, 1e5])."""
    k_mw = 10.0 ** (0.7 * logP - 0.3)
    return max(1.0, min(1.0e5, k_mw))


def predict_solubilization(
    drug_name: str,
    surfactant: str,
    logP: float,
    surfactant_conc_mM: float,
    intrinsic_solubility_mg_mL: float = 0.1,
) -> SurfactantSolubilizationResult:
    """Predict drug solubilization by a specific surfactant.

    Parameters
    ----------
    drug_name : str
        Name of the drug.
    surfactant : str
        Surfactant name. Must be one of: SDS, Tween_80, CTAB, Pluronic_F127, sodium_cholate.
    logP : float
        Drug octanol-water partition coefficient. Must be in [-5, 10].
    surfactant_conc_mM : float
        Surfactant concentration (mM). Must be > 0.
    intrinsic_solubility_mg_mL : float
        Intrinsic (aqueous) solubility of the drug (mg/mL). Must be > 0.

    Returns
    -------
    SurfactantSolubilizationResult

    Raises
    ------
    ValueError
        If surfactant is unknown, logP is out of range, or other inputs are invalid.
    """
    if surfactant not in _SURFACTANTS:
        raise ValueError(
            f"surfactant '{surfactant}' not recognised. Choose from: {sorted(_SURFACTANTS)}"
        )
    if logP < _LOG_P_MIN or logP > _LOG_P_MAX:
        raise ValueError(f"logP must be in [{_LOG_P_MIN}, {_LOG_P_MAX}], got {logP}")
    if surfactant_conc_mM <= 0:
        raise ValueError("surfactant_conc_mM must be > 0")
    if intrinsic_solubility_mg_mL <= 0:
        raise ValueError("intrinsic_solubility_mg_mL must be > 0")

    cmc_mM, _micelle_vol = _SURFACTANTS[surfactant]
    above_cmc = surfactant_conc_mM > cmc_mM

    k_mw = _compute_k_mw(logP)

    if above_cmc:
        # Micellar solubilization: S_total = S_aq + K_mw * [surf_free - CMC]  (mM → L/L via 1e-3)
        # K_mw is dimensionless (drug-in-micelle / drug-in-water concentration ratio)
        # [micelle phase] = (conc_mM - CMC_mM) * 1e-3 mol/L * micelle_vol_L/mol
        free_surf_mM = surfactant_conc_mM - cmc_mM
        # Simplify: micellar volume fraction ≈ free_surf_mM * micelle_vol * 1e-3
        phi_m = free_surf_mM * _micelle_vol * 1.0e-3  # dimensionless volume fraction
        # S_total_mg_mL = S_aq + K_mw * phi_m * S_aq  = S_aq * (1 + K_mw * phi_m)
        s_total = intrinsic_solubility_mg_mL * (1.0 + k_mw * phi_m)
    else:
        s_total = intrinsic_solubility_mg_mL

    fold_enhancement = s_total / intrinsic_solubility_mg_mL

    # Solubilization efficiency category
    if fold_enhancement < 1.1:
        efficiency = "none"
    elif fold_enhancement < 2.0:
        efficiency = "low"
    elif fold_enhancement < 10.0:
        efficiency = "moderate"
    else:
        efficiency = "high"

    above_str = "above" if above_cmc else "below"
    notes = (
        f"{drug_name} with {surfactant} ({surfactant_conc_mM:.3g} mM, "
        f"{above_str} CMC={cmc_mM} mM): "
        f"K_mw={k_mw:.1f}, S_total={s_total:.4g} mg/mL "
        f"({fold_enhancement:.2f}× enhancement, {efficiency} efficiency)."
    )

    return SurfactantSolubilizationResult(
        drug_name=drug_name,
        surfactant=surfactant,
        logP=logP,
        surfactant_conc_mM=surfactant_conc_mM,
        cmc_mM=cmc_mM,
        s_aqueous_mg_mL=intrinsic_solubility_mg_mL,
        s_total_mg_mL=s_total,
        fold_enhancement=fold_enhancement,
        k_mw=k_mw,
        above_cmc=above_cmc,
        solubilization_efficiency=efficiency,
        notes=notes,
    )


def screen_surfactants(
    drug_name: str,
    logP: float,
    surfactant_conc_mM: float = 10.0,
    intrinsic_solubility_mg_mL: float = 0.1,
) -> list[SurfactantSolubilizationResult]:
    """Screen all known surfactants for drug solubilization potential.

    Parameters
    ----------
    drug_name : str
        Drug name.
    logP : float
        Drug logP. Must be in [-5, 10].
    surfactant_conc_mM : float
        Surfactant concentration to evaluate (mM).
    intrinsic_solubility_mg_mL : float
        Intrinsic aqueous solubility (mg/mL).

    Returns
    -------
    list[SurfactantSolubilizationResult]
        All 5 surfactants tested, sorted by fold_enhancement descending.
    """
    results = [
        predict_solubilization(
            drug_name,
            surf,
            logP,
            surfactant_conc_mM,
            intrinsic_solubility_mg_mL,
        )
        for surf in _SURFACTANTS
    ]
    return sorted(results, key=lambda r: r.fold_enhancement, reverse=True)
