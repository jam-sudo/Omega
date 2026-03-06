"""Plasma protein binding (PPB) competition model.

Models competitive displacement of two drugs from albumin binding sites using
a Scatchard-based equilibrium model.  Drug A and Drug B compete for the same
protein binding sites; higher concentrations of B displace A, increasing the
free fraction of A.
"""

from __future__ import annotations

from dataclasses import dataclass

# Standard physiological parameters
_ALBUMIN_MW_G_PER_MOL = 66_500.0  # g/mol
_ALBUMIN_NORMAL_G_DL = 4.0  # g/dL (normal plasma albumin)
_PLASMA_VOL_DL = 30.0  # typical plasma volume expressed in dL (≈ 3 L / 0.1 L·dL⁻¹)


def _albumin_molar_conc(albumin_g_dL: float) -> float:
    """Convert albumin g/dL to molar concentration (mol/L).

    Plasma volume: 3 L → 30 dL.  Molar conc = g/dL * 10 dL/L / MW.
    """
    return albumin_g_dL * 10.0 / _ALBUMIN_MW_G_PER_MOL  # mol/L


def _infer_kd(fu_alone: float, albumin_g_dL: float, n_sites: float) -> float:
    """Infer dissociation constant Kd (mol/L) from fu measured at given albumin.

    Scatchard single-class: fu = 1 / (1 + n * [P] / Kd)
    → Kd = n * [P] * (1 - fu) / fu  (in mol/L, assuming fu_alone is at conc ca ≪ Kd)
    """
    p_mol_per_L = _albumin_molar_conc(albumin_g_dL)
    # fu = 1 / (1 + n*P/Kd)  →  n*P/Kd = (1-fu)/fu  →  Kd = n*P*fu/(1-fu)
    bound_ratio = (1.0 - fu_alone) / fu_alone
    if bound_ratio <= 0.0:
        return float("inf")  # completely unbound drug
    return n_sites * p_mol_per_L / bound_ratio


def _fu_competition(
    kd_a: float,
    kd_b: float,
    p_mol_per_L: float,
    n_sites: float,
    cb_free_mol_per_L: float,
) -> float:
    """Scatchard competition: fu_A at equilibrium when B is present.

    fu_A = 1 / (1 + n * [P] / (Kd_A * (1 + [B_free] / Kd_B)))
    """
    if kd_a == float("inf"):
        return 1.0
    apparent_kd_a = kd_a * (1.0 + cb_free_mol_per_L / kd_b)
    return 1.0 / (1.0 + n_sites * p_mol_per_L / apparent_kd_a)


def _clinical_significance(delta_fu: float) -> str:
    """Classify clinical significance of displacement based on delta_fu_a."""
    if delta_fu < 0.01:
        return "none"
    elif delta_fu < 0.05:
        return "minor"
    elif delta_fu < 0.15:
        return "moderate"
    else:
        return "major"


@dataclass(frozen=True)
class PPBCompetitionResult:
    """Result of a plasma protein binding competition analysis."""

    drug_a_name: str
    drug_b_name: str
    ca_mg_L: float
    cb_mg_L: float
    fu_a_alone: float
    fu_b_alone: float
    fu_a_displaced: float  # free fraction of A in presence of B
    fu_b_displaced: float  # free fraction of B in presence of A
    delta_fu_a: float  # fu_a_displaced - fu_a_alone
    displacement_pct_a: float  # (fu_a_displaced - fu_a_alone) / fu_a_alone * 100
    clinical_significance: str  # "none", "minor", "moderate", "major"
    notes: str


def compute_ppb_competition(
    drug_a_name: str,
    drug_b_name: str,
    ca_mg_L: float,
    cb_mg_L: float,
    fu_a_alone: float,
    fu_b_alone: float,
    albumin_g_dL: float = 4.0,
    n_sites: int = 1,
) -> PPBCompetitionResult:
    """Compute competitive displacement of Drug A by Drug B at albumin.

    Uses the Scatchard single-class competition model.  Kd values are inferred
    from the measured free fractions (``fu_*_alone``) at the specified albumin
    concentration.

    Parameters
    ----------
    drug_a_name, drug_b_name:
        Names of the two competing drugs.
    ca_mg_L:
        Total plasma concentration of Drug A (mg/L).
    cb_mg_L:
        Total plasma concentration of Drug B (mg/L).  Set to 0 to get baseline.
    fu_a_alone, fu_b_alone:
        Free fractions measured in the absence of the competitor (0, 1].
    albumin_g_dL:
        Plasma albumin concentration (g/dL).  Default 4.0 g/dL.
    n_sites:
        Number of equivalent binding sites per albumin molecule.  Default 1.

    Returns
    -------
    PPBCompetitionResult
    """
    # --- validation ---
    if not (0.0 < fu_a_alone <= 1.0):
        raise ValueError(f"fu_a_alone must be in (0, 1], got {fu_a_alone}")
    if not (0.0 < fu_b_alone <= 1.0):
        raise ValueError(f"fu_b_alone must be in (0, 1], got {fu_b_alone}")
    if ca_mg_L <= 0.0:
        raise ValueError(f"ca_mg_L must be positive, got {ca_mg_L}")
    if cb_mg_L < 0.0:
        raise ValueError(f"cb_mg_L must be >= 0, got {cb_mg_L}")
    if albumin_g_dL <= 0.0:
        raise ValueError(f"albumin_g_dL must be positive, got {albumin_g_dL}")
    if n_sites < 1:
        raise ValueError(f"n_sites must be >= 1, got {n_sites}")

    p_mol_per_L = _albumin_molar_conc(albumin_g_dL)

    # Infer Kd for each drug from their measured fu_alone
    kd_a = _infer_kd(fu_a_alone, albumin_g_dL, n_sites)
    kd_b = _infer_kd(fu_b_alone, albumin_g_dL, n_sites)

    # For the competition model we need free concentrations of B.
    # Approximate: C_B_free ≈ fu_b * C_B_total (ignoring second-order correction)
    # This is the standard simplification used in clinical DDI models.
    cb_free_mol_per_L: float
    if kd_b == float("inf"):
        cb_free_mol_per_L = 0.0
    else:
        # Drug B molecular weight unknown; use a representative MW of 400 g/mol
        # for unit conversion (mg/L → mol/L)
        mw_b_g_per_mol = 400.0
        cb_total_mol_per_L = (cb_mg_L / 1000.0) / mw_b_g_per_mol  # mg/L → g/L → mol/L
        cb_free_mol_per_L = fu_b_alone * cb_total_mol_per_L

    # Compute displaced free fractions
    if kd_b == float("inf") or cb_mg_L == 0.0:
        fu_a_disp = fu_a_alone
        fu_b_disp = fu_b_alone
    else:
        fu_a_disp = _fu_competition(kd_a, kd_b, p_mol_per_L, n_sites, cb_free_mol_per_L)
        # symmetric: B displaced by A
        mw_a_g_per_mol = 400.0
        ca_total_mol_per_L = (ca_mg_L / 1000.0) / mw_a_g_per_mol
        ca_free_mol_per_L = fu_a_alone * ca_total_mol_per_L
        fu_b_disp = _fu_competition(kd_b, kd_a, p_mol_per_L, n_sites, ca_free_mol_per_L)

    # Clamp to valid range
    fu_a_disp = min(1.0, max(fu_a_alone, fu_a_disp))
    fu_b_disp = min(1.0, max(fu_b_alone, fu_b_disp))

    delta_fu_a = fu_a_disp - fu_a_alone
    disp_pct_a = (delta_fu_a / fu_a_alone) * 100.0
    sig = _clinical_significance(delta_fu_a)

    notes = (
        f"Kd_A={kd_a:.2e} mol/L, Kd_B={kd_b:.2e} mol/L; "
        f"albumin={albumin_g_dL} g/dL, n_sites={n_sites}"
    )

    return PPBCompetitionResult(
        drug_a_name=drug_a_name,
        drug_b_name=drug_b_name,
        ca_mg_L=ca_mg_L,
        cb_mg_L=cb_mg_L,
        fu_a_alone=fu_a_alone,
        fu_b_alone=fu_b_alone,
        fu_a_displaced=fu_a_disp,
        fu_b_displaced=fu_b_disp,
        delta_fu_a=delta_fu_a,
        displacement_pct_a=disp_pct_a,
        clinical_significance=sig,
        notes=notes,
    )


def ppb_displacement_sensitivity(
    drug_a_name: str,
    drug_b_name: str,
    fu_a_alone: float,
    fu_b_alone: float,
    cb_range_mg_L: list[float],
    ca_mg_L: float = 10.0,
    albumin_g_dL: float = 4.0,
    n_sites: int = 1,
) -> list[PPBCompetitionResult]:
    """Sweep Drug B concentrations to quantify displacement of Drug A.

    Parameters
    ----------
    drug_a_name, drug_b_name, fu_a_alone, fu_b_alone:
        As in :func:`compute_ppb_competition`.
    cb_range_mg_L:
        List of Drug B total plasma concentrations (mg/L) to evaluate.
    ca_mg_L:
        Fixed Drug A total concentration (mg/L).  Default 10.0.
    albumin_g_dL:
        Plasma albumin (g/dL).  Default 4.0.
    n_sites:
        Binding sites per albumin.  Default 1.

    Returns
    -------
    list[PPBCompetitionResult]
        One result per entry in ``cb_range_mg_L``.
    """
    if not cb_range_mg_L:
        raise ValueError("cb_range_mg_L must not be empty")

    return [
        compute_ppb_competition(
            drug_a_name=drug_a_name,
            drug_b_name=drug_b_name,
            ca_mg_L=ca_mg_L,
            cb_mg_L=cb,
            fu_a_alone=fu_a_alone,
            fu_b_alone=fu_b_alone,
            albumin_g_dL=albumin_g_dL,
            n_sites=n_sites,
        )
        for cb in cb_range_mg_L
    ]
