"""Mechanistic drug solubility prediction — Phase 503."""

from dataclasses import dataclass


@dataclass(frozen=True)
class SolubilityProfileResult:
    """Result of mechanistic solubility prediction."""

    logP: float
    tm_C: float
    intrinsic_solubility_mg_mL: float
    solubility_class: str
    yalkowsky_estimate_mg_mL: float
    hbond_correction_factor: float
    dose_solubility_ratio_100mg: float
    potentially_solubility_limited: bool
    notes: str


def yalkowsky_solubility(logP: float, tm_C: float, mw: float = None) -> float:
    """Predict intrinsic aqueous solubility using the Yalkowsky-Valvani GSE.

    log(S_mg/mL) = -logP - 0.01*(Tm - 25) + correction
    correction = 0.5 for most drugs; +0.1 if MW > 500

    Args:
        logP: Octanol-water partition coefficient.
        tm_C: Melting point in Celsius.
        mw: Molecular weight in g/mol (optional, affects correction).

    Returns:
        Intrinsic solubility in mg/mL.

    Raises:
        ValueError: If tm_C <= 25 (crystalline drug requirement).
    """
    if tm_C <= 25.0:
        raise ValueError(f"Melting point must be > 25°C for crystalline drugs, got {tm_C}°C")

    correction = 0.5
    if mw is not None and mw > 500.0:
        correction += 0.1

    log_s = -logP - 0.01 * (tm_C - 25.0) + correction
    return 10.0**log_s


def alogps_correction(logP: float, n_hbd: int, n_hba: int, mw: float) -> float:
    """Compute H-bonding correction factor for solubility.

    Increases solubility by 0.15 per H-bond donor,
    decreases by 0.05 per H-bond acceptor.

    Args:
        logP: Octanol-water partition coefficient.
        n_hbd: Number of H-bond donors.
        n_hba: Number of H-bond acceptors.
        mw: Molecular weight in g/mol.

    Returns:
        Correction factor (multiplicative on solubility in log space as additive).
    """
    if n_hbd < 0:
        raise ValueError(f"n_hbd must be >= 0, got {n_hbd}")
    if n_hba < 0:
        raise ValueError(f"n_hba must be >= 0, got {n_hba}")
    if mw <= 0.0:
        raise ValueError(f"mw must be > 0, got {mw}")

    log_correction = 0.15 * n_hbd - 0.05 * n_hba
    return log_correction


def predict_intrinsic_solubility(
    logP: float,
    tm_C: float,
    n_hbd: int = 0,
    n_hba: int = 0,
    mw: float = 300.0,
) -> float:
    """Predict intrinsic aqueous solubility from thermodynamic principles.

    Combines Yalkowsky-Valvani GSE with H-bonding corrections.

    Args:
        logP: Octanol-water partition coefficient.
        tm_C: Melting point in Celsius.
        n_hbd: Number of H-bond donors.
        n_hba: Number of H-bond acceptors.
        mw: Molecular weight in g/mol.

    Returns:
        Intrinsic solubility in mg/mL.

    Raises:
        ValueError: If tm_C <= 25 or mw <= 0.
    """
    if mw <= 0.0:
        raise ValueError(f"mw must be > 0, got {mw}")

    # Base Yalkowsky estimate
    correction = 0.5
    if mw > 500.0:
        correction += 0.1

    if tm_C <= 25.0:
        raise ValueError(f"Melting point must be > 25°C for crystalline drugs, got {tm_C}°C")

    log_s_base = -logP - 0.01 * (tm_C - 25.0) + correction

    # H-bond correction
    hb_log_correction = alogps_correction(logP, n_hbd, n_hba, mw)

    log_s = log_s_base + hb_log_correction
    return 10.0**log_s


def solubility_class(solubility_mg_mL: float) -> str:
    """Classify solubility into BCS-inspired categories.

    Args:
        solubility_mg_mL: Solubility in mg/mL.

    Returns:
        One of: 'highly_soluble', 'soluble', 'sparingly_soluble', 'poorly_soluble'.

    Raises:
        ValueError: If solubility_mg_mL <= 0.
    """
    if solubility_mg_mL <= 0.0:
        raise ValueError(f"solubility_mg_mL must be > 0, got {solubility_mg_mL}")

    if solubility_mg_mL >= 1.0:
        return "highly_soluble"
    elif solubility_mg_mL >= 0.1:
        return "soluble"
    elif solubility_mg_mL >= 0.01:
        return "sparingly_soluble"
    else:
        return "poorly_soluble"


def predict_solubility_profile(
    logP: float,
    tm_C: float,
    n_hbd: int = 0,
    n_hba: int = 0,
    mw: float = 300.0,
) -> SolubilityProfileResult:
    """Generate a comprehensive solubility profile.

    Args:
        logP: Octanol-water partition coefficient.
        tm_C: Melting point in Celsius.
        n_hbd: Number of H-bond donors.
        n_hba: Number of H-bond acceptors.
        mw: Molecular weight in g/mol.

    Returns:
        SolubilityProfileResult with all relevant predictions.

    Raises:
        ValueError: If inputs are invalid.
    """
    if mw <= 0.0:
        raise ValueError(f"mw must be > 0, got {mw}")
    if tm_C <= 25.0:
        raise ValueError(f"Melting point must be > 25°C for crystalline drugs, got {tm_C}°C")
    if n_hbd < 0:
        raise ValueError(f"n_hbd must be >= 0, got {n_hbd}")
    if n_hba < 0:
        raise ValueError(f"n_hba must be >= 0, got {n_hba}")

    # Yalkowsky base estimate
    yalkowsky_est = yalkowsky_solubility(logP, tm_C, mw)

    # H-bond correction factor (as log-space additive)
    hb_log_corr = alogps_correction(logP, n_hbd, n_hba, mw)
    hbond_correction_factor = 10.0**hb_log_corr

    # Combined intrinsic solubility
    intrinsic_sol = yalkowsky_est * hbond_correction_factor

    # Classification
    sol_class = solubility_class(intrinsic_sol)

    # Dose/solubility ratio: 100 mg in 250 mL → threshold 0.4 mg/mL
    dose_threshold_mg_mL = 0.4  # 100 mg / 250 mL
    dose_sol_ratio = intrinsic_sol / dose_threshold_mg_mL
    potentially_limited = intrinsic_sol < dose_threshold_mg_mL

    # Build notes
    note_parts = []
    if logP > 3.0:
        note_parts.append("high logP reduces solubility")
    if tm_C > 200.0:
        note_parts.append("high melting point indicates strong crystal lattice")
    if mw > 500.0:
        note_parts.append("MW >500 Da applies BCS high-MW correction")
    if n_hbd > 3:
        note_parts.append("multiple H-bond donors improve solubility")
    if sol_class == "poorly_soluble":
        note_parts.append("BCS Class II/IV concern")
    if potentially_limited:
        note_parts.append("absorption may be solubility-limited at 100 mg dose")

    notes = "; ".join(note_parts) if note_parts else "no special flags"

    return SolubilityProfileResult(
        logP=logP,
        tm_C=tm_C,
        intrinsic_solubility_mg_mL=intrinsic_sol,
        solubility_class=sol_class,
        yalkowsky_estimate_mg_mL=yalkowsky_est,
        hbond_correction_factor=hbond_correction_factor,
        dose_solubility_ratio_100mg=dose_sol_ratio,
        potentially_solubility_limited=potentially_limited,
        notes=notes,
    )
