"""Ion channel pharmacology prediction module.

Predicts ion channel block probability and pharmacological effects using the
Hill equation. Supports cardiac safety (hERG, Nav1.5, Cav1.2, KATP) and CNS
(Nav1.1, Nav1.2) ion channels.
"""

from __future__ import annotations

from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class IonChannelResult:
    """Result of ion channel block prediction for a given drug concentration."""

    drug_name: str
    channel_name: str
    ic50_nM: float
    drug_conc_nM: float
    hill_coef: float
    block_pct: float
    use_dependence: str  # "high", "moderate", "low"
    state_dependent: bool
    therapeutic_window_margin: float  # IC50_therapeutic / IC50_channel
    cardiac_safety_risk: str  # "low", "moderate", "high", "critical"
    cns_risk: str  # "low", "moderate", "high"
    notes: str


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _cardiac_risk_from_block(block_pct: float, scale: float = 1.0) -> str:
    """Classify cardiac risk from block_pct with optional scaling of thresholds."""
    if block_pct > 50.0 * scale:
        return "critical"
    if block_pct > 30.0 * scale:
        return "high"
    if block_pct > 10.0 * scale:
        return "moderate"
    return "low"


def _cns_risk_from_block(block_pct: float) -> str:
    """Classify CNS risk from block_pct."""
    if block_pct > 50.0:
        return "high"
    if block_pct > 20.0:
        return "moderate"
    return "low"


# ---------------------------------------------------------------------------
# Core prediction function
# ---------------------------------------------------------------------------


def predict_channel_block(
    drug_name: str,
    channel_name: str,
    drug_conc_nM: float,
    ic50_nM: float,
    hill_coef: float,
    therapeutic_ic50_nM: float,
) -> IonChannelResult:
    """Predict ion channel block percentage and safety risk for a drug.

    Uses the Hill equation: block_pct = 100 * C^n / (IC50^n + C^n)

    Args:
        drug_name: Name of the drug.
        channel_name: Target channel (e.g. "hERG", "Nav1.5", "Cav1.2").
        drug_conc_nM: Drug free concentration at the channel (nM). Must be >= 0.
        ic50_nM: IC50 for channel inhibition (nM). Must be > 0.
        hill_coef: Hill coefficient (cooperativity). Typically 1.0.
        therapeutic_ic50_nM: IC50 for therapeutic target (nM). Must be > 0.

    Returns:
        IonChannelResult with block percentage and safety classification.

    Raises:
        ValueError: If ic50_nM <= 0, drug_conc_nM < 0, or therapeutic_ic50_nM <= 0.
    """
    if ic50_nM <= 0:
        raise ValueError(f"ic50_nM must be > 0, got {ic50_nM}")
    if drug_conc_nM < 0:
        raise ValueError(f"drug_conc_nM must be >= 0, got {drug_conc_nM}")
    if therapeutic_ic50_nM <= 0:
        raise ValueError(f"therapeutic_ic50_nM must be > 0, got {therapeutic_ic50_nM}")

    # Hill equation
    if drug_conc_nM == 0.0:
        block_pct = 0.0
    else:
        c_n = drug_conc_nM**hill_coef
        ic50_n = ic50_nM**hill_coef
        block_pct = 100.0 * c_n / (ic50_n + c_n)

    # Use-dependence classification
    if block_pct > 70.0:
        use_dependence = "high"
    elif block_pct > 30.0:
        use_dependence = "moderate"
    else:
        use_dependence = "low"

    # State dependence: sensitive channels have low IC50
    state_dependent = ic50_nM < 100.0

    # Therapeutic window margin
    therapeutic_window_margin = therapeutic_ic50_nM / ic50_nM

    # Channel-specific risk classification
    ch = channel_name.lower()
    if ch == "herg":
        cardiac_safety_risk = _cardiac_risk_from_block(block_pct, scale=1.0)
        cns_risk = "low"
    elif ch == "nav1.5":
        cardiac_safety_risk = _cardiac_risk_from_block(block_pct, scale=1.0)
        cns_risk = "low"
    elif ch == "cav1.2":
        # Thresholds multiplied by 0.8 => effectively the same formula but
        # thresholds are 40/24/8 instead of 50/30/10
        cardiac_safety_risk = _cardiac_risk_from_block(block_pct, scale=0.8)
        cns_risk = "low"
    elif ch == "nav1.1":
        cardiac_safety_risk = "low"
        cns_risk = _cns_risk_from_block(block_pct)
    elif ch == "nav1.2":
        cardiac_safety_risk = "low"
        cns_risk = _cns_risk_from_block(block_pct)
    elif ch == "katp":
        # Thresholds multiplied by 1.2 => effectively the same formula but
        # thresholds are 60/36/12 instead of 50/30/10
        cardiac_safety_risk = _cardiac_risk_from_block(block_pct, scale=1.2)
        cns_risk = "low"
    else:
        # Unknown channel: use default thresholds for both
        cardiac_safety_risk = _cardiac_risk_from_block(block_pct, scale=1.0)
        cns_risk = _cns_risk_from_block(block_pct)

    notes = (
        f"{channel_name} block at {drug_conc_nM:.1f} nM: {block_pct:.1f}%. "
        f"IC50={ic50_nM:.1f} nM, Hill={hill_coef:.2f}. "
        f"Therapeutic window margin={therapeutic_window_margin:.2f} "
        f"({'safe' if therapeutic_window_margin >= 30 else 'concern' if therapeutic_window_margin >= 10 else 'risk'})."
    )

    return IonChannelResult(
        drug_name=drug_name,
        channel_name=channel_name,
        ic50_nM=ic50_nM,
        drug_conc_nM=drug_conc_nM,
        hill_coef=hill_coef,
        block_pct=block_pct,
        use_dependence=use_dependence,
        state_dependent=state_dependent,
        therapeutic_window_margin=therapeutic_window_margin,
        cardiac_safety_risk=cardiac_safety_risk,
        cns_risk=cns_risk,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Multi-channel screening
# ---------------------------------------------------------------------------


def screen_channels(
    drug_name: str,
    drug_conc_nM: float,
    channel_data: list[dict],
) -> list[IonChannelResult]:
    """Screen a drug against multiple ion channels.

    Args:
        drug_name: Name of the drug.
        drug_conc_nM: Drug concentration for all channels (nM).
        channel_data: List of dicts, each with keys:
            "name" (str), "ic50_nM" (float), "hill_coef" (float),
            "therapeutic_ic50_nM" (float).

    Returns:
        List of IonChannelResult sorted by block_pct descending (most blocked first).
    """
    results: list[IonChannelResult] = []
    for entry in channel_data:
        result = predict_channel_block(
            drug_name=drug_name,
            channel_name=entry["name"],
            drug_conc_nM=drug_conc_nM,
            ic50_nM=entry["ic50_nM"],
            hill_coef=entry["hill_coef"],
            therapeutic_ic50_nM=entry["therapeutic_ic50_nM"],
        )
        results.append(result)

    # Sort by block_pct descending
    results.sort(key=lambda r: r.block_pct, reverse=True)
    return results
