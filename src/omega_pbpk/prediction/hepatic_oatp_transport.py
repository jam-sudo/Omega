"""Phase 1107 — Drug Hepatic Sinusoidal Transport (OATP).

Models OATP1B1/OATP1B3-mediated hepatic uptake using the extended
clearance concept (active + passive uptake), well-stirred liver model,
and perpetrator-mediated inhibition leading to DDI AUCR calculation.
"""

from __future__ import annotations

from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Default substrate type parameters (Vmax pmol/min/mg protein, Km µM)
# ---------------------------------------------------------------------------
_SUBSTRATE_DEFAULTS: dict[str, tuple[float, float]] = {
    "statin": (500.0, 5.0),
    "sartan": (200.0, 10.0),
    "other": (100.0, 20.0),
}

# Passive permeability x SA / V (simplified, µL/min/mg protein)
_CLINT_PASSIVE_DEFAULT = 5.0  # µL/min/mg protein

# Hepatocyte protein content conversion: 1 g liver ≈ 120 mg microsomal protein;
# adult liver ~1500 g → ~180 000 mg protein.  Used to convert CLhep units.
# We keep internal units as µL/min/mg protein; convert to L/h at output.
# 1 µL/min/mg * 180 000 mg / 1e6 µL/L * 60 min/h  = 10.8  L/h per 1 µL/min/mg
_SCALE_TO_L_PER_H = 180_000 * 60 / 1_000_000  # = 10.8


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class HepaticOATPResult:
    """Result from OATP-mediated hepatic transport prediction."""

    victim_drug: str
    substrate_type: str
    perpetrator_drug: str
    clint_active_uL_per_min_per_mg: float
    clint_passive_uL_per_min_per_mg: float
    clint_total_uL_per_min_per_mg: float
    clhep_baseline_L_per_h: float
    clhep_inhibited_L_per_h: float
    aucr: float
    ddi_severity: str
    active_uptake_fraction: float
    regulatory_flag: bool
    notes: str


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _validate_inputs(
    substrate_type: str,
    ki_oatp_uM: float,
    fu_plasma: float,
    q_hepatic_L_per_h: float,
    vmax: float | None,
    km: float | None,
) -> None:
    if substrate_type not in _SUBSTRATE_DEFAULTS:
        raise ValueError(
            f"substrate_type must be one of {set(_SUBSTRATE_DEFAULTS)}; got {substrate_type!r}"
        )
    if ki_oatp_uM < 0:
        raise ValueError(f"ki_oatp_uM must be >= 0; got {ki_oatp_uM}")
    if not (0 < fu_plasma <= 1):
        raise ValueError(f"fu_plasma must be in (0, 1]; got {fu_plasma}")
    if q_hepatic_L_per_h <= 0:
        raise ValueError(f"q_hepatic_L_per_h must be > 0; got {q_hepatic_L_per_h}")
    if vmax is not None and vmax <= 0:
        raise ValueError(f"vmax_pmol_per_min_per_mg must be > 0; got {vmax}")
    if km is not None and km <= 0:
        raise ValueError(f"km_uM must be > 0; got {km}")


def _clint_active(vmax: float, km: float, fu_plasma: float) -> float:
    """CLint_active = (Vmax / Km) * fu_plasma  [µL/min/mg protein]."""
    return (vmax / km) * fu_plasma


def _clhep_well_stirred(q_h: float, fu: float, clint_total_uL_per_min_per_mg: float) -> float:
    """Well-stirred hepatic clearance in L/h.

    q_h is already in L/h.  We scale CLint to L/h first.
    """
    clint_L_per_h = clint_total_uL_per_min_per_mg * _SCALE_TO_L_PER_H
    numerator = q_h * fu * clint_L_per_h
    denominator = q_h + fu * clint_L_per_h
    return numerator / denominator


def _severity(aucr: float) -> str:
    if aucr < 1.25:
        return "none"
    if aucr < 2.0:
        return "mild"
    if aucr < 5.0:
        return "moderate"
    return "severe"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def predict_oatp_transport(
    victim_drug: str,
    substrate_type: str,
    perpetrator_drug: str = "",
    ki_oatp_uM: float = 0.0,
    perpetrator_cmax_uM: float = 0.0,
    fu_plasma: float = 0.1,
    q_hepatic_L_per_h: float = 90.0,
    vmax_pmol_per_min_per_mg: float | None = None,
    km_uM: float | None = None,
) -> HepaticOATPResult:
    """Predict OATP-mediated hepatic uptake and DDI AUCR.

    Parameters
    ----------
    victim_drug:
        Name of the victim (substrate) drug.
    substrate_type:
        One of "statin", "sartan", "other".
    perpetrator_drug:
        Name of the OATP inhibitor (empty string if none).
    ki_oatp_uM:
        Ki of perpetrator for OATP (µM).  0 → no inhibition.
    perpetrator_cmax_uM:
        Unbound Cmax of perpetrator at the OATP inlet (µM).
    fu_plasma:
        Unbound fraction in plasma of victim drug.
    q_hepatic_L_per_h:
        Hepatic blood flow (L/h), default 90 L/h (typical adult).
    vmax_pmol_per_min_per_mg:
        Vmax for active uptake; if None uses substrate_type defaults.
    km_uM:
        Km for active uptake; if None uses substrate_type defaults.

    Returns
    -------
    HepaticOATPResult
    """
    _validate_inputs(
        substrate_type, ki_oatp_uM, fu_plasma, q_hepatic_L_per_h, vmax_pmol_per_min_per_mg, km_uM
    )

    # Use defaults if not provided
    default_vmax, default_km = _SUBSTRATE_DEFAULTS[substrate_type]
    vmax = vmax_pmol_per_min_per_mg if vmax_pmol_per_min_per_mg is not None else default_vmax
    km = km_uM if km_uM is not None else default_km

    # Intrinsic clearances
    cli_active = _clint_active(vmax, km, fu_plasma)
    cli_passive = _CLINT_PASSIVE_DEFAULT
    cli_total = cli_active + cli_passive

    # Baseline hepatic clearance
    clhep_base = _clhep_well_stirred(q_hepatic_L_per_h, fu_plasma, cli_total)

    # Inhibited active uptake
    has_inhibitor = bool(perpetrator_drug) and ki_oatp_uM > 0 and perpetrator_cmax_uM > 0
    if has_inhibitor:
        inhibition_factor = 1.0 / (1.0 + perpetrator_cmax_uM / ki_oatp_uM)
        cli_active_inh = cli_active * inhibition_factor
    else:
        cli_active_inh = cli_active

    cli_total_inh = cli_active_inh + cli_passive
    clhep_inh = _clhep_well_stirred(q_hepatic_L_per_h, fu_plasma, cli_total_inh)

    # AUCR: when OATP is inhibited, hepatic uptake decreases → AUC increases
    # AUCR = CLhep_baseline / CLhep_inhibited
    if clhep_inh > 0:
        aucr = clhep_base / clhep_inh
    else:
        aucr = 1.0  # degenerate case

    # Active uptake fraction
    active_frac = cli_active / cli_total if cli_total > 0 else 0.0

    severity = _severity(aucr)
    reg_flag = aucr > 1.5

    notes_parts = [
        f"Substrate type: {substrate_type}.",
        f"Vmax={vmax:.1f} pmol/min/mg, Km={km:.1f} µM.",
    ]
    if has_inhibitor:
        notes_parts.append(
            f"Perpetrator {perpetrator_drug!r}: Ki={ki_oatp_uM} µM, "
            f"Cmax={perpetrator_cmax_uM} µM → AUCR={aucr:.2f} ({severity})."
        )
    else:
        notes_parts.append("No perpetrator; AUCR=1.0.")

    return HepaticOATPResult(
        victim_drug=victim_drug,
        substrate_type=substrate_type,
        perpetrator_drug=perpetrator_drug,
        clint_active_uL_per_min_per_mg=cli_active,
        clint_passive_uL_per_min_per_mg=cli_passive,
        clint_total_uL_per_min_per_mg=cli_total,
        clhep_baseline_L_per_h=clhep_base,
        clhep_inhibited_L_per_h=clhep_inh,
        aucr=aucr,
        ddi_severity=severity,
        active_uptake_fraction=active_frac,
        regulatory_flag=reg_flag,
        notes=" ".join(notes_parts),
    )


def screen_substrates(
    substrates: list[dict],
    perpetrator_drug: str,
    ki_oatp_uM: float,
    perpetrator_cmax_uM: float,
) -> list[HepaticOATPResult]:
    """Screen a list of victim substrates against a single perpetrator.

    Each entry in `substrates` is a dict with at minimum:
      - "victim_drug" (str)
      - "substrate_type" (str)
    Optional keys mirror keyword arguments of `predict_oatp_transport`.

    Returns results sorted by AUCR descending (highest DDI risk first).
    """
    results: list[HepaticOATPResult] = []
    for sub in substrates:
        victim = sub["victim_drug"]
        stype = sub["substrate_type"]
        result = predict_oatp_transport(
            victim_drug=victim,
            substrate_type=stype,
            perpetrator_drug=perpetrator_drug,
            ki_oatp_uM=ki_oatp_uM,
            perpetrator_cmax_uM=perpetrator_cmax_uM,
            fu_plasma=sub.get("fu_plasma", 0.1),
            q_hepatic_L_per_h=sub.get("q_hepatic_L_per_h", 90.0),
            vmax_pmol_per_min_per_mg=sub.get("vmax_pmol_per_min_per_mg", None),
            km_uM=sub.get("km_uM", None),
        )
        results.append(result)

    results.sort(key=lambda r: r.aucr, reverse=True)
    return results
