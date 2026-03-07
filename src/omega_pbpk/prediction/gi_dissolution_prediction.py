"""GI Dissolution Prediction — Phase 1047.

Predict drug dissolution kinetics in simulated GI fluids at different segments.
"""

from __future__ import annotations

from dataclasses import dataclass

# ---------------------------------------------------------------------------
# GI segment properties
# ---------------------------------------------------------------------------

_GI_SEGMENTS: dict[str, dict[str, float]] = {
    "stomach": {"pH": 1.5, "bile_mM": 0.0, "volume_mL": 250.0, "transit_min": 30.0},
    "small_intestine": {"pH": 6.5, "bile_mM": 5.0, "volume_mL": 500.0, "transit_min": 180.0},
    "colon": {"pH": 7.0, "bile_mM": 1.0, "volume_mL": 150.0, "transit_min": 1200.0},
}

_VALID_IONIZATION = {"acid", "base", "neutral"}
_VALID_BCS_SOL = {"high", "low"}


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class GIDissolutionResult:
    drug_name: str
    gi_segment: str
    times_min: list
    fraction_dissolved: list
    dissolution_rate_mg_min: list
    t50_min: float
    t90_min: float
    intrinsic_dissolution_rate: float
    biorelevant_solubility_mg_mL: float
    bcs_absorption_potential: str
    notes: str


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _trapz(times: list[float], values: list[float]) -> float:
    """Manual trapezoidal integration."""
    area = 0.0
    for i in range(1, len(times)):
        area += 0.5 * (values[i] + values[i - 1]) * (times[i] - times[i - 1])
    return area


# ---------------------------------------------------------------------------
# Main prediction function
# ---------------------------------------------------------------------------


def predict_gi_dissolution(
    drug_name: str,
    mw_Da: float,
    logp: float,
    pka: float,
    ionization_type: str,
    dose_mg: float,
    particle_size_um: float = 50.0,
    gi_segment: str = "small_intestine",
    bcs_solubility_class: str = "low",
    surface_area_cm2: float = 1.0,
    t_end_min: float = 60.0,
    dt_min: float = 0.5,
) -> GIDissolutionResult:
    """Predict drug dissolution in simulated GI fluids.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    mw_Da:
        Molecular weight in Daltons.
    logp:
        Octanol-water partition coefficient.
    pka:
        Ionization constant.
    ionization_type:
        "acid", "base", or "neutral".
    dose_mg:
        Administered dose in mg.
    particle_size_um:
        Mean particle diameter in micrometers.
    gi_segment:
        GI compartment: "stomach", "small_intestine", or "colon".
    bcs_solubility_class:
        "high" or "low" (BCS class).
    surface_area_cm2:
        Total particle surface area in cm².
    t_end_min:
        Simulation duration in minutes.
    dt_min:
        Integration step size in minutes.

    Returns
    -------
    GIDissolutionResult
    """
    # --- validation ---
    if dose_mg <= 0:
        raise ValueError(f"dose_mg must be > 0, got {dose_mg}")
    if particle_size_um <= 0:
        raise ValueError(f"particle_size_um must be > 0, got {particle_size_um}")
    if surface_area_cm2 <= 0:
        raise ValueError(f"surface_area_cm2 must be > 0, got {surface_area_cm2}")
    if gi_segment not in _GI_SEGMENTS:
        raise ValueError(f"gi_segment must be one of {set(_GI_SEGMENTS)}, got {gi_segment!r}")
    if ionization_type not in _VALID_IONIZATION:
        raise ValueError(
            f"ionization_type must be one of {_VALID_IONIZATION}, got {ionization_type!r}"
        )
    if bcs_solubility_class not in _VALID_BCS_SOL:
        raise ValueError(
            f"bcs_solubility_class must be 'high' or 'low', got {bcs_solubility_class!r}"
        )

    seg = _GI_SEGMENTS[gi_segment]
    seg_pH = seg["pH"]
    bile_mM = seg["bile_mM"]

    # --- biorelevant solubility ---
    # Intrinsic aqueous solubility estimate
    s_intrinsic = (10 ** (-logp / 2)) * 10 / mw_Da * 1000  # mg/mL
    s_intrinsic = max(0.001, min(500.0, s_intrinsic))

    # Henderson-Hasselbalch ionization correction
    if ionization_type == "acid":
        s_corrected = s_intrinsic * (1 + 10 ** (seg_pH - pka))
    elif ionization_type == "base":
        s_corrected = s_intrinsic * (1 + 10 ** (pka - seg_pH))
    else:
        s_corrected = s_intrinsic

    # Bile solubilization
    bile_factor = 1 + 0.3 * bile_mM / 5.0
    s_bio = s_corrected * bile_factor
    s_bio = max(0.001, min(500.0, s_bio))

    # --- IDR (mg / min / cm²) ---
    k_d = 10 ** (-0.002 * mw_Da)
    idr = s_bio * k_d
    idr = max(1e-6, min(10.0, idr))

    # BCS high-solubility multiplier
    if bcs_solubility_class == "high":
        idr *= 5.0
        idr = min(10.0, idr)

    # --- ODE: shrinking sphere dissolution ---
    n_steps = max(2, int(t_end_min / dt_min) + 1)

    times_min: list[float] = []
    fraction_dissolved: list[float] = []
    dissolution_rate_mg_min: list[float] = []

    m_undissolved = dose_mg  # mass remaining undissolved
    fd = 0.0  # fraction dissolved

    for step in range(n_steps):
        t = step * dt_min
        times_min.append(t)
        fraction_dissolved.append(fd)
        rate = idr * surface_area_cm2 * max(0.0, (1.0 - fd) ** (2.0 / 3.0))
        dissolution_rate_mg_min.append(rate)

        # Forward Euler integration
        dm = rate * dt_min
        m_undissolved = max(0.0, m_undissolved - dm)
        fd = max(0.0, min(1.0, (dose_mg - m_undissolved) / dose_mg))

    # --- derived metrics ---
    t50 = t_end_min
    t90 = t_end_min

    for _i, (t, f) in enumerate(zip(times_min, fraction_dissolved)):
        if f >= 0.5 and t50 == t_end_min:
            t50 = t
        if f >= 0.9 and t90 == t_end_min:
            t90 = t
            break

    # t50 <= t90 by construction (monotone fd)

    if t50 < 15.0:
        bcs_absorption_potential = "high"
    elif t50 < 45.0:
        bcs_absorption_potential = "moderate"
    else:
        bcs_absorption_potential = "low"

    notes = (
        f"Segment pH={seg_pH:.1f}, bile={bile_mM:.1f} mM. "
        f"IDR={idr:.4e} mg/(min·cm²). "
        f"Biorelevant solubility={s_bio:.4f} mg/mL. "
        f"BCS class (solubility): {bcs_solubility_class}."
    )

    return GIDissolutionResult(
        drug_name=drug_name,
        gi_segment=gi_segment,
        times_min=times_min,
        fraction_dissolved=fraction_dissolved,
        dissolution_rate_mg_min=dissolution_rate_mg_min,
        t50_min=t50,
        t90_min=t90,
        intrinsic_dissolution_rate=idr,
        biorelevant_solubility_mg_mL=s_bio,
        bcs_absorption_potential=bcs_absorption_potential,
        notes=notes,
    )
