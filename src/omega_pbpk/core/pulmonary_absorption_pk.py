"""Phase 1093 — Pulmonary Drug Absorption (Inhaled Therapeutics).

Models absorption of inhaled drugs (bronchodilators, corticosteroids) from the
lungs into systemic circulation.  Three regions: tracheobronchial (TB),
alveolar, and GI (swallowed fraction).
"""

from __future__ import annotations

from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class PulmonaryAbsorptionResult:
    """Result of a pulmonary absorption simulation."""

    drug_name: str
    dose_mg: float
    particle_size_um: float
    times_h: list
    c_plasma_mg_L: list
    a_alveolar_mg: list
    a_tb_mg: list
    f_lung_absorbed: float
    f_gi_absorbed: float
    f_total_absorbed: float
    cmax: float
    tmax_h: float
    auc: float
    lung_selectivity_ratio: float
    notes: str


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

# Particle deposition fractions keyed by (tb, alv, gi)
_DEPOSITION: dict[str, tuple[float, float, float]] = {
    "1-3": (0.1, 0.6, 0.3),
    "3-5": (0.3, 0.3, 0.4),
    "5-10": (0.5, 0.05, 0.45),
    ">10": (0.0, 0.0, 1.0),
}

# Rate constants (per hour)
_K_ABS_ALV = 10.0
_K_ABS_TB = 0.2
_K_MC = 0.5  # mucociliary clearance TB → GI
_K_ABS_GI = 0.3


def _deposition_fractions(particle_size_um: float) -> tuple[float, float, float]:
    """Return (f_tb, f_alv, f_gi) for the given particle size in µm."""
    if particle_size_um <= 3.0:
        return _DEPOSITION["1-3"]
    elif particle_size_um <= 5.0:
        return _DEPOSITION["3-5"]
    elif particle_size_um <= 10.0:
        return _DEPOSITION["5-10"]
    else:
        return _DEPOSITION[">10"]


def _trapz(times: list, values: list) -> float:
    """Trapezoidal integration."""
    result = 0.0
    for i in range(1, len(times)):
        result += 0.5 * (values[i - 1] + values[i]) * (times[i] - times[i - 1])
    return result


# ---------------------------------------------------------------------------
# Main simulation function
# ---------------------------------------------------------------------------


def simulate_pulmonary_absorption(
    drug_name: str,
    dose_mg: float,
    particle_size_um: float = 3.0,
    cl_L_per_h: float = 10.0,
    vd_L: float = 50.0,
    t_end_h: float = 12.0,
    dt_h: float = 0.05,
) -> PulmonaryAbsorptionResult:
    """Simulate pulmonary drug absorption via Forward Euler integration.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    dose_mg:
        Inhaled dose in mg.
    particle_size_um:
        Median particle/aerosol size in µm.
    cl_L_per_h:
        Systemic clearance in L/h.
    vd_L:
        Volume of distribution in L.
    t_end_h:
        Simulation end time in hours.
    dt_h:
        Forward-Euler step size in hours.
    """
    # --- Validation ---
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if particle_size_um <= 0:
        raise ValueError("particle_size_um must be > 0")
    if cl_L_per_h <= 0:
        raise ValueError("cl_L_per_h must be > 0")
    if vd_L <= 0:
        raise ValueError("vd_L must be > 0")

    ke = cl_L_per_h / vd_L  # elimination rate constant

    f_tb, f_alv, f_gi = _deposition_fractions(particle_size_um)

    # Initial conditions (amounts in mg, concentration in mg/L)
    a_tb = f_tb * dose_mg
    a_alv = f_alv * dose_mg
    a_gi = f_gi * dose_mg
    c_plasma = 0.0

    times: list[float] = []
    c_plasma_list: list[float] = []
    a_alveolar_list: list[float] = []
    a_tb_list: list[float] = []

    t = 0.0
    n_steps = int(round(t_end_h / dt_h))

    # Track absorbed amounts for bioavailability calculation
    absorbed_lung = 0.0  # via alveolar + TB absorption
    absorbed_gi = 0.0  # via GI absorption

    for _ in range(n_steps + 1):
        times.append(t)
        c_plasma_list.append(c_plasma)
        a_alveolar_list.append(a_alv)
        a_tb_list.append(a_tb)

        if _ < n_steps:
            # Rates
            rate_abs_alv = _K_ABS_ALV * a_alv
            rate_abs_tb = _K_ABS_TB * a_tb
            rate_mc = _K_MC * a_tb
            rate_abs_gi = _K_ABS_GI * a_gi

            # Absorbed amounts (for bioavailability tracking)
            absorbed_lung += (rate_abs_alv + rate_abs_tb) * dt_h
            absorbed_gi += rate_abs_gi * dt_h

            # ODEs – Forward Euler
            da_tb = -(rate_abs_tb + rate_mc) * dt_h
            da_alv = -rate_abs_alv * dt_h
            da_gi = (rate_mc - rate_abs_gi) * dt_h
            dc_plasma = ((rate_abs_alv + rate_abs_tb + rate_abs_gi) / vd_L - ke * c_plasma) * dt_h

            a_tb = max(0.0, a_tb + da_tb)
            a_alv = max(0.0, a_alv + da_alv)
            a_gi = max(0.0, a_gi + da_gi)
            c_plasma = max(0.0, c_plasma + dc_plasma)

            t += dt_h

    # --- Derived PK metrics ---
    auc = _trapz(times, c_plasma_list)
    cmax = max(c_plasma_list)
    tmax_h = times[c_plasma_list.index(cmax)]

    f_lung_absorbed = absorbed_lung / dose_mg
    f_gi_absorbed = absorbed_gi / dose_mg
    f_total_absorbed = min(1.0, f_lung_absorbed + f_gi_absorbed)

    if f_total_absorbed > 0:
        lung_selectivity_ratio = f_lung_absorbed / f_total_absorbed
    else:
        lung_selectivity_ratio = 0.0

    # Clamp fractions to [0, 1]
    f_lung_absorbed = min(1.0, max(0.0, f_lung_absorbed))
    f_gi_absorbed = min(1.0, max(0.0, f_gi_absorbed))
    lung_selectivity_ratio = min(1.0, max(0.0, lung_selectivity_ratio))

    notes = (
        f"Particle size {particle_size_um} µm → "
        f"TB={f_tb:.0%}, Alv={f_alv:.0%}, GI={f_gi:.0%}. "
        f"Lung selectivity: {lung_selectivity_ratio:.2f}."
    )

    return PulmonaryAbsorptionResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        particle_size_um=particle_size_um,
        times_h=times,
        c_plasma_mg_L=c_plasma_list,
        a_alveolar_mg=a_alveolar_list,
        a_tb_mg=a_tb_list,
        f_lung_absorbed=f_lung_absorbed,
        f_gi_absorbed=f_gi_absorbed,
        f_total_absorbed=f_total_absorbed,
        cmax=cmax,
        tmax_h=tmax_h,
        auc=auc,
        lung_selectivity_ratio=lung_selectivity_ratio,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Particle size optimiser
# ---------------------------------------------------------------------------

_REPRESENTATIVE_SIZES = [2.0, 4.0, 7.0, 12.0]


def optimize_particle_size(
    drug_name: str,
    dose_mg: float,
    target: str = "lung_selectivity",
) -> PulmonaryAbsorptionResult:
    """Return the particle size that maximises lung selectivity ratio.

    Representative sizes tested: 2, 4, 7, 12 µm.

    Parameters
    ----------
    drug_name:
        Name of the drug.
    dose_mg:
        Inhaled dose in mg.
    target:
        Optimisation target (currently only 'lung_selectivity' is supported).
    """
    best: PulmonaryAbsorptionResult | None = None
    for size in _REPRESENTATIVE_SIZES:
        result = simulate_pulmonary_absorption(
            drug_name=drug_name,
            dose_mg=dose_mg,
            particle_size_um=size,
        )
        if best is None or result.lung_selectivity_ratio > best.lung_selectivity_ratio:
            best = result
    assert best is not None
    return best
