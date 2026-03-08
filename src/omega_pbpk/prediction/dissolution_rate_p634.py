"""
Phase 634 — Drug Dissolution Rate Prediction
Predict drug dissolution rate and extent using BCS framework and particle properties.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class DissolutionRateResult:
    """Result of dissolution rate prediction."""

    drug_name: str
    particle_size_um: float
    bulk_solubility_mg_mL: float
    dose_number: float
    noyes_whitney_k: float
    t_50pct_h: float
    t_80pct_h: float
    t_90pct_h: float
    fraction_dissolved_2h: float
    dissolution_efficiency: float
    bcs_class_predicted: int
    qc1_pass: bool
    absorption_rate_limited: bool
    notes: str


def _validate_inputs(
    dose_mg: float,
    particle_size_um: float,
    bulk_solubility_mg_mL: float,
    volume_dissolution_mL: float,
    bcs_perm_class: str,
) -> None:
    if dose_mg <= 0:
        raise ValueError(f"dose_mg must be > 0; got {dose_mg}")
    if particle_size_um <= 0:
        raise ValueError(f"particle_size_um must be > 0; got {particle_size_um}")
    if bulk_solubility_mg_mL <= 0:
        raise ValueError(f"bulk_solubility_mg_mL must be > 0; got {bulk_solubility_mg_mL}")
    if volume_dissolution_mL <= 0:
        raise ValueError(f"volume_dissolution_mL must be > 0; got {volume_dissolution_mL}")
    if bcs_perm_class not in ("high", "low"):
        raise ValueError(f"bcs_perm_class must be 'high' or 'low'; got {bcs_perm_class!r}")


def _compute_noyes_whitney_k(
    diffusion_coeff_cm2_s: float,
    bulk_solubility_mg_mL: float,
    drug_density_g_cm3: float,
    particle_size_um: float,
) -> float:
    """
    k_NW = 6 * D [cm2/h] * Cs [mg/mL] / (rho [mg/cm3] * d^2 [cm^2])   [/h]

    Parameters
    ----------
    diffusion_coeff_cm2_s : cm²/s
    bulk_solubility_mg_mL : mg/mL  (= mg/cm³)
    drug_density_g_cm3 : g/cm³
    particle_size_um : µm

    Returns
    -------
    float : rate constant [/h], clamped to [0.001, 100.0]
    """
    D_cm2_h = diffusion_coeff_cm2_s * 3600.0  # cm²/h
    Cs = bulk_solubility_mg_mL  # mg/mL = mg/cm³
    rho_mg_cm3 = drug_density_g_cm3 * 1000.0  # mg/cm³
    d_cm = particle_size_um * 1e-4  # cm

    k_NW = 6.0 * D_cm2_h * Cs / (rho_mg_cm3 * d_cm**2)

    # Clamp
    k_NW = max(0.001, min(100.0, k_NW))
    return k_NW


def _run_dissolution_simulation(
    dose_mg: float,
    bulk_solubility_mg_mL: float,
    volume_dissolution_mL: float,
    k_NW: float,
    t_end_h: float,
    dt_h: float,
):
    """
    Forward Euler dissolution simulation.

    State: A_undiss (mg) = undissolved drug amount
    dA_undiss/dt = -k_NW * A_undiss * (1 - C_dissolved/C_sat)   when C_dissolved < C_sat

    C_dissolved = (dose_mg - A_undiss) / volume_dissolution_mL
    Cap C_dissolved at C_sat.

    Returns
    -------
    times : list of float
    fractions : list of float (fraction dissolved)
    """
    C_sat = bulk_solubility_mg_mL

    A_undiss = dose_mg
    t = 0.0

    times = [0.0]
    fractions = [0.0]

    n_steps = int(round(t_end_h / dt_h))

    for _ in range(n_steps):
        # C_dissolved from mass balance
        mass_dissolved = dose_mg - A_undiss
        C_dissolved = mass_dissolved / volume_dissolution_mL
        if C_dissolved > C_sat:
            C_dissolved = C_sat

        # Driving force: (1 - C_dissolved/C_sat)
        if C_sat > 0:
            driving = 1.0 - C_dissolved / C_sat
        else:
            driving = 0.0

        if driving < 0.0:
            driving = 0.0

        dA_dt = -k_NW * A_undiss * driving

        A_undiss += dA_dt * dt_h
        if A_undiss < 0.0:
            A_undiss = 0.0

        t += dt_h
        frac = (dose_mg - A_undiss) / dose_mg
        if frac > 1.0:
            frac = 1.0

        times.append(t)
        fractions.append(frac)

    return times, fractions


def _find_time_at_fraction(times, fractions, target_frac):
    """Return first time when fraction >= target_frac, or -1.0 if not reached."""
    for t, f in zip(times, fractions):
        if f >= target_frac:
            return t
    return -1.0


def _fraction_at_time(times, fractions, target_t):
    """Return fraction at target_t (linear interpolation between steps)."""
    for i in range(1, len(times)):
        if times[i] >= target_t:
            # Interpolate
            t0, t1 = times[i - 1], times[i]
            f0, f1 = fractions[i - 1], fractions[i]
            if t1 == t0:
                return f1
            alpha = (target_t - t0) / (t1 - t0)
            return f0 + alpha * (f1 - f0)
    # Beyond end
    return fractions[-1]


def _trapezoidal_auc(x, y):
    """Manual trapezoidal AUC."""
    return sum(0.5 * (y[i] + y[i - 1]) * (x[i] - x[i - 1]) for i in range(1, len(x)))


def predict_dissolution_rate(
    drug_name: str,
    dose_mg: float,
    particle_size_um: float,
    bulk_solubility_mg_mL: float,
    drug_density_g_cm3: float = 1.2,
    diffusion_coeff_cm2_s: float = 5e-6,
    volume_dissolution_mL: float = 900.0,
    bcs_perm_class: str = "high",
    t_end_h: float = 4.0,
    dt_h: float = 0.01,
) -> DissolutionRateResult:
    """
    Predict dissolution rate and extent using Noyes-Whitney model.

    Parameters
    ----------
    drug_name : str
    dose_mg : float
    particle_size_um : float
    bulk_solubility_mg_mL : float
    drug_density_g_cm3 : float
    diffusion_coeff_cm2_s : float
    volume_dissolution_mL : float
    bcs_perm_class : str  — 'high' or 'low'
    t_end_h : float
    dt_h : float

    Returns
    -------
    DissolutionRateResult
    """
    _validate_inputs(
        dose_mg, particle_size_um, bulk_solubility_mg_mL, volume_dissolution_mL, bcs_perm_class
    )

    # Dose number
    dose_number = dose_mg / (bulk_solubility_mg_mL * volume_dissolution_mL)

    # Noyes-Whitney rate constant
    k_NW = _compute_noyes_whitney_k(
        diffusion_coeff_cm2_s, bulk_solubility_mg_mL, drug_density_g_cm3, particle_size_um
    )

    # Run simulation
    times, fractions = _run_dissolution_simulation(
        dose_mg, bulk_solubility_mg_mL, volume_dissolution_mL, k_NW, t_end_h, dt_h
    )

    # Key timepoints
    t_50pct = _find_time_at_fraction(times, fractions, 0.5)
    t_80pct = _find_time_at_fraction(times, fractions, 0.8)
    t_90pct = _find_time_at_fraction(times, fractions, 0.9)

    # Fraction at specific times
    frac_2h = _fraction_at_time(times, fractions, 2.0)
    frac_30min = _fraction_at_time(times, fractions, 0.5)

    # Dissolution efficiency: AUC under fraction_dissolved curve / t_end
    auc = _trapezoidal_auc(times, fractions)
    dissolution_efficiency = auc / t_end_h

    # BCS class
    high_sol = dose_number <= 1.0
    high_perm = bcs_perm_class == "high"

    if high_sol and high_perm:
        bcs_class = 1
    elif not high_sol and high_perm:
        bcs_class = 2
    elif high_sol and not high_perm:
        bcs_class = 3
    else:
        bcs_class = 4

    # QC1 pass: 80% dissolved at 30 min (0.5 h)
    qc1_pass = frac_30min >= 0.80

    # Absorption rate limited: t90 > 2h
    absorption_rate_limited = (t_90pct < 0.0) or (t_90pct > 2.0)

    notes = (
        f"Drug: {drug_name}, BCS Class {bcs_class} (Do={dose_number:.2f}, perm={bcs_perm_class}). "
        f"k_NW={k_NW:.4f}/h, f_dissolved@2h={frac_2h * 100:.1f}%. "
        f"QC1 ({'PASS' if qc1_pass else 'FAIL'}). "
        f"{'Absorption rate limited.' if absorption_rate_limited else 'Dissolution adequate.'}"
    )

    return DissolutionRateResult(
        drug_name=drug_name,
        particle_size_um=particle_size_um,
        bulk_solubility_mg_mL=bulk_solubility_mg_mL,
        dose_number=dose_number,
        noyes_whitney_k=k_NW,
        t_50pct_h=t_50pct,
        t_80pct_h=t_80pct,
        t_90pct_h=t_90pct,
        fraction_dissolved_2h=frac_2h,
        dissolution_efficiency=dissolution_efficiency,
        bcs_class_predicted=bcs_class,
        qc1_pass=qc1_pass,
        absorption_rate_limited=absorption_rate_limited,
        notes=notes,
    )
