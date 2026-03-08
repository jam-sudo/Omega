"""Phase 256: Splenic PK model.

Models drug distribution in the spleen, relevant for nanoparticle and
liposome capture. Uses a 2-compartment ODE (plasma + spleen) with
formulation-dependent uptake and release parameters.
"""

from dataclasses import dataclass

# Formulation-specific parameters: (k_spleen_uptake, k_spleen_release, f_spleen)
_FORMULATION_PARAMS = {
    "small_molecule": (0.02, 0.5, 0.02),
    "liposome": (0.3, 0.05, 0.15),
    "nanoparticle_100nm": (0.5, 0.02, 0.25),
    "nanoparticle_50nm": (0.2, 0.1, 0.10),
    "pegylated_nanoparticle": (0.1, 0.08, 0.05),
}

_VALID_FORMULATIONS = set(_FORMULATION_PARAMS.keys())

# Compartment volumes
_V_PLASMA = 3.0  # L
_V_SPLEEN = 0.2  # L

# Degradation in spleen
_K_DEG_SPLEEN = 0.01  # 1/h


def _trapz(times: list[float], values: list[float]) -> float:
    """Trapezoidal integration."""
    return sum(
        0.5 * (values[i] + values[i - 1]) * (times[i] - times[i - 1]) for i in range(1, len(times))
    )


@dataclass
class SplenicPKResult:
    """Result of splenic PK simulation."""

    drug_name: str
    dose_mg: float
    formulation_type: str
    times_h: list[float]
    c_plasma_mg_L: list[float]
    c_spleen_mg_L: list[float]
    cmax_plasma_mg_L: float
    cmax_spleen_mg_L: float
    tmax_spleen_h: float
    auc_plasma_mg_h_per_L: float
    auc_spleen_mg_h_per_L: float
    splenic_accumulation_ratio: float
    fraction_in_spleen_at_1h: float  # percent
    notes: str


def simulate_splenic_pk(
    drug_name: str,
    dose_mg: float,
    formulation_type: str,
    cl_L_per_h: float = 10.0,
    t_end_h: float = 48.0,
    dt_h: float = 0.05,
) -> SplenicPKResult:
    """Simulate drug distribution in plasma and spleen.

    Parameters
    ----------
    drug_name : str
        Name of the drug or formulation.
    dose_mg : float
        Administered dose in mg. Must be > 0.
    formulation_type : str
        One of: "small_molecule", "liposome", "nanoparticle_100nm",
        "nanoparticle_50nm", "pegylated_nanoparticle".
    cl_L_per_h : float
        Systemic clearance (L/h).
    t_end_h : float
        Simulation end time (h).
    dt_h : float
        ODE integration step size (h).

    Returns
    -------
    SplenicPKResult
    """
    if dose_mg <= 0:
        raise ValueError(f"dose_mg must be > 0, got {dose_mg}")
    if formulation_type not in _VALID_FORMULATIONS:
        raise ValueError(
            f"formulation_type must be one of {sorted(_VALID_FORMULATIONS)}, "
            f"got '{formulation_type}'"
        )

    k_uptake, k_release, f_spleen = _FORMULATION_PARAMS[formulation_type]
    ke = cl_L_per_h / _V_PLASMA  # elimination rate constant from plasma

    # Initial conditions
    c_plasma = dose_mg / _V_PLASMA  # mg/L
    c_spleen = 0.0  # mg/L

    times = []
    c_plasma_list = []
    c_spleen_list = []

    t = 0.0
    n_steps = int(round(t_end_h / dt_h))

    for step in range(n_steps + 1):
        t = step * dt_h
        times.append(t)
        c_plasma_list.append(c_plasma)
        c_spleen_list.append(c_spleen)

        if step < n_steps:
            # dC_plasma/dt = -ke*C_plasma - k_uptake*C_plasma + k_release*C_spleen*V_spleen/V_plasma
            dc_plasma = (
                -ke * c_plasma - k_uptake * c_plasma + k_release * c_spleen * _V_SPLEEN / _V_PLASMA
            )
            # dC_spleen/dt = k_uptake*C_plasma*V_plasma/V_spleen - k_release*C_spleen - k_deg*C_spleen
            dc_spleen = (
                k_uptake * c_plasma * _V_PLASMA / _V_SPLEEN
                - k_release * c_spleen
                - _K_DEG_SPLEEN * c_spleen
            )
            c_plasma = max(0.0, c_plasma + dc_plasma * dt_h)
            c_spleen = max(0.0, c_spleen + dc_spleen * dt_h)

    cmax_plasma = max(c_plasma_list)
    cmax_spleen = max(c_spleen_list)

    # tmax_spleen: time at which spleen concentration is maximum
    idx_tmax = c_spleen_list.index(cmax_spleen)
    tmax_spleen = times[idx_tmax]

    auc_plasma = _trapz(times, c_plasma_list)
    auc_spleen = _trapz(times, c_spleen_list)

    # splenic accumulation ratio
    if auc_plasma > 0:
        splenic_accumulation_ratio = auc_spleen / auc_plasma
    else:
        splenic_accumulation_ratio = 0.0

    # Fraction in spleen at 1h: C_spleen(1h)*V_spleen / dose * 100
    # Find index closest to 1h
    idx_1h = min(range(len(times)), key=lambda i: abs(times[i] - 1.0))
    fraction_in_spleen_at_1h = (c_spleen_list[idx_1h] * _V_SPLEEN / dose_mg) * 100.0

    # Build notes
    note_parts = [f"Formulation: {formulation_type}."]
    if splenic_accumulation_ratio > 2.0:
        note_parts.append("High splenic accumulation — MPS uptake dominant.")
    elif splenic_accumulation_ratio > 0.5:
        note_parts.append("Moderate splenic accumulation.")
    else:
        note_parts.append("Low splenic accumulation — primarily systemic distribution.")
    note_parts.append(f"Splenic Cmax {cmax_spleen:.3f} mg/L at Tmax {tmax_spleen:.1f} h.")
    notes = " ".join(note_parts)

    return SplenicPKResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        formulation_type=formulation_type,
        times_h=times,
        c_plasma_mg_L=c_plasma_list,
        c_spleen_mg_L=c_spleen_list,
        cmax_plasma_mg_L=cmax_plasma,
        cmax_spleen_mg_L=cmax_spleen,
        tmax_spleen_h=tmax_spleen,
        auc_plasma_mg_h_per_L=auc_plasma,
        auc_spleen_mg_h_per_L=auc_spleen,
        splenic_accumulation_ratio=splenic_accumulation_ratio,
        fraction_in_spleen_at_1h=fraction_in_spleen_at_1h,
        notes=notes,
    )


def compare_formulations(
    drug_name: str,
    dose_mg: float,
    cl_L_per_h: float = 10.0,
) -> list[SplenicPKResult]:
    """Compare all formulation types for splenic PK.

    Parameters
    ----------
    drug_name : str
        Name of the drug or formulation.
    dose_mg : float
        Administered dose in mg.
    cl_L_per_h : float
        Systemic clearance (L/h).

    Returns
    -------
    list of SplenicPKResult sorted by splenic_accumulation_ratio descending.
    """
    results = []
    for formulation in _VALID_FORMULATIONS:
        result = simulate_splenic_pk(
            drug_name=drug_name,
            dose_mg=dose_mg,
            formulation_type=formulation,
            cl_L_per_h=cl_L_per_h,
        )
        results.append(result)

    results.sort(key=lambda r: r.splenic_accumulation_ratio, reverse=True)
    return results
