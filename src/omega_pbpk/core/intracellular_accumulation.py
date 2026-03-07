"""Phase 867 — Intracellular Drug Accumulation.

Models drug accumulation inside cells due to pH trapping, lipid partitioning,
and active transport using a 3-compartment Forward Euler ODE system.
"""

from dataclasses import dataclass

# pH constants
_PH_LYSOSOME = 5.0
_PH_CYTOSOL = 7.0

# Cell volume fractions
_V_CYTOSOL_FRACTION = 0.7
_V_LYSOSOME_FRACTION = 0.05
_V_CELL_L_PER_TOTAL_L = 0.3

_VALID_IONIZATION_TYPES = {"base", "acid", "neutral"}


def _trapezoidal_auc(times: list[float], values: list[float]) -> float:
    """Compute AUC using trapezoidal rule."""
    auc = 0.0
    for i in range(1, len(times)):
        auc += 0.5 * (values[i - 1] + values[i]) * (times[i] - times[i - 1])
    return auc


def _compute_kp_lysosome(ionization_type: str, pka_base: float) -> float:
    """Compute lysosomal partition coefficient based on pH trapping."""
    if ionization_type == "base":
        numerator = 1.0 + 10.0 ** (pka_base - _PH_LYSOSOME)
        denominator = 1.0 + 10.0 ** (pka_base - _PH_CYTOSOL)
        if denominator == 0.0:
            return 1.0
        return numerator / denominator
    return 1.0


def _classify_accumulation(kp_lysosome: float) -> str:
    """Classify lysosomal accumulation based on Kp."""
    if kp_lysosome < 2.0:
        return "negligible"
    elif kp_lysosome < 10.0:
        return "moderate"
    else:
        return "extensive"


@dataclass
class IntracellularAccumulationResult:
    """Result of intracellular accumulation simulation."""

    drug_name: str
    ionization_type: str
    pka_base: float
    times_h: list[float]
    c_plasma_mg_L: list[float]
    c_cytosol_mg_L: list[float]
    c_lysosome_mg_L: list[float]
    kp_lysosome: float
    cmax_plasma: float
    cmax_cytosol: float
    cmax_lysosome: float
    auc_plasma: float
    auc_cytosol: float
    auc_lysosome: float
    lysosome_to_plasma_ratio: float
    accumulation_class: str
    notes: str


def simulate_intracellular_accumulation(
    drug_name: str,
    dose_mg: float,
    ionization_type: str = "base",
    pka_base: float = 7.0,
    k_uptake_per_h: float = 0.5,
    k_efflux_per_h: float = 0.2,
    k_lyso_in_per_h: float = 0.3,
    vd_L: float = 10.0,
    cl_L_per_h: float = 1.0,
    t_end_h: float = 24.0,
    dt_h: float = 0.1,
) -> IntracellularAccumulationResult:
    """Simulate intracellular drug accumulation using 3-compartment Forward Euler ODE.

    Parameters
    ----------
    drug_name : str
        Name of the drug.
    dose_mg : float
        Dose in mg (must be > 0).
    ionization_type : str
        One of "base", "acid", "neutral".
    pka_base : float
        pKa of basic nitrogen (default 7.0); use 0 for neutral/acid.
    k_uptake_per_h : float
        Passive + active uptake rate (h^-1).
    k_efflux_per_h : float
        Efflux rate from cytosol to plasma (h^-1).
    k_lyso_in_per_h : float
        Lysosomal sequestration rate (h^-1).
    vd_L : float
        Apparent volume of distribution for plasma (L).
    cl_L_per_h : float
        Systemic clearance from plasma (L/h).
    t_end_h : float
        Simulation end time (h).
    dt_h : float
        Time step for Forward Euler (h).

    Returns
    -------
    IntracellularAccumulationResult
    """
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if ionization_type not in _VALID_IONIZATION_TYPES:
        raise ValueError(
            f"ionization_type must be one of {_VALID_IONIZATION_TYPES}, got '{ionization_type}'"
        )
    if vd_L <= 0:
        raise ValueError("vd_L must be > 0")
    if cl_L_per_h <= 0:
        raise ValueError("cl_L_per_h must be > 0")

    kp_lysosome = _compute_kp_lysosome(ionization_type, pka_base)
    k_lyso_out_per_h = k_lyso_in_per_h / kp_lysosome if kp_lysosome > 0 else k_lyso_in_per_h

    # Elimination rate constant
    ke = cl_L_per_h / vd_L

    # Initial conditions
    c_plasma = dose_mg / vd_L
    c_cytosol = 0.0
    c_lysosome = 0.0

    # V_cell per unit total volume
    v_cell = _V_CELL_L_PER_TOTAL_L

    times: list[float] = []
    c_plasma_list: list[float] = []
    c_cytosol_list: list[float] = []
    c_lysosome_list: list[float] = []

    t = 0.0
    n_steps = int(round(t_end_h / dt_h)) + 1

    for _ in range(n_steps):
        times.append(t)
        c_plasma_list.append(c_plasma)
        c_cytosol_list.append(c_cytosol)
        c_lysosome_list.append(c_lysosome)

        # ODEs
        dCp_dt = -k_uptake_per_h * c_plasma + k_efflux_per_h * c_cytosol - ke * c_plasma
        # cytosol: uptake from plasma scaled by cell volume fraction, efflux to plasma,
        # lysosomal in/out
        dCc_dt = (
            k_uptake_per_h * c_plasma / v_cell
            - k_efflux_per_h * c_cytosol
            - k_lyso_in_per_h * c_cytosol
            + k_lyso_out_per_h * c_lysosome
        )
        dCl_dt = k_lyso_in_per_h * c_cytosol - k_lyso_out_per_h * c_lysosome

        # Forward Euler update
        c_plasma = max(0.0, c_plasma + dCp_dt * dt_h)
        c_cytosol = max(0.0, c_cytosol + dCc_dt * dt_h)
        c_lysosome = max(0.0, c_lysosome + dCl_dt * dt_h)
        t = round(t + dt_h, 10)

    cmax_plasma = max(c_plasma_list)
    cmax_cytosol = max(c_cytosol_list)
    cmax_lysosome = max(c_lysosome_list)

    auc_plasma = _trapezoidal_auc(times, c_plasma_list)
    auc_cytosol = _trapezoidal_auc(times, c_cytosol_list)
    auc_lysosome = _trapezoidal_auc(times, c_lysosome_list)

    lysosome_to_plasma_ratio = auc_lysosome / auc_plasma if auc_plasma > 0.0 else 0.0

    accumulation_class = _classify_accumulation(kp_lysosome)

    notes = (
        f"kp_lysosome={kp_lysosome:.2f}; "
        f"accumulation={accumulation_class}; "
        f"ionization={ionization_type}"
    )

    return IntracellularAccumulationResult(
        drug_name=drug_name,
        ionization_type=ionization_type,
        pka_base=pka_base,
        times_h=times,
        c_plasma_mg_L=c_plasma_list,
        c_cytosol_mg_L=c_cytosol_list,
        c_lysosome_mg_L=c_lysosome_list,
        kp_lysosome=kp_lysosome,
        cmax_plasma=cmax_plasma,
        cmax_cytosol=cmax_cytosol,
        cmax_lysosome=cmax_lysosome,
        auc_plasma=auc_plasma,
        auc_cytosol=auc_cytosol,
        auc_lysosome=auc_lysosome,
        lysosome_to_plasma_ratio=lysosome_to_plasma_ratio,
        accumulation_class=accumulation_class,
        notes=notes,
    )


def compare_ionization_types(
    drug_name: str,
    dose_mg: float,
    pka_base: float = 8.0,
) -> list[IntracellularAccumulationResult]:
    """Compare intracellular accumulation across ionization types.

    Parameters
    ----------
    drug_name : str
        Name of the drug.
    dose_mg : float
        Dose in mg.
    pka_base : float
        pKa of basic nitrogen (used for "base" type; default 8.0).

    Returns
    -------
    list[IntracellularAccumulationResult]
        Results for "base", "acid", "neutral" sorted by lysosome_to_plasma_ratio descending.
    """
    results = []
    for ion_type in ["base", "acid", "neutral"]:
        r = simulate_intracellular_accumulation(
            drug_name=drug_name,
            dose_mg=dose_mg,
            ionization_type=ion_type,
            pka_base=pka_base,
        )
        results.append(r)

    results.sort(key=lambda r: r.lysosome_to_plasma_ratio, reverse=True)
    return results
