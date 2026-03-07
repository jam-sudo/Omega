"""Tissue binding kinetics — time-dependent drug binding to tissue proteins.

Models non-specific and specific binding via Langmuir isotherm with quasi-steady-state
assumption. Plasma PK drives tissue free drug; bound fraction computed at each step.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TissueBindingResult:
    drug_name: str
    tissue: str
    dose_mg: float

    kd_uM: float                   # Dissociation constant (µM)
    bmax_umol_per_g: float          # Maximum binding capacity (µmol/g tissue)

    times_h: list[float]
    c_free_tissue_uM: list[float]   # Free (unbound) tissue concentration (µM)
    c_bound_tissue_uM: list[float]  # Bound concentration (µmol/g -> stored as µM equiv)
    c_total_tissue_uM: list[float]  # Total tissue concentration
    c_plasma_mg_L: list[float]      # Plasma concentration (driver)

    # Equilibrium metrics
    f_bound_at_cmax: float          # Fraction bound at peak
    apparent_kp: float              # Apparent tissue/plasma partition coefficient
    true_kp: float                  # Unbound tissue/plasma partition (Kpu)

    # Kinetics
    t_equilibration_h: float        # Time to 90% of steady-state free tissue conc
    saturation_pct_at_cmax: float   # % of binding sites occupied at Cmax

    notes: str


def simulate_tissue_binding(
    drug_name: str,
    tissue: str,
    dose_mg: float,
    cl_L_per_h: float,
    vd_L: float,
    mw: float,
    kd_uM: float,
    bmax_umol_per_g: float,
    tissue_mass_g: float = 200.0,
    kp_true: float = 1.5,
    ka_tissue_per_h: float = 2.0,
    route: str = "oral",
    ka_per_h: float = 1.5,
    f_oral: float = 0.9,
    t_end_h: float = 24.0,
    dt_h: float = 0.1,
) -> TissueBindingResult:
    """Simulate time-dependent tissue binding kinetics driven by plasma PK.

    Uses a quasi-equilibrium (QSS) Langmuir binding model:
      c_bound(t) = bmax * c_free(t) / (kd + c_free(t))

    Plasma PK: standard 1-compartment oral or IV model (forward Euler).

    Parameters
    ----------
    drug_name : str
        Drug identifier.
    tissue : str
        Tissue name (e.g., "liver", "muscle").
    dose_mg : float
        Administered dose (mg).
    cl_L_per_h : float
        Systemic clearance (L/h).
    vd_L : float
        Volume of distribution (L).
    mw : float
        Molecular weight (g/mol).
    kd_uM : float
        Dissociation constant of tissue binding (µM).
    bmax_umol_per_g : float
        Maximum binding capacity (µmol/g tissue).
    tissue_mass_g : float
        Mass of the tissue compartment (g).
    kp_true : float
        True unbound tissue/plasma partition coefficient (Kpu).
    ka_tissue_per_h : float
        Rate constant for free drug equilibration with tissue (h^-1).
    route : str
        "oral" or "iv".
    ka_per_h : float
        Oral absorption rate constant (h^-1); ignored for IV.
    f_oral : float
        Oral bioavailability fraction.
    t_end_h : float
        Simulation duration (h).
    dt_h : float
        Time step (h).
    """
    # Input validation
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if cl_L_per_h <= 0:
        raise ValueError("cl_L_per_h must be > 0")
    if vd_L <= 0:
        raise ValueError("vd_L must be > 0")
    if kd_uM <= 0:
        raise ValueError("kd_uM must be > 0")
    if bmax_umol_per_g <= 0:
        raise ValueError("bmax_umol_per_g must be > 0")
    if mw <= 0:
        raise ValueError("mw must be > 0")
    if tissue_mass_g <= 0:
        raise ValueError("tissue_mass_g must be > 0")

    ke = cl_L_per_h / vd_L  # elimination rate constant (h^-1)

    n_steps = int(t_end_h / dt_h) + 1
    times_h: list[float] = []
    c_plasma_mg_L: list[float] = []
    c_free_tissue_uM: list[float] = []
    c_bound_tissue_uM: list[float] = []
    c_total_tissue_uM: list[float] = []

    # Initial conditions
    if route == "oral":
        a_gut = dose_mg  # amount in gut (mg) — whole body dose
        c_plasma = 0.0
    else:
        # IV bolus: distribute instantaneously
        a_gut = 0.0
        c_plasma = dose_mg / vd_L  # mg/L

    c_free = 0.0  # µM — free tissue drug

    for i in range(n_steps):
        t = i * dt_h
        times_h.append(t)
        c_plasma_mg_L.append(c_plasma)

        # Plasma concentration in µM
        c_plasma_uM = c_plasma * 1000.0 / mw

        # QSS Langmuir: c_bound at current c_free
        c_bound = bmax_umol_per_g * c_free / (kd_uM + c_free) if c_free > 0 else 0.0
        c_total = c_free + c_bound

        c_free_tissue_uM.append(c_free)
        c_bound_tissue_uM.append(c_bound)
        c_total_tissue_uM.append(c_total)

        # --- Forward Euler update ---
        # Plasma PK
        if route == "oral":
            dc_plasma = (ka_per_h * a_gut * f_oral / vd_L) - ke * c_plasma
            da_gut = -ka_per_h * a_gut
            a_gut = max(a_gut + da_gut * dt_h, 0.0)
        else:
            dc_plasma = -ke * c_plasma
        c_plasma = max(c_plasma + dc_plasma * dt_h, 0.0)

        # Tissue free drug driven by plasma (simplified: instantaneous redistribution
        # toward kp_true * c_plasma_uM)
        target_free = kp_true * c_plasma_uM
        dc_free = ka_tissue_per_h * (target_free - c_free)
        c_free = max(c_free + dc_free * dt_h, 0.0)

    # --- Compute metrics ---
    # Cmax index in plasma
    cmax_idx = int(max(range(len(c_plasma_mg_L)), key=lambda i: c_plasma_mg_L[i]))
    cmax_plasma = c_plasma_mg_L[cmax_idx]
    c_bound_at_cmax = c_bound_tissue_uM[cmax_idx]
    c_total_at_cmax = c_total_tissue_uM[cmax_idx]

    if c_total_at_cmax > 0:
        f_bound_at_cmax = c_bound_at_cmax / c_total_at_cmax
    else:
        f_bound_at_cmax = 0.0

    # Apparent Kp: (total tissue conc in mg/L equivalent) / plasma conc
    # Convert µM back to mg/L: c [µM] * mw / 1000
    if cmax_plasma > 0 and c_total_at_cmax > 0:
        c_total_mgL = c_total_at_cmax * mw / 1000.0
        apparent_kp = c_total_mgL / cmax_plasma
    else:
        apparent_kp = kp_true

    # Saturation at Cmax
    saturation_pct_at_cmax = min(c_bound_at_cmax / bmax_umol_per_g * 100.0, 100.0)

    # t_equilibration: time when c_free reaches 90% of its own maximum
    c_free_max = max(c_free_tissue_uM) if c_free_tissue_uM else 0.0
    threshold_90 = 0.9 * c_free_max
    t_equilibration_h = t_end_h  # default: never reached
    for idx, (t, cf) in enumerate(zip(times_h, c_free_tissue_uM)):
        if cf >= threshold_90:
            t_equilibration_h = t
            break

    notes = (
        f"QSS Langmuir binding; kp_true={kp_true}; "
        f"route={route}; tissue_mass_g={tissue_mass_g}"
    )

    return TissueBindingResult(
        drug_name=drug_name,
        tissue=tissue,
        dose_mg=dose_mg,
        kd_uM=kd_uM,
        bmax_umol_per_g=bmax_umol_per_g,
        times_h=times_h,
        c_free_tissue_uM=c_free_tissue_uM,
        c_bound_tissue_uM=c_bound_tissue_uM,
        c_total_tissue_uM=c_total_tissue_uM,
        c_plasma_mg_L=c_plasma_mg_L,
        f_bound_at_cmax=f_bound_at_cmax,
        apparent_kp=apparent_kp,
        true_kp=kp_true,
        t_equilibration_h=t_equilibration_h,
        saturation_pct_at_cmax=saturation_pct_at_cmax,
        notes=notes,
    )


def compare_tissue_binding_sites(
    drug_name: str,
    dose_mg: float,
    cl_L_per_h: float,
    vd_L: float,
    mw: float,
    tissues_params: list[dict],
) -> list[TissueBindingResult]:
    """Compare drug binding across multiple tissues.

    Parameters
    ----------
    tissues_params : list of dict
        Each entry must contain: {"tissue": str, "kd_uM": float, "bmax_umol_per_g": float}
        Optional keys forwarded to simulate_tissue_binding: tissue_mass_g, kp_true,
        ka_tissue_per_h, route, ka_per_h, f_oral, t_end_h, dt_h.

    Returns
    -------
    list[TissueBindingResult]
        Sorted by saturation_pct_at_cmax descending (highest saturation first).
    """
    results: list[TissueBindingResult] = []
    for params in tissues_params:
        tissue = params["tissue"]
        kd_uM = params["kd_uM"]
        bmax_umol_per_g = params["bmax_umol_per_g"]

        # Optional kwargs
        kwargs = {
            k: params[k]
            for k in (
                "tissue_mass_g",
                "kp_true",
                "ka_tissue_per_h",
                "route",
                "ka_per_h",
                "f_oral",
                "t_end_h",
                "dt_h",
            )
            if k in params
        }

        result = simulate_tissue_binding(
            drug_name=drug_name,
            tissue=tissue,
            dose_mg=dose_mg,
            cl_L_per_h=cl_L_per_h,
            vd_L=vd_L,
            mw=mw,
            kd_uM=kd_uM,
            bmax_umol_per_g=bmax_umol_per_g,
            **kwargs,
        )
        results.append(result)

    results.sort(key=lambda r: r.saturation_pct_at_cmax, reverse=True)
    return results
