"""Tissue binding kinetics simulation — Phase 474.

Models drug binding to tissue proteins (non-specific and specific),
and how that binding affects apparent volume of distribution and free
drug fraction over time.

ODE model (2-compartment: plasma + tissue):
- d(C_plasma_total)/dt = input - ke_plasma * C_plasma_free - k12 + k21
- k12 = Q * C_plasma_free / V_plasma
- k21 = Q * C_tissue_free / V_tissue
- C_plasma_free = C_plasma_total * plasma_fu
- Tissue binding modelled via Langmuir saturable binding (QSS):
    C_tissue_bound = Bmax * C_tissue_free / (Kd + C_tissue_free)

Forward Euler integration (pure Python, no numpy/scipy).
"""

from __future__ import annotations

from dataclasses import dataclass

# Molecular weight used when converting nM concentrations to mg/kg
# Callers supply the drug's MW via the function signature.

# Conversion: 1 nM * MW (g/mol) / 1e6 = mg/L  → mg/kg using tissue density ~1 g/mL
_NM_TO_MG_PER_KG_FACTOR = 1e-6  # nM * MW = µg/L = ng/mL; /1000 => µg/mL = mg/L


@dataclass(frozen=True)
class TissueBindingResult:
    """Results from a tissue binding kinetics simulation."""

    drug_name: str
    times_h: list[float]
    c_plasma_free_mg_L: list[float]
    c_tissue_bound_mg_kg: list[float]
    c_tissue_free_mg_kg: list[float]
    vd_apparent_L: float
    t_half_h: float
    kp_tissue: float
    notes: str


def calculate_tissue_fu(
    tissue_kd_nM: float,
    tissue_bmax_nM: float,
    free_conc_nM: float,
) -> float:
    """Compute tissue unbound fraction using Langmuir binding.

    fu_tissue = C_free / C_total = C_free / (C_free + C_bound)
    C_bound = Bmax * C_free / (Kd + C_free)

    Parameters
    ----------
    tissue_kd_nM:
        Dissociation constant (nM). Must be > 0.
    tissue_bmax_nM:
        Maximum binding capacity (nM). Must be >= 0.
    free_conc_nM:
        Free (unbound) drug concentration in tissue (nM). Must be >= 0.

    Returns
    -------
    float
        Unbound fraction in tissue (0–1).
    """
    if tissue_kd_nM <= 0.0:
        raise ValueError("tissue_kd_nM must be > 0")
    if tissue_bmax_nM < 0.0:
        raise ValueError("tissue_bmax_nM must be >= 0")
    if free_conc_nM < 0.0:
        raise ValueError("free_conc_nM must be >= 0")
    if tissue_bmax_nM == 0.0 or free_conc_nM == 0.0:
        return 1.0
    c_bound = tissue_bmax_nM * free_conc_nM / (tissue_kd_nM + free_conc_nM)
    c_total = free_conc_nM + c_bound
    if c_total <= 0.0:
        return 1.0
    return float(free_conc_nM / c_total)


def apparent_vd(
    vd_plasma_L: float,
    vd_tissue_L: float,
    plasma_fu: float,
    tissue_fu: float,
) -> float:
    """Compute apparent volume of distribution.

    Vd_app = Vp + (fu_plasma / fu_tissue) * Vt

    Parameters
    ----------
    vd_plasma_L:
        Plasma volume (L). Must be > 0.
    vd_tissue_L:
        Tissue volume (L). Must be > 0.
    plasma_fu:
        Unbound fraction in plasma (0–1). Must be in (0, 1].
    tissue_fu:
        Unbound fraction in tissue (0–1). Must be in (0, 1].

    Returns
    -------
    float
        Apparent volume of distribution (L).
    """
    if vd_plasma_L <= 0.0:
        raise ValueError("vd_plasma_L must be > 0")
    if vd_tissue_L <= 0.0:
        raise ValueError("vd_tissue_L must be > 0")
    if plasma_fu <= 0.0 or plasma_fu > 1.0:
        raise ValueError("plasma_fu must be in (0, 1]")
    if tissue_fu <= 0.0 or tissue_fu > 1.0:
        raise ValueError("tissue_fu must be in (0, 1]")
    return float(vd_plasma_L + (plasma_fu / tissue_fu) * vd_tissue_L)


def kp_from_binding(plasma_fu: float, tissue_fu: float) -> float:
    """Compute tissue-to-plasma partition coefficient from free fractions.

    Kp = fu_plasma / fu_tissue

    Parameters
    ----------
    plasma_fu:
        Plasma unbound fraction (0–1]. Must be > 0.
    tissue_fu:
        Tissue unbound fraction (0–1]. Must be > 0.

    Returns
    -------
    float
        Kp value. Equal fu values → Kp = 1.
    """
    if plasma_fu <= 0.0 or plasma_fu > 1.0:
        raise ValueError("plasma_fu must be in (0, 1]")
    if tissue_fu <= 0.0 or tissue_fu > 1.0:
        raise ValueError("tissue_fu must be in (0, 1]")
    return float(plasma_fu / tissue_fu)


def simulate_tissue_binding(
    drug_name: str,
    dose_mg: float,
    cl_L_per_h: float,
    plasma_fu: float,
    tissue_kd_nM: float,
    tissue_bmax_nM: float,
    vd_plasma_L: float = 5.0,
    vd_tissue_L: float = 65.0,
    t_end_h: float = 24.0,
    dt_h: float = 0.1,
    route: str = "iv",
) -> TissueBindingResult:
    """Simulate plasma and tissue PK with saturable tissue binding.

    Two-compartment model (plasma + tissue) with forward Euler integration.

    Plasma compartment:
        C_plasma_total  (mg/L)
        C_plasma_free  = plasma_fu * C_plasma_total

    Tissue compartment:
        C_tissue_free  (mg/kg, tissue density ~1 g/mL assumed)
        C_tissue_bound = Bmax_mg_kg * C_tissue_free / (Kd_mg_kg + C_tissue_free)

    ODE (plasma):
        dC_p/dt = -ke * C_p_free - k12 * C_p_free + k21 * C_t_free
    ODE (tissue free):
        dC_t_free/dt = k12 * C_p_free - k21 * C_t_free

    ke = CL_free / V_plasma  (clearance acts on free plasma drug)
    k12 = Q / V_plasma;  k21 = Q / V_tissue  (inter-compartmental exchange)

    Parameters
    ----------
    drug_name:
        Identifier for the drug.
    dose_mg:
        Administered dose (mg).
    cl_L_per_h:
        Intrinsic systemic clearance (L/h). Operates on free drug.
    plasma_fu:
        Plasma unbound fraction (0–1].
    tissue_kd_nM:
        Tissue binding dissociation constant (nM). Must be > 0.
    tissue_bmax_nM:
        Tissue maximum binding capacity (nM). Must be >= 0.
    vd_plasma_L:
        Plasma (central) volume (L).
    vd_tissue_L:
        Tissue volume (L).
    t_end_h:
        Simulation end time (h).
    dt_h:
        Forward Euler time step (h).
    route:
        Dosing route: 'iv' (bolus) or 'oral' (first-order absorption).

    Returns
    -------
    TissueBindingResult
    """
    # --- Input validation ---
    if not drug_name or not isinstance(drug_name, str):
        raise ValueError("drug_name must be a non-empty string")
    if dose_mg <= 0.0:
        raise ValueError("dose_mg must be > 0")
    if cl_L_per_h <= 0.0:
        raise ValueError("cl_L_per_h must be > 0")
    if plasma_fu <= 0.0 or plasma_fu > 1.0:
        raise ValueError("plasma_fu must be in (0, 1]")
    if tissue_kd_nM <= 0.0:
        raise ValueError("tissue_kd_nM must be > 0")
    if tissue_bmax_nM < 0.0:
        raise ValueError("tissue_bmax_nM must be >= 0")
    if vd_plasma_L <= 0.0:
        raise ValueError("vd_plasma_L must be > 0")
    if vd_tissue_L <= 0.0:
        raise ValueError("vd_tissue_L must be > 0")
    if t_end_h <= 0.0:
        raise ValueError("t_end_h must be > 0")
    if dt_h <= 0.0:
        raise ValueError("dt_h must be > 0")
    if route not in ("iv", "oral"):
        raise ValueError("route must be 'iv' or 'oral'")

    # --- Derived rate constants ---
    # Effective elimination rate (on free drug in plasma)
    ke = cl_L_per_h / vd_plasma_L  # h^-1

    # Intercompartmental transfer rate — tissue equilibration half-life ~2h
    Q = 0.5 * vd_plasma_L  # L/h (empirical flow)
    k12 = Q / vd_plasma_L  # h^-1
    k21 = Q / vd_tissue_L  # h^-1

    # Oral absorption (first-order ka, default 1.5 h^-1)
    ka_oral = 1.5  # h^-1
    f_oral = 0.9

    # --- Convert Kd and Bmax from nM to mg/kg ---
    # Assume representative MW ~350 g/mol for unit conversion
    # nM * MW_g_mol / 1e6 => mg/L; tissue density 1 g/mL => mg/kg == mg/L
    # We use a fixed nominal MW=350 for the nM conversion, stored internally.
    # This preserves linear scaling; the caller sets tissue_kd/bmax in nM.
    _mw_nominal = 350.0  # g/mol (assumed for nM→mg/L conversion)
    kd_mg_kg = tissue_kd_nM * _mw_nominal / 1.0e6  # mg/kg
    bmax_mg_kg = tissue_bmax_nM * _mw_nominal / 1.0e6  # mg/kg

    # --- Initial conditions ---
    if route == "iv":
        c_plasma = dose_mg / vd_plasma_L  # mg/L
        gut_amount = 0.0
    else:
        c_plasma = 0.0
        gut_amount = dose_mg  # mg

    c_tissue_free = 0.0  # mg/kg

    # --- Time grid ---
    n_steps = int(t_end_h / dt_h) + 1

    times_h: list[float] = []
    c_plasma_free_list: list[float] = []
    c_tissue_free_list: list[float] = []
    c_tissue_bound_list: list[float] = []

    for step in range(n_steps):
        t = step * dt_h
        times_h.append(t)

        c_plasma_free = plasma_fu * c_plasma
        c_plasma_free_list.append(c_plasma_free)

        # Tissue bound via Langmuir (QSS)
        if bmax_mg_kg > 0.0 and c_tissue_free > 0.0:
            c_tissue_bound = bmax_mg_kg * c_tissue_free / (kd_mg_kg + c_tissue_free)
        else:
            c_tissue_bound = 0.0

        c_tissue_free_list.append(c_tissue_free)
        c_tissue_bound_list.append(c_tissue_bound)

        if step == n_steps - 1:
            break

        # --- Forward Euler: plasma ---
        if route == "oral":
            d_gut = -ka_oral * gut_amount * dt_h
            absorption_flux = ka_oral * gut_amount * f_oral / vd_plasma_L  # mg/L/h
        else:
            d_gut = 0.0
            absorption_flux = 0.0

        dc_plasma = (
            absorption_flux - ke * c_plasma_free - k12 * c_plasma_free + k21 * c_tissue_free
        ) * dt_h

        # --- Forward Euler: tissue free drug ---
        dc_tissue_free = (k12 * c_plasma_free - k21 * c_tissue_free) * dt_h

        # Update states
        gut_amount = max(gut_amount + d_gut, 0.0)
        c_plasma = max(c_plasma + dc_plasma, 0.0)
        c_tissue_free = max(c_tissue_free + dc_tissue_free, 0.0)

    # --- Compute summary metrics ---

    # Peak plasma free concentration
    c_pf_max = max(c_plasma_free_list) if c_plasma_free_list else 0.0

    # Half-life: find time for free plasma to decay to half of Cmax
    half_val = 0.5 * c_pf_max
    t_half_h = t_end_h  # default
    found_peak = False
    for t, cpf in zip(times_h, c_plasma_free_list, strict=False):
        if not found_peak and cpf >= c_pf_max * 0.99:
            found_peak = True
        if found_peak and cpf <= half_val:
            t_half_h = float(t)
            break

    # Apparent Vd: use ratio at steady-state (late time)
    # Vd_app = Vp + (fu_p / fu_t) * Vt
    # Estimate fu_t from tissue concentrations at Cmax index
    cmax_idx = 0
    for i, v in enumerate(c_plasma_free_list):
        if v >= c_pf_max * 0.99:
            cmax_idx = i
            break

    c_tf_at_cmax = c_tissue_free_list[cmax_idx] if cmax_idx < len(c_tissue_free_list) else 0.0
    c_tb_at_cmax = c_tissue_bound_list[cmax_idx] if cmax_idx < len(c_tissue_bound_list) else 0.0
    c_t_total_at_cmax = c_tf_at_cmax + c_tb_at_cmax

    if c_t_total_at_cmax > 0.0:
        tissue_fu_at_cmax = c_tf_at_cmax / c_t_total_at_cmax
    else:
        tissue_fu_at_cmax = 1.0

    tissue_fu_eff = max(tissue_fu_at_cmax, 1e-6)
    vd_app = apparent_vd(vd_plasma_L, vd_tissue_L, plasma_fu, tissue_fu_eff)

    # Kp_tissue
    kp_tissue = kp_from_binding(plasma_fu, tissue_fu_eff)

    notes = (
        f"2-cpt tissue binding; route={route}; "
        f"plasma_fu={plasma_fu}; tissue_kd_nM={tissue_kd_nM}; "
        f"tissue_bmax_nM={tissue_bmax_nM}; "
        f"vd_plasma={vd_plasma_L} L; vd_tissue={vd_tissue_L} L."
    )

    return TissueBindingResult(
        drug_name=drug_name,
        times_h=times_h,
        c_plasma_free_mg_L=c_plasma_free_list,
        c_tissue_bound_mg_kg=c_tissue_bound_list,
        c_tissue_free_mg_kg=c_tissue_free_list,
        vd_apparent_L=vd_app,
        t_half_h=t_half_h,
        kp_tissue=kp_tissue,
        notes=notes,
    )
