"""
Mucus Penetration Model
========================
Models drug diffusion through mucus layers in GI tract, lungs, and
cervicovaginal system. Mucus is a viscoelastic barrier that slows hydrophilic
and large molecules.

Key equations
-------------
- Stokes-Einstein free diffusivity: D_free = k_B * T / (3 * pi * eta * d)
- Hydrodynamic diameter: d_nm = 0.065 * MW^0.333 (protein-like scaling)
- Mucus partition factor: f_mucus = exp(-MW/1000 * mucus_solid_fraction * interaction_factor)
- Penetration flux: J = D_eff * C / h_mucus
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

# Physical constants
_KB = 1.380649e-23  # Boltzmann constant, J/K
_T_K = 310.15  # Body temperature, K (37 °C)
_ETA_PA_S = 1e-3  # Water viscosity approximation, Pa·s
_PI = math.pi

# Interaction factors per charge class
_INTERACTION_FACTOR: dict[str, float] = {
    "cationic": 2.0,  # mucus is anionic → repels cations more
    "neutral": 1.0,
    "anionic": 0.5,  # anionic drugs partially repelled by anionic mucus
}

# Route-specific mucus layer thickness (µm)
_ROUTE_H_UM: dict[str, float] = {
    "oral": 100.0,
    "inhaled": 10.0,
    "vaginal": 150.0,
}

_VALID_CHARGES = frozenset(_INTERACTION_FACTOR.keys())
_VALID_ROUTES = frozenset(_ROUTE_H_UM.keys())


@dataclass
class MucusDiffResult:
    """Mucus diffusivity calculation result."""

    mw: float
    logP: float
    charge: str
    mucus_solid_pct: float
    d_free_m2_s: float
    d_eff_m2_s: float
    f_mucus: float
    penetration_time_h: float  # for a 100 µm mucus layer


@dataclass
class MucusPKResult:
    """Mucus-limited PK simulation result."""

    drug_name: str
    dose_mg: float
    mw: float
    logP: float
    charge: str
    route: str
    mucus_diff: MucusDiffResult
    times_h: list[float] = field(default_factory=list)
    c_plasma_mg_L: list[float] = field(default_factory=list)
    cmax_mg_L: float = 0.0
    tmax_h: float = 0.0
    auc_mg_h_per_L: float = 0.0
    f_absorbed: float = 0.0
    notes: str = ""


def _validate_charge(charge: str) -> None:
    if charge not in _VALID_CHARGES:
        raise ValueError(f"charge must be one of {sorted(_VALID_CHARGES)}, got '{charge}'")


def mucus_diffusivity(
    mw: float,
    logP: float,
    charge: str,
    mucus_solid_pct: float = 4.0,
) -> MucusDiffResult:
    """
    Compute effective diffusivity of a drug in mucus.

    Parameters
    ----------
    mw : float
        Molecular weight (g/mol). Must be > 0.
    logP : float
        Octanol-water partition coefficient (log scale).
    charge : str
        Drug charge class: "cationic", "anionic", or "neutral".
    mucus_solid_pct : float
        Mucus solid content as percentage (default 4 %).

    Returns
    -------
    MucusDiffResult
    """
    if mw <= 0:
        raise ValueError("mw must be > 0")
    _validate_charge(charge)

    # Hydrodynamic diameter (protein-like scaling), convert nm → m
    d_nm = 0.065 * mw**0.333
    d_m = d_nm * 1e-9

    # Free diffusivity — Stokes-Einstein
    d_free = _KB * _T_K / (3.0 * _PI * _ETA_PA_S * d_m)

    # Mucus solid fraction (fraction, not percent)
    mucus_solid_fraction = mucus_solid_pct / 100.0

    # Interaction factor based on charge
    interaction_factor = _INTERACTION_FACTOR[charge]

    # Mucus partition factor
    f_mucus = math.exp(-mw / 1000.0 * mucus_solid_fraction * interaction_factor)
    f_mucus = max(1e-9, min(1.0, f_mucus))

    # Effective diffusivity
    d_eff = d_free * f_mucus

    # Penetration time for 100 µm layer: t = h² / (2 * D_eff)
    h_m = 100.0e-6  # 100 µm in metres
    penetration_time_s = h_m**2 / (2.0 * d_eff)
    penetration_time_h = penetration_time_s / 3600.0

    return MucusDiffResult(
        mw=mw,
        logP=logP,
        charge=charge,
        mucus_solid_pct=mucus_solid_pct,
        d_free_m2_s=d_free,
        d_eff_m2_s=d_eff,
        f_mucus=f_mucus,
        penetration_time_h=penetration_time_h,
    )


def simulate_mucus_pk(
    drug_name: str,
    dose_mg: float,
    mw: float,
    logP: float,
    charge: str,
    route: str,
    cl_L_per_h: float,
    vd_L: float,
    h_mucus_um: float = 100.0,
    mucus_solid_pct: float = 4.0,
    t_end_h: float = 12.0,
    dt_h: float = 0.1,
) -> MucusPKResult:
    """
    Simulate mucus-limited drug absorption and systemic PK.

    A mucus compartment (amount A_mucus) is drained into systemic 1-compartment
    (C_plasma) at a rate proportional to the diffusion flux through the mucus
    layer.  The epithelial absorption rate constant from the mucus side is:

        ka_mucus = D_eff / h²

    where h is the mucus layer thickness.

    Parameters
    ----------
    drug_name : str
    dose_mg : float > 0
    mw : float > 0
    logP : float
    charge : str  "cationic", "anionic", or "neutral"
    route : str   "oral", "inhaled", or "vaginal"
    cl_L_per_h : float > 0
    vd_L : float > 0
    h_mucus_um : float > 0   mucus layer thickness (µm)
    mucus_solid_pct : float  mucus solid content (%)
    t_end_h : float          simulation end time (h)
    dt_h : float             time step (h)

    Returns
    -------
    MucusPKResult
    """
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if mw <= 0:
        raise ValueError("mw must be > 0")
    if cl_L_per_h <= 0:
        raise ValueError("cl_L_per_h must be > 0")
    if vd_L <= 0:
        raise ValueError("vd_L must be > 0")
    if h_mucus_um <= 0:
        raise ValueError("h_mucus_um must be > 0")
    _validate_charge(charge)
    if route not in _VALID_ROUTES:
        raise ValueError(f"route must be one of {sorted(_VALID_ROUTES)}, got '{route}'")

    # Use route-specific default thickness when caller uses the function default (100 µm)
    # If the caller explicitly passes a different value, respect it.
    if h_mucus_um == 100.0:
        h_mucus_um = _ROUTE_H_UM[route]

    # Compute diffusivity
    diff = mucus_diffusivity(mw, logP, charge, mucus_solid_pct)

    # Mucus layer thickness in metres
    h_m = h_mucus_um * 1e-6

    # Absorption rate constant through mucus: ka_mucus (1/h)
    # D_eff / h² gives the Fickian exchange rate; cap to avoid stiff ODE
    ka_mucus_s = diff.d_eff_m2_s / (h_m**2)
    ka_mucus = min(ka_mucus_s * 3600.0, 100.0)  # convert to 1/h, cap at 100 h⁻¹

    # Fraction of dose that can permeate the mucus barrier.
    # f_mucus captures the equilibrium partitioning into/through mucus
    # (interaction with mucin fibres).  For a 100 µm reference layer the
    # ratio (h_reference / h_actual)^0.5 scales the barrier effect.
    h_ref_um = 100.0
    # Route-specific thickness is already embedded in h_mucus_um;
    # scale barrier relative to the reference 100 µm layer.
    barrier_scale = (h_mucus_um / h_ref_um) ** 0.5
    # Available fraction = f_mucus ^ barrier_scale  (more mucus → tighter barrier)
    f_avail = diff.f_mucus**barrier_scale
    f_avail = max(0.0, min(1.0, f_avail))

    # Effective absorbed dose after mucus barrier
    dose_avail_mg = dose_mg * f_avail

    ke = cl_L_per_h / vd_L  # elimination rate constant (1/h)

    # Forward Euler ODE: A_mucus (mg), C_plasma (mg/L)
    a_mucus = dose_avail_mg
    c_plasma = 0.0

    times: list[float] = [0.0]
    concs: list[float] = [0.0]

    t = 0.0
    n_steps = int(round(t_end_h / dt_h))

    absorbed_total = 0.0  # track absorbed mass (into plasma)

    for _ in range(n_steps):
        # Rate from mucus reservoir → plasma (mg/h)
        rate_abs = ka_mucus * a_mucus

        # Rate of elimination from plasma (mg/h)
        rate_elim = ke * c_plasma * vd_L

        da_mucus = -rate_abs * dt_h
        dc_plasma = (rate_abs - rate_elim) / vd_L * dt_h

        # Accumulate absorbed into plasma
        absorbed_total += rate_abs * dt_h

        a_mucus = max(0.0, a_mucus + da_mucus)
        c_plasma = max(0.0, c_plasma + dc_plasma)
        t += dt_h

        times.append(round(t, 10))
        concs.append(c_plasma)

    # PK parameters — f_absorbed relative to original dose
    cmax = max(concs)
    tmax = times[concs.index(cmax)]
    auc = sum(
        (concs[i] + concs[i - 1]) / 2.0 * (times[i] - times[i - 1]) for i in range(1, len(times))
    )
    f_absorbed = min(
        1.0, f_avail * min(1.0, absorbed_total / dose_avail_mg) if dose_avail_mg > 0 else 0.0
    )

    notes = (
        f"Route: {route}. Mucus h={h_mucus_um} µm, "
        f"ka_mucus={ka_mucus:.3f} h⁻¹, f_mucus={diff.f_mucus:.4f}. "
        f"f_absorbed={f_absorbed:.3f}."
    )

    return MucusPKResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        mw=mw,
        logP=logP,
        charge=charge,
        route=route,
        mucus_diff=diff,
        times_h=times,
        c_plasma_mg_L=concs,
        cmax_mg_L=cmax,
        tmax_h=tmax,
        auc_mg_h_per_L=auc,
        f_absorbed=f_absorbed,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Phase 441: simulate_mucus_penetration — 1D Fickian diffusion model
# ---------------------------------------------------------------------------

# Pore radii (nm) for each mucus type
_PORE_RADIUS_NM: dict[str, float] = {
    "gastric": 100.0,
    "intestinal": 200.0,
    "respiratory": 150.0,
    "cervical": 50.0,
}

_VALID_MUCUS_TYPES = set(_PORE_RADIUS_NM.keys())

# Simulation fixed geometry
_AREA_CM2 = 1.0  # cm²
_VOL_LUMEN_ML = 1.0  # mL
_VOL_EPITHELIUM_ML = 0.1  # mL
_K_ABS_PER_H = 1.0  # 1/h, epithelial → systemic absorption


@dataclass
class MucusPenetrationResult:
    """Result of mucus penetration simulation (Phase 441)."""

    drug_name: str
    dose_mg: float
    mucus_type: str
    mucus_thickness_um: float
    diffusion_coeff_effective: float  # cm²/s
    times_h: list[float] = field(default_factory=list)
    c_lumen_mg_mL: list[float] = field(default_factory=list)
    c_epithelium_mg_mL: list[float] = field(default_factory=list)
    f_penetrated: float = 0.0
    t50_penetration_h: float = float("nan")
    notes: str = ""


def _hydrodynamic_radius_nm(mw: float) -> float:
    """Estimate hydrodynamic radius from MW: r_h = 0.066 * mw^0.43 (nm)."""
    return 0.066 * (mw**0.43)


def _size_exclusion_factor(r_h_nm: float, pore_radius_nm: float) -> float:
    """Steric size-exclusion: factor = (1 - r_h/pore_radius)^2; 0 if r_h >= pore."""
    if r_h_nm >= pore_radius_nm:
        return 0.0
    return (1.0 - r_h_nm / pore_radius_nm) ** 2


def _charge_factor_penetration(charge: str) -> float:
    """Mucin is negatively charged. neutral=1.0, positive=0.3, negative=0.7."""
    c = charge.lower()
    if c in ("positive", "cationic", "+"):
        return 0.3
    if c in ("negative", "anionic", "-"):
        return 0.7
    return 1.0  # neutral


def _lipophilicity_factor_penetration(logP: float) -> float:
    """logP > 2 → 1.5 (partitions into mucus gel); else 1.0."""
    return 1.5 if logP > 2.0 else 1.0


def _effective_diffusion_cm2_s(
    d_free_cm2_s: float,
    mw: float,
    logP: float,
    charge: str,
    pore_radius_nm: float,
) -> float:
    """D_effective = D_free * size_factor * k_int * k_lipo."""
    r_h = _hydrodynamic_radius_nm(mw)
    f_size = _size_exclusion_factor(r_h, pore_radius_nm)
    k_int = _charge_factor_penetration(charge)
    k_lipo = _lipophilicity_factor_penetration(logP)
    return d_free_cm2_s * f_size * k_int * k_lipo


def simulate_mucus_penetration(
    drug_name: str,
    dose_mg: float,
    mucus_type: str,
    mucus_thickness_um: float,
    diffusion_coeff_cm2_s: float,
    mw: float,
    logP: float,
    charge: str,
    t_end_h: float = 4.0,
    dt_h: float = 0.01,
) -> MucusPenetrationResult:
    """Simulate drug penetration through a mucus barrier using 1D Fick diffusion.

    Parameters
    ----------
    drug_name : str
        Name of the drug compound.
    dose_mg : float
        Dose (mg) applied to the lumen compartment.
    mucus_type : str
        Type of mucus barrier: "gastric", "intestinal", "respiratory", "cervical".
    mucus_thickness_um : float
        Thickness of the mucus layer (micrometers, µm).
    diffusion_coeff_cm2_s : float
        Free aqueous diffusion coefficient (cm²/s).
    mw : float
        Molecular weight (g/mol).
    logP : float
        Lipophilicity (log octanol/water partition coefficient).
    charge : str
        Charge state: "neutral", "positive"/"cationic", or "negative"/"anionic".
    t_end_h : float
        Simulation end time (hours).
    dt_h : float
        Time step (hours).

    Returns
    -------
    MucusPenetrationResult
        Time-course concentrations and penetration summary metrics.
    """
    # --- Input validation ---
    if not drug_name or not drug_name.strip():
        raise ValueError("drug_name must be a non-empty string")
    if dose_mg <= 0:
        raise ValueError(f"dose_mg must be > 0, got {dose_mg}")
    if mucus_type not in _VALID_MUCUS_TYPES:
        raise ValueError(
            f"mucus_type must be one of {sorted(_VALID_MUCUS_TYPES)}, got '{mucus_type}'"
        )
    if mucus_thickness_um <= 0:
        raise ValueError(f"mucus_thickness_um must be > 0, got {mucus_thickness_um}")
    if diffusion_coeff_cm2_s <= 0:
        raise ValueError(f"diffusion_coeff_cm2_s must be > 0, got {diffusion_coeff_cm2_s}")
    if mw <= 0:
        raise ValueError(f"mw must be > 0, got {mw}")
    if t_end_h <= 0:
        raise ValueError(f"t_end_h must be > 0, got {t_end_h}")
    if dt_h <= 0:
        raise ValueError(f"dt_h must be > 0, got {dt_h}")

    charge_lower = charge.lower()
    _valid_charges_pen = {"neutral", "positive", "cationic", "+", "negative", "anionic", "-"}
    if charge_lower not in _valid_charges_pen:
        raise ValueError(
            f"charge must be neutral/positive/cationic/negative/anionic, got '{charge}'"
        )

    pore_radius_nm = _PORE_RADIUS_NM[mucus_type]

    # Effective diffusion coefficient (cm²/s)
    d_eff = _effective_diffusion_cm2_s(
        diffusion_coeff_cm2_s, mw, logP, charge, pore_radius_nm
    )

    # Convert thickness from µm to cm for Fick's law (all in cm units)
    thickness_cm = mucus_thickness_um * 1e-4  # µm → cm

    # Convert D_eff from cm²/s to cm²/h for forward Euler in hours
    d_eff_per_h = d_eff * 3600.0

    # Initial conditions (mg/mL)
    c_lumen = dose_mg / _VOL_LUMEN_ML  # mg/mL
    c_epi = 0.0  # mg/mL

    # t50: first time c_epi reaches half of expected equilibrium peak
    half_max_epi = (dose_mg * 0.5) / (_VOL_LUMEN_ML + _VOL_EPITHELIUM_ML)
    t50_h = float("nan")

    n_steps = max(int(t_end_h / dt_h), 1)
    times: list[float] = [0.0]
    c_lumen_arr: list[float] = [c_lumen]
    c_epi_arr: list[float] = [c_epi]

    for i in range(n_steps):
        # Flux (mg/cm²/h) through mucus: Fick's first law
        # C_lumen in mg/mL = mg/cm³, thickness in cm → flux in mg/cm²/h
        if thickness_cm > 0 and d_eff_per_h > 0:
            flux_mg_cm2_h = d_eff_per_h * c_lumen / thickness_cm
        else:
            flux_mg_cm2_h = 0.0

        # Rate of change (mg/mL/h)
        dc_lumen = -flux_mg_cm2_h * _AREA_CM2 / _VOL_LUMEN_ML
        dc_epi = (flux_mg_cm2_h * _AREA_CM2 / _VOL_EPITHELIUM_ML) - _K_ABS_PER_H * c_epi

        # Forward Euler update
        c_lumen = max(c_lumen + dc_lumen * dt_h, 0.0)
        c_epi = max(c_epi + dc_epi * dt_h, 0.0)

        t_next = (i + 1) * dt_h
        times.append(t_next)
        c_lumen_arr.append(c_lumen)
        c_epi_arr.append(c_epi)

        # Track t50: first time c_epi reaches the half_max threshold
        if math.isnan(t50_h) and c_epi >= half_max_epi:
            t50_h = t_next

    # Fraction penetrated: mass that left lumen (excluding residual epi and lumen)
    mass_remaining_lumen = c_lumen_arr[-1] * _VOL_LUMEN_ML
    mass_remaining_epi = c_epi_arr[-1] * _VOL_EPITHELIUM_ML
    mass_penetrated = dose_mg - mass_remaining_lumen - mass_remaining_epi
    f_penetrated = float(max(mass_penetrated / dose_mg, 0.0))

    # Build notes string
    r_h = _hydrodynamic_radius_nm(mw)
    notes_parts = [
        f"r_h={r_h:.2f} nm vs pore={pore_radius_nm:.0f} nm",
        f"D_eff={d_eff:.3e} cm2/s",
        f"k_int(charge)={_charge_factor_penetration(charge):.2f}",
        f"k_lipo={_lipophilicity_factor_penetration(logP):.2f}",
    ]
    if r_h >= pore_radius_nm:
        notes_parts.append("WARNING: molecule exceeds pore size, D_eff=0")
    notes = "; ".join(notes_parts)

    return MucusPenetrationResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        mucus_type=mucus_type,
        mucus_thickness_um=mucus_thickness_um,
        diffusion_coeff_effective=d_eff,
        times_h=times,
        c_lumen_mg_mL=c_lumen_arr,
        c_epithelium_mg_mL=c_epi_arr,
        f_penetrated=float(f_penetrated),
        t50_penetration_h=t50_h,
        notes=notes,
    )


def compare_charges(
    drug_name: str,
    dose_mg: float,
    mw: float,
    logP: float,
    cl_L_per_h: float,
    vd_L: float,
    **kwargs,
) -> list[MucusPKResult]:
    """
    Compare cationic, anionic, and neutral forms of a drug through mucus.

    Parameters
    ----------
    drug_name : str
    dose_mg : float
    mw : float
    logP : float
    cl_L_per_h : float
    vd_L : float
    **kwargs : passed to simulate_mucus_pk (route, h_mucus_um, etc.)

    Returns
    -------
    list[MucusPKResult]
        Sorted by f_absorbed descending (anionic typically highest).
    """
    results: list[MucusPKResult] = []
    for charge in ("cationic", "anionic", "neutral"):
        r = simulate_mucus_pk(
            drug_name=drug_name,
            dose_mg=dose_mg,
            mw=mw,
            logP=logP,
            charge=charge,
            cl_L_per_h=cl_L_per_h,
            vd_L=vd_L,
            **kwargs,
        )
        results.append(r)

    results.sort(key=lambda r: r.f_absorbed, reverse=True)
    return results
