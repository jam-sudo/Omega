"""Microbiome drug metabolism — gut lumen pre-absorptive metabolism and prodrug activation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from omega_pbpk._compat import np_trapz


@dataclass(frozen=True)
class MicrobiomePKResult:
    """Result from gut microbiome PK simulation."""

    drug_name: str
    microbiome_density: float
    times_h: list[float]
    c_plasma: list[float]
    cmax: float
    tmax_h: float
    auc: float
    f_metabolized_by_microbiome: float  # fraction of dose lost to microbiome
    notes: str


@dataclass(frozen=True)
class ProdrugActivationResult:
    """Result from prodrug microbiome activation simulation."""

    drug_name: str
    microbiome_density: float
    times_h: list[float]
    c_active_plasma: list[float]
    cmax: float
    tmax_h: float
    auc: float
    f_activated: float  # fraction of prodrug dose converted to active drug
    notes: str


def microbiome_metabolism_rate(
    c_lumen_mg_L: float,
    vmax_gut_mg_per_h: float,
    km_gut_mg_L: float,
    microbiome_density: float = 1.0,
) -> float:
    """Michaelis-Menten gut lumen metabolism by microbiome.

    Rate = Vmax * microbiome_density * C / (Km + C)

    Parameters
    ----------
    c_lumen_mg_L : float
        Drug concentration in gut lumen (mg/L).
    vmax_gut_mg_per_h : float
        Maximum microbiome metabolism rate (mg/h).
    km_gut_mg_L : float
        Michaelis constant for gut metabolism (mg/L).
    microbiome_density : float
        Relative microbiome density (0–1); 1.0 = normal, 0 = absent.

    Returns
    -------
    float
        Metabolism rate (mg/h) — amount metabolised per hour.
    """
    if c_lumen_mg_L <= 0.0:
        return 0.0
    return vmax_gut_mg_per_h * microbiome_density * c_lumen_mg_L / (km_gut_mg_L + c_lumen_mg_L)


def simulate_microbiome_drug_fate(
    drug_name: str,
    dose_mg: float,
    ka_abs_per_h: float,
    vmax_gut_mg_per_h: float,
    km_gut_mg_L: float,
    cl_L_per_h: float,
    vd_L: float,
    microbiome_density: float = 1.0,
    gut_volume_L: float = 1.0,
    t_end_h: float = 24.0,
    dt_h: float = 0.05,
) -> MicrobiomePKResult:
    """Simulate 3-compartment gut-lumen / plasma / microbiome-metabolite PK.

    Compartments:
    - A_gut   : drug mass in gut lumen (mg)
    - C_plasma: drug concentration in plasma (mg/L)

    ODEs (Forward Euler):
      dA_gut/dt   = -ka_abs * A_gut - microbiome_rate * gut_volume_L
      dC_plasma/dt = ka_abs * A_gut / vd_L - (cl/vd) * C_plasma

    Here microbiome_rate is expressed per litre of lumen (mg/h/L), so total
    rate removed from lumen = microbiome_rate * gut_volume_L.
    """
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if ka_abs_per_h <= 0:
        raise ValueError("ka_abs_per_h must be > 0")
    if vmax_gut_mg_per_h <= 0:
        raise ValueError("vmax_gut_mg_per_h must be > 0")
    if km_gut_mg_L <= 0:
        raise ValueError("km_gut_mg_L must be > 0")
    if cl_L_per_h <= 0:
        raise ValueError("cl_L_per_h must be > 0")
    if vd_L <= 0:
        raise ValueError("vd_L must be > 0")
    if not 0.0 <= microbiome_density <= 1.0:
        raise ValueError("microbiome_density must be in [0, 1]")

    n_steps = max(int(t_end_h / dt_h), 1)
    times = np.linspace(0.0, t_end_h, n_steps + 1)
    c_plasma = np.zeros(n_steps + 1)

    a_gut = dose_mg  # initial lumen mass (mg)
    c_p = 0.0
    ke = cl_L_per_h / vd_L

    total_metabolized = 0.0  # mg

    for i in range(n_steps):
        c_lumen = a_gut / gut_volume_L  # mg/L
        met_rate_per_h = microbiome_metabolism_rate(c_lumen, vmax_gut_mg_per_h, km_gut_mg_L,
                                                    microbiome_density)
        # mass metabolized per step (mg)
        mass_met_step = min(met_rate_per_h * dt_h, a_gut)

        abs_rate = ka_abs_per_h * a_gut  # mg/h absorbed

        # Gut lumen: remove absorbed + metabolized
        da_gut = -abs_rate * dt_h - mass_met_step
        a_gut = max(a_gut + da_gut, 0.0)

        # Track microbiome metabolized mass
        total_metabolized += mass_met_step

        # Plasma
        dc_p = (abs_rate / vd_L - ke * c_p) * dt_h
        c_p = max(c_p + dc_p, 0.0)
        c_plasma[i + 1] = c_p

    auc = float(np_trapz(c_plasma, times))
    cmax = float(np.max(c_plasma))
    tmax_h = float(times[int(np.argmax(c_plasma))])
    f_metabolized = total_metabolized / dose_mg

    density_label = f"microbiome_density={microbiome_density:.2f}"
    notes = (
        f"{drug_name}: {density_label}; "
        f"f_metabolized_by_microbiome={f_metabolized:.3f}"
    )

    return MicrobiomePKResult(
        drug_name=drug_name,
        microbiome_density=microbiome_density,
        times_h=times.tolist(),
        c_plasma=c_plasma.tolist(),
        cmax=cmax,
        tmax_h=tmax_h,
        auc=auc,
        f_metabolized_by_microbiome=f_metabolized,
        notes=notes,
    )


def antibiotic_disruption_effect(
    drug_name: str,
    dose_mg: float,
    ka_abs_per_h: float,
    vmax_gut_mg_per_h: float,
    km_gut_mg_L: float,
    cl_L_per_h: float,
    vd_L: float,
    disruption_factor: float = 0.1,
    **kwargs,
) -> dict:
    """Compare PK with normal vs antibiotic-disrupted microbiome.

    Parameters
    ----------
    disruption_factor : float
        Relative microbiome density after antibiotic disruption (default 0.1).

    Returns
    -------
    dict
        Keys: 'normal' (MicrobiomePKResult), 'disrupted' (MicrobiomePKResult),
        'auc_ratio' (disrupted AUC / normal AUC).
    """
    normal = simulate_microbiome_drug_fate(
        drug_name, dose_mg, ka_abs_per_h, vmax_gut_mg_per_h, km_gut_mg_L,
        cl_L_per_h, vd_L, microbiome_density=1.0, **kwargs
    )
    disrupted = simulate_microbiome_drug_fate(
        drug_name, dose_mg, ka_abs_per_h, vmax_gut_mg_per_h, km_gut_mg_L,
        cl_L_per_h, vd_L, microbiome_density=disruption_factor, **kwargs
    )
    auc_ratio = disrupted.auc / max(normal.auc, 1e-12)
    return {
        "normal": normal,
        "disrupted": disrupted,
        "auc_ratio": auc_ratio,
    }


def prodrug_activation(
    drug_name: str,
    prodrug_dose_mg: float,
    ka_abs_per_h: float,
    activation_vmax: float,
    activation_km: float,
    cl_L_per_h: float,
    vd_L: float,
    microbiome_density: float = 1.0,
    gut_volume_L: float = 1.0,
    t_end_h: float = 24.0,
    dt_h: float = 0.05,
) -> ProdrugActivationResult:
    """Simulate gut microbiome converting prodrug to active drug.

    The active drug released in the gut lumen is then absorbed.

    Compartments:
    - A_prodrug   : prodrug mass in lumen (mg)  — undergoes activation
    - A_active_gut: active drug mass in lumen (mg) — then absorbed
    - C_plasma    : active drug plasma concentration (mg/L)

    ODEs:
      dA_prodrug/dt   = -activation_rate * gut_volume_L
      dA_active_gut/dt = activation_rate * gut_volume_L - ka_abs * A_active_gut
      dC_plasma/dt    = ka_abs * A_active_gut / vd_L - (cl/vd) * C_plasma

    activation_rate (mg/h/L) = Vmax*density*C_prodrug/(Km+C_prodrug)

    Parameters
    ----------
    activation_vmax : float
        Maximum prodrug activation rate by microbiome (mg/h), at density=1.
    activation_km : float
        Michaelis constant for activation (mg/L).
    """
    if prodrug_dose_mg <= 0:
        raise ValueError("prodrug_dose_mg must be > 0")
    if ka_abs_per_h <= 0:
        raise ValueError("ka_abs_per_h must be > 0")
    if activation_vmax <= 0:
        raise ValueError("activation_vmax must be > 0")
    if activation_km <= 0:
        raise ValueError("activation_km must be > 0")
    if cl_L_per_h <= 0:
        raise ValueError("cl_L_per_h must be > 0")
    if vd_L <= 0:
        raise ValueError("vd_L must be > 0")
    if not 0.0 <= microbiome_density <= 1.0:
        raise ValueError("microbiome_density must be in [0, 1]")

    n_steps = max(int(t_end_h / dt_h), 1)
    times = np.linspace(0.0, t_end_h, n_steps + 1)
    c_plasma = np.zeros(n_steps + 1)

    a_prodrug = prodrug_dose_mg  # lumen prodrug mass (mg)
    a_active = 0.0              # lumen active drug mass (mg)
    c_p = 0.0
    ke = cl_L_per_h / vd_L

    total_activated = 0.0  # mg of prodrug converted

    for i in range(n_steps):
        c_prodrug_lumen = a_prodrug / gut_volume_L  # mg/L
        act_rate_per_L = microbiome_metabolism_rate(
            c_prodrug_lumen, activation_vmax, activation_km, microbiome_density
        )
        mass_activated = min(act_rate_per_L * dt_h, a_prodrug)

        # Prodrug lumen
        a_prodrug = max(a_prodrug - mass_activated, 0.0)
        total_activated += mass_activated

        # Active drug lumen
        abs_rate = ka_abs_per_h * a_active
        a_active = max(a_active + mass_activated - abs_rate * dt_h, 0.0)

        # Plasma
        dc_p = (abs_rate / vd_L - ke * c_p) * dt_h
        c_p = max(c_p + dc_p, 0.0)
        c_plasma[i + 1] = c_p

    auc = float(np_trapz(c_plasma, times))
    cmax = float(np.max(c_plasma))
    tmax_h = float(times[int(np.argmax(c_plasma))])
    f_activated = total_activated / prodrug_dose_mg

    notes = (
        f"{drug_name} prodrug: microbiome_density={microbiome_density:.2f}; "
        f"f_activated={f_activated:.3f}"
    )

    return ProdrugActivationResult(
        drug_name=drug_name,
        microbiome_density=microbiome_density,
        times_h=times.tolist(),
        c_active_plasma=c_plasma.tolist(),
        cmax=cmax,
        tmax_h=tmax_h,
        auc=auc,
        f_activated=f_activated,
        notes=notes,
    )


__all__ = [
    "MicrobiomePKResult",
    "ProdrugActivationResult",
    "microbiome_metabolism_rate",
    "simulate_microbiome_drug_fate",
    "antibiotic_disruption_effect",
    "prodrug_activation",
]
