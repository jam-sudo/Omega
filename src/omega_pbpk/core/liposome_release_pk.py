"""Drug release from liposomal formulations with systemic PK.

Models the two-compartment liposome-plasma system:
  1. Liposome pool (mg) -- drug encapsulated, slowly leaking and cleared.
  2. Free plasma drug (mg/L) -- released drug, cleared rapidly like a small molecule.

Four liposome types are supported, each with different intrinsic release rates
and circulation half-lives:

  * conventional  -- fastest release, short half-life (passive diffusion)
  * pegylated     -- slowest release, long circulation (PEGylated stealth)
  * thermosensitive -- triggered by mild hyperthermia (41 °C)
  * pH_sensitive  -- triggered by endosomal/tumour acidic pH

ODEs (Forward Euler)
--------------------
  dA_lipo/dt = -(k_release + k_lipo_elim) * A_lipo
  dC_free/dt = k_release * A_lipo / Vd_free - (CL_free / Vd_free) * C_free

where A_lipo [mg] is encapsulated drug, C_free [mg/L] is free-plasma concentration.

References
----------
- Allen TM & Cullis PR. Science. 2004;303(5665):1818-22
- Needham D et al., Cancer Res. 2000;60(5):1197-201 (thermosensitive)
- Drummond DC et al., Pharmacol Rev. 1999;51(4):691-744 (review)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

__all__ = [
    "LiposomeReleasePKResult",
    "simulate_liposome_release",
    "compare_liposome_types",
]

# ---------------------------------------------------------------------------
# Valid liposome types
# ---------------------------------------------------------------------------

_VALID_TYPES = frozenset({"conventional", "pegylated", "thermosensitive", "ph_sensitive"})

# Default (k_release [/h], t_half_lipo [h], mechanism)
_TYPE_DEFAULTS: dict[str, tuple[float, float, str]] = {
    "conventional": (0.3, 6.0, "passive_diffusion"),
    "pegylated": (0.05, 48.0, "passive_diffusion"),
    "thermosensitive": (0.5, 12.0, "triggered"),
    "ph_sensitive": (0.4, 8.0, "pH_triggered"),
}

_LN2 = math.log(2.0)

# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class LiposomeReleasePKResult:
    """Results from liposome release PK simulation.

    Attributes
    ----------
    drug_name : Name of the drug.
    liposome_type : One of ``conventional``, ``pegylated``,
        ``thermosensitive``, ``ph_sensitive``.
    dose_mg : Total encapsulated dose (mg).
    times_h : Simulation time grid (h).
    c_encapsulated_mg_L : Drug remaining in liposomes, normalised by Vd_free
        (mg/L) — used as a comparable concentration metric.
    c_released_mg_L : Free plasma drug concentration (mg/L).
    c_total_mg_L : Sum of encapsulated and free (mg/L).
    cmax_free_mg_L : Peak free plasma concentration.
    tmax_free_h : Time of peak free plasma concentration.
    auc_free_mg_h_per_L : AUC of free plasma drug (trapezoidal).
    auc_total_mg_h_per_L : AUC of total drug (trapezoidal).
    encapsulated_fraction_at_24h : Fraction of dose still encapsulated at t=24 h.
    t_half_liposome_h : Liposome circulation half-life (h).
    release_mechanism : ``passive_diffusion``, ``triggered``, or ``pH_triggered``.
    notes : Human-readable summary.
    """

    drug_name: str
    liposome_type: str
    dose_mg: float
    times_h: list = field(default_factory=list)
    c_encapsulated_mg_L: list = field(default_factory=list)
    c_released_mg_L: list = field(default_factory=list)
    c_total_mg_L: list = field(default_factory=list)
    cmax_free_mg_L: float = 0.0
    tmax_free_h: float = 0.0
    auc_free_mg_h_per_L: float = 0.0
    auc_total_mg_h_per_L: float = 0.0
    encapsulated_fraction_at_24h: float = 0.0
    t_half_liposome_h: float = 0.0
    release_mechanism: str = ""
    notes: str = ""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _trapz(x: list[float], y: list[float]) -> float:
    """Manual trapezoidal integration."""
    total = 0.0
    for i in range(1, len(x)):
        total += 0.5 * (y[i - 1] + y[i]) * (x[i] - x[i - 1])
    return total


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_DEFAULT_K_RELEASE = 0.1  # sentinel for "user did not override"


def simulate_liposome_release(
    drug_name: str,
    dose_mg: float,
    liposome_type: str = "pegylated",
    cl_free_L_per_h: float = 10.0,
    vd_free_L: float = 50.0,
    cl_lipo_L_per_h: float = 0.05,
    k_release_per_h: float = _DEFAULT_K_RELEASE,
    t_end_h: float = 72.0,
    dt_h: float = 0.1,
) -> LiposomeReleasePKResult:
    """Simulate drug release from liposomes and resultant free-drug PK.

    Parameters
    ----------
    drug_name : Name of the drug.
    dose_mg : Total encapsulated dose (mg). Must be > 0.
    liposome_type : ``conventional``, ``pegylated``,
        ``thermosensitive``, or ``ph_sensitive``.
    cl_free_L_per_h : Clearance of free (released) drug (L/h). Must be > 0.
    vd_free_L : Volume of distribution for free drug (L). Must be > 0.
    cl_lipo_L_per_h : Liposome clearance rate (L/h).  Used with vd_free_L
        to compute k_lipo_elim.
    k_release_per_h : Release rate constant (/h). Defaults to the
        liposome-type value if 0.1 is passed (the default sentinel).
    t_end_h : Simulation duration (h).
    dt_h : Forward-Euler time step (h).

    Returns
    -------
    LiposomeReleasePKResult
    """
    # --- Validation ---
    if dose_mg <= 0.0:
        raise ValueError(f"dose_mg must be > 0; got {dose_mg}")
    if cl_free_L_per_h <= 0.0:
        raise ValueError(f"cl_free_L_per_h must be > 0; got {cl_free_L_per_h}")
    if vd_free_L <= 0.0:
        raise ValueError(f"vd_free_L must be > 0; got {vd_free_L}")
    if liposome_type not in _VALID_TYPES:
        raise ValueError(
            f"Invalid liposome_type '{liposome_type}'. Must be one of {sorted(_VALID_TYPES)}."
        )

    # --- Type-specific defaults ---
    type_k_release, type_t_half_lipo, mechanism = _TYPE_DEFAULTS[liposome_type]

    # Apply type defaults when user passes the sentinel value
    if k_release_per_h == _DEFAULT_K_RELEASE:
        k_rel = type_k_release
    else:
        k_rel = k_release_per_h

    # k_lipo_elim: use type-specific t_half when cl_lipo is the default sentinel (0.05).
    # The spec gives per-type t_half_lipo values; derive k_lipo_elim from them so that
    # reported half-lives match the documented liposome biology.
    _DEFAULT_CL_LIPO = 0.05
    vd_lipo = vd_free_L
    if cl_lipo_L_per_h == _DEFAULT_CL_LIPO:
        # Use the type-specific t_half to set k_lipo_elim
        t_half_lipo = type_t_half_lipo
        k_lipo_elim = _LN2 / t_half_lipo
    else:
        k_lipo_elim = cl_lipo_L_per_h / vd_lipo
        if k_lipo_elim > 0.0:
            t_half_lipo = _LN2 / k_lipo_elim
        else:
            t_half_lipo = float("inf")

    # Elimination rate constants
    k_elim_free = cl_free_L_per_h / vd_free_L  # for free drug

    # --- Initial conditions ---
    a_lipo = dose_mg  # mg
    c_free = 0.0  # mg/L

    times: list[float] = [0.0]
    c_enc: list[float] = [a_lipo / vd_free_L]
    c_rel: list[float] = [c_free]
    c_tot: list[float] = [a_lipo / vd_free_L + c_free]

    # --- Forward Euler ODE ---
    n_steps = max(1, round(t_end_h / dt_h))
    for i in range(n_steps):
        da_lipo = -(k_rel + k_lipo_elim) * a_lipo
        dc_free = k_rel * a_lipo / vd_free_L - k_elim_free * c_free

        a_lipo = max(0.0, a_lipo + da_lipo * dt_h)
        c_free = max(0.0, c_free + dc_free * dt_h)

        t = (i + 1) * dt_h
        times.append(t)
        c_enc.append(a_lipo / vd_free_L)
        c_rel.append(c_free)
        c_tot.append(a_lipo / vd_free_L + c_free)

    # --- Summary statistics ---
    cmax_free = max(c_rel)
    tmax_free = times[c_rel.index(cmax_free)]
    auc_free = _trapz(times, c_rel)
    auc_total = _trapz(times, c_tot)

    # Encapsulated fraction at 24 h
    t24_idx = 0
    for idx, t in enumerate(times):
        if t <= 24.0:
            t24_idx = idx
    enc_frac_24h = c_enc[t24_idx] * vd_free_L / dose_mg

    notes = (
        f"{liposome_type.capitalize()} liposome: k_release={k_rel:.3f}/h, "
        f"k_lipo_elim={k_lipo_elim:.4f}/h, t_half_lipo={t_half_lipo:.1f}h. "
        f"Cmax_free={cmax_free:.4f} mg/L at {tmax_free:.1f}h, "
        f"AUC_free={auc_free:.2f} mg·h/L. "
        f"Encapsulated fraction at 24h: {enc_frac_24h:.3f}. "
        f"Mechanism: {mechanism}."
    )

    return LiposomeReleasePKResult(
        drug_name=drug_name,
        liposome_type=liposome_type,
        dose_mg=dose_mg,
        times_h=times,
        c_encapsulated_mg_L=c_enc,
        c_released_mg_L=c_rel,
        c_total_mg_L=c_tot,
        cmax_free_mg_L=cmax_free,
        tmax_free_h=tmax_free,
        auc_free_mg_h_per_L=auc_free,
        auc_total_mg_h_per_L=auc_total,
        encapsulated_fraction_at_24h=enc_frac_24h,
        t_half_liposome_h=t_half_lipo,
        release_mechanism=mechanism,
        notes=notes,
    )


def compare_liposome_types(
    drug_name: str,
    dose_mg: float,
    cl_free_L_per_h: float = 10.0,
    vd_free_L: float = 50.0,
) -> list[LiposomeReleasePKResult]:
    """Compare all four liposome types for the same drug and dose.

    Parameters
    ----------
    drug_name : Name of the drug.
    dose_mg : Total encapsulated dose (mg).
    cl_free_L_per_h : Clearance of free drug (L/h).
    vd_free_L : Volume of distribution for free drug (L).

    Returns
    -------
    list[LiposomeReleasePKResult]
        Results for all four types, sorted by auc_free_mg_h_per_L descending.
    """
    results = [
        simulate_liposome_release(
            drug_name=drug_name,
            dose_mg=dose_mg,
            liposome_type=ltype,
            cl_free_L_per_h=cl_free_L_per_h,
            vd_free_L=vd_free_L,
        )
        for ltype in sorted(_VALID_TYPES)
    ]
    results.sort(key=lambda r: r.auc_free_mg_h_per_L, reverse=True)
    return results
