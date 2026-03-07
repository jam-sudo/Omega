"""Nanoparticle drug delivery PK — lipid NPs, polymeric NPs, dendrimers."""

from __future__ import annotations

from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class NanomedicinePKResult:
    """PK result for nanoparticle drug delivery system."""

    drug_name: str
    dose_mg: float
    nanoparticle_type: str
    size_nm: float
    surface_charge_mv: float
    times_h: list[float]
    c_np_mg_L: list[float]
    c_free_mg_L: list[float]
    cmax_free: float
    auc_free: float
    t_half_np_h: float
    cl_effective_L_per_h: float
    notes: str


# ---------------------------------------------------------------------------
# Valid nanoparticle types
# ---------------------------------------------------------------------------

_VALID_NP_TYPES = ("lipid_np", "polymeric_np", "dendrimer")

_NP_RELEASE_MODIFIERS: dict[str, float] = {
    "lipid_np": 1.0,
    "polymeric_np": 0.7,
    "dendrimer": 1.5,
}


# ---------------------------------------------------------------------------
# Core simulation
# ---------------------------------------------------------------------------


def simulate_nanomedicine_pk(
    drug_name: str,
    dose_mg: float,
    nanoparticle_type: str,
    size_nm: float,
    surface_charge_mv: float,
    cl_L_per_h: float,
    vd_L: float,
    k_release_per_h: float,
    t_end_h: float = 24.0,
    dt_h: float = 0.1,
) -> NanomedicinePKResult:
    """Simulate nanoparticle drug delivery PK (2-compartment NP depot → free drug).

    Parameters
    ----------
    drug_name :
        Name of the drug / formulation.
    dose_mg :
        Administered dose (mg).
    nanoparticle_type :
        One of ``"lipid_np"``, ``"polymeric_np"``, ``"dendrimer"``.
    size_nm :
        Nanoparticle diameter (nm). Larger particles clear faster (opsonization).
    surface_charge_mv :
        Zeta potential (mV). Negative values mimic PEG-like stealth; positive
        values increase clearance.
    cl_L_per_h :
        Base systemic clearance of free drug (L/h).
    vd_L :
        Volume of distribution of free drug (L).
    k_release_per_h :
        Base first-order drug release rate from nanoparticle (1/h).
    t_end_h :
        Simulation end time (h).
    dt_h :
        Forward-Euler integration step (h).
    """
    # ------------------------------------------------------------------
    # Input validation
    # ------------------------------------------------------------------
    if not drug_name:
        raise ValueError("drug_name must be a non-empty string")
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if nanoparticle_type not in _VALID_NP_TYPES:
        raise ValueError(
            f"nanoparticle_type must be one of {_VALID_NP_TYPES}, got '{nanoparticle_type}'"
        )
    if size_nm <= 0:
        raise ValueError("size_nm must be > 0")
    if cl_L_per_h <= 0:
        raise ValueError("cl_L_per_h must be > 0")
    if vd_L <= 0:
        raise ValueError("vd_L must be > 0")
    if k_release_per_h < 0:
        raise ValueError("k_release_per_h must be >= 0")
    if t_end_h <= 0:
        raise ValueError("t_end_h must be > 0")
    if dt_h <= 0:
        raise ValueError("dt_h must be > 0")

    # ------------------------------------------------------------------
    # Size effect on NP clearance (opsonization increases above 100 nm)
    # ------------------------------------------------------------------
    size_factor = 1.0 + max(0.0, (size_nm - 100.0) / 200.0)

    # ------------------------------------------------------------------
    # Surface charge effect on NP clearance
    # ------------------------------------------------------------------
    if surface_charge_mv < 0:
        # Negative charge (PEG-like stealth): reduces clearance
        charge_factor = 1.0 - max(0.0, min(0.5, abs(surface_charge_mv) / 100.0 * 0.5))
    else:
        # Positive charge: increases clearance
        charge_factor = 1.2

    # Effective NP clearance (first-order elimination of NP from blood)
    # Expressed as an effective elimination rate constant for the NP depot
    k_np_elim = (cl_L_per_h / vd_L) * size_factor * charge_factor  # 1/h (proxy)

    # ------------------------------------------------------------------
    # Nanoparticle-type modifier on release rate
    # ------------------------------------------------------------------
    k_release_eff = k_release_per_h * _NP_RELEASE_MODIFIERS[nanoparticle_type]

    # ------------------------------------------------------------------
    # Effective free-drug clearance (CL is unchanged — NP only affects PK
    # of the NP depot, not the intrinsic CL of the released drug)
    # ------------------------------------------------------------------
    ke_free = cl_L_per_h / vd_L  # 1/h

    # ------------------------------------------------------------------
    # Forward-Euler integration
    # State variables:
    #   A_np  : amount of drug still inside NPs (mg)
    #   A_free: amount of free drug in plasma (mg)
    # ------------------------------------------------------------------
    n_steps = max(int(t_end_h / dt_h), 1)
    times_h: list[float] = []
    c_np_mg_L: list[float] = []
    c_free_mg_L: list[float] = []

    a_np = dose_mg
    a_free = 0.0

    for i in range(n_steps + 1):
        t = i * dt_h
        times_h.append(t)
        c_np_mg_L.append(a_np / vd_L)
        c_free_mg_L.append(a_free / vd_L)

        if i < n_steps:
            # Release from NP depot and NP elimination (MPS/opsonization)
            released = k_release_eff * a_np * dt_h
            np_eliminated = k_np_elim * a_np * dt_h  # phagocytosed without release
            d_a_np = -(released + np_eliminated)

            # Free drug: gains from release, loses via systemic clearance
            d_a_free = released - ke_free * a_free * dt_h

            a_np = max(a_np + d_a_np, 0.0)
            a_free = max(a_free + d_a_free, 0.0)

    # ------------------------------------------------------------------
    # PK metrics
    # ------------------------------------------------------------------
    cmax_free = max(c_free_mg_L)
    # Trapezoidal AUC (manual)
    auc_free = 0.0
    for j in range(len(times_h) - 1):
        auc_free += 0.5 * (c_free_mg_L[j] + c_free_mg_L[j + 1]) * (times_h[j + 1] - times_h[j])

    # Half-life of NP depot
    import math

    k_np_total = k_release_eff + k_np_elim
    t_half_np_h = math.log(2.0) / k_np_total if k_np_total > 0 else float("inf")

    cl_effective = cl_L_per_h * size_factor * charge_factor

    notes = (
        f"NP type: {nanoparticle_type}; size_factor={size_factor:.3f}; "
        f"charge_factor={charge_factor:.3f}; k_release_eff={k_release_eff:.4f} 1/h"
    )

    return NanomedicinePKResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        nanoparticle_type=nanoparticle_type,
        size_nm=size_nm,
        surface_charge_mv=surface_charge_mv,
        times_h=times_h,
        c_np_mg_L=c_np_mg_L,
        c_free_mg_L=c_free_mg_L,
        cmax_free=cmax_free,
        auc_free=auc_free,
        t_half_np_h=t_half_np_h,
        cl_effective_L_per_h=cl_effective,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Sensitivity analysis
# ---------------------------------------------------------------------------


def size_sensitivity(
    drug_name: str,
    dose_mg: float,
    sizes_nm: list[float],
    cl_L_per_h: float,
    vd_L: float,
    k_release_per_h: float,
    nanoparticle_type: str = "lipid_np",
    surface_charge_mv: float = -20.0,
    t_end_h: float = 24.0,
    dt_h: float = 0.1,
) -> list[dict]:
    """Evaluate free-drug AUC and Cmax across a range of particle sizes.

    Parameters
    ----------
    sizes_nm :
        List of particle diameters (nm) to evaluate.

    Returns
    -------
    list[dict]
        Each dict has keys: ``size_nm``, ``auc_free``, ``cmax_free``.
    """
    if not drug_name:
        raise ValueError("drug_name must be a non-empty string")
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if not sizes_nm:
        raise ValueError("sizes_nm must be a non-empty list")
    if cl_L_per_h <= 0:
        raise ValueError("cl_L_per_h must be > 0")
    if vd_L <= 0:
        raise ValueError("vd_L must be > 0")
    if k_release_per_h < 0:
        raise ValueError("k_release_per_h must be >= 0")

    results = []
    for size in sizes_nm:
        if size <= 0:
            raise ValueError(f"All sizes_nm must be > 0, got {size}")
        res = simulate_nanomedicine_pk(
            drug_name=drug_name,
            dose_mg=dose_mg,
            nanoparticle_type=nanoparticle_type,
            size_nm=size,
            surface_charge_mv=surface_charge_mv,
            cl_L_per_h=cl_L_per_h,
            vd_L=vd_L,
            k_release_per_h=k_release_per_h,
            t_end_h=t_end_h,
            dt_h=dt_h,
        )
        results.append(
            {
                "size_nm": size,
                "auc_free": res.auc_free,
                "cmax_free": res.cmax_free,
            }
        )
    return results


__all__ = [
    "NanomedicinePKResult",
    "simulate_nanomedicine_pk",
    "size_sensitivity",
]
