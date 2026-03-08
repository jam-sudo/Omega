"""Extended-release ocular drug delivery PK model (Phase 1017).

Models implant/punctal plug/contact lens/insert sustained release:
  Depot → Aqueous humor → Systemic
"""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = [
    "OcularDrugDeliveryResult",
    "simulate_ocular_drug_delivery",
]

_VALID_DEVICE_TYPES = {"implant", "punctal_plug", "contact_lens", "insert"}

_RELEASE_RATES = {
    "implant": 0.003,  # /h — very slow, months-long
    "punctal_plug": 0.01,  # /h — weeks
    "contact_lens": 0.05,  # /h — days
    "insert": 0.02,  # /h — 1-4 weeks
}

_V_AQUEOUS_L = 0.00025  # 0.25 mL in litres
_K_DRAIN = 2.5  # /h — aqueous humor turnover
_K_SYS = 0.1  # /h — nasolacrimal drainage
_V_SYSTEMIC_L = 5.0  # L — systemic volume of distribution
_KE_SYSTEMIC = 0.3  # /h — systemic elimination


@dataclass
class OcularDrugDeliveryResult:
    """Result of extended-release ocular drug delivery simulation."""

    drug_name: str
    dose_mg: float
    device_type: str
    times_h: list = field(default_factory=list)
    a_implant_mg: list = field(default_factory=list)
    c_aqueous_mg_L: list = field(default_factory=list)
    c_systemic_mg_L: list = field(default_factory=list)
    cmax_aqueous_mg_L: float = 0.0
    cmax_systemic_mg_L: float = 0.0
    steady_state_aqueous_mg_L: float = 0.0
    t_to_steady_state_h: float = 0.0
    duration_effective_h: float = 0.0
    drug_load_remaining_pct_at_30d: float = 0.0
    notes: str = ""


def simulate_ocular_drug_delivery(
    drug_name: str,
    dose_mg: float,
    device_type: str = "implant",
    logp: float = 2.0,
    solubility_mg_mL: float = 1.0,
    t_end_h: float = 720.0,
    dt_h: float = 1.0,
) -> OcularDrugDeliveryResult:
    """Simulate sustained-release ocular device PK using Forward Euler.

    Parameters
    ----------
    drug_name:
        Name of the drug compound.
    dose_mg:
        Total drug load in the device (mg).
    device_type:
        One of "implant", "punctal_plug", "contact_lens", "insert".
    logp:
        Logarithm of octanol-water partition coefficient.
    solubility_mg_mL:
        Intrinsic aqueous solubility (mg/mL) — currently informational.
    t_end_h:
        Simulation end time in hours (default 720 h = 30 days).
    dt_h:
        Time step in hours (default 1 h).

    Returns
    -------
    OcularDrugDeliveryResult
    """
    if dose_mg <= 0:
        raise ValueError(f"dose_mg must be > 0, got {dose_mg}")
    if device_type not in _VALID_DEVICE_TYPES:
        raise ValueError(f"device_type must be one of {_VALID_DEVICE_TYPES}, got {device_type!r}")

    k_release = _RELEASE_RATES[device_type]

    # logP-dependent corneal absorption factor (lipophilic = faster)
    k_cornea_factor = max(0.5, 1.0 + 0.2 * logp)

    # Steady-state approximation for aqueous concentration
    # At steady state: k_release * A_implant / V_aq * k_cornea_factor == (k_drain + k_sys) * Css
    # For t << 1/k_release, A_implant ≈ dose_mg, so:
    css_aqueous = k_release * dose_mg / _V_AQUEOUS_L * k_cornea_factor / (_K_DRAIN + _K_SYS)

    # ------------------------------------------------------------------
    # Forward Euler integration
    # ------------------------------------------------------------------
    n_steps = int(t_end_h / dt_h) + 1

    times_h: list[float] = []
    a_implant_mg: list[float] = []
    c_aqueous_mg_L: list[float] = []
    c_systemic_mg_L: list[float] = []

    a_imp = dose_mg
    c_aq = 0.0
    c_sys = 0.0

    cmax_aq = 0.0
    cmax_sys = 0.0
    t_to_ss = t_end_h  # default: never reached

    for step in range(n_steps):
        t = step * dt_h

        times_h.append(t)
        a_implant_mg.append(a_imp)
        c_aqueous_mg_L.append(c_aq)
        c_systemic_mg_L.append(c_sys)

        if c_aq > cmax_aq:
            cmax_aq = c_aq
        if c_sys > cmax_sys:
            cmax_sys = c_sys

        # Check for t_to_steady_state (90% of Css)
        if t_to_ss == t_end_h and c_aq >= 0.9 * css_aqueous:
            t_to_ss = t

        # ODEs
        da_imp = -k_release * a_imp
        dc_aq = k_release * a_imp / _V_AQUEOUS_L * k_cornea_factor - (_K_DRAIN + _K_SYS) * c_aq
        dc_sys = _K_DRAIN * c_aq * _V_AQUEOUS_L / _V_SYSTEMIC_L - _KE_SYSTEMIC * c_sys

        a_imp = max(0.0, a_imp + da_imp * dt_h)
        c_aq = max(0.0, c_aq + dc_aq * dt_h)
        c_sys = max(0.0, c_sys + dc_sys * dt_h)

    # ------------------------------------------------------------------
    # Derived metrics
    # ------------------------------------------------------------------

    # duration_effective_h: time above 10% of cmax_aq
    threshold = 0.1 * cmax_aq
    duration_eff = 0.0
    for val in c_aqueous_mg_L:
        if val >= threshold:
            duration_eff += dt_h

    # drug_load_remaining at 720 h
    idx_720 = min(int(720.0 / dt_h), len(a_implant_mg) - 1)
    drug_load_remaining = a_implant_mg[idx_720] / dose_mg * 100.0

    notes = (
        f"k_release={k_release}/h, k_cornea_factor={k_cornea_factor:.2f}, "
        f"Css_approx={css_aqueous:.4f} mg/L"
    )

    return OcularDrugDeliveryResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        device_type=device_type,
        times_h=times_h,
        a_implant_mg=a_implant_mg,
        c_aqueous_mg_L=c_aqueous_mg_L,
        c_systemic_mg_L=c_systemic_mg_L,
        cmax_aqueous_mg_L=cmax_aq,
        cmax_systemic_mg_L=cmax_sys,
        steady_state_aqueous_mg_L=css_aqueous,
        t_to_steady_state_h=t_to_ss,
        duration_effective_h=duration_eff,
        drug_load_remaining_pct_at_30d=drug_load_remaining,
        notes=notes,
    )
