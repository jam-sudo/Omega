"""Prodrug-to-active drug bioconversion kinetics model (Phase 980).

Models the pharmacokinetics of a prodrug and its active metabolite,
including oral and IV routes of administration.
"""

from __future__ import annotations

from dataclasses import dataclass, field

_VALID_ROUTES = {"oral", "iv"}
_VALID_SITES = {"intestinal", "hepatic", "plasma", "systemic"}


@dataclass
class ProdrugActivationResult:
    """Result of prodrug activation PK simulation."""

    prodrug_name: str = ""
    active_drug_name: str = ""
    route: str = "oral"
    times_h: list = field(default_factory=list)
    c_prodrug_mg_L: list = field(default_factory=list)
    c_active_mg_L: list = field(default_factory=list)
    cmax_prodrug_mg_L: float = 0.0
    cmax_active_mg_L: float = 0.0
    tmax_active_h: float = 0.0
    auc_prodrug_mg_h_per_L: float = 0.0
    auc_active_mg_h_per_L: float = 0.0
    conversion_efficiency: float = 0.0
    activation_site: str = "hepatic"
    notes: str = ""


def simulate_prodrug_activation(
    prodrug_name: str,
    active_drug_name: str,
    dose_mg: float,
    route: str = "oral",
    k_conv_per_h: float = 0.5,
    k_elim_prodrug_per_h: float = 0.3,
    k_elim_active_per_h: float = 0.2,
    vd_prodrug_L: float = 20.0,
    vd_active_L: float = 30.0,
    f_conversion: float = 0.8,
    activation_site: str = "hepatic",
    t_end_h: float = 24.0,
    dt_h: float = 0.1,
) -> ProdrugActivationResult:
    """Simulate prodrug bioconversion to active drug.

    Parameters
    ----------
    prodrug_name:
        Name of the prodrug.
    active_drug_name:
        Name of the active metabolite.
    dose_mg:
        Administered dose (mg).
    route:
        Administration route: "oral" or "iv".
    k_conv_per_h:
        First-order conversion rate from prodrug to active drug (1/h).
    k_elim_prodrug_per_h:
        First-order elimination rate of prodrug (1/h).
    k_elim_active_per_h:
        First-order elimination rate of active drug (1/h).
    vd_prodrug_L:
        Volume of distribution of prodrug (L).
    vd_active_L:
        Volume of distribution of active drug (L).
    f_conversion:
        Fraction of prodrug converted to active drug (0-1].
    activation_site:
        Anatomical site of bioactivation.
    t_end_h:
        Simulation end time (h).
    dt_h:
        Forward Euler step size (h).

    Returns
    -------
    ProdrugActivationResult
    """
    # --- Validation ---
    if dose_mg <= 0:
        raise ValueError(f"dose_mg must be > 0, got {dose_mg}")
    if k_conv_per_h <= 0:
        raise ValueError(f"k_conv_per_h must be > 0, got {k_conv_per_h}")
    if vd_prodrug_L <= 0:
        raise ValueError(f"vd_prodrug_L must be > 0, got {vd_prodrug_L}")
    if vd_active_L <= 0:
        raise ValueError(f"vd_active_L must be > 0, got {vd_active_L}")
    if not (0 < f_conversion <= 1):
        raise ValueError(f"f_conversion must be in (0, 1], got {f_conversion}")
    if route not in _VALID_ROUTES:
        raise ValueError(f"route must be one of {_VALID_ROUTES}, got {route!r}")
    if activation_site not in _VALID_SITES:
        raise ValueError(f"activation_site must be one of {_VALID_SITES}, got {activation_site!r}")

    # --- Initial conditions ---
    n_steps = int(round(t_end_h / dt_h)) + 1

    if route == "oral":
        ka = 1.0  # 1/h
        f_abs = 0.9  # oral bioavailability of prodrug
        a_gut = f_abs * dose_mg  # mg in gut compartment
        c_prod = 0.0  # mg/L
    else:
        # IV bolus
        a_gut = 0.0
        ka = 0.0
        c_prod = dose_mg / vd_prodrug_L  # mg/L

    c_act = 0.0  # mg/L active drug

    times_h_list: list[float] = []
    c_prodrug_list: list[float] = []
    c_active_list: list[float] = []

    # Combined elimination of prodrug (conversion + direct elimination)
    k_prod_total = k_conv_per_h + k_elim_prodrug_per_h

    for step in range(n_steps):
        t = step * dt_h
        times_h_list.append(t)
        c_prodrug_list.append(c_prod)
        c_active_list.append(c_act)

        # Input from gut (oral only)
        if route == "oral" and a_gut > 0:
            gut_input = ka * a_gut / vd_prodrug_L
            da_gut = -ka * a_gut
        else:
            gut_input = 0.0
            da_gut = 0.0

        # ODEs
        dc_prod = gut_input - k_prod_total * c_prod
        dc_act = (
            k_conv_per_h * c_prod * vd_prodrug_L / vd_active_L * f_conversion
            - k_elim_active_per_h * c_act
        )

        # Forward Euler update
        a_gut += da_gut * dt_h
        c_prod += dc_prod * dt_h
        c_act += dc_act * dt_h

        # Clamp to zero (numerical safety)
        if c_prod < 0.0:
            c_prod = 0.0
        if c_act < 0.0:
            c_act = 0.0
        if a_gut < 0.0:
            a_gut = 0.0

    # --- Derived metrics ---
    cmax_prodrug = max(c_prodrug_list)
    cmax_active = max(c_active_list)

    tmax_active = times_h_list[c_active_list.index(cmax_active)]

    auc_prodrug = _trapz(times_h_list, c_prodrug_list)
    auc_active = _trapz(times_h_list, c_active_list)

    total_auc = auc_prodrug + auc_active
    conv_eff = auc_active / total_auc if total_auc > 0 else 0.0

    notes = (
        f"k_conv={k_conv_per_h}/h, k_elim_prod={k_elim_prodrug_per_h}/h, "
        f"k_elim_act={k_elim_active_per_h}/h, f_conv={f_conversion:.2f}, "
        f"route={route}, site={activation_site}"
    )

    return ProdrugActivationResult(
        prodrug_name=prodrug_name,
        active_drug_name=active_drug_name,
        route=route,
        times_h=times_h_list,
        c_prodrug_mg_L=c_prodrug_list,
        c_active_mg_L=c_active_list,
        cmax_prodrug_mg_L=cmax_prodrug,
        cmax_active_mg_L=cmax_active,
        tmax_active_h=tmax_active,
        auc_prodrug_mg_h_per_L=auc_prodrug,
        auc_active_mg_h_per_L=auc_active,
        conversion_efficiency=conv_eff,
        activation_site=activation_site,
        notes=notes,
    )


def compare_prodrug_routes(
    prodrug_name: str,
    active_drug_name: str,
    dose_mg: float,
    **kwargs,
) -> list[ProdrugActivationResult]:
    """Simulate prodrug PK for both oral and IV routes.

    Returns results sorted by auc_active_mg_h_per_L descending.

    Parameters
    ----------
    prodrug_name:
        Name of the prodrug.
    active_drug_name:
        Name of the active metabolite.
    dose_mg:
        Administered dose (mg).
    **kwargs:
        Additional keyword arguments forwarded to simulate_prodrug_activation
        (except ``route``).

    Returns
    -------
    list[ProdrugActivationResult]
        Two results (oral and IV) sorted by AUC of active drug descending.
    """
    results = []
    for route in ("oral", "iv"):
        # Remove route from kwargs if accidentally passed
        kw = {k: v for k, v in kwargs.items() if k != "route"}
        result = simulate_prodrug_activation(
            prodrug_name=prodrug_name,
            active_drug_name=active_drug_name,
            dose_mg=dose_mg,
            route=route,
            **kw,
        )
        results.append(result)

    results.sort(key=lambda r: r.auc_active_mg_h_per_L, reverse=True)
    return results


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _trapz(times: list[float], values: list[float]) -> float:
    """Manual trapezoidal integration."""
    total = 0.0
    for i in range(1, len(times)):
        dt = times[i] - times[i - 1]
        total += 0.5 * (values[i - 1] + values[i]) * dt
    return total
