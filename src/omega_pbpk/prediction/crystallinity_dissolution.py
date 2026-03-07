"""Phase 1080 — Drug Crystallinity Effects on Dissolution.

Models how drug crystallinity (0-100%) affects dissolution rate and oral absorption.
Amorphous > partially crystalline > fully crystalline in dissolution rate.
"""

from __future__ import annotations

from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Rate constants
# ---------------------------------------------------------------------------
_K_DISS_AMORPHOUS = 0.5  # /h  — rapid dissolution for amorphous
_K_DISS_CRYSTALLINE = 0.05  # /h  — slow dissolution for crystalline
_K_PRECIP_DEFAULT = 0.5  # /h  — precipitation rate (spring-parachute)
_T_SUPER_H = 2.0  # h   — supersaturation window
_SUPERSATURATION_RATIO = 3.0  # fold above crystalline solubility limit

_LARGE_TIME = 999.0  # sentinel for "not reached"


def _trapz(times: list[float], values: list[float]) -> float:
    """Manual trapezoidal integration."""
    auc = 0.0
    for i in range(1, len(times)):
        auc += 0.5 * (values[i - 1] + values[i]) * (times[i] - times[i - 1])
    return auc


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------
@dataclass
class CrystallinityDissolutionResult:
    drug_name: str
    crystallinity_pct: float
    dose_mg: float
    times_h: list
    a_solid_mg: list
    a_dissolved_mg: list
    dissolution_fraction_list: list
    f_dissolved_at_1h: float
    f_dissolved_at_4h: float
    f_dissolved_complete: float
    t_80pct_dissolved_h: float
    absorption_enhancement_factor: float
    notes: str


# ---------------------------------------------------------------------------
# Core simulation
# ---------------------------------------------------------------------------
def simulate_crystallinity_dissolution(
    drug_name: str,
    dose_mg: float,
    crystallinity_pct: float,
    solubility_mg_mL: float = 0.1,
    volume_lumen_mL: float = 250.0,
    k_precip_per_h: float = _K_PRECIP_DEFAULT,
    t_end_h: float = 8.0,
    dt_h: float = 0.05,
) -> CrystallinityDissolutionResult:
    """Simulate drug dissolution as function of crystallinity fraction."""
    # --- Validation ---
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if not (0.0 <= crystallinity_pct <= 100.0):
        raise ValueError("crystallinity_pct must be in [0, 100]")
    if solubility_mg_mL <= 0:
        raise ValueError("solubility_mg_mL must be > 0")
    if volume_lumen_mL <= 0:
        raise ValueError("volume_lumen_mL must be > 0")

    f_cryst = crystallinity_pct / 100.0
    f_amor = 1.0 - f_cryst

    # Blended dissolution rate constant
    k_diss = _K_DISS_AMORPHOUS * f_amor + _K_DISS_CRYSTALLINE * f_cryst

    # Maximum dissolved amount based on thermodynamic solubility
    S_max_mg = solubility_mg_mL * volume_lumen_mL  # total capacity in lumen

    # Crystalline solubility limit (no supersaturation)
    S_cryst_limit = S_max_mg  # same thermodynamic solubility

    # Supersaturation ceiling for amorphous fraction
    S_super_limit = _SUPERSATURATION_RATIO * S_cryst_limit

    # Allow supersaturation only for predominantly amorphous (f_cryst < 0.5)
    allow_super = f_cryst < 0.5

    # Initial conditions
    A_solid = dose_mg
    A_diss = 0.0

    times: list[float] = []
    a_solid_list: list[float] = []
    a_diss_list: list[float] = []

    t = 0.0
    n_steps = int(round(t_end_h / dt_h))

    for step in range(n_steps + 1):
        times.append(t)
        a_solid_list.append(A_solid)
        a_diss_list.append(A_diss)

        if step == n_steps:
            break

        # Dissolution — only when solid remains and below dissolution ceiling
        if A_solid > 0:
            diss_ceiling = S_super_limit if allow_super else S_cryst_limit
            if A_diss < diss_ceiling:
                # driving force toward ceiling
                dA_diss = k_diss * (diss_ceiling - A_diss) * dt_h
                dA_diss = min(dA_diss, A_solid)
                A_solid -= dA_diss
                A_diss += dA_diss
                A_solid = max(0.0, A_solid)

        # Spring-parachute precipitation after supersaturation window
        if allow_super and t >= _T_SUPER_H and A_diss > S_cryst_limit:
            excess = A_diss - S_cryst_limit
            precip = k_precip_per_h * excess * dt_h
            A_diss -= precip
            A_solid += precip
            A_diss = max(0.0, A_diss)

        # Clip dissolved to thermodynamic max when not supersaturating
        if not allow_super:
            if A_diss > S_cryst_limit:
                overflow = A_diss - S_cryst_limit
                A_diss = S_cryst_limit
                A_solid += overflow

        t += dt_h

    # Compute dissolution fraction at each time
    diss_frac_list: list[float] = [min(1.0, a / dose_mg) for a in a_diss_list]

    def _interp_at(target_t: float) -> float:
        if target_t >= times[-1]:
            return diss_frac_list[-1]
        for i in range(1, len(times)):
            if times[i] >= target_t:
                t0, t1 = times[i - 1], times[i]
                v0, v1 = diss_frac_list[i - 1], diss_frac_list[i]
                frac = (target_t - t0) / (t1 - t0)
                return v0 + frac * (v1 - v0)
        return diss_frac_list[-1]

    f_1h = _interp_at(1.0)
    f_4h = _interp_at(4.0)
    f_final = diss_frac_list[-1]

    # Time to 80% dissolution
    t_80 = _LARGE_TIME
    for i, f in enumerate(diss_frac_list):
        if f >= 0.80:
            if i > 0:
                t0, t1 = times[i - 1], times[i]
                v0, v1 = diss_frac_list[i - 1], diss_frac_list[i]
                if v1 > v0:
                    t_80 = t0 + (0.80 - v0) / (v1 - v0) * (t1 - t0)
                else:
                    t_80 = times[i]
            else:
                t_80 = times[0]
            break

    # Absorption enhancement factor: AUC of dissolved curve vs 100% crystalline
    # Use AUC of dissolved amount as proxy; reference = crystalline case
    auc_dissolved = _trapz(times, a_diss_list)

    # Compute reference AUC for 100% crystalline inline (avoid recursion)
    k_ref = _K_DISS_CRYSTALLINE
    S_ref = S_cryst_limit
    A_sol_ref = dose_mg
    A_dis_ref = 0.0
    a_dis_ref_list: list[float] = []
    for step2 in range(n_steps + 1):
        a_dis_ref_list.append(A_dis_ref)
        if step2 == n_steps:
            break
        if A_sol_ref > 0 and A_dis_ref < S_ref:
            dA = k_ref * (S_ref - A_dis_ref) * dt_h
            dA = min(dA, A_sol_ref)
            A_sol_ref -= dA
            A_dis_ref += dA
            A_sol_ref = max(0.0, A_sol_ref)

    auc_ref = _trapz(times, a_dis_ref_list)
    enhancement = (auc_dissolved / auc_ref) if auc_ref > 0 else 1.0

    notes = (
        f"crystallinity={crystallinity_pct:.1f}%, "
        f"k_diss={k_diss:.4f}/h, "
        f"S_max={S_max_mg:.2f} mg, "
        f"supersaturation={'yes' if allow_super else 'no'}"
    )

    return CrystallinityDissolutionResult(
        drug_name=drug_name,
        crystallinity_pct=crystallinity_pct,
        dose_mg=dose_mg,
        times_h=times,
        a_solid_mg=a_solid_list,
        a_dissolved_mg=a_diss_list,
        dissolution_fraction_list=diss_frac_list,
        f_dissolved_at_1h=f_1h,
        f_dissolved_at_4h=f_4h,
        f_dissolved_complete=f_final,
        t_80pct_dissolved_h=t_80,
        absorption_enhancement_factor=enhancement,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Comparison helper
# ---------------------------------------------------------------------------
def compare_crystallinity_levels(
    drug_name: str,
    dose_mg: float,
    crystallinity_list: list[float],
    solubility_mg_mL: float = 0.1,
    volume_lumen_mL: float = 250.0,
    k_precip_per_h: float = _K_PRECIP_DEFAULT,
    t_end_h: float = 8.0,
    dt_h: float = 0.05,
) -> list[CrystallinityDissolutionResult]:
    """Simulate across crystallinity levels; sorted by f_dissolved_at_4h descending."""
    results = [
        simulate_crystallinity_dissolution(
            drug_name=drug_name,
            dose_mg=dose_mg,
            crystallinity_pct=c,
            solubility_mg_mL=solubility_mg_mL,
            volume_lumen_mL=volume_lumen_mL,
            k_precip_per_h=k_precip_per_h,
            t_end_h=t_end_h,
            dt_h=dt_h,
        )
        for c in crystallinity_list
    ]
    results.sort(key=lambda r: r.f_dissolved_at_4h, reverse=True)
    return results
