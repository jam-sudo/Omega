"""Phase 176: Drug PK in retinal tissue (intravitreal / systemic routes)."""

from dataclasses import dataclass, field

# Compartment volumes
_V_VIT = 0.004  # L  (4.0 mL)
_V_RET = 0.0001  # L  (0.1 mL)
_V_CHO = 0.0003  # L  (0.3 mL)
_VD_SYS = 5.0  # L

# Rate constants (1/h)
_KVR_BASE = 0.05  # vitreous → retina (base; scaled by MW)
_KRV = 0.10  # retina → vitreous
_KRC = 0.02  # retina → choroid
_KCR = 0.50  # choroid → retina
_K_AQ_DRAIN = 0.05  # vitreous drainage to aqueous
_KE_CHO = 0.10  # choroid → systemic
_KE_SYS = 1.00  # systemic elimination


@dataclass
class RetinalPKResult:
    drug_name: str
    dose_mg: float
    route: str
    mw_Da: float
    times_h: list = field(default_factory=list)
    c_vitreous_mg_L: list = field(default_factory=list)
    c_retina_mg_L: list = field(default_factory=list)
    c_systemic_mg_L: list = field(default_factory=list)
    cmax_retina_mg_L: float = 0.0
    tmax_retina_h: float = 0.0
    auc_retina_mg_h_per_L: float = 0.0
    auc_vitreous_mg_h_per_L: float = 0.0
    retina_to_vitreous_auc_ratio: float = 0.0
    t_half_vitreous_h: float = 0.0
    systemic_exposure_pct: float = 0.0
    notes: str = ""


def _trapz(xs: list, ys: list) -> float:
    auc = 0.0
    for i in range(1, len(xs)):
        auc += 0.5 * (ys[i] + ys[i - 1]) * (xs[i] - xs[i - 1])
    return auc


def simulate_retinal_pk(
    drug_name: str,
    dose_mg: float,
    route: str = "intravitreal",
    mw_Da: float = 500.0,
    t_end_h: float = 720.0,
    dt_h: float = 1.0,
) -> RetinalPKResult:
    """Simulate drug PK in retinal tissue using 4-compartment forward Euler ODE."""
    if dose_mg <= 0:
        raise ValueError(f"dose_mg must be > 0, got {dose_mg}")
    if route not in {"intravitreal", "systemic"}:
        raise ValueError(f"route must be 'intravitreal' or 'systemic', got {route!r}")
    if mw_Da <= 0:
        raise ValueError(f"mw_Da must be > 0, got {mw_Da}")

    # MW scaling for kvr
    kvr = _KVR_BASE * max(0.1, min(1.0, 1000.0 / mw_Da))

    # Initial conditions (mg/L)
    if route == "intravitreal":
        c_vit = dose_mg / _V_VIT
        c_ret = 0.0
        c_cho = 0.0
        c_sys = 0.0
    else:  # systemic
        c_vit = 0.0
        c_ret = 0.0
        c_cho = 0.0
        c_sys = dose_mg / _VD_SYS

    times: list = [0.0]
    c_vitreous_list: list = [c_vit]
    c_retina_list: list = [c_ret]
    c_systemic_list: list = [c_sys]

    n_steps = int(t_end_h / dt_h)
    t = 0.0

    for _ in range(n_steps):
        # ODEs
        dv = -kvr * c_vit + _KRV * c_ret * _V_RET / _V_VIT - _K_AQ_DRAIN * c_vit
        dr = (
            kvr * _V_VIT / _V_RET * c_vit
            - _KRV * c_ret
            - _KRC * c_ret
            + _KCR * c_cho * _V_CHO / _V_RET
        )
        dc = _KRC * _V_RET / _V_CHO * c_ret - _KCR * c_cho - _KE_CHO * c_cho
        ds = _KE_CHO * c_cho * _V_CHO / _VD_SYS - _KE_SYS * c_sys

        c_vit = max(0.0, c_vit + dv * dt_h)
        c_ret = max(0.0, c_ret + dr * dt_h)
        c_cho = max(0.0, c_cho + dc * dt_h)
        c_sys = max(0.0, c_sys + ds * dt_h)

        t += dt_h
        times.append(t)
        c_vitreous_list.append(c_vit)
        c_retina_list.append(c_ret)
        c_systemic_list.append(c_sys)

    # Derived metrics
    cmax_retina = max(c_retina_list)
    tmax_retina = times[c_retina_list.index(cmax_retina)]

    auc_retina = _trapz(times, c_retina_list)
    auc_vitreous = _trapz(times, c_vitreous_list)
    auc_systemic = _trapz(times, c_systemic_list)

    if auc_vitreous > 0:
        retina_to_vitreous_ratio = auc_retina / auc_vitreous
        systemic_exposure_pct = auc_systemic / auc_vitreous * 100.0
    else:
        retina_to_vitreous_ratio = 0.0
        systemic_exposure_pct = 0.0

    # Estimate t_half_vitreous from decay phase
    peak_vit = max(c_vitreous_list)
    half_peak = peak_vit / 2.0
    t_half_vit = 0.0
    if peak_vit > 0 and half_peak > 0:
        peak_idx = c_vitreous_list.index(peak_vit)
        # Search after peak for crossing half
        for i in range(peak_idx + 1, len(c_vitreous_list)):
            if c_vitreous_list[i] <= half_peak:
                # Linear interpolation
                c_prev = c_vitreous_list[i - 1]
                c_curr = c_vitreous_list[i]
                t_prev = times[i - 1]
                t_curr = times[i]
                if c_prev != c_curr:
                    frac = (c_prev - half_peak) / (c_prev - c_curr)
                    t_half_vit = (t_curr - t_prev) * frac + t_prev - times[peak_idx]
                else:
                    t_half_vit = t_curr - times[peak_idx]
                break

    mw_note = f"MW={mw_Da:.0f} Da, kvr={kvr:.4f}/h."
    route_note = (
        "Intravitreal injection; vitreous is primary depot."
        if route == "intravitreal"
        else "Systemic administration; retinal exposure via choroidal supply."
    )
    notes = f"{route_note} {mw_note}"

    return RetinalPKResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        route=route,
        mw_Da=mw_Da,
        times_h=times,
        c_vitreous_mg_L=c_vitreous_list,
        c_retina_mg_L=c_retina_list,
        c_systemic_mg_L=c_systemic_list,
        cmax_retina_mg_L=cmax_retina,
        tmax_retina_h=tmax_retina,
        auc_retina_mg_h_per_L=auc_retina,
        auc_vitreous_mg_h_per_L=auc_vitreous,
        retina_to_vitreous_auc_ratio=retina_to_vitreous_ratio,
        t_half_vitreous_h=t_half_vit,
        systemic_exposure_pct=systemic_exposure_pct,
        notes=notes,
    )
