"""Indirect response PK/PD models — Types I–IV (Dayneka/Mager framework)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from omega_pbpk._compat import np_trapz


class IRType(str, Enum):
    TYPE_I = "type_i"      # inhibit kin
    TYPE_II = "type_ii"    # inhibit kout
    TYPE_III = "type_iii"  # stimulate kin
    TYPE_IV = "type_iv"    # stimulate kout


@dataclass(frozen=True)
class IndirectResponseResult:
    drug_name: str
    ir_type: str
    times_h: list[float]
    cp_mg_L: list[float]
    response: list[float]
    r0: float
    rmax: float
    rmin: float
    tmax_h: float
    auc_response: float
    return_to_baseline_h: float
    kin: float
    kout: float
    emax: float
    ec50_mg_L: float
    hill: float


def simulate_indirect_response(
    drug_name: str,
    times_h: list[float],
    cp_mg_L: list[float],
    ir_type: IRType,
    kin: float,
    kout: float,
    emax: float = 1.0,
    ec50_mg_L: float = 1.0,
    hill: float = 1.0,
) -> IndirectResponseResult:
    """Simulate an indirect response PD model given a plasma concentration profile."""
    if kin <= 0:
        raise ValueError("kin must be > 0")
    if kout <= 0:
        raise ValueError("kout must be > 0")
    if ec50_mg_L <= 0:
        raise ValueError("ec50_mg_L must be > 0")
    if emax < 0:
        raise ValueError("emax must be >= 0")

    n = len(times_h)
    if n < 2 or len(cp_mg_L) != n:
        raise ValueError("times_h and cp_mg_L must have equal length >= 2")

    r0 = kin / kout
    r = r0
    response = [r0]

    ir = IRType(ir_type)

    for i in range(1, n):
        dt = times_h[i] - times_h[i - 1]
        cp = max(cp_mg_L[i - 1], 0.0)

        # Hill function
        cp_h = cp ** hill
        ec_h = ec50_mg_L ** hill
        h = emax * cp_h / (ec_h + cp_h) if (ec_h + cp_h) > 0 else 0.0

        if ir == IRType.TYPE_I:
            drdt = kin * (1.0 - h) - kout * r
        elif ir == IRType.TYPE_II:
            drdt = kin - kout * (1.0 - h) * r
        elif ir == IRType.TYPE_III:
            drdt = kin * (1.0 + h) - kout * r
        else:  # TYPE_IV
            drdt = kin - kout * (1.0 + h) * r

        r = max(r + drdt * dt, 0.0)
        response.append(r)

    # Derived metrics
    rmax = max(response)
    rmin = min(response)
    tmax_idx = response.index(rmax)
    tmax_h_val = times_h[tmax_idx]
    auc_response = float(np_trapz(response, times_h))

    # Return to baseline: first time AFTER peak where |R - R0|/R0 < 0.10
    rtb = times_h[-1]
    if r0 > 0:
        for i in range(tmax_idx + 1, n):
            if abs(response[i] - r0) / r0 < 0.10:
                rtb = times_h[i]
                break

    return IndirectResponseResult(
        drug_name=drug_name,
        ir_type=ir_type if isinstance(ir_type, str) else ir_type.value,
        times_h=list(times_h),
        cp_mg_L=list(cp_mg_L),
        response=response,
        r0=r0,
        rmax=rmax,
        rmin=rmin,
        tmax_h=tmax_h_val,
        auc_response=auc_response,
        return_to_baseline_h=rtb,
        kin=kin,
        kout=kout,
        emax=emax,
        ec50_mg_L=ec50_mg_L,
        hill=hill,
    )


def simulate_indirect_iv(
    drug_name: str,
    dose_mg: float,
    cl_L_per_h: float,
    vd_L: float,
    ir_type: IRType,
    kin: float,
    kout: float,
    emax: float = 1.0,
    ec50_mg_L: float = 1.0,
    hill: float = 1.0,
    t_end_h: float = 48.0,
    dt_h: float = 0.05,
) -> IndirectResponseResult:
    """Convenience: IV bolus PK + indirect response PD."""
    ke = cl_L_per_h / vd_L
    t = np.linspace(0.0, t_end_h, max(int(t_end_h / dt_h), 1) + 1)
    cp = (dose_mg / vd_L) * np.exp(-ke * t)
    return simulate_indirect_response(
        drug_name, t.tolist(), cp.tolist(), ir_type, kin, kout, emax, ec50_mg_L, hill
    )


__all__ = [
    "IRType",
    "IndirectResponseResult",
    "simulate_indirect_response",
    "simulate_indirect_iv",
]
