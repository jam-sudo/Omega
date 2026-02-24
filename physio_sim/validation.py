from __future__ import annotations

import warnings

import pandas as pd

from physio_sim.config import CompoundConfig, SubjectConfig


def mass_balance_check(
    timecourse: pd.DataFrame, dose_mg: float, tolerance_mg: float = 1e-3
) -> list[str]:
    """Check whether total model mass is conserved when accounting for urine sink."""
    amount_cols = [c for c in timecourse.columns if c.startswith("A_")]
    total_mass = float(timecourse[amount_cols].iloc[-1].sum())
    deviation = total_mass - dose_mg
    warnings_out: list[str] = []
    if abs(deviation) > tolerance_mg:
        msg = (
            f"Mass balance deviation exceeds tolerance: deviation={deviation:.6g} mg, "
            f"tolerance={tolerance_mg:.6g} mg"
        )
        warnings.warn(msg, RuntimeWarning, stacklevel=2)
        warnings_out.append(msg)
    return warnings_out


def physiologic_sanity_check(
    subject: SubjectConfig,
    compound: CompoundConfig,
    flow_tolerance_frac: float = 0.15,
) -> list[str]:
    """Run non-fatal physiological plausibility checks."""
    warnings_out: list[str] = []

    tissue_flow_sum = (
        subject.liver.flow_L_per_h
        + subject.kidney.flow_L_per_h
        + sum(t.flow_L_per_h for t in subject.tissues.values())
        - subject.tissues["portal_vein"].flow_L_per_h
    )
    co = subject.cardiac_output_L_per_h
    if co > 0:
        rel_err = abs(tissue_flow_sum - co) / co
        if rel_err > flow_tolerance_frac:
            msg = (
                "Total tissue blood flows are not close to cardiac output: "
                f"sum_flows={tissue_flow_sum:.3f} L/h, cardiac_output={co:.3f} L/h"
            )
            warnings.warn(msg, RuntimeWarning, stacklevel=2)
            warnings_out.append(msg)

    volumes = [subject.plasma_volume_L, subject.liver.volume_L, subject.kidney.volume_L] + [
        t.volume_L for t in subject.tissues.values()
    ]
    if any(v <= 0 for v in volumes):
        msg = "One or more compartment volumes are non-positive."
        warnings.warn(msg, RuntimeWarning, stacklevel=2)
        warnings_out.append(msg)

    if not (0.0 < compound.fu_plasma < 1.0):
        msg = (
            "fu_plasma should be strictly between 0 and 1 for plausibility; "
            f"got {compound.fu_plasma:.4g}."
        )
        warnings.warn(msg, RuntimeWarning, stacklevel=2)
        warnings_out.append(msg)

    return warnings_out
