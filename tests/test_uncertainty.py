from __future__ import annotations

import numpy as np

from physio_sim.config import load_compound, load_subject
from physio_sim.uncertainty import draw_parameter, monte_carlo_propagation


def test_draw_parameter_fixed_value() -> None:
    value = draw_parameter(1.25, rng=np.random.default_rng(0))
    assert value == 1.25


def test_monte_carlo_outputs_metrics() -> None:
    subject = load_subject("examples/subject_default.yaml")
    compound = load_compound("examples/compound_caffeine.yaml")

    result = monte_carlo_propagation(
        subject=subject,
        compound=compound,
        dose_mg=50.0,
        route="oral",
        t_end_h=4.0,
        dt_out_h=0.5,
        n_samples=8,
        parameter_specs={
            "ka_per_h": {"dist": "lognormal", "mean": 0.0, "sd": 0.2},
            "clint_L_per_h": {"dist": "normal", "mean": 8.0, "sd": 1.0},
        },
        seed=123,
    )

    assert len(result.sample_metrics) == 8
    assert {"Cmax_mg_per_L", "AUC0_tend_mg_h_per_L", "ka_per_h", "clint_L_per_h"}.issubset(
        result.sample_metrics.columns
    )
