from __future__ import annotations

import numpy as np

from physio_sim.qsp.registry import get_qsp_model


def test_turnover_rhs_negative_when_drug_and_no_input() -> None:
    model = get_qsp_model("turnover")
    config = {"kin": 0.0, "kout": 0.1, "emax": 1.0, "ec50_mg_per_L": 1.0, "hill": 1.0}
    rhs = model.rhs(0.0, np.array([5.0], dtype=float), {"C_signal": 10.0}, config)
    assert rhs[0] < 0.0


def test_turnover_nonnegative_guard_at_zero() -> None:
    model = get_qsp_model("turnover")
    config = {"kin": 0.0, "kout": 0.1, "emax": 1.0, "ec50_mg_per_L": 1.0, "hill": 1.0}
    rhs = model.rhs(0.0, np.array([0.0], dtype=float), {"C_signal": 10.0}, config)
    assert rhs[0] >= 0.0
