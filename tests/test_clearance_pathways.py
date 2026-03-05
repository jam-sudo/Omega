"""Tests verifying that well-stirred hepatic and renal clearance pathways are
correctly implemented in the PBPK engine.

TestHepatic: caffeine IV, analytical AUC vs simulated AUC (low-extraction approx).
TestRenal:   metformin IV, >95% of dose recovered in renal excretion sink.
"""

from __future__ import annotations

import numpy as np
import pytest

from omega_pbpk.core.body import IDX_EXC_RENAL, WholeBodyPBPK
from omega_pbpk.drugs.caffeine import CAFFEINE
from omega_pbpk.drugs.metformin import METFORMIN


@pytest.mark.integration
class TestHepatic:
    """Analytical AUC check for a low-extraction hepatically cleared drug."""

    def test_caffeine_iv_auc_vs_analytical(self) -> None:
        """Simulated AUC for caffeine IV should be within 20% of analytical value.

        Low-extraction approximation:
            CLh ≈ fup * CLint_hepatic  (when fup*CLint << Q_liver ≈ 103 L/h)
            CLtotal = fup * CLint_hepatic + CLr
            AUC_analytical = Dose / CLtotal
        """
        dose_mg = 100.0
        model = WholeBodyPBPK(CAFFEINE)
        model.setup_iv(dose_mg=dose_mg)
        result = model.simulate(t_end_h=120.0, dt_h=0.1)

        cp = result.plasma_concentration()
        t = result.time_h
        auc_sim = float(np.trapz(cp, t))

        # Analytical CL under low-extraction approximation
        fup = CAFFEINE.fup
        clint = CAFFEINE.clint_hepatic_L_per_h
        clr = CAFFEINE.clr_L_per_h
        cl_total = fup * clint + clr  # ~1.575 L/h
        auc_analytical = dose_mg / cl_total

        ratio = auc_sim / auc_analytical
        assert abs(ratio - 1.0) < 0.20, (
            f"AUC ratio {ratio:.3f} outside 20% tolerance: "
            f"simulated={auc_sim:.1f} mg·h/L, analytical={auc_analytical:.1f} mg·h/L"
        )


@pytest.mark.integration
class TestRenal:
    """Mass-sink check for a purely renally cleared drug (metformin)."""

    def test_metformin_iv_renal_recovery(self) -> None:
        """After 72 h, >95% of the metformin IV dose should be in the renal sink.

        Metformin: CLint_hepatic=0, CLr=25 L/h — pure renal elimination.
        IDX_EXC_RENAL = 31.
        """
        dose_mg = 500.0
        model = WholeBodyPBPK(METFORMIN)
        model.setup_iv(dose_mg=dose_mg)
        result = model.simulate(t_end_h=72.0, dt_h=0.1)

        amounts_final = result.amounts[-1]
        renal_fraction = amounts_final[IDX_EXC_RENAL] / dose_mg

        assert renal_fraction >= 0.95, (
            f"Renal recovery {renal_fraction:.3f} is below 95% threshold. "
            f"Renal sink: {amounts_final[IDX_EXC_RENAL]:.2f} mg of {dose_mg} mg dose."
        )
