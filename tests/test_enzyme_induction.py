"""Tests for Phase 37 enzyme induction modeling (PXR/CAR pathway).

Covers:
  - TestInductionUnit (7): core math functions and single-call behavior
  - TestInductionIntegration (5): full simulation pipelines and formatting
  - TestInductionAPI (2): POST /induction endpoint contract
"""

from __future__ import annotations

import math

from fastapi.testclient import TestClient

from omega_pbpk.api.app import app
from omega_pbpk.clinical.enzyme_induction import (
    AutoInductionResult,
    InductionParameters,
    compute_ksyn,
    compute_net_ddi_effect,
    format_induction_summary,
    simulate_auto_induction,
    simulate_induction_time_course,
    steady_state_induction_factor,
    time_to_steady_state,
)

client = TestClient(app)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

_RIFAMPIN_PARAMS = InductionParameters(
    inducer_name="Rifampin",
    target_enzyme="CYP3A4",
    emax=8.0,
    ec50_uM=0.5,
    inducer_conc_uM=10.0,
    kdeg_h=0.03,
    baseline_activity=1.0,
)

_AUTO_PARAMS = InductionParameters(
    inducer_name="AutoDrug",
    target_enzyme="CYP3A4",
    emax=5.0,
    ec50_uM=1.0,
    inducer_conc_uM=5.0,
    kdeg_h=0.03,
    baseline_activity=1.0,
)


# ---------------------------------------------------------------------------
# TestInductionUnit — core math functions
# ---------------------------------------------------------------------------


class TestInductionUnit:
    def test_compute_ksyn(self):
        """ksyn = kdeg * baseline_activity = 0.03 * 1.0 = 0.03."""
        assert abs(compute_ksyn(0.03, 1.0) - 0.03) < 1e-9

    def test_steady_state_induction_factor(self):
        """Rifampin: Emax=8, C=10 µM, EC50=0.5 µM → ~8.619."""
        # fold = 1 + 8 * 10 / (0.5 + 10) = 1 + 80/10.5 ≈ 8.619
        expected = 1.0 + 8.0 * 10.0 / (0.5 + 10.0)
        result = steady_state_induction_factor(emax=8.0, conc_uM=10.0, ec50_uM=0.5)
        assert abs(result - expected) < 1e-9

    def test_steady_state_induction_factor_zero_conc(self):
        """Zero inducer concentration → fold change = 1.0 (no induction)."""
        assert steady_state_induction_factor(emax=8.0, conc_uM=0.0, ec50_uM=0.5) == 1.0

    def test_time_to_steady_state(self):
        """t90% = -ln(0.1) / 0.03 ≈ 76.75 h for CYP3A4 kdeg=0.03 /h."""
        expected = -math.log(0.1) / 0.03
        result = time_to_steady_state(kdeg_h=0.03, fraction=0.9)
        assert abs(result - expected) < 1e-9

    def test_simulate_induction_reaches_ss(self):
        """Final enzyme activity must reach within 1% of steady-state fold."""
        tc = simulate_induction_time_course(_RIFAMPIN_PARAMS, t_end_h=500.0, n_points=300)
        final_activity = tc.enzyme_activity[-1]
        expected_ss = tc.steady_state_fold * _RIFAMPIN_PARAMS.baseline_activity
        assert abs(final_activity - expected_ss) / expected_ss < 0.01

    def test_simulate_induction_starts_at_baseline(self):
        """First enzyme activity point must equal baseline_activity."""
        tc = simulate_induction_time_course(_RIFAMPIN_PARAMS, t_end_h=336.0, n_points=200)
        assert abs(tc.enzyme_activity[0] - _RIFAMPIN_PARAMS.baseline_activity) < 1e-6

    def test_net_ddi_inhibition_dominant(self):
        """R1=5, induction=1.5, fm=1.0 → inhibition-dominant mechanism."""
        result = compute_net_ddi_effect(
            inhibition_r1=5.0,
            induction_fold=1.5,
            fm_enzyme=1.0,
            target_enzyme="CYP3A4",
        )
        assert result.dominant_mechanism == "inhibition"
        assert len(result.warnings) >= 1  # both inhibition and induction present


# ---------------------------------------------------------------------------
# TestInductionIntegration — full simulation pipelines
# ---------------------------------------------------------------------------


class TestInductionIntegration:
    def test_auto_induction_increases_clearance(self):
        """Auto-induction simulation must show CL_induced > CL_baseline."""
        ar = simulate_auto_induction(
            params=_AUTO_PARAMS,
            initial_conc_uM=10.0,
            cl_baseline_L_per_h=5.0,
            vd_L=70.0,
            t_end_h=336.0,
            n_points=500,
        )
        assert isinstance(ar, AutoInductionResult)
        assert ar.cl_induced_L_per_h > ar.cl_baseline_L_per_h

    def test_auto_induction_decreases_half_life(self):
        """Auto-induction must shorten elimination half-life."""
        ar = simulate_auto_induction(
            params=_AUTO_PARAMS,
            initial_conc_uM=10.0,
            cl_baseline_L_per_h=5.0,
            vd_L=70.0,
            t_end_h=336.0,
            n_points=500,
        )
        assert ar.half_life_induced_h < ar.half_life_baseline_h

    def test_auto_induction_concentration_declines_faster(self):
        """Concentration at t=48h must be lower with auto-induction than without."""
        # No induction: emax=0 → flat CL; use 48h so drug is not yet fully eliminated
        no_induction_params = InductionParameters(
            inducer_name="NoInduction",
            target_enzyme="CYP3A4",
            emax=0.0,
            ec50_uM=1.0,
            inducer_conc_uM=5.0,
            kdeg_h=0.03,
            baseline_activity=1.0,
        )
        ar_no = simulate_auto_induction(
            params=no_induction_params,
            initial_conc_uM=10.0,
            cl_baseline_L_per_h=5.0,
            vd_L=70.0,
            t_end_h=48.0,
            n_points=500,
        )
        ar_ind = simulate_auto_induction(
            params=_AUTO_PARAMS,
            initial_conc_uM=10.0,
            cl_baseline_L_per_h=5.0,
            vd_L=70.0,
            t_end_h=48.0,
            n_points=500,
        )
        # Final concentration must be lower when induction is active
        assert ar_ind.concentration_uM[-1] < ar_no.concentration_uM[-1]

    def test_net_ddi_induction_dominant(self):
        """R1=1.2, induction=5.0 → induction-dominant DDI."""
        result = compute_net_ddi_effect(
            inhibition_r1=1.2,
            induction_fold=5.0,
            fm_enzyme=1.0,
            target_enzyme="CYP3A4",
        )
        assert result.dominant_mechanism == "induction"

    def test_format_induction_summary(self):
        """Summary string must contain inducer name, enzyme, fold, and kdeg."""
        tc = simulate_induction_time_course(_RIFAMPIN_PARAMS, t_end_h=336.0, n_points=200)
        summary = format_induction_summary(tc)
        assert isinstance(summary, str)
        assert "Rifampin" in summary
        assert "CYP3A4" in summary
        assert "Steady-state fold" in summary
        assert "kdeg" in summary


# ---------------------------------------------------------------------------
# TestInductionAPI — POST /induction endpoint
# ---------------------------------------------------------------------------

_BASE_INDUCTION_PAYLOAD = {
    "inducer_name": "Rifampin",
    "target_enzyme": "CYP3A4",
    "emax": 8.0,
    "ec50_uM": 0.5,
    "inducer_conc_uM": 10.0,
    "kdeg_h": 0.03,
    "baseline_activity": 1.0,
    "t_end_h": 336.0,
    "n_points": 100,
}


class TestInductionAPI:
    def test_induction_endpoint_basic(self):
        """POST /induction → 200, mode=standard, steady_state_fold > 1."""
        r = client.post("/induction", json=_BASE_INDUCTION_PAYLOAD)
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["mode"] == "standard"
        assert isinstance(body["summary"], str)
        tc = body["time_course"]
        assert tc is not None
        assert tc["steady_state_fold"] > 1.0
        assert len(tc["time_points"]) == 100

    def test_induction_endpoint_auto_induction(self):
        """POST /induction with auto_induction=True → cl_induced > cl_baseline."""
        payload = {
            **_BASE_INDUCTION_PAYLOAD,
            "auto_induction": True,
            "initial_conc_uM": 10.0,
            "cl_baseline_L_per_h": 5.0,
            "vd_L": 70.0,
        }
        r = client.post("/induction", json=payload)
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["mode"] == "auto_induction"
        ar = body["auto_induction"]
        assert ar is not None
        assert ar["cl_induced_L_per_h"] > ar["cl_baseline_L_per_h"]
