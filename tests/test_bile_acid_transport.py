"""Tests for Phase 413 — bile_acid_transport module."""

from __future__ import annotations

import pytest

from omega_pbpk.core.bile_acid_transport import (
    BileAcidTransportResult,
    simulate_bile_acid_transport,
    transporter_inhibition_effect,
)

# ---------------------------------------------------------------------------
# Shared default parameters
# ---------------------------------------------------------------------------

BASE = dict(
    drug_name="TestDrug",
    dose_mg=100.0,
    cl_hep_L_per_h=1.0,
    vd_L=50.0,
    km_OATP_mg_L=0.5,
    vmax_OATP_mg_h=5.0,
    km_NTCP_mg_L=0.5,
    vmax_NTCP_mg_h=3.0,
    bile_flow_L_per_h=0.9,
    t_end_h=24.0,
    dt_h=0.1,
)


# ---------------------------------------------------------------------------
# Dataclass structure
# ---------------------------------------------------------------------------


class TestBileAcidTransportResultStructure:
    def test_result_is_dataclass_instance(self):
        res = simulate_bile_acid_transport(**BASE)
        assert isinstance(res, BileAcidTransportResult)

    def test_required_fields_present(self):
        res = simulate_bile_acid_transport(**BASE)
        for field in (
            "drug_name",
            "dose_mg",
            "km_OATP_mg_L",
            "vmax_OATP_mg_h",
            "times_h",
            "c_portal_mg_L",
            "c_hep_mg_L",
            "a_bile_mg",
            "a_intestine_mg",
            "auc_portal",
            "n_cycles_estimated",
            "notes",
        ):
            assert hasattr(res, field), f"Missing field: {field}"

    def test_list_fields_are_lists(self):
        res = simulate_bile_acid_transport(**BASE)
        assert isinstance(res.times_h, list)
        assert isinstance(res.c_portal_mg_L, list)
        assert isinstance(res.c_hep_mg_L, list)
        assert isinstance(res.a_bile_mg, list)
        assert isinstance(res.a_intestine_mg, list)

    def test_list_lengths_match(self):
        res = simulate_bile_acid_transport(**BASE)
        n = len(res.times_h)
        assert len(res.c_portal_mg_L) == n
        assert len(res.c_hep_mg_L) == n
        assert len(res.a_bile_mg) == n
        assert len(res.a_intestine_mg) == n

    def test_drug_name_stored(self):
        res = simulate_bile_acid_transport(**BASE)
        assert res.drug_name == "TestDrug"

    def test_dose_mg_stored(self):
        res = simulate_bile_acid_transport(**BASE)
        assert res.dose_mg == BASE["dose_mg"]


# ---------------------------------------------------------------------------
# Initial conditions
# ---------------------------------------------------------------------------


class TestInitialConditions:
    def test_portal_initial_concentration(self):
        # C_portal[0] = dose_mg / 5 (portal volume = 5 L)
        res = simulate_bile_acid_transport(**BASE)
        expected = BASE["dose_mg"] / 5.0
        assert abs(res.c_portal_mg_L[0] - expected) < 1e-9

    def test_hep_initial_concentration_zero(self):
        res = simulate_bile_acid_transport(**BASE)
        assert res.c_hep_mg_L[0] == 0.0

    def test_bile_initial_amount_zero(self):
        res = simulate_bile_acid_transport(**BASE)
        assert res.a_bile_mg[0] == 0.0

    def test_intestine_initial_amount_zero(self):
        res = simulate_bile_acid_transport(**BASE)
        assert res.a_intestine_mg[0] == 0.0

    def test_time_starts_at_zero(self):
        res = simulate_bile_acid_transport(**BASE)
        assert res.times_h[0] == 0.0


# ---------------------------------------------------------------------------
# Physical constraints
# ---------------------------------------------------------------------------


class TestPhysicalConstraints:
    def test_portal_concentration_nonnegative(self):
        res = simulate_bile_acid_transport(**BASE)
        assert all(c >= 0.0 for c in res.c_portal_mg_L)

    def test_hep_concentration_nonnegative(self):
        res = simulate_bile_acid_transport(**BASE)
        assert all(c >= 0.0 for c in res.c_hep_mg_L)

    def test_bile_amount_nonnegative(self):
        res = simulate_bile_acid_transport(**BASE)
        assert all(a >= 0.0 for a in res.a_bile_mg)

    def test_intestine_amount_nonnegative(self):
        res = simulate_bile_acid_transport(**BASE)
        assert all(a >= 0.0 for a in res.a_intestine_mg)

    def test_auc_portal_positive(self):
        res = simulate_bile_acid_transport(**BASE)
        assert res.auc_portal > 0.0

    def test_n_cycles_nonnegative(self):
        res = simulate_bile_acid_transport(**BASE)
        assert res.n_cycles_estimated >= 0.0


# ---------------------------------------------------------------------------
# Dose-response behaviour
# ---------------------------------------------------------------------------


class TestDoseResponse:
    def test_higher_dose_higher_portal_auc(self):
        res_low = simulate_bile_acid_transport(**{**BASE, "dose_mg": 50.0})
        res_high = simulate_bile_acid_transport(**{**BASE, "dose_mg": 200.0})
        assert res_high.auc_portal > res_low.auc_portal

    def test_higher_vmax_oatp_lowers_portal_auc(self):
        """More efficient hepatic uptake reduces portal exposure."""
        res_slow = simulate_bile_acid_transport(**{**BASE, "vmax_OATP_mg_h": 1.0})
        res_fast = simulate_bile_acid_transport(**{**BASE, "vmax_OATP_mg_h": 20.0})
        assert res_fast.auc_portal < res_slow.auc_portal

    def test_zero_oatp_keeps_drug_in_portal(self):
        """No OATP uptake: drug stays in portal vein, no biliary cycling."""
        res = simulate_bile_acid_transport(**{**BASE, "vmax_OATP_mg_h": 0.0})
        # Bile should remain near zero
        assert max(res.a_bile_mg) < 0.01

    def test_zero_ntcp_no_biliary_excretion(self):
        """No NTCP: drug enters hepatocyte but cannot exit via bile."""
        res = simulate_bile_acid_transport(**{**BASE, "vmax_NTCP_mg_h": 0.0})
        assert max(res.a_bile_mg) < 1e-6


# ---------------------------------------------------------------------------
# Simulation length and resolution
# ---------------------------------------------------------------------------


class TestSimulationParameters:
    def test_longer_simulation_more_time_points(self):
        res_short = simulate_bile_acid_transport(**{**BASE, "t_end_h": 12.0})
        res_long = simulate_bile_acid_transport(**{**BASE, "t_end_h": 48.0})
        assert len(res_long.times_h) > len(res_short.times_h)

    def test_finer_dt_more_time_points(self):
        res_coarse = simulate_bile_acid_transport(**{**BASE, "dt_h": 0.5})
        res_fine = simulate_bile_acid_transport(**{**BASE, "dt_h": 0.05})
        assert len(res_fine.times_h) > len(res_coarse.times_h)

    def test_notes_field_is_string(self):
        res = simulate_bile_acid_transport(**BASE)
        assert isinstance(res.notes, str)
        assert len(res.notes) > 0


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


class TestInputValidation:
    def test_empty_drug_name_raises(self):
        with pytest.raises(ValueError, match="drug_name"):
            simulate_bile_acid_transport(**{**BASE, "drug_name": ""})

    def test_zero_dose_raises(self):
        with pytest.raises(ValueError, match="dose_mg"):
            simulate_bile_acid_transport(**{**BASE, "dose_mg": 0.0})

    def test_negative_dose_raises(self):
        with pytest.raises(ValueError, match="dose_mg"):
            simulate_bile_acid_transport(**{**BASE, "dose_mg": -10.0})

    def test_zero_km_oatp_raises(self):
        with pytest.raises(ValueError, match="km_OATP"):
            simulate_bile_acid_transport(**{**BASE, "km_OATP_mg_L": 0.0})

    def test_zero_km_ntcp_raises(self):
        with pytest.raises(ValueError, match="km_NTCP"):
            simulate_bile_acid_transport(**{**BASE, "km_NTCP_mg_L": 0.0})

    def test_negative_vmax_oatp_raises(self):
        with pytest.raises(ValueError, match="vmax_OATP"):
            simulate_bile_acid_transport(**{**BASE, "vmax_OATP_mg_h": -1.0})

    def test_zero_t_end_raises(self):
        with pytest.raises(ValueError, match="t_end_h"):
            simulate_bile_acid_transport(**{**BASE, "t_end_h": 0.0})

    def test_zero_dt_raises(self):
        with pytest.raises(ValueError, match="dt_h"):
            simulate_bile_acid_transport(**{**BASE, "dt_h": 0.0})

    def test_negative_vd_raises(self):
        with pytest.raises(ValueError, match="vd_L"):
            simulate_bile_acid_transport(**{**BASE, "vd_L": -5.0})


# ---------------------------------------------------------------------------
# Transporter inhibition effect
# ---------------------------------------------------------------------------


class TestTransporterInhibitionEffect:
    def test_returns_dict(self):
        result = transporter_inhibition_effect("TestDrug", 100.0, 50.0, 0.0)
        assert isinstance(result, dict)

    def test_expected_keys_present(self):
        result = transporter_inhibition_effect("TestDrug", 100.0, 50.0, 0.0)
        for key in (
            "auc_portal_normal",
            "auc_portal_inhibited",
            "auc_ratio",
            "oatp_inhibition_pct",
            "ntcp_inhibition_pct",
            "interpretation",
        ):
            assert key in result, f"Missing key: {key}"

    def test_oatp_inhibition_raises_portal_auc(self):
        """OATP inhibition should increase portal AUC (reduced hepatic uptake)."""
        result = transporter_inhibition_effect("TestDrug", 100.0, 80.0, 0.0)
        assert result["auc_portal_inhibited"] > result["auc_portal_normal"]

    def test_zero_inhibition_auc_ratio_near_one(self):
        result = transporter_inhibition_effect("TestDrug", 100.0, 0.0, 0.0)
        assert abs(result["auc_ratio"] - 1.0) < 1e-6

    def test_invalid_drug_name_raises(self):
        with pytest.raises(ValueError):
            transporter_inhibition_effect("", 100.0, 50.0, 0.0)

    def test_invalid_dose_raises(self):
        with pytest.raises(ValueError):
            transporter_inhibition_effect("Drug", 0.0, 50.0, 0.0)

    def test_inhibition_above_100_raises(self):
        with pytest.raises(ValueError):
            transporter_inhibition_effect("Drug", 100.0, 110.0, 0.0)

    def test_inhibition_below_0_raises(self):
        with pytest.raises(ValueError):
            transporter_inhibition_effect("Drug", 100.0, -5.0, 0.0)

    def test_interpretation_is_string(self):
        result = transporter_inhibition_effect("TestDrug", 100.0, 50.0, 50.0)
        assert isinstance(result["interpretation"], str)
        assert len(result["interpretation"]) > 0
