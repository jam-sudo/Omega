"""Tests for Phase 1144: prediction/excipient_safety_screener.py"""

import math

import pytest

from omega_pbpk.prediction.excipient_safety_screener import (
    ExcipientSafetyResult,
    screen_excipient,
    screen_formulation,
)

# ------------------------------------------------------------------ helpers --


def pvp_result(level=100.0, bw=60.0, route="oral"):
    return screen_excipient("PVP", route, level, bw)


def mcc_result(level=200.0, bw=60.0, route="oral"):
    return screen_excipient("microcrystalline_cellulose", route, level, bw)


# -------------------------------------------------------- basic PVP tests -----


class TestPVP:
    def test_returns_result_type(self):
        r = pvp_result()
        assert isinstance(r, ExcipientSafetyResult)

    def test_gras_true(self):
        r = pvp_result()
        assert r.gras_status is True

    def test_echa_false(self):
        r = pvp_result()
        assert r.echa_concern is False

    def test_adi_is_50(self):
        r = pvp_result()
        assert r.adi_mg_per_kg == 50.0

    def test_100mg_per_60kg_adi_fraction_small(self):
        # 100 / 60 / 50 = 0.03333
        r = pvp_result(level=100.0, bw=60.0)
        assert r.adi_fraction is not None
        assert math.isclose(r.adi_fraction, 100.0 / 60.0 / 50.0, rel_tol=1e-9)

    def test_100mg_green_flag(self):
        r = pvp_result(level=100.0, bw=60.0)
        assert r.safety_flag == "green"

    def test_safety_concern_false_at_low_dose(self):
        r = pvp_result(level=100.0, bw=60.0)
        assert r.safety_concern is False

    def test_3500mg_adi_fraction_exceeds_1(self):
        # 3500 / 60 / 50 ≈ 1.167
        r = pvp_result(level=3500.0, bw=60.0)
        assert r.adi_fraction is not None
        assert r.adi_fraction > 1.0

    def test_3500mg_red_flag(self):
        r = pvp_result(level=3500.0, bw=60.0)
        assert r.safety_flag == "red"

    def test_3500mg_safety_concern_true(self):
        r = pvp_result(level=3500.0, bw=60.0)
        assert r.safety_concern is True

    def test_route_approved_oral(self):
        r = pvp_result(route="oral")
        assert r.route_approved is True

    def test_route_approved_iv(self):
        r = pvp_result(route="iv")
        assert r.route_approved is True

    def test_daily_intake_per_kg_computed(self):
        r = pvp_result(level=300.0, bw=60.0)
        assert math.isclose(r.daily_intake_mg_per_kg, 5.0)

    def test_yellow_flag_half_adi(self):
        # Need adi_fraction > 0.5 but safety_concern False
        # PVP adi=50; fraction > 0.5 means intake > 25 mg/kg → level > 25*60=1500
        # but < 50*60=3000 for no exceedance
        r = pvp_result(level=2000.0, bw=60.0)
        assert r.safety_flag == "yellow"


# ------------------------------------------------------ MCC (no ADI) tests ---


class TestMCC:
    def test_gras_true(self):
        r = mcc_result()
        assert r.gras_status is True

    def test_echa_false(self):
        r = mcc_result()
        assert r.echa_concern is False

    def test_adi_is_none(self):
        r = mcc_result()
        assert r.adi_mg_per_kg is None

    def test_adi_fraction_is_none(self):
        r = mcc_result()
        assert r.adi_fraction is None

    def test_green_flag(self):
        r = mcc_result()
        assert r.safety_flag == "green"

    def test_safety_concern_false(self):
        r = mcc_result()
        assert r.safety_concern is False

    def test_route_approved_oral(self):
        r = mcc_result(route="oral")
        assert r.route_approved is True

    def test_route_not_approved_iv(self):
        r = mcc_result(route="iv")
        assert r.route_approved is False


# ---------------------------------------------------- DMSO (ECHA) tests -----


class TestDMSO:
    def test_echa_concern_true(self):
        r = screen_excipient("DMSO", "topical", 50.0)
        assert r.echa_concern is True

    def test_red_flag_due_to_echa(self):
        r = screen_excipient("DMSO", "topical", 50.0)
        assert r.safety_flag == "red"

    def test_safety_concern_true(self):
        r = screen_excipient("DMSO", "topical", 50.0)
        assert r.safety_concern is True


# --------------------------------------------- screen_formulation tests -----


class TestScreenFormulation:
    def test_returns_list(self):
        excipients = [
            {"name": "microcrystalline_cellulose", "level_mg_per_day": 200.0},
            {"name": "PVP", "level_mg_per_day": 100.0},
        ]
        results = screen_formulation(excipients, "oral")
        assert isinstance(results, list)
        assert len(results) == 2

    def test_sorted_concerns_first(self):
        excipients = [
            {"name": "microcrystalline_cellulose", "level_mg_per_day": 200.0},
            {"name": "DMSO", "level_mg_per_day": 50.0},
            {"name": "PVP", "level_mg_per_day": 100.0},
        ]
        results = screen_formulation(excipients, "topical")
        # DMSO (red) should be first
        assert results[0].safety_flag == "red"

    def test_formulation_results_are_correct_type(self):
        excipients = [{"name": "mannitol", "level_mg_per_day": 500.0}]
        results = screen_formulation(excipients, "iv")
        assert isinstance(results[0], ExcipientSafetyResult)


# ----------------------------------------------- validation tests -----------


class TestValidation:
    def test_unknown_excipient_raises(self):
        with pytest.raises(ValueError, match="Unknown excipient"):
            screen_excipient("polyvinylXYZ", "oral", 100.0)

    def test_empty_route_raises(self):
        with pytest.raises(ValueError, match="route_of_administration"):
            screen_excipient("PVP", "", 100.0)

    def test_zero_level_raises(self):
        with pytest.raises(ValueError, match="level_mg_per_day"):
            screen_excipient("PVP", "oral", 0.0)

    def test_negative_level_raises(self):
        with pytest.raises(ValueError, match="level_mg_per_day"):
            screen_excipient("PVP", "oral", -10.0)

    def test_zero_body_weight_raises(self):
        with pytest.raises(ValueError, match="body_weight_kg"):
            screen_excipient("PVP", "oral", 100.0, body_weight_kg=0.0)

    def test_negative_body_weight_raises(self):
        with pytest.raises(ValueError, match="body_weight_kg"):
            screen_excipient("PVP", "oral", 100.0, body_weight_kg=-50.0)
