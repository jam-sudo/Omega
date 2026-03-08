"""Tests for Phase 323 — Tablet Disintegration Time Predictor."""

import pytest

from omega_pbpk.prediction.tablet_disintegration_predictor import (
    TabletDisintegrationResult,
    predict_disintegration_time,
    screen_disintegrant_levels,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _default(**kwargs):
    params = dict(
        drug_name="TestDrug",
        tablet_weight_mg=200.0,
        compression_force_kN=5.0,
        disintegrant_pct=5.0,
        disintegrant_type="CCS",
        binder_pct=10.0,
        binder_type="MCC",
        porosity_pct=20.0,
    )
    params.update(kwargs)
    return predict_disintegration_time(**params)


# ---------------------------------------------------------------------------
# Return type and field preservation
# ---------------------------------------------------------------------------


class TestReturnType:
    def test_returns_correct_type(self):
        r = _default()
        assert isinstance(r, TabletDisintegrationResult)

    def test_drug_name_preserved(self):
        r = _default(drug_name="Ibuprofen")
        assert r.drug_name == "Ibuprofen"

    def test_tablet_weight_preserved(self):
        r = _default(tablet_weight_mg=500.0)
        assert r.tablet_weight_mg == 500.0

    def test_compression_force_preserved(self):
        r = _default(compression_force_kN=8.0)
        assert r.compression_force_kN == 8.0

    def test_disintegrant_pct_preserved(self):
        r = _default(disintegrant_pct=7.0)
        assert r.disintegrant_pct == 7.0

    def test_disintegrant_type_preserved(self):
        r = _default(disintegrant_type="SSG")
        assert r.disintegrant_type == "SSG"

    def test_binder_pct_preserved(self):
        r = _default(binder_pct=15.0)
        assert r.binder_pct == 15.0

    def test_binder_type_preserved(self):
        r = _default(binder_type="PVP")
        assert r.binder_type == "PVP"

    def test_porosity_pct_preserved(self):
        r = _default(porosity_pct=30.0)
        assert r.porosity_pct == 30.0


# ---------------------------------------------------------------------------
# Numerical bounds
# ---------------------------------------------------------------------------


class TestNumericalBounds:
    def test_disintegration_time_at_least_0_1(self):
        r = _default()
        assert r.disintegration_time_min >= 0.1

    def test_disintegration_time_at_most_120(self):
        r = _default()
        assert r.disintegration_time_min <= 120.0

    def test_water_uptake_positive(self):
        r = _default()
        assert r.water_uptake_rate_mg_per_min > 0


# ---------------------------------------------------------------------------
# Physics
# ---------------------------------------------------------------------------


class TestPhysics:
    def test_higher_compression_longer_disintegration(self):
        low_cf = _default(compression_force_kN=2.0)
        high_cf = _default(compression_force_kN=15.0)
        assert high_cf.disintegration_time_min > low_cf.disintegration_time_min

    def test_higher_disintegrant_shorter_time(self):
        low_d = _default(disintegrant_pct=2.0)
        high_d = _default(disintegrant_pct=15.0)
        assert high_d.disintegration_time_min < low_d.disintegration_time_min

    def test_ccs_faster_than_starch(self):
        ccs = _default(disintegrant_type="CCS")
        starch = _default(disintegrant_type="starch")
        assert ccs.disintegration_time_min < starch.disintegration_time_min

    def test_hpmc_longer_than_mcc(self):
        hpmc = _default(binder_type="HPMC")
        mcc = _default(binder_type="MCC")
        assert hpmc.disintegration_time_min > mcc.disintegration_time_min

    def test_higher_porosity_faster_disintegration(self):
        low_p = _default(porosity_pct=5.0)
        high_p = _default(porosity_pct=50.0)
        assert high_p.disintegration_time_min < low_p.disintegration_time_min

    def test_water_uptake_scales_with_weight(self):
        r1 = _default(tablet_weight_mg=100.0, porosity_pct=20.0)
        r2 = _default(tablet_weight_mg=200.0, porosity_pct=20.0)
        assert abs(r2.water_uptake_rate_mg_per_min - 2 * r1.water_uptake_rate_mg_per_min) < 1e-9


# ---------------------------------------------------------------------------
# USP standard and classification
# ---------------------------------------------------------------------------


class TestClassification:
    def test_fast_tablet_meets_usp(self):
        # Very high disintegrant, low compression, high porosity → fast
        r = _default(
            disintegrant_pct=20.0,
            compression_force_kN=1.0,
            binder_type="MCC",
            porosity_pct=50.0,
        )
        assert r.meets_usp_standard is True

    def test_slow_tablet_fails_usp(self):
        # starch (efficacy 0.4) + HPMC (resistance 1.5) + high compression + very low dis%
        r = predict_disintegration_time(
            drug_name="TestDrug",
            tablet_weight_mg=200.0,
            compression_force_kN=20.0,
            disintegrant_pct=1.0,
            disintegrant_type="starch",
            binder_pct=25.0,
            binder_type="HPMC",
            porosity_pct=2.0,
        )
        assert r.meets_usp_standard is False

    def test_disintegration_class_rapid(self):
        r = _default(
            disintegrant_pct=20.0,
            compression_force_kN=1.0,
            binder_type="MCC",
            porosity_pct=50.0,
        )
        assert r.disintegration_class == "rapid"

    def test_disintegration_class_slow(self):
        # starch + HPMC + high compression + 1% disintegrant → > 15 min
        r = predict_disintegration_time(
            drug_name="TestDrug",
            tablet_weight_mg=200.0,
            compression_force_kN=20.0,
            disintegrant_pct=1.0,
            disintegrant_type="starch",
            binder_pct=25.0,
            binder_type="HPMC",
            porosity_pct=2.0,
        )
        assert r.disintegration_class == "slow"


# ---------------------------------------------------------------------------
# screen_disintegrant_levels
# ---------------------------------------------------------------------------


class TestScreen:
    def test_returns_list(self):
        result = screen_disintegrant_levels(
            drug_name="Drug",
            tablet_weight_mg=200.0,
            compression_force_kN=5.0,
            binder_pct=10.0,
            disintegrant_type="CCS",
        )
        assert isinstance(result, list)

    def test_sorted_ascending(self):
        result = screen_disintegrant_levels(
            drug_name="Drug",
            tablet_weight_mg=200.0,
            compression_force_kN=5.0,
            binder_pct=10.0,
            disintegrant_type="CCS",
        )
        times = [r.disintegration_time_min for r in result]
        assert times == sorted(times)

    def test_all_elements_are_results(self):
        results = screen_disintegrant_levels(
            drug_name="Drug",
            tablet_weight_mg=200.0,
            compression_force_kN=5.0,
            binder_pct=10.0,
            disintegrant_type="SSG",
        )
        for r in results:
            assert isinstance(r, TabletDisintegrationResult)

    def test_non_empty(self):
        results = screen_disintegrant_levels(
            drug_name="Drug",
            tablet_weight_mg=200.0,
            compression_force_kN=5.0,
            binder_pct=10.0,
            disintegrant_type="starch",
        )
        assert len(results) > 0


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


class TestValidation:
    def test_negative_tablet_weight(self):
        with pytest.raises(ValueError):
            _default(tablet_weight_mg=-100.0)

    def test_zero_compression_force(self):
        with pytest.raises(ValueError):
            _default(compression_force_kN=0.0)

    def test_disintegrant_pct_zero(self):
        with pytest.raises(ValueError):
            _default(disintegrant_pct=0.0)

    def test_disintegrant_pct_too_high(self):
        with pytest.raises(ValueError):
            _default(disintegrant_pct=21.0)

    def test_binder_pct_zero(self):
        with pytest.raises(ValueError):
            _default(binder_pct=0.0)

    def test_binder_pct_too_high(self):
        with pytest.raises(ValueError):
            _default(binder_pct=31.0)

    def test_porosity_zero(self):
        with pytest.raises(ValueError):
            _default(porosity_pct=0.0)

    def test_porosity_too_high(self):
        with pytest.raises(ValueError):
            _default(porosity_pct=60.0)

    def test_invalid_disintegrant_type(self):
        with pytest.raises(ValueError):
            _default(disintegrant_type="UnknownDis")

    def test_invalid_binder_type(self):
        with pytest.raises(ValueError):
            _default(binder_type="UnknownBinder")
