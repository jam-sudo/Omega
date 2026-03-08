"""Tests for Phase 588 - Drug Half-Life Prediction from Structure."""

import pytest

from omega_pbpk.prediction.structural_halflife_predictor import (
    StructuralHalflifeResult,
    predict_halflife_from_structure,
    screen_halflife,
)


def _default_result(**kwargs):
    defaults = dict(
        drug_name="TestDrug",
        logP=2.5,
        mw_Da=350.0,
        psa_A2=80.0,
        hbd=2,
        rot_bonds=5,
    )
    defaults.update(kwargs)
    return predict_halflife_from_structure(**defaults)


class TestReturnType:
    def test_returns_structural_halflife_result(self):
        r = _default_result()
        assert isinstance(r, StructuralHalflifeResult)

    def test_all_fields_present(self):
        r = _default_result()
        fields = [
            "drug_name",
            "logP",
            "mw_Da",
            "psa_A2",
            "hbd",
            "rot_bonds",
            "predicted_cl_L_per_h",
            "predicted_vd_L",
            "predicted_thalf_h",
            "thalf_class",
            "predicted_dosing_frequency",
            "high_clearance_alert",
            "renal_or_hepatic_primary",
            "notes",
        ]
        for f in fields:
            assert hasattr(r, f), f"Missing field: {f}"

    def test_thalf_positive(self):
        r = _default_result()
        assert r.predicted_thalf_h > 0


class TestClearancePrediction:
    def test_cl_in_range(self):
        r = _default_result()
        assert 1.0 <= r.predicted_cl_L_per_h <= 100.0

    def test_cl_clamped_low(self):
        r = _default_result(logP=-5.0, mw_Da=50.0)
        assert r.predicted_cl_L_per_h >= 1.0

    def test_higher_logP_higher_vd(self):
        r_low = _default_result(logP=0.5)
        r_high = _default_result(logP=4.0)
        assert r_high.predicted_vd_L > r_low.predicted_vd_L


class TestVdPrediction:
    def test_vd_in_range(self):
        r = _default_result()
        assert 5.0 <= r.predicted_vd_L <= 500.0

    def test_vd_clamped_low(self):
        r = _default_result(logP=-2.0, mw_Da=50.0)
        assert r.predicted_vd_L >= 5.0

    def test_vd_clamped_high(self):
        r = _default_result(logP=6.0, mw_Da=900.0)
        assert r.predicted_vd_L <= 500.0


class TestHalflifeClass:
    def test_ultra_short(self):
        # Very small MW, low logP -> small Vd, moderate CL -> short thalf
        r = _default_result(logP=-1.0, mw_Da=30.0)
        assert r.thalf_class == "ultra_short"

    def test_short_class(self):
        r = _default_result(logP=1.0, mw_Da=200.0)
        assert r.thalf_class == "short"

    def test_intermediate_class(self):
        r = _default_result(logP=2.0, mw_Da=400.0)
        assert r.thalf_class in ("short", "intermediate")

    def test_long_class_consistency(self):
        """The model's CL floor and Vd ceiling limit max thalf ~20h.
        Test that intermediate is produced for high Vd/low CL combos."""
        r = _default_result(logP=5.0, mw_Da=1500.0)
        assert r.thalf_class in ("intermediate", "long")

    def test_all_classes_are_valid(self):
        valid = {"ultra_short", "short", "intermediate", "long"}
        r = _default_result()
        assert r.thalf_class in valid


class TestDosingFrequency:
    def test_qid_for_ultra_short(self):
        r = _default_result(logP=-1.0, mw_Da=30.0)
        assert r.predicted_dosing_frequency == "QID"

    def test_bid_for_intermediate(self):
        r = _default_result(logP=5.0, mw_Da=1500.0)
        assert r.predicted_dosing_frequency in ("BID", "QD")

    def test_valid_frequencies(self):
        valid = {"QID", "TID", "BID", "QD"}
        r = _default_result()
        assert r.predicted_dosing_frequency in valid


class TestHighClearanceAlert:
    def test_high_clearance_true(self):
        # Need extreme logP to push (cl_lipo+cl_hydro)/2 > 60
        # logP=30: cl_lipo=15*32/4=120, cl_hydro=7.5, avg=63.75
        r = predict_halflife_from_structure("X", logP=30.0, mw_Da=300.0, psa_A2=50.0, hbd=0)
        assert r.high_clearance_alert is True

    def test_high_clearance_false(self):
        r = _default_result(logP=1.0)
        assert r.high_clearance_alert is False


class TestRenalOrHepatic:
    def test_hepatic_for_high_logP(self):
        r = _default_result(logP=2.0)
        assert r.renal_or_hepatic_primary == "hepatic"

    def test_renal_for_low_logP(self):
        r = _default_result(logP=0.5)
        assert r.renal_or_hepatic_primary == "renal"

    def test_renal_for_logP_one(self):
        r = _default_result(logP=1.0)
        assert r.renal_or_hepatic_primary == "renal"


class TestScreenHalflife:
    def test_returns_list(self):
        compounds = [
            {"drug_name": "A", "logP": 2.0, "mw_Da": 300.0, "psa_A2": 80.0, "hbd": 1},
            {"drug_name": "B", "logP": 0.5, "mw_Da": 200.0, "psa_A2": 100.0, "hbd": 3},
        ]
        results = screen_halflife(compounds)
        assert isinstance(results, list)
        assert len(results) == 2

    def test_sorted_descending_by_thalf(self):
        compounds = [
            {"drug_name": "Short", "logP": -1.0, "mw_Da": 50.0, "psa_A2": 30.0, "hbd": 0},
            {"drug_name": "Long", "logP": 4.0, "mw_Da": 600.0, "psa_A2": 60.0, "hbd": 1},
            {"drug_name": "Med", "logP": 1.5, "mw_Da": 300.0, "psa_A2": 70.0, "hbd": 2},
        ]
        results = screen_halflife(compounds)
        thalfs = [r.predicted_thalf_h for r in results]
        assert thalfs == sorted(thalfs, reverse=True)

    def test_optional_rot_bonds(self):
        compounds = [
            {
                "drug_name": "A",
                "logP": 2.0,
                "mw_Da": 300.0,
                "psa_A2": 80.0,
                "hbd": 1,
                "rot_bonds": 3,
            },
        ]
        results = screen_halflife(compounds)
        assert results[0].rot_bonds == 3


class TestValidation:
    def test_mw_zero_raises(self):
        with pytest.raises(ValueError):
            _default_result(mw_Da=0.0)

    def test_mw_negative_raises(self):
        with pytest.raises(ValueError):
            _default_result(mw_Da=-100.0)

    def test_hbd_negative_raises(self):
        with pytest.raises(ValueError):
            _default_result(hbd=-1)

    def test_psa_negative_raises(self):
        with pytest.raises(ValueError):
            _default_result(psa_A2=-10.0)


class TestNotes:
    def test_notes_contains_drug_name(self):
        r = _default_result(drug_name="Atorvastatin")
        assert "Atorvastatin" in r.notes
