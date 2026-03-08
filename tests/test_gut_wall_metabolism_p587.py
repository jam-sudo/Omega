"""Tests for Phase 587 - Drug Absorption Limited by Gut Wall Metabolism."""

import pytest

from omega_pbpk.core.gut_wall_metabolism_p587 import (
    GutWallMetabolismResult,
    compare_cyp3a4_fractions,
    predict_gut_wall_metabolism,
)


def _default_result(**kwargs):
    defaults = dict(
        drug_name="TestDrug",
        dose_mg=100.0,
        fa=0.9,
        cl_gut_intrinsic_L_per_h=50.0,
        cl_hepatic_intrinsic_L_per_h=30.0,
        fup=0.5,
    )
    defaults.update(kwargs)
    return predict_gut_wall_metabolism(**defaults)


class TestReturnType:
    def test_returns_gut_wall_metabolism_result(self):
        r = _default_result()
        assert isinstance(r, GutWallMetabolismResult)

    def test_drug_name(self):
        r = _default_result(drug_name="Midazolam")
        assert r.drug_name == "Midazolam"

    def test_dose_mg(self):
        r = _default_result(dose_mg=50.0)
        assert r.dose_mg == 50.0

    def test_fa_stored(self):
        r = _default_result(fa=0.85)
        assert r.fa == 0.85

    def test_all_fields_present(self):
        r = _default_result()
        fields = [
            "drug_name",
            "dose_mg",
            "fa",
            "fg",
            "fh",
            "f_oral",
            "e_gut",
            "e_hepatic",
            "cl_gut_total_L_per_h",
            "cl_hepatic_L_per_h",
            "cyp3a4_contribution_pct",
            "effect_of_grapefruit",
            "notes",
        ]
        for f in fields:
            assert hasattr(r, f), f"Missing field: {f}"


class TestOralBioavailability:
    def test_f_oral_equals_fa_times_fg_times_fh(self):
        r = _default_result()
        assert abs(r.f_oral - r.fa * r.fg * r.fh) < 1e-10

    def test_f_oral_less_than_fa(self):
        r = _default_result()
        assert r.f_oral <= r.fa

    def test_higher_cl_gut_lowers_fg(self):
        r_low = _default_result(cl_gut_intrinsic_L_per_h=10.0)
        r_high = _default_result(cl_gut_intrinsic_L_per_h=100.0)
        assert r_high.fg < r_low.fg

    def test_higher_cl_gut_lowers_f_oral(self):
        r_low = _default_result(cl_gut_intrinsic_L_per_h=10.0)
        r_high = _default_result(cl_gut_intrinsic_L_per_h=100.0)
        assert r_high.f_oral < r_low.f_oral


class TestExtractionRatios:
    def test_e_gut_in_range(self):
        r = _default_result()
        assert 0.0 <= r.e_gut <= 1.0

    def test_e_hepatic_in_range(self):
        r = _default_result()
        assert 0.0 <= r.e_hepatic <= 1.0

    def test_fg_plus_e_gut_equals_one(self):
        r = _default_result()
        assert abs(r.fg + r.e_gut - 1.0) < 1e-10

    def test_fh_plus_e_hepatic_equals_one(self):
        r = _default_result()
        assert abs(r.fh + r.e_hepatic - 1.0) < 1e-10


class TestClearance:
    def test_cl_gut_total_positive(self):
        r = _default_result()
        assert r.cl_gut_total_L_per_h > 0

    def test_cl_hepatic_positive(self):
        r = _default_result()
        assert r.cl_hepatic_L_per_h > 0


class TestCYP3A4:
    def test_cyp3a4_contribution_default(self):
        r = _default_result()
        assert r.cyp3a4_contribution_pct == 80.0

    def test_cyp3a4_contribution_custom(self):
        r = _default_result(cyp3a4_fraction_gut=0.5)
        assert r.cyp3a4_contribution_pct == 50.0

    def test_cyp3a4_contribution_in_range(self):
        r = _default_result()
        assert 0 <= r.cyp3a4_contribution_pct <= 100


class TestGrapefruitEffect:
    def test_grapefruit_increases_f_oral(self):
        r = _default_result(cyp3a4_fraction_gut=0.8)
        assert r.effect_of_grapefruit > r.f_oral

    def test_grapefruit_no_effect_zero_cyp3a4(self):
        r = _default_result(cyp3a4_fraction_gut=0.0)
        assert abs(r.effect_of_grapefruit - r.f_oral) < 1e-10


class TestCompareCYP3A4Fractions:
    def test_returns_list(self):
        results = compare_cyp3a4_fractions("TestDrug", 100.0, 0.9, 50.0, 30.0, 0.5)
        assert isinstance(results, list)

    def test_sorted_ascending_by_f_oral(self):
        results = compare_cyp3a4_fractions("TestDrug", 100.0, 0.9, 50.0, 30.0, 0.5)
        f_orals = [r.f_oral for r in results]
        assert f_orals == sorted(f_orals)

    def test_custom_fractions(self):
        results = compare_cyp3a4_fractions(
            "TestDrug",
            100.0,
            0.9,
            50.0,
            30.0,
            0.5,
            fractions=[0.1, 0.9],
        )
        assert len(results) == 2


class TestValidation:
    def test_fa_zero_raises(self):
        with pytest.raises(ValueError):
            _default_result(fa=0.0)

    def test_fa_negative_raises(self):
        with pytest.raises(ValueError):
            _default_result(fa=-0.1)

    def test_fa_above_one_raises(self):
        with pytest.raises(ValueError):
            _default_result(fa=1.1)

    def test_fup_zero_raises(self):
        with pytest.raises(ValueError):
            _default_result(fup=0.0)

    def test_fup_above_one_raises(self):
        with pytest.raises(ValueError):
            _default_result(fup=1.5)

    def test_cl_gut_negative_raises(self):
        with pytest.raises(ValueError):
            _default_result(cl_gut_intrinsic_L_per_h=-1.0)

    def test_cl_hepatic_zero_raises(self):
        with pytest.raises(ValueError):
            _default_result(cl_hepatic_intrinsic_L_per_h=0.0)

    def test_dose_zero_raises(self):
        with pytest.raises(ValueError):
            _default_result(dose_mg=0.0)


class TestNotes:
    def test_notes_contains_drug_name(self):
        r = _default_result(drug_name="Felodipine")
        assert "Felodipine" in r.notes
