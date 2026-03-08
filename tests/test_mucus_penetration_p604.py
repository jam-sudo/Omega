"""Tests for Phase 604: Drug mucus penetration prediction."""

import pytest

from omega_pbpk.prediction.mucus_penetration_p604 import (
    MucusPenetrationResult,
    compare_mucus_types,
    predict_mucus_penetration,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _default(**kwargs):
    params = dict(
        drug_name="TestDrug",
        mw_Da=300.0,
        charge="neutral",
        mucus_type="gi",
        logP=0.0,
        peg_coated=False,
    )
    params.update(kwargs)
    return predict_mucus_penetration(**params)


# ---------------------------------------------------------------------------
# Return type & frozen dataclass
# ---------------------------------------------------------------------------


class TestReturnType:
    def test_returns_correct_type(self):
        res = _default()
        assert isinstance(res, MucusPenetrationResult)

    def test_result_is_frozen(self):
        res = _default()
        with pytest.raises((AttributeError, TypeError)):
            res.d_ratio = 999.0  # type: ignore[misc]

    def test_drug_name_preserved(self):
        res = _default(drug_name="Ibuprofen")
        assert res.drug_name == "Ibuprofen"

    def test_mucus_type_preserved(self):
        res = _default(mucus_type="pulmonary")
        assert res.mucus_type == "pulmonary"

    def test_charge_preserved(self):
        res = _default(charge="anionic")
        assert res.charge == "anionic"

    def test_mw_preserved(self):
        res = _default(mw_Da=500.0)
        assert res.mw_Da == pytest.approx(500.0)


# ---------------------------------------------------------------------------
# Diffusivity values
# ---------------------------------------------------------------------------


class TestDiffusivity:
    def test_d_free_positive(self):
        res = _default()
        assert res.d_free_cm2_s > 0.0

    def test_d_mucus_positive(self):
        res = _default()
        assert res.d_mucus_cm2_s > 0.0

    def test_d_mucus_leq_d_free(self):
        res = _default()
        assert res.d_mucus_cm2_s <= res.d_free_cm2_s + 1e-20

    def test_d_ratio_between_0_and_1(self):
        # PEG can push d_mucus close to d_free but not exceed it by definition
        res = _default()
        assert 0.0 < res.d_ratio <= 1.0 + 1e-9

    def test_d_ratio_equals_d_mucus_over_d_free(self):
        res = _default()
        assert res.d_ratio == pytest.approx(res.d_mucus_cm2_s / res.d_free_cm2_s)

    def test_flux_relative_equals_d_ratio(self):
        res = _default()
        assert res.flux_relative == pytest.approx(res.d_ratio)


# ---------------------------------------------------------------------------
# Charge effect
# ---------------------------------------------------------------------------


class TestChargeEffect:
    def test_cationic_lower_d_ratio_than_neutral(self):
        r_cat = _default(charge="cationic")
        r_neu = _default(charge="neutral")
        assert r_cat.d_ratio < r_neu.d_ratio

    def test_anionic_similar_to_neutral(self):
        r_an = _default(charge="anionic")
        r_neu = _default(charge="neutral")
        assert r_an.d_ratio == pytest.approx(r_neu.d_ratio)

    def test_zwitterionic_between_cationic_and_neutral(self):
        r_cat = _default(charge="cationic")
        r_zwit = _default(charge="zwitterionic")
        r_neu = _default(charge="neutral")
        assert r_cat.d_ratio < r_zwit.d_ratio < r_neu.d_ratio + 1e-9

    def test_cationic_high_mucoadhesion_risk(self):
        res = _default(charge="cationic")
        assert res.mucoadhesion_risk == "high"

    def test_neutral_low_mucoadhesion_risk(self):
        res = _default(charge="neutral")
        assert res.mucoadhesion_risk == "low"

    def test_anionic_low_mucoadhesion_risk(self):
        res = _default(charge="anionic")
        assert res.mucoadhesion_risk == "low"


# ---------------------------------------------------------------------------
# MW effect
# ---------------------------------------------------------------------------


class TestMWEffect:
    def test_higher_mw_lower_d_ratio(self):
        r_small = _default(mw_Da=200.0)
        r_large = _default(mw_Da=800.0)
        assert r_small.d_ratio > r_large.d_ratio

    def test_higher_mw_lower_d_free(self):
        r_small = _default(mw_Da=100.0)
        r_large = _default(mw_Da=500.0)
        # Larger MW → larger radius → lower D_free
        assert r_small.d_free_cm2_s > r_large.d_free_cm2_s

    def test_higher_mw_longer_permeation_time(self):
        r_small = _default(mw_Da=150.0)
        r_large = _default(mw_Da=1000.0)
        assert r_large.permeation_time_s > r_small.permeation_time_s


# ---------------------------------------------------------------------------
# PEG coating effect
# ---------------------------------------------------------------------------


class TestPEGCoating:
    def test_peg_increases_d_mucus(self):
        r_no_peg = _default(peg_coated=False)
        r_peg = _default(peg_coated=True)
        assert r_peg.d_mucus_cm2_s >= r_no_peg.d_mucus_cm2_s

    def test_peg_increases_d_ratio(self):
        r_no_peg = _default(peg_coated=False)
        r_peg = _default(peg_coated=True)
        assert r_peg.d_ratio >= r_no_peg.d_ratio

    def test_peg_shortens_permeation_time(self):
        r_no_peg = _default(peg_coated=False)
        r_peg = _default(peg_coated=True)
        assert r_peg.permeation_time_s <= r_no_peg.permeation_time_s


# ---------------------------------------------------------------------------
# logP effect
# ---------------------------------------------------------------------------


class TestLogPEffect:
    def test_high_logP_reduces_d_ratio(self):
        r_low = _default(logP=0.0)
        r_high = _default(logP=5.0)
        assert r_low.d_ratio >= r_high.d_ratio

    def test_logP_below_1_no_penalty(self):
        r0 = _default(logP=0.0)
        r1 = _default(logP=0.9)
        # Both should have the same logP_factor (1.0)
        assert r0.d_ratio == pytest.approx(r1.d_ratio, rel=0.01)


# ---------------------------------------------------------------------------
# Mucus types
# ---------------------------------------------------------------------------


class TestMucusTypes:
    def test_gi_thickness(self):
        res = _default(mucus_type="gi")
        assert res.mucus_thickness_um == pytest.approx(300.0)

    def test_pulmonary_thickness(self):
        res = _default(mucus_type="pulmonary")
        assert res.mucus_thickness_um == pytest.approx(10.0)

    def test_cervical_thickness(self):
        res = _default(mucus_type="cervical")
        assert res.mucus_thickness_um == pytest.approx(100.0)

    def test_gi_slowest_permeation(self):
        """GI mucus is thickest → longest permeation time (same diffusivity)."""
        r_gi = _default(mucus_type="gi")
        r_pulm = _default(mucus_type="pulmonary")
        assert r_gi.permeation_time_s > r_pulm.permeation_time_s

    def test_pulmonary_fastest_permeation(self):
        r_pulm = _default(mucus_type="pulmonary")
        r_cerv = _default(mucus_type="cervical")
        assert r_pulm.permeation_time_s < r_cerv.permeation_time_s


# ---------------------------------------------------------------------------
# Penetration class
# ---------------------------------------------------------------------------


class TestPenetrationClass:
    def test_rapid_class_for_small_thin(self):
        # Small MW, pulmonary (thin) → rapid
        res = predict_mucus_penetration(
            "SmallDrug",
            mw_Da=100.0,
            charge="neutral",
            mucus_type="pulmonary",
            logP=0.0,
            peg_coated=True,
        )
        assert res.penetration_class == "rapid"

    def test_slow_class_for_large_gi(self):
        # Large MW, GI (thick) → slow
        res = predict_mucus_penetration(
            "BigDrug", mw_Da=1500.0, charge="cationic", mucus_type="gi", logP=0.0
        )
        assert res.penetration_class == "slow"

    def test_penetration_class_in_valid_set(self):
        for mt in ("gi", "pulmonary", "cervical"):
            res = _default(mucus_type=mt)
            assert res.penetration_class in ("rapid", "moderate", "slow")


# ---------------------------------------------------------------------------
# compare_mucus_types
# ---------------------------------------------------------------------------


class TestCompareMucusTypes:
    def test_returns_list(self):
        result = compare_mucus_types("TestDrug", mw_Da=300.0, charge="neutral")
        assert isinstance(result, list)

    def test_returns_three_results(self):
        result = compare_mucus_types("TestDrug", mw_Da=300.0, charge="neutral")
        assert len(result) == 3

    def test_all_results_are_correct_type(self):
        result = compare_mucus_types("TestDrug", mw_Da=300.0, charge="neutral")
        for r in result:
            assert isinstance(r, MucusPenetrationResult)

    def test_sorted_by_d_ratio_descending(self):
        result = compare_mucus_types("TestDrug", mw_Da=300.0, charge="neutral")
        for i in range(1, len(result)):
            assert result[i - 1].d_ratio >= result[i].d_ratio

    def test_all_three_mucus_types_present(self):
        result = compare_mucus_types("TestDrug", mw_Da=300.0, charge="neutral")
        types_found = {r.mucus_type for r in result}
        assert types_found == {"gi", "pulmonary", "cervical"}


# ---------------------------------------------------------------------------
# Validation / error cases
# ---------------------------------------------------------------------------


class TestValidation:
    def test_zero_mw_raises(self):
        with pytest.raises(ValueError, match="mw_Da"):
            _default(mw_Da=0.0)

    def test_negative_mw_raises(self):
        with pytest.raises(ValueError, match="mw_Da"):
            _default(mw_Da=-100.0)

    def test_invalid_mucus_type_raises(self):
        with pytest.raises(ValueError, match="mucus_type"):
            _default(mucus_type="nasal")

    def test_invalid_charge_raises(self):
        with pytest.raises(ValueError, match="charge"):
            _default(charge="positive")

    def test_notes_not_empty(self):
        res = _default()
        assert isinstance(res.notes, str)
        assert len(res.notes) > 0
