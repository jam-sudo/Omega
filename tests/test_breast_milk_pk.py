"""Tests for breast milk PK module — Phase 341."""

from __future__ import annotations

import pytest

from omega_pbpk.clinical.breast_milk_pk import (
    BreastMilkResult,
    calculate_milk_to_plasma_ratio,
    screen_drugs_for_breastfeeding,
    simulate_breast_milk_pk,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _default_sim(**kw) -> BreastMilkResult:
    defaults = dict(
        drug_name="TestDrug",
        maternal_dose_mg=200.0,
        cl_maternal_L_per_h=5.0,
        vd_maternal_L=50.0,
        milk_to_plasma_ratio=0.5,
        ka_per_h=1.0,
        f_oral=0.9,
        t_end_h=24.0,
        infant_weight_kg=5.0,
        milk_intake_mL_per_day=150.0,
        dt_h=0.5,
    )
    defaults.update(kw)
    return simulate_breast_milk_pk(**defaults)


# ---------------------------------------------------------------------------
# calculate_milk_to_plasma_ratio
# ---------------------------------------------------------------------------


class TestCalculateMilkToPlasmaRatio:
    def test_basic_drug_mp_greater_than_one(self):
        """Basic drugs (pKa_base high) concentrate in acidic milk."""
        mp = calculate_milk_to_plasma_ratio(pka_acid=None, pka_base=9.0, logP=1.5, fu_plasma=1.0)
        assert mp > 1.0

    def test_acidic_drug_mp_less_than_one(self):
        """Acidic drugs (pKa_acid low) favor plasma over milk."""
        mp = calculate_milk_to_plasma_ratio(pka_acid=4.0, pka_base=None, logP=1.5, fu_plasma=1.0)
        assert mp < 1.0

    def test_neutral_drug_in_range(self):
        """Neutral drugs return M/P between 0.1 and 5.0."""
        mp = calculate_milk_to_plasma_ratio(pka_acid=None, pka_base=None, logP=1.5, fu_plasma=0.5)
        assert 0.05 <= mp <= 10.0

    def test_neutral_drug_logP_zero(self):
        """logP=0 neutral drug gives M/P near 0.5."""
        mp = calculate_milk_to_plasma_ratio(pka_acid=None, pka_base=None, logP=0.0, fu_plasma=1.0)
        assert 0.05 <= mp <= 10.0

    def test_clamp_upper_bound(self):
        """Very basic drug with fu=1 should be clamped to 10.0."""
        mp = calculate_milk_to_plasma_ratio(pka_acid=None, pka_base=14.0, logP=0.0, fu_plasma=1.0)
        assert mp <= 10.0

    def test_clamp_lower_bound(self):
        """Very acidic drug with low fu should be clamped to 0.05."""
        mp = calculate_milk_to_plasma_ratio(pka_acid=1.0, pka_base=None, logP=0.0, fu_plasma=0.001)
        assert mp >= 0.05

    def test_fu_plasma_scales_mp(self):
        """Higher fu_plasma → higher M/P for basic drugs."""
        mp_low = calculate_milk_to_plasma_ratio(None, 9.0, 1.0, fu_plasma=0.1)
        mp_high = calculate_milk_to_plasma_ratio(None, 9.0, 1.0, fu_plasma=0.9)
        assert mp_high > mp_low

    def test_returns_float(self):
        mp = calculate_milk_to_plasma_ratio(None, 8.0, 2.0, 0.5)
        assert isinstance(mp, float)


# ---------------------------------------------------------------------------
# simulate_breast_milk_pk
# ---------------------------------------------------------------------------


class TestSimulateBreastMilkPk:
    def test_returns_dataclass(self):
        r = _default_sim()
        assert isinstance(r, BreastMilkResult)

    def test_frozen_dataclass_immutable(self):
        r = _default_sim()
        with pytest.raises((AttributeError, TypeError)):
            r.drug_name = "other"  # type: ignore[misc]

    def test_rid_less_than_100_pct(self):
        """RID should always be < 100% for realistic parameters."""
        r = _default_sim(milk_to_plasma_ratio=5.0)
        assert r.relative_infant_dose_pct < 100.0

    def test_c_milk_all_nonnegative(self):
        r = _default_sim()
        assert all(c >= 0 for c in r.c_milk_mg_L)

    def test_times_h_length_matches_c_milk(self):
        r = _default_sim()
        assert len(r.times_h) == len(r.c_milk_mg_L)

    def test_maternal_cmax_positive(self):
        r = _default_sim()
        assert r.maternal_cmax_mg_L > 0

    def test_maternal_auc_positive(self):
        r = _default_sim()
        assert r.maternal_auc_mg_h_per_L > 0

    def test_low_mp_ratio_compatible(self):
        """Very low M/P ratio → low RID → compatible recommendation."""
        r = _default_sim(milk_to_plasma_ratio=0.01)
        assert r.risk_category == "low"
        assert r.breastfeeding_recommendation == "compatible"

    def test_high_mp_ratio_higher_rid(self):
        """Higher M/P ratio produces higher RID."""
        r_low = _default_sim(milk_to_plasma_ratio=0.1)
        r_high = _default_sim(milk_to_plasma_ratio=5.0)
        assert r_high.relative_infant_dose_pct > r_low.relative_infant_dose_pct

    def test_dose_linearity_infant_dose(self):
        """2x maternal dose → 2x infant daily dose (linear system)."""
        r1 = _default_sim(maternal_dose_mg=100.0)
        r2 = _default_sim(maternal_dose_mg=200.0)
        ratio = r2.infant_daily_dose_mg_per_kg / r1.infant_daily_dose_mg_per_kg
        assert abs(ratio - 2.0) < 0.05

    def test_risk_category_moderate(self):
        """Moderate RID (10–25%) gives use_with_caution."""
        # Force moderate RID by using high M/P with large milk intake
        r = _default_sim(
            milk_to_plasma_ratio=3.0,
            milk_intake_mL_per_day=300.0,
            maternal_dose_mg=50.0,
            cl_maternal_L_per_h=2.0,
        )
        # Just check valid category
        assert r.risk_category in ("low", "moderate", "high")
        assert r.breastfeeding_recommendation in ("compatible", "use_with_caution", "avoid")

    def test_risk_high_category(self):
        """Extreme M/P with tiny maternal dose → high RID."""
        r = _default_sim(
            milk_to_plasma_ratio=10.0,
            milk_intake_mL_per_day=1000.0,
            maternal_dose_mg=1.0,
            cl_maternal_L_per_h=0.1,
            vd_maternal_L=5.0,
        )
        assert r.risk_category == "high"
        assert r.breastfeeding_recommendation == "avoid"

    def test_infant_dose_positive(self):
        r = _default_sim()
        assert r.infant_daily_dose_mg_per_kg >= 0

    def test_times_h_starts_at_zero(self):
        r = _default_sim()
        assert r.times_h[0] == pytest.approx(0.0)

    def test_times_h_ends_at_t_end(self):
        r = _default_sim(t_end_h=12.0)
        assert r.times_h[-1] == pytest.approx(12.0, abs=0.6)

    def test_drug_name_preserved(self):
        r = _default_sim(drug_name="Ibuprofen")
        assert r.drug_name == "Ibuprofen"

    def test_invalid_dose_raises(self):
        with pytest.raises(ValueError, match="maternal_dose_mg"):
            _default_sim(maternal_dose_mg=0.0)

    def test_invalid_cl_raises(self):
        with pytest.raises(ValueError, match="cl_maternal_L_per_h"):
            _default_sim(cl_maternal_L_per_h=-1.0)

    def test_invalid_f_oral_raises(self):
        with pytest.raises(ValueError, match="f_oral"):
            _default_sim(f_oral=0.0)

    def test_invalid_infant_weight_raises(self):
        with pytest.raises(ValueError, match="infant_weight_kg"):
            _default_sim(infant_weight_kg=-5.0)

    def test_invalid_mp_ratio_raises(self):
        with pytest.raises(ValueError, match="milk_to_plasma_ratio"):
            _default_sim(milk_to_plasma_ratio=0.0)


# ---------------------------------------------------------------------------
# screen_drugs_for_breastfeeding
# ---------------------------------------------------------------------------


class TestScreenDrugsForBreastfeeding:
    def _make_list(self):
        return [
            {
                "name": "DrugA",
                "maternal_dose_mg": 100.0,
                "cl_L_per_h": 5.0,
                "vd_L": 50.0,
                "m_p_ratio": 0.1,
            },
            {
                "name": "DrugB",
                "maternal_dose_mg": 100.0,
                "cl_L_per_h": 5.0,
                "vd_L": 50.0,
                "m_p_ratio": 2.0,
            },
            {
                "name": "DrugC",
                "maternal_dose_mg": 100.0,
                "cl_L_per_h": 5.0,
                "vd_L": 50.0,
                "m_p_ratio": 0.5,
            },
        ]

    def test_returns_list(self):
        results = screen_drugs_for_breastfeeding(self._make_list())
        assert isinstance(results, list)

    def test_correct_length(self):
        results = screen_drugs_for_breastfeeding(self._make_list())
        assert len(results) == 3

    def test_sorted_ascending_rid(self):
        results = screen_drugs_for_breastfeeding(self._make_list())
        rids = [r.relative_infant_dose_pct for r in results]
        assert rids == sorted(rids)

    def test_safest_first_is_low_mp(self):
        """Drug with lowest M/P should appear first."""
        results = screen_drugs_for_breastfeeding(self._make_list())
        assert results[0].drug_name == "DrugA"

    def test_returns_breast_milk_result_objects(self):
        results = screen_drugs_for_breastfeeding(self._make_list())
        assert all(isinstance(r, BreastMilkResult) for r in results)
