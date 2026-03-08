"""Tests for Phase 533 — Metabolite Safety Monitoring."""

import pytest

from omega_pbpk.risk.metabolite_safety_monitoring import (
    MetaboliteSafetyResult,
    assess_metabolite_safety,
    screen_metabolites,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_result(**kwargs):
    defaults = dict(
        drug_name="DrugA",
        metabolite_name="MetA",
        dose_mg=100.0,
        cl_parent_L_per_h=10.0,
        vd_L=50.0,
        cl_metabolite_L_per_h=10.0,
        fm=0.8,
        mw_parent_Da=400.0,
        mw_metabolite_Da=370.0,
    )
    defaults.update(kwargs)
    return assess_metabolite_safety(**defaults)


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------


class TestReturnType:
    def test_returns_dataclass(self):
        r = make_result()
        assert isinstance(r, MetaboliteSafetyResult)

    def test_frozen(self):
        r = make_result()
        with pytest.raises((AttributeError, TypeError)):
            r.drug_name = "X"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Fields presence
# ---------------------------------------------------------------------------


class TestFields:
    def test_drug_name(self):
        r = make_result(drug_name="TestDrug")
        assert r.drug_name == "TestDrug"

    def test_metabolite_name(self):
        r = make_result(metabolite_name="MetX")
        assert r.metabolite_name == "MetX"

    def test_dose_stored(self):
        r = make_result(dose_mg=200.0)
        assert r.dose_mg == pytest.approx(200.0)

    def test_fm_stored(self):
        r = make_result(fm=0.5)
        assert r.fm == pytest.approx(0.5)

    def test_mw_parent_stored(self):
        r = make_result(mw_parent_Da=500.0)
        assert r.mw_parent_Da == pytest.approx(500.0)

    def test_mw_metabolite_stored(self):
        r = make_result(mw_metabolite_Da=450.0)
        assert r.mw_metabolite_Da == pytest.approx(450.0)


# ---------------------------------------------------------------------------
# AUC parent = dose / CL
# ---------------------------------------------------------------------------


class TestAUCParent:
    def test_auc_parent_formula(self):
        r = make_result(dose_mg=200.0, cl_parent_L_per_h=20.0)
        assert r.auc_parent_mg_h_per_L == pytest.approx(200.0 / 20.0)

    def test_auc_parent_different_values(self):
        r = make_result(dose_mg=50.0, cl_parent_L_per_h=5.0)
        assert r.auc_parent_mg_h_per_L == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# Cmax parent = dose / Vd
# ---------------------------------------------------------------------------


class TestCmaxParent:
    def test_cmax_parent_formula(self):
        r = make_result(dose_mg=100.0, vd_L=50.0)
        assert r.cmax_parent_mg_L == pytest.approx(100.0 / 50.0)


# ---------------------------------------------------------------------------
# AUC ratio formula
# ---------------------------------------------------------------------------


class TestAUCRatio:
    def test_ratio_equal_cl_same_mw(self):
        """When CL_met = CL_parent and molar_ratio=1, ratio = fm."""
        r = make_result(
            dose_mg=100.0,
            cl_parent_L_per_h=10.0,
            cl_metabolite_L_per_h=10.0,
            fm=0.5,
            mw_parent_Da=400.0,
            mw_metabolite_Da=400.0,
        )
        # AUC_met = AUC_parent * fm * 1 * (10/10) = AUC_parent * fm
        # ratio = fm = 0.5
        assert r.auc_ratio_met_to_parent == pytest.approx(0.5)

    def test_ratio_uses_molar_ratio(self):
        """Molar ratio = mw_parent / mw_metabolite scales ratio."""
        r = make_result(
            cl_parent_L_per_h=10.0,
            cl_metabolite_L_per_h=10.0,
            fm=0.5,
            mw_parent_Da=400.0,
            mw_metabolite_Da=200.0,  # molar_ratio = 2
        )
        # ratio = fm * molar_ratio * (CL_parent / CL_met) = 0.5 * 2 * 1 = 1.0
        assert r.auc_ratio_met_to_parent == pytest.approx(1.0)

    def test_ratio_with_lower_cl_met(self):
        """Lower metabolite CL → higher AUC ratio."""
        r_low = make_result(
            cl_metabolite_L_per_h=2.0, mw_parent_Da=400.0, mw_metabolite_Da=400.0, fm=0.5
        )
        r_high = make_result(
            cl_metabolite_L_per_h=20.0, mw_parent_Da=400.0, mw_metabolite_Da=400.0, fm=0.5
        )
        assert r_low.auc_ratio_met_to_parent > r_high.auc_ratio_met_to_parent


# ---------------------------------------------------------------------------
# MIST threshold (> 0.10)
# ---------------------------------------------------------------------------


class TestMISTThreshold:
    def test_below_threshold(self):
        # Make ratio < 0.10: fm=0.05, same MW, same CL → ratio=0.05
        r = make_result(
            fm=0.05,
            cl_parent_L_per_h=10.0,
            cl_metabolite_L_per_h=10.0,
            mw_parent_Da=400.0,
            mw_metabolite_Da=400.0,
        )
        assert r.mist_threshold_exceeded is False

    def test_above_threshold(self):
        # Make ratio > 0.10: fm=0.5, same MW, same CL → ratio=0.5
        r = make_result(
            fm=0.5,
            cl_parent_L_per_h=10.0,
            cl_metabolite_L_per_h=10.0,
            mw_parent_Da=400.0,
            mw_metabolite_Da=400.0,
        )
        assert r.mist_threshold_exceeded is True


# ---------------------------------------------------------------------------
# Major metabolite (> 0.25)
# ---------------------------------------------------------------------------


class TestMajorMetabolite:
    def test_not_major(self):
        # fm=0.2, same MW, same CL → ratio=0.2 (> 0.10 but < 0.25?)
        # 0.2 < 0.25 → not major
        r = make_result(
            fm=0.2,
            cl_parent_L_per_h=10.0,
            cl_metabolite_L_per_h=10.0,
            mw_parent_Da=400.0,
            mw_metabolite_Da=400.0,
        )
        assert r.major_metabolite is False

    def test_is_major(self):
        # fm=0.5, same MW, same CL → ratio=0.5 > 0.25
        r = make_result(
            fm=0.5,
            cl_parent_L_per_h=10.0,
            cl_metabolite_L_per_h=10.0,
            mw_parent_Da=400.0,
            mw_metabolite_Da=400.0,
        )
        assert r.major_metabolite is True


# ---------------------------------------------------------------------------
# Safety testing classification
# ---------------------------------------------------------------------------


class TestSafetyTestingRequired:
    def test_none(self):
        # ratio < 0.10
        r = make_result(
            fm=0.05,
            cl_parent_L_per_h=10.0,
            cl_metabolite_L_per_h=10.0,
            mw_parent_Da=400.0,
            mw_metabolite_Da=400.0,
        )
        assert r.safety_testing_required == "none"

    def test_characterization(self):
        # 0.10 < ratio < 0.25: fm=0.2, same MW, same CL → ratio=0.2
        r = make_result(
            fm=0.2,
            cl_parent_L_per_h=10.0,
            cl_metabolite_L_per_h=10.0,
            mw_parent_Da=400.0,
            mw_metabolite_Da=400.0,
        )
        assert r.safety_testing_required == "characterization"

    def test_full_ich_m3(self):
        # ratio > 0.25
        r = make_result(
            fm=0.5,
            cl_parent_L_per_h=10.0,
            cl_metabolite_L_per_h=10.0,
            mw_parent_Da=400.0,
            mw_metabolite_Da=400.0,
        )
        assert r.safety_testing_required == "full_ich_m3"


# ---------------------------------------------------------------------------
# Higher fm → higher ratio
# ---------------------------------------------------------------------------


class TestFmEffect:
    def test_higher_fm_higher_ratio(self):
        r_low = make_result(fm=0.3, mw_parent_Da=400.0, mw_metabolite_Da=400.0)
        r_high = make_result(fm=0.7, mw_parent_Da=400.0, mw_metabolite_Da=400.0)
        assert r_high.auc_ratio_met_to_parent > r_low.auc_ratio_met_to_parent


# ---------------------------------------------------------------------------
# Screen function
# ---------------------------------------------------------------------------


class TestScreenMetabolites:
    def _make_metabolites(self):
        return [
            {
                "metabolite_name": "MetA",
                "cl_metabolite_L_per_h": 10.0,
                "fm": 0.2,
                "mw_metabolite_Da": 400.0,
            },
            {
                "metabolite_name": "MetB",
                "cl_metabolite_L_per_h": 5.0,
                "fm": 0.5,
                "mw_metabolite_Da": 400.0,
            },
            {
                "metabolite_name": "MetC",
                "cl_metabolite_L_per_h": 20.0,
                "fm": 0.05,
                "mw_metabolite_Da": 400.0,
            },
        ]

    def test_returns_list(self):
        mets = self._make_metabolites()
        results = screen_metabolites("DrugA", 100.0, 10.0, 50.0, mets)
        assert isinstance(results, list)

    def test_length(self):
        mets = self._make_metabolites()
        results = screen_metabolites("DrugA", 100.0, 10.0, 50.0, mets)
        assert len(results) == 3

    def test_sorted_descending(self):
        mets = self._make_metabolites()
        results = screen_metabolites("DrugA", 100.0, 10.0, 50.0, mets)
        ratios = [r.auc_ratio_met_to_parent for r in results]
        assert ratios == sorted(ratios, reverse=True)

    def test_first_is_highest_ratio(self):
        mets = self._make_metabolites()
        results = screen_metabolites("DrugA", 100.0, 10.0, 50.0, mets)
        # MetB: fm=0.5, CL_met=5 → highest ratio
        assert results[0].metabolite_name == "MetB"


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


class TestValidation:
    def test_dose_zero_raises(self):
        with pytest.raises(ValueError, match="dose_mg"):
            make_result(dose_mg=0.0)

    def test_dose_negative_raises(self):
        with pytest.raises(ValueError, match="dose_mg"):
            make_result(dose_mg=-10.0)

    def test_cl_parent_zero_raises(self):
        with pytest.raises(ValueError, match="cl_parent"):
            make_result(cl_parent_L_per_h=0.0)

    def test_vd_zero_raises(self):
        with pytest.raises(ValueError, match="vd_L"):
            make_result(vd_L=0.0)

    def test_cl_metabolite_zero_raises(self):
        with pytest.raises(ValueError, match="cl_metabolite"):
            make_result(cl_metabolite_L_per_h=0.0)

    def test_fm_zero_raises(self):
        with pytest.raises(ValueError, match="fm"):
            make_result(fm=0.0)

    def test_fm_one_raises(self):
        with pytest.raises(ValueError, match="fm"):
            make_result(fm=1.0)

    def test_fm_negative_raises(self):
        with pytest.raises(ValueError, match="fm"):
            make_result(fm=-0.1)

    def test_notes_is_str(self):
        r = make_result()
        assert isinstance(r.notes, str)
        assert len(r.notes) > 0
