"""Tests for Phase 676 — Dialysis Drug Removal."""

import pytest

from omega_pbpk.clinical.dialysis_drug_removal_p676 import (
    DialysisDrugRemovalResult,
    predict_dialysis_removal,
)


def _default_pred(**kwargs):
    defaults = dict(
        drug_name="TestDrug",
        pre_dialysis_conc_mg_L=20.0,
        vd_L=40.0,
        protein_binding_pct=30.0,
        mw_Da=300.0,
    )
    defaults.update(kwargs)
    return predict_dialysis_removal(**defaults)


class TestReturnType:
    def test_returns_result_type(self):
        r = _default_pred()
        assert isinstance(r, DialysisDrugRemovalResult)

    def test_frozen(self):
        r = _default_pred()
        with pytest.raises(AttributeError):
            r.drug_name = "Other"

    def test_mutation_raises(self):
        r = _default_pred()
        with pytest.raises(AttributeError):
            r.removal_pct = 50.0


class TestProteinBinding:
    def test_high_binding_lower_removal(self):
        r_low = _default_pred(protein_binding_pct=10.0)
        r_high = _default_pred(protein_binding_pct=90.0)
        assert r_high.removal_pct < r_low.removal_pct

    def test_high_binding_lower_sc(self):
        r_low = _default_pred(protein_binding_pct=10.0)
        r_high = _default_pred(protein_binding_pct=90.0)
        assert r_high.sieving_coefficient < r_low.sieving_coefficient


class TestMolecularWeight:
    def test_high_mw_lower_sieving(self):
        r_low = _default_pred(mw_Da=200.0)
        r_high = _default_pred(mw_Da=2000.0)
        assert r_high.sieving_coefficient < r_low.sieving_coefficient

    def test_high_mw_less_removal(self):
        r_low = _default_pred(mw_Da=200.0)
        r_high = _default_pred(mw_Da=2000.0)
        assert r_high.removal_pct < r_low.removal_pct


class TestBloodFlow:
    def test_higher_flow_higher_cl(self):
        r_low = _default_pred(blood_flow_mL_min=200.0)
        r_high = _default_pred(blood_flow_mL_min=400.0)
        assert r_high.dialysis_clearance_L_per_h > r_low.dialysis_clearance_L_per_h


class TestRemovalPct:
    def test_in_range(self):
        r = _default_pred()
        assert 0 <= r.removal_pct <= 100

    def test_removal_pct_positive(self):
        r = _default_pred()
        assert r.removal_pct > 0


class TestPostDialysis:
    def test_post_less_than_pre(self):
        r = _default_pred()
        assert r.post_dialysis_conc_mg_L < r.pre_dialysis_conc_mg_L

    def test_post_non_negative(self):
        r = _default_pred()
        assert r.post_dialysis_conc_mg_L >= 0


class TestReboundFactor:
    def test_rebound_at_least_one(self):
        r = _default_pred()
        assert r.rebound_factor >= 1.0

    def test_rebound_capped(self):
        r = _default_pred(vd_L=500.0)
        assert r.rebound_factor <= 3.0


class TestSupplementalDose:
    def test_supplemental_non_negative(self):
        r = _default_pred()
        assert r.supplemental_dose_needed_mg >= 0

    def test_supplemental_at_least_removed(self):
        r = _default_pred()
        assert r.supplemental_dose_needed_mg >= r.total_removed_mg


class TestDialyzabilityClass:
    def test_highly_dialyzable(self):
        r = _default_pred(protein_binding_pct=0.0, mw_Da=100.0, vd_L=10.0)
        assert r.dialyzability_class == "highly_dialyzable"

    def test_not_dialyzable(self):
        r = _default_pred(protein_binding_pct=99.0, mw_Da=5000.0)
        assert r.dialyzability_class == "not_dialyzable"

    def test_class_is_valid_string(self):
        r = _default_pred()
        assert r.dialyzability_class in {
            "highly_dialyzable",
            "dialyzable",
            "partially_dialyzable",
            "not_dialyzable",
        }


class TestValidation:
    def test_zero_conc(self):
        with pytest.raises(ValueError):
            _default_pred(pre_dialysis_conc_mg_L=0)

    def test_negative_vd(self):
        with pytest.raises(ValueError):
            _default_pred(vd_L=-1)

    def test_binding_over_100(self):
        with pytest.raises(ValueError):
            _default_pred(protein_binding_pct=101)

    def test_binding_negative(self):
        with pytest.raises(ValueError):
            _default_pred(protein_binding_pct=-1)

    def test_zero_mw(self):
        with pytest.raises(ValueError):
            _default_pred(mw_Da=0)

    def test_zero_blood_flow(self):
        with pytest.raises(ValueError):
            _default_pred(blood_flow_mL_min=0)

    def test_zero_session(self):
        with pytest.raises(ValueError):
            _default_pred(session_duration_h=0)


class TestNotes:
    def test_notes_present(self):
        r = _default_pred()
        assert len(r.notes) > 0
