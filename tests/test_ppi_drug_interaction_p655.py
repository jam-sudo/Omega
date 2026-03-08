"""Tests for PPI drug interaction model (phase 655)."""

from __future__ import annotations

import pytest

from omega_pbpk.clinical.ppi_drug_interaction_p655 import (
    PPIDrugInteractionResult,
    predict_ppi_interaction,
    screen_ppi_interactions,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _base_result(**kw):
    defaults = dict(drug_name="drugA", logP=2.0, mw_Da=300.0, pka=8.0, is_base=True)
    defaults.update(kw)
    return predict_ppi_interaction(**defaults)


def _acid_result(**kw):
    defaults = dict(drug_name="drugB", logP=1.5, mw_Da=250.0, pka=4.0, is_base=False)
    defaults.update(kw)
    return predict_ppi_interaction(**defaults)


# ---------------------------------------------------------------------------
# Return type & frozen
# ---------------------------------------------------------------------------


class TestReturnType:
    def test_returns_frozen_dataclass(self):
        r = _base_result()
        assert isinstance(r, PPIDrugInteractionResult)

    def test_mutation_raises(self):
        r = _base_result()
        with pytest.raises(AttributeError):
            r.drug_name = "other"  # type: ignore[misc]

    def test_drug_name_preserved(self):
        r = _base_result(drug_name="TestDrug")
        assert r.drug_name == "TestDrug"

    def test_ppi_drug_default(self):
        r = _base_result()
        assert r.ppi_drug == "omeprazole"


# ---------------------------------------------------------------------------
# Gastric pH
# ---------------------------------------------------------------------------


class TestGastricPH:
    def test_baseline_ph(self):
        r = _base_result()
        assert r.baseline_gastric_ph == 1.5

    def test_ppi_ph(self):
        r = _base_result()
        assert r.ppi_gastric_ph == 4.5

    def test_ppi_ph_gt_baseline(self):
        r = _base_result()
        assert r.ppi_gastric_ph > r.baseline_gastric_ph


# ---------------------------------------------------------------------------
# Basic drug: PPI reduces absorption
# ---------------------------------------------------------------------------


class TestBasicDrug:
    def test_high_pka_base_ppi_reduces_fa(self):
        r = _base_result(pka=5.0, dose_mg=1000.0, solubility_at_ph1_mg_mL=0.001)
        assert r.ppi_fa_pct < r.baseline_fa_pct

    def test_delta_fa_negative_for_base(self):
        r = _base_result(pka=5.0, dose_mg=1000.0, solubility_at_ph1_mg_mL=0.001)
        assert r.delta_fa_pct < 0

    def test_solubility_decreases_with_ppi_for_base(self):
        r = _base_result(pka=9.0)
        assert r.ppi_solubility_mg_mL < r.baseline_solubility_mg_mL

    def test_delta_equals_diff(self):
        r = _base_result()
        assert abs(r.delta_fa_pct - (r.ppi_fa_pct - r.baseline_fa_pct)) < 1e-9


# ---------------------------------------------------------------------------
# Acidic drug: PPI increases absorption
# ---------------------------------------------------------------------------


class TestAcidicDrug:
    def test_acid_ppi_increases_fa(self):
        r = _acid_result(pka=3.0, dose_mg=500.0, solubility_at_ph1_mg_mL=0.01)
        assert r.ppi_fa_pct >= r.baseline_fa_pct

    def test_acid_solubility_increases(self):
        r = _acid_result(pka=3.0)
        assert r.ppi_solubility_mg_mL >= r.baseline_solubility_mg_mL

    def test_acid_delta_fa_non_negative(self):
        r = _acid_result(pka=3.0, dose_mg=500.0, solubility_at_ph1_mg_mL=0.01)
        assert r.delta_fa_pct >= 0


# ---------------------------------------------------------------------------
# AUC ratio clamping
# ---------------------------------------------------------------------------


class TestAUCRatio:
    def test_baseline_auc_ratio_is_one(self):
        r = _base_result()
        assert r.baseline_auc_ratio == 1.0

    def test_ppi_auc_ratio_in_range(self):
        r = _base_result()
        assert 0.1 <= r.ppi_auc_ratio <= 5.0

    def test_ppi_auc_ratio_clamped_low(self):
        # very high pKa base drug: massive solubility drop
        r = _base_result(pka=12.0, dose_mg=1000.0, solubility_at_ph1_mg_mL=0.001)
        assert r.ppi_auc_ratio >= 0.1


# ---------------------------------------------------------------------------
# DDI magnitude & clinical significance
# ---------------------------------------------------------------------------


class TestDDIMagnitude:
    def test_minor(self):
        # Mild base with low pKa — small effect
        r = _base_result(pka=2.0)
        assert r.ddi_magnitude == "minor"
        assert r.clinical_significance == "no_action"

    def test_moderate_or_major_high_pka(self):
        r = _base_result(pka=5.0, dose_mg=1000.0, solubility_at_ph1_mg_mL=0.001)
        assert r.ddi_magnitude in ("moderate", "major")
        assert r.clinical_significance in ("monitor", "avoid")

    def test_major_implies_avoid(self):
        r = _base_result(pka=10.0, dose_mg=1000.0, solubility_at_ph1_mg_mL=0.01)
        if r.ddi_magnitude == "major":
            assert r.clinical_significance == "avoid"


# ---------------------------------------------------------------------------
# Screening
# ---------------------------------------------------------------------------


class TestScreening:
    def test_screen_returns_list(self):
        compounds = [
            dict(drug_name="A", logP=2.0, mw_Da=300.0, pka=9.0, is_base=True),
            dict(drug_name="B", logP=1.0, mw_Da=250.0, pka=3.0, is_base=False),
        ]
        results = screen_ppi_interactions(compounds)
        assert isinstance(results, list)
        assert len(results) == 2

    def test_screen_sorted_by_abs_delta_fa(self):
        compounds = [
            dict(drug_name="A", logP=2.0, mw_Da=300.0, pka=2.0, is_base=True),
            dict(drug_name="B", logP=2.0, mw_Da=300.0, pka=9.0, is_base=True),
        ]
        results = screen_ppi_interactions(compounds)
        assert abs(results[0].delta_fa_pct) >= abs(results[1].delta_fa_pct)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TestValidation:
    def test_mw_zero_raises(self):
        with pytest.raises(ValueError):
            _base_result(mw_Da=0)

    def test_mw_negative_raises(self):
        with pytest.raises(ValueError):
            _base_result(mw_Da=-100)

    def test_solubility_zero_raises(self):
        with pytest.raises(ValueError):
            _base_result(solubility_at_ph1_mg_mL=0)

    def test_dose_zero_raises(self):
        with pytest.raises(ValueError):
            _base_result(dose_mg=0)

    def test_dose_negative_raises(self):
        with pytest.raises(ValueError):
            _base_result(dose_mg=-10)
