"""Tests for Phase 430 — Dissolution medium effect prediction."""

import pytest

from omega_pbpk.prediction.dissolution_medium_effect import (
    MEDIA,
    DissolutionMediumResult,
    compare_media,
    predict_dissolution_medium,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _base(**kwargs):
    defaults = dict(
        drug_name="ibuprofen",
        medium="FaSSIF",
        drug_pka=4.4,
        logp=3.5,
        ionization_type="acid",
        dose_mg=200.0,
        intrinsic_solubility_mg_mL=0.05,
    )
    defaults.update(kwargs)
    return predict_dissolution_medium(**defaults)


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------


class TestReturnType:
    def test_return_type(self):
        result = _base()
        assert isinstance(result, DissolutionMediumResult)


# ---------------------------------------------------------------------------
# Field preservation
# ---------------------------------------------------------------------------


class TestFieldPreservation:
    def test_drug_name_preserved(self):
        result = _base(drug_name="ketoconazole")
        assert result.drug_name == "ketoconazole"

    def test_medium_preserved(self):
        result = _base(medium="FeSSIF")
        assert result.medium == "FeSSIF"

    def test_pka_preserved(self):
        result = _base(drug_pka=5.5)
        assert result.drug_pka == 5.5

    def test_logp_preserved(self):
        result = _base(logp=2.0)
        assert result.logp == 2.0

    def test_dose_preserved(self):
        result = _base(dose_mg=100.0)
        assert result.dose_mg == 100.0

    def test_intrinsic_solubility_preserved(self):
        result = _base(intrinsic_solubility_mg_mL=0.1)
        assert result.intrinsic_solubility_mg_mL == 0.1


# ---------------------------------------------------------------------------
# Medium solubility
# ---------------------------------------------------------------------------


class TestMediumSolubility:
    def test_medium_solubility_ge_intrinsic(self):
        """Medium solubility should be at least equal to intrinsic."""
        result = _base()
        assert result.medium_solubility_mg_mL >= result.intrinsic_solubility_mg_mL - 1e-9

    def test_solubility_enhancement_ge_1(self):
        result = _base()
        assert result.solubility_enhancement >= 1.0 - 1e-9

    def test_acid_drug_high_ph_enhanced(self):
        """Acid drug in phosphate buffer pH 6.8 should have enhanced solubility."""
        result = predict_dissolution_medium(
            drug_name="acid_drug",
            medium="phosphate_buffer_pH68",
            drug_pka=4.4,
            logp=1.0,
            ionization_type="acid",
            dose_mg=100.0,
            intrinsic_solubility_mg_mL=0.1,
        )
        # pH 6.8 >> pKa 4.4 → strong enhancement
        assert result.medium_solubility_mg_mL > result.intrinsic_solubility_mg_mL

    def test_fassif_higher_solubility_than_sgf_for_lipophilic(self):
        """FaSSIF has bile salts, SGF does not — lipophilic drug should dissolve better."""
        fassif = predict_dissolution_medium(
            drug_name="drug",
            medium="FaSSIF",
            drug_pka=7.0,
            logp=3.0,
            ionization_type="neutral",
            dose_mg=100.0,
            intrinsic_solubility_mg_mL=0.02,
        )
        sgf = predict_dissolution_medium(
            drug_name="drug",
            medium="SGF",
            drug_pka=7.0,
            logp=3.0,
            ionization_type="neutral",
            dose_mg=100.0,
            intrinsic_solubility_mg_mL=0.02,
        )
        assert fassif.medium_solubility_mg_mL > sgf.medium_solubility_mg_mL

    def test_fessif_higher_than_fassif_lipophilic(self):
        """FeSSIF has more bile (15 mM) than FaSSIF (3 mM) → higher solubility for lipophilic drug."""
        fessif = predict_dissolution_medium(
            drug_name="drug",
            medium="FeSSIF",
            drug_pka=7.0,
            logp=3.0,
            ionization_type="neutral",
            dose_mg=100.0,
            intrinsic_solubility_mg_mL=0.02,
        )
        fassif = predict_dissolution_medium(
            drug_name="drug",
            medium="FaSSIF",
            drug_pka=7.0,
            logp=3.0,
            ionization_type="neutral",
            dose_mg=100.0,
            intrinsic_solubility_mg_mL=0.02,
        )
        assert fessif.medium_solubility_mg_mL > fassif.medium_solubility_mg_mL


# ---------------------------------------------------------------------------
# Dose number and dissolution class
# ---------------------------------------------------------------------------


class TestDoseNumber:
    def test_dose_number_positive(self):
        result = _base()
        assert result.dose_number > 0.0

    def test_sink_conditions_when_dose_number_lt_1(self):
        # Very high intrinsic solubility → dose_number < 1
        result = predict_dissolution_medium(
            drug_name="drug",
            medium="water",
            drug_pka=7.0,
            logp=0.5,
            ionization_type="neutral",
            dose_mg=1.0,
            intrinsic_solubility_mg_mL=100.0,
        )
        assert result.dose_number < 1.0
        assert result.dissolution_class == "sink_conditions"

    def test_dissolution_limited_when_dose_number_gt_3(self):
        # Low solubility, high dose → dose_number > 3
        result = predict_dissolution_medium(
            drug_name="drug",
            medium="SGF",
            drug_pka=7.0,
            logp=0.5,
            ionization_type="neutral",
            dose_mg=1000.0,
            intrinsic_solubility_mg_mL=0.001,
        )
        assert result.dose_number > 3.0
        assert result.dissolution_class == "dissolution_limited"

    def test_borderline_class(self):
        # Tune dose_number to be between 1 and 3
        result = predict_dissolution_medium(
            drug_name="drug",
            medium="water",
            drug_pka=7.0,
            logp=0.5,
            ionization_type="neutral",
            dose_mg=100.0,
            intrinsic_solubility_mg_mL=0.3,  # 0.3 * 250 = 75 mg → dose_number = 100/75 ≈ 1.33
        )
        assert 1.0 <= result.dose_number <= 3.0
        assert result.dissolution_class == "borderline"


# ---------------------------------------------------------------------------
# Predicted f_abs
# ---------------------------------------------------------------------------


class TestPredictedFAbs:
    def test_f_abs_between_0_and_1(self):
        result = _base()
        assert 0.0 <= result.predicted_f_abs <= 1.0

    def test_sink_conditions_f_abs_is_one(self):
        result = predict_dissolution_medium(
            drug_name="drug",
            medium="water",
            drug_pka=7.0,
            logp=0.5,
            ionization_type="neutral",
            dose_mg=1.0,
            intrinsic_solubility_mg_mL=100.0,
        )
        assert result.predicted_f_abs == 1.0


# ---------------------------------------------------------------------------
# Compare media
# ---------------------------------------------------------------------------


class TestCompareMedia:
    def test_compare_returns_6_results(self):
        results = compare_media(
            drug_name="ibuprofen",
            drug_pka=4.4,
            logp=3.5,
            ionization_type="acid",
            dose_mg=200.0,
            intrinsic_solubility_mg_mL=0.05,
        )
        assert len(results) == 6

    def test_compare_sorted_descending(self):
        results = compare_media(
            drug_name="ibuprofen",
            drug_pka=4.4,
            logp=3.5,
            ionization_type="acid",
            dose_mg=200.0,
            intrinsic_solubility_mg_mL=0.05,
        )
        solubilities = [r.medium_solubility_mg_mL for r in results]
        assert solubilities == sorted(solubilities, reverse=True)

    def test_compare_all_media_represented(self):
        results = compare_media(
            drug_name="drug",
            drug_pka=7.0,
            logp=2.0,
            ionization_type="neutral",
            dose_mg=100.0,
            intrinsic_solubility_mg_mL=0.1,
        )
        mediums = {r.medium for r in results}
        assert mediums == set(MEDIA.keys())


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


class TestValidation:
    def test_invalid_medium_raises(self):
        with pytest.raises(ValueError):
            _base(medium="FaSSJEJUNUM")

    def test_zero_intrinsic_solubility_raises(self):
        with pytest.raises(ValueError):
            _base(intrinsic_solubility_mg_mL=0.0)

    def test_negative_intrinsic_solubility_raises(self):
        with pytest.raises(ValueError):
            _base(intrinsic_solubility_mg_mL=-0.1)

    def test_zero_dose_raises(self):
        with pytest.raises(ValueError):
            _base(dose_mg=0.0)

    def test_invalid_ionization_type_raises(self):
        with pytest.raises(ValueError):
            _base(ionization_type="zwitterion")
