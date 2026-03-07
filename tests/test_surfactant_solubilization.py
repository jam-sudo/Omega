"""Tests for prediction/surfactant_solubilization.py (Phase 1074)."""

import pytest

from omega_pbpk.prediction.surfactant_solubilization import (
    SurfactantSolubilizationResult,
    predict_solubilization,
    screen_surfactants,
)

_ALL_SURFACTANTS = ["SDS", "Tween_80", "CTAB", "Pluronic_F127", "sodium_cholate"]
_VALID_EFFICIENCIES = {"none", "low", "moderate", "high"}


def _make_result(**kwargs) -> SurfactantSolubilizationResult:
    defaults = dict(
        drug_name="TestDrug",
        surfactant="SDS",
        logP=2.0,
        surfactant_conc_mM=20.0,
        intrinsic_solubility_mg_mL=0.1,
    )
    defaults.update(kwargs)
    return predict_solubilization(**defaults)


# ---------------------------------------------------------------------------
# 1. Return type
# ---------------------------------------------------------------------------


class TestReturnType:
    def test_returns_solubilization_result(self):
        result = _make_result()
        assert isinstance(result, SurfactantSolubilizationResult)

    def test_result_is_frozen(self):
        result = _make_result()
        with pytest.raises((AttributeError, TypeError)):
            result.fold_enhancement = 999.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# 2. fold_enhancement >= 1.0
# ---------------------------------------------------------------------------


class TestFoldEnhancement:
    def test_fold_enhancement_at_least_one(self):
        result = _make_result()
        assert result.fold_enhancement >= 1.0

    def test_fold_enhancement_at_least_one_below_cmc(self):
        # SDS CMC=8 mM; use 1 mM → below CMC
        result = _make_result(surfactant="SDS", surfactant_conc_mM=1.0)
        assert result.fold_enhancement >= 1.0

    def test_fold_enhancement_equals_s_total_over_s_aqueous(self):
        result = _make_result()
        expected = result.s_total_mg_mL / result.s_aqueous_mg_mL
        assert abs(result.fold_enhancement - expected) < 1e-9


# ---------------------------------------------------------------------------
# 3. Below CMC behaviour
# ---------------------------------------------------------------------------


class TestBelowCMC:
    def test_below_cmc_flag_false(self):
        # SDS CMC=8 mM; 1 mM < CMC
        result = _make_result(surfactant="SDS", surfactant_conc_mM=1.0)
        assert result.above_cmc is False

    def test_below_cmc_s_total_equals_s_aqueous(self):
        result = _make_result(surfactant="SDS", surfactant_conc_mM=1.0)
        assert abs(result.s_total_mg_mL - result.s_aqueous_mg_mL) < 1e-12

    def test_below_cmc_fold_enhancement_is_one(self):
        result = _make_result(surfactant="SDS", surfactant_conc_mM=1.0)
        assert abs(result.fold_enhancement - 1.0) < 1e-9


# ---------------------------------------------------------------------------
# 4. Above CMC behaviour
# ---------------------------------------------------------------------------


class TestAboveCMC:
    def test_above_cmc_flag_true(self):
        result = _make_result(surfactant="SDS", surfactant_conc_mM=20.0)
        assert result.above_cmc is True

    def test_above_cmc_s_total_gt_s_aqueous(self):
        result = _make_result(surfactant="SDS", surfactant_conc_mM=20.0)
        assert result.s_total_mg_mL > result.s_aqueous_mg_mL


# ---------------------------------------------------------------------------
# 5. Higher logP → higher fold_enhancement
# ---------------------------------------------------------------------------


class TestLogPEffect:
    def test_higher_logp_higher_fold_enhancement(self):
        r_low = _make_result(logP=0.0)
        r_high = _make_result(logP=4.0)
        assert r_high.fold_enhancement > r_low.fold_enhancement

    def test_negative_logp_has_lower_fold_than_positive(self):
        r_neg = _make_result(logP=-2.0)
        r_pos = _make_result(logP=3.0)
        assert r_pos.fold_enhancement > r_neg.fold_enhancement


# ---------------------------------------------------------------------------
# 6. Higher surfactant concentration → higher s_total
# ---------------------------------------------------------------------------


class TestConcentrationEffect:
    def test_higher_conc_higher_s_total(self):
        r_low = _make_result(surfactant_conc_mM=20.0)
        r_high = _make_result(surfactant_conc_mM=100.0)
        assert r_high.s_total_mg_mL > r_low.s_total_mg_mL


# ---------------------------------------------------------------------------
# 7. k_mw > 0
# ---------------------------------------------------------------------------


class TestKmw:
    def test_k_mw_positive(self):
        result = _make_result()
        assert result.k_mw > 0.0

    def test_k_mw_increases_with_logp(self):
        r_low = _make_result(logP=0.0)
        r_high = _make_result(logP=5.0)
        assert r_high.k_mw > r_low.k_mw


# ---------------------------------------------------------------------------
# 8. Solubilization efficiency valid
# ---------------------------------------------------------------------------


class TestSolubilizationEfficiency:
    def test_efficiency_valid_value(self):
        result = _make_result()
        assert result.solubilization_efficiency in _VALID_EFFICIENCIES

    def test_high_fold_high_efficiency(self):
        # Very high logP and high conc should give "high"
        result = _make_result(logP=8.0, surfactant_conc_mM=50.0)
        assert result.solubilization_efficiency == "high"

    def test_below_cmc_efficiency_none(self):
        # Below CMC → no enhancement → "none"
        result = _make_result(surfactant="SDS", surfactant_conc_mM=1.0)
        assert result.solubilization_efficiency == "none"


# ---------------------------------------------------------------------------
# 9. screen_surfactants
# ---------------------------------------------------------------------------


class TestScreenSurfactants:
    def test_returns_five_results(self):
        results = screen_surfactants("Drug", logP=2.0)
        assert len(results) == 5

    def test_all_surfactants_present(self):
        results = screen_surfactants("Drug", logP=2.0)
        surfs = {r.surfactant for r in results}
        assert surfs == set(_ALL_SURFACTANTS)

    def test_sorted_by_fold_enhancement_descending(self):
        results = screen_surfactants("Drug", logP=2.0)
        folds = [r.fold_enhancement for r in results]
        assert folds == sorted(folds, reverse=True)

    def test_all_results_are_correct_type(self):
        results = screen_surfactants("Drug", logP=2.0)
        for r in results:
            assert isinstance(r, SurfactantSolubilizationResult)


# ---------------------------------------------------------------------------
# 10. All 5 surfactants work individually
# ---------------------------------------------------------------------------


class TestAllSurfactants:
    @pytest.mark.parametrize("surf", _ALL_SURFACTANTS)
    def test_each_surfactant_runs(self, surf):
        result = predict_solubilization("Drug", surf, logP=2.0, surfactant_conc_mM=50.0)
        assert isinstance(result, SurfactantSolubilizationResult)

    @pytest.mark.parametrize("surf", _ALL_SURFACTANTS)
    def test_each_surfactant_has_correct_name(self, surf):
        result = predict_solubilization("Drug", surf, logP=2.0, surfactant_conc_mM=50.0)
        assert result.surfactant == surf


# ---------------------------------------------------------------------------
# 11. Validation errors
# ---------------------------------------------------------------------------


class TestValidation:
    def test_invalid_surfactant_raises(self):
        with pytest.raises(ValueError, match="surfactant"):
            predict_solubilization("Drug", "UnknownSurf", logP=2.0, surfactant_conc_mM=10.0)

    def test_logp_too_low_raises(self):
        with pytest.raises(ValueError):
            _make_result(logP=-6.0)

    def test_logp_too_high_raises(self):
        with pytest.raises(ValueError):
            _make_result(logP=11.0)

    def test_negative_conc_raises(self):
        with pytest.raises(ValueError):
            _make_result(surfactant_conc_mM=-1.0)

    def test_zero_conc_raises(self):
        with pytest.raises(ValueError):
            _make_result(surfactant_conc_mM=0.0)

    def test_zero_intrinsic_solubility_raises(self):
        with pytest.raises(ValueError):
            _make_result(intrinsic_solubility_mg_mL=0.0)

    def test_negative_intrinsic_solubility_raises(self):
        with pytest.raises(ValueError):
            _make_result(intrinsic_solubility_mg_mL=-0.1)


# ---------------------------------------------------------------------------
# 12. Notes nonempty
# ---------------------------------------------------------------------------


def test_notes_nonempty():
    result = _make_result()
    assert len(result.notes) > 0


# ---------------------------------------------------------------------------
# 13. CMC fields correct
# ---------------------------------------------------------------------------


class TestCMCFields:
    def test_sds_cmc_is_8(self):
        result = _make_result(surfactant="SDS")
        assert abs(result.cmc_mM - 8.0) < 1e-9

    def test_tween80_cmc_is_0_012(self):
        result = _make_result(surfactant="Tween_80", surfactant_conc_mM=0.1)
        assert abs(result.cmc_mM - 0.012) < 1e-9

    def test_pluronic_above_cmc_at_1mM(self):
        # Pluronic CMC=0.001 mM; 1 mM >> CMC
        result = _make_result(surfactant="Pluronic_F127", surfactant_conc_mM=1.0)
        assert result.above_cmc is True
