"""Tests for Phase 379 — Drug Solubility Enhancement Predictor."""

import pytest

from omega_pbpk.prediction.solubility_enhancement_predictor import (
    SolubilityEnhancementResult,
    compare_strategies,
    predict_solubility_enhancement,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _base_kwargs(**overrides):
    """Return minimal valid kwargs for predict_solubility_enhancement."""
    kwargs = dict(
        drug_name="TestDrug",
        intrinsic_solubility_mg_mL=0.01,
        logp=3.0,
        pka=8.0,
        mw_da=400.0,
        strategy="parent",
    )
    kwargs.update(overrides)
    return kwargs


# ---------------------------------------------------------------------------
# Return type tests
# ---------------------------------------------------------------------------


class TestReturnType:
    def test_returns_result_instance(self):
        result = predict_solubility_enhancement(**_base_kwargs())
        assert isinstance(result, SolubilityEnhancementResult)

    def test_result_is_frozen(self):
        result = predict_solubility_enhancement(**_base_kwargs())
        with pytest.raises((AttributeError, TypeError)):
            result.drug_name = "Changed"  # type: ignore[misc]

    def test_all_fields_present(self):
        result = predict_solubility_enhancement(**_base_kwargs())
        assert hasattr(result, "drug_name")
        assert hasattr(result, "strategy")
        assert hasattr(result, "intrinsic_solubility_mg_mL")
        assert hasattr(result, "enhanced_solubility_mg_mL")
        assert hasattr(result, "enhancement_fold")
        assert hasattr(result, "predicted_bioavailability_pct")
        assert hasattr(result, "physical_stability_score")
        assert hasattr(result, "manufacturing_complexity")
        assert hasattr(result, "notes")


# ---------------------------------------------------------------------------
# Parent strategy
# ---------------------------------------------------------------------------


class TestParentStrategy:
    def test_parent_fold_is_one(self):
        result = predict_solubility_enhancement(**_base_kwargs(strategy="parent"))
        assert result.enhancement_fold == pytest.approx(1.0)

    def test_parent_enhanced_equals_intrinsic(self):
        sol = 0.05
        result = predict_solubility_enhancement(
            **_base_kwargs(intrinsic_solubility_mg_mL=sol, strategy="parent")
        )
        assert result.enhanced_solubility_mg_mL == pytest.approx(sol)

    def test_parent_stability_score(self):
        result = predict_solubility_enhancement(**_base_kwargs(strategy="parent"))
        assert result.physical_stability_score == pytest.approx(90.0)

    def test_parent_manufacturing_complexity(self):
        result = predict_solubility_enhancement(**_base_kwargs(strategy="parent"))
        assert result.manufacturing_complexity == "low"


# ---------------------------------------------------------------------------
# Salt strategy
# ---------------------------------------------------------------------------


class TestSaltStrategy:
    def test_salt_fold_ionizable_drug(self):
        # pKa=5 → abs(5-7)=2 → fold = 10^(0.5*2) = 10
        result = predict_solubility_enhancement(**_base_kwargs(pka=5.0, strategy="salt"))
        expected = 10.0 ** (0.5 * abs(5.0 - 7.0))
        assert result.enhancement_fold == pytest.approx(expected, rel=1e-6)

    def test_salt_fold_neutral_drug_pka_out_of_range(self):
        # pKa=1.0 (outside [2,12]) → fold = 1.5
        result = predict_solubility_enhancement(**_base_kwargs(pka=1.0, strategy="salt"))
        assert result.enhancement_fold == pytest.approx(1.5)

    def test_salt_fold_capped_at_100(self):
        # pKa=2 → abs(2-7)=5 → 10^2.5 = 316 → capped at 100
        result = predict_solubility_enhancement(**_base_kwargs(pka=2.0, strategy="salt"))
        assert result.enhancement_fold <= 100.0 + 1e-9

    def test_salt_stability_score(self):
        result = predict_solubility_enhancement(**_base_kwargs(strategy="salt"))
        assert result.physical_stability_score == pytest.approx(75.0)

    def test_salt_manufacturing_complexity(self):
        result = predict_solubility_enhancement(**_base_kwargs(strategy="salt"))
        assert result.manufacturing_complexity == "low"


# ---------------------------------------------------------------------------
# Amorphous strategy
# ---------------------------------------------------------------------------


class TestAmorphousStrategy:
    def test_amorphous_fold_greater_than_parent_for_lipophilic(self):
        """For logP=3, amorphous fold should be > 1."""
        result_amo = predict_solubility_enhancement(**_base_kwargs(strategy="amorphous"))
        result_par = predict_solubility_enhancement(**_base_kwargs(strategy="parent"))
        assert result_amo.enhancement_fold > result_par.enhancement_fold

    def test_amorphous_min_fold_is_two(self):
        # logP=-5 → 10^(-1) = 0.1 → clamped to max(2.0, ...) = 2.0
        result = predict_solubility_enhancement(**_base_kwargs(logp=-5.0, strategy="amorphous"))
        assert result.enhancement_fold >= 2.0

    def test_amorphous_fold_capped_at_1000(self):
        # logP=10 → 10^2 = 100; well below 1000, but test cap doesn't break
        result = predict_solubility_enhancement(**_base_kwargs(logp=10.0, strategy="amorphous"))
        assert result.enhancement_fold <= 1000.0 + 1e-9

    def test_amorphous_stability_score(self):
        result = predict_solubility_enhancement(**_base_kwargs(strategy="amorphous"))
        assert result.physical_stability_score == pytest.approx(40.0)

    def test_amorphous_manufacturing_complexity_high(self):
        result = predict_solubility_enhancement(**_base_kwargs(strategy="amorphous"))
        assert result.manufacturing_complexity == "high"


# ---------------------------------------------------------------------------
# Cyclodextrin strategy
# ---------------------------------------------------------------------------


class TestCyclodextrinStrategy:
    def test_cyclodextrin_lipophilic_higher_than_hydrophilic(self):
        result_lipo = predict_solubility_enhancement(
            **_base_kwargs(logp=4.0, strategy="cyclodextrin")
        )
        result_hydro = predict_solubility_enhancement(
            **_base_kwargs(logp=-1.0, strategy="cyclodextrin")
        )
        assert result_lipo.enhancement_fold > result_hydro.enhancement_fold

    def test_cyclodextrin_negative_logp_fold_is_1_5(self):
        result = predict_solubility_enhancement(**_base_kwargs(logp=-2.0, strategy="cyclodextrin"))
        assert result.enhancement_fold == pytest.approx(1.5)

    def test_cyclodextrin_fold_capped_at_100(self):
        result = predict_solubility_enhancement(**_base_kwargs(logp=10.0, strategy="cyclodextrin"))
        assert result.enhancement_fold <= 100.0 + 1e-9

    def test_cyclodextrin_stability_score(self):
        result = predict_solubility_enhancement(**_base_kwargs(strategy="cyclodextrin"))
        assert result.physical_stability_score == pytest.approx(80.0)


# ---------------------------------------------------------------------------
# Cocrystal strategy
# ---------------------------------------------------------------------------


class TestCocrystalStrategy:
    def test_cocrystal_fold_formula(self):
        # fold = max(1.5, 2.0 + logp*0.5) capped at 50
        logp = 3.0
        expected = min(50.0, max(1.5, 2.0 + logp * 0.5))
        result = predict_solubility_enhancement(**_base_kwargs(logp=logp, strategy="cocrystal"))
        assert result.enhancement_fold == pytest.approx(expected, rel=1e-6)

    def test_cocrystal_min_fold(self):
        # logP=-10 (outside valid range), but within [-5,10]: logP=-5 → 2.0+(-5)*0.5=-0.5 → max(1.5,−0.5)=1.5
        result = predict_solubility_enhancement(**_base_kwargs(logp=-5.0, strategy="cocrystal"))
        assert result.enhancement_fold >= 1.5 - 1e-9

    def test_cocrystal_stability_score(self):
        result = predict_solubility_enhancement(**_base_kwargs(strategy="cocrystal"))
        assert result.physical_stability_score == pytest.approx(70.0)

    def test_cocrystal_manufacturing_medium(self):
        result = predict_solubility_enhancement(**_base_kwargs(strategy="cocrystal"))
        assert result.manufacturing_complexity == "medium"


# ---------------------------------------------------------------------------
# Nano strategy
# ---------------------------------------------------------------------------


class TestNanoStrategy:
    def test_nano_fold_proportional_to_mw(self):
        result_low = predict_solubility_enhancement(**_base_kwargs(mw_da=200.0, strategy="nano"))
        result_high = predict_solubility_enhancement(**_base_kwargs(mw_da=800.0, strategy="nano"))
        assert result_high.enhancement_fold > result_low.enhancement_fold

    def test_nano_fold_formula(self):
        mw = 400.0
        expected = min(50.0, max(2.0, mw / 200.0))
        result = predict_solubility_enhancement(**_base_kwargs(mw_da=mw, strategy="nano"))
        assert result.enhancement_fold == pytest.approx(expected, rel=1e-6)

    def test_nano_fold_min_two(self):
        result = predict_solubility_enhancement(**_base_kwargs(mw_da=50.0, strategy="nano"))
        assert result.enhancement_fold >= 2.0

    def test_nano_fold_capped_at_50(self):
        result = predict_solubility_enhancement(**_base_kwargs(mw_da=20000.0, strategy="nano"))
        assert result.enhancement_fold <= 50.0 + 1e-9

    def test_nano_stability_score(self):
        result = predict_solubility_enhancement(**_base_kwargs(strategy="nano"))
        assert result.physical_stability_score == pytest.approx(55.0)


# ---------------------------------------------------------------------------
# Bioavailability
# ---------------------------------------------------------------------------


class TestBioavailability:
    def test_bioavailability_range(self):
        for strat in ["parent", "salt", "cocrystal", "amorphous", "cyclodextrin", "nano"]:
            result = predict_solubility_enhancement(**_base_kwargs(strategy=strat))
            assert 0.0 <= result.predicted_bioavailability_pct <= 100.0

    def test_high_solubility_gives_high_bioavailability(self):
        # enhanced_solubility = 10 mg/mL >> 1 mg/mL reference → ba = 100%
        result = predict_solubility_enhancement(
            **_base_kwargs(intrinsic_solubility_mg_mL=10.0, strategy="parent")
        )
        assert result.predicted_bioavailability_pct == pytest.approx(100.0)


# ---------------------------------------------------------------------------
# compare_strategies
# ---------------------------------------------------------------------------


class TestCompareStrategies:
    def test_compare_returns_list(self):
        results = compare_strategies("Drug", 0.01, 3.0, 7.0, 400.0)
        assert isinstance(results, list)

    def test_compare_returns_six_strategies(self):
        results = compare_strategies("Drug", 0.01, 3.0, 7.0, 400.0)
        assert len(results) == 6

    def test_compare_sorted_descending(self):
        results = compare_strategies("Drug", 0.01, 3.0, 7.0, 400.0)
        folds = [r.enhancement_fold for r in results]
        assert folds == sorted(folds, reverse=True)

    def test_compare_all_are_result_instances(self):
        results = compare_strategies("Drug", 0.01, 3.0, 7.0, 400.0)
        for r in results:
            assert isinstance(r, SolubilityEnhancementResult)


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


class TestValidation:
    def test_invalid_intrinsic_solubility(self):
        with pytest.raises(ValueError, match="intrinsic_solubility"):
            predict_solubility_enhancement(**_base_kwargs(intrinsic_solubility_mg_mL=-0.1))

    def test_invalid_logp_too_low(self):
        with pytest.raises(ValueError, match="logp"):
            predict_solubility_enhancement(**_base_kwargs(logp=-6.0))

    def test_invalid_logp_too_high(self):
        with pytest.raises(ValueError, match="logp"):
            predict_solubility_enhancement(**_base_kwargs(logp=11.0))

    def test_invalid_pka(self):
        with pytest.raises(ValueError, match="pka"):
            predict_solubility_enhancement(**_base_kwargs(pka=15.0))

    def test_invalid_mw(self):
        with pytest.raises(ValueError, match="mw_da"):
            predict_solubility_enhancement(**_base_kwargs(mw_da=0.0))

    def test_invalid_strategy(self):
        with pytest.raises(ValueError, match="strategy"):
            predict_solubility_enhancement(**_base_kwargs(strategy="spray_drying"))
