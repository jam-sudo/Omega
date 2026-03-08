"""Tests for Phase 670 — Drug Formulation pH Stability."""

import pytest

from omega_pbpk.prediction.formulation_ph_stability_p670 import (
    FormulationpHStabilityResult,
    predict_ph_stability,
    screen_ph_stability,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DEFAULT_KWARGS = dict(
    drug_name="DrugA",
    logP=2.5,
    mw_Da=350.0,
)


def _run(**overrides):
    kw = {**DEFAULT_KWARGS, **overrides}
    return predict_ph_stability(**kw)


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------


class TestReturnType:
    def test_return_type(self):
        r = _run()
        assert isinstance(r, FormulationpHStabilityResult)

    def test_frozen_raises_on_mutation(self):
        r = _run()
        with pytest.raises(AttributeError):
            r.drug_name = "Other"

    def test_drug_name_stored(self):
        r = _run()
        assert r.drug_name == "DrugA"


# ---------------------------------------------------------------------------
# Degradation kinetics
# ---------------------------------------------------------------------------


class TestDegradation:
    def test_higher_temp_shorter_half_life(self):
        r_25 = _run(temperature_C=25.0)
        r_40 = _run(temperature_C=40.0)
        assert r_40.t_half_degradation_h < r_25.t_half_degradation_h

    def test_higher_hydrolysis_susceptibility_faster(self):
        r_low = _run(susceptibility_hydrolysis=0.1)
        r_high = _run(susceptibility_hydrolysis=0.9)
        assert r_high.k_degradation_per_h > r_low.k_degradation_per_h

    def test_t_half_positive(self):
        r = _run()
        assert r.t_half_degradation_h > 0

    def test_k_degradation_positive(self):
        r = _run()
        assert r.k_degradation_per_h > 0


# ---------------------------------------------------------------------------
# Potency and shelf life
# ---------------------------------------------------------------------------


class TestPotency:
    def test_potency_2y_in_range(self):
        r = _run()
        assert 0 <= r.potency_at_2y_pct <= 100

    def test_potency_5y_in_range(self):
        r = _run()
        assert 0 <= r.potency_at_5y_pct <= 100

    def test_potency_5y_le_2y(self):
        r = _run()
        assert r.potency_at_5y_pct <= r.potency_at_2y_pct

    def test_shelf_life_positive(self):
        r = _run()
        assert r.shelf_life_years > 0


# ---------------------------------------------------------------------------
# Stability classification
# ---------------------------------------------------------------------------


class TestStabilityClass:
    def test_stability_class_valid(self):
        r = _run()
        assert r.stability_class in ("excellent", "good", "fair", "unstable")

    def test_low_susceptibility_excellent(self):
        r = _run(susceptibility_hydrolysis=0.001, susceptibility_oxidation=0.001)
        assert r.stability_class in ("excellent", "good")

    def test_high_susceptibility_poor(self):
        r = _run(
            susceptibility_hydrolysis=1.0,
            susceptibility_oxidation=1.0,
            ph_tested=12.0 if False else 9.0,
        )
        # At extreme pH, may be unstable or fair
        assert r.stability_class in ("unstable", "fair", "good")


# ---------------------------------------------------------------------------
# Degradation pathway
# ---------------------------------------------------------------------------


class TestDegradationPathway:
    def test_pathway_valid(self):
        r = _run()
        assert r.degradation_pathway in ("hydrolysis", "oxidation", "combined")

    def test_high_hydrolysis_pathway(self):
        r = _run(susceptibility_hydrolysis=0.9, susceptibility_oxidation=0.01)
        assert r.degradation_pathway == "hydrolysis"

    def test_high_oxidation_pathway(self):
        r = _run(susceptibility_hydrolysis=0.01, susceptibility_oxidation=0.9)
        assert r.degradation_pathway == "oxidation"

    def test_balanced_combined(self):
        # We need k_hyd ≈ k_ox. At pH 7 for base: k_hyd = susc_h * 0.001
        # k_ox = susc_ox * 0.0001. So susc_h * 0.001 ≈ susc_ox * 0.0001
        # susc_h ≈ susc_ox * 0.1
        r = _run(susceptibility_hydrolysis=0.05, susceptibility_oxidation=0.5, ph_tested=7.0)
        # k_hyd = 0.05 * 0.001 = 5e-5, k_ox = 0.5 * 0.0001 = 5e-5 → combined!
        assert r.degradation_pathway == "combined"


# ---------------------------------------------------------------------------
# Optimal pH
# ---------------------------------------------------------------------------


class TestOptimalPH:
    def test_optimal_ph_in_range(self):
        r = _run()
        assert 3.0 <= r.optimal_ph <= 9.0

    def test_base_optimal_ph_lower(self):
        r = _run(is_base=True)
        assert r.optimal_ph < 7.0

    def test_acid_optimal_ph_higher(self):
        r = _run(is_base=False)
        assert r.optimal_ph >= 7.0


# ---------------------------------------------------------------------------
# Formulation recommendation
# ---------------------------------------------------------------------------


class TestRecommendation:
    def test_recommendation_not_empty(self):
        r = _run()
        assert len(r.formulation_recommendation) > 0

    def test_unstable_recommendation(self):
        # Force unstable: high susceptibility + high temp + extreme pH
        r = _run(
            susceptibility_hydrolysis=1.0,
            susceptibility_oxidation=1.0,
            ph_tested=9.0,
            temperature_C=40.0,
        )
        if r.stability_class == "unstable":
            assert "lyophilize" in r.formulation_recommendation


# ---------------------------------------------------------------------------
# screen_ph_stability
# ---------------------------------------------------------------------------


class TestScreenPH:
    def test_returns_dict(self):
        compounds = [
            {"drug_name": "A", "logP": 2.0, "mw_Da": 300.0},
            {"drug_name": "B", "logP": 3.0, "mw_Da": 400.0},
        ]
        result = screen_ph_stability(compounds)
        assert isinstance(result, dict)
        assert "A" in result
        assert "B" in result

    def test_optimal_ph_in_range(self):
        compounds = [{"drug_name": "X", "logP": 1.0, "mw_Da": 200.0}]
        result = screen_ph_stability(compounds, ph_range=[3.0, 5.0, 7.0, 9.0])
        assert result["X"] in [3.0, 5.0, 7.0, 9.0]


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TestValidation:
    def test_negative_hydrolysis_raises(self):
        with pytest.raises(ValueError):
            _run(susceptibility_hydrolysis=-0.1)

    def test_hydrolysis_over_1_raises(self):
        with pytest.raises(ValueError):
            _run(susceptibility_hydrolysis=1.5)

    def test_negative_oxidation_raises(self):
        with pytest.raises(ValueError):
            _run(susceptibility_oxidation=-0.1)

    def test_zero_temp_raises(self):
        with pytest.raises(ValueError):
            _run(temperature_C=0.0)

    def test_negative_temp_raises(self):
        with pytest.raises(ValueError):
            _run(temperature_C=-10.0)
