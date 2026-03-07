"""Tests for drug solubility enhancement strategies (Phase 296)."""

from __future__ import annotations

import math

import pytest

from omega_pbpk.biopharmaceutics.solubility_enhancement import (
    SolubilityComparisonResult,
    compare_enhancement_strategies,
    cosolvent_solubility,
    cyclodextrin_solubility,
    surfactant_solubilization,
)

# ---------------------------------------------------------------------------
# cyclodextrin_solubility
# ---------------------------------------------------------------------------


class TestCyclodextrinSolubility:
    def test_no_cd_returns_free_solubility(self):
        s = cyclodextrin_solubility(
            c_free_mg_mL=0.1, k11_M_inv=5000.0, cd_conc_mM=0.0, drug_mw=300.0
        )
        assert abs(s - 0.1) < 1e-9

    def test_positive_enhancement_with_cd(self):
        s_with = cyclodextrin_solubility(0.1, 5000.0, 10.0, 300.0)
        assert s_with > 0.1

    def test_higher_cd_higher_solubility(self):
        s_low = cyclodextrin_solubility(0.1, 5000.0, 10.0, 300.0)
        s_high = cyclodextrin_solubility(0.1, 5000.0, 50.0, 300.0)
        assert s_high > s_low

    def test_linear_with_cd_concentration(self):
        # S_total = S_free * (1 + K11 * [CD])
        # At [CD]=0.01 M (10 mM): S = 0.1 * (1 + 5000 * 0.01) = 0.1 * 51 = 5.1
        s = cyclodextrin_solubility(0.1, 5000.0, 10.0, 300.0)
        expected = 0.1 * (1.0 + 5000.0 * 0.01)
        assert abs(s - expected) < 1e-6

    def test_higher_k11_more_solubilization(self):
        s_low_k = cyclodextrin_solubility(0.1, 1000.0, 10.0, 300.0)
        s_high_k = cyclodextrin_solubility(0.1, 10000.0, 10.0, 300.0)
        assert s_high_k > s_low_k

    def test_invalid_free_solubility_raises(self):
        with pytest.raises(ValueError, match="c_free_mg_mL must be > 0"):
            cyclodextrin_solubility(0.0, 5000.0, 10.0, 300.0)

    def test_negative_cd_conc_raises(self):
        with pytest.raises(ValueError, match="cd_conc_mM must be >= 0"):
            cyclodextrin_solubility(0.1, 5000.0, -1.0, 300.0)

    def test_invalid_drug_mw_raises(self):
        with pytest.raises(ValueError, match="drug_mw must be > 0"):
            cyclodextrin_solubility(0.1, 5000.0, 10.0, 0.0)

    def test_custom_cd_mw(self):
        # Should not raise and should give valid result
        s = cyclodextrin_solubility(0.1, 5000.0, 10.0, 300.0, cd_mw=972.0)
        assert s > 0.1


# ---------------------------------------------------------------------------
# cosolvent_solubility
# ---------------------------------------------------------------------------


class TestCosolventSolubility:
    def test_zero_cosolvent_returns_water_solubility(self):
        s = cosolvent_solubility(0.5, logP=2.0, cosolvent_fraction=0.0)
        assert abs(s - 0.5) < 1e-9

    def test_solubility_increases_with_cosolvent(self):
        s0 = cosolvent_solubility(0.5, 2.0, 0.0)
        s30 = cosolvent_solubility(0.5, 2.0, 0.3)
        assert s30 > s0

    def test_log_linear_relationship(self):
        # log10(S_mix) = log10(0.5) + 3.0 * 0.3
        s = cosolvent_solubility(0.5, 2.0, 0.3, sigma_cosolvent=3.0)
        expected = 10.0 ** (math.log10(0.5) + 3.0 * 0.3)
        assert abs(s - expected) < 1e-9

    def test_higher_sigma_more_enhancement(self):
        s_low_sigma = cosolvent_solubility(0.1, 2.0, 0.3, sigma_cosolvent=2.0)
        s_high_sigma = cosolvent_solubility(0.1, 2.0, 0.3, sigma_cosolvent=4.0)
        assert s_high_sigma > s_low_sigma

    def test_invalid_water_solubility_raises(self):
        with pytest.raises(ValueError, match="solubility_water_mg_mL must be > 0"):
            cosolvent_solubility(0.0, 2.0, 0.3)

    def test_cosolvent_fraction_one_raises(self):
        with pytest.raises(ValueError, match="cosolvent_fraction must be in"):
            cosolvent_solubility(0.5, 2.0, 1.0)

    def test_negative_fraction_raises(self):
        with pytest.raises(ValueError):
            cosolvent_solubility(0.5, 2.0, -0.1)

    def test_peg400_sigma_2_5(self):
        # Lower sigma than ethanol → less enhancement at same fraction
        s_peg = cosolvent_solubility(0.1, 3.0, 0.3, sigma_cosolvent=2.5)
        s_etoh = cosolvent_solubility(0.1, 3.0, 0.3, sigma_cosolvent=3.0)
        assert s_etoh > s_peg


# ---------------------------------------------------------------------------
# surfactant_solubilization
# ---------------------------------------------------------------------------


class TestSurfactantSolubilization:
    def test_below_cmc_no_enhancement(self):
        # Surfactant below CMC: no micelles, S = S_water
        s = surfactant_solubilization(0.1, logP=2.0, surfactant_conc_mM=0.5, cmc_mM=1.0)
        assert abs(s - 0.1) < 1e-9

    def test_above_cmc_enhancement(self):
        s = surfactant_solubilization(0.1, logP=2.0, surfactant_conc_mM=5.0, cmc_mM=1.0)
        assert s > 0.1

    def test_higher_concentration_more_solubilization(self):
        s_low = surfactant_solubilization(0.1, 2.0, 5.0)
        s_high = surfactant_solubilization(0.1, 2.0, 20.0)
        assert s_high > s_low

    def test_high_logP_more_solubilization(self):
        s_low_logP = surfactant_solubilization(0.1, logP=1.0, surfactant_conc_mM=10.0)
        s_high_logP = surfactant_solubilization(0.1, logP=4.0, surfactant_conc_mM=10.0)
        assert s_high_logP > s_low_logP

    def test_at_cmc_no_enhancement(self):
        # At exactly CMC: [micelles] = 0, S_total = S_water
        s = surfactant_solubilization(0.5, 2.0, surfactant_conc_mM=1.0, cmc_mM=1.0)
        assert abs(s - 0.5) < 1e-9

    def test_invalid_solubility_raises(self):
        with pytest.raises(ValueError, match="solubility_water_mg_mL must be > 0"):
            surfactant_solubilization(0.0, 2.0, 5.0)

    def test_zero_cmc_raises(self):
        with pytest.raises(ValueError, match="cmc_mM must be > 0"):
            surfactant_solubilization(0.1, 2.0, 5.0, cmc_mM=0.0)

    def test_negative_ksp_raises(self):
        with pytest.raises(ValueError, match="ksp must be > 0"):
            surfactant_solubilization(0.1, 2.0, 5.0, ksp=-1.0)

    def test_negative_surfactant_raises(self):
        with pytest.raises(ValueError, match="surfactant_conc_mM must be >= 0"):
            surfactant_solubilization(0.1, 2.0, -1.0)

    def test_ksp_scales_enhancement(self):
        s_low_ksp = surfactant_solubilization(0.1, 2.0, 10.0, ksp=50.0)
        s_high_ksp = surfactant_solubilization(0.1, 2.0, 10.0, ksp=200.0)
        assert s_high_ksp > s_low_ksp


# ---------------------------------------------------------------------------
# compare_enhancement_strategies
# ---------------------------------------------------------------------------


class TestCompareEnhancementStrategies:
    def _run(self, **kw):
        defaults = dict(
            drug_name="DrugX",
            solubility_water_mg_mL=0.01,
            logP=3.5,
            drug_mw=450.0,
        )
        defaults.update(kw)
        return compare_enhancement_strategies(**defaults)

    def test_returns_result_type(self):
        assert isinstance(self._run(), SolubilityComparisonResult)

    def test_frozen_dataclass(self):
        r = self._run()
        with pytest.raises(Exception):
            r.drug_name = "other"  # type: ignore[misc]

    def test_has_7_strategies(self):
        r = self._run()
        assert len(r.strategies) == 7

    def test_water_strategy_equals_intrinsic(self):
        r = self._run(solubility_water_mg_mL=0.05)
        assert abs(r.strategies["water"] - 0.05) < 1e-9

    def test_all_strategies_positive(self):
        r = self._run()
        for k, v in r.strategies.items():
            assert v > 0.0, f"Strategy {k} is non-positive"

    def test_cd_50mM_better_than_cd_10mM(self):
        r = self._run()
        assert r.strategies["cd_50mM"] > r.strategies["cd_10mM"]

    def test_cosolvent_30pct_better_than_10pct(self):
        r = self._run()
        assert r.strategies["cosolvent_30pct"] > r.strategies["cosolvent_10pct"]

    def test_surfactant_20mM_better_than_5mM(self):
        r = self._run()
        assert r.strategies["surfactant_20mM"] > r.strategies["surfactant_5mM"]

    def test_max_fold_is_positive(self):
        r = self._run()
        assert r.max_enhancement_fold > 1.0

    def test_max_strategy_not_water(self):
        r = self._run()
        assert r.max_enhancement_strategy != "water"

    def test_max_strategy_gives_max_solubility(self):
        r = self._run()
        non_water = {k: v for k, v in r.strategies.items() if k != "water"}
        expected_max = max(non_water, key=lambda k: non_water[k])
        assert r.max_enhancement_strategy == expected_max

    def test_max_fold_consistent_with_strategies(self):
        r = self._run()
        max_sol = r.strategies[r.max_enhancement_strategy]
        expected_fold = max_sol / r.solubility_water
        assert abs(r.max_enhancement_fold - expected_fold) < 1e-3

    def test_invalid_solubility_raises(self):
        with pytest.raises(ValueError, match="solubility_water_mg_mL must be > 0"):
            compare_enhancement_strategies("X", 0.0, 2.0, 300.0)

    def test_invalid_drug_mw_raises(self):
        with pytest.raises(ValueError, match="drug_mw must be > 0"):
            compare_enhancement_strategies("X", 0.1, 2.0, 0.0)

    def test_notes_contains_logP(self):
        r = self._run(logP=3.5)
        assert "3.5" in r.notes or "logP" in r.notes

    def test_high_logP_surfactant_dominant(self):
        # Very lipophilic drug: surfactant should be very effective
        r = compare_enhancement_strategies("Lipophilic", 0.001, 5.0, 400.0)
        # surfactant_20mM should be substantial
        assert r.strategies["surfactant_20mM"] > r.strategies["water"] * 2
