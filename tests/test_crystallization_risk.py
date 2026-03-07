"""Tests for biopharmaceutics.crystallization_risk — Phase 253."""

from __future__ import annotations

import math
import pytest

from omega_pbpk.biopharmaceutics.crystallization_risk import (
    ASDStabilityResult,
    CrystTendencyResult,
    assess_asd_stability,
    crystallization_tendency,
    induction_time,
)


# ---------------------------------------------------------------------------
# crystallization_tendency tests
# ---------------------------------------------------------------------------


class TestCrystallizationTendency:
    """Tests for crystallization_tendency()."""

    def test_low_tg_tm_ratio_gives_high_tendency(self):
        """Tg/Tm < 0.7 in Kelvin -> high crystallization tendency."""
        # Tg=50C (323K), Tm=230C (503K) => Tg/Tm ≈ 0.642 < 0.7
        result = crystallization_tendency(logP=2.0, mw=350.0, tm_C=230.0, tg_C=50.0)
        assert result.crystallization_tendency == "high"

    def test_high_tg_tm_ratio_gives_low_tendency(self):
        """Tg/Tm > 0.8 -> low crystallization tendency."""
        # Tg=150C (423K), Tm=200C (473K) => Tg/Tm ≈ 0.894 > 0.8
        result = crystallization_tendency(logP=1.5, mw=300.0, tm_C=200.0, tg_C=150.0)
        assert result.crystallization_tendency == "low"

    def test_moderate_tg_tm_ratio_gives_moderate_tendency(self):
        """Tg/Tm in [0.7, 0.8] -> moderate tendency."""
        # Tg=95C (368K), Tm=160C (433K) => ratio ≈ 0.85 - check boundary
        # Let's pick Tg=80C (353K), Tm=160C (433K) => 353/433 ≈ 0.815 -- low
        # Tg=70C (343K), Tm=185C (458K) => 343/458 ≈ 0.749 -> moderate
        result = crystallization_tendency(logP=3.0, mw=400.0, tm_C=185.0, tg_C=70.0)
        assert result.crystallization_tendency == "moderate"

    def test_tg_tm_ratio_value(self):
        """tg_tm_ratio is Tg_K / Tm_K."""
        result = crystallization_tendency(logP=2.0, mw=300.0, tm_C=200.0, tg_C=100.0)
        expected_ratio = (100.0 + 273.15) / (200.0 + 273.15)
        assert abs(result.tg_tm_ratio - expected_ratio) < 1e-6

    def test_tg_tm_ratio_in_zero_one(self):
        """tg_tm_ratio must be in (0, 1)."""
        result = crystallization_tendency(logP=2.0, mw=400.0, tm_C=250.0, tg_C=80.0)
        assert 0.0 < result.tg_tm_ratio < 1.0

    def test_fragility_m_formula(self):
        """fragility_m = 17.7 + 0.0017 * mw."""
        mw = 400.0
        result = crystallization_tendency(logP=2.0, mw=mw, tm_C=200.0, tg_C=100.0)
        expected_m = 17.7 + 0.0017 * mw
        assert abs(result.fragility_m - expected_m) < 1e-9

    def test_result_fields_types(self):
        """Result fields have correct types."""
        result = crystallization_tendency(logP=2.5, mw=350.0, tm_C=210.0, tg_C=90.0)
        assert isinstance(result, CrystTendencyResult)
        assert isinstance(result.logP, float)
        assert isinstance(result.mw, float)
        assert isinstance(result.tm_C, float)
        assert isinstance(result.tg_C, float)
        assert isinstance(result.tg_tm_ratio, float)
        assert isinstance(result.fragility_m, float)
        assert isinstance(result.crystallization_tendency, str)
        assert isinstance(result.notes, str)

    def test_validation_mw_zero(self):
        with pytest.raises(ValueError, match="mw"):
            crystallization_tendency(logP=2.0, mw=0.0, tm_C=200.0, tg_C=100.0)

    def test_validation_tm_zero(self):
        with pytest.raises(ValueError, match="tm_C"):
            crystallization_tendency(logP=2.0, mw=300.0, tm_C=0.0, tg_C=50.0)

    def test_validation_tg_zero(self):
        with pytest.raises(ValueError, match="tg_C"):
            crystallization_tendency(logP=2.0, mw=300.0, tm_C=200.0, tg_C=0.0)

    def test_validation_tg_ge_tm(self):
        """tg_C must be strictly less than tm_C."""
        with pytest.raises(ValueError):
            crystallization_tendency(logP=2.0, mw=300.0, tm_C=100.0, tg_C=100.0)

    def test_notes_not_empty(self):
        result = crystallization_tendency(logP=3.0, mw=400.0, tm_C=200.0, tg_C=80.0)
        assert len(result.notes) > 10


# ---------------------------------------------------------------------------
# induction_time tests
# ---------------------------------------------------------------------------


class TestInductionTime:
    """Tests for induction_time()."""

    def test_higher_supersaturation_shorter_induction(self):
        """Higher S -> shorter induction time."""
        t_low_S = induction_time(solubility_crystal_mg_mL=0.1, c_supersaturated_mg_mL=1.0)
        t_high_S = induction_time(solubility_crystal_mg_mL=0.1, c_supersaturated_mg_mL=10.0)
        assert t_high_S < t_low_S

    def test_no_supersaturation_returns_large_value(self):
        """S = 1 (saturated, not supersaturated) -> very large induction time."""
        t = induction_time(solubility_crystal_mg_mL=1.0, c_supersaturated_mg_mL=1.0)
        assert t >= 1e5

    def test_subsaturated_returns_large_value(self):
        """S < 1 (undersaturated) -> effectively infinite induction time."""
        t = induction_time(solubility_crystal_mg_mL=1.0, c_supersaturated_mg_mL=0.5)
        assert t >= 1e5

    def test_induction_time_formula(self):
        """t_ind = A / log(S)^2 with A=10 h."""
        sol = 0.1
        c_sup = 1.0  # S = 10
        S = 10.0
        expected = 10.0 / (math.log(S) ** 2)
        result = induction_time(sol, c_sup, temperature_C=37.0)
        assert abs(result - expected) < 1e-9

    def test_positive_induction_time(self):
        """Induction time must be positive for S > 1."""
        t = induction_time(0.01, 1.0)
        assert t > 0.0

    def test_validation_solubility_zero(self):
        with pytest.raises(ValueError, match="solubility"):
            induction_time(0.0, 1.0)

    def test_validation_concentration_zero(self):
        with pytest.raises(ValueError, match="c_supersaturated"):
            induction_time(1.0, 0.0)

    def test_induction_time_hours_reasonable(self):
        """For moderate supersaturation (S=5), induction time should be a few hours."""
        t = induction_time(0.1, 0.5)  # S=5
        # t = 10 / log(5)^2 ≈ 10 / (1.609)^2 ≈ 3.86 h
        assert 0.5 < t < 50.0


# ---------------------------------------------------------------------------
# assess_asd_stability tests
# ---------------------------------------------------------------------------


class TestAssessAsdStability:
    """Tests for assess_asd_stability()."""

    def test_hpmc_raises_tg(self):
        """HPMC provides +15C Tg boost -> lower risk than no polymer."""
        result_hpmc = assess_asd_stability(
            "DrugA",
            logP=3.0,
            mw=400.0,
            tm_C=220.0,
            tg_C=60.0,
            polymer_type="HPMC",
            humidity_pct=40.0,
            temperature_C=25.0,
        )
        # Raw Tg=60C, HPMC adds 15C -> effective 75C
        assert result_hpmc.tg_effective_C == pytest.approx(75.0, abs=1e-6)

    def test_pvp_raises_tg(self):
        """PVP provides +10C Tg boost."""
        result = assess_asd_stability(
            "DrugB",
            logP=2.0,
            mw=350.0,
            tm_C=200.0,
            tg_C=70.0,
            polymer_type="PVP",
            humidity_pct=40.0,
            temperature_C=25.0,
        )
        assert result.tg_effective_C == pytest.approx(80.0, abs=1e-6)

    def test_high_humidity_lowers_tg(self):
        """Higher humidity -> lower effective Tg -> worse stability."""
        r_low_rh = assess_asd_stability(
            "DrugC",
            logP=2.0,
            mw=350.0,
            tm_C=200.0,
            tg_C=80.0,
            polymer_type="HPMC",
            humidity_pct=40.0,
            temperature_C=25.0,
        )
        r_high_rh = assess_asd_stability(
            "DrugC",
            logP=2.0,
            mw=350.0,
            tm_C=200.0,
            tg_C=80.0,
            polymer_type="HPMC",
            humidity_pct=80.0,
            temperature_C=25.0,
        )
        assert r_high_rh.tg_effective_C < r_low_rh.tg_effective_C

    def test_high_humidity_increases_risk(self):
        """Very high humidity can push risk to 'high'."""
        r_high = assess_asd_stability(
            "DrugD",
            logP=3.0,
            mw=500.0,
            tm_C=300.0,
            tg_C=30.0,
            polymer_type="PVP",
            humidity_pct=100.0,
            temperature_C=25.0,
        )
        # tg_effective = 30 + 10 - (60/10)*2 = 40-12 = 28C -> tg_K/tm_K very low
        assert r_high.crystallization_risk == "high"

    def test_tg_tm_ratio_in_zero_one(self):
        """tg_tm_ratio_effective must be in (0, 1]."""
        result = assess_asd_stability(
            "DrugE",
            logP=2.0,
            mw=300.0,
            tm_C=200.0,
            tg_C=90.0,
            polymer_type="HPMC",
            humidity_pct=40.0,
            temperature_C=25.0,
        )
        assert 0.0 < result.tg_tm_ratio_effective <= 1.0

    def test_estimated_shelf_life_positive(self):
        """estimated_shelf_life_months must be > 0."""
        result = assess_asd_stability(
            "DrugF",
            logP=2.0,
            mw=400.0,
            tm_C=200.0,
            tg_C=60.0,
            polymer_type="PVPVA",
            humidity_pct=40.0,
            temperature_C=25.0,
        )
        assert result.estimated_shelf_life_months > 0

    def test_low_risk_high_shelf_life(self):
        """Very high effective Tg should give 24+ months shelf life."""
        # Tg=160C + HPMC15 - 0 humidity penalty = 175C; Tg margin above 25C = 150C
        result = assess_asd_stability(
            "DrugG",
            logP=2.0,
            mw=300.0,
            tm_C=220.0,
            tg_C=160.0,
            polymer_type="HPMC",
            humidity_pct=40.0,
            temperature_C=25.0,
        )
        assert result.estimated_shelf_life_months >= 24.0

    def test_risk_categories_valid(self):
        """Risk category must be one of the valid values."""
        result = assess_asd_stability(
            "DrugH",
            logP=3.0,
            mw=400.0,
            tm_C=200.0,
            tg_C=80.0,
            polymer_type="HPMC",
            humidity_pct=50.0,
            temperature_C=25.0,
        )
        assert result.crystallization_risk in {"low", "moderate", "high"}

    def test_result_type(self):
        """Result is ASDStabilityResult."""
        result = assess_asd_stability(
            "DrugI",
            logP=2.5,
            mw=350.0,
            tm_C=210.0,
            tg_C=85.0,
            polymer_type="copovidone",
            humidity_pct=40.0,
            temperature_C=25.0,
        )
        assert isinstance(result, ASDStabilityResult)

    def test_validation_humidity_out_of_range(self):
        with pytest.raises(ValueError, match="humidity"):
            assess_asd_stability(
                "DrugJ", logP=2.0, mw=300.0, tm_C=200.0, tg_C=80.0, humidity_pct=110.0
            )

    def test_validation_tg_ge_tm(self):
        with pytest.raises(ValueError):
            assess_asd_stability("DrugK", logP=2.0, mw=300.0, tm_C=100.0, tg_C=100.0)

    def test_validation_invalid_polymer(self):
        with pytest.raises(ValueError, match="polymer_type"):
            assess_asd_stability(
                "DrugL", logP=2.0, mw=300.0, tm_C=200.0, tg_C=80.0, polymer_type="INVALID_POLYMER"
            )

    def test_recommendation_not_empty(self):
        result = assess_asd_stability(
            "DrugM",
            logP=2.0,
            mw=300.0,
            tm_C=200.0,
            tg_C=80.0,
            polymer_type="HPMC",
            humidity_pct=40.0,
            temperature_C=25.0,
        )
        assert len(result.recommendation) > 10

    def test_field_types(self):
        """Check all result field types."""
        result = assess_asd_stability(
            "DrugN",
            logP=2.5,
            mw=350.0,
            tm_C=210.0,
            tg_C=90.0,
            polymer_type="PVP",
            humidity_pct=50.0,
            temperature_C=25.0,
        )
        assert isinstance(result.drug_name, str)
        assert isinstance(result.polymer_type, str)
        assert isinstance(result.humidity_pct, float)
        assert isinstance(result.temperature_C, float)
        assert isinstance(result.tg_effective_C, float)
        assert isinstance(result.tg_tm_ratio_effective, float)
        assert isinstance(result.crystallization_risk, str)
        assert isinstance(result.estimated_shelf_life_months, float)
        assert isinstance(result.recommendation, str)
        assert isinstance(result.notes, str)
