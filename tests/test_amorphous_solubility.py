"""Tests for biopharmaceutics.amorphous_solubility module."""

from __future__ import annotations

import math

import pytest

from omega_pbpk.biopharmaceutics.amorphous_solubility import (
    AmorphousSolubilityResult,
    SPSimResult,
    amorphous_solubility_advantage,
    spring_and_parachute_sim,
)

# ---------------------------------------------------------------------------
# AmorphousSolubilityResult / amorphous_solubility_advantage
# ---------------------------------------------------------------------------


class TestAmorphousSolubilityResult:
    def test_returns_correct_type(self):
        res = amorphous_solubility_advantage("DrugA", tm_C=150.0, tg_C=80.0, logP=2.0)
        assert isinstance(res, AmorphousSolubilityResult)

    def test_fields_preserved(self):
        res = amorphous_solubility_advantage("DrugA", tm_C=150.0, tg_C=80.0, logP=2.0)
        assert res.drug_name == "DrugA"
        assert res.tm_C == pytest.approx(150.0)
        assert res.tg_C == pytest.approx(80.0)
        assert res.temperature_C == pytest.approx(37.0)

    def test_sa_sc_ratio_greater_than_one(self):
        """Amorphous form is always more soluble than crystalline (T < Tm)."""
        res = amorphous_solubility_advantage("DrugB", tm_C=200.0, tg_C=100.0, logP=3.0)
        assert res.sa_sc_ratio > 1.0

    def test_ratio_increases_with_higher_tm(self):
        """Higher Tm → larger ΔHfus → larger solubility advantage."""
        res_low = amorphous_solubility_advantage("D", tm_C=120.0, tg_C=70.0, logP=2.0)
        res_high = amorphous_solubility_advantage("D", tm_C=250.0, tg_C=130.0, logP=2.0)
        assert res_high.sa_sc_ratio > res_low.sa_sc_ratio

    def test_ratio_decreases_at_higher_temperature(self):
        """At temperatures closer to Tm the solubility advantage is smaller."""
        res_body = amorphous_solubility_advantage("D", tm_C=200.0, tg_C=100.0, logP=2.0,
                                                  temperature_C=37.0)
        res_warm = amorphous_solubility_advantage("D", tm_C=200.0, tg_C=100.0, logP=2.0,
                                                  temperature_C=100.0)
        assert res_body.sa_sc_ratio > res_warm.sa_sc_ratio

    def test_delta_hfus_formula(self):
        """ΔHfus ≈ 50 * Tm [J/mol]."""
        tm = 180.0
        res = amorphous_solubility_advantage("D", tm_C=tm, tg_C=90.0, logP=1.5)
        expected_hfus = 50.0 * (tm + 273.15)
        assert res.delta_hfus_J_per_mol == pytest.approx(expected_hfus, rel=1e-6)

    def test_murdande_formula_manual(self):
        """Verify the numerical result against a hand-calculated reference."""
        R = 8.314
        tm_C, T_C = 150.0, 37.0
        Tm = tm_C + 273.15
        T = T_C + 273.15
        hfus = 50.0 * Tm
        expected_ratio = math.exp(hfus * (Tm - T) / (R * T * Tm))
        res = amorphous_solubility_advantage("D", tm_C=tm_C, tg_C=80.0, logP=2.0,
                                             temperature_C=T_C)
        assert res.sa_sc_ratio == pytest.approx(expected_ratio, rel=1e-6)

    def test_notes_not_empty(self):
        res = amorphous_solubility_advantage("DrugC", tm_C=130.0, tg_C=60.0, logP=1.0)
        assert len(res.notes) > 10

    def test_good_glass_former_note(self):
        """Tg/Tm > 0.7 → good glass-forming stability note."""
        # Tg/Tm > 0.7: Tg = 0.72 * Tm (K)
        tm_C = 150.0
        Tm_K = tm_C + 273.15
        tg_K = 0.72 * Tm_K
        tg_C = tg_K - 273.15
        res = amorphous_solubility_advantage("D", tm_C=tm_C, tg_C=tg_C, logP=2.0)
        assert "Good glass-forming" in res.notes

    def test_moderate_glass_former_note(self):
        """Tg/Tm < 0.7 → moderate glass-forming stability note."""
        tm_C = 150.0
        Tm_K = tm_C + 273.15
        tg_K = 0.60 * Tm_K
        tg_C = tg_K - 273.15
        res = amorphous_solubility_advantage("D", tm_C=tm_C, tg_C=tg_C, logP=2.0)
        assert "Moderate" in res.notes

    def test_logP_preserved_in_notes(self):
        res = amorphous_solubility_advantage("D", tm_C=160.0, tg_C=90.0, logP=4.5)
        assert "4.50" in res.notes or "4.5" in res.notes


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


class TestAmorphousValidation:
    def test_raises_tm_zero(self):
        with pytest.raises(ValueError, match="tm_C"):
            amorphous_solubility_advantage("D", tm_C=0.0, tg_C=50.0, logP=2.0)

    def test_raises_tm_negative(self):
        with pytest.raises(ValueError, match="tm_C"):
            amorphous_solubility_advantage("D", tm_C=-10.0, tg_C=-50.0, logP=2.0)

    def test_raises_temperature_zero(self):
        with pytest.raises(ValueError, match="temperature_C"):
            amorphous_solubility_advantage("D", tm_C=150.0, tg_C=80.0, logP=2.0,
                                           temperature_C=0.0)

    def test_raises_temperature_negative(self):
        with pytest.raises(ValueError, match="temperature_C"):
            amorphous_solubility_advantage("D", tm_C=150.0, tg_C=80.0, logP=2.0,
                                           temperature_C=-10.0)

    def test_raises_temperature_equals_tm(self):
        with pytest.raises(ValueError):
            amorphous_solubility_advantage("D", tm_C=150.0, tg_C=80.0, logP=2.0,
                                           temperature_C=150.0)

    def test_raises_temperature_above_tm(self):
        with pytest.raises(ValueError):
            amorphous_solubility_advantage("D", tm_C=100.0, tg_C=60.0, logP=2.0,
                                           temperature_C=120.0)


# ---------------------------------------------------------------------------
# SPSimResult / spring_and_parachute_sim
# ---------------------------------------------------------------------------


class TestSPSim:
    def _default_kwargs(self, **overrides):
        base = dict(
            drug_name="DrugX",
            crystalline_sol_mg_mL=0.1,
            amorphous_sol_mg_mL=2.0,
            k_precipitation_per_h=1.0,
            k_dissolution_per_h=0.5,
            dose_mg=50.0,
            volume_GI_mL=250.0,
            t_end_h=4.0,
            dt_h=0.05,
        )
        base.update(overrides)
        return base

    def test_returns_correct_type(self):
        res = spring_and_parachute_sim(**self._default_kwargs())
        assert isinstance(res, SPSimResult)

    def test_times_length_matches_concentrations(self):
        res = spring_and_parachute_sim(**self._default_kwargs())
        assert len(res.times_h) == len(res.c_supersaturated) == len(res.c_cryst_sed)

    def test_times_starts_at_zero(self):
        res = spring_and_parachute_sim(**self._default_kwargs())
        assert res.times_h[0] == pytest.approx(0.0)

    def test_cmax_is_max_of_c_supersaturated(self):
        res = spring_and_parachute_sim(**self._default_kwargs())
        assert res.cmax == pytest.approx(max(res.c_supersaturated), rel=1e-6)

    def test_auc_positive(self):
        res = spring_and_parachute_sim(**self._default_kwargs())
        assert res.auc > 0.0

    def test_f_precipitated_between_0_and_1(self):
        res = spring_and_parachute_sim(**self._default_kwargs())
        assert 0.0 <= res.f_precipitated <= 1.0

    def test_no_precipitation_when_dose_below_crystalline_sol(self):
        """If dose/V < crystalline sol, no supersaturation → no precipitation."""
        res = spring_and_parachute_sim(
            drug_name="D",
            crystalline_sol_mg_mL=1.0,
            amorphous_sol_mg_mL=10.0,
            k_precipitation_per_h=5.0,
            k_dissolution_per_h=0.0,
            dose_mg=1.0,     # dose/V = 0.004 mg/mL < crystalline_sol=1.0
            volume_GI_mL=250.0,
            t_end_h=3.0,
            dt_h=0.05,
        )
        # No supersaturation: concentration should stay near initial or rise slightly
        assert res.f_precipitated == pytest.approx(0.0, abs=1e-3)

    def test_high_precipitation_rate_drives_conc_down(self):
        """Very high kp should rapidly reduce supersaturated concentration."""
        res = spring_and_parachute_sim(
            drug_name="D",
            crystalline_sol_mg_mL=0.1,
            amorphous_sol_mg_mL=5.0,
            k_precipitation_per_h=100.0,
            k_dissolution_per_h=0.0,
            dose_mg=100.0,
            volume_GI_mL=250.0,
            t_end_h=4.0,
            dt_h=0.01,
        )
        # Final concentration should be close to crystalline solubility
        assert res.c_supersaturated[-1] <= res.c_supersaturated[0]

    def test_zero_precipitation_rate_no_fall(self):
        """kp=0 with kd=0 → concentration stays constant (no sediment to dissolve)."""
        res = spring_and_parachute_sim(
            drug_name="D",
            crystalline_sol_mg_mL=0.1,
            amorphous_sol_mg_mL=1.0,
            k_precipitation_per_h=0.0,
            k_dissolution_per_h=0.0,
            dose_mg=50.0,
            volume_GI_mL=250.0,
            t_end_h=3.0,
            dt_h=0.05,
        )
        # With no precipitation and no dissolution, concentration is constant
        assert res.c_supersaturated[-1] == pytest.approx(res.c_supersaturated[0], rel=1e-3)

    def test_notes_not_empty(self):
        res = spring_and_parachute_sim(**self._default_kwargs())
        assert len(res.notes) > 5

    def test_drug_name_preserved(self):
        res = spring_and_parachute_sim(**self._default_kwargs(drug_name="TestDrug"))
        assert res.drug_name == "TestDrug"

    def test_auc_manual_trapezoid_consistency(self):
        """AUC should match manual trapezoid calculation."""
        res = spring_and_parachute_sim(**self._default_kwargs())
        manual_auc = sum(
            0.5 * (res.c_supersaturated[i] + res.c_supersaturated[i + 1])
            * (res.times_h[i + 1] - res.times_h[i])
            for i in range(len(res.times_h) - 1)
        )
        assert res.auc == pytest.approx(manual_auc, rel=1e-5)


# ---------------------------------------------------------------------------
# SP Sim validation errors
# ---------------------------------------------------------------------------


class TestSPSimValidation:
    def _base(self, **overrides):
        base = dict(
            drug_name="D",
            crystalline_sol_mg_mL=0.1,
            amorphous_sol_mg_mL=2.0,
            k_precipitation_per_h=1.0,
            k_dissolution_per_h=0.5,
            dose_mg=50.0,
            volume_GI_mL=250.0,
        )
        base.update(overrides)
        return base

    def test_raises_crystalline_zero(self):
        with pytest.raises(ValueError, match="crystalline_sol_mg_mL"):
            spring_and_parachute_sim(**self._base(crystalline_sol_mg_mL=0.0))

    def test_raises_amorphous_zero(self):
        with pytest.raises(ValueError, match="amorphous_sol_mg_mL"):
            spring_and_parachute_sim(**self._base(amorphous_sol_mg_mL=0.0))

    def test_raises_amorphous_less_than_crystalline(self):
        with pytest.raises(ValueError):
            spring_and_parachute_sim(**self._base(
                crystalline_sol_mg_mL=2.0, amorphous_sol_mg_mL=1.0
            ))

    def test_raises_negative_kp(self):
        with pytest.raises(ValueError, match="k_precipitation"):
            spring_and_parachute_sim(**self._base(k_precipitation_per_h=-1.0))

    def test_raises_negative_kd(self):
        with pytest.raises(ValueError, match="k_dissolution"):
            spring_and_parachute_sim(**self._base(k_dissolution_per_h=-0.1))

    def test_raises_dose_zero(self):
        with pytest.raises(ValueError, match="dose_mg"):
            spring_and_parachute_sim(**self._base(dose_mg=0.0))

    def test_raises_volume_zero(self):
        with pytest.raises(ValueError, match="volume_GI_mL"):
            spring_and_parachute_sim(**self._base(volume_GI_mL=0.0))

    def test_raises_t_end_zero(self):
        with pytest.raises(ValueError, match="t_end_h"):
            spring_and_parachute_sim(**self._base(t_end_h=0.0))

    def test_raises_dt_zero(self):
        with pytest.raises(ValueError, match="dt_h"):
            spring_and_parachute_sim(**self._base(dt_h=0.0))
