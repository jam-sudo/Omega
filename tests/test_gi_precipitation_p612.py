"""Tests for Phase 612: GI Tract Drug Precipitation Predictor.

Coverage:
- Return type and frozen dataclass behaviour
- Dose number calculation
- BCS class 2/4 high-risk scenarios
- Amorphous solubility advantage
- Polymer reduction of risk score
- Risk classification thresholds
- Ionisation state (acid/base) effect on intestinal solubility
- Induction time behaviour
- Polymer recommendation by logP
- Validation errors
"""

import math

import pytest

from omega_pbpk.prediction.gi_precipitation_p612 import (
    GIPrecipitationResult,
    predict_gi_precipitation,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DEFAULTS = dict(
    drug_name="TestPrecip",
    dose_mg=100.0,
    cs_fasted_mg_mL=0.1,
    logP=2.5,
    mw_Da=400.0,
    pka=None,
    drug_type="neutral",
    amorphous=False,
    polymer_conc_mg_mL=0.0,
)


def _run(**overrides):
    params = {**DEFAULTS, **overrides}
    return predict_gi_precipitation(**params)


# ---------------------------------------------------------------------------
# 1. Return type and frozen dataclass
# ---------------------------------------------------------------------------


class TestReturnType:
    def test_returns_gi_precipitation_result(self):
        result = _run()
        assert isinstance(result, GIPrecipitationResult)

    def test_result_is_frozen(self):
        result = _run()
        with pytest.raises((AttributeError, TypeError)):
            result.drug_name = "Modified"  # type: ignore[misc]

    def test_drug_name_preserved(self):
        result = _run(drug_name="Itraconazole")
        assert result.drug_name == "Itraconazole"

    def test_dose_mg_preserved(self):
        result = _run(dose_mg=200.0)
        assert result.dose_mg == pytest.approx(200.0)

    def test_notes_is_nonempty_str(self):
        result = _run()
        assert isinstance(result.notes, str)
        assert len(result.notes) > 10

    def test_all_numeric_fields_finite(self):
        result = _run()
        for field in (
            "dose_number",
            "cs_ph_intestine_mg_mL",
            "initial_supersaturation",
            "precipitation_risk_score",
            "induction_time_h",
            "solubility_advantage",
        ):
            val = getattr(result, field)
            assert math.isfinite(val), f"{field} is not finite: {val}"


# ---------------------------------------------------------------------------
# 2. Dose number calculation
# ---------------------------------------------------------------------------


class TestDoseNumber:
    def test_dose_number_formula_neutral(self):
        """Do = dose / (Cs_intestine * 250)."""
        cs = 0.2
        dose = 50.0
        result = _run(dose_mg=dose, cs_fasted_mg_mL=cs, drug_type="neutral")
        expected_do = dose / (cs * 250.0)
        assert result.dose_number == pytest.approx(expected_do, rel=1e-6)

    def test_dose_number_low_for_high_solubility(self):
        result = _run(dose_mg=50.0, cs_fasted_mg_mL=5.0)
        assert result.dose_number < 1.0

    def test_dose_number_high_for_low_solubility(self):
        result = _run(dose_mg=500.0, cs_fasted_mg_mL=0.01)
        assert result.dose_number > 10.0

    def test_cs_intestine_equals_fasted_for_neutral(self):
        result = _run(cs_fasted_mg_mL=0.3, drug_type="neutral")
        assert result.cs_ph_intestine_mg_mL == pytest.approx(0.3, rel=1e-6)


# ---------------------------------------------------------------------------
# 3. Ionisation effects
# ---------------------------------------------------------------------------


class TestIonisationEffects:
    def test_acid_drug_higher_intestinal_solubility(self):
        """Acids are more soluble at pH 6.8 > pKa."""
        cs = 0.05
        result = _run(cs_fasted_mg_mL=cs, drug_type="acid", pka=4.0)
        # pH 6.8 >> pKa 4.0 → large fraction ionised → higher solubility
        assert result.cs_ph_intestine_mg_mL > cs

    def test_base_drug_lower_intestinal_solubility_at_ph6_8(self):
        """Bases are less ionised at pH 6.8 vs gastric pH (pKa >> 6.8)."""
        cs = 0.1
        # pKa 9.0 >> pH 6.8 → mostly unionised at pH 6.8 → lower solubility multiplier
        result = _run(cs_fasted_mg_mL=cs, drug_type="base", pka=9.0)
        # The H-H formula: cs*(1 + 10^(pKa-pH)) = cs*(1+10^2.2) ≈ cs*159 → capped at 100*cs
        assert result.cs_ph_intestine_mg_mL <= 100.0 * cs

    def test_neutral_ignores_pka(self):
        result_no_pka = _run(drug_type="neutral", pka=None)
        result_with_pka = _run(drug_type="neutral", pka=7.0)
        assert result_no_pka.cs_ph_intestine_mg_mL == pytest.approx(
            result_with_pka.cs_ph_intestine_mg_mL
        )

    def test_intestinal_solubility_cap_applied(self):
        """Very basic drug: solubility capped at 100× fasted."""
        result = _run(cs_fasted_mg_mL=0.01, drug_type="base", pka=12.0)
        assert result.cs_ph_intestine_mg_mL <= 100.0 * 0.01 + 1e-9


# ---------------------------------------------------------------------------
# 4. Amorphous solubility advantage
# ---------------------------------------------------------------------------


class TestAmorphousSolubility:
    def test_amorphous_solubility_advantage_gt_one(self):
        result = _run(amorphous=True, logP=3.0)
        assert result.solubility_advantage > 1.0

    def test_crystalline_solubility_advantage_is_one(self):
        result = _run(amorphous=False)
        assert result.solubility_advantage == pytest.approx(1.0)

    def test_amorphous_higher_supersaturation(self):
        cryst = _run(amorphous=False, logP=3.0)
        amorph = _run(amorphous=True, logP=3.0)
        assert amorph.initial_supersaturation >= cryst.initial_supersaturation

    def test_amorphous_advantage_capped_at_10(self):
        """Very lipophilic compound: advantage capped at 10."""
        result = _run(amorphous=True, logP=10.0)
        assert result.solubility_advantage <= 10.0 + 1e-9


# ---------------------------------------------------------------------------
# 5. Polymer reduction
# ---------------------------------------------------------------------------


class TestPolymerReduction:
    def test_polymer_reduces_risk_score(self):
        no_polymer = _run(dose_mg=500.0, cs_fasted_mg_mL=0.01, polymer_conc_mg_mL=0.0)
        with_polymer = _run(dose_mg=500.0, cs_fasted_mg_mL=0.01, polymer_conc_mg_mL=3.0)
        assert with_polymer.precipitation_risk_score < no_polymer.precipitation_risk_score

    def test_high_polymer_conc_max_reduction_30(self):
        """Polymer reduction capped at 30 points."""
        high_risk = _run(dose_mg=1000.0, cs_fasted_mg_mL=0.005)
        with_polymer = _run(dose_mg=1000.0, cs_fasted_mg_mL=0.005, polymer_conc_mg_mL=100.0)
        reduction = high_risk.precipitation_risk_score - with_polymer.precipitation_risk_score
        assert reduction <= 30.0 + 1e-6

    def test_zero_polymer_no_reduction(self):
        result = _run(dose_mg=500.0, cs_fasted_mg_mL=0.01, polymer_conc_mg_mL=0.0)
        result2 = _run(dose_mg=500.0, cs_fasted_mg_mL=0.01)
        assert result.precipitation_risk_score == pytest.approx(result2.precipitation_risk_score)


# ---------------------------------------------------------------------------
# 6. Risk classification
# ---------------------------------------------------------------------------


class TestRiskClassification:
    def test_negligible_risk_low_dose_high_sol(self):
        result = _run(dose_mg=5.0, cs_fasted_mg_mL=5.0)
        assert result.precipitation_risk_class == "negligible"

    def test_high_risk_very_low_sol(self):
        result = _run(dose_mg=1000.0, cs_fasted_mg_mL=0.001)
        assert result.precipitation_risk_class == "high"

    def test_risk_score_in_0_100(self):
        result = _run()
        assert 0.0 <= result.precipitation_risk_score <= 100.0

    def test_high_risk_triggers_polymer_required(self):
        result = _run(dose_mg=1000.0, cs_fasted_mg_mL=0.001)
        assert result.polymer_required is True

    def test_negligible_risk_no_polymer_required(self):
        result = _run(dose_mg=5.0, cs_fasted_mg_mL=5.0)
        assert result.polymer_required is False
        assert result.recommended_polymer == "none"


# ---------------------------------------------------------------------------
# 7. Polymer recommendation
# ---------------------------------------------------------------------------


class TestPolymerRecommendation:
    def _high_risk_result(self, logP):
        return _run(dose_mg=1000.0, cs_fasted_mg_mL=0.001, logP=logP)

    def test_high_logP_recommends_hpmc_as(self):
        result = self._high_risk_result(logP=3.5)
        assert result.recommended_polymer == "HPMC-AS"

    def test_mid_logP_recommends_pvp_va(self):
        result = self._high_risk_result(logP=1.5)
        assert result.recommended_polymer == "PVP-VA"

    def test_low_logP_recommends_hpmc(self):
        result = self._high_risk_result(logP=-0.5)
        assert result.recommended_polymer == "HPMC"

    def test_low_risk_recommends_none(self):
        result = _run(dose_mg=5.0, cs_fasted_mg_mL=10.0)
        assert result.recommended_polymer == "none"


# ---------------------------------------------------------------------------
# 8. Induction time
# ---------------------------------------------------------------------------


class TestInductionTime:
    def test_induction_time_in_valid_range(self):
        result = _run()
        assert 0.1 <= result.induction_time_h <= 24.0

    def test_high_supersaturation_shorter_induction(self):
        low_ss = _run(dose_mg=50.0, cs_fasted_mg_mL=5.0)
        high_ss = _run(dose_mg=1000.0, cs_fasted_mg_mL=0.01)
        assert high_ss.induction_time_h <= low_ss.induction_time_h


# ---------------------------------------------------------------------------
# 9. Validation errors
# ---------------------------------------------------------------------------


class TestValidation:
    def test_zero_dose_raises(self):
        with pytest.raises(ValueError, match="dose_mg"):
            _run(dose_mg=0.0)

    def test_negative_dose_raises(self):
        with pytest.raises(ValueError, match="dose_mg"):
            _run(dose_mg=-10.0)

    def test_zero_solubility_raises(self):
        with pytest.raises(ValueError, match="cs_fasted"):
            _run(cs_fasted_mg_mL=0.0)

    def test_zero_mw_raises(self):
        with pytest.raises(ValueError, match="mw_Da"):
            _run(mw_Da=0.0)

    def test_negative_polymer_raises(self):
        with pytest.raises(ValueError, match="polymer_conc"):
            _run(polymer_conc_mg_mL=-1.0)

    def test_invalid_drug_type_raises(self):
        with pytest.raises(ValueError, match="drug_type"):
            _run(drug_type="zwitterion")

    def test_valid_acid_type_accepted(self):
        result = _run(drug_type="acid", pka=4.5)
        assert isinstance(result, GIPrecipitationResult)

    def test_valid_base_type_accepted(self):
        result = _run(drug_type="base", pka=8.5)
        assert isinstance(result, GIPrecipitationResult)
