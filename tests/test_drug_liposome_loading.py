"""Tests for Phase 1114 — Drug liposome loading efficiency prediction."""

import pytest

from omega_pbpk.prediction.drug_liposome_loading import (
    LiposomeLoadingResult,
    predict_loading_efficiency,
    screen_lipid_types,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def default_result(**kwargs):
    params = dict(
        drug_name="TestDrug",
        logP=2.5,
        pka=8.0,
        ionization_type="neutral",
        lipid_type="POPC",
        ph_internal=4.0,
        ph_external=7.4,
        drug_to_lipid_ratio=0.1,
    )
    params.update(kwargs)
    return predict_loading_efficiency(**params)


# ---------------------------------------------------------------------------
# Structural tests
# ---------------------------------------------------------------------------


class TestResultType:
    def test_returns_correct_type(self):
        res = default_result()
        assert isinstance(res, LiposomeLoadingResult)

    def test_drug_name_preserved(self):
        res = predict_loading_efficiency("Doxorubicin", 1.2, 8.3, "base", "DOPG")
        assert res.drug_name == "Doxorubicin"

    def test_logP_preserved(self):
        res = default_result(logP=3.5)
        assert res.logP == pytest.approx(3.5)

    def test_pka_preserved(self):
        res = default_result(pka=6.5)
        assert res.pka == pytest.approx(6.5)

    def test_lipid_type_preserved(self):
        res = default_result(lipid_type="DPPC")
        assert res.lipid_type == "DPPC"

    def test_ionization_type_preserved(self):
        res = default_result(ionization_type="acid")
        assert res.ionization_type == "acid"

    def test_drug_to_lipid_ratio_preserved(self):
        res = default_result(drug_to_lipid_ratio=0.2)
        assert res.drug_to_lipid_ratio == pytest.approx(0.2)


# ---------------------------------------------------------------------------
# Efficiency range tests
# ---------------------------------------------------------------------------


class TestEfficiencyRange:
    def test_efficiency_in_0_to_100(self):
        for lipid in ["DPPC", "DSPC", "POPC", "DOPC", "DOPG"]:
            res = predict_loading_efficiency("D", 3.0, 7.0, "base", lipid)
            assert 0.0 <= res.loading_efficiency_pct <= 100.0

    def test_efficiency_nonzero_for_hydrophobic_drug(self):
        res = default_result(logP=4.0)
        assert res.loading_efficiency_pct > 0.0

    def test_efficiency_lower_for_very_hydrophilic(self):
        res_hphil = default_result(logP=-3.0, ionization_type="neutral")
        res_hphob = default_result(logP=5.0, ionization_type="neutral")
        assert res_hphob.loading_efficiency_pct > res_hphil.loading_efficiency_pct


# ---------------------------------------------------------------------------
# Ionization and pH gradient tests
# ---------------------------------------------------------------------------


class TestIonizationEffects:
    def test_base_with_ph_gradient_higher_than_neutral(self):
        # Same lipid, high logP
        res_neutral = predict_loading_efficiency("D", 1.0, 9.0, "neutral", "POPC")
        res_base = predict_loading_efficiency("D", 1.0, 9.0, "base", "POPC")
        assert res_base.loading_efficiency_pct > res_neutral.loading_efficiency_pct

    def test_base_mechanism_is_active_ph_gradient(self):
        res = predict_loading_efficiency(
            "D", 2.0, 8.0, "base", "POPC", ph_internal=4.0, ph_external=7.4
        )
        assert res.trapping_mechanism == "active_pH_gradient"

    def test_neutral_mechanism_is_passive(self):
        res = predict_loading_efficiency("D", 2.0, 8.0, "neutral", "POPC")
        assert res.trapping_mechanism == "passive_partitioning"

    def test_no_ph_gradient_base_is_passive(self):
        res = predict_loading_efficiency(
            "D", 2.0, 8.0, "base", "POPC", ph_internal=7.4, ph_external=7.4
        )
        assert res.trapping_mechanism == "passive_partitioning"

    def test_acid_active_loading_when_ph_internal_above_pka(self):
        # ph_internal=7.4 > pka=4.0: acid mostly ionized inside → trapped
        res = predict_loading_efficiency(
            "D", 2.0, 4.0, "acid", "POPC", ph_internal=7.4, ph_external=7.4
        )
        assert res.trapping_mechanism == "active_pH_gradient"


# ---------------------------------------------------------------------------
# Lipid-specific tests
# ---------------------------------------------------------------------------


class TestLipidEffects:
    def test_cationic_base_in_DOPG_higher_than_neutral_in_DOPG(self):
        res_base = predict_loading_efficiency("D", 2.0, 8.5, "base", "DOPG")
        res_neutral = predict_loading_efficiency("D", 2.0, 8.5, "neutral", "DOPG")
        assert res_base.loading_efficiency_pct > res_neutral.loading_efficiency_pct

    def test_dppc_bonus_for_hydrophobic_drug(self):
        res_dppc = predict_loading_efficiency("D", 3.0, 7.0, "neutral", "DPPC")
        res_popc = predict_loading_efficiency("D", 3.0, 7.0, "neutral", "POPC")
        assert res_dppc.loading_efficiency_pct > res_popc.loading_efficiency_pct

    def test_dopc_lower_than_popc_for_neutral(self):
        res_popc = predict_loading_efficiency("D", 3.0, 7.0, "neutral", "POPC")
        res_dopc = predict_loading_efficiency("D", 3.0, 7.0, "neutral", "DOPC")
        assert res_popc.loading_efficiency_pct > res_dopc.loading_efficiency_pct

    def test_dspc_similar_to_dppc_for_hydrophobic(self):
        res_dppc = predict_loading_efficiency("D", 3.5, 7.0, "neutral", "DPPC")
        res_dspc = predict_loading_efficiency("D", 3.5, 7.0, "neutral", "DSPC")
        # Both should have +10% bonus; efficiency should be equal
        assert res_dppc.loading_efficiency_pct == pytest.approx(
            res_dspc.loading_efficiency_pct, abs=0.01
        )


# ---------------------------------------------------------------------------
# Encapsulation class tests
# ---------------------------------------------------------------------------


class TestEncapsulationClass:
    def test_poor_class_threshold(self):
        # Very hydrophilic, neutral → poor
        res = predict_loading_efficiency("D", -4.0, 7.0, "neutral", "DOPC")
        assert res.encapsulation_class == "poor"

    def test_high_class_threshold(self):
        # Hydrophobic base, DOPG, pH gradient → high
        res = predict_loading_efficiency(
            "D", 5.0, 8.0, "base", "DOPG", ph_internal=4.0, ph_external=7.4
        )
        assert res.encapsulation_class == "high"

    def test_encapsulation_class_consistent_with_efficiency(self):
        for lipid in ["DPPC", "POPC", "DOPC"]:
            res = predict_loading_efficiency("D", 2.0, 7.0, "neutral", lipid)
            eff = res.loading_efficiency_pct
            if eff < 30:
                assert res.encapsulation_class == "poor"
            elif eff <= 70:
                assert res.encapsulation_class == "moderate"
            else:
                assert res.encapsulation_class == "high"


# ---------------------------------------------------------------------------
# Screen lipid types tests
# ---------------------------------------------------------------------------


class TestScreenLipidTypes:
    def test_returns_five_results(self):
        results = screen_lipid_types("Dox", 1.5, 8.3, "base")
        assert len(results) == 5

    def test_returns_list_of_correct_type(self):
        results = screen_lipid_types("Dox", 1.5, 8.3, "base")
        for r in results:
            assert isinstance(r, LiposomeLoadingResult)

    def test_sorted_descending_by_efficiency(self):
        results = screen_lipid_types("Dox", 1.5, 8.3, "base")
        efficiencies = [r.loading_efficiency_pct for r in results]
        assert efficiencies == sorted(efficiencies, reverse=True)

    def test_all_five_lipid_types_present(self):
        results = screen_lipid_types("Dox", 1.5, 8.3, "base")
        lipid_types = {r.lipid_type for r in results}
        assert lipid_types == {"DPPC", "DSPC", "POPC", "DOPC", "DOPG"}

    def test_screen_neutral_drug_sorted(self):
        results = screen_lipid_types("NeutralDrug", 3.0, 7.0, "neutral")
        efficiencies = [r.loading_efficiency_pct for r in results]
        assert efficiencies == sorted(efficiencies, reverse=True)


# ---------------------------------------------------------------------------
# Validation error tests
# ---------------------------------------------------------------------------


class TestValidation:
    def test_invalid_ionization_type_raises(self):
        with pytest.raises(ValueError, match="ionization_type"):
            predict_loading_efficiency("D", 2.0, 7.0, "zwitterion", "DPPC")

    def test_invalid_lipid_type_raises(self):
        with pytest.raises(ValueError, match="lipid_type"):
            predict_loading_efficiency("D", 2.0, 7.0, "neutral", "DLPC")

    def test_drug_to_lipid_ratio_zero_raises(self):
        with pytest.raises(ValueError, match="drug_to_lipid_ratio"):
            predict_loading_efficiency("D", 2.0, 7.0, "neutral", "POPC", drug_to_lipid_ratio=0.0)

    def test_drug_to_lipid_ratio_above_one_raises(self):
        with pytest.raises(ValueError, match="drug_to_lipid_ratio"):
            predict_loading_efficiency("D", 2.0, 7.0, "neutral", "POPC", drug_to_lipid_ratio=1.5)

    def test_ph_internal_out_of_range_raises(self):
        with pytest.raises(ValueError, match="ph_internal"):
            predict_loading_efficiency("D", 2.0, 7.0, "neutral", "POPC", ph_internal=-1.0)

    def test_ph_external_out_of_range_raises(self):
        with pytest.raises(ValueError, match="ph_external"):
            predict_loading_efficiency("D", 2.0, 7.0, "neutral", "POPC", ph_external=15.0)

    def test_logP_below_range_raises(self):
        with pytest.raises(ValueError, match="logP"):
            predict_loading_efficiency("D", -6.0, 7.0, "neutral", "POPC")

    def test_logP_above_range_raises(self):
        with pytest.raises(ValueError, match="logP"):
            predict_loading_efficiency("D", 11.0, 7.0, "neutral", "POPC")
