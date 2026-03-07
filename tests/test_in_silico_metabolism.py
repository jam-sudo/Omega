"""
Tests for Phase 584: prediction/in_silico_metabolism.py
"""


from omega_pbpk.prediction.in_silico_metabolism import (
    clint_from_structure,
    cyp_substrate_probability,
    predict_metabolic_sites,
    predict_phase1_products,
)

# ---------------------------------------------------------------------------
# predict_metabolic_sites tests
# ---------------------------------------------------------------------------


class TestPredictMetabolicSites:
    def test_returns_dict_with_required_keys(self):
        result = predict_metabolic_sites([], logP=2.0, mw=350.0)
        assert "likely_enzymes" in result
        assert "metabolism_class" in result
        assert "probability_phase1" in result

    def test_high_logp_mid_mw_gives_cyp3a4(self):
        result = predict_metabolic_sites([], logP=4.0, mw=400.0)
        assert "CYP3A4" in result["likely_enzymes"]

    def test_high_logp_mid_mw_gives_phase1(self):
        result = predict_metabolic_sites([], logP=4.0, mw=400.0)
        assert result["metabolism_class"] == "phase1"

    def test_low_logp_small_mw_gives_phase2(self):
        result = predict_metabolic_sites([], logP=0.5, mw=200.0)
        assert result["metabolism_class"] == "phase2"

    def test_low_logp_small_mw_probability_phase1_low(self):
        result = predict_metabolic_sites([], logP=0.5, mw=200.0)
        assert result["probability_phase1"] < 0.5

    def test_probability_phase1_in_zero_one(self):
        for logp, mw in [(0.5, 200), (2.0, 350), (5.0, 450)]:
            result = predict_metabolic_sites([], logP=logp, mw=mw)
            assert 0.0 <= result["probability_phase1"] <= 1.0

    def test_mixed_metabolism_class(self):
        result = predict_metabolic_sites([], logP=2.0, mw=350.0)
        assert result["metabolism_class"] in {"phase1", "phase2", "mixed"}

    def test_likely_enzymes_is_list(self):
        result = predict_metabolic_sites([], logP=3.5, mw=400.0)
        assert isinstance(result["likely_enzymes"], list)
        assert len(result["likely_enzymes"]) > 0


# ---------------------------------------------------------------------------
# predict_phase1_products tests
# ---------------------------------------------------------------------------


class TestPredictPhase1Products:
    def test_returns_list_with_four_reactions(self):
        products = predict_phase1_products("test_drug", logP=2.5, mw=350.0)
        assert len(products) == 4

    def test_each_product_has_required_keys(self):
        products = predict_phase1_products("test_drug", logP=2.5, mw=350.0)
        required = {"reaction", "product_mw_estimate", "likely_enzyme", "notes"}
        for p in products:
            assert required.issubset(p.keys())

    def test_hydroxylation_adds_16(self):
        mw = 300.0
        products = predict_phase1_products("drug", logP=2.5, mw=mw)
        hydrox = next(p for p in products if p["reaction"] == "hydroxylation")
        assert abs(hydrox["product_mw_estimate"] - (mw + 16)) < 1e-9

    def test_n_dealkylation_subtracts_14(self):
        mw = 300.0
        products = predict_phase1_products("drug", logP=2.5, mw=mw)
        ndea = next(p for p in products if p["reaction"] == "N-dealkylation")
        assert abs(ndea["product_mw_estimate"] - (mw - 14)) < 1e-9

    def test_o_demethylation_subtracts_14(self):
        mw = 300.0
        products = predict_phase1_products("drug", logP=2.5, mw=mw)
        odem = next(p for p in products if p["reaction"] == "O-demethylation")
        assert abs(odem["product_mw_estimate"] - (mw - 14)) < 1e-9

    def test_oxidation_adds_16(self):
        mw = 300.0
        products = predict_phase1_products("drug", logP=2.5, mw=mw)
        oxid = next(p for p in products if p["reaction"] == "oxidation")
        assert abs(oxid["product_mw_estimate"] - (mw + 16)) < 1e-9

    def test_likely_enzyme_is_string(self):
        products = predict_phase1_products("drug", logP=2.5, mw=350.0)
        for p in products:
            assert isinstance(p["likely_enzyme"], str)
            assert len(p["likely_enzyme"]) > 0

    def test_reaction_names_are_correct(self):
        products = predict_phase1_products("drug", logP=2.5, mw=350.0)
        names = {p["reaction"] for p in products}
        assert names == {"hydroxylation", "N-dealkylation", "O-demethylation", "oxidation"}


# ---------------------------------------------------------------------------
# clint_from_structure tests
# ---------------------------------------------------------------------------


class TestClintFromStructure:
    def test_returns_required_keys(self):
        result = clint_from_structure(logP=2.0, mw=350.0, psa=60.0, n_hba=4)
        assert "clint_uL_min_mg" in result
        assert "clint_unbound" in result
        assert "hepatic_cl_estimate_L_per_h" in result
        assert "prediction_confidence" in result

    def test_clint_in_valid_range(self):
        result = clint_from_structure(logP=2.0, mw=350.0, psa=60.0, n_hba=4)
        assert 0.1 <= result["clint_uL_min_mg"] <= 1000.0

    def test_higher_logp_gives_higher_clint(self):
        low = clint_from_structure(logP=0.5, mw=300.0, psa=80.0, n_hba=5)
        high = clint_from_structure(logP=4.0, mw=300.0, psa=80.0, n_hba=5)
        assert high["clint_uL_min_mg"] >= low["clint_uL_min_mg"]

    def test_clint_unbound_is_clint_divided_by_fu_mic(self):
        result = clint_from_structure(logP=2.0, mw=350.0, psa=60.0, n_hba=4)
        expected_fu_mic = 0.1
        assert abs(result["clint_unbound"] - result["clint_uL_min_mg"] / expected_fu_mic) < 1e-6

    def test_hepatic_cl_positive(self):
        result = clint_from_structure(logP=2.0, mw=350.0, psa=60.0, n_hba=4)
        assert result["hepatic_cl_estimate_L_per_h"] > 0.0

    def test_hepatic_cl_less_than_liver_blood_flow(self):
        result = clint_from_structure(logP=2.0, mw=350.0, psa=60.0, n_hba=4)
        # Liver blood flow = 90 L/h
        assert result["hepatic_cl_estimate_L_per_h"] <= 90.0

    def test_prediction_confidence_valid_value(self):
        result = clint_from_structure(logP=2.0, mw=350.0, psa=60.0, n_hba=4)
        assert result["prediction_confidence"] in {"low", "medium", "high"}


# ---------------------------------------------------------------------------
# cyp_substrate_probability tests
# ---------------------------------------------------------------------------


class TestCypSubstrateProbability:
    def test_returns_required_keys(self):
        result = cyp_substrate_probability(logP=3.0, mw=400.0, psa=60.0)
        assert "cyp3a4_prob" in result
        assert "cyp2d6_prob" in result
        assert "cyp2c9_prob" in result
        assert "dominant_cyp" in result

    def test_all_probabilities_in_zero_one(self):
        result = cyp_substrate_probability(logP=3.0, mw=400.0, psa=60.0)
        for key in ("cyp3a4_prob", "cyp2d6_prob", "cyp2c9_prob"):
            assert 0.0 <= result[key] <= 1.0

    def test_dominant_cyp_is_valid(self):
        result = cyp_substrate_probability(logP=3.0, mw=400.0, psa=60.0)
        assert result["dominant_cyp"] in {"CYP3A4", "CYP2D6", "CYP2C9"}

    def test_lipophilic_drug_dominant_cyp3a4(self):
        # High logP, moderate MW, low PSA → CYP3A4 should dominate
        result = cyp_substrate_probability(logP=5.0, mw=450.0, psa=30.0)
        assert result["dominant_cyp"] == "CYP3A4"

    def test_cyp3a4_high_prob_for_lipophilic(self):
        result = cyp_substrate_probability(logP=5.0, mw=450.0, psa=30.0)
        assert result["cyp3a4_prob"] > 0.5

    def test_cyp2d6_prob_less_than_cyp3a4_for_lipophilic(self):
        # CYP2D6 is capped at 0.6 × sigmoid, so generally lower
        result = cyp_substrate_probability(logP=5.0, mw=450.0, psa=20.0)
        # cyp2d6 has 0.6 cap factor; for extreme lipophilic, 3A4 may dominate
        assert result["cyp3a4_prob"] >= result["cyp2d6_prob"]

    def test_dominant_cyp_matches_maximum_probability(self):
        result = cyp_substrate_probability(logP=3.0, mw=400.0, psa=60.0)
        probs = {
            "CYP3A4": result["cyp3a4_prob"],
            "CYP2D6": result["cyp2d6_prob"],
            "CYP2C9": result["cyp2c9_prob"],
        }
        expected_dominant = max(probs, key=lambda k: probs[k])
        assert result["dominant_cyp"] == expected_dominant
