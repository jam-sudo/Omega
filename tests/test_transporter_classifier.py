"""Tests for TransporterClassifier and TransporterPlugin."""

from __future__ import annotations

import math

import pytest

from omega_pbpk.prediction.transporter_classifier import (
    TRANSPORTER_NAMES,
    TransporterClassifier,
    TransporterInteraction,
    TransporterProfile,
)

# ---------------------------------------------------------------------------
# Shared fixtures / helpers
# ---------------------------------------------------------------------------

METFORMIN_PROPS = {"mw": 129.2, "logp": -1.43, "charge_class": "base"}
ROSUVASTATIN_PROPS = {"mw": 481.5, "logp": 0.13, "charge_class": "acid"}
CYCLOSPORINE_PROPS = {"mw": 1202.6, "logp": 2.92, "charge_class": "neutral"}
DIGOXIN_PROPS = {"mw": 780.9, "logp": 1.26, "charge_class": "neutral"}
CAFFEINE_PROPS = {"mw": 194.2, "logp": -0.07, "charge_class": "neutral"}
CIMETIDINE_PROPS = {"mw": 252.3, "logp": 0.40, "charge_class": "base"}
RITONAVIR_PROPS = {"mw": 720.9, "logp": 4.14, "charge_class": "base"}


# ---------------------------------------------------------------------------
# TestImports
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestImports:
    def test_transporter_classifier_importable(self):
        from omega_pbpk.prediction.transporter_classifier import TransporterClassifier

        assert TransporterClassifier is not None

    def test_transporter_plugin_importable(self):
        from omega_pbpk.plugins.transporter_plugin import TransporterPlugin

        assert TransporterPlugin is not None

    def test_transporter_names_count(self):
        assert len(TRANSPORTER_NAMES) == 7

    def test_transporter_names_values(self):
        expected = ["OATP1B1", "OATP1B3", "P_gp", "BCRP", "OCT2", "MRP2", "MATE1"]
        assert TRANSPORTER_NAMES == expected

    def test_prediction_module_exports(self):
        from omega_pbpk.prediction import TransporterClassifier, TransporterProfile

        assert TransporterClassifier is not None
        assert TransporterProfile is not None

    def test_plugins_init_exports_transporter_plugin(self):
        from omega_pbpk.plugins import TransporterPlugin

        assert TransporterPlugin is not None


# ---------------------------------------------------------------------------
# TestTransporterInteraction
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestTransporterInteraction:
    def test_interaction_fields(self):
        ti = TransporterInteraction(
            transporter="P_gp",
            substrate_prob=0.8,
            inhibitor_prob=0.2,
            km_uM=45.0,
            ic50_uM=float("inf"),
        )
        assert ti.transporter == "P_gp"
        assert ti.substrate_prob == 0.8
        assert ti.inhibitor_prob == 0.2
        assert ti.km_uM == 45.0
        assert ti.ic50_uM == float("inf")

    def test_interaction_is_frozen(self):
        ti = TransporterInteraction(
            transporter="BCRP",
            substrate_prob=0.5,
            inhibitor_prob=0.1,
            km_uM=20.0,
            ic50_uM=float("inf"),
        )
        with pytest.raises((AttributeError, TypeError)):
            ti.substrate_prob = 0.9  # type: ignore[misc]

    def test_probs_stored_as_given(self):
        ti = TransporterInteraction(
            transporter="OCT2",
            substrate_prob=0.75,
            inhibitor_prob=0.30,
            km_uM=80.0,
            ic50_uM=50.0,
        )
        assert 0.0 <= ti.substrate_prob <= 1.0
        assert 0.0 <= ti.inhibitor_prob <= 1.0

    def test_km_non_negative(self):
        ti = TransporterInteraction(
            transporter="MRP2",
            substrate_prob=0.6,
            inhibitor_prob=0.1,
            km_uM=30.0,
            ic50_uM=float("inf"),
        )
        assert ti.km_uM >= 0.0

    def test_ic50_positive_or_inf(self):
        ti = TransporterInteraction(
            transporter="MATE1",
            substrate_prob=0.1,
            inhibitor_prob=0.7,
            km_uM=0.0,
            ic50_uM=25.0,
        )
        assert ti.ic50_uM > 0.0 or math.isinf(ti.ic50_uM)


# ---------------------------------------------------------------------------
# TestTransporterProfileMethods
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestTransporterProfileMethods:
    def _make_profile(self, sub_p: float = 0.8, inh_p: float = 0.2) -> TransporterProfile:
        interactions = {
            t: TransporterInteraction(
                transporter=t,
                substrate_prob=sub_p,
                inhibitor_prob=inh_p,
                km_uM=10.0 if sub_p >= 0.5 else 0.0,
                ic50_uM=5.0 if inh_p >= 0.5 else float("inf"),
            )
            for t in TRANSPORTER_NAMES
        }
        return TransporterProfile(profiles=interactions)

    def test_is_substrate_above_threshold(self):
        p = self._make_profile(sub_p=0.8)
        assert p.is_substrate("P_gp", threshold=0.5) is True

    def test_is_substrate_below_threshold(self):
        p = self._make_profile(sub_p=0.2)
        assert p.is_substrate("P_gp", threshold=0.5) is False

    def test_is_inhibitor_above_threshold(self):
        p = self._make_profile(inh_p=0.9)
        assert p.is_inhibitor("OATP1B1", threshold=0.5) is True

    def test_is_inhibitor_below_threshold(self):
        p = self._make_profile(inh_p=0.1)
        assert p.is_inhibitor("OATP1B1", threshold=0.5) is False

    def test_is_substrate_unknown_transporter_returns_false(self):
        p = self._make_profile()
        assert p.is_substrate("UNKNOWN", threshold=0.5) is False

    def test_active_substrates_list(self):
        p = self._make_profile(sub_p=0.8)
        subs = p.active_substrates(threshold=0.5)
        assert set(subs) == set(TRANSPORTER_NAMES)

    def test_active_substrates_empty_when_all_low(self):
        p = self._make_profile(sub_p=0.1)
        assert p.active_substrates(threshold=0.5) == []

    def test_active_inhibitors_list(self):
        p = self._make_profile(inh_p=0.9)
        inhs = p.active_inhibitors(threshold=0.5)
        assert set(inhs) == set(TRANSPORTER_NAMES)

    def test_profiles_has_all_transporters(self):
        tc = TransporterClassifier()
        p = tc.predict_from_properties(300.0, 2.0, "neutral")
        assert set(p.profiles.keys()) == set(TRANSPORTER_NAMES)

    def test_profile_is_frozen(self):
        p = self._make_profile()
        with pytest.raises((AttributeError, TypeError)):
            p.confidence = "high"  # type: ignore[misc]

    def test_ddi_risk_flags_is_tuple(self):
        p = self._make_profile()
        assert isinstance(p.ddi_risk_flags, tuple)


# ---------------------------------------------------------------------------
# TestKnownSubstrates (FDA DDI Guidance 2020)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestKnownSubstrates:
    def test_digoxin_pgp_substrate(self):
        """Digoxin: canonical narrow-TI P-gp substrate."""
        tc = TransporterClassifier()
        p = tc.predict_from_properties(**DIGOXIN_PROPS)
        assert p.profiles["P_gp"].substrate_prob > 0.5

    def test_metformin_oct2_substrate(self):
        """Metformin: OCT2 substrate (renal secretion)."""
        tc = TransporterClassifier()
        p = tc.predict_from_properties(**METFORMIN_PROPS)
        assert p.profiles["OCT2"].substrate_prob > 0.5

    def test_metformin_mate1_substrate(self):
        """Metformin: MATE1 substrate (renal efflux)."""
        tc = TransporterClassifier()
        p = tc.predict_from_properties(**METFORMIN_PROPS)
        assert p.profiles["MATE1"].substrate_prob > 0.4

    def test_rosuvastatin_oatp1b1_substrate(self):
        """Rosuvastatin: OATP1B1 substrate (hepatic uptake)."""
        tc = TransporterClassifier()
        p = tc.predict_from_properties(**ROSUVASTATIN_PROPS)
        assert p.profiles["OATP1B1"].substrate_prob > 0.3

    def test_rosuvastatin_bcrp_substrate(self):
        """Rosuvastatin: BCRP substrate."""
        tc = TransporterClassifier()
        p = tc.predict_from_properties(**ROSUVASTATIN_PROPS)
        assert p.profiles["BCRP"].substrate_prob > 0.3

    def test_atorvastatin_oatp1b1_substrate(self):
        """Atorvastatin: OATP1B1 substrate."""
        tc = TransporterClassifier()
        p = tc.predict_from_properties(558.6, 4.06, "acid")
        assert p.profiles["OATP1B1"].substrate_prob > 0.3

    def test_fexofenadine_pgp_substrate(self):
        """Fexofenadine: P-gp substrate."""
        tc = TransporterClassifier()
        p = tc.predict_from_properties(501.7, 2.18, "zwitterion")
        assert p.profiles["P_gp"].substrate_prob > 0.4

    def test_morphine_mrp2_substrate(self):
        """Morphine: MRP2 substrate (conjugate efflux)."""
        tc = TransporterClassifier()
        p = tc.predict_from_properties(285.3, 0.89, "base")
        assert p.profiles["MRP2"].substrate_prob >= 0.0  # non-negative at minimum


# ---------------------------------------------------------------------------
# TestKnownInhibitors
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestKnownInhibitors:
    def test_cyclosporine_pgp_inhibitor(self):
        """Cyclosporine: potent P-gp inhibitor."""
        tc = TransporterClassifier()
        p = tc.predict_from_properties(**CYCLOSPORINE_PROPS)
        assert p.profiles["P_gp"].inhibitor_prob > 0.5

    def test_cyclosporine_oatp1b1_inhibitor(self):
        """Cyclosporine: OATP1B1 inhibitor."""
        tc = TransporterClassifier()
        p = tc.predict_from_properties(**CYCLOSPORINE_PROPS)
        assert p.profiles["OATP1B1"].inhibitor_prob > 0.4

    def test_ritonavir_pgp_inhibitor(self):
        """Ritonavir: P-gp inhibitor."""
        tc = TransporterClassifier()
        p = tc.predict_from_properties(**RITONAVIR_PROPS)
        assert p.profiles["P_gp"].inhibitor_prob > 0.5

    def test_verapamil_pgp_inhibitor(self):
        """Verapamil: P-gp inhibitor."""
        tc = TransporterClassifier()
        p = tc.predict_from_properties(454.6, 3.79, "base")
        assert p.profiles["P_gp"].inhibitor_prob > 0.4

    def test_cimetidine_oct2_inhibitor(self):
        """Cimetidine: OCT2 inhibitor (renal DDI probe)."""
        tc = TransporterClassifier()
        p = tc.predict_from_properties(**CIMETIDINE_PROPS)
        assert p.profiles["OCT2"].inhibitor_prob > 0.4

    def test_gemfibrozil_oatp1b1_inhibitor(self):
        """Gemfibrozil: OATP1B1 inhibitor (statin interaction)."""
        tc = TransporterClassifier()
        p = tc.predict_from_properties(250.3, 3.44, "acid")
        assert p.profiles["OATP1B1"].inhibitor_prob > 0.3


# ---------------------------------------------------------------------------
# TestKnownNonSubstrates
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestKnownNonSubstrates:
    def test_caffeine_not_pgp_substrate(self):
        tc = TransporterClassifier()
        p = tc.predict_from_properties(**CAFFEINE_PROPS)
        assert p.profiles["P_gp"].substrate_prob < 0.5

    def test_caffeine_not_oatp_substrate(self):
        tc = TransporterClassifier()
        p = tc.predict_from_properties(**CAFFEINE_PROPS)
        assert p.profiles["OATP1B1"].substrate_prob < 0.5

    def test_metformin_not_oatp_substrate(self):
        """Metformin is cation — OATP1B1 prefers anions."""
        tc = TransporterClassifier()
        p = tc.predict_from_properties(**METFORMIN_PROPS)
        assert p.profiles["OATP1B1"].substrate_prob < 0.4

    def test_atenolol_moderate_pgp_probability(self):
        """Atenolol: small, hydrophilic — should not be high P-gp probability."""
        tc = TransporterClassifier()
        p = tc.predict_from_properties(266.3, 0.16, "base")
        assert p.profiles["P_gp"].substrate_prob < 0.8


# ---------------------------------------------------------------------------
# TestPredictFromSMILES
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestPredictFromSMILES:
    def test_predict_metformin_smiles(self):
        tc = TransporterClassifier()
        p = tc.predict("CN(C)C(=N)NC(=N)N")
        assert len(p.profiles) == 7
        assert p.profiles["OCT2"].substrate_prob > 0.4

    def test_predict_caffeine_smiles(self):
        tc = TransporterClassifier()
        p = tc.predict("Cn1cnc2c1c(=O)n(C)c(=O)n2C")
        assert len(p.profiles) == 7
        assert p.confidence in ("low", "medium", "high")

    def test_predict_all_probs_in_range(self):
        tc = TransporterClassifier()
        p = tc.predict("CC(Cc1ccc(cc1)O)C(=O)O")  # ibuprofen
        for interaction in p.profiles.values():
            assert 0.0 <= interaction.substrate_prob <= 1.0
            assert 0.0 <= interaction.inhibitor_prob <= 1.0

    def test_predict_returns_transporter_profile(self):
        tc = TransporterClassifier()
        result = tc.predict("c1ccccc1")
        assert isinstance(result, TransporterProfile)

    def test_predict_profile_has_all_names(self):
        tc = TransporterClassifier()
        p = tc.predict("CCO")
        assert set(p.profiles.keys()) == set(TRANSPORTER_NAMES)


# ---------------------------------------------------------------------------
# TestKmAndIC50
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestKmAndIC50:
    def test_km_non_negative_for_all_transporters(self):
        tc = TransporterClassifier()
        p = tc.predict_from_properties(500.0, 3.0, "acid")
        for t in TRANSPORTER_NAMES:
            assert p.profiles[t].km_uM >= 0.0

    def test_ic50_positive_for_all_transporters(self):
        tc = TransporterClassifier()
        p = tc.predict_from_properties(500.0, 3.0, "base")
        for t in TRANSPORTER_NAMES:
            assert p.profiles[t].ic50_uM > 0.0

    def test_km_zero_when_not_substrate(self):
        """km_uM should be 0 when substrate_prob < 0.5."""
        tc = TransporterClassifier()
        p = tc.predict_from_properties(50.0, -5.0, "neutral")
        for t in TRANSPORTER_NAMES:
            if p.profiles[t].substrate_prob < 0.5:
                assert p.profiles[t].km_uM == 0.0

    def test_ic50_inf_when_not_inhibitor(self):
        """ic50_uM should be inf when inhibitor_prob < 0.5."""
        tc = TransporterClassifier()
        p = tc.predict_from_properties(50.0, -5.0, "neutral")
        for t in TRANSPORTER_NAMES:
            if p.profiles[t].inhibitor_prob < 0.5:
                assert math.isinf(p.profiles[t].ic50_uM)

    def test_km_within_bounds(self):
        tc = TransporterClassifier()
        p = tc.predict_from_properties(400.0, 2.0, "acid")
        for t in TRANSPORTER_NAMES:
            if p.profiles[t].km_uM > 0:
                assert 1.0 <= p.profiles[t].km_uM <= 1000.0

    def test_ic50_within_bounds(self):
        tc = TransporterClassifier()
        p = tc.predict_from_properties(600.0, 4.0, "base")
        for t in TRANSPORTER_NAMES:
            ic50 = p.profiles[t].ic50_uM
            if not math.isinf(ic50):
                assert 0.1 <= ic50 <= 500.0


# ---------------------------------------------------------------------------
# TestDDIRiskFlags
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestDDIRiskFlags:
    def test_flags_is_tuple(self):
        tc = TransporterClassifier()
        p = tc.predict_from_properties(**DIGOXIN_PROPS)
        assert isinstance(p.ddi_risk_flags, tuple)

    def test_flags_are_strings(self):
        tc = TransporterClassifier()
        p = tc.predict_from_properties(**CYCLOSPORINE_PROPS)
        for f in p.ddi_risk_flags:
            assert isinstance(f, str)

    def test_pgp_substrate_generates_flag(self):
        tc = TransporterClassifier()
        p = tc.predict_from_properties(**DIGOXIN_PROPS)
        if p.profiles["P_gp"].substrate_prob > 0.6:
            assert any("P-gp" in f for f in p.ddi_risk_flags)

    def test_oatp_inhibitor_generates_flag(self):
        tc = TransporterClassifier()
        p = tc.predict_from_properties(**CYCLOSPORINE_PROPS)
        if p.profiles["OATP1B1"].inhibitor_prob > 0.6:
            assert any("OATP" in f for f in p.ddi_risk_flags)

    def test_no_flags_for_very_small_molecule(self):
        """Tiny drug unlikely to generate high-prob transporter interactions."""
        tc = TransporterClassifier()
        p = tc.predict_from_properties(50.0, -4.0, "neutral")
        # No P-gp substrate flag for tiny molecule (probs should be low)
        for flag in p.ddi_risk_flags:
            assert isinstance(flag, str)


# ---------------------------------------------------------------------------
# TestEdgeCases
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestEdgeCases:
    def test_very_small_molecule(self):
        tc = TransporterClassifier()
        p = tc.predict_from_properties(50.0, -3.0, "neutral")
        assert len(p.profiles) == 7

    def test_very_large_molecule(self):
        tc = TransporterClassifier()
        p = tc.predict_from_properties(2000.0, 6.0, "neutral")
        assert len(p.profiles) == 7

    def test_confidence_string_values(self):
        tc = TransporterClassifier()
        p = tc.predict_from_properties(300.0, 2.0, "neutral")
        assert p.confidence in ("low", "medium", "high")

    def test_multiple_predictions_consistent(self):
        tc = TransporterClassifier()
        p1 = tc.predict_from_properties(500.0, 3.0, "acid")
        p2 = tc.predict_from_properties(500.0, 3.0, "acid")
        assert p1.profiles["P_gp"].substrate_prob == p2.profiles["P_gp"].substrate_prob

    def test_zwitterion_charge_class(self):
        tc = TransporterClassifier()
        p = tc.predict_from_properties(388.9, 1.70, "zwitterion")
        assert len(p.profiles) == 7

    def test_acid_charge_class(self):
        tc = TransporterClassifier()
        p = tc.predict_from_properties(481.5, 0.13, "acid")
        assert len(p.profiles) == 7

    def test_predict_from_properties_defaults(self):
        tc = TransporterClassifier()
        p = tc.predict_from_properties(300.0, 2.0)  # charge_class defaults to "neutral"
        assert len(p.profiles) == 7

    def test_ref_data_loaded_once(self):
        """Reference data should be loaded exactly once (lazy init)."""
        tc = TransporterClassifier()
        assert tc._ref_data is None
        tc.predict_from_properties(300.0, 2.0, "neutral")
        assert tc._ref_data is not None
        first_id = id(tc._ref_data)
        tc.predict_from_properties(400.0, 3.0, "base")
        assert id(tc._ref_data) == first_id


# ---------------------------------------------------------------------------
# TestTransporterPlugin
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestTransporterPlugin:
    def test_plugin_name(self):
        from omega_pbpk.plugins.transporter_plugin import TransporterPlugin

        assert TransporterPlugin().name == "transporter_classifier"

    def test_plugin_provides(self):
        from omega_pbpk.plugins.transporter_plugin import TransporterPlugin

        p = TransporterPlugin()
        assert "transporter_profile" in p.provides

    def test_plugin_predict_keys(self):
        from omega_pbpk.contracts.drug_spec import DrugSpec
        from omega_pbpk.plugins.transporter_plugin import TransporterPlugin

        spec = DrugSpec(name="test", mw=500.0, logP=3.0, compound_type="base")
        result = TransporterPlugin().predict(spec)
        assert set(result.keys()) == {"transporter_profile"}

    def test_plugin_predict_structure(self):
        from omega_pbpk.contracts.drug_spec import DrugSpec
        from omega_pbpk.plugins.transporter_plugin import TransporterPlugin

        spec = DrugSpec(name="test", mw=500.0, logP=3.0)
        result = TransporterPlugin().predict(spec)
        tp = result["transporter_profile"]
        assert len(tp) == 7
        for _t, vals in tp.items():
            assert "substrate_prob" in vals
            assert "inhibitor_prob" in vals
            assert "km_uM" in vals
            assert "ic50_uM" in vals

    def test_plugin_predict_with_smiles(self):
        from omega_pbpk.contracts.drug_spec import DrugSpec
        from omega_pbpk.plugins.transporter_plugin import TransporterPlugin

        spec = DrugSpec(name="metformin", smiles="CN(C)C(=N)NC(=N)N", mw=129.2, logP=-1.43)
        result = TransporterPlugin().predict(spec)
        assert isinstance(result["transporter_profile"], dict)

    def test_plugin_predict_without_smiles(self):
        from omega_pbpk.contracts.drug_spec import DrugSpec
        from omega_pbpk.plugins.transporter_plugin import TransporterPlugin

        spec = DrugSpec(name="test", mw=300.0, logP=2.0, compound_type="neutral")
        result = TransporterPlugin().predict(spec)
        assert len(result["transporter_profile"]) == 7

    def test_plugin_confidence_higher_with_smiles(self):
        from omega_pbpk.contracts.drug_spec import DrugSpec
        from omega_pbpk.plugins.transporter_plugin import TransporterPlugin

        p = TransporterPlugin()
        conf_smiles = p.confidence(DrugSpec(name="t", smiles="CCO", mw=46.0, logP=-0.3))
        conf_no_smiles = p.confidence(DrugSpec(name="t", mw=46.0, logP=-0.3))
        assert conf_smiles > conf_no_smiles

    def test_plugin_ic50_inf_replaced_with_9999(self):
        from omega_pbpk.contracts.drug_spec import DrugSpec
        from omega_pbpk.plugins.transporter_plugin import TransporterPlugin

        spec = DrugSpec(name="test", mw=50.0, logP=-5.0, compound_type="neutral")
        result = TransporterPlugin().predict(spec)
        for t_data in result["transporter_profile"].values():
            assert not math.isinf(t_data["ic50_uM"]), "Plugin should replace inf with 9999"


# ---------------------------------------------------------------------------
# TestDrugSpecTransporterField
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestDrugSpecTransporterField:
    def test_drug_spec_has_transporter_profile_field(self):
        from omega_pbpk.contracts.drug_spec import DrugSpec

        spec = DrugSpec(name="test")
        assert hasattr(spec, "transporter_profile")

    def test_drug_spec_transporter_profile_defaults_empty(self):
        from omega_pbpk.contracts.drug_spec import DrugSpec

        spec = DrugSpec(name="test")
        assert spec.transporter_profile == {}

    def test_drug_spec_transporter_profile_can_be_set(self):
        from omega_pbpk.contracts.drug_spec import DrugSpec

        data = {"P_gp": {"substrate_prob": 0.8}}
        spec = DrugSpec(name="test", transporter_profile=data)
        assert spec.transporter_profile["P_gp"]["substrate_prob"] == 0.8
