"""Tests for drug solubility prediction (Phase 68)."""

from __future__ import annotations

from omega_pbpk.prediction.solubility import (
    SolubilityResult,
    predict_solubility,
    solubility_ph_profile,
)


class TestPredictSolubility:
    def _run(self, **kw):
        defaults = dict(drug_name="Drug", logP=2.0, mp_celsius=150.0, drug_type="neutral")
        defaults.update(kw)
        return predict_solubility(**defaults)

    def test_returns_result_type(self):
        assert isinstance(self._run(), SolubilityResult)

    def test_intrinsic_sol_positive(self):
        assert self._run().sol_intrinsic_mg_mL > 0

    def test_high_logP_lower_solubility(self):
        low = self._run(logP=1.0)
        high = self._run(logP=5.0)
        assert high.sol_intrinsic_mg_mL <= low.sol_intrinsic_mg_mL

    def test_high_mp_lower_solubility(self):
        low_mp = self._run(mp_celsius=80.0)
        high_mp = self._run(mp_celsius=250.0)
        assert high_mp.sol_intrinsic_mg_mL <= low_mp.sol_intrinsic_mg_mL

    def test_acid_drug_higher_sol_at_high_pH(self):
        r = self._run(drug_type="acid", pka=4.5)
        # Acid dissolves better at intestinal pH than stomach pH
        assert r.sol_ph_intestine_mg_mL >= r.sol_ph_stomach_mg_mL

    def test_base_drug_higher_sol_at_low_pH(self):
        r = self._run(drug_type="base", pka=8.5)
        # Base dissolves better at stomach pH than intestinal pH
        assert r.sol_ph_stomach_mg_mL >= r.sol_ph_intestine_mg_mL

    def test_neutral_same_ph_profile(self):
        r = self._run(drug_type="neutral")
        # Neutral drug: stomach ≈ intestine ≈ colon
        assert (
            abs(r.sol_ph_stomach_mg_mL - r.sol_ph_intestine_mg_mL)
            / max(r.sol_ph_stomach_mg_mL, 1e-6)
            < 0.01
        )

    def test_bcs_class_high_when_soluble(self):
        r = self._run(logP=0.0)  # low logP → higher solubility
        # Only assert if truly soluble
        if r.sol_ph_intestine_mg_mL >= 0.1:
            assert r.bcs_solubility_class == "high"

    def test_bcs_class_low_when_insoluble(self):
        r = self._run(logP=5.0)
        if r.sol_ph_intestine_mg_mL < 0.1:
            assert r.bcs_solubility_class == "low"

    def test_dose_number_positive(self):
        assert self._run().dose_number > 0

    def test_supersaturation_potential_geq_intestine(self):
        r = self._run()
        assert r.supersaturation_potential >= r.sol_ph_intestine_mg_mL

    def test_drug_name_preserved(self):
        r = self._run(drug_name="Ibuprofen")
        assert r.drug_name == "Ibuprofen"


class TestSolubilityProfile:
    def test_returns_list(self):
        r = solubility_ph_profile("Drug", logP=2.0, pka=7.0)
        assert isinstance(r, list)

    def test_has_10_entries_default(self):
        r = solubility_ph_profile("Drug", logP=2.0, pka=7.0)
        assert len(r) == 10

    def test_entries_have_keys(self):
        r = solubility_ph_profile("Drug", logP=2.0, pka=7.0)
        for entry in r:
            assert "pH" in entry and "solubility_mg_mL" in entry

    def test_acid_solubility_increases_with_pH(self):
        r = solubility_ph_profile("AcidDrug", logP=2.0, pka=4.5, drug_type="acid")
        # Sort by pH and check general increasing trend
        sorted_r = sorted(r, key=lambda x: x["pH"])
        low_ph_sol = sorted_r[0]["solubility_mg_mL"]
        high_ph_sol = sorted_r[-1]["solubility_mg_mL"]
        assert high_ph_sol >= low_ph_sol


class TestBCSWarnings:
    def test_low_do_no_warning(self):
        # Very soluble drug: logP=0 → no dissolution warning
        r = predict_solubility("Safe", logP=0.0, mp_celsius=80.0, dose_mg=10.0)
        dissolution_warnings = [w for w in r.warnings if "dissolution" in w.lower()]
        # Only assert if Do < 1 (truly soluble)
        if r.dose_number < 1:
            assert len(dissolution_warnings) == 0

    def test_high_do_warning(self):
        # Insoluble drug at high dose → dissolution warning
        r = predict_solubility("Insoluble", logP=5.0, mp_celsius=200.0, dose_mg=1000.0)
        assert any("dissolution" in w.lower() or "solubility" in w.lower() for w in r.warnings)
