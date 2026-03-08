"""Tests for solubility pH profile prediction (Phase 594)."""

import pytest

from omega_pbpk.prediction.solubility_ph_profile_p594 import (
    SolubilityPHProfileResult,
    predict_solubility_ph_profile,
    screen_solubility_ph,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def acid_result():
    return predict_solubility_ph_profile("AsprinLike", 0.1, "acid", pka1=4.5)


@pytest.fixture
def base_result():
    return predict_solubility_ph_profile("AmineDrug", 0.05, "base", pka1=8.5)


@pytest.fixture
def neutral_result():
    return predict_solubility_ph_profile("Neutral", 2.0, "neutral")


@pytest.fixture
def amphoteric_result():
    return predict_solubility_ph_profile("AmphoChar", 0.01, "amphoteric", pka1=4.0, pka2=9.0)


# ---------------------------------------------------------------------------
# Return type tests
# ---------------------------------------------------------------------------


class TestReturnType:
    def test_returns_correct_type(self, acid_result):
        assert isinstance(acid_result, SolubilityPHProfileResult)

    def test_drug_name_preserved(self, acid_result):
        assert acid_result.drug_name == "AsprinLike"

    def test_drug_type_preserved(self, acid_result):
        assert acid_result.drug_type == "acid"

    def test_s0_preserved(self, acid_result):
        assert acid_result.s0_mg_mL == pytest.approx(0.1)

    def test_ph_values_is_list(self, acid_result):
        assert isinstance(acid_result.ph_values, list)

    def test_solubility_is_list(self, acid_result):
        assert isinstance(acid_result.solubility_mg_mL, list)

    def test_pka_values_is_list(self, acid_result):
        assert isinstance(acid_result.pka_values, list)

    def test_lists_same_length(self, acid_result):
        assert len(acid_result.ph_values) == len(acid_result.solubility_mg_mL)

    def test_default_n_points(self, acid_result):
        assert len(acid_result.ph_values) == 37

    def test_ph_range_starts_at_1(self, acid_result):
        assert acid_result.ph_values[0] == pytest.approx(1.0)

    def test_ph_range_ends_at_10(self, acid_result):
        assert acid_result.ph_values[-1] == pytest.approx(10.0)

    def test_notes_is_str(self, acid_result):
        assert isinstance(acid_result.notes, str)
        assert len(acid_result.notes) > 0

    def test_bcs_class_is_str(self, acid_result):
        assert acid_result.bcs_solubility_class in ("high", "low")


# ---------------------------------------------------------------------------
# Acid profile tests
# ---------------------------------------------------------------------------


class TestAcidProfile:
    def test_acid_pka_in_result(self, acid_result):
        assert 4.5 in acid_result.pka_values

    def test_acid_increases_with_ph(self, acid_result):
        """Acid solubility increases at alkaline pH."""
        ph_vals = acid_result.ph_values
        sol_vals = acid_result.solubility_mg_mL
        # Compare pH 2 region vs pH 9 region
        idx_low = min(range(len(ph_vals)), key=lambda i: abs(ph_vals[i] - 2.0))
        idx_high = min(range(len(ph_vals)), key=lambda i: abs(ph_vals[i] - 9.0))
        assert sol_vals[idx_high] > sol_vals[idx_low]

    def test_acid_s0_at_very_low_ph(self):
        """At pH << pKa, acid S(pH) ≈ S0."""
        r = predict_solubility_ph_profile("Acid", 1.0, "acid", pka1=7.0)
        # At pH 1, 10^(1-7)=1e-6, so S ≈ S0*(1+1e-6) ≈ S0
        sol_at_1 = r.solubility_mg_mL[0]
        assert sol_at_1 == pytest.approx(1.0 * (1 + 10 ** (1.0 - 7.0)), rel=1e-4)

    def test_acid_pka_formula(self):
        """At pH=pKa, S(pKa) = 2*S0."""
        pka = 5.0
        s0 = 0.5
        r = predict_solubility_ph_profile("Acid", s0, "acid", pka1=pka)
        # Find closest pH to pKa
        idx = min(range(len(r.ph_values)), key=lambda i: abs(r.ph_values[i] - pka))
        ph_at_idx = r.ph_values[idx]
        expected = s0 * (1 + 10 ** (ph_at_idx - pka))
        assert r.solubility_mg_mL[idx] == pytest.approx(expected, rel=1e-5)


# ---------------------------------------------------------------------------
# Base profile tests
# ---------------------------------------------------------------------------


class TestBaseProfile:
    def test_base_pka_in_result(self, base_result):
        assert 8.5 in base_result.pka_values

    def test_base_decreases_with_ph(self, base_result):
        """Base solubility decreases at alkaline pH."""
        ph_vals = base_result.ph_values
        sol_vals = base_result.solubility_mg_mL
        idx_low = min(range(len(ph_vals)), key=lambda i: abs(ph_vals[i] - 2.0))
        idx_high = min(range(len(ph_vals)), key=lambda i: abs(ph_vals[i] - 9.0))
        assert sol_vals[idx_low] > sol_vals[idx_high]

    def test_base_s0_at_very_high_ph(self):
        """At pH >> pKa, base S(pH) ≈ S0."""
        r = predict_solubility_ph_profile("Base", 1.0, "base", pka1=3.0)
        # At pH 10, 10^(3-10)=1e-7, so S ≈ S0
        sol_at_10 = r.solubility_mg_mL[-1]
        assert sol_at_10 == pytest.approx(1.0 * (1 + 10 ** (3.0 - 10.0)), rel=1e-4)

    def test_base_pka_formula(self):
        """At pH=pKa, S(pKa) = 2*S0 for base."""
        pka = 6.0
        s0 = 0.3
        r = predict_solubility_ph_profile("Base", s0, "base", pka1=pka)
        idx = min(range(len(r.ph_values)), key=lambda i: abs(r.ph_values[i] - pka))
        ph_at_idx = r.ph_values[idx]
        expected = s0 * (1 + 10 ** (pka - ph_at_idx))
        assert r.solubility_mg_mL[idx] == pytest.approx(expected, rel=1e-5)


# ---------------------------------------------------------------------------
# Neutral profile tests
# ---------------------------------------------------------------------------


class TestNeutralProfile:
    def test_neutral_constant_solubility(self, neutral_result):
        """Neutral drug solubility is constant across all pH."""
        sols = neutral_result.solubility_mg_mL
        assert all(abs(s - 2.0) < 1e-9 for s in sols)

    def test_neutral_pka_values_empty(self, neutral_result):
        assert neutral_result.pka_values == []

    def test_neutral_bcs_high_when_s0_large(self):
        r = predict_solubility_ph_profile("Neutral", 5.0, "neutral")
        assert r.bcs_solubility_class == "high"

    def test_neutral_bcs_low_when_s0_small(self):
        r = predict_solubility_ph_profile("Neutral", 0.001, "neutral")
        assert r.bcs_solubility_class == "low"


# ---------------------------------------------------------------------------
# Amphoteric profile tests
# ---------------------------------------------------------------------------


class TestAmphotericProfile:
    def test_amphoteric_pka_values(self, amphoteric_result):
        assert 4.0 in amphoteric_result.pka_values
        assert 9.0 in amphoteric_result.pka_values

    def test_amphoteric_u_shape(self, amphoteric_result):
        """Amphoteric drugs have higher solubility at extremes, lower near pI."""
        ph_vals = amphoteric_result.ph_values
        sol_vals = amphoteric_result.solubility_mg_mL
        idx_low = min(range(len(ph_vals)), key=lambda i: abs(ph_vals[i] - 1.5))
        idx_mid = min(range(len(ph_vals)), key=lambda i: abs(ph_vals[i] - 6.5))
        idx_high = min(range(len(ph_vals)), key=lambda i: abs(ph_vals[i] - 9.5))
        # At extremes (low and high pH) solubility should be higher than mid
        assert sol_vals[idx_low] > sol_vals[idx_mid]
        assert sol_vals[idx_high] > sol_vals[idx_mid]

    def test_amphoteric_formula(self):
        """Verify amphoteric formula at a specific pH."""
        s0, pka1, pka2 = 0.01, 4.0, 9.0
        ph = 7.0
        r = predict_solubility_ph_profile("Amph", s0, "amphoteric", pka1=pka1, pka2=pka2)
        idx = min(range(len(r.ph_values)), key=lambda i: abs(r.ph_values[i] - ph))
        ph_at_idx = r.ph_values[idx]
        expected = s0 * (1 + 10 ** (ph_at_idx - pka1) + 10 ** (pka2 - ph_at_idx))
        assert r.solubility_mg_mL[idx] == pytest.approx(expected, rel=1e-5)


# ---------------------------------------------------------------------------
# BCS class and key pH points
# ---------------------------------------------------------------------------


class TestBCSAndKeyPH:
    def test_bcs_high_acid_alkaline(self):
        """Acid with high S0 dissolved at GI pH should be high BCS."""
        r = predict_solubility_ph_profile("HighSolAcid", 5.0, "acid", pka1=4.0)
        assert r.bcs_solubility_class == "high"

    def test_bcs_low_hydrophobic_neutral(self):
        """Very insoluble neutral drug should be BCS low."""
        r = predict_solubility_ph_profile("Insoluble", 0.0001, "neutral")
        assert r.bcs_solubility_class == "low"

    def test_solubility_fasted_gi_computed(self, acid_result):
        """solubility_fasted_gi should match formula at pH 6.5."""
        s0 = acid_result.s0_mg_mL
        pka1 = acid_result.pka_values[0]
        expected = min(1000.0, s0 * (1 + 10 ** (6.5 - pka1)))
        assert acid_result.solubility_fasted_gi == pytest.approx(expected, rel=1e-5)

    def test_solubility_gastric_computed(self, base_result):
        """solubility_gastric should match formula at pH 1.2 for base."""
        s0 = base_result.s0_mg_mL
        pka1 = base_result.pka_values[0]
        expected = min(1000.0, s0 * (1 + 10 ** (pka1 - 1.2)))
        assert base_result.solubility_gastric == pytest.approx(expected, rel=1e-4)

    def test_solubility_at_ph_max_equals_max(self, acid_result):
        assert acid_result.solubility_at_ph_max == pytest.approx(
            max(acid_result.solubility_mg_mL), rel=1e-9
        )

    def test_cap_at_1000_mg_mL(self):
        """Very soluble drug capped at 1000 mg/mL."""
        r = predict_solubility_ph_profile("SuperSoluble", 500.0, "acid", pka1=4.0)
        assert all(s <= 1000.0 for s in r.solubility_mg_mL)


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


class TestValidation:
    def test_invalid_drug_type_raises(self):
        with pytest.raises(ValueError, match="drug_type"):
            predict_solubility_ph_profile("D", 1.0, "zwitterion")

    def test_acid_without_pka_raises(self):
        with pytest.raises(ValueError, match="pka1"):
            predict_solubility_ph_profile("D", 1.0, "acid")

    def test_base_without_pka_raises(self):
        with pytest.raises(ValueError, match="pka1"):
            predict_solubility_ph_profile("D", 1.0, "base")

    def test_amphoteric_without_pka2_raises(self):
        with pytest.raises(ValueError, match="pka"):
            predict_solubility_ph_profile("D", 1.0, "amphoteric", pka1=4.0)

    def test_zero_s0_raises(self):
        with pytest.raises(ValueError, match="s0_mg_mL"):
            predict_solubility_ph_profile("D", 0.0, "neutral")

    def test_negative_s0_raises(self):
        with pytest.raises(ValueError, match="s0_mg_mL"):
            predict_solubility_ph_profile("D", -0.1, "acid", pka1=5.0)


# ---------------------------------------------------------------------------
# Screening function
# ---------------------------------------------------------------------------


class TestScreenSolubilityPH:
    def test_returns_list(self):
        compounds = [
            {"drug_name": "A", "s0_mg_mL": 1.0, "drug_type": "neutral"},
            {"drug_name": "B", "s0_mg_mL": 0.01, "drug_type": "acid", "pka1": 4.0},
        ]
        results = screen_solubility_ph(compounds)
        assert isinstance(results, list)

    def test_sorted_by_fasted_gi_descending(self):
        compounds = [
            {"drug_name": "Low", "s0_mg_mL": 0.001, "drug_type": "neutral"},
            {"drug_name": "Mid", "s0_mg_mL": 0.1, "drug_type": "neutral"},
            {"drug_name": "High", "s0_mg_mL": 10.0, "drug_type": "neutral"},
        ]
        results = screen_solubility_ph(compounds)
        fgi = [r.solubility_fasted_gi for r in results]
        assert fgi == sorted(fgi, reverse=True)

    def test_returns_all_compounds(self):
        compounds = [
            {"drug_name": f"Drug{i}", "s0_mg_mL": 0.1 * (i + 1), "drug_type": "neutral"}
            for i in range(5)
        ]
        results = screen_solubility_ph(compounds)
        assert len(results) == 5

    def test_returns_solubility_ph_profile_results(self):
        compounds = [{"drug_name": "D", "s0_mg_mL": 1.0, "drug_type": "neutral"}]
        results = screen_solubility_ph(compounds)
        assert isinstance(results[0], SolubilityPHProfileResult)
