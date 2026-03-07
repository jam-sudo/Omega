"""Tests for biopharmaceutics/solubilization_predictor.py — Phase 465."""

from __future__ import annotations

import pytest

from omega_pbpk.biopharmaceutics.solubilization_predictor import (
    GISolubilityResult,
    absorption_limited_by,
    dose_number,
    micellar_solubilization,
    predict_gi_solubility,
    solubilization_ratio,
)

# ---------------------------------------------------------------------------
# micellar_solubilization
# ---------------------------------------------------------------------------


class TestMicellarSolubilization:
    def test_returns_positive_float(self):
        s = micellar_solubilization(drug_logP=2.0, drug_mw=300.0)
        assert s > 0.0

    def test_higher_logp_greater_micellar_enhancement(self):
        """More lipophilic drug → larger Kmc → more micellar enhancement."""
        s_low = micellar_solubilization(
            drug_logP=1.0, drug_mw=300.0, bile_salt_mM=5.0, lecithin_mM=1.5
        )
        s_high = micellar_solubilization(
            drug_logP=5.0, drug_mw=300.0, bile_salt_mM=5.0, lecithin_mM=1.5
        )
        assert s_high > s_low

    def test_higher_bile_salt_increases_solubility(self):
        s_low_bile = micellar_solubilization(
            drug_logP=3.0, drug_mw=300.0, bile_salt_mM=2.0, lecithin_mM=1.0
        )
        s_high_bile = micellar_solubilization(
            drug_logP=3.0, drug_mw=300.0, bile_salt_mM=15.0, lecithin_mM=1.0
        )
        assert s_high_bile >= s_low_bile

    def test_fed_concentrations_give_higher_than_fasted(self):
        """Fed: bile ~15 mM, lecithin ~3.75 mM >> fasted: ~3/0.8 mM."""
        s_fasted = micellar_solubilization(
            drug_logP=3.0, drug_mw=300.0, bile_salt_mM=3.0, lecithin_mM=0.8, state="fasted"
        )
        s_fed = micellar_solubilization(
            drug_logP=3.0, drug_mw=300.0, bile_salt_mM=15.0, lecithin_mM=3.75, state="fed"
        )
        assert s_fed > s_fasted

    def test_zero_bile_returns_intrinsic_only(self):
        """No micelles → solubility approaches intrinsic (CMC not exceeded)."""
        s = micellar_solubilization(drug_logP=2.0, drug_mw=300.0, bile_salt_mM=0.0, lecithin_mM=0.0)
        s_baseline = micellar_solubilization(
            drug_logP=2.0, drug_mw=300.0, bile_salt_mM=5.0, lecithin_mM=1.5
        )
        assert s <= s_baseline

    def test_fasted_state_accepted(self):
        s = micellar_solubilization(drug_logP=2.0, drug_mw=300.0, state="fasted")
        assert s > 0

    def test_fed_state_accepted(self):
        s = micellar_solubilization(drug_logP=2.0, drug_mw=300.0, state="fed")
        assert s > 0

    def test_invalid_state_raises(self):
        with pytest.raises(ValueError, match="state must be one of"):
            micellar_solubilization(drug_logP=2.0, drug_mw=300.0, state="unknown")

    def test_invalid_mw_raises(self):
        with pytest.raises(ValueError, match="drug_mw must be positive"):
            micellar_solubilization(drug_logP=2.0, drug_mw=0.0)

    def test_negative_bile_raises(self):
        with pytest.raises(ValueError):
            micellar_solubilization(drug_logP=2.0, drug_mw=300.0, bile_salt_mM=-1.0)

    def test_hydrophilic_drug_low_logp(self):
        """logP < 0 → very low micellar enhancement but still valid."""
        s = micellar_solubilization(drug_logP=-1.0, drug_mw=200.0)
        assert s > 0

    def test_very_lipophilic_drug(self):
        s = micellar_solubilization(
            drug_logP=6.0, drug_mw=500.0, bile_salt_mM=15.0, lecithin_mM=3.75
        )
        assert s > 0.01


# ---------------------------------------------------------------------------
# solubilization_ratio
# ---------------------------------------------------------------------------


class TestSolubilizationRatio:
    def test_fed_greater_than_fasted_lipophilic(self):
        """For lipophilic drugs, fed > fasted → ratio > 1."""
        s_fasted = micellar_solubilization(
            drug_logP=4.0, drug_mw=350.0, bile_salt_mM=3.0, lecithin_mM=0.8, state="fasted"
        )
        s_fed = micellar_solubilization(
            drug_logP=4.0, drug_mw=350.0, bile_salt_mM=15.0, lecithin_mM=3.75, state="fed"
        )
        ratio = solubilization_ratio(s_fasted, s_fed)
        assert ratio > 1.0

    def test_ratio_equal_when_same(self):
        ratio = solubilization_ratio(0.5, 0.5)
        assert abs(ratio - 1.0) < 1e-9

    def test_ratio_less_than_one_when_fed_lower(self):
        ratio = solubilization_ratio(fasted_solubility=2.0, fed_solubility=1.0)
        assert ratio < 1.0

    def test_zero_fasted_raises(self):
        with pytest.raises(ValueError, match="fasted_solubility must be positive"):
            solubilization_ratio(0.0, 1.0)

    def test_negative_fed_raises(self):
        with pytest.raises(ValueError):
            solubilization_ratio(1.0, -0.5)


# ---------------------------------------------------------------------------
# dose_number
# ---------------------------------------------------------------------------


class TestDoseNumber:
    def test_expected_value(self):
        """dose=100mg, S=0.4mg/mL, V=250mL → Dn=1.0."""
        dn = dose_number(dose_mg=100.0, solubility_mg_mL=0.4, volume_mL=250.0)
        assert abs(dn - 1.0) < 1e-9

    def test_high_solubility_low_dn(self):
        dn = dose_number(dose_mg=100.0, solubility_mg_mL=10.0, volume_mL=250.0)
        assert dn < 1.0

    def test_low_solubility_high_dn(self):
        dn = dose_number(dose_mg=100.0, solubility_mg_mL=0.05, volume_mL=250.0)
        assert dn > 1.0

    def test_default_volume_250(self):
        dn = dose_number(100.0, 0.4)
        assert abs(dn - 1.0) < 1e-9

    def test_invalid_dose_raises(self):
        with pytest.raises(ValueError, match="dose_mg must be positive"):
            dose_number(0.0, 1.0)

    def test_invalid_solubility_raises(self):
        with pytest.raises(ValueError, match="solubility_mg_mL must be positive"):
            dose_number(100.0, 0.0)

    def test_invalid_volume_raises(self):
        with pytest.raises(ValueError, match="volume_mL must be positive"):
            dose_number(100.0, 1.0, volume_mL=0.0)


# ---------------------------------------------------------------------------
# absorption_limited_by
# ---------------------------------------------------------------------------


class TestAbsorptionLimitedBy:
    def test_both_high_dn_low_papp(self):
        result = absorption_limited_by(dose_number_val=5.0, papp_cm_s=1e-8)
        assert result == "both"

    def test_solubility_only(self):
        result = absorption_limited_by(dose_number_val=5.0, papp_cm_s=1e-5)
        assert result == "solubility"

    def test_permeability_only(self):
        result = absorption_limited_by(dose_number_val=0.1, papp_cm_s=1e-8)
        assert result == "permeability"

    def test_neither(self):
        result = absorption_limited_by(dose_number_val=0.1, papp_cm_s=1e-5)
        assert result == "neither"

    def test_boundary_dn_exactly_one_not_sol_limited(self):
        # Dn == 1.0 → not strictly > 1, so not solubility limited
        result = absorption_limited_by(dose_number_val=1.0, papp_cm_s=1e-5)
        assert result == "neither"

    def test_negative_dn_raises(self):
        with pytest.raises(ValueError, match="dose_number_val must be non-negative"):
            absorption_limited_by(-1.0, 1e-6)

    def test_negative_papp_raises(self):
        with pytest.raises(ValueError, match="papp_cm_s must be non-negative"):
            absorption_limited_by(1.0, -1e-6)


# ---------------------------------------------------------------------------
# predict_gi_solubility
# ---------------------------------------------------------------------------


class TestPredictGISolubility:
    def test_returns_gi_solubility_result(self):
        result = predict_gi_solubility(drug_logP=2.0, drug_mw=300.0)
        assert isinstance(result, GISolubilityResult)

    def test_default_segment_is_jejunum(self):
        result = predict_gi_solubility(drug_logP=2.0, drug_mw=300.0)
        assert result.gi_segment == "jejunum"
        assert abs(result.ph - 6.5) < 1e-9

    def test_stomach_ph_1_5(self):
        result = predict_gi_solubility(drug_logP=2.0, drug_mw=300.0, gi_segment="stomach")
        assert abs(result.ph - 1.5) < 1e-9

    def test_duodenum_ph_6(self):
        result = predict_gi_solubility(drug_logP=2.0, drug_mw=300.0, gi_segment="duodenum")
        assert abs(result.ph - 6.0) < 1e-9

    def test_ileum_ph_7_2(self):
        result = predict_gi_solubility(drug_logP=2.0, drug_mw=300.0, gi_segment="ileum")
        assert abs(result.ph - 7.2) < 1e-9

    def test_colon_ph_7_4(self):
        result = predict_gi_solubility(drug_logP=2.0, drug_mw=300.0, gi_segment="colon")
        assert abs(result.ph - 7.4) < 1e-9

    def test_different_segments_different_ph(self):
        r_jej = predict_gi_solubility(drug_logP=2.0, drug_mw=300.0, gi_segment="jejunum")
        r_col = predict_gi_solubility(drug_logP=2.0, drug_mw=300.0, gi_segment="colon")
        assert r_jej.ph != r_col.ph

    def test_high_dose_solubility_limited(self):
        # Very lipophilic drug, high dose → Dn > 1
        result = predict_gi_solubility(
            drug_logP=5.0, drug_mw=500.0, drug_dose_mg=500.0, gi_segment="jejunum"
        )
        assert result.dose_number > 0

    def test_dose_number_field_consistent(self):
        result = predict_gi_solubility(drug_logP=2.0, drug_mw=300.0, drug_dose_mg=100.0)
        expected_dn = dose_number(100.0, result.total_solubility_mg_mL, 250.0)
        assert abs(result.dose_number - expected_dn) < 1e-6

    def test_solubility_limited_flag(self):
        result = predict_gi_solubility(drug_logP=2.0, drug_mw=300.0, drug_dose_mg=100.0)
        assert result.solubility_limited == (result.dose_number > 1.0)

    def test_pka_applied_acidic_drug(self):
        # Acidic drug at high pH → ionized → higher solubility
        r_no_pka = predict_gi_solubility(drug_logP=3.0, drug_mw=300.0, gi_segment="colon")
        r_pka = predict_gi_solubility(
            drug_logP=3.0, drug_mw=300.0, drug_pka=4.5, gi_segment="colon"
        )
        assert r_pka.total_solubility_mg_mL >= r_no_pka.total_solubility_mg_mL

    def test_notes_contains_segment(self):
        result = predict_gi_solubility(drug_logP=2.0, drug_mw=300.0, gi_segment="duodenum")
        assert "duodenum" in result.notes

    def test_invalid_segment_raises(self):
        with pytest.raises(ValueError, match="gi_segment must be one of"):
            predict_gi_solubility(drug_logP=2.0, drug_mw=300.0, gi_segment="appendix")

    def test_invalid_mw_raises(self):
        with pytest.raises(ValueError):
            predict_gi_solubility(drug_logP=2.0, drug_mw=0.0)

    def test_invalid_dose_raises(self):
        with pytest.raises(ValueError):
            predict_gi_solubility(drug_logP=2.0, drug_mw=300.0, drug_dose_mg=0.0)

    def test_total_solubility_positive(self):
        result = predict_gi_solubility(drug_logP=3.0, drug_mw=350.0)
        assert result.total_solubility_mg_mL > 0
