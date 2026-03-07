"""Tests for drug-drug interaction severity classification (Phase 564)."""

from __future__ import annotations

import pytest

from omega_pbpk.risk.drug_interaction_severity import (
    classify_ddi_severity,
    induction_fold_change,
    mbi_r2_value,
    net_ddi_aucr,
    static_r1_gut,
    static_r1_value,
)

# ---------------------------------------------------------------------------
# classify_ddi_severity
# ---------------------------------------------------------------------------


class TestClassifyDdiSeverity:
    """Tests for FDA 2020 DDI severity classification."""

    def test_aucr_5_strong(self):
        r = classify_ddi_severity(aucr=5.0)
        assert r["aucr_class"] == "strong"

    def test_aucr_above_5_strong(self):
        r = classify_ddi_severity(aucr=10.0)
        assert r["aucr_class"] == "strong"

    def test_aucr_2_5_moderate(self):
        r = classify_ddi_severity(aucr=2.5)
        assert r["aucr_class"] == "moderate"

    def test_aucr_2_moderate(self):
        r = classify_ddi_severity(aucr=2.0)
        assert r["aucr_class"] == "moderate"

    def test_aucr_1_5_weak(self):
        r = classify_ddi_severity(aucr=1.5)
        assert r["aucr_class"] == "weak"

    def test_aucr_1_25_weak(self):
        r = classify_ddi_severity(aucr=1.25)
        assert r["aucr_class"] == "weak"

    def test_aucr_1_1_negligible(self):
        r = classify_ddi_severity(aucr=1.1)
        assert r["aucr_class"] == "negligible"

    def test_cmax_ratio_classified_when_provided(self):
        r = classify_ddi_severity(aucr=2.0, cmax_ratio=1.5)
        assert r["cmax_class"] == "weak"
        assert r["cmax_ratio"] == 1.5

    def test_cmax_class_none_when_not_provided(self):
        r = classify_ddi_severity(aucr=2.0)
        assert r["cmax_class"] is None
        assert r["cmax_ratio"] is None

    def test_clinical_action_populated(self):
        r = classify_ddi_severity(aucr=3.0)
        assert isinstance(r["clinical_action"], str)
        assert len(r["clinical_action"]) > 0

    def test_notes_non_empty(self):
        r = classify_ddi_severity(aucr=2.0)
        assert len(r["notes"]) > 0

    def test_returns_dict_with_required_keys(self):
        r = classify_ddi_severity(aucr=2.0)
        for key in ("aucr", "aucr_class", "cmax_ratio", "cmax_class", "clinical_action", "notes"):
            assert key in r


# ---------------------------------------------------------------------------
# static_r1_value
# ---------------------------------------------------------------------------


class TestStaticR1Value:
    """Tests for basic static R1 hepatic inhibition model."""

    def test_r1_at_i_equals_ki_is_2(self):
        """R1 = 1 + I/Ki; at I=Ki, R1=2."""
        r1 = static_r1_value(inhibitor_conc_uM=10.0, ki_uM=10.0)
        assert abs(r1 - 2.0) < 1e-9

    def test_r1_at_zero_inhibitor_is_1(self):
        r1 = static_r1_value(inhibitor_conc_uM=0.0, ki_uM=5.0)
        assert abs(r1 - 1.0) < 1e-9

    def test_r1_increases_with_inhibitor(self):
        r1_high = static_r1_value(inhibitor_conc_uM=20.0, ki_uM=5.0)
        r1_low = static_r1_value(inhibitor_conc_uM=1.0, ki_uM=5.0)
        assert r1_high > r1_low

    def test_zero_ki_raises(self):
        with pytest.raises(ValueError):
            static_r1_value(inhibitor_conc_uM=5.0, ki_uM=0.0)

    def test_negative_ki_raises(self):
        with pytest.raises(ValueError):
            static_r1_value(inhibitor_conc_uM=5.0, ki_uM=-1.0)


# ---------------------------------------------------------------------------
# static_r1_gut
# ---------------------------------------------------------------------------


class TestStaticR1Gut:
    """Tests for gut R1 static model."""

    def test_returns_float_geq_1(self):
        r1 = static_r1_gut(dose_mg=100.0, mw=300.0, ki_uM=10.0)
        assert r1 >= 1.0

    def test_zero_dose_gives_1(self):
        r1 = static_r1_gut(dose_mg=0.0, mw=300.0, ki_uM=10.0)
        assert abs(r1 - 1.0) < 1e-9

    def test_fa_0_gives_1(self):
        r1 = static_r1_gut(dose_mg=100.0, mw=300.0, ki_uM=10.0, fa=0.0)
        assert abs(r1 - 1.0) < 1e-9

    def test_higher_dose_higher_r1(self):
        r1_high = static_r1_gut(dose_mg=500.0, mw=300.0, ki_uM=10.0)
        r1_low = static_r1_gut(dose_mg=50.0, mw=300.0, ki_uM=10.0)
        assert r1_high > r1_low

    def test_zero_ki_raises(self):
        with pytest.raises(ValueError):
            static_r1_gut(dose_mg=100.0, mw=300.0, ki_uM=0.0)


# ---------------------------------------------------------------------------
# mbi_r2_value
# ---------------------------------------------------------------------------


class TestMbiR2Value:
    """Tests for mechanism-based inhibitor R2."""

    def test_r2_less_than_1_when_kinact_positive(self):
        r2 = mbi_r2_value(kinact_per_h=0.5, ki_MBI_uM=1.0, inhibitor_conc_uM=5.0)
        assert r2 < 1.0

    def test_r2_equals_1_when_kinact_zero(self):
        r2 = mbi_r2_value(kinact_per_h=0.0, ki_MBI_uM=1.0, inhibitor_conc_uM=5.0)
        assert abs(r2 - 1.0) < 1e-9

    def test_r2_between_0_and_1(self):
        r2 = mbi_r2_value(kinact_per_h=1.0, ki_MBI_uM=1.0, inhibitor_conc_uM=10.0)
        assert 0.0 < r2 <= 1.0

    def test_r2_decreases_with_more_inhibitor(self):
        r2_high = mbi_r2_value(kinact_per_h=0.5, ki_MBI_uM=1.0, inhibitor_conc_uM=50.0)
        r2_low = mbi_r2_value(kinact_per_h=0.5, ki_MBI_uM=1.0, inhibitor_conc_uM=0.1)
        assert r2_high < r2_low

    def test_negative_ki_MBI_raises(self):
        with pytest.raises(ValueError):
            mbi_r2_value(kinact_per_h=0.5, ki_MBI_uM=-1.0, inhibitor_conc_uM=5.0)

    def test_zero_kdeg_raises(self):
        with pytest.raises(ValueError):
            mbi_r2_value(kinact_per_h=0.5, ki_MBI_uM=1.0, inhibitor_conc_uM=5.0, kdeg_per_h=0.0)


# ---------------------------------------------------------------------------
# induction_fold_change
# ---------------------------------------------------------------------------


class TestInductionFoldChange:
    """Tests for enzyme induction fold change."""

    def test_at_zero_concentration_returns_1(self):
        fold = induction_fold_change(emax_fold=5.0, ec50_uM=1.0, inducer_conc_uM=0.0)
        assert abs(fold - 1.0) < 1e-9

    def test_at_ec50_returns_1_plus_emax_over_2(self):
        """At C=EC50, fold = 1 + Emax/2."""
        fold = induction_fold_change(emax_fold=4.0, ec50_uM=2.0, inducer_conc_uM=2.0)
        assert abs(fold - (1.0 + 4.0 / 2.0)) < 1e-9

    def test_fold_increases_with_concentration(self):
        fold_high = induction_fold_change(emax_fold=5.0, ec50_uM=1.0, inducer_conc_uM=10.0)
        fold_low = induction_fold_change(emax_fold=5.0, ec50_uM=1.0, inducer_conc_uM=0.5)
        assert fold_high > fold_low

    def test_emax_0_always_1(self):
        fold = induction_fold_change(emax_fold=0.0, ec50_uM=1.0, inducer_conc_uM=100.0)
        assert abs(fold - 1.0) < 1e-9

    def test_negative_emax_raises(self):
        with pytest.raises(ValueError):
            induction_fold_change(emax_fold=-1.0, ec50_uM=1.0, inducer_conc_uM=5.0)

    def test_zero_ec50_raises(self):
        with pytest.raises(ValueError):
            induction_fold_change(emax_fold=5.0, ec50_uM=0.0, inducer_conc_uM=5.0)


# ---------------------------------------------------------------------------
# net_ddi_aucr
# ---------------------------------------------------------------------------


class TestNetDdiAucr:
    """Tests for net AUCR calculation."""

    def test_net_aucr_equals_r1_over_r2(self):
        r1 = 3.0
        r2 = 0.5
        assert abs(net_ddi_aucr(r1=r1, r2=r2) - r1 / r2) < 1e-9

    def test_r1_1_r2_1_gives_1(self):
        assert abs(net_ddi_aucr(r1=1.0, r2=1.0) - 1.0) < 1e-9

    def test_lower_r2_higher_aucr(self):
        aucr_high = net_ddi_aucr(r1=2.0, r2=0.3)
        aucr_low = net_ddi_aucr(r1=2.0, r2=0.8)
        assert aucr_high > aucr_low

    def test_zero_r2_raises(self):
        with pytest.raises(ValueError):
            net_ddi_aucr(r1=2.0, r2=0.0)

    def test_induction_fold_parameter_accepted(self):
        """induction_fold parameter is accepted without error."""
        result = net_ddi_aucr(r1=2.0, r2=0.5, induction_fold=2.0)
        assert result > 0.0
