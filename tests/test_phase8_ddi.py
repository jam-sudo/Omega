"""Phase 8 — Advanced DDI Modeling tests.

Tests cover:
- Static R1 helper and per-enzyme values
- Gut R1gut (intestinal CYP3A4)
- MBI R2 helper and multi-enzyme values
- CYP induction (Emax model)
- Transporter DDI (P-gp, OATP1B1/1B3, OCT2)
- AUCR prediction
- assess_ddi_risk integration
- format_report output
- KDEG constants
- Legacy backward-compatibility
"""

import math
import pytest

from omega_pbpk.clinical.ddi_report import (
    KDEG,
    DDIInhibitor,
    DDIRiskReport,
    _aucr,
    _induction_ratio,
    _r1,
    _r2_mbi,
    assess_ddi_risk,
    format_report,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def weak_inhibitor():
    """Cmax 0.1 µM — all Ki values far above threshold → no risk flags expected.

    Ki values ≥ 10 000 µM and small dose ensure R1 ≪ 2, R1gut ≪ 11.
    """
    return DDIInhibitor(
        name="Weak",
        cmax_uM=0.1,
        ki_3a4_uM=10_000.0,
        ki_2d6_uM=10_000.0,
        ki_2c9_uM=10_000.0,
        ki_1a2_uM=10_000.0,
        ki_2c19_uM=10_000.0,
        dose_mg=10.0,   # small dose → R1gut ≪ 11
    )


@pytest.fixture
def strong_cyp3a4_inhibitor():
    """Potent CYP3A4 inhibitor (ketoconazole-like): low Ki, high Cmax."""
    return DDIInhibitor(
        name="StrongCYP3A4",
        cmax_uM=5.0,
        ki_3a4_uM=0.1,        # R1 = 1 + 5/0.1 = 51 → well above 2
        mw=531.0,
        dose_mg=200.0,
        fa=0.9,
        ka_per_h=1.0,
        fup=0.01,
    )


@pytest.fixture
def mbi_inhibitor():
    """Strong MBI perpetrator (clarithromycin-like)."""
    return DDIInhibitor(
        name="MBIStrong",
        cmax_uM=3.0,
        ki_3a4_uM=1000.0,         # no static inhibition
        kinact_3a4_per_h=4.2,     # high kinact
        ki_mbi_3a4_uM=0.5,
        mw=748.0,
        dose_mg=500.0,
    )


@pytest.fixture
def inducer():
    """Potent CYP3A4 inducer (rifampin-like)."""
    return DDIInhibitor(
        name="Inducer",
        cmax_uM=10.0,
        ind_max_3a4=8.0,    # IndMax = 8
        ind_c50_3a4_uM=2.0, # EC50 = 2 µM
        mw=823.0,
        dose_mg=600.0,
    )


@pytest.fixture
def transporter_inhibitor():
    """Inhibitor hitting P-gp and OATP1B1."""
    return DDIInhibitor(
        name="TransporterHit",
        cmax_uM=2.0,
        ic50_pgp_uM=1.0,      # R_pgp will be large due to gut lumen concentration
        ic50_oatp1b1_uM=0.5,
        ic50_oatp1b3_uM=0.5,
        ic50_oct2_uM=1.0,
        mw=400.0,
        dose_mg=200.0,
        fa=0.9,
        ka_per_h=1.5,
    )


@pytest.fixture
def multi_mechanism_inhibitor():
    """Perpetrator with static + MBI + induction (to test AUCR interaction)."""
    return DDIInhibitor(
        name="MultiMech",
        cmax_uM=4.0,
        ki_3a4_uM=2.0,             # R1 = 3.0
        kinact_3a4_per_h=2.0,
        ki_mbi_3a4_uM=1.0,
        ind_max_3a4=3.0,
        ind_c50_3a4_uM=1.0,        # induction ratio at Cmax=4 → 1 + 3*4/(1+4) = 3.4
        mw=400.0,
        dose_mg=100.0,
    )


# ---------------------------------------------------------------------------
# KDEG constants
# ---------------------------------------------------------------------------

class TestKdegConstants:
    def test_all_cyps_present(self):
        for cyp in ("CYP3A4", "CYP2D6", "CYP2C9", "CYP1A2", "CYP2C19"):
            assert cyp in KDEG

    def test_values_positive(self):
        for val in KDEG.values():
            assert val > 0

    def test_cyp3a4_value(self):
        # FDA 2020: 0.0005 /min → 0.03 /h
        assert abs(KDEG["CYP3A4"] - 0.03) < 1e-9

    def test_cyp2d6_value(self):
        assert abs(KDEG["CYP2D6"] - 0.018) < 1e-9


# ---------------------------------------------------------------------------
# _r1 helper
# ---------------------------------------------------------------------------

class TestR1Helper:
    def test_no_inhibition_at_zero_cmax(self):
        assert _r1(0.0, 1.0) == 1.0

    def test_ki_inf_gives_r1_one(self):
        assert _r1(10.0, float("inf")) == pytest.approx(1.0, abs=1e-9)

    def test_known_value(self):
        # cmax=1 µM, Ki=1 µM → R1 = 2.0
        assert _r1(1.0, 1.0) == pytest.approx(2.0, rel=1e-6)

    def test_high_cmax_scales_linearly(self):
        r1a = _r1(2.0, 1.0)
        r1b = _r1(4.0, 1.0)
        # R1 = 1+[I]/Ki → (r1b-1) / (r1a-1) == 2
        assert (r1b - 1.0) / (r1a - 1.0) == pytest.approx(2.0, rel=1e-6)

    def test_ki_very_small_does_not_raise(self):
        r = _r1(1.0, 1e-15)
        assert math.isfinite(r)
        assert r > 1.0


# ---------------------------------------------------------------------------
# _r2_mbi helper
# ---------------------------------------------------------------------------

class TestR2MbiHelper:
    def test_no_mbi_returns_one(self):
        assert _r2_mbi(1.0, 0.0, 1.0, KDEG["CYP3A4"]) == 1.0

    def test_inf_ki_mbi_returns_one(self):
        assert _r2_mbi(1.0, 1.0, float("inf"), KDEG["CYP3A4"]) == 1.0

    def test_zero_cmax_returns_one(self):
        assert _r2_mbi(0.0, 2.0, 0.5, KDEG["CYP3A4"]) == 1.0

    def test_r2_greater_than_one_with_mbi(self):
        r2 = _r2_mbi(cmax=3.0, kinact=4.2, ki_mbi=0.5, kdeg=KDEG["CYP3A4"])
        assert r2 > 1.25

    def test_r2_increases_with_kinact(self):
        r2_low = _r2_mbi(cmax=1.0, kinact=0.5, ki_mbi=1.0, kdeg=0.03)
        r2_high = _r2_mbi(cmax=1.0, kinact=5.0, ki_mbi=1.0, kdeg=0.03)
        assert r2_high > r2_low

    def test_r2_decreases_with_kdeg(self):
        r2_low_kdeg = _r2_mbi(cmax=1.0, kinact=2.0, ki_mbi=1.0, kdeg=0.01)
        r2_high_kdeg = _r2_mbi(cmax=1.0, kinact=2.0, ki_mbi=1.0, kdeg=0.5)
        assert r2_low_kdeg > r2_high_kdeg

    def test_numerical_stability_extreme_kinact(self):
        r2 = _r2_mbi(cmax=100.0, kinact=1000.0, ki_mbi=0.001, kdeg=0.03)
        assert math.isfinite(r2)
        assert r2 > 1.0


# ---------------------------------------------------------------------------
# _induction_ratio helper
# ---------------------------------------------------------------------------

class TestInductionRatioHelper:
    def test_no_induction_when_ind_max_zero(self):
        assert _induction_ratio(10.0, 0.0, 1.0) == 1.0

    def test_no_induction_at_zero_cmax(self):
        assert _induction_ratio(0.0, 5.0, 1.0) == 1.0

    def test_emax_model_at_ec50(self):
        # At Cmax = EC50: ratio = 1 + IndMax * 0.5
        ratio = _induction_ratio(1.0, 8.0, 1.0)
        assert ratio == pytest.approx(1.0 + 8.0 * 0.5, rel=1e-6)

    def test_emax_model_saturation(self):
        # At very high Cmax (>> EC50): ratio ≈ 1 + IndMax
        ratio = _induction_ratio(10000.0, 8.0, 1.0)
        assert ratio == pytest.approx(9.0, rel=0.001)

    def test_ratio_always_ge_one(self):
        ratio = _induction_ratio(5.0, 2.0, 10.0)
        assert ratio >= 1.0


# ---------------------------------------------------------------------------
# _aucr helper
# ---------------------------------------------------------------------------

class TestAucrHelper:
    def test_no_ddi_aucr_is_one(self):
        aucr = _aucr(fm=0.5, R1=1.0, R2=1.0, induction_ratio=1.0)
        assert aucr == pytest.approx(1.0, rel=1e-6)

    def test_full_inhibition_fm1_pure_inhibitor(self):
        # fm=1, R1 large, R2=1, no induction → AUCR ≈ R1
        aucr = _aucr(fm=1.0, R1=10.0, R2=1.0, induction_ratio=1.0)
        assert aucr == pytest.approx(10.0, rel=1e-4)

    def test_full_induction_reduces_aucr(self):
        # fm=1, no inhibition, 10× induction → AUCR = 1/10
        aucr = _aucr(fm=1.0, R1=1.0, R2=1.0, induction_ratio=10.0)
        assert aucr == pytest.approx(0.1, rel=1e-4)

    def test_combined_inhibition_and_induction(self):
        # fm=1, R1=2, R2=1, induction_ratio=2 → clint_scalar=2/2=1 → AUCR=1
        aucr = _aucr(fm=1.0, R1=2.0, R2=1.0, induction_ratio=2.0)
        assert aucr == pytest.approx(1.0, rel=1e-6)

    def test_fm_zero_aucr_is_one(self):
        # fm=0 → drug not metabolized by CYP → AUCR=1
        aucr = _aucr(fm=0.0, R1=50.0, R2=5.0, induction_ratio=1.0)
        assert aucr == pytest.approx(1.0, rel=1e-6)

    def test_fm_clipped_to_range(self):
        # fm > 1 should be clipped to 1
        aucr_clamped = _aucr(fm=2.0, R1=10.0, R2=1.0, induction_ratio=1.0)
        aucr_one = _aucr(fm=1.0, R1=10.0, R2=1.0, induction_ratio=1.0)
        assert aucr_clamped == pytest.approx(aucr_one, rel=1e-6)


# ---------------------------------------------------------------------------
# assess_ddi_risk — individual sections
# ---------------------------------------------------------------------------

class TestStaticR1:
    def test_weak_inhibitor_no_flags(self, weak_inhibitor):
        report = assess_ddi_risk(weak_inhibitor)
        r1_flags = [f for f in report.flags if "static inhibition" in f]
        assert len(r1_flags) == 0

    def test_strong_cyp3a4_r1_above_threshold(self, strong_cyp3a4_inhibitor):
        report = assess_ddi_risk(strong_cyp3a4_inhibitor)
        assert report.R1_3a4 >= 2.0

    def test_strong_cyp3a4_flag_raised(self, strong_cyp3a4_inhibitor):
        report = assess_ddi_risk(strong_cyp3a4_inhibitor)
        assert any("CYP3A4" in f and "static inhibition" in f for f in report.flags)

    def test_all_five_cyp_r1_computed(self, weak_inhibitor):
        report = assess_ddi_risk(weak_inhibitor)
        for attr in ("R1_3a4", "R1_2d6", "R1_2c9", "R1_1a2", "R1_2c19"):
            val = getattr(report, attr)
            assert val >= 1.0
            assert math.isfinite(val)

    def test_r1_values_match_helper(self):
        inh = DDIInhibitor(name="Test", cmax_uM=3.0, ki_3a4_uM=1.5)
        report = assess_ddi_risk(inh)
        expected = _r1(3.0, 1.5)
        assert report.R1_3a4 == pytest.approx(expected, rel=1e-9)


class TestR1Gut:
    def test_r1gut_ge_one(self, strong_cyp3a4_inhibitor):
        report = assess_ddi_risk(strong_cyp3a4_inhibitor)
        assert report.R1gut_3a4 >= 1.0

    def test_r1gut_triggered_by_high_dose(self):
        # Very low Ki (0.01 µM) + 500 mg dose → R1gut should be huge
        inh = DDIInhibitor(
            name="GutHit",
            cmax_uM=0.1,
            ki_3a4_uM=0.01,
            dose_mg=500.0,
            mw=300.0,
            fa=0.9,
            ka_per_h=1.0,
        )
        report = assess_ddi_risk(inh)
        assert report.R1gut_3a4 >= 11.0

    def test_r1gut_flag_present_when_high(self):
        inh = DDIInhibitor(
            name="GutHit",
            cmax_uM=0.1,
            ki_3a4_uM=0.01,
            dose_mg=500.0,
            mw=300.0,
            fa=0.9,
            ka_per_h=1.0,
        )
        report = assess_ddi_risk(inh)
        assert any("R1gut" in f for f in report.flags)

    def test_r1gut_is_one_with_no_ki(self):
        inh = DDIInhibitor(name="NoGut", cmax_uM=5.0)  # ki defaults to inf
        report = assess_ddi_risk(inh)
        # With Ki=inf, R1gut should equal 1.0
        assert report.R1gut_3a4 == pytest.approx(1.0, abs=1e-6)


class TestMbiR2:
    def test_no_mbi_r2_is_one(self, weak_inhibitor):
        report = assess_ddi_risk(weak_inhibitor)
        assert report.R2_3a4 == pytest.approx(1.0, abs=1e-9)
        assert report.R2_2d6 == pytest.approx(1.0, abs=1e-9)
        assert report.R2_2c9 == pytest.approx(1.0, abs=1e-9)
        assert report.R2_1a2 == pytest.approx(1.0, abs=1e-9)

    def test_strong_mbi_r2_above_threshold(self, mbi_inhibitor):
        report = assess_ddi_risk(mbi_inhibitor)
        assert report.R2_3a4 >= 1.25

    def test_mbi_flag_raised(self, mbi_inhibitor):
        report = assess_ddi_risk(mbi_inhibitor)
        assert any("MBI" in f and "CYP3A4" in f for f in report.flags)

    def test_multi_enzyme_mbi(self):
        inh = DDIInhibitor(
            name="MultiMBI",
            cmax_uM=5.0,
            kinact_2d6_per_h=3.0,
            ki_mbi_2d6_uM=0.5,
            kinact_2c9_per_h=2.0,
            ki_mbi_2c9_uM=0.8,
        )
        report = assess_ddi_risk(inh)
        assert report.R2_2d6 >= 1.25
        assert report.R2_2c9 >= 1.25

    def test_r2_values_all_ge_one(self, mbi_inhibitor):
        report = assess_ddi_risk(mbi_inhibitor)
        for attr in ("R2_3a4", "R2_2d6", "R2_2c9", "R2_1a2"):
            assert getattr(report, attr) >= 1.0


class TestCypInduction:
    def test_no_induction_by_default(self, weak_inhibitor):
        report = assess_ddi_risk(weak_inhibitor)
        assert report.induction_ratio_3a4 == pytest.approx(1.0, abs=1e-9)
        assert report.induction_ratio_1a2 == pytest.approx(1.0, abs=1e-9)

    def test_rifampin_like_induction_flag(self, inducer):
        report = assess_ddi_risk(inducer)
        assert report.induction_ratio_3a4 >= 2.0
        assert any("CYP3A4" in f and "induction" in f for f in report.flags)

    def test_induction_ratio_emax_formula(self, inducer):
        report = assess_ddi_risk(inducer)
        # cmax=10, IndMax=8, EC50=2 → ratio = 1 + 8*10/(2+10) = 1 + 6.67 = 7.67
        expected = 1.0 + 8.0 * 10.0 / (2.0 + 10.0)
        assert report.induction_ratio_3a4 == pytest.approx(expected, rel=1e-6)

    def test_legacy_induction_fold_property(self):
        inh = DDIInhibitor(
            name="Inducer",
            cmax_uM=5.0,
            ind_max_3a4=4.0,
            ind_c50_3a4_uM=2.0,
        )
        report = assess_ddi_risk(inh)
        # Legacy property should mirror induction_ratio_3a4
        assert report.induction_fold_3a4 == report.induction_ratio_3a4

    def test_1a2_induction_separate(self):
        inh = DDIInhibitor(
            name="1A2Inducer",
            cmax_uM=10.0,
            ind_max_1a2=5.0,
            ind_c50_1a2_uM=2.0,
        )
        report = assess_ddi_risk(inh)
        assert report.induction_ratio_1a2 >= 2.0
        assert any("CYP1A2" in f and "induction" in f for f in report.flags)


class TestTransporterDDI:
    def test_no_transporter_hit_with_inf_ic50(self, weak_inhibitor):
        # weak_inhibitor has no IC50 values set (default to inf)
        report = assess_ddi_risk(weak_inhibitor)
        assert report.R_pgp == pytest.approx(1.0, abs=1e-6)
        assert report.R_oatp1b1 == pytest.approx(1.0, abs=1e-6)
        assert report.R_oatp1b3 == pytest.approx(1.0, abs=1e-6)
        assert report.R_oct2 == pytest.approx(1.0, abs=1e-6)

    def test_pgp_r_above_threshold(self, transporter_inhibitor):
        report = assess_ddi_risk(transporter_inhibitor)
        # Gut concentration >> systemic Cmax → large R_pgp
        assert report.R_pgp >= 1.25

    def test_oatp_r_above_threshold(self, transporter_inhibitor):
        report = assess_ddi_risk(transporter_inhibitor)
        assert report.R_oatp1b1 >= 1.25

    def test_oct2_r_value(self):
        inh = DDIInhibitor(
            name="OCT2Hit",
            cmax_uM=3.0,
            ic50_oct2_uM=2.0,   # R_oct2 = 1 + 3/2 = 2.5
        )
        report = assess_ddi_risk(inh)
        assert report.R_oct2 == pytest.approx(2.5, rel=1e-4)
        assert any("OCT2" in f for f in report.flags)

    def test_transporter_flags_raised(self, transporter_inhibitor):
        report = assess_ddi_risk(transporter_inhibitor)
        transporter_flags = [f for f in report.flags if any(
            t in f for t in ("P-gp", "OATP1B1", "OATP1B3", "OCT2")
        )]
        assert len(transporter_flags) >= 1

    def test_transporter_r_values_finite(self, transporter_inhibitor):
        report = assess_ddi_risk(transporter_inhibitor)
        for attr in ("R_pgp", "R_oatp1b1", "R_oatp1b3", "R_oct2"):
            assert math.isfinite(getattr(report, attr))


class TestAucrPrediction:
    def test_no_ddi_aucr_one(self, weak_inhibitor):
        report = assess_ddi_risk(weak_inhibitor)
        assert report.aucr_3a4 == pytest.approx(1.0, abs=0.01)

    def test_strong_inhibitor_aucr_above_one(self, strong_cyp3a4_inhibitor):
        report = assess_ddi_risk(strong_cyp3a4_inhibitor, victim_fm_3a4=0.9)
        assert report.aucr_3a4 > 2.0

    def test_inducer_aucr_below_one(self, inducer):
        report = assess_ddi_risk(inducer, victim_fm_3a4=1.0)
        assert report.aucr_3a4 < 1.0

    def test_fm_zero_aucr_one_regardless_of_inhibitor(self, strong_cyp3a4_inhibitor):
        report = assess_ddi_risk(strong_cyp3a4_inhibitor, victim_fm_3a4=0.0)
        assert report.aucr_3a4 == pytest.approx(1.0, abs=1e-4)

    def test_multi_mechanism_aucr_intermediate(self, multi_mechanism_inhibitor):
        report = assess_ddi_risk(multi_mechanism_inhibitor, victim_fm_3a4=0.8)
        # Inhibition and induction partially cancel; should be different from 1
        assert math.isfinite(report.aucr_3a4)
        assert report.aucr_3a4 > 0


# ---------------------------------------------------------------------------
# Full assess_ddi_risk integration
# ---------------------------------------------------------------------------

class TestAssessDdiRiskIntegration:
    def test_returns_ddi_risk_report(self, weak_inhibitor):
        report = assess_ddi_risk(weak_inhibitor)
        assert isinstance(report, DDIRiskReport)

    def test_clean_compound_no_flags(self, weak_inhibitor):
        report = assess_ddi_risk(weak_inhibitor)
        assert len(report.flags) == 0
        assert report.recommendation == "No clinical DDI study warranted"

    def test_problematic_compound_has_recommendation(self, strong_cyp3a4_inhibitor):
        report = assess_ddi_risk(strong_cyp3a4_inhibitor)
        assert "recommended" in report.recommendation.lower()

    def test_inhibitor_name_preserved(self, mbi_inhibitor):
        report = assess_ddi_risk(mbi_inhibitor)
        assert report.inhibitor_name == "MBIStrong"

    def test_all_r_values_ge_one(self, strong_cyp3a4_inhibitor):
        report = assess_ddi_risk(strong_cyp3a4_inhibitor)
        for attr in ("R1_3a4", "R1_2d6", "R1_2c9", "R1_1a2", "R1_2c19",
                     "R2_3a4", "R2_2d6", "R2_2c9", "R2_1a2",
                     "R_pgp", "R_oatp1b1", "R_oatp1b3", "R_oct2"):
            val = getattr(report, attr)
            assert val >= 1.0, f"{attr} = {val} < 1.0"

    def test_induction_ratios_ge_one(self, inducer):
        report = assess_ddi_risk(inducer)
        assert report.induction_ratio_3a4 >= 1.0
        assert report.induction_ratio_1a2 >= 1.0

    def test_multi_mechanism_multiple_flags(self, multi_mechanism_inhibitor):
        report = assess_ddi_risk(multi_mechanism_inhibitor, victim_fm_3a4=0.9)
        assert len(report.flags) >= 2


# ---------------------------------------------------------------------------
# format_report
# ---------------------------------------------------------------------------

class TestFormatReport:
    def test_returns_string(self, weak_inhibitor):
        report = assess_ddi_risk(weak_inhibitor)
        text = format_report(report)
        assert isinstance(text, str)
        assert len(text) > 100

    def test_contains_inhibitor_name(self, strong_cyp3a4_inhibitor):
        report = assess_ddi_risk(strong_cyp3a4_inhibitor)
        text = format_report(report)
        assert "StrongCYP3A4" in text

    def test_contains_all_cyp_sections(self, strong_cyp3a4_inhibitor):
        report = assess_ddi_risk(strong_cyp3a4_inhibitor)
        text = format_report(report)
        for cyp in ("CYP3A4", "CYP2D6", "CYP2C9", "CYP1A2", "CYP2C19"):
            assert cyp in text

    def test_warning_marker_present_for_flagged(self, strong_cyp3a4_inhibitor):
        report = assess_ddi_risk(strong_cyp3a4_inhibitor)
        text = format_report(report)
        assert "⚠" in text

    def test_no_warning_for_clean(self, weak_inhibitor):
        report = assess_ddi_risk(weak_inhibitor)
        text = format_report(report)
        assert "No risk flags raised" in text

    def test_contains_transporter_section(self, transporter_inhibitor):
        report = assess_ddi_risk(transporter_inhibitor)
        text = format_report(report)
        assert "Transporter" in text
        assert "OATP1B1" in text

    def test_contains_aucr(self, strong_cyp3a4_inhibitor):
        report = assess_ddi_risk(strong_cyp3a4_inhibitor)
        text = format_report(report)
        assert "AUCR" in text

    def test_recommendation_present(self, weak_inhibitor):
        report = assess_ddi_risk(weak_inhibitor)
        text = format_report(report)
        assert "Recommendation:" in text


# ---------------------------------------------------------------------------
# Backward compatibility
# ---------------------------------------------------------------------------

class TestBackwardCompatibility:
    def test_legacy_induction_fold_3a4_attribute(self):
        """DDIInhibitor.induction_fold_3a4 still accepted."""
        inh = DDIInhibitor(name="Legacy", cmax_uM=1.0, induction_fold_3a4=3.0)
        assert inh.induction_fold_3a4 == 3.0

    def test_legacy_report_property(self):
        """DDIRiskReport.induction_fold_3a4 mirrors induction_ratio_3a4."""
        inh = DDIInhibitor(
            name="LegacyInd",
            cmax_uM=5.0,
            ind_max_3a4=5.0,
            ind_c50_3a4_uM=2.5,
        )
        report = assess_ddi_risk(inh)
        assert report.induction_fold_3a4 == report.induction_ratio_3a4

    def test_r2_3a4_value_property(self):
        inh = DDIInhibitor(
            name="MBILegacy",
            cmax_uM=3.0,
            kinact_3a4_per_h=3.0,
            ki_mbi_3a4_uM=0.5,
        )
        report = assess_ddi_risk(inh)
        assert report.R2_3a4_value == report.R2_3a4
