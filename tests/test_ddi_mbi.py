"""Tests for FDA 2020 static MBI DDI model in ddi_report.py.

Covers:
  - Happy path: erythromycin-like MBI inhibitor (combined mechanism)
  - Backward compat: kinact=None → competitive mechanism, unchanged R values
  - Edge case: very high kinact → R2 approaches 0
"""

from __future__ import annotations

import pytest

from omega_pbpk.clinical.ddi_report import (
    DDI_MBI_LIMITATION,
    KDEG_CYP3A4_OBACH,
    DDIInhibitor,
    assess_ddi_risk,
)


class TestMBICombined:
    """Erythromycin-like inhibitor with both reversible and MBI parameters."""

    def test_erythromycin_like_combined_mechanism(self) -> None:
        """kinact=3.6 h⁻¹, KI=5.9 µM, cmax=3.6 µM → mechanism=='combined'."""
        inh = DDIInhibitor(
            name="ErythroLike",
            cmax_uM=3.6,
            ki_3a4_uM=10.0,  # reversible Ki present
            kinact=3.6,
            KI=5.9,
        )
        report = assess_ddi_risk(inh)
        assert report.mechanism == "combined"

    def test_erythromycin_like_r2_less_than_one(self) -> None:
        """R2 should be < 1.0 (MBI reduces enzyme activity)."""
        inh = DDIInhibitor(
            name="ErythroLike",
            cmax_uM=3.6,
            ki_3a4_uM=10.0,
            kinact=3.6,
            KI=5.9,
        )
        report = assess_ddi_risk(inh)
        assert report.R2_3a4 < 1.0

    def test_erythromycin_like_r2_formula(self) -> None:
        """Verify R2 matches FDA 2020 static equation exactly."""
        kinact = 3.6
        ki = 5.9
        cmax = 3.6
        expected_r2 = 1.0 / (1.0 + (kinact / KDEG_CYP3A4_OBACH) * cmax / (ki + cmax))

        inh = DDIInhibitor(
            name="ErythroLike",
            cmax_uM=cmax,
            ki_3a4_uM=10.0,
            kinact=kinact,
            KI=ki,
        )
        report = assess_ddi_risk(inh)
        assert report.R2_3a4 == pytest.approx(expected_r2, rel=1e-9)

    def test_mbi_limitation_warning_present(self) -> None:
        """MBI parameters should trigger the DDI_MBI_LIMITATION warning."""
        inh = DDIInhibitor(
            name="ErythroLike",
            cmax_uM=3.6,
            ki_3a4_uM=10.0,
            kinact=3.6,
            KI=5.9,
        )
        report = assess_ddi_risk(inh)
        limitation_flags = [f for f in report.flags if DDI_MBI_LIMITATION in f]
        assert len(limitation_flags) == 1


class TestMBIOnly:
    """MBI inhibitor without reversible Ki (pure MBI mechanism)."""

    def test_mbi_only_mechanism(self) -> None:
        """When ki_3a4_uM is inf but kinact/KI present → mechanism=='MBI'."""
        inh = DDIInhibitor(
            name="PureMBI",
            cmax_uM=5.0,
            kinact=2.0,
            KI=3.0,
            # ki_3a4_uM defaults to inf → no reversible inhibition
        )
        report = assess_ddi_risk(inh)
        assert report.mechanism == "MBI"


class TestBackwardCompat:
    """kinact=None (default) → skip MBI, mechanism stays 'competitive'."""

    def test_no_mbi_params_mechanism_competitive(self) -> None:
        inh = DDIInhibitor(
            name="ReversibleOnly",
            cmax_uM=5.0,
            ki_3a4_uM=10.0,
        )
        report = assess_ddi_risk(inh)
        assert report.mechanism == "competitive"

    def test_no_mbi_params_r2_unchanged(self) -> None:
        """Without kinact/KI, R2 should stay at default 1.0."""
        inh = DDIInhibitor(
            name="ReversibleOnly",
            cmax_uM=5.0,
            ki_3a4_uM=10.0,
        )
        report = assess_ddi_risk(inh)
        assert report.R2_3a4 == 1.0

    def test_no_mbi_params_no_limitation_warning(self) -> None:
        """No MBI params → no DDI_MBI_LIMITATION in flags."""
        inh = DDIInhibitor(
            name="ReversibleOnly",
            cmax_uM=5.0,
            ki_3a4_uM=10.0,
        )
        report = assess_ddi_risk(inh)
        limitation_flags = [f for f in report.flags if DDI_MBI_LIMITATION in f]
        assert len(limitation_flags) == 0

    def test_r1_unchanged_with_no_mbi(self) -> None:
        """R1 values should be the same whether or not MBI fields are None."""
        inh = DDIInhibitor(
            name="ReversibleOnly",
            cmax_uM=5.0,
            ki_3a4_uM=10.0,
        )
        report = assess_ddi_risk(inh)
        expected_r1 = 1.0 + 5.0 / 10.0
        assert report.R1_3a4 == pytest.approx(expected_r1, rel=1e-9)


class TestEdgeCases:
    """Edge cases: very high kinact, R2 approaches 0."""

    def test_very_high_kinact_r2_approaches_zero(self) -> None:
        """With extremely high kinact, R2 → 0 (near-complete inactivation)."""
        inh = DDIInhibitor(
            name="SuperMBI",
            cmax_uM=10.0,
            kinact=1e6,  # very high kinact
            KI=1.0,  # reasonable KI
        )
        report = assess_ddi_risk(inh)
        # R2 = 1/(1 + (1e6/0.000289) * 10/(1+10)) → extremely small
        assert report.R2_3a4 < 0.001
        assert report.R2_3a4 > 0.0  # still positive

    def test_very_high_kinact_strong_mbi_warning(self) -> None:
        """Very high kinact should trigger strong MBI inhibitor warning (R2 < 0.8)."""
        inh = DDIInhibitor(
            name="SuperMBI",
            cmax_uM=10.0,
            kinact=1e6,
            KI=1.0,
        )
        report = assess_ddi_risk(inh)
        strong_flags = [f for f in report.flags if "Strong MBI inhibitor" in f]
        assert len(strong_flags) == 1

    def test_erythromycin_like_strong_mbi_warning(self) -> None:
        """Erythromycin-like params (kinact=3.6, KI=5.9, cmax=3.6) should
        trigger strong MBI warning since R2 will be << 0.8."""
        inh = DDIInhibitor(
            name="ErythroLike",
            cmax_uM=3.6,
            ki_3a4_uM=10.0,
            kinact=3.6,
            KI=5.9,
        )
        report = assess_ddi_risk(inh)
        # R2 = 1/(1 + (3.6/0.000289) * 3.6/(5.9+3.6)) ≈ very small
        assert report.R2_3a4 < 0.8
        strong_flags = [f for f in report.flags if "Strong MBI inhibitor" in f]
        assert len(strong_flags) == 1

    def test_kinact_none_ki_set_skips_mbi(self) -> None:
        """If only one of kinact/KI is set, MBI is skipped."""
        inh = DDIInhibitor(
            name="Partial",
            cmax_uM=5.0,
            kinact=None,
            KI=5.0,
        )
        report = assess_ddi_risk(inh)
        assert report.mechanism == "competitive"
        assert report.R2_3a4 == 1.0
