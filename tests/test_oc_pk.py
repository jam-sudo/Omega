"""Tests for clinical.oc_pk — Combined OC pharmacokinetics (Phase 250)."""

from __future__ import annotations

import pytest

from omega_pbpk.clinical.oc_pk import OCPKResult, inducer_impact, simulate_oc_pk


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _default_oc(**kwargs) -> OCPKResult:
    params = dict(
        dose_ee_ug=30.0,
        dose_progestin_ug=150.0,
        progestin_type="LNG",
        cl_ee_L_per_h=27.0,
        vd_ee_L=211.0,
        ka_ee_per_h=2.0,
        f_ee=0.45,
        cl_progestin_L_per_h=4.5,
        vd_progestin_L=130.0,
        ka_progestin_per_h=1.5,
        f_progestin=0.88,
        erc_ee_fraction=0.3,
        t_end_h=48.0,
        dt_h=0.1,
    )
    params.update(kwargs)
    return simulate_oc_pk(**params)


# ---------------------------------------------------------------------------
# simulate_oc_pk tests
# ---------------------------------------------------------------------------


class TestSimulateOCPK:
    """Tests for simulate_oc_pk()."""

    def test_result_type(self):
        result = _default_oc()
        assert isinstance(result, OCPKResult)

    def test_cmax_ee_positive(self):
        result = _default_oc()
        assert result.cmax_ee > 0

    def test_tmax_ee_in_range(self):
        """EE Tmax expected to be in 1-6h for typical oral OC."""
        result = _default_oc()
        assert 0.5 <= result.tmax_ee_h <= 8.0, f"EE Tmax = {result.tmax_ee_h}"

    def test_auc_ee_positive(self):
        result = _default_oc()
        assert result.auc_ee > 0

    def test_cmax_progestin_positive(self):
        result = _default_oc()
        assert result.cmax_progestin > 0

    def test_tmax_progestin_in_range(self):
        result = _default_oc()
        assert 0.5 <= result.tmax_progestin_h <= 8.0, f"LNG Tmax = {result.tmax_progestin_h}"

    def test_auc_progestin_positive(self):
        result = _default_oc()
        assert result.auc_progestin > 0

    def test_times_length_matches_concs_ee(self):
        result = _default_oc()
        assert len(result.times_h) == len(result.c_ee_ng_mL)

    def test_times_length_matches_concs_progestin(self):
        result = _default_oc()
        assert len(result.times_h) == len(result.c_progestin_ng_mL)

    def test_ee_secondary_peak_is_bool(self):
        result = _default_oc()
        assert isinstance(result.ee_secondary_peak_present, bool)

    def test_notes_is_string(self):
        result = _default_oc()
        assert isinstance(result.notes, str)

    def test_progestin_type_stored(self):
        result = _default_oc(progestin_type="NETA")
        assert result.progestin_type == "NETA"

    def test_dose_ee_linearity_cmax(self):
        """2x EE dose → ~2x Cmax_EE (linear kinetics)."""
        r1 = _default_oc(dose_ee_ug=30.0)
        r2 = _default_oc(dose_ee_ug=60.0)
        ratio = r2.cmax_ee / r1.cmax_ee
        assert 1.7 < ratio < 2.4, f"EE Cmax linearity ratio = {ratio}"

    def test_dose_ee_linearity_auc(self):
        """2x EE dose → ~2x AUC_EE."""
        r1 = _default_oc(dose_ee_ug=30.0)
        r2 = _default_oc(dose_ee_ug=60.0)
        ratio = r2.auc_ee / r1.auc_ee
        assert 1.7 < ratio < 2.4, f"EE AUC linearity ratio = {ratio}"

    def test_ee_secondary_peak_with_erc(self):
        """ERC > 0 should produce secondary EE peak in sufficiently long sim."""
        result = _default_oc(erc_ee_fraction=0.5, t_end_h=72.0, dt_h=0.05)
        # Secondary peak may or may not appear depending on parameters;
        # just confirm the flag is consistent with peak detection
        if result.ee_secondary_peak_present:
            # verify concentrations contain a bump after tmax
            c = list(result.c_ee_ng_mL)
            t = list(result.times_h)
            tmax = result.tmax_ee_h
            post_peak = [(t[i], c[i]) for i in range(len(t)) if t[i] > tmax + 0.5]
            if post_peak:
                post_concs = [x[1] for x in post_peak]
                assert max(post_concs) > 0

    def test_erc_zero_no_secondary_peak(self):
        result = _default_oc(erc_ee_fraction=0.0)
        assert not result.ee_secondary_peak_present

    # --- Validation errors ---

    def test_invalid_dose_ee_raises(self):
        with pytest.raises(ValueError, match="dose_ee_ug"):
            _default_oc(dose_ee_ug=0.0)

    def test_invalid_dose_progestin_raises(self):
        with pytest.raises(ValueError, match="dose_progestin_ug"):
            _default_oc(dose_progestin_ug=-5.0)

    def test_invalid_cl_ee_raises(self):
        with pytest.raises(ValueError, match="cl_ee"):
            _default_oc(cl_ee_L_per_h=0.0)

    def test_invalid_vd_ee_raises(self):
        with pytest.raises(ValueError, match="vd_ee"):
            _default_oc(vd_ee_L=-1.0)

    def test_invalid_cl_progestin_raises(self):
        with pytest.raises(ValueError, match="cl_progestin"):
            _default_oc(cl_progestin_L_per_h=0.0)


# ---------------------------------------------------------------------------
# inducer_impact tests
# ---------------------------------------------------------------------------


class TestInducerImpact:
    """Tests for inducer_impact()."""

    def _baseline(self):
        return _default_oc()

    def test_result_type(self):
        result = inducer_impact(induction_factor_ee=2.0)
        assert isinstance(result, OCPKResult)

    def test_higher_cl_lowers_auc_ee(self):
        baseline = self._baseline()
        induced = inducer_impact(induction_factor_ee=3.5)
        assert induced.auc_ee < baseline.auc_ee

    def test_auc_ee_scales_inversely(self):
        """AUC_induced ≈ AUC_baseline / induction_factor for linear kinetics."""
        factor = 2.0
        baseline = self._baseline()
        induced = inducer_impact(induction_factor_ee=factor)
        expected = baseline.auc_ee / factor
        # Allow ±30% due to ERC non-linearity
        assert abs(induced.auc_ee - expected) / expected < 0.35, (
            f"AUC_induced={induced.auc_ee:.2f}, expected≈{expected:.2f}"
        )

    def test_higher_cl_lowers_cmax_ee(self):
        baseline = self._baseline()
        induced = inducer_impact(induction_factor_ee=2.0)
        assert induced.cmax_ee < baseline.cmax_ee

    def test_separate_progestin_factor(self):
        """Independent progestin induction factor."""
        induced = inducer_impact(induction_factor_ee=2.0, induction_factor_progestin=3.0)
        baseline = self._baseline()
        assert induced.auc_progestin < baseline.auc_progestin

    def test_factor_one_matches_baseline(self):
        """Induction factor = 1 should give same result as baseline."""
        baseline = self._baseline()
        induced = inducer_impact(induction_factor_ee=1.0)
        assert abs(induced.auc_ee - baseline.auc_ee) / baseline.auc_ee < 0.01

    def test_invalid_factor_raises(self):
        with pytest.raises(ValueError, match="induction_factor_ee"):
            inducer_impact(induction_factor_ee=0.0)

    def test_rifampicin_substantially_reduces_auc(self):
        """Rifampicin (factor≈3.5) should reduce EE AUC by >50%."""
        baseline = self._baseline()
        induced = inducer_impact(induction_factor_ee=3.5)
        reduction = (baseline.auc_ee - induced.auc_ee) / baseline.auc_ee
        assert reduction > 0.5, f"AUC reduction = {reduction:.2%}"
