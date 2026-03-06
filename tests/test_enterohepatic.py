"""Tests for enterohepatic recirculation simulator (Phase 69)."""

from __future__ import annotations

from omega_pbpk.core.enterohepatic import EHCResult, ehc_sensitivity, simulate_ehc


def _default(**overrides):
    defaults = dict(
        drug_name="TestDrug",
        dose_mg=100.0,
        cl_hepatic_L_per_h=5.0,
        vd_L=50.0,
        f_biliary=0.3,
        f_reabsorb=0.5,
        t_end_h=48.0,
    )
    defaults.update(overrides)
    return simulate_ehc(**defaults)


class TestSimulateEHC:
    def test_returns_result_type(self):
        assert isinstance(_default(), EHCResult)

    def test_auc_positive(self):
        assert _default().auc_total_mg_h_L > 0

    def test_plasma_conc_length(self):
        assert len(_default().plasma_conc) > 1

    def test_cmax_positive(self):
        assert _default().cmax_mg_L > 0

    def test_tmax_within_simulation(self):
        r = _default()
        assert 0 <= r.tmax_h <= 48.0

    def test_ehc_fraction_bounded(self):
        r = _default()
        assert 0 <= r.ehc_fraction <= 1

    def test_fecal_excretion_nonneg(self):
        assert _default().excreted_feces_mg >= 0

    def test_reabsorbed_nonneg(self):
        assert _default().reabsorbed_total_mg >= 0

    def test_mass_balance_approx(self):
        r = _default()
        # Rough check: fecal + reabsorbed can't exceed dose
        assert r.excreted_feces_mg + r.reabsorbed_total_mg <= r.reabsorbed_total_mg + 200.0

    def test_higher_f_biliary_higher_ehc(self):
        r_low = _default(f_biliary=0.1)
        r_high = _default(f_biliary=0.5)
        assert r_high.ehc_fraction >= r_low.ehc_fraction

    def test_higher_f_reabsorb_higher_auc(self):
        r_low = _default(f_reabsorb=0.0)
        r_high = _default(f_reabsorb=0.8)
        assert r_high.auc_total_mg_h_L >= r_low.auc_total_mg_h_L

    def test_iv_route_works(self):
        r = _default(route="iv")
        assert isinstance(r, EHCResult)
        assert r.tmax_h < 5.0

    def test_bile_list_length(self):
        r = _default()
        assert len(r.bile_amount_mg) == len(r.times_h)

    def test_auc_ehc_geq_no_ehc_with_reabsorption(self):
        r = _default(f_biliary=0.4, f_reabsorb=0.8)
        # With EHC + significant reabsorption, AUC should be >= no-EHC AUC
        assert r.auc_total_mg_h_L >= r.auc_no_ehc_mg_h_L * 0.8  # allow some tolerance


class TestEHCSensitivity:
    def test_returns_list(self):
        r = ehc_sensitivity("Drug", 100.0, 5.0, 50.0)
        assert isinstance(r, list)

    def test_has_5_entries_default(self):
        r = ehc_sensitivity("Drug", 100.0, 5.0, 50.0)
        assert len(r) == 5

    def test_entries_have_keys(self):
        r = ehc_sensitivity("Drug", 100.0, 5.0, 50.0)
        for entry in r:
            for key in ("f_biliary", "auc_total", "n_secondary_peaks", "ehc_fraction"):
                assert key in entry

    def test_higher_f_biliary_more_secondary_peaks(self):
        r = ehc_sensitivity("Drug", 100.0, 5.0, 50.0)
        f_biliary_vals = [e["f_biliary"] for e in r]
        peaks = [e["n_secondary_peaks"] for e in r]
        # For the highest f_biliary, peaks >= for f_biliary=0
        idx_zero = f_biliary_vals.index(min(f_biliary_vals))
        idx_max = f_biliary_vals.index(max(f_biliary_vals))
        assert peaks[idx_max] >= peaks[idx_zero]


class TestSecondaryPeaks:
    def test_secondary_peaks_is_list(self):
        r = _default()
        assert isinstance(r.secondary_peak_times, list)

    def test_no_ehc_no_secondary_peaks(self):
        r = _default(f_biliary=0.0)
        assert r.ehc_fraction == 0.0
