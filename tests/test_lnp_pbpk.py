"""Tests for Lipid Nanoparticle (LNP) PBPK model."""

from __future__ import annotations

import pytest

from omega_pbpk.core.lnp_pbpk import LNPPBPKResult, optimize_pka, simulate_lnp_pk


@pytest.fixture()
def default_result() -> LNPPBPKResult:
    return simulate_lnp_pk("TestLNP", dose_mg_lipid=10.0)


# --- Basic result type and structure ---


def test_returns_lnp_pbpk_result(default_result):
    assert isinstance(default_result, LNPPBPKResult)


def test_times_h_length_gt_1(default_result):
    assert len(default_result.times_h) > 1


def test_a_plasma_mg_all_non_negative(default_result):
    assert all(v >= 0 for v in default_result.a_plasma_mg)


# --- Validation ---


def test_dose_le_zero_raises():
    with pytest.raises(ValueError, match="dose_mg_lipid"):
        simulate_lnp_pk("Bad", dose_mg_lipid=0)


def test_cargo_loading_pct_out_of_range_raises():
    with pytest.raises(ValueError, match="cargo_loading_pct"):
        simulate_lnp_pk("Bad", dose_mg_lipid=10.0, cargo_loading_pct=0)
    with pytest.raises(ValueError, match="cargo_loading_pct"):
        simulate_lnp_pk("Bad", dose_mg_lipid=10.0, cargo_loading_pct=100)


# --- Liver dominance ---


def test_liver_pct_at_24h_gt_spleen(default_result):
    assert default_result.liver_pct_at_24h > default_result.spleen_pct_at_24h


def test_liver_to_spleen_ratio_gt_1(default_result):
    assert default_result.liver_to_spleen_ratio > 1.0


# --- ApoE effect ---


def test_higher_apoe_higher_liver():
    low = simulate_lnp_pk("LNP", dose_mg_lipid=10.0, apoe_adsorption=0.5)
    high = simulate_lnp_pk("LNP", dose_mg_lipid=10.0, apoe_adsorption=2.0)
    assert high.liver_pct_at_24h > low.liver_pct_at_24h


# --- PEG effects ---


def test_higher_peg_longer_half_life():
    low_peg = simulate_lnp_pk("LNP", dose_mg_lipid=10.0, peg_mol_pct=0.5)
    high_peg = simulate_lnp_pk("LNP", dose_mg_lipid=10.0, peg_mol_pct=2.0)
    assert high_peg.t_half_plasma_h > low_peg.t_half_plasma_h


def test_higher_peg_lower_liver_uptake():
    low_peg = simulate_lnp_pk("LNP", dose_mg_lipid=10.0, peg_mol_pct=0.5)
    high_peg = simulate_lnp_pk("LNP", dose_mg_lipid=10.0, peg_mol_pct=2.0)
    assert high_peg.liver_pct_at_24h < low_peg.liver_pct_at_24h


# --- Cargo ---


def test_peak_cargo_liver_gt_0(default_result):
    assert default_result.peak_cargo_liver_mg > 0


def test_endosomal_escape_in_0_1(default_result):
    assert 0 < default_result.endosomal_escape_efficiency < 1


def test_pka_6_5_better_cargo_than_7_5():
    r65 = simulate_lnp_pk("LNP", dose_mg_lipid=10.0, pka_apparent=6.5)
    r75 = simulate_lnp_pk("LNP", dose_mg_lipid=10.0, pka_apparent=7.5)
    assert r65.peak_cargo_liver_mg > r75.peak_cargo_liver_mg


def test_sirna_degrades_faster_than_mrna():
    mrna = simulate_lnp_pk("LNP", dose_mg_lipid=10.0, cargo_type="mrna", mrna_half_life_h=48.0)
    sirna = simulate_lnp_pk("LNP", dose_mg_lipid=10.0, cargo_type="sirna")
    # siRNA half-life 24h < mRNA 48h → faster decay → lower peak
    assert sirna.peak_cargo_liver_mg < mrna.peak_cargo_liver_mg


def test_cargo_liver_end_lt_peak(default_result):
    assert default_result.cargo_liver_mg[-1] < default_result.peak_cargo_liver_mg


# --- Dose linearity ---


def test_2x_dose_2x_peak_cargo():
    r1 = simulate_lnp_pk("LNP", dose_mg_lipid=5.0)
    r2 = simulate_lnp_pk("LNP", dose_mg_lipid=10.0)
    ratio = r2.peak_cargo_liver_mg / r1.peak_cargo_liver_mg
    assert 1.9 < ratio < 2.1


# --- PK metrics ---


def test_auc_plasma_gt_0(default_result):
    assert default_result.auc_plasma > 0


def test_cmax_plasma_gt_0(default_result):
    assert default_result.cmax_plasma_mg_L > 0


def test_t_half_plasma_gt_0(default_result):
    assert default_result.t_half_plasma_h > 0


# --- Mass balance ---


def test_liver_plus_spleen_pct_le_100(default_result):
    end_liver = default_result.a_liver_pct[-1]
    end_spleen = default_result.a_spleen_pct[-1]
    assert end_liver + end_spleen <= 100.0


# --- Liver accumulation ---


def test_liver_pct_72h_ge_24h():
    # With negligible hepatic clearance, LNP accumulates in liver over time
    r = simulate_lnp_pk("LNP", dose_mg_lipid=10.0, cl_liver_L_per_h=0.0)
    assert r.liver_pct_at_72h >= r.liver_pct_at_24h


# --- Notes ---


def test_notes_contain_name_and_cargo(default_result):
    assert "TestLNP" in default_result.notes
    assert "mrna" in default_result.notes


# --- optimize_pka ---


def test_optimize_pka_returns_6_results():
    results = optimize_pka("LNP", dose_mg_lipid=10.0)
    assert len(results) == 6


def test_optimize_pka_sorted_descending():
    results = optimize_pka("LNP", dose_mg_lipid=10.0)
    peaks = [r.peak_cargo_liver_mg for r in results]
    assert peaks == sorted(peaks, reverse=True)


# --- mRNA half-life effect ---


def test_mrna_half_life_effect():
    short = simulate_lnp_pk("LNP", dose_mg_lipid=10.0, mrna_half_life_h=24.0)
    long = simulate_lnp_pk("LNP", dose_mg_lipid=10.0, mrna_half_life_h=72.0)
    # Shorter half-life → cargo decays faster → lower peak
    assert short.peak_cargo_liver_mg < long.peak_cargo_liver_mg
