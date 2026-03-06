"""Tests for core/mucus_penetration.py — Phase 238."""

import pytest

from omega_pbpk.core.mucus_penetration import (
    MucusDiffResult,
    MucusPKResult,
    compare_charges,
    mucus_diffusivity,
    simulate_mucus_pk,
)


# ---------------------------------------------------------------------------
# mucus_diffusivity — basic checks
# ---------------------------------------------------------------------------


def test_cationic_lower_f_mucus_than_anionic():
    r_cat = mucus_diffusivity(mw=400.0, logP=2.0, charge="cationic")
    r_ani = mucus_diffusivity(mw=400.0, logP=2.0, charge="anionic")
    assert r_cat.f_mucus < r_ani.f_mucus


def test_neutral_between_cationic_and_anionic():
    mw, logp = 400.0, 2.0
    r_cat = mucus_diffusivity(mw, logp, "cationic")
    r_neu = mucus_diffusivity(mw, logp, "neutral")
    r_ani = mucus_diffusivity(mw, logp, "anionic")
    assert r_ani.f_mucus >= r_neu.f_mucus >= r_cat.f_mucus


def test_high_mw_lower_d_eff():
    r_low = mucus_diffusivity(mw=200.0, logP=1.0, charge="neutral")
    r_high = mucus_diffusivity(mw=2000.0, logP=1.0, charge="neutral")
    assert r_high.d_eff_m2_s < r_low.d_eff_m2_s


def test_f_mucus_in_range():
    for mw in [100.0, 500.0, 1500.0]:
        r = mucus_diffusivity(mw, 1.0, "neutral")
        assert 0.0 < r.f_mucus <= 1.0


def test_penetration_time_positive():
    r = mucus_diffusivity(400.0, 1.5, "neutral")
    assert r.penetration_time_h > 0.0


def test_d_free_positive():
    r = mucus_diffusivity(300.0, 1.0, "anionic")
    assert r.d_free_m2_s > 0.0


def test_mucus_solid_pct_effect():
    r_low = mucus_diffusivity(500.0, 1.0, "neutral", mucus_solid_pct=1.0)
    r_high = mucus_diffusivity(500.0, 1.0, "neutral", mucus_solid_pct=8.0)
    assert r_low.f_mucus > r_high.f_mucus


def test_charge_validation_error():
    with pytest.raises(ValueError, match="charge"):
        mucus_diffusivity(300.0, 1.0, charge="zwitterion")


# ---------------------------------------------------------------------------
# simulate_mucus_pk — validation
# ---------------------------------------------------------------------------


def test_simulate_dose_validation():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_mucus_pk(
            "D", dose_mg=0, mw=300, logP=1, charge="neutral", route="oral", cl_L_per_h=5, vd_L=50
        )


def test_simulate_cl_validation():
    with pytest.raises(ValueError, match="cl_L_per_h"):
        simulate_mucus_pk(
            "D", dose_mg=100, mw=300, logP=1, charge="neutral", route="oral", cl_L_per_h=0, vd_L=50
        )


def test_simulate_vd_validation():
    with pytest.raises(ValueError, match="vd_L"):
        simulate_mucus_pk(
            "D", dose_mg=100, mw=300, logP=1, charge="neutral", route="oral", cl_L_per_h=5, vd_L=-1
        )


def test_simulate_charge_validation():
    with pytest.raises(ValueError, match="charge"):
        simulate_mucus_pk(
            "D",
            dose_mg=100,
            mw=300,
            logP=1,
            charge="amphoteric",
            route="oral",
            cl_L_per_h=5,
            vd_L=50,
        )


def test_simulate_route_validation():
    with pytest.raises(ValueError, match="route"):
        simulate_mucus_pk(
            "D",
            dose_mg=100,
            mw=300,
            logP=1,
            charge="neutral",
            route="rectal",
            cl_L_per_h=5,
            vd_L=50,
        )


# ---------------------------------------------------------------------------
# simulate_mucus_pk — results
# ---------------------------------------------------------------------------


def test_f_absorbed_in_range():
    r = simulate_mucus_pk("Drug", 100.0, 300.0, 2.0, "neutral", "oral", 5.0, 50.0)
    assert 0.0 <= r.f_absorbed <= 1.0


def test_cmax_positive():
    r = simulate_mucus_pk("Drug", 100.0, 300.0, 2.0, "neutral", "oral", 5.0, 50.0)
    assert r.cmax_mg_L > 0.0


def test_auc_positive():
    r = simulate_mucus_pk("Drug", 100.0, 300.0, 2.0, "neutral", "oral", 5.0, 50.0)
    assert r.auc_mg_h_per_L > 0.0


def test_inhaled_faster_than_vaginal():
    """Thinner mucus (inhaled 10 µm) → higher f_absorbed than vaginal (150 µm)."""
    base = dict(
        drug_name="D",
        dose_mg=100.0,
        mw=300.0,
        logP=1.0,
        charge="neutral",
        cl_L_per_h=5.0,
        vd_L=50.0,
        t_end_h=12.0,
    )
    r_inh = simulate_mucus_pk(**base, route="inhaled")
    r_vag = simulate_mucus_pk(**base, route="vaginal")
    # inhaled route has thinner default mucus → more absorbed
    assert r_inh.f_absorbed >= r_vag.f_absorbed


def test_result_types():
    r = simulate_mucus_pk("Drug", 100.0, 300.0, 2.0, "neutral", "oral", 5.0, 50.0)
    assert isinstance(r, MucusPKResult)
    assert isinstance(r.mucus_diff, MucusDiffResult)
    assert isinstance(r.times_h, list)
    assert isinstance(r.c_plasma_mg_L, list)
    assert isinstance(r.cmax_mg_L, float)
    assert isinstance(r.tmax_h, float)
    assert isinstance(r.auc_mg_h_per_L, float)
    assert isinstance(r.f_absorbed, float)
    assert isinstance(r.notes, str)


# ---------------------------------------------------------------------------
# compare_charges
# ---------------------------------------------------------------------------


def test_compare_charges_returns_three():
    results = compare_charges("Drug", 100.0, 300.0, 2.0, 5.0, 50.0, route="oral")
    assert len(results) == 3


def test_compare_charges_sorted_by_f_absorbed():
    results = compare_charges("Drug", 100.0, 300.0, 2.0, 5.0, 50.0, route="oral")
    absorbed = [r.f_absorbed for r in results]
    assert absorbed == sorted(absorbed, reverse=True)


def test_compare_charges_anionic_highest():
    results = compare_charges("Drug", 100.0, 500.0, 2.0, 5.0, 50.0, route="oral")
    assert results[0].charge == "anionic"


def test_compare_charges_cationic_lowest():
    results = compare_charges("Drug", 100.0, 500.0, 2.0, 5.0, 50.0, route="oral")
    assert results[-1].charge == "cationic"
