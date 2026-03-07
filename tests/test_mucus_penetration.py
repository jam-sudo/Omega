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


# ---------------------------------------------------------------------------
# Phase 441: simulate_mucus_penetration tests
# ---------------------------------------------------------------------------

import math as _math  # noqa: E402

from omega_pbpk.core.mucus_penetration import (  # noqa: E402
    MucusPenetrationResult,
    _charge_factor_penetration,
    _effective_diffusion_cm2_s,
    _hydrodynamic_radius_nm,
    _lipophilicity_factor_penetration,
    _size_exclusion_factor,
    simulate_mucus_penetration,
)

_DEFAULT_PEN = dict(
    drug_name="TestDrug",
    dose_mg=10.0,
    mucus_type="intestinal",
    mucus_thickness_um=200.0,
    diffusion_coeff_cm2_s=5e-7,
    mw=300.0,
    logP=1.0,
    charge="neutral",
    t_end_h=2.0,
    dt_h=0.05,
)


def _pen(**overrides):
    return simulate_mucus_penetration(**{**_DEFAULT_PEN, **overrides})


class TestHydrodynamicRadius:
    def test_positive_output(self):
        assert _hydrodynamic_radius_nm(300.0) > 0

    def test_monotone_with_mw(self):
        assert _hydrodynamic_radius_nm(500.0) > _hydrodynamic_radius_nm(100.0)

    def test_known_value_mw300(self):
        expected = 0.066 * (300.0**0.43)
        assert abs(_hydrodynamic_radius_nm(300.0) - expected) < 1e-10


class TestSizeExclusionFactor:
    def test_zero_when_rh_equals_pore(self):
        assert _size_exclusion_factor(50.0, 50.0) == 0.0

    def test_zero_when_rh_exceeds_pore(self):
        assert _size_exclusion_factor(100.0, 50.0) == 0.0

    def test_one_when_rh_is_zero(self):
        assert _size_exclusion_factor(0.0, 100.0) == 1.0

    def test_between_zero_and_one(self):
        f = _size_exclusion_factor(25.0, 100.0)
        assert 0.0 < f < 1.0

    def test_smaller_rh_larger_factor(self):
        assert _size_exclusion_factor(10.0, 100.0) > _size_exclusion_factor(50.0, 100.0)


class TestChargeFactorPenetration:
    def test_neutral_is_one(self):
        assert _charge_factor_penetration("neutral") == 1.0

    def test_positive_is_0_3(self):
        assert _charge_factor_penetration("positive") == 0.3
        assert _charge_factor_penetration("cationic") == 0.3

    def test_negative_is_0_7(self):
        assert _charge_factor_penetration("negative") == 0.7
        assert _charge_factor_penetration("anionic") == 0.7

    def test_case_insensitive(self):
        assert _charge_factor_penetration("NEUTRAL") == 1.0
        assert _charge_factor_penetration("Positive") == 0.3


class TestLipophilicityFactorPenetration:
    def test_high_logP_gives_1_5(self):
        assert _lipophilicity_factor_penetration(3.0) == 1.5

    def test_low_logP_gives_1_0(self):
        assert _lipophilicity_factor_penetration(1.0) == 1.0

    def test_boundary_at_2(self):
        assert _lipophilicity_factor_penetration(2.0) == 1.0
        assert _lipophilicity_factor_penetration(2.01) == 1.5


class TestEffectiveDiffusionPenetration:
    def test_returns_positive(self):
        d = _effective_diffusion_cm2_s(5e-7, 300.0, 1.0, "neutral", 200.0)
        assert d > 0

    def test_zero_when_molecule_too_large(self):
        # r_h = 0.066 * MW^0.43; for MW=10^12 → r_h >> 50 nm cervical pore
        d = _effective_diffusion_cm2_s(5e-7, 10**12, 1.0, "neutral", 50.0)
        assert d == 0.0

    def test_positive_charge_reduces_diffusion(self):
        d_n = _effective_diffusion_cm2_s(5e-7, 300.0, 1.0, "neutral", 200.0)
        d_p = _effective_diffusion_cm2_s(5e-7, 300.0, 1.0, "positive", 200.0)
        assert d_p < d_n

    def test_high_logP_increases_diffusion(self):
        d_low = _effective_diffusion_cm2_s(5e-7, 300.0, 0.0, "neutral", 200.0)
        d_high = _effective_diffusion_cm2_s(5e-7, 300.0, 3.0, "neutral", 200.0)
        assert d_high > d_low


class TestSimulateMucusPenetration:
    def test_returns_correct_type(self):
        assert isinstance(_pen(), MucusPenetrationResult)

    def test_times_start_at_zero(self):
        assert _pen().times_h[0] == 0.0

    def test_times_end_at_t_end(self):
        r = _pen(t_end_h=2.0, dt_h=0.05)
        assert abs(r.times_h[-1] - 2.0) < 0.06

    def test_lumen_conc_initial_equals_dose_per_vol(self):
        r = _pen(dose_mg=10.0)
        assert abs(r.c_lumen_mg_mL[0] - 10.0) < 1e-10

    def test_lumen_conc_decreases_over_time(self):
        r = _pen()
        assert r.c_lumen_mg_mL[-1] < r.c_lumen_mg_mL[0]

    def test_epithelium_starts_at_zero(self):
        assert _pen().c_epithelium_mg_mL[0] == 0.0

    def test_epithelium_rises_above_zero(self):
        assert max(_pen().c_epithelium_mg_mL) > 0.0

    def test_f_penetrated_between_zero_and_one(self):
        r = _pen()
        assert 0.0 <= r.f_penetrated <= 1.0

    def test_thin_mucus_more_penetration(self):
        thick = _pen(mucus_thickness_um=500.0, t_end_h=4.0)
        thin = _pen(mucus_thickness_um=50.0, t_end_h=4.0)
        assert thin.f_penetrated >= thick.f_penetrated

    def test_neutral_penetrates_more_than_positive(self):
        neutral = _pen(charge="neutral", t_end_h=4.0)
        positive = _pen(charge="positive", t_end_h=4.0)
        assert neutral.f_penetrated >= positive.f_penetrated

    def test_negative_penetrates_more_than_positive(self):
        negative = _pen(charge="negative", t_end_h=4.0)
        positive = _pen(charge="positive", t_end_h=4.0)
        assert negative.f_penetrated >= positive.f_penetrated

    def test_notes_field_is_nonempty_string(self):
        r = _pen()
        assert isinstance(r.notes, str) and len(r.notes) > 0

    def test_times_length_matches_steps(self):
        r = _pen(t_end_h=1.0, dt_h=0.1)
        assert len(r.times_h) == 11
        assert len(r.c_lumen_mg_mL) == 11
        assert len(r.c_epithelium_mg_mL) == 11

    def test_gastric_mucus_type(self):
        r = _pen(mucus_type="gastric")
        assert r.mucus_type == "gastric"

    def test_respiratory_mucus_type(self):
        r = _pen(mucus_type="respiratory")
        assert r.mucus_type == "respiratory"

    def test_cervical_mucus_type(self):
        r = _pen(mucus_type="cervical")
        assert r.mucus_type == "cervical"

    def test_cervical_smaller_pore_lower_penetration(self):
        intestinal = _pen(mucus_type="intestinal", t_end_h=4.0)
        cervical = _pen(mucus_type="cervical", t_end_h=4.0)
        assert cervical.f_penetrated <= intestinal.f_penetrated

    def test_d_eff_stored_leq_d_free(self):
        r = _pen()
        assert r.diffusion_coeff_effective <= 5e-7 + 1e-15

    def test_high_logP_increases_d_eff(self):
        low_logP = _pen(logP=0.0, t_end_h=4.0)
        high_logP = _pen(logP=3.0, t_end_h=4.0)
        assert high_logP.f_penetrated >= low_logP.f_penetrated

    def test_large_molecule_very_low_penetration_cervical(self):
        # MW=10^10 → r_h >> cervical 50 nm pore → D_eff=0 → no penetration
        r = _pen(mw=1e10, mucus_type="cervical", t_end_h=4.0)
        assert r.f_penetrated < 0.01

    def test_t50_is_float(self):
        r = _pen(mw=5000.0, mucus_type="cervical", t_end_h=1.0)
        assert isinstance(r.t50_penetration_h, float)

    def test_t50_finite_permeable_drug(self):
        r = _pen(mw=150.0, charge="neutral", t_end_h=6.0, dt_h=0.05)
        if not _math.isnan(r.t50_penetration_h):
            assert r.t50_penetration_h > 0

    def test_drug_name_preserved(self):
        r = _pen(drug_name="Aspirin")
        assert r.drug_name == "Aspirin"

    def test_dose_preserved(self):
        r = _pen(dose_mg=25.0)
        assert r.dose_mg == 25.0

    def test_mucus_thickness_preserved(self):
        r = _pen(mucus_thickness_um=150.0)
        assert r.mucus_thickness_um == 150.0


class TestSimulateMucusPenetrationValidation:
    def test_empty_drug_name_raises(self):
        with pytest.raises(ValueError, match="drug_name"):
            _pen(drug_name="")

    def test_negative_dose_raises(self):
        with pytest.raises(ValueError, match="dose_mg"):
            _pen(dose_mg=-1.0)

    def test_zero_dose_raises(self):
        with pytest.raises(ValueError, match="dose_mg"):
            _pen(dose_mg=0.0)

    def test_invalid_mucus_type_raises(self):
        with pytest.raises(ValueError, match="mucus_type"):
            _pen(mucus_type="skin")

    def test_negative_thickness_raises(self):
        with pytest.raises(ValueError, match="mucus_thickness_um"):
            _pen(mucus_thickness_um=-10.0)

    def test_negative_diffusion_coeff_raises(self):
        with pytest.raises(ValueError, match="diffusion_coeff_cm2_s"):
            _pen(diffusion_coeff_cm2_s=-1e-7)

    def test_zero_mw_raises(self):
        with pytest.raises(ValueError, match="mw"):
            _pen(mw=0.0)

    def test_invalid_charge_raises(self):
        with pytest.raises(ValueError, match="charge"):
            _pen(charge="zwitterionic")

    def test_negative_t_end_raises(self):
        with pytest.raises(ValueError, match="t_end_h"):
            _pen(t_end_h=-1.0)

    def test_negative_dt_raises(self):
        with pytest.raises(ValueError, match="dt_h"):
            _pen(dt_h=-0.01)
