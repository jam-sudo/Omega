"""Tests for Phase 589 — Antimicrobial PK/PD Target Attainment."""

import pytest

from omega_pbpk.clinical.antimicrobial_pk_pd import (
    antibiotic_pta_simulation,
    mbc_to_mic_ratio,
    pk_pd_index,
    target_attainment,
)

# ---------------------------------------------------------------------------
# pk_pd_index tests
# ---------------------------------------------------------------------------


def test_pk_pd_index_beta_lactam():
    assert pk_pd_index("beta_lactam") == "fT_MIC"


def test_pk_pd_index_aminoglycoside():
    assert pk_pd_index("aminoglycoside") == "Cmax_MIC"


def test_pk_pd_index_fluoroquinolone():
    assert pk_pd_index("fluoroquinolone") == "AUC_MIC"


def test_pk_pd_index_vancomycin():
    assert pk_pd_index("vancomycin") == "AUC_MIC"


def test_pk_pd_index_azithromycin():
    assert pk_pd_index("azithromycin") == "AUC_MIC"


def test_pk_pd_index_invalid_raises():
    with pytest.raises(ValueError):
        pk_pd_index("unknown_drug")


# ---------------------------------------------------------------------------
# target_attainment tests
# ---------------------------------------------------------------------------


def test_target_attainment_beta_lactam_achieved():
    # t_above_mic_h = 6h, tau = 10h → 60% → target 50% → achieved
    result = target_attainment(
        auc_mg_h_L=50.0,
        cmax_mg_L=10.0,
        mic_mg_L=1.0,
        t_above_mic_h=6.0,
        tau_h=10.0,
        antibiotic_class="beta_lactam",
    )
    assert result["pk_pd_index_name"] == "fT_MIC"
    assert result["attainment_achieved"] is True
    assert abs(result["index_value"] - 60.0) < 1e-6


def test_target_attainment_beta_lactam_not_achieved():
    # t_above_mic_h = 4h, tau = 10h → 40% < 50%
    result = target_attainment(
        auc_mg_h_L=50.0,
        cmax_mg_L=10.0,
        mic_mg_L=1.0,
        t_above_mic_h=4.0,
        tau_h=10.0,
        antibiotic_class="beta_lactam",
    )
    assert result["attainment_achieved"] is False


def test_target_attainment_aminoglycoside_achieved():
    # Cmax/MIC = 12 → target 10 → achieved
    result = target_attainment(
        auc_mg_h_L=60.0,
        cmax_mg_L=12.0,
        mic_mg_L=1.0,
        t_above_mic_h=2.0,
        tau_h=24.0,
        antibiotic_class="aminoglycoside",
    )
    assert result["pk_pd_index_name"] == "Cmax_MIC"
    assert result["attainment_achieved"] is True
    assert abs(result["index_value"] - 12.0) < 1e-6


def test_target_attainment_aminoglycoside_not_achieved():
    # Cmax/MIC = 8 < 10
    result = target_attainment(
        auc_mg_h_L=60.0,
        cmax_mg_L=8.0,
        mic_mg_L=1.0,
        t_above_mic_h=2.0,
        tau_h=24.0,
        antibiotic_class="aminoglycoside",
    )
    assert result["attainment_achieved"] is False


def test_target_attainment_fluoroquinolone_achieved():
    # AUC/MIC = 200 → target 125 → achieved
    result = target_attainment(
        auc_mg_h_L=200.0,
        cmax_mg_L=10.0,
        mic_mg_L=1.0,
        t_above_mic_h=12.0,
        tau_h=24.0,
        antibiotic_class="fluoroquinolone",
    )
    assert result["pk_pd_index_name"] == "AUC_MIC"
    assert result["attainment_achieved"] is True


def test_target_attainment_vancomycin_target():
    result = target_attainment(
        auc_mg_h_L=500.0,
        cmax_mg_L=20.0,
        mic_mg_L=1.0,
        t_above_mic_h=12.0,
        tau_h=12.0,
        antibiotic_class="vancomycin",
    )
    assert result["target_value"] == 400.0
    assert result["attainment_achieved"] is True


def test_target_attainment_attainment_pct():
    result = target_attainment(
        auc_mg_h_L=250.0,
        cmax_mg_L=10.0,
        mic_mg_L=1.0,
        t_above_mic_h=10.0,
        tau_h=24.0,
        antibiotic_class="fluoroquinolone",
    )
    # index=250, target=125 → 200%
    assert abs(result["attainment_pct"] - 200.0) < 1e-4


# ---------------------------------------------------------------------------
# mbc_to_mic_ratio tests
# ---------------------------------------------------------------------------


def test_mbc_to_mic_ratio_bactericidal():
    result = mbc_to_mic_ratio(mbc=2.0, mic=1.0)
    assert result["ratio"] == 2.0
    assert result["kill_type"] == "bactericidal"


def test_mbc_to_mic_ratio_intermediate():
    result = mbc_to_mic_ratio(mbc=8.0, mic=1.0)
    assert result["kill_type"] == "intermediate"


def test_mbc_to_mic_ratio_bacteriostatic():
    result = mbc_to_mic_ratio(mbc=32.0, mic=1.0)
    assert result["kill_type"] == "bacteriostatic"


def test_mbc_to_mic_ratio_exact_boundary_4():
    result = mbc_to_mic_ratio(mbc=4.0, mic=1.0)
    assert result["kill_type"] == "intermediate"


# ---------------------------------------------------------------------------
# antibiotic_pta_simulation tests
# ---------------------------------------------------------------------------


def test_pta_simulation_c_free_positive():
    result = antibiotic_pta_simulation(
        drug_name="ampicillin",
        dose_mg=500.0,
        interval_h=6.0,
        cl_L_per_h=5.0,
        vd_L=20.0,
        f_unbound=0.8,
        mic_mg_L=1.0,
        antibiotic_class="beta_lactam",
        n_doses=5,
    )
    assert all(cf >= 0 for cf in result["c_free"])
    assert max(result["c_free"]) > 0.0


def test_pta_simulation_t_above_mic_in_range():
    result = antibiotic_pta_simulation(
        drug_name="ampicillin",
        dose_mg=500.0,
        interval_h=6.0,
        cl_L_per_h=5.0,
        vd_L=20.0,
        f_unbound=0.8,
        mic_mg_L=1.0,
        antibiotic_class="beta_lactam",
        n_doses=5,
    )
    assert 0.0 <= result["t_above_mic_pct"] <= 100.0


def test_pta_simulation_higher_dose_higher_attainment():
    low_dose = antibiotic_pta_simulation(
        drug_name="cipro",
        dose_mg=200.0,
        interval_h=12.0,
        cl_L_per_h=8.0,
        vd_L=150.0,
        f_unbound=0.7,
        mic_mg_L=0.1,
        antibiotic_class="fluoroquinolone",
        n_doses=5,
    )
    high_dose = antibiotic_pta_simulation(
        drug_name="cipro",
        dose_mg=1000.0,
        interval_h=12.0,
        cl_L_per_h=8.0,
        vd_L=150.0,
        f_unbound=0.7,
        mic_mg_L=0.1,
        antibiotic_class="fluoroquinolone",
        n_doses=5,
    )
    assert high_dose["auc_free_mg_h_L"] > low_dose["auc_free_mg_h_L"]


def test_pta_simulation_invalid_class_raises():
    with pytest.raises(ValueError):
        antibiotic_pta_simulation(
            drug_name="x",
            dose_mg=500.0,
            interval_h=8.0,
            cl_L_per_h=5.0,
            vd_L=20.0,
            f_unbound=0.5,
            mic_mg_L=1.0,
            antibiotic_class="invalid_class",
        )


def test_pta_simulation_returns_pk_pd_attainment_dict():
    result = antibiotic_pta_simulation(
        drug_name="gentamicin",
        dose_mg=480.0,
        interval_h=24.0,
        cl_L_per_h=4.0,
        vd_L=20.0,
        f_unbound=0.9,
        mic_mg_L=1.0,
        antibiotic_class="aminoglycoside",
        n_doses=5,
    )
    pk_pd = result["pk_pd_attainment"]
    assert "attainment_achieved" in pk_pd
    assert "index_value" in pk_pd


def test_pta_simulation_cmax_free_positive():
    result = antibiotic_pta_simulation(
        drug_name="vancomycin",
        dose_mg=1000.0,
        interval_h=12.0,
        cl_L_per_h=3.0,
        vd_L=40.0,
        f_unbound=0.7,
        mic_mg_L=1.0,
        antibiotic_class="vancomycin",
        n_doses=5,
    )
    assert result["cmax_free"] > 0.0
