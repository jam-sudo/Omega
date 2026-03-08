"""
Tests for Phase 445 — Antimicrobial PK/PD Target Attainment
"""

import pytest

from omega_pbpk.clinical.antimicrobial_pk_pd_p445 import (
    AntimicrobialPKPDResult,
    assess_antimicrobial_pkpd,
    compare_dosing_regimens,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def beta_lactam_iv():
    """Amoxicillin-like: CL=5 L/h, Vd=20 L, dose=1000 mg, q8h, MIC=1 mg/L."""
    return assess_antimicrobial_pkpd(
        drug_name="amoxicillin",
        antibiotic_class="beta_lactam",
        dose_mg=1000.0,
        dosing_interval_h=8.0,
        mic_mg_L=1.0,
        cl_L_per_h=5.0,
        vd_L=20.0,
        route="iv",
    )


@pytest.fixture
def aminoglycoside_iv():
    """Gentamicin-like: CL=4 L/h, Vd=15 L, dose=500 mg, q24h, MIC=2 mg/L."""
    return assess_antimicrobial_pkpd(
        drug_name="gentamicin",
        antibiotic_class="aminoglycoside",
        dose_mg=500.0,
        dosing_interval_h=24.0,
        mic_mg_L=2.0,
        cl_L_per_h=4.0,
        vd_L=15.0,
        route="iv",
    )


@pytest.fixture
def fluoroquinolone_iv():
    """Ciprofloxacin-like: CL=20 L/h, Vd=200 L, dose=400 mg, q12h, MIC=0.25 mg/L."""
    return assess_antimicrobial_pkpd(
        drug_name="ciprofloxacin",
        antibiotic_class="fluoroquinolone",
        dose_mg=400.0,
        dosing_interval_h=12.0,
        mic_mg_L=0.25,
        cl_L_per_h=20.0,
        vd_L=200.0,
        route="iv",
    )


@pytest.fixture
def vancomycin_iv():
    """Vancomycin-like: CL=3 L/h, Vd=50 L, dose=1500 mg, q12h, MIC=1 mg/L."""
    return assess_antimicrobial_pkpd(
        drug_name="vancomycin",
        antibiotic_class="vancomycin",
        dose_mg=1500.0,
        dosing_interval_h=12.0,
        mic_mg_L=1.0,
        cl_L_per_h=3.0,
        vd_L=50.0,
        route="iv",
    )


# ---------------------------------------------------------------------------
# Return type tests
# ---------------------------------------------------------------------------


def test_return_type(beta_lactam_iv):
    assert isinstance(beta_lactam_iv, AntimicrobialPKPDResult)


def test_drug_name_preserved(beta_lactam_iv):
    assert beta_lactam_iv.drug_name == "amoxicillin"


# ---------------------------------------------------------------------------
# Basic PK metric tests
# ---------------------------------------------------------------------------


def test_t_half_positive(beta_lactam_iv):
    assert beta_lactam_iv.t_half_h > 0


def test_t_greater_mic_pct_range(beta_lactam_iv):
    assert 0.0 <= beta_lactam_iv.t_greater_mic_pct <= 100.0


def test_auc_over_mic_positive(beta_lactam_iv):
    assert beta_lactam_iv.auc_over_mic > 0


def test_cmax_over_mic_positive(beta_lactam_iv):
    assert beta_lactam_iv.cmax_over_mic > 0


# ---------------------------------------------------------------------------
# Dose-response relationship tests
# ---------------------------------------------------------------------------


def test_higher_dose_higher_t_greater_mic():
    """Higher dose should give higher T>MIC for beta-lactam.

    Use parameters where drug drops below MIC within the dosing interval:
    CL=10 L/h, Vd=5 L → ke=2/h, t_half=0.35h.
    dose=50 mg → C0=10 mg/L → t_mic=ln(10/8)/2=0.11h (short)
    dose=200 mg → C0=40 mg/L → t_mic=ln(40/8)/2=0.80h (longer)
    MIC=8 mg/L, interval=2h → drug drops below MIC in both cases.
    """
    low = assess_antimicrobial_pkpd(
        drug_name="drug",
        antibiotic_class="beta_lactam",
        dose_mg=50.0,
        dosing_interval_h=2.0,
        mic_mg_L=8.0,
        cl_L_per_h=10.0,
        vd_L=5.0,
        route="iv",
    )
    high = assess_antimicrobial_pkpd(
        drug_name="drug",
        antibiotic_class="beta_lactam",
        dose_mg=200.0,
        dosing_interval_h=2.0,
        mic_mg_L=8.0,
        cl_L_per_h=10.0,
        vd_L=5.0,
        route="iv",
    )
    assert high.t_greater_mic_pct > low.t_greater_mic_pct


def test_higher_mic_lower_t_greater_mic():
    """Higher MIC should give lower T>MIC.

    Use parameters where drug drops below MIC within the dosing interval:
    CL=10 L/h, Vd=5 L → ke=2/h, C0=100/5=20 mg/L, interval=2h.
    MIC=2: t_mic=ln(20/2)/2=1.15h → ~58% of 2h
    MIC=16: t_mic=ln(20/16)/2=0.11h → ~6% of 2h
    """
    low_mic = assess_antimicrobial_pkpd(
        drug_name="drug",
        antibiotic_class="beta_lactam",
        dose_mg=100.0,
        dosing_interval_h=2.0,
        mic_mg_L=2.0,
        cl_L_per_h=10.0,
        vd_L=5.0,
        route="iv",
    )
    high_mic = assess_antimicrobial_pkpd(
        drug_name="drug",
        antibiotic_class="beta_lactam",
        dose_mg=100.0,
        dosing_interval_h=2.0,
        mic_mg_L=16.0,
        cl_L_per_h=10.0,
        vd_L=5.0,
        route="iv",
    )
    assert high_mic.t_greater_mic_pct < low_mic.t_greater_mic_pct


# ---------------------------------------------------------------------------
# PK/PD index and threshold tests
# ---------------------------------------------------------------------------


def test_beta_lactam_pkpd_index(beta_lactam_iv):
    assert beta_lactam_iv.pkpd_index == "%T_MIC"


def test_beta_lactam_threshold(beta_lactam_iv):
    assert beta_lactam_iv.pharmacodynamic_target == 40.0


def test_aminoglycoside_pkpd_index(aminoglycoside_iv):
    assert aminoglycoside_iv.pkpd_index == "Cmax_MIC"


def test_aminoglycoside_threshold(aminoglycoside_iv):
    assert aminoglycoside_iv.pharmacodynamic_target == 10.0


def test_fluoroquinolone_pkpd_index(fluoroquinolone_iv):
    assert fluoroquinolone_iv.pkpd_index == "AUC_MIC"


def test_fluoroquinolone_threshold(fluoroquinolone_iv):
    assert fluoroquinolone_iv.pharmacodynamic_target == 125.0


def test_vancomycin_pkpd_index(vancomycin_iv):
    assert vancomycin_iv.pkpd_index == "AUC_MIC"


def test_vancomycin_threshold(vancomycin_iv):
    assert vancomycin_iv.pharmacodynamic_target == 400.0


# ---------------------------------------------------------------------------
# Target attainment tests
# ---------------------------------------------------------------------------


def test_pta_target_met_true_aminoglycoside(aminoglycoside_iv):
    """Gentamicin 500mg with MIC=2 mg/L, Vd=15L → Cmax=33 mg/L, ratio=16.7 ≥ 10."""
    assert aminoglycoside_iv.pta_target_met is True


def test_pta_target_not_met_when_low_dose():
    """Very low dose aminoglycoside should not meet target."""
    r = assess_antimicrobial_pkpd(
        drug_name="gentamicin",
        antibiotic_class="aminoglycoside",
        dose_mg=10.0,
        dosing_interval_h=24.0,
        mic_mg_L=2.0,
        cl_L_per_h=4.0,
        vd_L=15.0,
        route="iv",
    )
    assert r.pta_target_met is False


def test_bactericidal_activity_values():
    assert aminoglycoside_iv.__class__  # just instantiation check
    valid = {"achieved", "partial", "not_achieved"}
    r = assess_antimicrobial_pkpd(
        drug_name="g",
        antibiotic_class="aminoglycoside",
        dose_mg=500.0,
        dosing_interval_h=24.0,
        mic_mg_L=2.0,
        cl_L_per_h=4.0,
        vd_L=15.0,
        route="iv",
    )
    assert r.bactericidal_activity in valid


def test_dose_recommendation_values():
    valid = {"adequate", "increase_dose", "increase_frequency", "combine"}
    r = assess_antimicrobial_pkpd(
        drug_name="g",
        antibiotic_class="beta_lactam",
        dose_mg=1000.0,
        dosing_interval_h=8.0,
        mic_mg_L=1.0,
        cl_L_per_h=5.0,
        vd_L=20.0,
        route="iv",
    )
    assert r.dose_recommendation in valid


# ---------------------------------------------------------------------------
# Oral route test
# ---------------------------------------------------------------------------


def test_oral_route_returns_result():
    r = assess_antimicrobial_pkpd(
        drug_name="amoxicillin_oral",
        antibiotic_class="beta_lactam",
        dose_mg=500.0,
        dosing_interval_h=8.0,
        mic_mg_L=1.0,
        cl_L_per_h=5.0,
        vd_L=20.0,
        ka_per_h=2.0,
        f_oral=0.9,
        route="oral",
    )
    assert isinstance(r, AntimicrobialPKPDResult)
    assert r.t_half_h > 0
    assert 0.0 <= r.t_greater_mic_pct <= 100.0


# ---------------------------------------------------------------------------
# compare_dosing_regimens tests
# ---------------------------------------------------------------------------


def test_compare_regimens_count():
    results = compare_dosing_regimens(
        drug_name="drug",
        antibiotic_class="beta_lactam",
        mic_mg_L=1.0,
        cl_L_per_h=5.0,
        vd_L=20.0,
        doses_mg=[500.0, 1000.0],
        intervals_h=[6.0, 8.0, 12.0],
    )
    assert len(results) == 6  # 2 doses x 3 intervals


def test_compare_regimens_sorted_descending():
    results = compare_dosing_regimens(
        drug_name="drug",
        antibiotic_class="beta_lactam",
        mic_mg_L=1.0,
        cl_L_per_h=5.0,
        vd_L=20.0,
        doses_mg=[500.0, 1000.0, 2000.0],
        intervals_h=[8.0, 12.0],
    )
    for i in range(1, len(results)):
        assert results[i - 1].t_greater_mic_pct >= results[i].t_greater_mic_pct


# ---------------------------------------------------------------------------
# Validation error tests
# ---------------------------------------------------------------------------


def test_invalid_antibiotic_class():
    with pytest.raises(ValueError, match="antibiotic_class"):
        assess_antimicrobial_pkpd(
            drug_name="x",
            antibiotic_class="unknown_class",
            dose_mg=100.0,
            dosing_interval_h=8.0,
            mic_mg_L=1.0,
            cl_L_per_h=5.0,
            vd_L=20.0,
        )


def test_invalid_mic_zero():
    with pytest.raises(ValueError, match="mic_mg_L"):
        assess_antimicrobial_pkpd(
            drug_name="x",
            antibiotic_class="beta_lactam",
            dose_mg=100.0,
            dosing_interval_h=8.0,
            mic_mg_L=0.0,
            cl_L_per_h=5.0,
            vd_L=20.0,
        )


def test_invalid_mic_negative():
    with pytest.raises(ValueError, match="mic_mg_L"):
        assess_antimicrobial_pkpd(
            drug_name="x",
            antibiotic_class="beta_lactam",
            dose_mg=100.0,
            dosing_interval_h=8.0,
            mic_mg_L=-1.0,
            cl_L_per_h=5.0,
            vd_L=20.0,
        )


def test_invalid_route():
    with pytest.raises(ValueError, match="route"):
        assess_antimicrobial_pkpd(
            drug_name="x",
            antibiotic_class="beta_lactam",
            dose_mg=100.0,
            dosing_interval_h=8.0,
            mic_mg_L=1.0,
            cl_L_per_h=5.0,
            vd_L=20.0,
            route="intramuscular",
        )
