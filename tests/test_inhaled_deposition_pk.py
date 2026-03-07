"""Tests for Phase 751 — Inhaled Drug Deposition PK."""

import sys

import pytest

sys.path.insert(0, "src")

from omega_pbpk.core.inhaled_deposition_pk import (
    InhaledDepositionResult,
    compare_particle_sizes,
    simulate_inhaled_deposition_pk,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _default(**kwargs) -> InhaledDepositionResult:
    defaults = dict(
        drug_name="TestDrug",
        dose_mg=1.0,
        mmad_um=2.0,
        ka_alveolar_per_h=3.0,
        ka_oral_per_h=0.5,
        f_oral=0.3,
        cl_L_per_h=5.0,
        vd_L=50.0,
        t_end_h=12.0,
        dt_h=0.05,
    )
    defaults.update(kwargs)
    return simulate_inhaled_deposition_pk(**defaults)


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------


def test_returns_inhaled_deposition_result():
    res = _default()
    assert isinstance(res, InhaledDepositionResult)


def test_drug_name_preserved():
    res = _default(drug_name="Salbutamol")
    assert res.drug_name == "Salbutamol"


def test_dose_mg_preserved():
    res = _default(dose_mg=2.5)
    assert res.dose_mg == pytest.approx(2.5)


def test_mmad_um_preserved():
    res = _default(mmad_um=3.0)
    assert res.mmad_um == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# Basic PK sanity
# ---------------------------------------------------------------------------


def test_cmax_positive():
    res = _default()
    assert res.cmax_mg_L > 0.0


def test_auc_positive():
    res = _default()
    assert res.auc_mg_h_per_L > 0.0


def test_tmax_in_range():
    res = _default()
    assert 0.0 <= res.tmax_h <= 12.0


def test_times_and_plasma_same_length():
    res = _default()
    assert len(res.times_h) == len(res.c_plasma_mg_L)


def test_plasma_non_negative():
    res = _default()
    assert all(c >= 0.0 for c in res.c_plasma_mg_L)


# ---------------------------------------------------------------------------
# Deposition fractions
# ---------------------------------------------------------------------------


def test_fine_particles_higher_alveolar_than_very_coarse():
    # At very large MMAD (15µm), high oropharyngeal deposition reduces f_alveolar
    # below the fine particle (1µm) level
    fine = _default(mmad_um=1.0)
    very_coarse = _default(mmad_um=15.0)
    assert fine.f_alveolar > very_coarse.f_alveolar


def test_coarse_particles_higher_oropharynx_than_fine():
    fine = _default(mmad_um=1.0)
    coarse = _default(mmad_um=5.0)
    assert coarse.f_oropharynx > fine.f_oropharynx


def test_fractions_sum_le_one():
    for mmad in [1.0, 2.0, 3.0, 5.0, 8.0]:
        res = _default(mmad_um=mmad)
        total = res.f_alveolar + res.f_tracheobronchial + res.f_oropharynx
        assert total <= 1.0 + 1e-9, f"MMAD={mmad}: total={total}"


def test_all_fractions_non_negative():
    for mmad in [0.5, 1.0, 2.0, 5.0, 10.0]:
        res = _default(mmad_um=mmad)
        assert res.f_alveolar >= 0.0
        assert res.f_tracheobronchial >= 0.0
        assert res.f_oropharynx >= 0.0


# ---------------------------------------------------------------------------
# Systemic F and fine particle dose
# ---------------------------------------------------------------------------


def test_systemic_f_between_0_and_1():
    for mmad in [1.0, 2.0, 5.0]:
        res = _default(mmad_um=mmad)
        assert 0.0 <= res.systemic_f <= 1.0, f"MMAD={mmad}: systemic_f={res.systemic_f}"


def test_fine_particle_dose_equals_dose_times_f_alveolar():
    res = _default(dose_mg=3.0, mmad_um=2.0)
    expected = res.dose_mg * res.f_alveolar
    assert res.fine_particle_dose_mg == pytest.approx(expected, rel=1e-6)


# ---------------------------------------------------------------------------
# Dose linearity
# ---------------------------------------------------------------------------


def test_linear_dose_scaling_cmax():
    res1 = _default(dose_mg=1.0, mmad_um=2.0)
    res2 = _default(dose_mg=2.0, mmad_um=2.0)
    assert res2.cmax_mg_L == pytest.approx(2.0 * res1.cmax_mg_L, rel=1e-4)


def test_linear_dose_scaling_auc():
    res1 = _default(dose_mg=1.0)
    res2 = _default(dose_mg=4.0)
    assert res2.auc_mg_h_per_L == pytest.approx(4.0 * res1.auc_mg_h_per_L, rel=1e-4)


# ---------------------------------------------------------------------------
# compare_particle_sizes
# ---------------------------------------------------------------------------


def test_compare_particle_sizes_returns_list():
    results = compare_particle_sizes("Drug", 1.0, [1.0, 2.0, 5.0])
    assert isinstance(results, list)


def test_compare_particle_sizes_correct_length():
    mmad_values = [1.0, 2.0, 3.0, 4.0, 5.0]
    results = compare_particle_sizes("Drug", 1.0, mmad_values)
    assert len(results) == len(mmad_values)


def test_compare_particle_sizes_each_is_result():
    results = compare_particle_sizes("Drug", 1.0, [1.5, 3.0])
    for r in results:
        assert isinstance(r, InhaledDepositionResult)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_validation_dose_zero_raises():
    with pytest.raises(ValueError):
        _default(dose_mg=0.0)


def test_validation_dose_negative_raises():
    with pytest.raises(ValueError):
        _default(dose_mg=-1.0)


def test_validation_mmad_zero_raises():
    with pytest.raises(ValueError):
        _default(mmad_um=0.0)


def test_validation_cl_zero_raises():
    with pytest.raises(ValueError):
        _default(cl_L_per_h=0.0)


def test_validation_vd_zero_raises():
    with pytest.raises(ValueError):
        _default(vd_L=0.0)


# ---------------------------------------------------------------------------
# Notes
# ---------------------------------------------------------------------------


def test_notes_nonempty():
    res = _default()
    assert isinstance(res.notes, str)
    assert len(res.notes) > 0


def test_t_half_positive():
    res = _default()
    assert res.t_half_h > 0.0
