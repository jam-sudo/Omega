"""Tests for saturable_tissue_binding module."""

import pytest

from omega_pbpk.core.saturable_tissue_binding import (
    TissueBindingResult,
    compare_tissues,
    fu_tissue,
    langmuir_bound,
    simulate_tissue_binding,
)

BMAX = 100.0  # nmol/g
KD = 10.0  # nM
THERAPEUTIC = 5.0  # nM


def make_result(**kwargs):
    defaults = dict(
        drug_name="TestDrug",
        tissue="liver",
        bmax_nmol_per_g=BMAX,
        kd_nM=KD,
        therapeutic_free_conc_nM=THERAPEUTIC,
    )
    defaults.update(kwargs)
    return simulate_tissue_binding(**defaults)


def test_returns_result():
    result = make_result()
    assert isinstance(result, TissueBindingResult)


def test_langmuir_bound_at_kd():
    # At C=Kd, bound = Bmax/2
    bound = langmuir_bound(KD, BMAX, KD)
    assert abs(bound - BMAX / 2) < 1e-9


def test_langmuir_bound_at_zero_is_zero():
    bound = langmuir_bound(0.0, BMAX, KD)
    assert bound == 0.0


def test_langmuir_bound_at_large_c_approaches_bmax():
    # At C = 1e6 * Kd, bound should be ~Bmax
    bound = langmuir_bound(1e6 * KD, BMAX, KD)
    assert abs(bound - BMAX) < BMAX * 0.001


def test_langmuir_bound_positive():
    bound = langmuir_bound(1.0, BMAX, KD)
    assert bound > 0.0


def test_fu_tissue_in_0_1():
    for c in [0.1, 1.0, 10.0, 100.0, 1000.0]:
        fu = fu_tissue(c, BMAX, KD)
        assert 0.0 <= fu <= 1.0


def test_fu_tissue_approaches_1_at_high_conc():
    # At very high concentration (saturated binding), fu -> 1 as free >> bound_equiv
    # bound is capped at Bmax=100 nmol/g = 100000 nM equiv; at C=1e9 nM, fu -> 1
    fu = fu_tissue(1e9, BMAX, KD)
    assert fu > 0.99


def test_fu_tissue_higher_at_high_conc_than_low():
    # fu increases as concentration increases (binding becomes saturated)
    fu_low = fu_tissue(0.01, BMAX, KD)
    fu_high = fu_tissue(10000.0, BMAX, KD)
    assert fu_high > fu_low


def test_occupancy_at_therapeutic_in_0_1():
    result = make_result()
    assert 0.0 <= result.occupancy_at_therapeutic <= 1.0


def test_occupancy_at_kd_is_half():
    # At therapeutic = Kd, occupancy = 0.5
    result = make_result(therapeutic_free_conc_nM=KD)
    assert abs(result.occupancy_at_therapeutic - 0.5) < 1e-9


def test_saturation_conc_equals_kd():
    result = make_result()
    assert result.saturation_conc_nM == KD


def test_linear_kp_positive():
    result = make_result()
    assert result.linear_kp > 0.0


def test_nonlinearity_flag_true_for_saturable():
    # Large Bmax, small Kd -> strong saturable binding -> nonlinearity
    result = make_result(bmax_nmol_per_g=10000.0, kd_nM=0.1)
    assert result.nonlinearity_flag is True


def test_nonlinearity_flag_false_for_linear():
    # Tiny Bmax, large Kd -> very little binding -> nearly linear, fu close to 1 everywhere
    result = make_result(bmax_nmol_per_g=0.001, kd_nM=1e6)
    assert result.nonlinearity_flag is False


def test_bound_fractions_length():
    conc_range = (0.1, 1.0, 10.0, 100.0, 1000.0)
    result = make_result(conc_range_nM=conc_range)
    assert len(result.bound_fractions) == len(conc_range)


def test_compare_tissues_returns_list():
    tissues = [
        {"name": "liver", "bmax": 100.0, "kd": 10.0},
        {"name": "kidney", "bmax": 50.0, "kd": 20.0},
        {"name": "brain", "bmax": 30.0, "kd": 5.0},
    ]
    results = compare_tissues("TestDrug", 5.0, tissues)
    assert isinstance(results, list)


def test_compare_tissues_length():
    tissues = [
        {"name": "liver", "bmax": 100.0, "kd": 10.0},
        {"name": "kidney", "bmax": 50.0, "kd": 20.0},
        {"name": "brain", "bmax": 30.0, "kd": 5.0},
    ]
    results = compare_tissues("TestDrug", 5.0, tissues)
    assert len(results) == 3


def test_compare_tissues_result_type():
    tissues = [
        {"name": "liver", "bmax": 100.0, "kd": 10.0},
        {"name": "kidney", "bmax": 50.0, "kd": 20.0},
    ]
    results = compare_tissues("TestDrug", 5.0, tissues)
    for r in results:
        assert isinstance(r, TissueBindingResult)


def test_invalid_bmax_raises():
    with pytest.raises(ValueError):
        simulate_tissue_binding(
            "Drug", "liver", bmax_nmol_per_g=-1.0, kd_nM=10.0, therapeutic_free_conc_nM=5.0
        )


def test_invalid_kd_raises():
    with pytest.raises(ValueError):
        simulate_tissue_binding(
            "Drug", "liver", bmax_nmol_per_g=100.0, kd_nM=0.0, therapeutic_free_conc_nM=5.0
        )


def test_notes_nonempty():
    result = make_result()
    assert len(result.notes) > 0
