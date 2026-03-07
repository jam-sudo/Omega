"""Tests for Phase 336 — Protein Binding Saturation Model."""

import pytest

from omega_pbpk.prediction.protein_binding_saturation import (
    ProteinBindingSaturationResult,
    calculate_protein_binding_saturation,
)

# --- Helpers ---


def make_basic(**kwargs):
    defaults = dict(
        drug_name="TestDrug",
        protein_conc_uM=600.0,  # albumin
        ka_per_uM=0.01,  # moderate affinity (Kd = 100 uM)
        n_binding_sites=1,
        total_drug_conc_mg_L=10.0,  # 10 mg/L
        mw_Da=400.0,  # MW 400 Da → ~25 uM
    )
    defaults.update(kwargs)
    return calculate_protein_binding_saturation(**defaults)


# --- Type checks ---


def test_returns_correct_type():
    result = make_basic()
    assert isinstance(result, ProteinBindingSaturationResult)


def test_result_is_plain_dataclass_not_frozen():
    result = make_basic()
    # Should NOT raise — plain dataclass is mutable
    result.drug_name = "Modified"
    assert result.drug_name == "Modified"


# --- fu validity ---


def test_fu_in_valid_range():
    result = make_basic()
    assert 0.0 < result.fu_at_therapeutic <= 1.0


def test_fu_profile_all_in_valid_range():
    result = make_basic()
    for fu in result.fu_profile:
        assert 0.0 < fu <= 1.0


# --- Saturation index ---


def test_saturation_index_in_range():
    result = make_basic()
    assert 0.0 <= result.saturation_index <= 1.0


def test_high_ka_gives_high_saturation():
    # High affinity (ka=10, Kd=0.1 uM) + very high drug conc → high saturation
    # total_drug_conc_uM = (1000/400)*1000 = 2500 uM >> total_sites = 600 uM
    result = make_basic(ka_per_uM=10.0, total_drug_conc_mg_L=1000.0)
    assert result.saturation_index > 0.5


def test_low_ka_gives_low_saturation():
    # Very low affinity (ka=0.0001, Kd=10000 uM) + low drug → low saturation
    result = make_basic(ka_per_uM=0.0001, total_drug_conc_mg_L=1.0)
    assert result.saturation_index < 0.5


# --- Concentration profile ---


def test_conc_range_length_is_20():
    result = make_basic()
    assert len(result.conc_range_uM) == 20


def test_fu_profile_length_is_20():
    result = make_basic()
    assert len(result.fu_profile) == 20


def test_conc_range_is_increasing():
    result = make_basic()
    for i in range(1, len(result.conc_range_uM)):
        assert result.conc_range_uM[i] > result.conc_range_uM[i - 1]


def test_higher_conc_lower_fu():
    """At higher concentrations, protein binding saturates, so fu increases."""
    result = make_basic(ka_per_uM=1.0, total_drug_conc_mg_L=50.0)
    # fu at first (lowest conc) should be <= fu at last (highest conc) when saturating
    # i.e., higher drug conc → more saturation → higher fu
    assert result.fu_profile[-1] >= result.fu_profile[0]


# --- Bound and free drug ---


def test_bound_plus_free_equals_total():
    result = make_basic()
    total_reconstructed = result.bound_drug_uM + result.free_drug_uM
    assert abs(total_reconstructed - result.total_drug_conc_uM) < 1e-6


def test_free_equals_fu_times_total():
    result = make_basic()
    expected_free = result.fu_at_therapeutic * result.total_drug_conc_uM
    assert abs(result.free_drug_uM - expected_free) < 1e-6


# --- Concentration conversion ---


def test_concentration_conversion_correct():
    # total_drug_conc_uM = (mg/L / Da) * 1000
    result = make_basic(total_drug_conc_mg_L=10.0, mw_Da=500.0)
    expected_uM = (10.0 / 500.0) * 1000.0  # = 20.0 uM
    assert abs(result.total_drug_conc_uM - expected_uM) < 1e-9


# --- 50% saturation concentration ---


def test_concentration_at_50pct_saturation_positive():
    result = make_basic()
    assert result.concentration_at_50pct_saturation_uM > 0.0


def test_50pct_saturation_formula():
    # C_50 = Kd + n*P/2
    protein_conc = 600.0
    ka = 0.01
    n = 1
    kd = 1.0 / ka  # = 100 uM
    expected = kd + n * protein_conc / 2.0  # = 400 uM
    result = make_basic(ka_per_uM=ka, protein_conc_uM=protein_conc, n_binding_sites=n)
    assert abs(result.concentration_at_50pct_saturation_uM - expected) < 1e-6


# --- Nonlinearity ---


def test_is_nonlinear_high_affinity_high_conc():
    # High affinity + drug conc near Kd → nonlinear
    result = make_basic(ka_per_uM=5.0, total_drug_conc_mg_L=500.0)
    assert result.is_nonlinear is True


def test_nonlinearity_index_consistent_with_is_nonlinear():
    result = make_basic()
    if result.binding_nonlinearity_index > 1.2:
        assert result.is_nonlinear is True
    else:
        assert result.is_nonlinear is False


# --- Validation errors ---


def test_raises_on_zero_protein_conc():
    with pytest.raises(ValueError, match="protein_conc_uM"):
        make_basic(protein_conc_uM=0.0)


def test_raises_on_negative_protein_conc():
    with pytest.raises(ValueError, match="protein_conc_uM"):
        make_basic(protein_conc_uM=-1.0)


def test_raises_on_zero_ka():
    with pytest.raises(ValueError, match="ka_per_uM"):
        make_basic(ka_per_uM=0.0)


def test_raises_on_negative_ka():
    with pytest.raises(ValueError, match="ka_per_uM"):
        make_basic(ka_per_uM=-0.5)


def test_raises_on_zero_mw():
    with pytest.raises(ValueError, match="mw_Da"):
        make_basic(mw_Da=0.0)


def test_raises_on_negative_mw():
    with pytest.raises(ValueError, match="mw_Da"):
        make_basic(mw_Da=-100.0)


def test_raises_on_zero_binding_sites():
    with pytest.raises(ValueError, match="n_binding_sites"):
        make_basic(n_binding_sites=0)


def test_raises_on_zero_drug_conc():
    with pytest.raises(ValueError, match="total_drug_conc_mg_L"):
        make_basic(total_drug_conc_mg_L=0.0)


def test_raises_on_negative_drug_conc():
    with pytest.raises(ValueError, match="total_drug_conc_mg_L"):
        make_basic(total_drug_conc_mg_L=-5.0)


# --- Notes non-empty ---


def test_notes_non_empty():
    result = make_basic()
    assert isinstance(result.notes, str)
    assert len(result.notes) > 0
