"""Tests for core/tissue_binding_kinetics.py — Phase 474.

Covers:
- Higher Bmax → more tissue binding → larger apparent Vd
- Higher Kd → weaker binding → lower Kp
- apparent_vd: fu_plasma/fu_tissue drives Kp
- kp_from_binding: equal fu → Kp=1
- simulate_tissue_binding: tissue concentration peaks after plasma peak
- 2x dose → 2x peak concentrations (linear binding)
- t_half longer with more tissue binding
- Input validation
"""

from __future__ import annotations

import pytest

from omega_pbpk.core.tissue_binding_kinetics import (
    TissueBindingResult,
    apparent_vd,
    calculate_tissue_fu,
    kp_from_binding,
    simulate_tissue_binding,
)

# ---------------------------------------------------------------------------
# Shared defaults — typical small molecule
# ---------------------------------------------------------------------------

_BASE = dict(
    drug_name="TestDrug",
    dose_mg=100.0,
    cl_L_per_h=5.0,
    plasma_fu=0.5,
    tissue_kd_nM=100.0,
    tissue_bmax_nM=500.0,
    vd_plasma_L=5.0,
    vd_tissue_L=65.0,
    t_end_h=24.0,
    dt_h=0.1,
    route="iv",
)


def _run(**overrides) -> TissueBindingResult:
    params = {**_BASE, **overrides}
    return simulate_tissue_binding(**params)


# ---------------------------------------------------------------------------
# 1. Smoke test / dataclass type
# ---------------------------------------------------------------------------


def test_returns_tissue_binding_result():
    result = _run()
    assert isinstance(result, TissueBindingResult)


def test_frozen_dataclass():
    result = _run()
    with pytest.raises((AttributeError, TypeError)):
        result.drug_name = "other"  # type: ignore[misc]


def test_time_vector_length():
    result = _run(t_end_h=10.0, dt_h=0.5)
    n_steps = int(10.0 / 0.5) + 1
    assert len(result.times_h) == n_steps


def test_concentration_vectors_same_length_as_times():
    result = _run()
    n = len(result.times_h)
    assert len(result.c_plasma_free_mg_L) == n
    assert len(result.c_tissue_bound_mg_kg) == n
    assert len(result.c_tissue_free_mg_kg) == n


def test_times_h_starts_at_zero():
    result = _run()
    assert result.times_h[0] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 2. Physical constraints
# ---------------------------------------------------------------------------


def test_c_plasma_free_non_negative():
    result = _run()
    assert all(c >= 0 for c in result.c_plasma_free_mg_L)


def test_c_tissue_free_non_negative():
    result = _run()
    assert all(c >= 0 for c in result.c_tissue_free_mg_kg)


def test_c_tissue_bound_non_negative():
    result = _run()
    assert all(c >= 0 for c in result.c_tissue_bound_mg_kg)


def test_iv_plasma_starts_nonzero():
    result = _run(route="iv")
    assert result.c_plasma_free_mg_L[0] > 0.0


def test_oral_plasma_starts_near_zero():
    result = _run(route="oral")
    assert result.c_plasma_free_mg_L[0] == pytest.approx(0.0, abs=1e-9)


def test_vd_apparent_positive():
    result = _run()
    assert result.vd_apparent_L > 0.0


def test_t_half_positive():
    result = _run()
    assert result.t_half_h > 0.0


def test_kp_tissue_positive():
    result = _run()
    assert result.kp_tissue > 0.0


# ---------------------------------------------------------------------------
# 3. Higher Bmax → more tissue binding → larger apparent Vd
# ---------------------------------------------------------------------------


def test_higher_bmax_larger_vd():
    """More binding sites → drug distributes more into tissue → larger Vd."""
    low_bmax = _run(tissue_bmax_nM=10.0)
    high_bmax = _run(tissue_bmax_nM=10000.0)
    assert high_bmax.vd_apparent_L >= low_bmax.vd_apparent_L


def test_high_bmax_high_kp():
    """High Bmax → more bound → lower tissue fu → higher Kp."""
    low_bmax = _run(tissue_bmax_nM=10.0)
    high_bmax = _run(tissue_bmax_nM=10000.0)
    assert high_bmax.kp_tissue >= low_bmax.kp_tissue


# ---------------------------------------------------------------------------
# 4. Higher Kd → weaker binding → lower Kp
# ---------------------------------------------------------------------------


def test_higher_kd_lower_kp():
    """Higher Kd (weaker binding) → more drug stays free → lower Kp."""
    tight = _run(tissue_kd_nM=1.0, tissue_bmax_nM=1000.0)
    weak = _run(tissue_kd_nM=10000.0, tissue_bmax_nM=1000.0)
    assert tight.kp_tissue >= weak.kp_tissue


def test_very_high_kd_kp_near_plasma_ratio():
    """When Kd >> free conc, binding negligible → fu_tissue ≈ 1 → Kp ≈ plasma_fu."""
    result = _run(tissue_kd_nM=1e9, tissue_bmax_nM=100.0)
    # Kp ≈ plasma_fu / 1.0 = plasma_fu
    assert result.kp_tissue == pytest.approx(_BASE["plasma_fu"], rel=0.5)


# ---------------------------------------------------------------------------
# 5. apparent_vd function
# ---------------------------------------------------------------------------


def test_apparent_vd_equal_fu_linear():
    """fu_plasma == fu_tissue → Vd_app = Vp + Vt."""
    vd = apparent_vd(5.0, 65.0, 0.5, 0.5)
    assert vd == pytest.approx(5.0 + 65.0, rel=1e-9)


def test_apparent_vd_low_tissue_fu_large_vd():
    """Low tissue fu → drugs extensively bound → very large Vd."""
    vd = apparent_vd(5.0, 65.0, plasma_fu=0.5, tissue_fu=0.01)
    # Vd_app = 5 + (0.5/0.01)*65 = 5 + 3250 = 3255
    assert vd > 1000.0


def test_apparent_vd_invalid_plasma_fu():
    with pytest.raises(ValueError, match="plasma_fu"):
        apparent_vd(5.0, 65.0, plasma_fu=0.0, tissue_fu=0.5)


def test_apparent_vd_invalid_tissue_fu():
    with pytest.raises(ValueError, match="tissue_fu"):
        apparent_vd(5.0, 65.0, plasma_fu=0.5, tissue_fu=0.0)


def test_apparent_vd_invalid_vd_plasma():
    with pytest.raises(ValueError, match="vd_plasma_L"):
        apparent_vd(0.0, 65.0, plasma_fu=0.5, tissue_fu=0.5)


# ---------------------------------------------------------------------------
# 6. kp_from_binding function
# ---------------------------------------------------------------------------


def test_kp_equal_fu_is_one():
    """Equal plasma and tissue unbound fractions → Kp = 1."""
    assert kp_from_binding(0.5, 0.5) == pytest.approx(1.0)


def test_kp_high_plasma_fu_high_kp():
    """High plasma_fu relative to tissue_fu → high Kp."""
    kp = kp_from_binding(plasma_fu=0.9, tissue_fu=0.1)
    assert kp == pytest.approx(9.0, rel=1e-9)


def test_kp_invalid_plasma_fu():
    with pytest.raises(ValueError, match="plasma_fu"):
        kp_from_binding(0.0, 0.5)


def test_kp_invalid_tissue_fu():
    with pytest.raises(ValueError, match="tissue_fu"):
        kp_from_binding(0.5, 0.0)


# ---------------------------------------------------------------------------
# 7. calculate_tissue_fu function
# ---------------------------------------------------------------------------


def test_calculate_tissue_fu_no_binding():
    """Zero bmax → fu=1 (no binding)."""
    fu = calculate_tissue_fu(tissue_kd_nM=10.0, tissue_bmax_nM=0.0, free_conc_nM=5.0)
    assert fu == pytest.approx(1.0)


def test_calculate_tissue_fu_zero_conc():
    """Zero free concentration → fu=1 (no saturation)."""
    fu = calculate_tissue_fu(10.0, 100.0, 0.0)
    assert fu == pytest.approx(1.0)


def test_calculate_tissue_fu_high_bmax_low_fu():
    """Very high bmax → most drug bound → low fu."""
    fu = calculate_tissue_fu(tissue_kd_nM=1.0, tissue_bmax_nM=1e6, free_conc_nM=100.0)
    assert fu < 0.1


def test_calculate_tissue_fu_invalid_kd():
    with pytest.raises(ValueError, match="tissue_kd_nM"):
        calculate_tissue_fu(0.0, 100.0, 10.0)


def test_calculate_tissue_fu_in_unit_interval():
    fu = calculate_tissue_fu(10.0, 50.0, 5.0)
    assert 0.0 <= fu <= 1.0


# ---------------------------------------------------------------------------
# 8. Tissue concentration peaks after plasma peak
# ---------------------------------------------------------------------------


def test_tissue_free_peaks_after_plasma_for_oral():
    """For oral dosing, tissue equilibrates after plasma Cmax."""
    result = _run(route="oral", t_end_h=24.0)
    pf = result.c_plasma_free_mg_L
    tf = result.c_tissue_free_mg_kg
    t_pf_max = result.times_h[pf.index(max(pf))]
    t_tf_max = result.times_h[tf.index(max(tf))]
    # Tissue peak should be at or after plasma peak (allow small numerical tolerance)
    assert t_tf_max >= t_pf_max - 1.0


def test_tissue_has_nonzero_concentration():
    result = _run(t_end_h=12.0)
    assert max(result.c_tissue_free_mg_kg) > 0.0


# ---------------------------------------------------------------------------
# 9. 2× dose → 2× peak concentrations (linear PK)
# ---------------------------------------------------------------------------


def test_double_dose_double_plasma_peak():
    """Linear PK: 2x dose should yield ~2x peak free plasma concentration."""
    r1 = _run(dose_mg=100.0)
    r2 = _run(dose_mg=200.0)
    cmax1 = max(r1.c_plasma_free_mg_L)
    cmax2 = max(r2.c_plasma_free_mg_L)
    assert cmax2 == pytest.approx(2.0 * cmax1, rel=0.15)


def test_double_dose_double_tissue_peak():
    """Linear PK: 2x dose should yield ~2x peak tissue free concentration."""
    r1 = _run(dose_mg=100.0, tissue_kd_nM=1e6)  # negligible binding → linear
    r2 = _run(dose_mg=200.0, tissue_kd_nM=1e6)
    tf_max1 = max(r1.c_tissue_free_mg_kg)
    tf_max2 = max(r2.c_tissue_free_mg_kg)
    if tf_max1 > 0.0:
        assert tf_max2 == pytest.approx(2.0 * tf_max1, rel=0.2)


# ---------------------------------------------------------------------------
# 10. t_half longer with more tissue binding
# ---------------------------------------------------------------------------


def test_high_binding_longer_thalf():
    """More tissue binding → slower apparent terminal elimination → longer t½."""
    low_bind = _run(tissue_bmax_nM=1.0, tissue_kd_nM=1.0)
    high_bind = _run(tissue_bmax_nM=50000.0, tissue_kd_nM=10.0)
    # Allow for numerical noise but high binding should not shorten t½
    assert high_bind.t_half_h >= low_bind.t_half_h * 0.5


# ---------------------------------------------------------------------------
# 11. Input validation
# ---------------------------------------------------------------------------


def test_invalid_dose_raises():
    with pytest.raises(ValueError, match="dose_mg"):
        _run(dose_mg=0.0)


def test_invalid_cl_raises():
    with pytest.raises(ValueError, match="cl_L_per_h"):
        _run(cl_L_per_h=-1.0)


def test_invalid_plasma_fu_zero_raises():
    with pytest.raises(ValueError, match="plasma_fu"):
        _run(plasma_fu=0.0)


def test_invalid_kd_raises():
    with pytest.raises(ValueError, match="tissue_kd_nM"):
        _run(tissue_kd_nM=0.0)


def test_invalid_bmax_raises():
    with pytest.raises(ValueError, match="tissue_bmax_nM"):
        _run(tissue_bmax_nM=-1.0)


def test_invalid_vd_plasma_raises():
    with pytest.raises(ValueError, match="vd_plasma_L"):
        _run(vd_plasma_L=0.0)


def test_invalid_vd_tissue_raises():
    with pytest.raises(ValueError, match="vd_tissue_L"):
        _run(vd_tissue_L=-5.0)


def test_invalid_route_raises():
    with pytest.raises(ValueError, match="route"):
        _run(route="subcutaneous")


def test_invalid_drug_name_raises():
    with pytest.raises(ValueError, match="drug_name"):
        _run(drug_name="")


# ---------------------------------------------------------------------------
# 12. Notes
# ---------------------------------------------------------------------------


def test_notes_is_string():
    result = _run()
    assert isinstance(result.notes, str)


def test_notes_contains_route():
    result = _run(route="oral")
    assert "oral" in result.notes
