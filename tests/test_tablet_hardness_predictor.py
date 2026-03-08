"""Tests for Phase 231: Tablet Hardness Predictor."""

from __future__ import annotations

import pytest

from omega_pbpk.prediction.tablet_hardness_predictor import (
    TabletHardnessResult,
    optimize_compression_force,
    predict_tablet_hardness,
)


def _pred(force=10.0, excipient="MCC", loading=10.0, **kw) -> TabletHardnessResult:
    return predict_tablet_hardness(
        drug_name="TestDrug",
        compression_force_kN=force,
        excipient_type=excipient,
        drug_loading_pct=loading,
        **kw,
    )


def test_return_type():
    assert isinstance(_pred(), TabletHardnessResult)


def test_hardness_positive():
    assert _pred().hardness_N > 0.0


def test_hardness_increases_with_force():
    assert _pred(force=20.0).hardness_N > _pred(force=5.0).hardness_N


def test_hardness_decreases_with_drug_loading():
    assert _pred(loading=80.0).hardness_N < _pred(loading=10.0).hardness_N


def test_hardness_bounded_by_excipient_max():
    r = _pred(force=1000.0, excipient="MCC", loading=0.0, particle_size_um=1.0)
    assert r.hardness_N < 220.0


def test_friability_non_negative():
    assert _pred().friability_pct >= 0.0


def test_friability_max_5():
    assert _pred(force=0.01).friability_pct <= 5.0


def test_friability_decreases_with_force():
    assert _pred(force=40.0).friability_pct <= _pred(force=2.0).friability_pct


def test_quality_status_values():
    assert _pred(force=30.0).quality_status in {"acceptable", "failing"}


def test_high_force_gives_acceptable():
    r = _pred(force=50.0, excipient="MCC", loading=5.0, particle_size_um=50.0)
    assert r.quality_status == "acceptable"


def test_low_force_gives_failing():
    assert _pred(force=0.1).quality_status == "failing"


def test_MCC_harder_than_starch_same_force():
    assert (
        _pred(excipient="MCC", force=10.0).hardness_N
        > _pred(excipient="starch", force=10.0).hardness_N
    )


def test_smaller_particles_harder():
    assert _pred(particle_size_um=10.0).hardness_N > _pred(particle_size_um=500.0).hardness_N


def test_default_notes_nonempty():
    assert len(_pred().notes) > 0


def test_custom_notes_preserved():
    r = predict_tablet_hardness("D", 10.0, "MCC", 10.0, notes="custom note")
    assert "custom note" in r.notes


def test_optimize_returns_dict():
    result = optimize_compression_force("D", "MCC", 10.0, target_hardness_N=60.0)
    assert isinstance(result, dict)


def test_optimize_keys_present():
    result = optimize_compression_force("D", "MCC", 10.0, target_hardness_N=60.0)
    for key in (
        "min_force_kN",
        "expected_hardness_N",
        "expected_friability_pct",
        "quality_status",
        "achievable",
    ):
        assert key in result


def test_optimize_achievable_target():
    result = optimize_compression_force("D", "MCC", 10.0, target_hardness_N=50.0)
    assert result["achievable"] is True
    assert result["min_force_kN"] > 0.0


def test_optimize_unachievable_target():
    result = optimize_compression_force("D", "MCC", 10.0, target_hardness_N=500.0)
    assert result["achievable"] is False


def test_error_force_zero():
    with pytest.raises(ValueError, match="compression_force_kN"):
        predict_tablet_hardness("D", 0.0, "MCC", 10.0)


def test_error_force_negative():
    with pytest.raises(ValueError, match="compression_force_kN"):
        predict_tablet_hardness("D", -1.0, "MCC", 10.0)


def test_error_loading_above_100():
    with pytest.raises(ValueError, match="drug_loading_pct"):
        predict_tablet_hardness("D", 10.0, "MCC", 110.0)


def test_error_invalid_excipient():
    with pytest.raises(ValueError, match="excipient_type"):
        predict_tablet_hardness("D", 10.0, "unknown_excipient", 10.0)
