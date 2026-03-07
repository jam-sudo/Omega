"""Tests for Phase 324 — Drug Deposition in Lung Airways."""

from __future__ import annotations

import pytest

from omega_pbpk.prediction.lung_airway_deposition import (
    AirwayDepositionResult,
    optimize_for_alveolar_delivery,
    predict_airway_deposition,
)


def _sim(**kw) -> AirwayDepositionResult:
    defaults = dict(
        drug_name="TestDrug",
        mmad_um=3.0,
        gsd=2.0,
        flow_rate_L_min=30.0,
        tidal_volume_mL=500.0,
        respiratory_rate_per_min=15.0,
        device_type="MDI",
    )
    defaults.update(kw)
    return predict_airway_deposition(**defaults)


def test_return_type():
    assert isinstance(_sim(), AirwayDepositionResult)


def test_drug_name_preserved():
    assert _sim(drug_name="Salbutamol").drug_name == "Salbutamol"


def test_fractions_between_0_and_1():
    r = _sim()
    assert 0.0 <= r.et_fraction <= 1.0
    assert 0.0 <= r.tb_fraction <= 1.0
    assert 0.0 <= r.alveolar_fraction <= 1.0


def test_total_deposited_plus_exhaled_approx_one():
    r = _sim()
    assert abs(r.total_deposited_fraction + r.exhaled_fraction - 1.0) < 1e-9


def test_lung_deposited_equals_tb_plus_alveolar():
    r = _sim()
    assert abs(r.lung_deposited_fraction - (r.tb_fraction + r.alveolar_fraction)) < 1e-9


def test_fpf_positive_for_small_mmad():
    assert _sim(mmad_um=1.0).fine_particle_fraction > 0.0


def test_fpf_zero_for_large_mmad():
    assert _sim(mmad_um=10.0).fine_particle_fraction == pytest.approx(0.0)


def test_nebulizer_lower_et_than_mdi():
    assert _sim(device_type="nebulizer").et_fraction < _sim(device_type="MDI").et_fraction


def test_soft_mist_lower_et_than_mdi():
    assert _sim(device_type="soft_mist").et_fraction < _sim(device_type="MDI").et_fraction


def test_large_particles_high_et_fraction():
    assert _sim(mmad_um=15.0).et_fraction > 0.5


def test_small_particles_higher_alveolar():
    assert _sim(mmad_um=1.0).alveolar_fraction >= _sim(mmad_um=10.0).alveolar_fraction


def test_higher_flow_increases_et():
    assert _sim(flow_rate_L_min=60.0).et_fraction >= _sim(flow_rate_L_min=15.0).et_fraction


def test_deposition_class_valid_string():
    assert _sim().deposition_efficiency_class in {"excellent", "good", "poor"}


def test_small_particles_nebulizer_good_or_excellent():
    r = _sim(mmad_um=1.5, device_type="nebulizer", flow_rate_L_min=15.0)
    assert r.deposition_efficiency_class in {"excellent", "good"}


def test_notes_nonempty():
    assert len(_sim().notes) > 0


def test_notes_contains_device():
    assert "DPI" in _sim(device_type="DPI").notes


def test_error_mmad_zero():
    with pytest.raises(ValueError, match="mmad_um"):
        _sim(mmad_um=0.0)


def test_error_gsd_below_one():
    with pytest.raises(ValueError, match="gsd"):
        _sim(gsd=0.5)


def test_error_flow_rate_zero():
    with pytest.raises(ValueError, match="flow_rate_L_min"):
        _sim(flow_rate_L_min=0.0)


def test_error_invalid_device():
    with pytest.raises(ValueError, match="device_type"):
        _sim(device_type="turbuhaler")


def test_optimize_returns_result():
    r = optimize_for_alveolar_delivery(
        "TestDrug",
        gsd=2.0,
        flow_rate_L_min=30.0,
        tidal_volume_mL=500.0,
        respiratory_rate_per_min=15.0,
        device_type="nebulizer",
    )
    assert isinstance(r, AirwayDepositionResult)


def test_optimize_alveolar_better_than_large_mmad():
    best = optimize_for_alveolar_delivery(
        "TestDrug",
        gsd=2.0,
        flow_rate_L_min=30.0,
        tidal_volume_mL=500.0,
        respiratory_rate_per_min=15.0,
        device_type="soft_mist",
    )
    r8 = predict_airway_deposition(
        "TestDrug",
        mmad_um=8.0,
        gsd=2.0,
        flow_rate_L_min=30.0,
        tidal_volume_mL=500.0,
        respiratory_rate_per_min=15.0,
        device_type="soft_mist",
    )
    assert best.alveolar_fraction >= r8.alveolar_fraction


def test_optimize_mmad_in_valid_range():
    r = optimize_for_alveolar_delivery(
        "TestDrug",
        gsd=2.0,
        flow_rate_L_min=30.0,
        tidal_volume_mL=500.0,
        respiratory_rate_per_min=15.0,
        device_type="DPI",
    )
    assert 0.5 <= r.mmad_um <= 10.0
