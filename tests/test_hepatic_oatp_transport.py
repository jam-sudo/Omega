"""Tests for Phase 1107 — Hepatic OATP Transport Prediction."""

from __future__ import annotations

import pytest

from omega_pbpk.prediction.hepatic_oatp_transport import (
    HepaticOATPResult,
    predict_oatp_transport,
    screen_substrates,
)


def test_return_type():
    r = predict_oatp_transport("Rosuvastatin", substrate_type="statin")
    assert isinstance(r, HepaticOATPResult)


def test_victim_drug_preserved():
    r = predict_oatp_transport("Atorvastatin", substrate_type="statin")
    assert r.victim_drug == "Atorvastatin"


def test_no_perpetrator_aucr_is_one():
    r = predict_oatp_transport("Rosuvastatin", substrate_type="statin")
    assert r.aucr == pytest.approx(1.0)


def test_perpetrator_raises_aucr():
    r_no = predict_oatp_transport("Rosuvastatin", substrate_type="statin")
    r_inh = predict_oatp_transport(
        "Rosuvastatin",
        substrate_type="statin",
        perpetrator_drug="Cyclosporine",
        ki_oatp_uM=0.2,
        perpetrator_cmax_uM=1.0,
    )
    assert r_inh.aucr > r_no.aucr


def test_clhep_baseline_positive():
    r = predict_oatp_transport("Drug", substrate_type="other")
    assert r.clhep_baseline_L_per_h > 0.0


def test_clhep_inhibited_leq_baseline():
    r = predict_oatp_transport(
        "Drug",
        substrate_type="statin",
        perpetrator_drug="Rifampin",
        ki_oatp_uM=1.0,
        perpetrator_cmax_uM=5.0,
    )
    assert r.clhep_inhibited_L_per_h <= r.clhep_baseline_L_per_h


def test_active_uptake_fraction_in_range():
    r = predict_oatp_transport("Drug", substrate_type="statin")
    assert 0.0 <= r.active_uptake_fraction <= 1.0


def test_statin_has_higher_active_fraction_than_other():
    r_statin = predict_oatp_transport("Statin", substrate_type="statin")
    r_other = predict_oatp_transport("Other", substrate_type="other")
    assert r_statin.active_uptake_fraction > r_other.active_uptake_fraction


def test_ddi_severity_none_without_perpetrator():
    r = predict_oatp_transport("Drug", substrate_type="sartan")
    assert r.ddi_severity == "none"


def test_ddi_severity_severe_with_strong_inhibitor():
    r = predict_oatp_transport(
        "Drug",
        substrate_type="statin",
        perpetrator_drug="StrongInhibitor",
        ki_oatp_uM=0.01,
        perpetrator_cmax_uM=10.0,
        vmax_pmol_per_min_per_mg=5000.0,
        km_uM=0.5,
    )
    assert r.ddi_severity == "severe"


def test_regulatory_flag_false_no_perpetrator():
    r = predict_oatp_transport("Drug", substrate_type="statin")
    assert r.regulatory_flag is False


def test_regulatory_flag_true_moderate_ddi():
    r = predict_oatp_transport(
        "Drug",
        substrate_type="statin",
        perpetrator_drug="Cyclosporine",
        ki_oatp_uM=0.5,
        perpetrator_cmax_uM=2.0,
    )
    assert r.regulatory_flag is True


def test_notes_nonempty():
    r = predict_oatp_transport("Drug", substrate_type="statin")
    assert len(r.notes) > 0


def test_invalid_substrate_type_raises():
    with pytest.raises(ValueError, match="substrate_type"):
        predict_oatp_transport("Drug", substrate_type="hormone")


def test_invalid_ki_negative_raises():
    with pytest.raises(ValueError, match="ki_oatp_uM"):
        predict_oatp_transport("Drug", substrate_type="statin", ki_oatp_uM=-1.0)


def test_invalid_fu_zero_raises():
    with pytest.raises(ValueError, match="fu_plasma"):
        predict_oatp_transport("Drug", substrate_type="statin", fu_plasma=0.0)


def test_invalid_q_hepatic_raises():
    with pytest.raises(ValueError, match="q_hepatic_L_per_h"):
        predict_oatp_transport("Drug", substrate_type="statin", q_hepatic_L_per_h=0.0)


def test_custom_vmax_km():
    r = predict_oatp_transport(
        "Drug",
        substrate_type="statin",
        vmax_pmol_per_min_per_mg=1000.0,
        km_uM=2.0,
    )
    assert r.clhep_baseline_L_per_h > 0.0


def test_screen_substrates_sorted_descending():
    substrates = [
        {"victim_drug": "A", "substrate_type": "other"},
        {"victim_drug": "B", "substrate_type": "statin"},
        {"victim_drug": "C", "substrate_type": "sartan"},
    ]
    results = screen_substrates(substrates, "Cyclosporine", ki_oatp_uM=0.5,
                                 perpetrator_cmax_uM=2.0)
    aucs = [r.aucr for r in results]
    assert aucs == sorted(aucs, reverse=True)


def test_screen_substrates_returns_correct_count():
    substrates = [
        {"victim_drug": "A", "substrate_type": "statin"},
        {"victim_drug": "B", "substrate_type": "sartan"},
    ]
    results = screen_substrates(substrates, "Inhibitor", ki_oatp_uM=1.0,
                                 perpetrator_cmax_uM=1.0)
    assert len(results) == 2
