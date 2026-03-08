"""Tests for Phase 1068 — Drug Aggregation Risk Profiler."""

from __future__ import annotations

import pytest

from omega_pbpk.prediction.aggregation_risk_profiler import (
    AggregationRiskResult,
    predict_aggregation_risk,
)

_DEFAULTS = dict(
    molecule_name="TestMab",
    mw_Da=148000.0,
    pi=7.0,
    hydrophobicity_score=5.0,
    molecule_type="mAb",
    formulation_ph=6.0,
    ionic_strength_mM=150.0,
    protein_conc_mg_mL=10.0,
    has_disulfide=True,
    has_glycosylation=False,
    storage_temp_C=4.0,
)


def _run(**kwargs):
    params = {**_DEFAULTS, **kwargs}
    return predict_aggregation_risk(**params)


def test_return_type():
    assert isinstance(_run(), AggregationRiskResult)


def test_molecule_name_preserved():
    assert _run(molecule_name="MyMab").molecule_name == "MyMab"


def test_molecule_type_preserved():
    assert _run(molecule_type="peptide").molecule_type == "peptide"


def test_aggregation_rate_positive():
    assert _run().aggregation_rate_per_h > 0.0


def test_higher_temp_higher_rate():
    assert (
        _run(storage_temp_C=37.0).aggregation_rate_per_h
        > _run(storage_temp_C=4.0).aggregation_rate_per_h
    )


def test_near_pi_higher_rate():
    assert (
        _run(formulation_ph=7.0, pi=7.0).aggregation_rate_per_h
        > _run(formulation_ph=5.0, pi=7.0).aggregation_rate_per_h
    )


def test_glycosylation_reduces_rate():
    assert (
        _run(has_glycosylation=True).aggregation_rate_per_h
        < _run(has_glycosylation=False).aggregation_rate_per_h
    )


def test_disulfide_reduces_rate():
    assert (
        _run(has_disulfide=True).aggregation_rate_per_h
        < _run(has_disulfide=False).aggregation_rate_per_h
    )


def test_higher_hydrophobicity_higher_rate():
    assert (
        _run(hydrophobicity_score=9.0).aggregation_rate_per_h
        > _run(hydrophobicity_score=1.0).aggregation_rate_per_h
    )


def test_higher_conc_higher_rate():
    assert (
        _run(protein_conc_mg_mL=100.0).aggregation_rate_per_h
        > _run(protein_conc_mg_mL=1.0).aggregation_rate_per_h
    )


def test_monomer_at_1y_in_range():
    r = _run()
    assert 0.0 < r.monomer_fraction_at_1y <= 1.0


def test_monomer_at_2y_in_range():
    r = _run()
    assert 0.0 < r.monomer_fraction_at_2y <= 1.0


def test_monomer_at_2y_less_than_or_equal_1y():
    r = _run()
    assert r.monomer_fraction_at_2y <= r.monomer_fraction_at_1y


def test_risk_score_in_range():
    r = _run()
    assert 0.0 <= r.aggregation_risk_score <= 100.0


def test_stable_molecule_low_risk_score():
    r = _run(storage_temp_C=4.0, formulation_ph=5.0, pi=8.0, has_glycosylation=True)
    assert r.aggregation_risk_score < 60.0


def test_unstable_molecule_high_risk_score():
    r = _run(storage_temp_C=37.0, formulation_ph=7.0, pi=7.0, hydrophobicity_score=9.0)
    assert r.aggregation_risk_score > 25.0


def test_risk_category_valid():
    assert _run().risk_category in {"low", "moderate", "high", "critical"}


def test_mechanism_valid_values():
    assert _run().primary_aggregation_mechanism in {
        "hydrophobic",
        "electrostatic",
        "stress_induced",
        "chemical",
    }


def test_near_pi_gives_electrostatic_mechanism():
    r = _run(formulation_ph=7.0, pi=7.0, hydrophobicity_score=3.0, storage_temp_C=4.0)
    assert r.primary_aggregation_mechanism == "electrostatic"


def test_high_hydrophobicity_gives_hydrophobic_mechanism():
    r = _run(formulation_ph=5.0, pi=9.0, hydrophobicity_score=8.0, storage_temp_C=4.0)
    assert r.primary_aggregation_mechanism == "hydrophobic"


def test_recommended_ph_in_range():
    assert 5.0 <= _run().recommended_ph <= 8.5


def test_recommended_ph_low_pi():
    r = _run(pi=5.0)
    assert r.recommended_ph == pytest.approx(7.0, abs=0.01)


def test_recommended_ph_high_pi():
    r = _run(pi=9.0)
    assert r.recommended_ph == pytest.approx(7.0, abs=0.01)


def test_recommended_excipients_is_tuple():
    assert isinstance(_run().recommended_excipients, tuple)


def test_recommended_excipients_nonempty():
    assert len(_run().recommended_excipients) > 0


def test_notes_nonempty():
    assert len(_run().notes) > 0


@pytest.mark.parametrize("mtype", ["mAb", "protein", "peptide", "ADC", "fusion"])
def test_all_molecule_types_valid(mtype):
    r = _run(molecule_type=mtype)
    assert r.molecule_type == mtype
    assert r.aggregation_rate_per_h > 0.0


def test_invalid_mw_raises():
    with pytest.raises(ValueError, match="mw_Da"):
        _run(mw_Da=-1.0)


def test_invalid_pi_low_raises():
    with pytest.raises(ValueError, match="pi"):
        _run(pi=1.0)


def test_invalid_pi_high_raises():
    with pytest.raises(ValueError, match="pi"):
        _run(pi=12.0)


def test_invalid_hydrophobicity_raises():
    with pytest.raises(ValueError, match="hydrophobicity_score"):
        _run(hydrophobicity_score=11.0)


def test_invalid_conc_raises():
    with pytest.raises(ValueError, match="protein_conc_mg_mL"):
        _run(protein_conc_mg_mL=0.0)


def test_invalid_molecule_type_raises():
    with pytest.raises(ValueError, match="molecule_type"):
        _run(molecule_type="small_molecule")


def test_invalid_storage_temp_low_raises():
    with pytest.raises(ValueError, match="storage_temp_C"):
        _run(storage_temp_C=-30.0)


def test_invalid_storage_temp_high_raises():
    with pytest.raises(ValueError, match="storage_temp_C"):
        _run(storage_temp_C=70.0)
