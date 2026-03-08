"""Tests for Phase 1054 — Drug Synaptic Cleft PK."""

import pytest

from omega_pbpk.prediction.synaptic_cleft_pk import (
    SynapticCleftResult,
    predict_synaptic_cleft_pk,
)


def _default(**kwargs):
    params = dict(
        drug_name="TestDrug",
        plasma_conc_mg_L=1.0,
        mw_Da=300.0,
        logp=2.0,
        fu_plasma=0.3,
        fu_brain=0.1,
        target_neurotransmitter="dopamine",
        ki_nM=100.0,
        plasma_to_brain_ratio=0.0,
        mechanism="reuptake_inhibition",
        therapeutic_conc_nM=10.0,
        toxic_conc_nM=1000.0,
    )
    params.update(kwargs)
    return predict_synaptic_cleft_pk(**params)


# --- Return type ---


def test_returns_synaptic_cleft_result():
    result = _default()
    assert isinstance(result, SynapticCleftResult)


def test_result_is_frozen():
    result = _default()
    with pytest.raises((AttributeError, TypeError)):
        result.c_synaptic_cleft_nM = 999.0  # type: ignore[misc]


# --- Range checks ---


def test_c_synaptic_positive():
    result = _default()
    assert result.c_synaptic_cleft_nM > 0


def test_receptor_occupancy_in_range():
    result = _default()
    assert 0.0 <= result.receptor_occupancy_pct <= 100.0


def test_synaptic_half_life_positive():
    result = _default()
    assert result.synaptic_half_life_ms > 0


def test_brain_to_synapse_ratio_positive():
    result = _default()
    assert result.brain_to_synapse_ratio > 0


def test_neurotransmitter_effect_valid():
    valid = {"inhibition", "enhancement", "reuptake_block", "none"}
    for mechanism in ("reuptake_inhibition", "receptor_antagonism", "receptor_agonism"):
        result = _default(mechanism=mechanism)
        assert result.neurotransmitter_effect in valid


def test_therapeutic_window_valid():
    valid = {"sub-therapeutic", "therapeutic", "supratherapeutic"}
    result = _default()
    assert result.therapeutic_window_synapse in valid


def test_notes_nonempty():
    result = _default()
    assert len(result.notes) > 0


# --- Comparative / directional ---


def test_higher_plasma_conc_higher_occupancy():
    low = _default(plasma_conc_mg_L=0.1)
    high = _default(plasma_conc_mg_L=10.0)
    assert high.receptor_occupancy_pct > low.receptor_occupancy_pct


def test_low_ki_higher_occupancy():
    high_ki = _default(ki_nM=1000.0)
    low_ki = _default(ki_nM=1.0)
    assert low_ki.receptor_occupancy_pct > high_ki.receptor_occupancy_pct


def test_high_logp_higher_c_synaptic():
    low_logp = _default(logp=0.0)
    high_logp = _default(logp=4.0)
    assert high_logp.c_synaptic_cleft_nM > low_logp.c_synaptic_cleft_nM


def test_reuptake_inhibition_high_occupancy_gives_reuptake_block():
    # Force high occupancy: very low ki_nM and high plasma conc
    result = _default(ki_nM=0.1, plasma_conc_mg_L=100.0, mechanism="reuptake_inhibition")
    assert result.neurotransmitter_effect == "reuptake_block"


def test_antagonism_high_occupancy_gives_inhibition():
    result = _default(ki_nM=0.1, plasma_conc_mg_L=100.0, mechanism="receptor_antagonism")
    assert result.neurotransmitter_effect == "inhibition"


def test_agonism_high_occupancy_gives_enhancement():
    result = _default(ki_nM=0.1, plasma_conc_mg_L=100.0, mechanism="receptor_agonism")
    assert result.neurotransmitter_effect == "enhancement"


def test_low_occupancy_gives_none_effect():
    # Very high ki_nM and low plasma conc → RO << 0.2
    result = _default(ki_nM=1e9, plasma_conc_mg_L=0.001, mechanism="reuptake_inhibition")
    assert result.neurotransmitter_effect == "none"


def test_brain_to_synapse_ratio_equals_inv_fu_brain():
    result = _default(fu_brain=0.2)
    assert abs(result.brain_to_synapse_ratio - (1.0 / 0.2)) < 1e-6


def test_sub_therapeutic_window():
    result = _default(plasma_conc_mg_L=0.0001, therapeutic_conc_nM=10000.0)
    assert result.therapeutic_window_synapse == "sub-therapeutic"


def test_supratherapeutic_window():
    result = _default(
        plasma_conc_mg_L=1000.0,
        ki_nM=0.1,
        therapeutic_conc_nM=0.001,
        toxic_conc_nM=0.01,
    )
    assert result.therapeutic_window_synapse == "supratherapeutic"


def test_plasma_to_brain_ratio_used_when_provided():
    r1 = _default(plasma_to_brain_ratio=0.5)
    r2 = _default(plasma_to_brain_ratio=1.5)
    assert r2.c_synaptic_cleft_nM > r1.c_synaptic_cleft_nM


# --- Validation ---


def test_raises_plasma_conc_zero():
    with pytest.raises(ValueError, match="plasma_conc_mg_L"):
        _default(plasma_conc_mg_L=0.0)


def test_raises_mw_zero():
    with pytest.raises(ValueError, match="mw_Da"):
        _default(mw_Da=0.0)


def test_raises_invalid_neurotransmitter():
    with pytest.raises(ValueError, match="target_neurotransmitter"):
        _default(target_neurotransmitter="acetylcholine")


def test_raises_ki_zero():
    with pytest.raises(ValueError, match="ki_nM"):
        _default(ki_nM=0.0)


def test_raises_invalid_mechanism():
    with pytest.raises(ValueError, match="mechanism"):
        _default(mechanism="allosteric_modulation")
