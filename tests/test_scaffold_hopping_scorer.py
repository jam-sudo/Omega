"""Tests for Phase 806 scaffold hopping scorer."""

from __future__ import annotations

import pytest

from omega_pbpk.prediction.scaffold_hopping_scorer import (
    ScaffoldHoppingResult,
    score_scaffold_hop,
    screen_analogs,
)

# Default values for a reference compound (ibuprofen-like)
_ORIG = dict(
    orig_mw=206.0,
    orig_logp=3.5,
    orig_psa=37.3,
    orig_hbd=1,
    orig_hba=2,
)
_SAME = dict(
    analog_mw=206.0,
    analog_logp=3.5,
    analog_psa=37.3,
    analog_hbd=1,
    analog_hba=2,
)


def _run(**kwargs):
    defaults = dict(
        original_compound="Ref",
        analog_compound="Analog",
        **_ORIG,
        **_SAME,
        bioisostere_type="none",
        known_active_scaffold=False,
    )
    defaults.update(kwargs)
    return score_scaffold_hop(**defaults)


class TestScaffoldHoppingResultType:
    def test_returns_correct_type(self):
        result = _run()
        assert isinstance(result, ScaffoldHoppingResult)

    def test_is_frozen(self):
        result = _run()
        with pytest.raises(Exception):
            result.mw_delta = 999.0  # type: ignore[misc]

    def test_has_all_fields(self):
        result = _run()
        assert hasattr(result, "original_compound")
        assert hasattr(result, "analog_compound")
        assert hasattr(result, "mw_delta")
        assert hasattr(result, "logp_delta")
        assert hasattr(result, "psa_delta")
        assert hasattr(result, "hbd_delta")
        assert hasattr(result, "hba_delta")
        assert hasattr(result, "tanimoto_similarity")
        assert hasattr(result, "property_similarity")
        assert hasattr(result, "bioisostere_score")
        assert hasattr(result, "scaffold_hop_feasibility")
        assert hasattr(result, "predicted_activity_retention_pct")
        assert hasattr(result, "adme_improvement")
        assert hasattr(result, "notes")


class TestScaffoldHoppingScoring:
    def test_identical_compound_high_activity_retention(self):
        result = _run()
        assert result.predicted_activity_retention_pct > 40.0

    def test_identical_compound_high_property_similarity(self):
        result = _run()
        assert result.property_similarity > 0.9

    def test_large_mw_change_low_property_similarity(self):
        result = _run(analog_mw=600.0)
        assert result.property_similarity < 0.5

    def test_mw_delta_correct(self):
        result = _run(analog_mw=300.0)
        assert result.mw_delta == pytest.approx(300.0 - 206.0)

    def test_logp_delta_correct(self):
        result = _run(analog_logp=1.0)
        assert result.logp_delta == pytest.approx(1.0 - 3.5)

    def test_psa_delta_correct(self):
        result = _run(analog_psa=70.0)
        assert result.psa_delta == pytest.approx(70.0 - 37.3, rel=0.01)

    def test_ring_replacement_higher_score_than_none(self):
        ring = _run(bioisostere_type="ring_replacement")
        none = _run(bioisostere_type="none")
        assert ring.predicted_activity_retention_pct > none.predicted_activity_retention_pct

    def test_known_active_scaffold_boosts_score(self):
        active = _run(known_active_scaffold=True)
        inactive = _run(known_active_scaffold=False)
        assert active.predicted_activity_retention_pct > inactive.predicted_activity_retention_pct

    def test_tanimoto_between_0_and_1(self):
        result = _run()
        assert 0.0 <= result.tanimoto_similarity <= 1.0

    def test_property_similarity_between_0_and_1(self):
        result = _run()
        assert 0.0 <= result.property_similarity <= 1.0

    def test_activity_retention_between_0_and_100(self):
        result = _run(analog_mw=900.0, analog_logp=8.0)
        assert 0.0 <= result.predicted_activity_retention_pct <= 100.0

    def test_feasibility_high_for_identical(self):
        result = _run(known_active_scaffold=True, bioisostere_type="ring_replacement")
        assert result.scaffold_hop_feasibility == "high"

    def test_feasibility_in_expected_set(self):
        result = _run()
        assert result.scaffold_hop_feasibility in {"high", "moderate", "low"}

    def test_adme_improvement_worsened_when_logp_increases(self):
        result = _run(analog_logp=6.0, analog_psa=37.3)
        assert result.adme_improvement == "worsened"

    def test_adme_improvement_improved_when_psa_increases(self):
        result = _run(analog_psa=80.0, analog_logp=3.5)
        assert result.adme_improvement == "improved"

    def test_adme_improvement_improved_when_logp_decreases(self):
        result = _run(analog_logp=1.0, analog_psa=37.3)
        assert result.adme_improvement == "improved"

    def test_notes_is_string(self):
        result = _run()
        assert isinstance(result.notes, str)


class TestValidation:
    def test_zero_orig_mw_raises(self):
        with pytest.raises(ValueError, match="orig_mw"):
            score_scaffold_hop(
                original_compound="Ref",
                analog_compound="Analog",
                orig_mw=0.0,
                orig_logp=2.0,
                orig_psa=50.0,
                orig_hbd=1,
                orig_hba=2,
                analog_mw=200.0,
                analog_logp=2.0,
                analog_psa=50.0,
                analog_hbd=1,
                analog_hba=2,
            )

    def test_negative_orig_mw_raises(self):
        with pytest.raises(ValueError, match="orig_mw"):
            score_scaffold_hop(
                original_compound="Ref",
                analog_compound="Analog",
                orig_mw=-100.0,
                orig_logp=2.0,
                orig_psa=50.0,
                orig_hbd=1,
                orig_hba=2,
                analog_mw=200.0,
                analog_logp=2.0,
                analog_psa=50.0,
                analog_hbd=1,
                analog_hba=2,
            )


class TestScreenAnalogs:
    def test_returns_list(self):
        orig_props = {"mw": 200.0, "logp": 2.0, "psa": 50.0, "hbd": 1, "hba": 2}
        analogs = [
            {"name": "A1", "mw": 210.0, "logp": 2.0, "psa": 50.0, "hbd": 1, "hba": 2},
            {"name": "A2", "mw": 300.0, "logp": 4.0, "psa": 30.0, "hbd": 0, "hba": 1},
        ]
        results = screen_analogs("Original", orig_props, analogs)
        assert isinstance(results, list)
        assert len(results) == 2

    def test_sorted_descending(self):
        orig_props = {"mw": 200.0, "logp": 2.0, "psa": 50.0, "hbd": 1, "hba": 2}
        analogs = [
            {"name": "Far", "mw": 600.0, "logp": 7.0, "psa": 10.0, "hbd": 0, "hba": 0},
            {
                "name": "Close",
                "mw": 205.0,
                "logp": 2.1,
                "psa": 52.0,
                "hbd": 1,
                "hba": 2,
                "bioisostere_type": "group_replacement",
            },
        ]
        results = screen_analogs("Original", orig_props, analogs)
        scores = [r.predicted_activity_retention_pct for r in results]
        assert scores == sorted(scores, reverse=True)
