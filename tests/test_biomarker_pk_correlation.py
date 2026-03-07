"""
Tests for Phase 587 — clinical/biomarker_pk_correlation.py
"""

import math

import pytest

from omega_pbpk.clinical.biomarker_pk_correlation import (
    hysteresis_detection,
    indirect_response_model,
    pk_pd_link_ke0,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_decay(n=50, peak=10.0, decay=0.3):
    """Mono-exponential decay plasma concentration profile."""
    times = [i * 0.5 for i in range(n)]
    c = [peak * math.exp(-decay * t) for t in times]
    return times, c


def _make_flat(n=50, val=5.0):
    times = [i * 0.5 for i in range(n)]
    c = [val] * n
    return times, c


# ---------------------------------------------------------------------------
# indirect_response_model — inhibition_kin
# ---------------------------------------------------------------------------


class TestIndirectResponseInhibitionKin:
    def test_returns_correct_keys(self):
        times, c = _make_decay()
        result = indirect_response_model(times, c, kin=1.0, kout=0.5, imax=0.8, ic50=5.0)
        assert "times_h" in result
        assert "response" in result
        assert "r_baseline" in result
        assert "r_min" in result
        assert "r_max" in result
        assert "auc_response" in result
        assert "notes" in result

    def test_baseline_equals_kin_over_kout(self):
        times, c = _make_decay()
        result = indirect_response_model(times, c, kin=2.0, kout=0.5, imax=0.8, ic50=5.0)
        assert abs(result["r_baseline"] - 4.0) < 1e-9

    def test_custom_baseline(self):
        times, c = _make_decay()
        result = indirect_response_model(
            times, c, kin=2.0, kout=0.5, imax=0.8, ic50=5.0, baseline=10.0
        )
        assert abs(result["r_baseline"] - 10.0) < 1e-9

    def test_response_decreases_with_inhibition_kin(self):
        """Drug inhibits production: biomarker should decrease below baseline."""
        times, c = _make_flat(n=80, val=20.0)
        result = indirect_response_model(times, c, kin=1.0, kout=0.5, imax=0.9, ic50=5.0)
        assert result["r_min"] < result["r_baseline"]

    def test_response_length_matches_times(self):
        times, c = _make_decay()
        result = indirect_response_model(times, c, kin=1.0, kout=0.5, imax=0.8, ic50=5.0)
        assert len(result["response"]) == len(times)

    def test_auc_response_positive(self):
        times, c = _make_decay()
        result = indirect_response_model(times, c, kin=1.0, kout=0.5, imax=0.8, ic50=5.0)
        assert result["auc_response"] > 0

    def test_invalid_model_type_raises(self):
        times, c = _make_decay()
        with pytest.raises(ValueError, match="Invalid model_type"):
            indirect_response_model(
                times, c, kin=1.0, kout=0.5, imax=0.8, ic50=5.0, model_type="bad_type"
            )

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            indirect_response_model([0, 1, 2], [1.0, 2.0], kin=1.0, kout=0.5, imax=0.8, ic50=5.0)


# ---------------------------------------------------------------------------
# indirect_response_model — stimulation_kin
# ---------------------------------------------------------------------------


class TestIndirectResponseStimulationKin:
    def test_response_increases_with_stimulation_kin(self):
        """Drug stimulates production: biomarker should rise above baseline."""
        times, c = _make_flat(n=80, val=20.0)
        result = indirect_response_model(
            times, c, kin=1.0, kout=0.5, imax=2.0, ic50=5.0, model_type="stimulation_kin"
        )
        assert result["r_max"] > result["r_baseline"]

    def test_notes_contain_model_type(self):
        times, c = _make_decay()
        result = indirect_response_model(
            times, c, kin=1.0, kout=0.5, imax=0.8, ic50=5.0, model_type="stimulation_kin"
        )
        assert "stimulation_kin" in result["notes"]


# ---------------------------------------------------------------------------
# indirect_response_model — inhibition_kout / stimulation_kout
# ---------------------------------------------------------------------------


class TestIndirectResponseKoutModels:
    def test_inhibition_kout_response_increases(self):
        """Inhibiting degradation: biomarker should increase."""
        times, c = _make_flat(n=100, val=20.0)
        result = indirect_response_model(
            times, c, kin=1.0, kout=0.5, imax=0.9, ic50=5.0, model_type="inhibition_kout"
        )
        assert result["r_max"] > result["r_baseline"]

    def test_stimulation_kout_response_decreases(self):
        """Stimulating degradation: biomarker should decrease."""
        times, c = _make_flat(n=100, val=20.0)
        result = indirect_response_model(
            times, c, kin=1.0, kout=0.5, imax=2.0, ic50=5.0, model_type="stimulation_kout"
        )
        assert result["r_min"] < result["r_baseline"]

    def test_hill_coefficient_effect(self):
        """Hill=2 should produce different response than hill=1 at off-EC50 concentration."""
        times, c = _make_flat(n=50, val=2.0)  # C != IC50 so Hill matters
        r1 = indirect_response_model(times, c, kin=1.0, kout=0.5, imax=0.8, ic50=5.0, hill=1.0)
        r2 = indirect_response_model(times, c, kin=1.0, kout=0.5, imax=0.8, ic50=5.0, hill=2.0)
        assert r1["r_min"] != r2["r_min"]


# ---------------------------------------------------------------------------
# hysteresis_detection
# ---------------------------------------------------------------------------


class TestHysteresisDetection:
    def test_returns_correct_keys(self):
        times = [0, 1, 2, 3, 4, 5]
        c = [0, 5, 10, 8, 4, 1]
        resp = [2, 2, 2, 2, 2, 2]
        result = hysteresis_detection(times, c, resp)
        assert "ascending_r" in result
        assert "descending_r" in result
        assert "hysteresis_detected" in result
        assert "notes" in result

    def test_flat_response_no_hysteresis(self):
        """Flat response: no meaningful correlation on either limb."""
        times = list(range(20))
        c = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
        resp = [5.0] * 20
        result = hysteresis_detection(times, c, resp)
        assert result["hysteresis_detected"] is False

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            hysteresis_detection([0, 1, 2], [1, 2], [3, 4, 5])

    def test_too_few_points_raises(self):
        with pytest.raises(ValueError):
            hysteresis_detection([0, 1, 2], [1, 2, 3], [4, 5, 6])

    def test_hysteresis_detected_is_boolean(self):
        times = list(range(20))
        c = [0, 2, 4, 6, 8, 10, 10, 8, 6, 4, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        resp = [0, 0, 1, 2, 4, 6, 8, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 0, 0]
        result = hysteresis_detection(times, c, resp)
        assert isinstance(result["hysteresis_detected"], bool)


# ---------------------------------------------------------------------------
# pk_pd_link_ke0
# ---------------------------------------------------------------------------


class TestPkPdLinkKe0:
    def test_returns_correct_keys(self):
        times, c = _make_decay(n=20)
        resp = [0.5 * ci for ci in c]
        result = pk_pd_link_ke0(c, resp, times)
        assert "times_h" in result
        assert "ce_effect_site" in result
        assert "correlation_r" in result

    def test_ce_starts_at_zero(self):
        times, c = _make_decay(n=20)
        resp = [0.5 * ci for ci in c]
        result = pk_pd_link_ke0(c, resp, times)
        assert result["ce_effect_site"][0] == 0.0

    def test_ce_increases_toward_cp(self):
        """Ce should rise from 0 toward Cp for constant Cp."""
        times, c = _make_flat(n=30, val=10.0)
        resp = [5.0] * 30
        result = pk_pd_link_ke0(c, resp, times, ke0_per_h=1.0)
        ce = result["ce_effect_site"]
        assert ce[-1] > ce[0]

    def test_correlation_r_range(self):
        times, c = _make_decay(n=30)
        resp = [ci * 0.5 for ci in c]
        result = pk_pd_link_ke0(c, resp, times)
        assert -1.0 <= result["correlation_r"] <= 1.0

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            pk_pd_link_ke0([1, 2, 3], [1, 2], [0, 1, 2])

    def test_too_short_raises(self):
        with pytest.raises(ValueError):
            pk_pd_link_ke0([5.0], [3.0], [0.0])
