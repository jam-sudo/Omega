"""Tests for steady-state PK predictor."""

import dataclasses

import pytest

from omega_pbpk.clinical.steady_state_predictor import (
    SteadyStateResult,
    predict_steady_state,
)
from omega_pbpk.drugs.caffeine import CAFFEINE
from omega_pbpk.drugs.metformin import METFORMIN


@pytest.fixture(scope="module")
def caffeine_ss():
    return predict_steady_state(
        CAFFEINE, dose_mg=100, interval_h=12, route="oral", n_doses=10
    )


def test_result_is_dataclass(caffeine_ss):
    assert isinstance(caffeine_ss, SteadyStateResult)
    assert dataclasses.is_dataclass(caffeine_ss)


def test_css_max_ge_css_min(caffeine_ss):
    assert caffeine_ss.css_max_mg_L >= caffeine_ss.css_min_mg_L


def test_accumulation_ratio_positive(caffeine_ss):
    assert caffeine_ss.accumulation_ratio > 0


def test_fluctuation_nonnegative(caffeine_ss):
    assert caffeine_ss.fluctuation_pct >= 0


@pytest.mark.integration
def test_caffeine_q12h_accumulates(caffeine_ss):
    # Caffeine apparent t1/2 in PBPK ~21h, so q12h dosing should accumulate
    assert caffeine_ss.accumulation_ratio > 1.0


@pytest.mark.integration
def test_metformin_iv_runs():
    result = predict_steady_state(
        METFORMIN, dose_mg=500, interval_h=12, route="iv", n_doses=5
    )
    assert isinstance(result, SteadyStateResult)
    assert result.css_max_mg_L > 0
