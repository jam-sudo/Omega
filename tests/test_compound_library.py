"""Tests for Phase 48 Compound Library Screening.

Covers:
  - TestDataClasses (2): CompoundEntry / ScreeningCriteria defaults
  - TestSimulatePK (3): IV cmax, oral AUC, t_half
  - TestEmaxHill (2): zero conc, ec50 midpoint
  - TestParetoRank (3): single, dominated pair, all non-dominated
  - TestClustering (3): length, range, determinism
  - TestCompositeScore (2): meeting criteria, far outside
  - TestScreenLibrary (6): structure, n_passed, lead count, rank-1 leads, sort, determinism
  - TestAPIEndpoint (2): valid POST, empty list 422
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from fastapi.testclient import TestClient

from omega_pbpk.api.app import app
from omega_pbpk.clinical.compound_library import (
    CompoundEntry,
    LibraryScreenResult,
    ScreeningCriteria,
    _composite_score,
    _emax_hill,
    _kmeans_cluster,
    _pareto_rank,
    _simulate_compound_pk,
    screen_library,
)

client = TestClient(app)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

_ENTRY_A = CompoundEntry(
    name="CompA",
    dose_mg=100,
    cl_L_per_h=5,
    vd_L=70,
    ka_per_h=1.2,
    f_bioavail=0.8,
)
_ENTRY_B = CompoundEntry(
    name="CompB",
    dose_mg=100,
    cl_L_per_h=10,
    vd_L=50,
    ka_per_h=0.8,
    f_bioavail=0.6,
)
_ENTRY_C = CompoundEntry(
    name="CompC",
    dose_mg=200,
    cl_L_per_h=3,
    vd_L=80,
    ka_per_h=2.0,
    f_bioavail=0.9,
)


# ---------------------------------------------------------------------------
# TestDataClasses
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestDataClasses:
    """CompoundEntry and ScreeningCriteria have sensible defaults."""

    def test_compound_entry_defaults(self):
        e = CompoundEntry(name="Test")
        assert e.name == "Test"
        assert e.dose_mg == 100.0
        assert e.route == "oral"
        assert e.f_bioavail == 0.8
        assert e.vd_L == 70.0
        assert e.cl_L_per_h == 5.0

    def test_screening_criteria_defaults(self):
        c = ScreeningCriteria()
        assert c.min_auc_mg_h_L == 0.0
        assert c.max_cmax_mg_L == 1e9
        assert c.min_f_bioavail == 0.0
        assert c.hill == 1.0
        assert c.ec50_mg_L == 1.0


# ---------------------------------------------------------------------------
# TestSimulatePK
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSimulatePK:
    """Analytical 1-compartment PK via _simulate_compound_pk."""

    def test_iv_cmax(self):
        entry = CompoundEntry(
            name="IV", route="iv", dose_mg=100, vd_L=50, cl_L_per_h=5, ka_per_h=1.0, f_bioavail=1.0
        )
        cmax, _auc, _tmax, _thalf = _simulate_compound_pk(entry)
        expected = 100.0 / 50.0  # dose / vd
        assert abs(cmax - expected) / expected < 0.01

    def test_oral_auc(self):
        entry = CompoundEntry(
            name="Oral",
            route="oral",
            dose_mg=200,
            cl_L_per_h=4,
            vd_L=60,
            ka_per_h=1.5,
            f_bioavail=0.7,
        )
        _cmax, auc, _tmax, _thalf = _simulate_compound_pk(entry)
        expected = 0.7 * 200.0 / 4.0  # f * dose / cl
        assert abs(auc - expected) / expected < 0.01

    def test_t_half(self):
        entry = CompoundEntry(
            name="THalf", dose_mg=100, cl_L_per_h=5, vd_L=70, ka_per_h=1.2, f_bioavail=0.8
        )
        _cmax, _auc, _tmax, t_half = _simulate_compound_pk(entry)
        expected = math.log(2) * 70.0 / 5.0
        assert abs(t_half - expected) / expected < 0.01


# ---------------------------------------------------------------------------
# TestEmaxHill
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestEmaxHill:
    """Emax Hill equation helper."""

    def test_zero_concentration(self):
        assert _emax_hill(0, 1.0, 1.0, 1.0) == 0.0

    def test_ec50_gives_half_emax(self):
        ec50, emax = 5.0, 2.0
        result = _emax_hill(ec50, emax, ec50, 1.0)
        assert abs(result - emax / 2.0) < 1e-6


# ---------------------------------------------------------------------------
# TestParetoRank
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestParetoRank:
    """Non-dominated sorting (Pareto ranking)."""

    def test_single_compound(self):
        assert _pareto_rank([(1.0, 2.0, 3.0, 4.0)]) == [1]

    def test_dominated_pair(self):
        # A dominates B on all objectives
        a = (10.0, 10.0, 0.0, 0.0)
        b = (5.0, 5.0, -1.0, -1.0)
        ranks = _pareto_rank([a, b])
        assert ranks[0] == 1
        assert ranks[1] == 2

    def test_all_non_dominated(self):
        # Each is best in one dimension → all rank 1
        a = (10.0, 1.0, 0.0, 0.0)
        b = (1.0, 10.0, 0.0, 0.0)
        c = (1.0, 1.0, 10.0, 0.0)
        ranks = _pareto_rank([a, b, c])
        assert ranks == [1, 1, 1]


# ---------------------------------------------------------------------------
# TestClustering
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestClustering:
    """K-means clustering helper."""

    def test_returns_correct_length(self):
        features = np.random.default_rng(0).random((10, 4))
        labels = _kmeans_cluster(features, 3, seed=42)
        assert len(labels) == 10

    def test_cluster_ids_in_range(self):
        features = np.random.default_rng(0).random((10, 4))
        labels = _kmeans_cluster(features, 3, seed=42)
        assert all(0 <= cid < 3 for cid in labels)

    def test_deterministic(self):
        features = np.random.default_rng(0).random((10, 4))
        a = _kmeans_cluster(features, 3, seed=42)
        b = _kmeans_cluster(features, 3, seed=42)
        assert a == b


# ---------------------------------------------------------------------------
# TestCompositeScore
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestCompositeScore:
    """Composite scoring (0-100)."""

    def test_meeting_criteria_scores_above_50(self):
        criteria = ScreeningCriteria(
            min_auc_mg_h_L=10.0,
            max_cmax_mg_L=50.0,
            min_effect=0.5,
            max_t_half_h=24.0,
            target_cmax_lower=1.0,
            target_cmax_upper=5.0,
        )
        # Compound that meets criteria
        score = _composite_score(
            cmax=3.0,
            auc=10.0,
            effect=0.5,
            t_half=10.0,
            criteria=criteria,
        )
        assert score > 50

    def test_far_outside_criteria_scores_below_30(self):
        criteria = ScreeningCriteria(
            min_auc_mg_h_L=100.0,
            max_cmax_mg_L=1.0,
            min_effect=0.9,
            max_t_half_h=2.0,
            target_cmax_lower=0.5,
            target_cmax_upper=1.0,
        )
        # Compound far outside criteria
        score = _composite_score(
            cmax=50.0,
            auc=0.01,
            effect=0.001,
            t_half=100.0,
            criteria=criteria,
        )
        assert score < 30


# ---------------------------------------------------------------------------
# TestScreenLibrary
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestScreenLibrary:
    """End-to-end screen_library() tests."""

    _entries = [_ENTRY_A, _ENTRY_B, _ENTRY_C]
    _criteria = ScreeningCriteria(ec50_mg_L=1.0, emax=1.0)

    def test_result_structure(self):
        result = screen_library(self._entries, self._criteria, n_clusters=2, seed=42)
        assert isinstance(result, LibraryScreenResult)
        assert len(result.results) == 3
        assert isinstance(result.clusters, dict)
        assert isinstance(result.lead_compounds, list)

    def test_n_passed_count(self):
        result = screen_library(self._entries, self._criteria, n_clusters=2, seed=42)
        manual_count = sum(1 for r in result.results if r.passes_criteria)
        assert result.n_passed == manual_count

    def test_lead_compounds_at_most_3(self):
        # Even with many compounds, leads capped at 3
        many = [CompoundEntry(name=f"C{i}", dose_mg=50 + i * 10) for i in range(10)]
        result = screen_library(many, self._criteria, n_clusters=3, seed=42)
        assert len(result.lead_compounds) <= 3

    def test_lead_compounds_are_rank1(self):
        result = screen_library(self._entries, self._criteria, n_clusters=2, seed=42)
        rank1_names = {r.name for r in result.results if r.pareto_rank == 1}
        if rank1_names:
            for lead in result.lead_compounds:
                assert lead in rank1_names

    def test_sorted_by_rank_then_score(self):
        result = screen_library(self._entries, self._criteria, n_clusters=2, seed=42)
        for i in range(len(result.results) - 1):
            a = result.results[i]
            b = result.results[i + 1]
            assert (a.pareto_rank, -a.composite_score) <= (b.pareto_rank, -b.composite_score)

    def test_deterministic(self):
        r1 = screen_library(self._entries, self._criteria, n_clusters=2, seed=42)
        r2 = screen_library(self._entries, self._criteria, n_clusters=2, seed=42)
        assert [r.name for r in r1.results] == [r.name for r in r2.results]
        assert [r.composite_score for r in r1.results] == [r.composite_score for r in r2.results]


# ---------------------------------------------------------------------------
# TestAPIEndpoint
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestAPIEndpoint:
    """POST /library/screen API endpoint."""

    def test_valid_request(self):
        payload = {
            "entries": [
                {
                    "name": "CompA",
                    "dose_mg": 100,
                    "cl_L_per_h": 5,
                    "vd_L": 70,
                    "ka_per_h": 1.2,
                    "f_bioavail": 0.8,
                },
                {
                    "name": "CompB",
                    "dose_mg": 100,
                    "cl_L_per_h": 10,
                    "vd_L": 50,
                    "ka_per_h": 0.8,
                    "f_bioavail": 0.6,
                },
                {
                    "name": "CompC",
                    "dose_mg": 200,
                    "cl_L_per_h": 3,
                    "vd_L": 80,
                    "ka_per_h": 2.0,
                    "f_bioavail": 0.9,
                },
            ],
            "criteria": {"ec50_mg_L": 1.0, "emax": 1.0},
            "n_clusters": 2,
        }
        resp = client.post("/library/screen", json=payload)
        assert resp.status_code == 200
        body = resp.json()
        assert "results" in body
        assert "lead_compounds" in body

    def test_empty_entries_422(self):
        payload = {"entries": [], "criteria": {}}
        resp = client.post("/library/screen", json=payload)
        assert resp.status_code == 422
