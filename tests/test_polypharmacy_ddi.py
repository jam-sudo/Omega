"""Tests for Phase 25 polypharmacy DDI simulation.

Covers:
  - TestPolypharmacyUnit (8): pairwise vs combined AUCR, synergy, profiles
  - TestPolypharmacyEdge (2): induction + inhibition combo, 10 perpetrators
  - TestPolypharmacyAPI (3): POST /ddi/polypharmacy endpoint contract
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from omega_pbpk.api.app import app
from omega_pbpk.clinical.ddi_simulation import (
    PerpetratorSpec,
    PolypharmacyDDIResult,
    simulate_ddi,
    simulate_polypharmacy_ddi,
)
from omega_pbpk.drugs.drug import Drug

client = TestClient(app)

# ---------------------------------------------------------------------------
# Victim drug — midazolam-like (CYP3A4 primary metabolism)
# ---------------------------------------------------------------------------

VICTIM = Drug(
    name="victim",
    mw=325.8,
    logP=3.89,
    pka=[6.15],
    drug_type="monoprotic_base",
    fup=0.032,
    rbp=0.66,
    clint_hepatic_L_per_h=750.0,
    peff=5.37,
    fm={"CYP3A4": 0.93, "other": 0.07},
    kp={
        "lung": 0.6,
        "liver": 5.8,
        "kidney": 5,
        "heart": 5,
        "brain": 6,
        "spleen": 3.8,
        "gut_wall": 4.2,
        "rest": 2.5,
    },
)

VICTIM_PAYLOAD = {
    "name": "victim",
    "mw": 325.8,
    "logP": 3.89,
    "pka": [6.15],
    "drug_type": "monoprotic_base",
    "fup": 0.032,
    "rbp": 0.66,
    "clint_hepatic_L_per_h": 750.0,
    "peff": 5.37,
    "fm": {"CYP3A4": 0.93, "other": 0.07},
}

# Shared fast simulation parameters for unit tests
SIM_KWARGS: dict = dict(dose_mg=1.0, t_end_h=12.0)


# ---------------------------------------------------------------------------
# Helper factories
# ---------------------------------------------------------------------------


def _inhibitor(name: str, ki_uM: float = 0.1, cmax_uM: float = 1.0) -> PerpetratorSpec:
    return PerpetratorSpec(name=name, ki_uM=ki_uM, cmax_uM=cmax_uM)


def _inducer(name: str, fold: float = 3.0) -> PerpetratorSpec:
    return PerpetratorSpec(
        name=name,
        ki_uM=1_000_000.0,
        fold_induction=fold,
        mechanism="induction",
        cmax_uM=1.0,
    )


# ===========================================================================
# TestPolypharmacyUnit
# ===========================================================================


class TestPolypharmacyUnit:
    def test_single_perpetrator_matches(self):
        """1 perpetrator: polypharmacy AUCR should closely match simulate_ddi() AUCR."""
        perp = _inhibitor("keto", ki_uM=0.018, cmax_uM=1.0)
        poly = simulate_polypharmacy_ddi(VICTIM, [perp], **SIM_KWARGS)
        single = simulate_ddi(VICTIM, perp, **SIM_KWARGS)
        assert isinstance(poly, PolypharmacyDDIResult)
        assert abs(poly.auc_ratio_combined - single.auc_ratio) < 0.05, (
            f"poly AUCR={poly.auc_ratio_combined:.4f} vs single={single.auc_ratio:.4f}"
        )

    def test_two_perpetrators_combined_gt_each(self):
        """2 inhibitors: combined AUCR must exceed the max individual pairwise AUCR."""
        perp1 = _inhibitor("p1", ki_uM=0.1)
        perp2 = _inhibitor("p2", ki_uM=0.1)
        poly = simulate_polypharmacy_ddi(VICTIM, [perp1, perp2], **SIM_KWARGS)
        max_pairwise = max(r.auc_ratio for r in poly.pairwise)
        assert poly.auc_ratio_combined > max_pairwise, (
            f"combined={poly.auc_ratio_combined:.4f} not > max_pairwise={max_pairwise:.4f}"
        )

    def test_no_inhibition_near_one(self):
        """All Ki=1e6 → combined AUCR ≈ 1.0 (negligible inhibition)."""
        perps = [
            PerpetratorSpec(name=f"inert{i}", ki_uM=1_000_000.0, cmax_uM=1.0) for i in range(3)
        ]
        poly = simulate_polypharmacy_ddi(VICTIM, perps, **SIM_KWARGS)
        assert abs(poly.auc_ratio_combined - 1.0) < 0.15, (
            f"AUCR={poly.auc_ratio_combined:.4f} expected ≈ 1.0"
        )

    def test_synergy_flag_true(self):
        """Two very potent inhibitors: combined > max_pairwise * 1.1 → synergy_flag=True.

        With Ki=0.01 µM and Cmax=1 µM each:
          individual R1 ≈ 101, combined R1 ≈ 201 → combined/individual > 1.1.
        """
        perp1 = _inhibitor("strong1", ki_uM=0.01, cmax_uM=1.0)
        perp2 = _inhibitor("strong2", ki_uM=0.01, cmax_uM=1.0)
        poly = simulate_polypharmacy_ddi(VICTIM, [perp1, perp2], **SIM_KWARGS)
        assert poly.synergy_flag is True, (
            f"synergy_flag=False; combined={poly.auc_ratio_combined:.4f}, "
            f"max_pairwise={max(r.auc_ratio for r in poly.pairwise):.4f}"
        )

    def test_synergy_flag_false(self):
        """Single perpetrator: synergy_flag must always be False."""
        perp = _inhibitor("solo", ki_uM=0.1)
        poly = simulate_polypharmacy_ddi(VICTIM, [perp], **SIM_KWARGS)
        assert poly.synergy_flag is False, "synergy_flag unexpectedly True for single perpetrator"

    def test_pairwise_count(self):
        """N perpetrators → exactly N pairwise DDISimulationResult objects."""
        for n in (1, 2, 5):
            perps = [_inhibitor(f"p{i}", ki_uM=0.5) for i in range(n)]
            poly = simulate_polypharmacy_ddi(VICTIM, perps, **SIM_KWARGS)
            assert len(poly.pairwise) == n, (
                f"n={n}: expected {n} pairwise results, got {len(poly.pairwise)}"
            )

    def test_profiles_equal_length(self):
        """time_h, cp_alone_mg_L, cp_combined_mg_L must be non-empty and same length."""
        perp = _inhibitor("profiler", ki_uM=0.5)
        poly = simulate_polypharmacy_ddi(VICTIM, [perp], **SIM_KWARGS)
        assert len(poly.time_h) > 0
        assert len(poly.cp_alone_mg_L) == len(poly.time_h)
        assert len(poly.cp_combined_mg_L) == len(poly.time_h)

    def test_classification_valid(self):
        """classification_combined must be one of the known FDA AUCR classes."""
        valid = {
            "no_interaction",
            "weak",
            "moderate",
            "strong",
            "weak_induction",
            "moderate_induction",
            "strong_induction",
        }
        perp = _inhibitor("classify_me", ki_uM=0.05)
        poly = simulate_polypharmacy_ddi(VICTIM, [perp], **SIM_KWARGS)
        assert poly.classification_combined in valid, (
            f"Unexpected classification: {poly.classification_combined!r}"
        )


# ===========================================================================
# TestPolypharmacyEdge
# ===========================================================================


class TestPolypharmacyEdge:
    def test_induction_plus_inhibition(self):
        """One inducer + one inhibitor: result is a valid PolypharmacyDDIResult."""
        inhibitor = _inhibitor("inhib", ki_uM=0.1)
        inducer = _inducer("induc", fold=3.0)
        poly = simulate_polypharmacy_ddi(VICTIM, [inhibitor, inducer], **SIM_KWARGS)
        assert isinstance(poly, PolypharmacyDDIResult)
        assert poly.n_perpetrators == 2
        assert poly.auc_ratio_combined > 0.0

    def test_max_perpetrators(self):
        """10 perpetrators (maximum allowed) completes without error."""
        perps = [_inhibitor(f"p{i}", ki_uM=10.0 * (i + 1)) for i in range(10)]
        poly = simulate_polypharmacy_ddi(VICTIM, perps, **SIM_KWARGS)
        assert isinstance(poly, PolypharmacyDDIResult)
        assert poly.n_perpetrators == 10
        assert len(poly.pairwise) == 10


# ===========================================================================
# TestPolypharmacyAPI
# ===========================================================================


def _poly_payload(**overrides) -> dict:
    payload: dict = {
        "victim_drug": VICTIM_PAYLOAD,
        "perpetrators": [
            {"name": "keto", "ki_uM": 0.018, "cmax_uM": 1.0},
            {"name": "clari", "ki_uM": 0.05, "cmax_uM": 1.0},
        ],
        "dose_mg": 1.0,
        "route": "oral",
        "duration_h": 12.0,
    }
    payload.update(overrides)
    return payload


class TestPolypharmacyAPI:
    def test_endpoint_200(self):
        """POST /ddi/polypharmacy returns 200."""
        r = client.post("/ddi/polypharmacy", json=_poly_payload())
        assert r.status_code == 200, r.text

    def test_response_structure(self):
        """Response body contains all expected top-level keys."""
        r = client.post("/ddi/polypharmacy", json=_poly_payload())
        assert r.status_code == 200, r.text
        body = r.json()
        expected_keys = {
            "victim_name",
            "perpetrators",
            "n_perpetrators",
            "auc_ratio_combined",
            "cmax_ratio_combined",
            "classification_combined",
            "pairwise",
            "synergy_flag",
            "time_h",
            "cp_alone_mg_L",
            "cp_combined_mg_L",
        }
        assert expected_keys.issubset(body.keys()), f"Missing keys: {expected_keys - body.keys()}"

    def test_too_many_perpetrators(self):
        """>10 perpetrators returns 422 Unprocessable Entity."""
        too_many = [{"name": f"p{i}", "ki_uM": 1.0, "cmax_uM": 1.0} for i in range(11)]
        r = client.post("/ddi/polypharmacy", json=_poly_payload(perpetrators=too_many))
        assert r.status_code == 422, f"Expected 422, got {r.status_code}: {r.text}"
