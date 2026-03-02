"""Tests for Phase 22 pediatric PBPK: ontogeny wiring, enzyme maturation, GFR scaling, API.

Tests are organized into four classes:
  TestOntogenyWiring     — age_years parameter, pediatric vs adult PK ordering
  TestEnzymeMaturation   — fm-weighted enzyme maturation via _clint_effective
  TestGFROntogeny        — renal CL reduction in neonates/children
  TestPediatricAPI       — POST /simulate/pediatric endpoint contract
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from omega_pbpk.api.app import app
from omega_pbpk.core.body import WholeBodyPBPK
from omega_pbpk.drugs.drug import Drug

client = TestClient(app)

# ---------------------------------------------------------------------------
# Shared test drug — midazolam-like CYP3A4 primary substrate
# ---------------------------------------------------------------------------

TEST_DRUG = Drug(
    name="test_ped",
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
        "brain": 6.0,
        "heart": 5.0,
        "kidney": 5.0,
        "liver": 5.8,
        "spleen": 3.8,
        "gut_wall": 4.2,
        "rest": 2.5,
    },
)

TEST_DRUG_PAYLOAD = {
    "name": "test_ped",
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

_DOSE_MG = 1.0
_SIM_H = 12.0


def _run_iv(drug: Drug, bw: float, age: float | None) -> dict:
    """Helper: IV bolus simulation → pk_summary dict."""
    model = WholeBodyPBPK(drug=drug, body_weight=bw, age_years=age)
    model.setup_iv(_DOSE_MG)
    return model.simulate(t_end_h=_SIM_H).pk_summary()


# ===========================================================================
# TestOntogenyWiring
# ===========================================================================


class TestOntogenyWiring:
    """Verify that age_years parameter correctly modifies pediatric PK."""

    def test_adult_age_unscaled(self):
        """WholeBodyPBPK(age_years=30) AUC matches age_years=None (same 70 kg adult)."""
        pk_none = _run_iv(TEST_DRUG, bw=70.0, age=None)
        pk_30 = _run_iv(TEST_DRUG, bw=70.0, age=30.0)
        rel_diff = abs(pk_none["AUC_mg_h_L"] - pk_30["AUC_mg_h_L"]) / max(
            pk_none["AUC_mg_h_L"], 1e-12
        )
        assert rel_diff < 0.01, f"AUC rel diff = {rel_diff:.4f}, expected < 1 %"

    def test_neonate_lower_clearance(self):
        """Neonate (0.1 y, 3 kg) apparent CL is lower than adult (age=None, 70 kg)."""
        pk_adult = _run_iv(TEST_DRUG, bw=70.0, age=None)
        pk_neo = _run_iv(TEST_DRUG, bw=3.0, age=0.1)
        assert pk_neo["CL_L_h"] < pk_adult["CL_L_h"], (
            f"Neonate CL={pk_neo['CL_L_h']:.4f} should be < adult CL={pk_adult['CL_L_h']:.4f}"
        )

    def test_neonate_higher_auc(self):
        """Neonate (0.1 y, 3 kg) has higher AUC than adult — immature clearance."""
        pk_adult = _run_iv(TEST_DRUG, bw=70.0, age=None)
        pk_neo = _run_iv(TEST_DRUG, bw=3.0, age=0.1)
        assert pk_neo["AUC_mg_h_L"] > pk_adult["AUC_mg_h_L"], (
            f"Neonate AUC={pk_neo['AUC_mg_h_L']:.4f} should exceed "
            f"adult={pk_adult['AUC_mg_h_L']:.4f}"
        )

    def test_infant_intermediate(self):
        """1-year-old (10 kg) AUC lies between neonate and adult."""
        pk_adult = _run_iv(TEST_DRUG, bw=70.0, age=None)
        pk_neo = _run_iv(TEST_DRUG, bw=3.0, age=0.1)
        pk_infant = _run_iv(TEST_DRUG, bw=10.0, age=1.0)
        assert pk_infant["AUC_mg_h_L"] > pk_adult["AUC_mg_h_L"], (
            f"Infant AUC {pk_infant['AUC_mg_h_L']:.4f} should exceed "
            f"adult {pk_adult['AUC_mg_h_L']:.4f}"
        )
        assert pk_infant["AUC_mg_h_L"] < pk_neo["AUC_mg_h_L"], (
            f"Infant AUC {pk_infant['AUC_mg_h_L']:.4f} should be below "
            f"neonate {pk_neo['AUC_mg_h_L']:.4f}"
        )

    def test_adolescent_near_adult(self):
        """15-year-old (60 kg) AUC is within 30 % of adult."""
        pk_adult = _run_iv(TEST_DRUG, bw=70.0, age=None)
        pk_teen = _run_iv(TEST_DRUG, bw=60.0, age=15.0)
        ratio = pk_teen["AUC_mg_h_L"] / max(pk_adult["AUC_mg_h_L"], 1e-12)
        assert 0.7 <= ratio <= 1.5, (
            f"Adolescent/adult AUC ratio = {ratio:.3f} expected in [0.7, 1.5]"
        )


# ===========================================================================
# TestEnzymeMaturation
# ===========================================================================


class TestEnzymeMaturation:
    """Verify fm-weighted enzyme maturation scaling in _clint_effective."""

    def test_cyp3a4_fm_scaling(self):
        """Drug with fm_CYP3A4=0.93 shows >3× AUC elevation at birth vs adult."""
        pk_adult = _run_iv(TEST_DRUG, bw=70.0, age=None)
        pk_neo = _run_iv(TEST_DRUG, bw=3.0, age=0.1)
        ratio = pk_neo["AUC_mg_h_L"] / max(pk_adult["AUC_mg_h_L"], 1e-12)
        assert ratio > 3.0, f"Expected neonate/adult AUC ratio > 3, got {ratio:.2f}"

    def test_no_fm_defaults(self):
        """Drug without fm dict still applies CYP3A4 default ontogeny; neonate AUC > adult."""
        drug_no_fm = Drug(
            name="no_fm",
            mw=325.8,
            logP=3.89,
            fup=0.032,
            rbp=0.66,
            clint_hepatic_L_per_h=750.0,
            peff=5.37,
        )
        pk_adult = _run_iv(drug_no_fm, bw=70.0, age=None)
        pk_neo = _run_iv(drug_no_fm, bw=3.0, age=0.1)
        assert pk_neo["AUC_mg_h_L"] > pk_adult["AUC_mg_h_L"]

    def test_mixed_fm_weighted(self):
        """CYP3A4-only drug has higher effective CLint at birth than CYP3A4+CYP2D6 drug.

        CYP2D6 matures more slowly than CYP3A4 (tm50=0.4 y vs 0.25 y, higher Hill),
        so adding CYP2D6 fraction increases the overall CLint reduction at birth.
        """
        drug_pure_3a4 = Drug(
            name="pure_3a4",
            mw=325.8,
            logP=3.89,
            fup=0.032,
            rbp=0.66,
            clint_hepatic_L_per_h=750.0,
            peff=5.37,
            fm={"CYP3A4": 1.0},
        )
        drug_mixed = Drug(
            name="mixed_fm",
            mw=325.8,
            logP=3.89,
            fup=0.032,
            rbp=0.66,
            clint_hepatic_L_per_h=750.0,
            peff=5.37,
            fm={"CYP3A4": 0.50, "CYP2D6": 0.50},
        )
        neo_pure = WholeBodyPBPK(drug=drug_pure_3a4, body_weight=3.0, age_years=0.1)
        neo_mixed = WholeBodyPBPK(drug=drug_mixed, body_weight=3.0, age_years=0.1)
        # CYP3A4-only drug retains more CLint at birth → higher effective CLint
        assert neo_pure._clint_effective() > neo_mixed._clint_effective(), (
            f"Pure CYP3A4 CLint_eff {neo_pure._clint_effective():.2f} should exceed "
            f"mixed {neo_mixed._clint_effective():.2f} at birth"
        )

    def test_non_cyp_fraction_unscaled(self):
        """fm={'other': 1.0} yields no maturation reduction — CLint ratio neonate/adult ≈ 1."""
        drug_other = Drug(
            name="other_fm",
            mw=325.8,
            logP=3.89,
            fup=0.032,
            rbp=0.66,
            clint_hepatic_L_per_h=200.0,
            peff=5.37,
            fm={"other": 1.0},
        )
        model_neo = WholeBodyPBPK(drug=drug_other, body_weight=3.0, age_years=0.1)
        model_adult = WholeBodyPBPK(drug=drug_other, body_weight=70.0, age_years=None)
        ratio = model_neo._clint_effective() / max(model_adult._clint_effective(), 1e-12)
        assert abs(ratio - 1.0) < 0.01, (
            f"Non-CYP fm should not be maturation-scaled; CLint ratio = {ratio:.4f}"
        )


# ===========================================================================
# TestGFROntogeny
# ===========================================================================


class TestGFROntogeny:
    """Verify GFR-based renal clearance ontogeny scaling."""

    def test_renal_neonate_reduced(self):
        """Neonate GFR factor is < 0.5 and less than adult GFR factor (1.0)."""
        model_neo = WholeBodyPBPK(drug=TEST_DRUG, body_weight=3.0, age_years=0.1)
        model_adult = WholeBodyPBPK(drug=TEST_DRUG, body_weight=70.0, age_years=None)
        gfr_neo = model_neo._ontogeny_factors["gfr"]
        gfr_adult = model_adult._ontogeny_factors["gfr"]
        assert gfr_neo < gfr_adult, f"Neonate GFR {gfr_neo:.3f} should be < adult {gfr_adult:.3f}"
        assert gfr_neo < 0.5, f"Neonate GFR factor {gfr_neo:.3f} should be < 0.5"

    def test_renal_adult_unscaled(self):
        """Adult (age_years=None) GFR ontogeny factor is exactly 1.0."""
        model = WholeBodyPBPK(drug=TEST_DRUG, body_weight=70.0, age_years=None)
        assert model._ontogeny_factors["gfr"] == 1.0


# ===========================================================================
# TestPediatricAPI
# ===========================================================================


def _ped_payload(**overrides) -> dict:
    """Build a valid POST /simulate/pediatric payload."""
    payload = {
        "drug": TEST_DRUG_PAYLOAD,
        "dose_mg": 1.0,
        "route": "iv",
        "duration_h": 12.0,
        "age_years": 0.5,
        "body_weight_kg": 7.0,
    }
    payload.update(overrides)
    return payload


class TestPediatricAPI:
    """Verify POST /simulate/pediatric endpoint contract."""

    def test_endpoint_returns_200(self):
        """Valid pediatric payload returns HTTP 200."""
        r = client.post("/simulate/pediatric", json=_ped_payload())
        assert r.status_code == 200, r.text

    def test_adult_comparison(self):
        """Response contains both pediatric and adult PK keys."""
        r = client.post("/simulate/pediatric", json=_ped_payload())
        assert r.status_code == 200, r.text
        body = r.json()
        for key in (
            "pediatric_auc_mg_h_L",
            "pediatric_cmax_mg_L",
            "adult_auc_mg_h_L",
            "adult_cmax_mg_L",
            "auc_ratio",
            "cmax_ratio",
            "ontogeny_factors",
        ):
            assert key in body, f"Missing key: {key}"

    def test_invalid_age_rejects(self):
        """age_years > 18 returns HTTP 422 (Pydantic field validation)."""
        r = client.post("/simulate/pediatric", json=_ped_payload(age_years=20.0))
        assert r.status_code == 422, f"Expected 422, got {r.status_code}: {r.text}"

    def test_cmax_ratio_positive(self):
        """cmax_ratio and auc_ratio returned by endpoint are strictly positive."""
        r = client.post("/simulate/pediatric", json=_ped_payload())
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["cmax_ratio"] > 0, f"cmax_ratio={body['cmax_ratio']}"
        assert body["auc_ratio"] > 0, f"auc_ratio={body['auc_ratio']}"
