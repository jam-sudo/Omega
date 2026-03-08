"""Tests for pericardial fluid drug distribution prediction (Phase 921)."""

import pytest

from omega_pbpk.prediction.pericardial_distribution import (
    PericardialDistributionResult,
    predict_pericardial_distribution,
    screen_intrapericardial_drugs,
)


class TestPericardialDistributionResult:
    def test_is_frozen_dataclass(self):
        result = predict_pericardial_distribution("TestDrug")
        assert isinstance(result, PericardialDistributionResult)
        with pytest.raises((AttributeError, TypeError)):
            result.drug_name = "OtherDrug"  # type: ignore[misc]

    def test_fields_present(self):
        r = predict_pericardial_distribution("Drug")
        assert hasattr(r, "drug_name")
        assert hasattr(r, "logp")
        assert hasattr(r, "mw_Da")
        assert hasattr(r, "route")
        assert hasattr(r, "dose_mg")
        assert hasattr(r, "pericardial_plasma_ratio")
        assert hasattr(r, "predicted_pericardial_conc_mg_L")
        assert hasattr(r, "plasma_conc_input_mg_L")
        assert hasattr(r, "pericardial_residence_h")
        assert hasattr(r, "epicardial_penetration_fraction")
        assert hasattr(r, "local_cardiac_advantage")
        assert hasattr(r, "notes")


class TestSystemicRoute:
    def test_default_systemic(self):
        r = predict_pericardial_distribution("Amiodarone", route="systemic")
        assert r.route == "systemic"
        assert r.drug_name == "Amiodarone"

    def test_pericardial_ratio_low_logp(self):
        # logp=0: ratio = 0.05 + 0*0.02 = 0.05
        r = predict_pericardial_distribution(
            "Drug", logp=0.0, route="systemic", plasma_conc_mg_L=1.0
        )
        assert abs(r.pericardial_plasma_ratio - 0.05) < 1e-9

    def test_pericardial_ratio_positive_logp(self):
        # logp=2: ratio = 0.05 + 2*0.02 = 0.09
        r = predict_pericardial_distribution(
            "Drug", logp=2.0, route="systemic", plasma_conc_mg_L=1.0
        )
        assert abs(r.pericardial_plasma_ratio - 0.09) < 1e-9

    def test_pericardial_ratio_clamped_max(self):
        # logp very high should clamp ratio to 0.5
        r = predict_pericardial_distribution(
            "Drug", logp=100.0, route="systemic", plasma_conc_mg_L=1.0
        )
        assert r.pericardial_plasma_ratio == pytest.approx(0.5, abs=1e-9)

    def test_pericardial_ratio_clamped_min(self):
        # logp very negative: 0.05 + (-100)*0.02 = -1.95 -> clamp to 0.01
        r = predict_pericardial_distribution(
            "Drug", logp=-100.0, route="systemic", plasma_conc_mg_L=1.0
        )
        assert r.pericardial_plasma_ratio == pytest.approx(0.01, abs=1e-9)

    def test_predicted_conc_systemic(self):
        r = predict_pericardial_distribution(
            "Drug", logp=0.0, route="systemic", plasma_conc_mg_L=2.0
        )
        # ratio = 0.05, conc = 0.05 * 2.0 = 0.1
        assert abs(r.predicted_pericardial_conc_mg_L - 0.1) < 1e-9

    def test_residence_systemic(self):
        # logp=2: 0.5 + 2*0.2 = 0.9
        r = predict_pericardial_distribution("Drug", logp=2.0, route="systemic")
        assert abs(r.pericardial_residence_h - 0.9) < 1e-9

    def test_residence_clamped_min_systemic(self):
        r = predict_pericardial_distribution("Drug", logp=-100.0, route="systemic")
        assert r.pericardial_residence_h == pytest.approx(0.1, abs=1e-9)

    def test_residence_clamped_max_systemic(self):
        r = predict_pericardial_distribution("Drug", logp=100.0, route="systemic")
        assert r.pericardial_residence_h == pytest.approx(5.0, abs=1e-9)

    def test_notes_limited_distribution(self):
        # logp=0 systemic -> penetration=0.05, no high advantage, no good penetration
        r = predict_pericardial_distribution(
            "Drug", logp=0.0, route="systemic", plasma_conc_mg_L=1.0
        )
        assert "Limited pericardial distribution" in r.notes


class TestIntrapericardialRoute:
    def test_intrapericardial_basic(self):
        r = predict_pericardial_distribution("Ibuprofen", dose_mg=0.6, route="intrapericardial")
        assert r.route == "intrapericardial"
        # pericardial_conc = 0.6 / 0.030 = 20 mg/L
        assert abs(r.predicted_pericardial_conc_mg_L - 20.0) < 1e-9

    def test_intrapericardial_ratio_high(self):
        # dose=1.0, plasma=0.001 -> pericardial = 1/0.03 = 33.3, ratio = 33.3/0.001 = 33333 -> clamp 1000
        r = predict_pericardial_distribution(
            "Drug", dose_mg=1.0, route="intrapericardial", plasma_conc_mg_L=0.001
        )
        assert r.pericardial_plasma_ratio == pytest.approx(1000.0, abs=1e-6)

    def test_intrapericardial_ratio_floor(self):
        # large plasma: pericardial=1/0.03=33.3, plasma=1000 -> ratio=0.033 -> clamp to 1.0
        r = predict_pericardial_distribution(
            "Drug", dose_mg=1.0, route="intrapericardial", plasma_conc_mg_L=1000.0
        )
        assert r.pericardial_plasma_ratio == pytest.approx(1.0, abs=1e-9)

    def test_residence_intrapericardial(self):
        # logp=2: 2.0 + 2*1.0 = 4.0
        r = predict_pericardial_distribution("Drug", logp=2.0, route="intrapericardial")
        assert abs(r.pericardial_residence_h - 4.0) < 1e-9

    def test_residence_clamped_max_ip(self):
        r = predict_pericardial_distribution("Drug", logp=100.0, route="intrapericardial")
        assert r.pericardial_residence_h == pytest.approx(48.0, abs=1e-9)

    def test_residence_clamped_min_ip(self):
        r = predict_pericardial_distribution("Drug", logp=-100.0, route="intrapericardial")
        assert r.pericardial_residence_h == pytest.approx(1.0, abs=1e-9)

    def test_notes_high_local_advantage(self):
        # local_advantage > 10 and intrapericardial -> high advantage note
        r = predict_pericardial_distribution(
            "Drug", dose_mg=1.0, route="intrapericardial", plasma_conc_mg_L=0.01
        )
        assert r.local_cardiac_advantage > 10
        assert "High local cardiac advantage" in r.notes


class TestEpicardialPenetration:
    def test_penetration_high_logp(self):
        r = predict_pericardial_distribution("Drug", logp=3.0, route="systemic")
        assert r.epicardial_penetration_fraction == pytest.approx(0.3)

    def test_penetration_medium_logp(self):
        r = predict_pericardial_distribution("Drug", logp=1.0, route="systemic")
        assert r.epicardial_penetration_fraction == pytest.approx(0.15)

    def test_penetration_low_logp(self):
        r = predict_pericardial_distribution("Drug", logp=-1.0, route="systemic")
        assert r.epicardial_penetration_fraction == pytest.approx(0.05)

    def test_notes_good_epicardial_penetration(self):
        # logp=3 systemic: penetration=0.3 > 0.2 -> good penetration note
        r = predict_pericardial_distribution(
            "Drug", logp=3.0, route="systemic", plasma_conc_mg_L=0.01
        )
        assert "Good epicardial penetration" in r.notes


class TestValidation:
    def test_invalid_dose(self):
        with pytest.raises(ValueError, match="dose_mg"):
            predict_pericardial_distribution("Drug", dose_mg=0.0)

    def test_invalid_mw(self):
        with pytest.raises(ValueError, match="mw_Da"):
            predict_pericardial_distribution("Drug", mw_Da=-1.0)

    def test_invalid_route(self):
        with pytest.raises(ValueError, match="route"):
            predict_pericardial_distribution("Drug", route="oral")


class TestScreenIntrapericardialDrugs:
    def test_returns_list(self):
        candidates = [
            {"name": "DrugA", "logp": 3.0},
            {"name": "DrugB", "logp": 0.5},
            {"name": "DrugC", "logp": -1.0},
        ]
        results = screen_intrapericardial_drugs(candidates)
        assert len(results) == 3

    def test_sorted_by_epicardial_penetration(self):
        candidates = [
            {"name": "Low", "logp": -1.0},
            {"name": "High", "logp": 3.0},
            {"name": "Med", "logp": 1.0},
        ]
        results = screen_intrapericardial_drugs(candidates)
        fractions = [r.epicardial_penetration_fraction for r in results]
        assert fractions == sorted(fractions, reverse=True)

    def test_mw_default(self):
        candidates = [{"name": "Drug", "logp": 2.0}]
        results = screen_intrapericardial_drugs(candidates)
        assert results[0].mw_Da == pytest.approx(300.0)

    def test_mw_custom(self):
        candidates = [{"name": "Drug", "logp": 2.0, "mw_Da": 450.0}]
        results = screen_intrapericardial_drugs(candidates)
        assert results[0].mw_Da == pytest.approx(450.0)

    def test_all_intrapericardial_route(self):
        candidates = [{"name": "Drug", "logp": 1.0}]
        results = screen_intrapericardial_drugs(candidates)
        assert results[0].route == "intrapericardial"
