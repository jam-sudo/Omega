"""Tests for ddi_severity_classifier module (Phase 532)."""

import pytest

from omega_pbpk.clinical.ddi_severity_classifier import (
    classify_ddi_severity,
    generate_ddi_interaction_table,
    polypharmacy_ddi_risk,
)

# ---------------------------------------------------------------------------
# classify_ddi_severity — inhibition
# ---------------------------------------------------------------------------


class TestClassifyInhibition:
    def test_aucr5_strong(self):
        result = classify_ddi_severity(aucr=5.0, mechanism="inhibition")
        assert result["severity"] == "strong"

    def test_aucr10_strong(self):
        result = classify_ddi_severity(aucr=10.0, mechanism="inhibition")
        assert result["severity"] == "strong"

    def test_aucr3_moderate(self):
        result = classify_ddi_severity(aucr=3.0, mechanism="inhibition")
        assert result["severity"] == "moderate"

    def test_aucr2_moderate_boundary(self):
        result = classify_ddi_severity(aucr=2.0, mechanism="inhibition")
        assert result["severity"] == "moderate"

    def test_aucr1_5_weak(self):
        result = classify_ddi_severity(aucr=1.5, mechanism="inhibition")
        assert result["severity"] == "weak"

    def test_aucr1_1_minimal(self):
        result = classify_ddi_severity(aucr=1.1, mechanism="inhibition")
        assert result["severity"] == "minimal"

    def test_label_warning_strong(self):
        result = classify_ddi_severity(aucr=5.0, mechanism="inhibition")
        assert result["label_warning_needed"] is True

    def test_label_warning_moderate(self):
        result = classify_ddi_severity(aucr=3.0, mechanism="inhibition")
        assert result["label_warning_needed"] is True

    def test_label_warning_minimal(self):
        result = classify_ddi_severity(aucr=1.1, mechanism="inhibition")
        assert result["label_warning_needed"] is False

    def test_result_contains_aucr(self):
        result = classify_ddi_severity(aucr=3.0, mechanism="inhibition")
        assert result["aucr"] == pytest.approx(3.0)

    def test_result_contains_mechanism(self):
        result = classify_ddi_severity(aucr=3.0, mechanism="inhibition")
        assert result["mechanism"] == "inhibition"

    def test_recommendation_present(self):
        result = classify_ddi_severity(aucr=5.0, mechanism="inhibition")
        assert isinstance(result["recommendation"], str)
        assert len(result["recommendation"]) > 0


# ---------------------------------------------------------------------------
# classify_ddi_severity — induction
# ---------------------------------------------------------------------------


class TestClassifyInduction:
    def test_aucr0_1_strong(self):
        result = classify_ddi_severity(aucr=0.1, mechanism="induction")
        assert result["severity"] == "strong"

    def test_aucr0_2_strong_boundary(self):
        result = classify_ddi_severity(aucr=0.2, mechanism="induction")
        assert result["severity"] == "strong"

    def test_aucr0_3_moderate(self):
        result = classify_ddi_severity(aucr=0.3, mechanism="induction")
        assert result["severity"] == "moderate"

    def test_aucr0_6_weak(self):
        result = classify_ddi_severity(aucr=0.6, mechanism="induction")
        assert result["severity"] == "weak"

    def test_aucr0_9_minimal(self):
        result = classify_ddi_severity(aucr=0.9, mechanism="induction")
        assert result["severity"] == "minimal"

    def test_induction_label_warning_strong(self):
        result = classify_ddi_severity(aucr=0.1, mechanism="induction")
        assert result["label_warning_needed"] is True


# ---------------------------------------------------------------------------
# classify_ddi_severity — NTI drug
# ---------------------------------------------------------------------------


class TestNTIDrug:
    def test_nti_inhibition_aucr3_strong(self):
        # NTI: AUCR >= 2 → strong
        result = classify_ddi_severity(
            aucr=3.0, mechanism="inhibition", narrow_therapeutic_index=True
        )
        assert result["severity"] == "strong"

    def test_nti_inhibition_aucr1_7_moderate(self):
        result = classify_ddi_severity(
            aucr=1.7, mechanism="inhibition", narrow_therapeutic_index=True
        )
        assert result["severity"] == "moderate"

    def test_nti_induction_aucr0_6_strong(self):
        # NTI: AUCR <= 0.5 → strong
        result = classify_ddi_severity(
            aucr=0.4, mechanism="induction", narrow_therapeutic_index=True
        )
        assert result["severity"] == "strong"


# ---------------------------------------------------------------------------
# classify_ddi_severity — transporter
# ---------------------------------------------------------------------------


class TestTransporterMechanism:
    def test_transporter_uses_inhibition_thresholds(self):
        r1 = classify_ddi_severity(aucr=5.0, mechanism="transporter")
        r2 = classify_ddi_severity(aucr=5.0, mechanism="inhibition")
        assert r1["severity"] == r2["severity"]


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


class TestClassifyInputValidation:
    def test_invalid_mechanism_raises(self):
        with pytest.raises(ValueError, match="mechanism"):
            classify_ddi_severity(aucr=2.0, mechanism="unknown")

    def test_zero_aucr_raises(self):
        with pytest.raises(ValueError, match="aucr"):
            classify_ddi_severity(aucr=0.0, mechanism="inhibition")

    def test_negative_aucr_raises(self):
        with pytest.raises(ValueError, match="aucr"):
            classify_ddi_severity(aucr=-1.0, mechanism="inhibition")


# ---------------------------------------------------------------------------
# polypharmacy_ddi_risk
# ---------------------------------------------------------------------------


def _drug(name, inhibitor=False, inducer=False, substrate=False, nti=False, cyp="CYP3A4"):
    return {
        "name": name,
        "is_cyp_inhibitor": inhibitor,
        "is_cyp_inducer": inducer,
        "is_cyp_substrate": substrate,
        "is_nti": nti,
        "cyp_enzyme": cyp,
    }


class TestPolypharmacyDDI:
    def test_inhibitor_substrate_same_cyp_detected(self):
        drugs = [
            _drug("Inh", inhibitor=True, cyp="CYP3A4"),
            _drug("Sub", substrate=True, cyp="CYP3A4"),
        ]
        result = polypharmacy_ddi_risk(drugs)
        assert result["n_interactions"] >= 1

    def test_no_overlap_no_interactions(self):
        drugs = [
            _drug("Inh", inhibitor=True, cyp="CYP3A4"),
            _drug("Sub", substrate=True, cyp="CYP2D6"),
        ]
        result = polypharmacy_ddi_risk(drugs)
        assert result["n_interactions"] == 0
        assert result["risk_level"] == "low"

    def test_high_risk_pair_listed(self):
        drugs = [
            _drug("Inh", inhibitor=True, cyp="CYP3A4"),
            _drug("Sub", substrate=True, nti=True, cyp="CYP3A4"),
        ]
        result = polypharmacy_ddi_risk(drugs)
        assert len(result["high_risk_pairs"]) >= 1

    def test_total_drugs_count(self):
        drugs = [_drug("A"), _drug("B"), _drug("C")]
        result = polypharmacy_ddi_risk(drugs)
        assert result["total_drugs"] == 3

    def test_empty_list(self):
        result = polypharmacy_ddi_risk([])
        assert result["total_drugs"] == 0
        assert result["n_interactions"] == 0

    def test_risk_level_present(self):
        drugs = [
            _drug("Inh", inhibitor=True, cyp="CYP3A4"),
            _drug("Sub", substrate=True, cyp="CYP3A4"),
        ]
        result = polypharmacy_ddi_risk(drugs)
        assert result["risk_level"] in ("low", "moderate", "high")

    def test_recommendations_is_list(self):
        drugs = [_drug("A"), _drug("B")]
        result = polypharmacy_ddi_risk(drugs)
        assert isinstance(result["recommendations"], list)


# ---------------------------------------------------------------------------
# generate_ddi_interaction_table
# ---------------------------------------------------------------------------


class TestGenerateDDITable:
    def _perp(self, name, mechanism="inhibition", aucr=3.0):
        return {"name": name, "mechanism": mechanism, "predicted_aucr": aucr}

    def _victim(self, name, nti=False):
        return {"name": name, "is_nti": nti}

    def test_cross_product_size(self):
        perps = [self._perp("P1"), self._perp("P2")]
        victims = [self._victim("V1"), self._victim("V2"), self._victim("V3")]
        table = generate_ddi_interaction_table(perps, victims)
        assert len(table) == 6  # 2 × 3

    def test_row_keys_present(self):
        table = generate_ddi_interaction_table([self._perp("P")], [self._victim("V")])
        row = table[0]
        assert "perpetrator" in row
        assert "victim" in row
        assert "aucr" in row
        assert "severity" in row
        assert "label_warning_needed" in row

    def test_strong_inhibitor_flagged(self):
        table = generate_ddi_interaction_table(
            [self._perp("P", mechanism="inhibition", aucr=8.0)],
            [self._victim("V")],
        )
        assert table[0]["severity"] == "strong"
        assert table[0]["label_warning_needed"] is True

    def test_nti_victim_tighter_threshold(self):
        # AUCR=3 for NTI victim → strong (NTI: >=2 strong)
        table = generate_ddi_interaction_table(
            [self._perp("P", mechanism="inhibition", aucr=3.0)],
            [self._victim("NTI_drug", nti=True)],
        )
        assert table[0]["severity"] == "strong"

    def test_empty_perpetrators_raises(self):
        with pytest.raises(ValueError):
            generate_ddi_interaction_table([], [self._victim("V")])

    def test_empty_victims_raises(self):
        with pytest.raises(ValueError):
            generate_ddi_interaction_table([self._perp("P")], [])
