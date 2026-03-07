"""Tests for Phase 1040 — physicochemical profiling."""

import pytest

from omega_pbpk.prediction.physicochemical_profiler import (
    PhysicochemicalProfile,
    compare_candidates,
    profile_drug,
)

_VALID_BCS = {"BCS I", "BCS II", "BCS III", "BCS IV"}
_VALID_RISK = {"low", "moderate", "high"}


def _default(**kwargs):
    """Call profile_drug with reasonable defaults."""
    params = dict(
        drug_name="TestDrug",
        mw_Da=300.0,
        logp=2.0,
        hbd=2,
        hba=5,
        psa_A2=60.0,
        rotatable_bonds=5,
        pka=7.0,
        ionization_type="neutral",
    )
    params.update(kwargs)
    return profile_drug(**params)


class TestReturnType:
    def test_returns_frozen_dataclass(self):
        result = _default()
        assert isinstance(result, PhysicochemicalProfile)

    def test_frozen(self):
        result = _default()
        with pytest.raises((AttributeError, TypeError)):
            result.drug_likeness_score = 42.0  # type: ignore[misc]


class TestNumericalBounds:
    def test_logd_in_range(self):
        result = _default()
        assert -5.0 <= result.logd_74 <= 10.0

    def test_cns_mpo_in_range(self):
        result = _default()
        assert 0.0 <= result.cns_mpo_score <= 6.0

    def test_drug_likeness_in_range(self):
        result = _default()
        assert 0.0 <= result.drug_likeness_score <= 100.0


class TestRulesPass:
    def test_lipinski_pass(self):
        # MW=300, logP=2, HBD=2, HBA=5 — all within limits
        result = _default(mw_Da=300.0, logp=2.0, hbd=2, hba=5)
        assert result.lipinski_pass is True

    def test_lipinski_fail_high_mw(self):
        result = _default(mw_Da=600.0, logp=6.0, hbd=2, hba=5)
        assert result.lipinski_pass is False

    def test_lipinski_fail_high_logp(self):
        result = _default(mw_Da=400.0, logp=6.0)
        assert result.lipinski_pass is False

    def test_veber_pass(self):
        result = _default(rotatable_bonds=5, psa_A2=80.0)
        assert result.veber_pass is True

    def test_veber_fail_high_rot_bonds(self):
        result = _default(rotatable_bonds=12, psa_A2=80.0)
        assert result.veber_pass is False

    def test_veber_fail_high_psa(self):
        result = _default(rotatable_bonds=5, psa_A2=150.0)
        assert result.veber_pass is False

    def test_egan_pass(self):
        result = _default(logp=2.0, psa_A2=80.0)
        assert result.egan_pass is True

    def test_ro3_pass_for_fragment(self):
        result = _default(mw_Da=200.0, logp=1.0, hbd=1, hba=2)
        assert result.ro3_pass is True

    def test_ro3_fail_large_molecule(self):
        result = _default(mw_Da=400.0, logp=4.0, hbd=4, hba=8)
        assert result.ro3_pass is False


class TestBCSClass:
    def test_bcs_class_in_valid_set(self):
        result = _default()
        assert result.bcs_class in _VALID_BCS

    def test_bcs_1_low_logp_low_mw(self):
        # high sol (logP<=2) + high perm (logP>=1 and MW<=450)
        result = _default(logp=1.5, mw_Da=300.0)
        assert result.bcs_class == "BCS I"

    def test_bcs_2_high_logp_low_mw(self):
        # low sol (logP>2) + high perm (logP>=1 and MW<=450)
        result = _default(logp=3.5, mw_Da=300.0)
        assert result.bcs_class == "BCS II"

    def test_bcs_3_low_logp_high_mw(self):
        # high sol (logP<=2) + low perm (MW>450)
        result = _default(logp=1.5, mw_Da=500.0)
        assert result.bcs_class == "BCS III"

    def test_bcs_4_high_logp_high_mw(self):
        # low sol + low perm
        result = _default(logp=3.5, mw_Da=500.0)
        assert result.bcs_class == "BCS IV"


class TestLogD:
    def test_neutral_logd_equals_logp(self):
        result = _default(logp=2.5, ionization_type="neutral", pka=7.0)
        assert abs(result.logd_74 - 2.5) < 1e-9

    def test_acid_logd_below_logp_when_ionized(self):
        # acid pka=4.0 at pH 7.4 → mostly ionized → logD << logP
        result = _default(logp=3.0, ionization_type="acid", pka=4.0)
        assert result.logd_74 < result.logp

    def test_base_logd_below_logp_when_ionized(self):
        # base pka=10 at pH 7.4 → mostly protonated → logD << logP
        result = _default(logp=3.0, ionization_type="base", pka=10.0)
        assert result.logd_74 < result.logp

    def test_logd_clamped_lower(self):
        # Very ionized acid: logP=-4 → logD should be clamped to -5
        result = _default(logp=-4.0, ionization_type="acid", pka=3.0)
        assert result.logd_74 >= -5.0


class TestCNSMPO:
    def test_high_cns_mpo_for_good_cns_drug(self):
        # MW=300, logP=2, HBD=0, PSA=40, neutral → should score >4
        result = _default(mw_Da=300.0, logp=2.0, hbd=0, psa_A2=40.0, ionization_type="neutral")
        assert result.cns_mpo_score > 4.0

    def test_low_cns_mpo_for_poor_cns_drug(self):
        # MW=600, logP=6, HBD=5, PSA=150 → should score low
        result = _default(mw_Da=600.0, logp=6.0, hbd=5, psa_A2=150.0)
        assert result.cns_mpo_score < 3.0


class TestDevelopmentRisk:
    def test_risk_in_valid_set(self):
        result = _default()
        assert result.development_risk in _VALID_RISK

    def test_low_risk_for_drug_like(self):
        # All rules pass → score >=70 → low risk
        result = _default(
            mw_Da=300.0, logp=2.0, hbd=0, hba=5, psa_A2=40.0, rotatable_bonds=5
        )
        assert result.development_risk == "low"

    def test_high_risk_for_poor_candidate(self):
        result = _default(
            mw_Da=700.0, logp=7.0, hbd=8, hba=15, psa_A2=200.0, rotatable_bonds=15
        )
        assert result.development_risk == "high"


class TestNotes:
    def test_notes_nonempty(self):
        result = _default()
        assert isinstance(result.notes, str) and len(result.notes) > 0


class TestCompare:
    def test_compare_returns_sorted(self):
        candidates = [
            dict(drug_name="A", mw_Da=700.0, logp=7.0, hbd=8, hba=15,
                 psa_A2=200.0, rotatable_bonds=15),
            dict(drug_name="B", mw_Da=300.0, logp=2.0, hbd=0, hba=5,
                 psa_A2=40.0, rotatable_bonds=5),
            dict(drug_name="C", mw_Da=450.0, logp=3.0, hbd=2, hba=7,
                 psa_A2=90.0, rotatable_bonds=8),
        ]
        profiles = compare_candidates(candidates)
        scores = [p.drug_likeness_score for p in profiles]
        assert scores == sorted(scores, reverse=True)

    def test_compare_returns_list(self):
        candidates = [
            dict(drug_name="X", mw_Da=300.0, logp=2.0, hbd=1, hba=4,
                 psa_A2=60.0, rotatable_bonds=4),
        ]
        result = compare_candidates(candidates)
        assert isinstance(result, list)
        assert len(result) == 1


class TestValidation:
    def test_mw_zero_raises(self):
        with pytest.raises(ValueError, match="mw_Da"):
            _default(mw_Da=0.0)

    def test_hbd_negative_raises(self):
        with pytest.raises(ValueError, match="hbd"):
            _default(hbd=-1)

    def test_hba_negative_raises(self):
        with pytest.raises(ValueError, match="hba"):
            _default(hba=-1)

    def test_psa_negative_raises(self):
        with pytest.raises(ValueError, match="psa_A2"):
            _default(psa_A2=-1.0)

    def test_rot_bonds_negative_raises(self):
        with pytest.raises(ValueError, match="rotatable_bonds"):
            _default(rotatable_bonds=-1)

    def test_invalid_ionization_type_raises(self):
        with pytest.raises(ValueError, match="ionization_type"):
            _default(ionization_type="zwitterion")

    def test_pka_above_14_raises(self):
        with pytest.raises(ValueError, match="pka"):
            _default(pka=15.0)
