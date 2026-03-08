"""Tests for Phase 556 - Blood-Brain Barrier Permeability Prediction."""

import pytest

from omega_pbpk.prediction.bbb_permeability import (
    BBBPermeabilityResult,
    predict_bbb_permeability,
    screen_bbb_permeability,
)


def _predict(**overrides):
    defaults = dict(drug_name="TestDrug", logP=2.0, mw_Da=300.0, hbd=1, psa_A2=60.0)
    kw = {**defaults, **overrides}
    return predict_bbb_permeability(**kw)


# 1. Return type
def test_return_type():
    r = _predict()
    assert isinstance(r, BBBPermeabilityResult)


# 2. log_PS clamped to [-6, 1]
def test_log_ps_range():
    r = _predict()
    assert -6.0 <= r.log_ps_cm3_per_s <= 1.0


# 3. PS = 10^log_PS
def test_ps_equals_10_to_log_ps():
    r = _predict()
    assert r.ps_cm3_per_s == pytest.approx(10**r.log_ps_cm3_per_s, rel=1e-6)


# 4. Low MW/PSA/HBD → higher log_PS
def test_low_descriptors_higher_logps():
    r_easy = _predict(logP=3.0, mw_Da=100.0, hbd=0, psa_A2=0.0)
    r_hard = _predict(logP=0.0, mw_Da=200.0, hbd=2, psa_A2=5.0)
    assert r_easy.log_ps_cm3_per_s > r_hard.log_ps_cm3_per_s


# 5. High MW/PSA → "none" or "low" class
def test_high_mw_psa_low_penetration():
    r = _predict(logP=-2.0, mw_Da=800.0, hbd=8, psa_A2=200.0)
    assert r.cns_penetration_class in ("low", "none")


# 6. cns_mpo_score in [0, 6]
def test_mpo_range():
    r = _predict()
    assert 0 <= r.cns_mpo_score <= 6


# 7. Kp_brain > 0
def test_kp_brain_positive():
    r = _predict()
    assert r.kp_brain > 0


# 8. fu_brain in [0.01, 1.0]
def test_fu_brain_range():
    r = _predict()
    assert 0.01 <= r.fraction_unbound_brain <= 1.0


# 9. screen sorted descending by log_ps
def test_screen_sorted_descending():
    compounds = [
        dict(drug_name="A", logP=1.0, mw_Da=400.0, hbd=3, psa_A2=100.0),
        dict(drug_name="B", logP=3.0, mw_Da=200.0, hbd=0, psa_A2=30.0),
        dict(drug_name="C", logP=0.0, mw_Da=350.0, hbd=2, psa_A2=80.0),
    ]
    results = screen_bbb_permeability(compounds)
    logps = [r.log_ps_cm3_per_s for r in results]
    assert logps == sorted(logps, reverse=True)


# 10. "high" class achievable
def test_high_class():
    # logPS = 0.152*5 - 0.0148*50 - 0 - 0 + 0.233 = 0.253
    r = _predict(logP=5.0, mw_Da=50.0, hbd=0, psa_A2=0.0)
    assert r.cns_penetration_class == "high"


# 11. "moderate" class achievable
def test_moderate_class():
    # logPS = 0.152*2 - 0.0148*200 - 0.139 - 0 + 0.233 = -2.562
    r = _predict(logP=2.0, mw_Da=200.0, hbd=1, psa_A2=0.0)
    assert r.cns_penetration_class == "moderate"


# 12. "low" class achievable
def test_low_class():
    # logPS = 0.152*2 - 0.0148*250 - 0.278 - 0.69 + 0.233 = -4.131
    r = _predict(logP=2.0, mw_Da=250.0, hbd=2, psa_A2=5.0)
    assert r.cns_penetration_class == "low"


# 13. "none" class achievable
def test_none_class():
    r = _predict(logP=-2.0, mw_Da=900.0, hbd=10, psa_A2=250.0)
    assert r.cns_penetration_class == "none"


# 14. Validation: mw_Da > 0
def test_invalid_mw():
    with pytest.raises(ValueError, match="mw_Da"):
        _predict(mw_Da=0)


# 15. Validation: hbd >= 0
def test_invalid_hbd():
    with pytest.raises(ValueError, match="hbd"):
        _predict(hbd=-1)


# 16. Validation: psa_A2 >= 0
def test_invalid_psa():
    with pytest.raises(ValueError, match="psa_A2"):
        _predict(psa_A2=-5.0)


# 17. drug_name preserved
def test_drug_name():
    r = _predict(drug_name="Caffeine")
    assert r.drug_name == "Caffeine"


# 18. logP stored correctly
def test_logp_stored():
    r = _predict(logP=3.5)
    assert r.logP == 3.5


# 19. Higher logP → higher Kp_brain
def test_logp_kp_brain_relationship():
    r_low = _predict(logP=0.5)
    r_high = _predict(logP=4.0)
    assert r_high.kp_brain > r_low.kp_brain


# 20. Higher logP → lower fu_brain
def test_logp_fu_brain_relationship():
    r_low = _predict(logP=0.5)
    r_high = _predict(logP=4.0)
    assert r_high.fraction_unbound_brain < r_low.fraction_unbound_brain


# 21. notes non-empty
def test_notes_nonempty():
    r = _predict()
    assert len(r.notes) > 0


# 22. log_PS upper clamp
def test_log_ps_upper_clamp():
    # Very favorable properties should still clamp at 1.0
    r = _predict(logP=10.0, mw_Da=50.0, hbd=0, psa_A2=0.0)
    assert r.log_ps_cm3_per_s == 1.0


# 23. MPO perfect score
def test_mpo_perfect_score():
    # logP in [0,3], mw<=360, psa<=90, hbd<=0, pKa<=10
    r = _predict(logP=1.0, mw_Da=200.0, hbd=0, psa_A2=50.0, pKa=7.0)
    assert r.cns_mpo_score == 6.0
