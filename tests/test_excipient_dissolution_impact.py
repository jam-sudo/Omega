"""
Tests for Phase 450: Excipient Dissolution Impact Prediction
"""

import pytest

from omega_pbpk.prediction.excipient_dissolution_impact import (
    EXCIPIENTS,
    ExcipientDissolutionResult,
    predict_excipient_dissolution,
    screen_excipients_dissolution,
)

# ─── Helper ───────────────────────────────────────────────────────────────────


def pred(excipient="SDS", conc=0.5, drug_logp=2.0, bcs=2, drug="TestDrug"):
    return predict_excipient_dissolution(
        drug_name=drug,
        excipient_name=excipient,
        excipient_concentration_pct=conc,
        drug_logp=drug_logp,
        drug_bcs_class=bcs,
    )


# ─── Return type ──────────────────────────────────────────────────────────────


def test_return_type():
    result = pred()
    assert isinstance(result, ExcipientDissolutionResult)


def test_frozen_dataclass():
    """ExcipientDissolutionResult should be frozen (immutable)."""
    result = pred()
    with pytest.raises((AttributeError, TypeError)):
        result.drug_name = "Modified"


# ─── Field preservation ───────────────────────────────────────────────────────


def test_drug_name_preserved():
    result = pred(drug="Itraconazole")
    assert result.drug_name == "Itraconazole"


def test_excipient_name_preserved():
    result = pred(excipient="PVP_K30")
    assert result.excipient_name == "PVP_K30"


def test_excipient_concentration_preserved():
    result = pred(excipient="HPMC_AS", conc=3.0)
    assert result.excipient_concentration_pct == pytest.approx(3.0)


# ─── Enhancement factor ───────────────────────────────────────────────────────


def test_enhancement_factor_at_least_one():
    for exc in EXCIPIENTS:
        result = pred(excipient=exc, conc=EXCIPIENTS[exc]["optimal_pct"])
        assert result.dissolution_enhancement_factor >= 1.0, f"Failed for {exc}"


def test_cyclodextrin_highest_max_factor():
    """cyclodextrin_beta has max_factor=10, highest in database."""
    result_cd = pred(excipient="cyclodextrin_beta", conc=10.0)
    result_sds = pred(excipient="SDS", conc=0.5)
    assert result_cd.dissolution_enhancement_factor > result_sds.dissolution_enhancement_factor


def test_higher_concentration_higher_enhancement():
    """Hill equation: more concentration → more enhancement (up to saturation)."""
    r_low = pred(excipient="PVP_K30", conc=1.0)
    r_high = pred(excipient="PVP_K30", conc=5.0)
    assert r_high.dissolution_enhancement_factor > r_low.dissolution_enhancement_factor


def test_enhancement_monotone_with_concentration():
    """Test at several concentrations that factor is monotonically non-decreasing."""
    concs = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    factors = [pred(excipient="HPMC_AS", conc=c).dissolution_enhancement_factor for c in concs]
    for i in range(1, len(factors)):
        assert factors[i] >= factors[i - 1]


# ─── t50 reduction ────────────────────────────────────────────────────────────


def test_t50_reduction_between_zero_and_ninety_five():
    for exc in EXCIPIENTS:
        result = pred(excipient=exc, conc=EXCIPIENTS[exc]["optimal_pct"])
        assert 0.0 <= result.t50_reduction_pct <= 95.0, f"Failed for {exc}"


# ─── Supersaturation ──────────────────────────────────────────────────────────


def test_supersaturation_maintenance_positive():
    result = pred(excipient="HPMC_E5", conc=1.0)
    assert result.supersaturation_maintenance_h > 0.0


def test_supersaturation_clamped_max_24():
    result = pred(excipient="HPMC_AS", conc=100.0)
    assert result.supersaturation_maintenance_h <= 24.0


# ─── Spring / parachute ───────────────────────────────────────────────────────


def test_hpmc_as_parachute_effect_true():
    result = pred(excipient="HPMC_AS")
    assert result.parachute_effect is True


def test_sds_spring_effect_true():
    result = pred(excipient="SDS")
    assert result.spring_effect is True


def test_pvp_k30_spring_effect_false():
    result = pred(excipient="PVP_K30")
    assert result.spring_effect is False


def test_hpmc_e5_parachute_effect_true():
    result = pred(excipient="HPMC_E5")
    assert result.parachute_effect is True


# ─── Mechanism ────────────────────────────────────────────────────────────────


def test_mechanism_preserved_sds():
    result = pred(excipient="SDS")
    assert result.mechanism == "wetting"


def test_mechanism_preserved_pvp():
    result = pred(excipient="PVP_K30")
    assert result.mechanism == "nucleation_inhibition"


def test_mechanism_preserved_cyclodextrin():
    result = pred(excipient="cyclodextrin_beta", conc=10.0)
    assert result.mechanism == "complexation"


# ─── Optimal concentration ────────────────────────────────────────────────────


def test_optimal_concentration_from_database():
    result = pred(excipient="cremophor_EL", conc=5.0)
    assert result.optimal_concentration_pct == pytest.approx(5.0)


# ─── screen_excipients_dissolution ───────────────────────────────────────────


def test_screen_returns_eight_results():
    results = screen_excipients_dissolution("TestDrug")
    assert len(results) == 8


def test_screen_sorted_descending():
    results = screen_excipients_dissolution("TestDrug")
    factors = [r.dissolution_enhancement_factor for r in results]
    assert factors == sorted(factors, reverse=True)


def test_screen_cyclodextrin_near_top():
    """cyclodextrin_beta at optimal pct should be near the top."""
    results = screen_excipients_dissolution("TestDrug", drug_bcs_class=2)
    names = [r.excipient_name for r in results]
    # cyclodextrin with max_factor=10 should be first
    assert names[0] == "cyclodextrin_beta"


# ─── Validation errors ────────────────────────────────────────────────────────


def test_invalid_excipient_raises_value_error():
    with pytest.raises(ValueError, match="not found"):
        predict_excipient_dissolution("Drug", "unknown_polymer", 1.0)


def test_concentration_zero_raises_value_error():
    with pytest.raises(ValueError, match="excipient_concentration_pct"):
        predict_excipient_dissolution("Drug", "SDS", 0.0)


def test_concentration_negative_raises_value_error():
    with pytest.raises(ValueError, match="excipient_concentration_pct"):
        predict_excipient_dissolution("Drug", "SDS", -1.0)
