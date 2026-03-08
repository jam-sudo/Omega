"""Tests for amorphous solid dispersion (ASD) performance prediction — Phase 1014."""

import pytest

from omega_pbpk.prediction.amorphous_solid_dispersion import (
    _VALID_POLYMERS,
    ASDResult,
    predict_asd_performance,
    screen_asd_polymers,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run_default(**kwargs):
    defaults = dict(
        drug_name="TestDrug",
        polymer_name="HPMC-AS",
        drug_loading_pct=20.0,
        logp=3.0,
        drug_pka=7.0,
        ionization_type="neutral",
    )
    defaults.update(kwargs)
    return predict_asd_performance(**defaults)


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------


def test_returns_asd_result():
    result = _run_default()
    assert isinstance(result, ASDResult)


def test_drug_name_preserved():
    result = _run_default(drug_name="Ibuprofen")
    assert result.drug_name == "Ibuprofen"


def test_polymer_name_preserved():
    result = _run_default(polymer_name="PVP-VA")
    assert result.polymer_name == "PVP-VA"


# ---------------------------------------------------------------------------
# Solubility enhancement
# ---------------------------------------------------------------------------


def test_asd_solubility_above_intrinsic():
    result = _run_default()
    assert result.asd_apparent_solubility_mg_mL > result.intrinsic_crystalline_solubility_mg_mL


def test_supersaturation_ratio_above_one():
    result = _run_default()
    assert result.supersaturation_ratio > 1.0


def test_spring_factor_above_one():
    result = _run_default()
    assert result.spring_factor > 1.0


# ---------------------------------------------------------------------------
# Parachute factor
# ---------------------------------------------------------------------------


def test_parachute_factor_in_bounds():
    result = _run_default()
    assert 0.0 < result.parachute_factor <= 1.0


def test_hpmc_as_parachute_factor():
    result = _run_default(polymer_name="HPMC-AS")
    assert result.parachute_factor == pytest.approx(0.85)


def test_pvp_parachute_factor():
    result = _run_default(polymer_name="PVP")
    assert result.parachute_factor == pytest.approx(0.55)


# ---------------------------------------------------------------------------
# Crystallization / duration
# ---------------------------------------------------------------------------


def test_crystallization_time_positive():
    result = _run_default()
    assert result.crystallization_induction_time_min > 0.0


def test_spring_parachute_duration_positive():
    result = _run_default()
    assert result.spring_parachute_duration_min > 0.0


def test_spring_parachute_duration_greater_than_induction():
    result = _run_default()
    assert result.spring_parachute_duration_min > result.crystallization_induction_time_min


# ---------------------------------------------------------------------------
# Bioavailability enhancement
# ---------------------------------------------------------------------------


def test_bioavailability_enhancement_above_one():
    result = _run_default()
    assert result.bioavailability_enhancement_factor > 1.0


def test_bioavailability_enhancement_capped_at_five():
    # Very high solubility boost polymer at low loading
    result = _run_default(polymer_name="HPMC-AS", drug_loading_pct=5.0, logp=0.0)
    assert result.bioavailability_enhancement_factor <= 5.0


# ---------------------------------------------------------------------------
# High logP → lower intrinsic solubility
# ---------------------------------------------------------------------------


def test_high_logp_lower_intrinsic_solubility():
    r_low = _run_default(logp=1.0)
    r_high = _run_default(logp=5.0)
    assert (
        r_high.intrinsic_crystalline_solubility_mg_mL < r_low.intrinsic_crystalline_solubility_mg_mL
    )


# ---------------------------------------------------------------------------
# Drug loading effect
# ---------------------------------------------------------------------------


def test_high_loading_lower_spring_factor():
    r_low = _run_default(drug_loading_pct=10.0)
    r_high = _run_default(drug_loading_pct=80.0)
    assert r_high.spring_factor < r_low.spring_factor


# ---------------------------------------------------------------------------
# Recommended polymer
# ---------------------------------------------------------------------------


def test_recommended_polymer_is_string():
    result = _run_default()
    assert isinstance(result.recommended_polymer, str)


def test_recommended_polymer_in_valid_set():
    result = _run_default()
    assert result.recommended_polymer in _VALID_POLYMERS


def test_recommended_polymer_acid():
    result = _run_default(ionization_type="acid")
    assert result.recommended_polymer == "HPMC-AS"


def test_recommended_polymer_base():
    result = _run_default(ionization_type="base")
    assert result.recommended_polymer == "PVP-VA"


def test_recommended_polymer_neutral():
    result = _run_default(ionization_type="neutral")
    assert result.recommended_polymer == "Soluplus"


# ---------------------------------------------------------------------------
# screen_asd_polymers
# ---------------------------------------------------------------------------


def test_screen_returns_six_results():
    results = screen_asd_polymers("Drug", 20.0, 3.0, 7.0, "neutral")
    assert len(results) == 6


def test_screen_sorted_descending():
    results = screen_asd_polymers("Drug", 20.0, 3.0, 7.0, "neutral")
    bafs = [r.bioavailability_enhancement_factor for r in results]
    assert bafs == sorted(bafs, reverse=True)


def test_screen_all_are_asd_results():
    results = screen_asd_polymers("Drug", 20.0, 3.0, 7.0, "neutral")
    for r in results:
        assert isinstance(r, ASDResult)


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_invalid_polymer_raises():
    with pytest.raises(ValueError, match="polymer_name"):
        predict_asd_performance("X", polymer_name="UnknownPolymer", drug_loading_pct=20.0)


def test_invalid_loading_zero_raises():
    with pytest.raises(ValueError, match="drug_loading_pct"):
        predict_asd_performance("X", polymer_name="HPMC-AS", drug_loading_pct=0.0)


def test_invalid_loading_100_raises():
    with pytest.raises(ValueError, match="drug_loading_pct"):
        predict_asd_performance("X", polymer_name="HPMC-AS", drug_loading_pct=100.0)


def test_invalid_ionization_raises():
    with pytest.raises(ValueError, match="ionization_type"):
        predict_asd_performance(
            "X", polymer_name="HPMC-AS", drug_loading_pct=20.0, ionization_type="zwitterion"
        )
