"""Tests for Drug Lyophilization Stability Predictor — Phase 845."""

import pytest

from omega_pbpk.prediction.lyophilization_stability import (
    LyophilizationResult,
    optimize_lyophilization_cycle,
    predict_lyophilization_stability,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_result(
    excipient: str = "sucrose",
    drug_loading_pct: float = 20.0,
    protein_content_pct: float = 30.0,
    drug_mw_kDa: float = 10.0,
) -> LyophilizationResult:
    return predict_lyophilization_stability(
        drug_name="TestDrug",
        drug_loading_pct=drug_loading_pct,
        protein_content_pct=protein_content_pct,
        excipient_name=excipient,
        drug_mw_kDa=drug_mw_kDa,
    )


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------


def test_return_type():
    result = _make_result()
    assert isinstance(result, LyophilizationResult)


def test_frozen_dataclass():
    result = _make_result()
    with pytest.raises((AttributeError, TypeError)):
        result.drug_name = "Other"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Tg_prime from excipient database
# ---------------------------------------------------------------------------


def test_tg_prime_sucrose():
    result = _make_result(excipient="sucrose")
    assert result.tg_prime_celsius == pytest.approx(-32.0)


def test_tg_prime_trehalose():
    result = _make_result(excipient="trehalose")
    assert result.tg_prime_celsius == pytest.approx(-28.0)


def test_tg_prime_mannitol():
    result = _make_result(excipient="mannitol")
    assert result.tg_prime_celsius == pytest.approx(-25.0)


def test_tg_prime_pvp():
    result = _make_result(excipient="PVP")
    assert result.tg_prime_celsius == pytest.approx(-20.0)


def test_tg_prime_dextran():
    result = _make_result(excipient="dextran")
    assert result.tg_prime_celsius == pytest.approx(-15.0)


# ---------------------------------------------------------------------------
# Collapse temperature = Tg' + 2
# ---------------------------------------------------------------------------


def test_collapse_temperature_sucrose():
    result = _make_result(excipient="sucrose")
    assert result.collapse_temperature_celsius == pytest.approx(result.tg_prime_celsius + 2.0)


def test_collapse_temperature_trehalose():
    result = _make_result(excipient="trehalose")
    assert result.collapse_temperature_celsius == pytest.approx(result.tg_prime_celsius + 2.0)


def test_collapse_temperature_pvp():
    result = _make_result(excipient="PVP")
    assert result.collapse_temperature_celsius == pytest.approx(result.tg_prime_celsius + 2.0)


# ---------------------------------------------------------------------------
# Primary drying temp = collapse_temperature - 5
# ---------------------------------------------------------------------------


def test_primary_drying_temp_sucrose():
    result = _make_result(excipient="sucrose")
    assert result.primary_drying_temp_celsius == pytest.approx(
        result.collapse_temperature_celsius - 5.0
    )


def test_primary_drying_temp_trehalose():
    result = _make_result(excipient="trehalose")
    assert result.primary_drying_temp_celsius == pytest.approx(
        result.collapse_temperature_celsius - 5.0
    )


# ---------------------------------------------------------------------------
# Shelf life > 0
# ---------------------------------------------------------------------------


def test_shelf_life_positive():
    for exc in ["sucrose", "trehalose", "mannitol", "PVP", "dextran"]:
        r = _make_result(excipient=exc)
        assert r.shelf_life_months > 0.0, f"shelf_life <= 0 for {exc}"


# ---------------------------------------------------------------------------
# ICH compliance = shelf_life >= 24 months
# ---------------------------------------------------------------------------


def test_ich_compliance_trehalose():
    result = _make_result(excipient="trehalose", drug_loading_pct=20.0, protein_content_pct=10.0)
    assert result.ich_stability_compliant == (result.shelf_life_months >= 24.0)


def test_ich_compliance_sucrose_low_loading():
    result = _make_result(excipient="sucrose", drug_loading_pct=20.0, protein_content_pct=10.0)
    assert result.ich_stability_compliant == (result.shelf_life_months >= 24.0)


def test_ich_compliance_high_loading_penalty():
    # High drug loading (>50%) triggers 0.7 multiplier on sucrose base=24 → 16.8 months < 24
    result = predict_lyophilization_stability(
        drug_name="HighLoad",
        drug_loading_pct=60.0,
        protein_content_pct=10.0,
        excipient_name="sucrose",
    )
    assert result.shelf_life_months < 24.0
    assert not result.ich_stability_compliant


# ---------------------------------------------------------------------------
# Trehalose → longer shelf life than sucrose (same conditions)
# ---------------------------------------------------------------------------


def test_trehalose_longer_shelf_life_than_sucrose():
    trehalose = _make_result(excipient="trehalose", drug_loading_pct=20.0, protein_content_pct=10.0)
    sucrose = _make_result(excipient="sucrose", drug_loading_pct=20.0, protein_content_pct=10.0)
    assert trehalose.shelf_life_months >= sucrose.shelf_life_months


# ---------------------------------------------------------------------------
# optimize_lyophilization_cycle: 5 results, sorted by shelf_life descending
# ---------------------------------------------------------------------------


def test_optimize_returns_five_results():
    results = optimize_lyophilization_cycle(
        drug_name="TestDrug",
        drug_loading_pct=20.0,
        protein_content_pct=30.0,
    )
    assert len(results) == 5


def test_optimize_sorted_descending():
    results = optimize_lyophilization_cycle(
        drug_name="TestDrug",
        drug_loading_pct=20.0,
        protein_content_pct=30.0,
    )
    shelf_lives = [r.shelf_life_months for r in results]
    assert shelf_lives == sorted(shelf_lives, reverse=True)


def test_optimize_all_lyophilization_result_type():
    results = optimize_lyophilization_cycle(
        drug_name="BioDrug",
        drug_loading_pct=10.0,
        protein_content_pct=80.0,
        drug_mw_kDa=150.0,
    )
    for r in results:
        assert isinstance(r, LyophilizationResult)


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_validation_drug_loading_zero():
    with pytest.raises(ValueError, match="drug_loading_pct"):
        predict_lyophilization_stability("Drug", 0.0, 30.0, "sucrose")


def test_validation_drug_loading_100():
    with pytest.raises(ValueError, match="drug_loading_pct"):
        predict_lyophilization_stability("Drug", 100.0, 30.0, "sucrose")


def test_validation_protein_content_negative():
    with pytest.raises(ValueError, match="protein_content_pct"):
        predict_lyophilization_stability("Drug", 20.0, -5.0, "sucrose")


def test_validation_protein_content_over_100():
    with pytest.raises(ValueError, match="protein_content_pct"):
        predict_lyophilization_stability("Drug", 20.0, 105.0, "sucrose")
