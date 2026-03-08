"""Tests for Phase 425 — drug_transporter_efflux module."""

import pytest

from omega_pbpk.clinical.drug_transporter_efflux import (
    TransporterEffluxResult,
    assess_transporter_efflux,
    compare_efflux_scenarios,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _base(**kwargs):
    defaults = dict(
        drug_name="TestDrug",
        transporter="P-gp",
        logp=2.5,
        mw_Da=400.0,
        in_vitro_efflux_ratio=5.0,
        fu_brain=0.05,
        ic50_uM=100.0,
    )
    defaults.update(kwargs)
    return assess_transporter_efflux(**defaults)


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------


def test_return_type():
    result = _base()
    assert isinstance(result, TransporterEffluxResult)


# ---------------------------------------------------------------------------
# Preserved fields
# ---------------------------------------------------------------------------


def test_drug_name_preserved():
    result = _base(drug_name="Digoxin")
    assert result.drug_name == "Digoxin"


def test_transporter_preserved():
    for t in ("P-gp", "BCRP", "MRP2", "combined"):
        result = _base(transporter=t)
        assert result.transporter == t


# ---------------------------------------------------------------------------
# Substrate classification
# ---------------------------------------------------------------------------


def test_efflux_ratio_high():
    result = _base(in_vitro_efflux_ratio=15.0)
    assert result.substrate_class == "high"


def test_efflux_ratio_moderate():
    result = _base(in_vitro_efflux_ratio=5.0)
    assert result.substrate_class == "moderate"


def test_efflux_ratio_low():
    result = _base(in_vitro_efflux_ratio=1.7)
    assert result.substrate_class == "low"


def test_efflux_ratio_non_substrate():
    result = _base(in_vitro_efflux_ratio=1.2)
    assert result.substrate_class == "non_substrate"


# ---------------------------------------------------------------------------
# CNS penetration
# ---------------------------------------------------------------------------


def test_high_efflux_low_cns():
    # High efflux ratio should result in low CNS penetration
    result = _base(in_vitro_efflux_ratio=20.0, fu_brain=0.05)
    assert result.cns_kp_uu < 0.1


def test_cns_kp_uu_between_0_and_1():
    for ratio in (1.0, 2.0, 5.0, 20.0, 50.0):
        result = _base(in_vitro_efflux_ratio=ratio, fu_brain=0.1)
        assert 0.0 < result.cns_kp_uu <= 1.0


def test_low_efflux_high_cns():
    result = _base(in_vitro_efflux_ratio=1.0, fu_brain=0.5)
    assert result.cns_kp_uu > 0.3


# ---------------------------------------------------------------------------
# Gut extraction & f_oral_reduction_factor
# ---------------------------------------------------------------------------


def test_high_efflux_high_gut_extraction():
    result = _base(in_vitro_efflux_ratio=20.0)
    assert result.gut_extraction_pct > 50.0


def test_gut_extraction_capped_at_80():
    result = _base(in_vitro_efflux_ratio=100.0)
    assert result.gut_extraction_pct <= 80.0


def test_f_oral_reduction_between_0_and_1():
    for ratio in (1.0, 2.0, 5.0, 20.0):
        result = _base(in_vitro_efflux_ratio=ratio)
        assert 0.0 <= result.f_oral_reduction_factor <= 1.0


def test_non_substrate_minimal_extraction():
    result = _base(in_vitro_efflux_ratio=1.0)
    assert result.gut_extraction_pct == 0.0
    assert result.f_oral_reduction_factor == 1.0


# ---------------------------------------------------------------------------
# Inhibitor classification
# ---------------------------------------------------------------------------


def test_low_ic50_strong_inhibitor():
    result = _base(ic50_uM=0.05)
    assert result.inhibitor_class == "strong"


def test_moderate_ic50_moderate_inhibitor():
    result = _base(ic50_uM=0.5)
    assert result.inhibitor_class == "moderate"


def test_high_ic50_non_inhibitor():
    result = _base(ic50_uM=50.0)
    assert result.inhibitor_class == "non_inhibitor"


# ---------------------------------------------------------------------------
# Combined transporter
# ---------------------------------------------------------------------------


def test_combined_transporter_higher_efflux():
    base_result = _base(transporter="P-gp", in_vitro_efflux_ratio=5.0)
    combined_result = _base(transporter="combined", in_vitro_efflux_ratio=5.0)
    assert combined_result.efflux_ratio > base_result.efflux_ratio


# ---------------------------------------------------------------------------
# compare_efflux_scenarios
# ---------------------------------------------------------------------------


def test_compare_efflux_scenarios_length():
    scenarios = [("P-gp", 2.0), ("BCRP", 8.0), ("MRP2", 15.0)]
    results = compare_efflux_scenarios("Drug", 2.5, 400.0, scenarios)
    assert len(results) == 3


def test_compare_efflux_scenarios_sorted_descending():
    scenarios = [("P-gp", 15.0), ("BCRP", 2.0), ("MRP2", 5.0)]
    results = compare_efflux_scenarios("Drug", 2.5, 400.0, scenarios)
    kp_uu_values = [r.cns_kp_uu for r in results]
    assert kp_uu_values == sorted(kp_uu_values, reverse=True)


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_invalid_transporter():
    with pytest.raises(ValueError, match="transporter"):
        assess_transporter_efflux(
            drug_name="X",
            transporter="Unknown",
            logp=2.0,
            mw_Da=300.0,
            in_vitro_efflux_ratio=5.0,
        )


def test_efflux_ratio_zero_raises():
    with pytest.raises(ValueError):
        assess_transporter_efflux(
            drug_name="X",
            transporter="P-gp",
            logp=2.0,
            mw_Da=300.0,
            in_vitro_efflux_ratio=0.0,
        )


def test_efflux_ratio_negative_raises():
    with pytest.raises(ValueError):
        assess_transporter_efflux(
            drug_name="X",
            transporter="P-gp",
            logp=2.0,
            mw_Da=300.0,
            in_vitro_efflux_ratio=-1.0,
        )


def test_fu_brain_zero_raises():
    with pytest.raises(ValueError):
        _base(fu_brain=0.0)


def test_fu_brain_greater_than_one_raises():
    with pytest.raises(ValueError):
        _base(fu_brain=1.5)
