"""Tests for mechanistic renal PBPK module."""

import math

import numpy as np
import pytest

from omega_pbpk.core.kidney_pk import KidneyPKResult, predict_renal_pk

# ---------------------------------------------------------------------------
# Synthetic data
# ---------------------------------------------------------------------------
KE = 0.1
VD = 50.0
DOSE = 200.0
T = np.linspace(0, 24, 97)
CP = np.maximum((DOSE / VD) * np.exp(-KE * T), 0.0)

BASE = dict(
    plasma_times=list(T),
    plasma_conc=list(CP),
    fup=0.8,
    gfr_mL_per_min=120.0,
    logP=0.0,
    secretion_vmax_mg_h=0.0,
    secretion_km_mg_L=1.0,
    urine_flow_mL_h=60.0,
    dose_mg=DOSE,
)


# ---------------------------------------------------------------------------
# Dataclass fields
# ---------------------------------------------------------------------------
class TestKidneyPKResult:
    def test_dataclass_fields(self):
        fields = {
            "cl_filtration_L_per_h",
            "cl_secretion_mean_L_per_h",
            "cl_renal_L_per_h",
            "reabsorption_fraction",
            "fe_urine",
            "urine_rate_mg_h",
            "urine_conc_mg_L",
            "urine_cumulative_mg",
            "gfr_mL_per_min",
            "secretion_vmax_mg_h",
            "secretion_km_mg_L",
        }
        result = predict_renal_pk(**BASE)
        for f in fields:
            assert hasattr(result, f), f"Missing field: {f}"
        assert len(fields) == 11


# ---------------------------------------------------------------------------
# Filtration-only tests
# ---------------------------------------------------------------------------
@pytest.mark.integration
class TestFiltrationOnly:
    def test_returns_result_type(self):
        result = predict_renal_pk(**BASE)
        assert isinstance(result, KidneyPKResult)

    def test_cl_filtration_formula(self):
        result = predict_renal_pk(**BASE)
        expected = 120.0 * 0.06 * 0.8  # GFR * 0.06 L/mL/min→L/h * fup
        assert result.cl_filtration_L_per_h == pytest.approx(expected, rel=1e-6)

    def test_no_secretion_cl_zero(self):
        result = predict_renal_pk(**BASE)
        assert result.cl_secretion_mean_L_per_h == pytest.approx(0.0)

    def test_hydrophilic_low_reabsorption(self):
        result = predict_renal_pk(**BASE)  # logP=0
        assert result.reabsorption_fraction < 0.3

    def test_urine_rate_length(self):
        result = predict_renal_pk(**BASE)
        assert len(result.urine_rate_mg_h) == len(T)

    def test_urine_rate_nonneg(self):
        result = predict_renal_pk(**BASE)
        assert all(r >= 0 for r in result.urine_rate_mg_h)

    def test_urine_cumulative_positive(self):
        result = predict_renal_pk(**BASE)
        assert result.urine_cumulative_mg > 0

    def test_fe_urine_in_range(self):
        result = predict_renal_pk(**BASE)
        assert 0.0 <= result.fe_urine <= 1.0

    def test_urine_conc_nonneg(self):
        result = predict_renal_pk(**BASE)
        assert all(c >= 0 for c in result.urine_conc_mg_L)


# ---------------------------------------------------------------------------
# Active secretion tests
# ---------------------------------------------------------------------------
@pytest.mark.integration
class TestActiveSecretion:
    def test_secretion_increases_cl_renal(self):
        base = predict_renal_pk(**BASE)
        sec = predict_renal_pk(**{**BASE, "secretion_vmax_mg_h": 50.0})
        assert sec.cl_renal_L_per_h > base.cl_renal_L_per_h

    def test_secretion_cl_positive(self):
        sec = predict_renal_pk(**{**BASE, "secretion_vmax_mg_h": 50.0})
        assert sec.cl_secretion_mean_L_per_h > 0

    def test_secretion_increases_urine(self):
        base = predict_renal_pk(**BASE)
        sec = predict_renal_pk(**{**BASE, "secretion_vmax_mg_h": 50.0})
        assert sec.urine_cumulative_mg > base.urine_cumulative_mg


# ---------------------------------------------------------------------------
# Reabsorption tests
# ---------------------------------------------------------------------------
@pytest.mark.integration
class TestReabsorption:
    def test_lipophilic_high_reabsorption(self):
        result = predict_renal_pk(**{**BASE, "logP": 3})
        assert result.reabsorption_fraction > 0.7

    def test_hydrophilic_low_reabsorption(self):
        result = predict_renal_pk(**{**BASE, "logP": -2})
        assert result.reabsorption_fraction < 0.2

    def test_reabsorption_reduces_urine(self):
        hydro = predict_renal_pk(**{**BASE, "logP": -2})
        lipo = predict_renal_pk(**{**BASE, "logP": 3})
        assert lipo.urine_cumulative_mg < hydro.urine_cumulative_mg


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------
@pytest.mark.integration
class TestEdgeCases:
    def test_no_dose_fe_nan(self):
        result = predict_renal_pk(**{**BASE, "dose_mg": 0})
        assert math.isnan(result.fe_urine)

    def test_high_fup_increases_filtration(self):
        lo = predict_renal_pk(**{**BASE, "fup": 0.5})
        hi = predict_renal_pk(**{**BASE, "fup": 1.0})
        assert hi.cl_filtration_L_per_h > lo.cl_filtration_L_per_h

    def test_cl_renal_positive(self):
        result = predict_renal_pk(**BASE)
        assert result.cl_renal_L_per_h > 0

    def test_invalid_urine_flow_raises(self):
        with pytest.raises(ValueError):
            predict_renal_pk(**{**BASE, "urine_flow_mL_h": 0})
