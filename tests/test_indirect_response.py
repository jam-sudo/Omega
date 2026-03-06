"""Tests for indirect response PK/PD models — Types I–IV."""

import pytest

from omega_pbpk.core.indirect_response import (
    IndirectResponseResult,
    IRType,
    simulate_indirect_iv,
    simulate_indirect_response,
)

# ── Shared helpers ──────────────────────────────────────────────────────
KIN, KOUT = 1.0, 0.1  # R0 = 10
EMAX, EC50, HILL = 1.0, 1.0, 1.0

# Simple concentration profile: ramp up then wash-out
_TIMES = [float(t) for t in range(49)]  # 0..48 h
_CP = [2.0 * max(1 - t / 24.0, 0.0) for t in _TIMES]  # peaks at t=0, zero by t=24


def _run(ir_type: IRType, *, emax: float = EMAX) -> IndirectResponseResult:
    return simulate_indirect_response(
        drug_name="TestDrug",
        times_h=_TIMES,
        cp_mg_L=_CP,
        ir_type=ir_type,
        kin=KIN,
        kout=KOUT,
        emax=emax,
        ec50_mg_L=EC50,
        hill=HILL,
    )


# ── IRType enum ─────────────────────────────────────────────────────────
class TestIRType:
    def test_type_i(self):
        assert IRType.TYPE_I.value == "type_i"

    def test_type_ii(self):
        assert IRType.TYPE_II.value == "type_ii"

    def test_type_iii(self):
        assert IRType.TYPE_III.value == "type_iii"

    def test_type_iv(self):
        assert IRType.TYPE_IV.value == "type_iv"


# ── Input validation ────────────────────────────────────────────────────
class TestInputValidation:
    def test_kin_zero_raises(self):
        with pytest.raises(ValueError):
            simulate_indirect_response("D", _TIMES, _CP, IRType.TYPE_I, kin=0, kout=0.1)

    def test_kout_zero_raises(self):
        with pytest.raises(ValueError):
            simulate_indirect_response("D", _TIMES, _CP, IRType.TYPE_I, kin=1, kout=0)

    def test_ec50_zero_raises(self):
        with pytest.raises(ValueError):
            simulate_indirect_response(
                "D", _TIMES, _CP, IRType.TYPE_I, kin=1, kout=0.1, ec50_mg_L=0
            )

    def test_emax_negative_raises(self):
        with pytest.raises(ValueError):
            simulate_indirect_response("D", _TIMES, _CP, IRType.TYPE_I, kin=1, kout=0.1, emax=-1)


# ── Type I: inhibit kin → response DECREASES ────────────────────────────
class TestTypeIInhibitKin:
    def test_baseline(self):
        r = _run(IRType.TYPE_I)
        assert r.r0 == pytest.approx(KIN / KOUT)

    def test_response_decreases(self):
        r = _run(IRType.TYPE_I)
        assert r.rmin < r.r0, "Type I should reduce response below baseline"

    def test_rmax_near_baseline(self):
        r = _run(IRType.TYPE_I)
        assert r.rmax <= r.r0 * 1.01  # should not rise significantly above R0

    def test_auc_positive(self):
        r = _run(IRType.TYPE_I)
        assert r.auc_response > 0

    def test_all_response_nonneg(self):
        r = _run(IRType.TYPE_I)
        assert all(v >= 0 for v in r.response)


# ── Type II: inhibit kout → response INCREASES ──────────────────────────
class TestTypeIIInhibitKout:
    def test_response_increases(self):
        r = _run(IRType.TYPE_II)
        assert r.rmax > r.r0, "Type II should raise response above baseline"

    def test_return_to_baseline(self):
        r = _run(IRType.TYPE_II)
        assert r.return_to_baseline_h > 0

    def test_all_response_nonneg(self):
        r = _run(IRType.TYPE_II)
        assert all(v >= 0 for v in r.response)


# ── Type III: stimulate kin → response INCREASES ─────────────────────────
class TestTypeIIIStimulateKin:
    def test_response_increases(self):
        r = _run(IRType.TYPE_III)
        assert r.rmax > r.r0, "Type III should raise response above baseline"


# ── Type IV: stimulate kout → response DECREASES ────────────────────────
class TestTypeIVStimulateKout:
    def test_response_decreases(self):
        r = _run(IRType.TYPE_IV)
        assert r.rmin < r.r0, "Type IV should reduce response below baseline"


# ── Baseline (no drug effect) ───────────────────────────────────────────
class TestBaseline:
    def test_emax_zero_stays_at_r0(self):
        r = _run(IRType.TYPE_I, emax=0.0)
        assert r.r0 == pytest.approx(KIN / KOUT)
        for val in r.response:
            assert val == pytest.approx(r.r0, rel=0.01)


# ── IV convenience function ─────────────────────────────────────────────
class TestSimulateIndirectIV:
    def test_returns_result(self):
        r = simulate_indirect_iv(
            "IVDrug",
            dose_mg=100,
            cl_L_per_h=5,
            vd_L=50,
            ir_type=IRType.TYPE_III,
            kin=KIN,
            kout=KOUT,
            emax=EMAX,
            ec50_mg_L=EC50,
            hill=HILL,
            t_end_h=48,
        )
        assert isinstance(r, IndirectResponseResult)

    def test_baseline_correct(self):
        r = simulate_indirect_iv(
            "IVDrug",
            100,
            5,
            50,
            IRType.TYPE_III,
            KIN,
            KOUT,
            t_end_h=48,
        )
        assert r.r0 == pytest.approx(KIN / KOUT)

    def test_times_nonempty(self):
        r = simulate_indirect_iv(
            "IVDrug",
            100,
            5,
            50,
            IRType.TYPE_III,
            KIN,
            KOUT,
            t_end_h=48,
        )
        assert len(r.times_h) > 0

    def test_lengths_match(self):
        r = simulate_indirect_iv(
            "IVDrug",
            100,
            5,
            50,
            IRType.TYPE_III,
            KIN,
            KOUT,
            t_end_h=48,
        )
        assert len(r.response) == len(r.times_h)

    def test_rmax_positive(self):
        r = simulate_indirect_iv(
            "IVDrug",
            100,
            5,
            50,
            IRType.TYPE_III,
            KIN,
            KOUT,
            t_end_h=48,
        )
        assert r.rmax > 0


# ── Higher Emax → stronger effect ───────────────────────────────────────
class TestHigherEmaxStrongerEffect:
    def test_type_iii_higher_emax(self):
        r_low = _run(IRType.TYPE_III, emax=0.5)
        r_high = _run(IRType.TYPE_III, emax=2.0)
        assert r_high.rmax > r_low.rmax


# ── Result structure ────────────────────────────────────────────────────
class TestResultStructure:
    def test_all_fields(self):
        r = _run(IRType.TYPE_I)
        assert isinstance(r.ir_type, str)
        assert r.kin == pytest.approx(KIN)
        assert r.kout == pytest.approx(KOUT)
        assert r.ec50_mg_L == pytest.approx(EC50)
        assert isinstance(r.times_h, list)
        assert isinstance(r.response, list)
        assert isinstance(r.cp_mg_L, list)
        assert isinstance(r.r0, float)
        assert isinstance(r.rmax, float)
        assert isinstance(r.rmin, float)
        assert isinstance(r.tmax_h, float)
        assert isinstance(r.auc_response, float)
        assert isinstance(r.return_to_baseline_h, float)
        assert isinstance(r.drug_name, str)
        assert isinstance(r.emax, float)
        assert isinstance(r.hill, float)
