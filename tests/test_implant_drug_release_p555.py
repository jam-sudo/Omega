"""Tests for Phase 555 - Drug Release from Implants."""

import math

import pytest

from omega_pbpk.core.implant_drug_release_p555 import (
    ImplantReleaseResult,
    simulate_implant_release,
)

# --- Defaults ---
DEF = dict(drug_name="TestDrug", dose_mg=100.0, cl_L_per_h=1.0, vd_L=50.0)


def _run(implant_type="rod", **overrides):
    kw = {**DEF, "implant_type": implant_type, **overrides}
    return simulate_implant_release(**kw)


# 1. Return type
def test_return_type():
    r = _run("rod")
    assert isinstance(r, ImplantReleaseResult)


# 2. List lengths match
def test_list_lengths_match():
    r = _run("rod")
    n = len(r.times_h)
    assert n == len(r.c_systemic_mg_L) == len(r.cumulative_released_pct)
    assert n > 0


# 3-5. All implant types produce valid results
@pytest.mark.parametrize("itype", ["rod", "pellet", "microsphere"])
def test_implant_type_valid(itype):
    r = _run(itype)
    assert r.implant_type == itype
    assert r.cmax_mg_L > 0
    assert r.auc_mg_h_per_L > 0


# 6. Cmax > 0
def test_cmax_positive():
    r = _run("pellet")
    assert r.cmax_mg_L > 0


# 7. AUC > 0
def test_auc_positive():
    r = _run("microsphere")
    assert r.auc_mg_h_per_L > 0


# 8. Rod releases slower than microsphere initially
def test_rod_slower_than_microsphere_initially():
    r_rod = _run("rod", t_end_h=48.0)
    r_ms = _run("microsphere", t_end_h=48.0)
    # At early times, microsphere (k=0.02) releases faster
    idx = min(24, len(r_rod.cumulative_released_pct) - 1)
    assert r_ms.cumulative_released_pct[idx] > r_rod.cumulative_released_pct[idx]


# 9. t50 > 0
def test_t50_positive():
    r = _run("pellet")
    assert r.t50_released_h > 0


# 10. t90 > t50
def test_t90_gt_t50():
    r = _run("pellet", t_end_h=1500.0)
    assert r.t90_released_h > r.t50_released_h


# 11. Dose linearity: 2x dose → ~2x AUC
def test_dose_linearity():
    r1 = _run("pellet", dose_mg=50.0)
    r2 = _run("pellet", dose_mg=100.0)
    ratio = r2.auc_mg_h_per_L / r1.auc_mg_h_per_L
    assert 1.8 < ratio < 2.2


# 12. Cumulative released starts near 0
def test_cumulative_starts_at_zero():
    r = _run("rod")
    assert r.cumulative_released_pct[0] == pytest.approx(0.0, abs=0.01)


# 13. Cumulative released ends near 100 (for long enough sim)
def test_cumulative_ends_near_100():
    r = _run("pellet", t_end_h=2000.0)
    assert r.cumulative_released_pct[-1] > 95.0


# 14. duration_days > 0
def test_duration_days_positive():
    r = _run("rod")
    assert r.duration_days > 0


# 15. Invalid implant_type
def test_invalid_implant_type():
    with pytest.raises(ValueError, match="implant_type"):
        _run("capsule")


# 16. Invalid dose
def test_invalid_dose():
    with pytest.raises(ValueError, match="dose_mg"):
        _run("rod", dose_mg=-10.0)


# 17. Invalid clearance
def test_invalid_clearance():
    with pytest.raises(ValueError, match="cl_L_per_h"):
        _run("rod", cl_L_per_h=0.0)


# 18. Invalid volume
def test_invalid_vd():
    with pytest.raises(ValueError, match="vd_L"):
        _run("rod", vd_L=-1.0)


# 19. drug_name preserved
def test_drug_name():
    r = _run("rod", drug_name="Aspirin")
    assert r.drug_name == "Aspirin"


# 20. tmax is within simulation range
def test_tmax_in_range():
    r = _run("microsphere")
    assert 0 <= r.tmax_h <= 720.0


# 21. notes is non-empty
def test_notes_nonempty():
    r = _run("rod")
    assert len(r.notes) > 0


# 22. Microsphere biphasic: cumulative at 72h is substantial
def test_microsphere_biphasic():
    r = _run("microsphere", t_end_h=200.0)
    idx_72 = 72  # dt=1h so index == hour
    cum_72 = r.cumulative_released_pct[idx_72]
    # Fast phase k=0.02 for 72h should release meaningful amount
    assert cum_72 > 50.0


# 23. Rod t90 reachable within ~600h
def test_rod_t90_reachable():
    r = _run("rod", t_end_h=800.0)
    assert r.t90_released_h < math.inf
