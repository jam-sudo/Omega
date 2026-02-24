from __future__ import annotations

from physio_sim.config import load_compound, load_subject
from physio_sim.pbpk.solver import simulate
from physio_sim.validation import mass_balance_check, physiologic_sanity_check


def test_physiologic_sanity_check_baseline_no_warnings() -> None:
    subject = load_subject("examples/subject_default.yaml")
    compound = load_compound("examples/compound_caffeine.yaml")
    warnings = physiologic_sanity_check(subject, compound)
    assert warnings == []


def test_mass_balance_check_no_warnings_for_closed_case() -> None:
    subject = load_subject("examples/subject_default.yaml")
    compound = load_compound("examples/compound_caffeine.yaml").model_copy(
        update={"clint_L_per_h": 0.0, "clr_L_per_h": 0.0}
    )
    df = simulate(
        subject, compound, dose_mg=100.0, route="iv", t_end_h=1.0, dt_out_h=0.1
    ).timecourse
    warnings = mass_balance_check(df, dose_mg=100.0)
    assert warnings == []
