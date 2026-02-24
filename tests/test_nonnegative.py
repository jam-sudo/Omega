from physio_sim.config import load_compound, load_subject
from physio_sim.pbpk.solver import simulate


def test_amounts_nonnegative_with_tolerance() -> None:
    subject = load_subject("examples/subject_default.yaml")
    compound = load_compound("examples/compound_caffeine.yaml")
    out = simulate(
        subject,
        compound,
        dose_mg=100.0,
        route="oral",
        t_end_h=4.0,
        dt_out_h=0.1,
    ).timecourse
    amount_cols = [c for c in out.columns if c.startswith("A_")]
    assert (out[amount_cols].to_numpy() >= -1e-8).all()
