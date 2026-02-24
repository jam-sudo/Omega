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


def test_unbound_concentration_column_present() -> None:
    subject = load_subject("examples/subject_default.yaml")
    compound = load_compound("examples/compound_caffeine.yaml")
    out = simulate(
        subject,
        compound,
        dose_mg=10.0,
        route="iv",
        t_end_h=0.2,
        dt_out_h=0.1,
    ).timecourse
    assert "Cu_plasma_mg_per_L" in out.columns
    assert (out["Cu_plasma_mg_per_L"] <= out["C_plasma_mg_per_L"]).all()
