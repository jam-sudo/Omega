from physio_sim.config import load_compound, load_subject
from physio_sim.pbpk.solver import simulate


def test_mass_balance_with_urine_sink() -> None:
    subject = load_subject("examples/subject_default.yaml")
    compound = load_compound("examples/compound_caffeine.yaml").model_copy(
        update={"clint_L_per_h": 0.0, "clr_L_per_h": 8.0}
    )

    out = simulate(
        subject,
        compound,
        dose_mg=100.0,
        route="iv",
        t_end_h=24.0,
        dt_out_h=0.1,
    ).timecourse
    amount_cols = [column for column in out.columns if column.startswith("A_")]
    total = out[amount_cols].sum(axis=1)

    assert abs(total.iloc[-1] - 100.0) < 1e-3
