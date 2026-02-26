"""PK validation tests — verify pharmacokinetic correctness of the PBPK engine.

These tests confirm:
  1. CLh flow-limited ceiling (well-stirred model)
  2. CLh enzyme-limited floor
  3. AUC linearity with dose (IV, no clearance)
  4. IV vs oral AUC parity (low-extraction, Fg = 1)
  5. Caffeine CLh regression guard

Run: pytest tests/test_pk_validation.py -v
"""

from __future__ import annotations

import pytest

# ------------------------------------------------------------------ helpers


def _make_drug(**overrides):
    """Create a Drug instance with sensible defaults, applying overrides."""
    from omega_pbpk.drugs.drug import Drug

    defaults = dict(
        name="TestPK",
        mw=300.0,
        logP=2.0,
        fup=0.5,
        rbp=1.0,
        pka=[7.0],
        drug_type="neutral",
        peff=5.0,
        solubility_mg_mL=10.0,
        particle_radius_um=25.0,
        kp={
            "lung": 1.0,
            "brain": 1.0,
            "heart": 1.0,
            "kidney": 1.0,
            "liver": 1.0,
            "spleen": 1.0,
            "gut_wall": 1.0,
            "pancreas": 1.0,
            "thymus": 1.0,
            "reproductive": 1.0,
            "rest": 1.0,
        },
        permeability_limited={
            "adipose": {"kp": 1.0, "ps": 20},
            "muscle": {"kp": 1.0, "ps": 40},
            "bone": {"kp": 1.0, "ps": 10},
            "skin": {"kp": 1.0, "ps": 15},
        },
    )
    defaults.update(overrides)
    return Drug(**defaults)


def _simulate_iv(drug, dose_mg: float, t_end_h: float = 48.0, dt_h: float = 0.05):
    """Run IV bolus simulation and return (result, pk_summary)."""
    from omega_pbpk.core.body import WholeBodyPBPK

    model = WholeBodyPBPK(drug, body_weight=70.0)
    model.setup_iv(dose_mg)
    result = model.simulate(t_end_h=t_end_h, dt_h=dt_h)
    return result, result.pk_summary()


def _simulate_oral(drug, dose_mg: float, t_end_h: float = 48.0, dt_h: float = 0.05):
    """Run oral simulation and return (result, pk_summary)."""
    from omega_pbpk.core.body import WholeBodyPBPK

    model = WholeBodyPBPK(drug, body_weight=70.0)
    model.setup_oral(dose_mg)
    result = model.simulate(t_end_h=t_end_h, dt_h=dt_h)
    return result, result.pk_summary()


def _liver_blood_flow(body_weight: float = 70.0) -> float:
    """Total liver blood flow = hepatic artery + portal organs."""
    co = 390.0 * (body_weight / 70.0)
    # liver HA (0.065) + spleen (0.03) + gut_wall (0.15) + pancreas (0.01)
    return co * (0.065 + 0.03 + 0.15 + 0.01)


def _well_stirred_clh(q: float, fup: float, clint: float) -> float:
    """Compute CLh using the well-stirred model."""
    return (q * fup * clint) / max(q + fup * clint, 1e-12)


# ------------------------------------------------------------------ tests


class TestPKValidation:
    def test_clh_flow_limited_ceiling(self) -> None:
        """For CLint=500 L/h (high extraction), fu=1.0, Q_liver≈99.5:
        CLh should approach Q but never exceed it.
        Simulated CL should be bounded below Q_liver.
        """
        q_liver = _liver_blood_flow(70.0)
        clint = 500.0
        fup = 1.0

        # Analytical well-stirred CLh
        clh = _well_stirred_clh(q_liver, fup, clint)
        assert clh < q_liver, "CLh must not exceed liver blood flow"
        # CLh = (99.5 * 500) / (99.5 + 500) ≈ 83 L/h
        assert clh == pytest.approx(q_liver * clint / (q_liver + clint), rel=0.01)

        # Verify via simulation: CL from PK must be below Q_liver
        drug = _make_drug(clint_hepatic_L_per_h=clint, fup=fup, clr_L_per_h=0.0)
        _, pk = _simulate_iv(drug, dose_mg=100.0, dt_h=0.01)
        cl_sim = pk["CL_L_h"]
        assert cl_sim < q_liver, f"Simulated CL={cl_sim:.1f} must be below Q_liver={q_liver:.1f}"
        assert cl_sim > 0.5 * clh, (
            f"Simulated CL={cl_sim:.1f} should not be far below CLh={clh:.1f}"
        )

    def test_clh_enzyme_limited_floor(self) -> None:
        """For CLint=1.0 L/h (low extraction), fu=1.0, Q_liver≈99.5:
        CLh ≈ fu × CLint (enzyme-limited regime).
        Simulated CL should be in the same order of magnitude.
        """
        q_liver = _liver_blood_flow(70.0)
        clint = 1.0
        fup = 1.0

        # Analytical: at low extraction, CLh ≈ fu × CLint
        clh = _well_stirred_clh(q_liver, fup, clint)
        assert clh == pytest.approx(fup * clint, rel=0.05), (
            f"At low CLint={clint}, CLh={clh:.3f} ≈ fu×CLint={fup * clint:.3f}"
        )

        # Verify via simulation
        drug = _make_drug(clint_hepatic_L_per_h=clint, fup=fup, clr_L_per_h=0.0)
        _, pk = _simulate_iv(drug, dose_mg=100.0, t_end_h=120.0, dt_h=0.05)
        cl_sim = pk["CL_L_h"]
        # Allow 30% tolerance for multi-compartment redistribution effects
        assert cl_sim == pytest.approx(clh, rel=0.30), (
            f"Simulated CL={cl_sim:.3f} should be near CLh={clh:.3f}"
        )

    def test_auc_linear_with_dose_iv(self) -> None:
        """AUC scales linearly with IV dose when there is no clearance."""
        drug = _make_drug(
            clint_hepatic_L_per_h=0.0,
            clr_L_per_h=0.0,
            clint={},
            fm={},
            gut_clint_multiplier=0.0,
        )

        aucs = {}
        for dose in [50.0, 100.0, 200.0]:
            _, pk = _simulate_iv(drug, dose_mg=dose, t_end_h=24.0)
            aucs[dose] = pk["AUC_mg_h_L"]

        ratio = aucs[200.0] / aucs[50.0]
        assert ratio == pytest.approx(4.0, rel=0.02), (
            f"AUC(200mg)/AUC(50mg) = {ratio:.3f}, expected 4.0"
        )

    def test_iv_oral_auc_parity_low_extraction(self) -> None:
        """For low-extraction drug with Fg≈1, oral AUC ≈ IV AUC."""
        drug = _make_drug(
            clint_hepatic_L_per_h=1.0,
            fup=1.0,
            clr_L_per_h=0.0,
            clint={},
            fm={},
            clint_gut_L_per_h=0.0,
            gut_clint_multiplier=0.0,
            peff=10.0,
            solubility_mg_mL=50.0,
        )

        _, pk_iv = _simulate_iv(drug, dose_mg=100.0, t_end_h=120.0)
        _, pk_oral = _simulate_oral(drug, dose_mg=100.0, t_end_h=120.0)

        auc_ratio = pk_oral["AUC_mg_h_L"] / max(pk_iv["AUC_mg_h_L"], 1e-12)
        assert 0.90 <= auc_ratio <= 1.10, (
            f"AUC_oral/AUC_iv = {auc_ratio:.3f}, expected ≈1.0 "
            f"(oral={pk_oral['AUC_mg_h_L']:.3f}, iv={pk_iv['AUC_mg_h_L']:.3f})"
        )

    def test_caffeine_clh_regression(self) -> None:
        """Caffeine CLh from well-stirred model should be in [0.5, 5.0] L/h.

        With default caffeine config (CLint=8.0, fup=0.65, Q_liver≈99.5):
        CLh = (99.5 × 0.65 × 8.0) / (99.5 + 0.65 × 8.0) ≈ 4.94 L/h
        """
        from omega_pbpk.config import load_compound

        drug = load_compound("compounds/caffeine.yaml")
        q_liver = _liver_blood_flow(70.0)

        clint = drug.clint_scaled_L_per_h
        fup = drug.fup

        clh = _well_stirred_clh(q_liver, fup, clint)
        assert 0.5 <= clh <= 5.0, (
            f"Caffeine CLh = {clh:.3f} L/h (CLint={clint:.2f}, fup={fup}, Q_liver={q_liver:.1f})"
        )
