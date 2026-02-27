"""Unit tests for DDI (drug-drug interaction) mechanisms in WholeBodyPBPK.

Covers:
  - Competitive inhibition reduces CLint proportionally
  - Full inhibition (high [I]/Ki) approaches maximum inhibition
  - MBI (mechanism-based inactivation) reduces CLint over time
  - Induction increases CLint
  - DDI with ki_uM=None-equivalent (non-inhibitor) does nothing
"""

from __future__ import annotations

import pytest

from omega_pbpk.core.body import DDIInhibitor, WholeBodyPBPK
from omega_pbpk.drugs.drug import Drug

# ------------------------------------------------------------------ helpers


def _make_drug(**overrides) -> Drug:
    """Drug with CYP3A4 as the single metabolising enzyme (fm=1.0)."""
    defaults = dict(
        name="TestDDI",
        mw=300.0,
        logP=2.0,
        fup=0.5,
        rbp=1.0,
        pka=[7.0],
        drug_type="neutral",
        peff=5.0,
        solubility_mg_mL=10.0,
        particle_radius_um=25.0,
        # Use calibrated in-vivo CLint so the value is directly controlled
        clint_hepatic_L_per_h=10.0,
        clint_gut_L_per_h=0.0,
        clr_L_per_h=0.0,
        fm={"CYP3A4": 1.0},
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


def _clint_eff(drug: Drug, inhibitors: list[DDIInhibitor]) -> float:
    """Invoke the private _clint_effective helper via a model instance."""
    model = WholeBodyPBPK(drug, body_weight=70.0)
    for inh in inhibitors:
        model.add_inhibitor(inh)
    return model._clint_effective()


# ------------------------------------------------------------------ tests


class TestCompetitiveInhibition:
    def test_competitive_reduces_clint(self) -> None:
        """Competitive inhibitor with [I]=Ki should reduce CLint to ~50% fm contribution."""
        drug = _make_drug(clint_hepatic_L_per_h=10.0, fm={"CYP3A4": 1.0})
        base = drug.clint_scaled_L_per_h  # 10.0

        inh = DDIInhibitor(
            name="Inhibitor",
            ki_uM=1.0,
            concentration_uM=1.0,  # [I] == Ki → inhibition_factor = 0.5
            target_enzyme="CYP3A4",
            mechanism="competitive",
        )
        result = _clint_eff(drug, [inh])

        # With fm=1.0 and [I]/Ki=1: CLint_eff = CLint * (1 - 1.0 * (1 - 0.5)) = CLint * 0.5
        expected = base * 0.5
        assert result == pytest.approx(expected, rel=1e-6)

    def test_competitive_partial_fm_reduces_proportionally(self) -> None:
        """With fm=0.5 for CYP3A4 and [I]=Ki, CLint reduces by 25% (half of fm)."""
        drug = _make_drug(clint_hepatic_L_per_h=10.0, fm={"CYP3A4": 0.5})
        base = drug.clint_scaled_L_per_h

        inh = DDIInhibitor(
            name="PartialInh",
            ki_uM=1.0,
            concentration_uM=1.0,
            target_enzyme="CYP3A4",
            mechanism="competitive",
        )
        result = _clint_eff(drug, [inh])

        # CLint_eff = CLint * (1 - 0.5 * (1 - 0.5)) = CLint * 0.75
        expected = base * 0.75
        assert result == pytest.approx(expected, rel=1e-6)

    def test_competitive_full_inhibition_approaches_maximum(self) -> None:
        """At very high [I]/Ki (1000x), inhibition_factor → ~0, approaching max inhibition."""
        drug = _make_drug(clint_hepatic_L_per_h=10.0, fm={"CYP3A4": 1.0})
        base = drug.clint_scaled_L_per_h

        inh = DDIInhibitor(
            name="StrongInh",
            ki_uM=0.001,
            concentration_uM=10.0,  # [I]/Ki = 10000
            target_enzyme="CYP3A4",
            mechanism="competitive",
        )
        result = _clint_eff(drug, [inh])

        # inhibition_factor ≈ 0, CLint_eff ≈ CLint * (1 - 1.0 * 1.0) ≈ 0
        # max(0, result) should be very small relative to base
        assert result < base * 0.01, (
            f"Expected near-full inhibition; CLint_eff={result:.4f}, base={base:.4f}"
        )
        assert result >= 0.0, "CLint must not be negative"

    def test_competitive_no_inhibition_when_fm_zero(self) -> None:
        """If fm for the target enzyme is 0, inhibitor has no effect."""
        drug = _make_drug(clint_hepatic_L_per_h=10.0, fm={"CYP3A4": 0.0})
        base = drug.clint_scaled_L_per_h

        inh = DDIInhibitor(
            name="IrrelevantInh",
            ki_uM=1.0,
            concentration_uM=1.0,
            target_enzyme="CYP3A4",
            mechanism="competitive",
        )
        result = _clint_eff(drug, [inh])
        assert result == pytest.approx(base, rel=1e-9)

    def test_non_inhibitor_ki_effectively_infinite_does_nothing(self) -> None:
        """An inhibitor with a very large Ki (essentially non-inhibiting) leaves CLint unchanged."""
        drug = _make_drug(clint_hepatic_L_per_h=10.0, fm={"CYP3A4": 1.0})
        base = drug.clint_scaled_L_per_h

        inh = DDIInhibitor(
            name="WeakInh",
            ki_uM=1e9,  # Effectively infinite Ki
            concentration_uM=1.0,
            target_enzyme="CYP3A4",
            mechanism="competitive",
        )
        result = _clint_eff(drug, [inh])

        # inhibition_factor ≈ 1.0, so CLint_eff ≈ CLint * (1 - 1.0 * 0) = CLint
        assert result == pytest.approx(base, rel=1e-3)


class TestMBIInhibition:
    def test_mbi_reduces_clint(self) -> None:
        """MBI mechanism reduces CLint when kinact/kdeg ratio produces partial inactivation."""
        drug = _make_drug(clint_hepatic_L_per_h=10.0, fm={"CYP3A4": 1.0})
        base = drug.clint_scaled_L_per_h

        inh = DDIInhibitor(
            name="MBIInhibitor",
            ki_uM=1.0,
            concentration_uM=1.0,
            target_enzyme="CYP3A4",
            mechanism="mbi",
            kinact_per_h=0.5,
            kdeg_per_h=0.02,
        )
        result = _clint_eff(drug, [inh])

        # Verify CLint is reduced (not unchanged)
        assert result < base, f"MBI should reduce CLint; got {result:.4f} vs base {base:.4f}"
        assert result >= 0.0

    def test_mbi_frac_inact_calculation(self) -> None:
        """Verify MBI fraction inactivated follows the formula exactly."""
        drug = _make_drug(clint_hepatic_L_per_h=10.0, fm={"CYP3A4": 1.0})
        base = drug.clint_scaled_L_per_h

        ki = 1.0
        conc = 2.0
        kinact = 0.6
        kdeg = 0.02

        numerator = kinact * conc
        denominator = kdeg * (ki + conc) + kinact * conc
        frac_inact = numerator / denominator
        expected = base * (1.0 - 1.0 * frac_inact)

        inh = DDIInhibitor(
            name="MBIExact",
            ki_uM=ki,
            concentration_uM=conc,
            target_enzyme="CYP3A4",
            mechanism="mbi",
            kinact_per_h=kinact,
            kdeg_per_h=kdeg,
        )
        result = _clint_eff(drug, [inh])
        assert result == pytest.approx(expected, rel=1e-6)

    def test_mbi_zero_kinact_no_effect(self) -> None:
        """MBI with kinact=0 should produce zero inactivation fraction → no change."""
        drug = _make_drug(clint_hepatic_L_per_h=10.0, fm={"CYP3A4": 1.0})
        base = drug.clint_scaled_L_per_h

        inh = DDIInhibitor(
            name="ZeroKinact",
            ki_uM=1.0,
            concentration_uM=1.0,
            target_enzyme="CYP3A4",
            mechanism="mbi",
            kinact_per_h=0.0,
            kdeg_per_h=0.02,
        )
        result = _clint_eff(drug, [inh])
        assert result == pytest.approx(base, rel=1e-9)


class TestInduction:
    def test_induction_increases_clint(self) -> None:
        """Induction with fold_induction>1 should increase CLint."""
        drug = _make_drug(clint_hepatic_L_per_h=10.0, fm={"CYP3A4": 1.0})
        base = drug.clint_scaled_L_per_h

        inh = DDIInhibitor(
            name="Inducer",
            ki_uM=1.0,
            concentration_uM=1.0,
            target_enzyme="CYP3A4",
            mechanism="induction",
            fold_induction=3.0,
        )
        result = _clint_eff(drug, [inh])

        # CLint_eff = CLint * (1 + fm * (fold - 1)) = 10 * (1 + 1.0 * 2.0) = 30
        expected = base * (1.0 + 1.0 * (3.0 - 1.0))
        assert result == pytest.approx(expected, rel=1e-6)
        assert result > base

    def test_induction_fold_one_no_change(self) -> None:
        """fold_induction=1.0 means no induction; CLint unchanged."""
        drug = _make_drug(clint_hepatic_L_per_h=10.0, fm={"CYP3A4": 1.0})
        base = drug.clint_scaled_L_per_h

        inh = DDIInhibitor(
            name="NoInducer",
            ki_uM=1.0,
            concentration_uM=1.0,
            target_enzyme="CYP3A4",
            mechanism="induction",
            fold_induction=1.0,
        )
        result = _clint_eff(drug, [inh])
        assert result == pytest.approx(base, rel=1e-9)

    def test_induction_partial_fm(self) -> None:
        """Induction with fm=0.5 gives proportionally smaller CLint increase."""
        drug = _make_drug(clint_hepatic_L_per_h=10.0, fm={"CYP3A4": 0.5})
        base = drug.clint_scaled_L_per_h

        inh = DDIInhibitor(
            name="PartialInducer",
            ki_uM=1.0,
            concentration_uM=1.0,
            target_enzyme="CYP3A4",
            mechanism="induction",
            fold_induction=3.0,
        )
        result = _clint_eff(drug, [inh])

        # CLint_eff = 10 * (1 + 0.5 * 2.0) = 20
        expected = base * (1.0 + 0.5 * (3.0 - 1.0))
        assert result == pytest.approx(expected, rel=1e-6)


class TestNoDDI:
    def test_no_inhibitors_returns_base_clint(self) -> None:
        """Without any inhibitors, _clint_effective returns the base CLint."""
        drug = _make_drug(clint_hepatic_L_per_h=10.0, fm={"CYP3A4": 1.0})
        model = WholeBodyPBPK(drug, body_weight=70.0)
        result = model._clint_effective()
        assert result == pytest.approx(drug.clint_scaled_L_per_h, rel=1e-9)

    def test_inhibitor_targeting_different_enzyme_no_effect(self) -> None:
        """Inhibitor targeting CYP2D6 when drug is only metabolised by CYP3A4 has no effect."""
        drug = _make_drug(clint_hepatic_L_per_h=10.0, fm={"CYP3A4": 1.0})
        base = drug.clint_scaled_L_per_h

        inh = DDIInhibitor(
            name="WrongEnzymeInh",
            ki_uM=0.001,
            concentration_uM=100.0,
            target_enzyme="CYP2D6",  # Drug has fm=0 for CYP2D6
            mechanism="competitive",
        )
        result = _clint_eff(drug, [inh])
        assert result == pytest.approx(base, rel=1e-9)
