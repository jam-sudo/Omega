"""Unit and integration tests for the in vitro assay simulators and pipeline.

Coverage:
  1. MicrosomesAssay — depletion kinetics, fu_mic correction, whole-liver scaling
  2. HepatocyteAssay — combined Phase 1+2 depletion, viability correction
  3. EquilibriumOccupancy — Langmuir isotherm occupancy and AUC
  4. KineticOccupancy — kon/koff ODE solver, steady-state consistency
  5. OperationalModel (Black-Leff) — occupancy → effect mapping
  6. EC50Extractor — Hill-equation fitting from dose-response data
  7. Caco2Assay — bidirectional transport ODE, efflux ratio, classification
  8. Phase2Predictor — UGT/SULT/MAO QSPR (SMILES-heuristic path)
  9. InVitroPipeline — 7-stage SMILES→Drug end-to-end integration
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from omega_pbpk.in_vitro.assay import (
    HPGL,
    LIVER_WEIGHT_G,
    MPPGL,
    HepatocyteAssay,
    MicrosomesAssay,
)
from omega_pbpk.in_vitro.caco2 import Caco2Assay
from omega_pbpk.in_vitro.receptor import (
    EC50Extractor,
    EquilibriumOccupancy,
    KineticOccupancy,
    OperationalModel,
)
from omega_pbpk.prediction.phase2_predictor import Phase2Predictor


# ---------------------------------------------------------------------------
# MicrosomesAssay
# ---------------------------------------------------------------------------


class TestMicrosomesAssay:
    """HLM substrate-depletion assay simulator."""

    def _assay(self, **kw) -> MicrosomesAssay:
        defaults = dict(protein_conc_mg_mL=0.5, volume_uL=200.0, f_unbound_mic=1.0)
        defaults.update(kw)
        return MicrosomesAssay(**defaults)

    def test_parent_decays_over_time(self):
        assay = self._assay()
        res = assay.simulate(clint_ul_min_mg=10.0, c0_uM=1.0, t_end_min=60.0)
        # Concentration should be strictly decreasing
        assert res.c_parent[-1] < res.c_parent[0]
        assert res.c_parent[0] == pytest.approx(1.0, rel=1e-3)

    def test_t_half_consistent_with_k_dep(self):
        assay = self._assay()
        res = assay.simulate(clint_ul_min_mg=5.0, c0_uM=1.0, t_end_min=120.0)
        # t_half = ln2 / k_dep_per_min; k_dep_per_h → k_dep_per_min = k_dep_per_h / 60
        k_dep_min = res.k_dep_per_h / 60.0
        expected_t_half = math.log(2.0) / k_dep_min if k_dep_min > 0 else float("inf")
        assert res.t_half_min == pytest.approx(expected_t_half, rel=1e-4)

    def test_at_t_half_parent_near_half_c0(self):
        assay = self._assay()
        clint = 20.0  # fast depletion so t_half is short (< 60 min)
        res = assay.simulate(clint_ul_min_mg=clint, c0_uM=2.0, t_end_min=120.0)
        t_half = res.t_half_min
        # Find index closest to t_half
        idx = int(np.argmin(np.abs(res.time_min - t_half)))
        c_at_half = res.c_parent[idx]
        # Should be near C0/2 = 1.0 (allow ±10%)
        assert c_at_half == pytest.approx(1.0, rel=0.10)

    def test_fu_mic_correction_shortens_half_life(self):
        """fu_mic < 1 increases effective CLint → faster depletion."""
        assay_no_bind = MicrosomesAssay(f_unbound_mic=1.0)
        assay_binding = MicrosomesAssay(f_unbound_mic=0.1)
        clint = 5.0
        res1 = assay_no_bind.simulate(clint_ul_min_mg=clint, c0_uM=1.0)
        res2 = assay_binding.simulate(clint_ul_min_mg=clint, c0_uM=1.0)
        assert res2.t_half_min < res1.t_half_min

    def test_zero_clint_no_depletion(self):
        assay = self._assay()
        res = assay.simulate(clint_ul_min_mg=0.0, c0_uM=1.0, t_end_min=60.0)
        assert res.c_parent[-1] == pytest.approx(1.0, abs=1e-6)

    def test_clint_liver_scaling(self):
        """Verify whole-liver CLint calculation against the published formula."""
        assay = MicrosomesAssay(f_unbound_mic=1.0)
        clint_nom = 10.0  # µL/min/mg
        res = assay.simulate(clint_ul_min_mg=clint_nom)
        expected = clint_nom * MPPGL * LIVER_WEIGHT_G / 1e6 * 60.0   # L/h
        assert res.clint_liver_L_per_h == pytest.approx(expected, rel=1e-4)

    def test_estimate_fu_mic_decreases_with_logp(self):
        """Austin (2002): higher logP → more microsomal binding → lower fu_mic."""
        fu1 = MicrosomesAssay.estimate_fu_mic(1.0)
        fu3 = MicrosomesAssay.estimate_fu_mic(3.0)
        fu5 = MicrosomesAssay.estimate_fu_mic(5.0)
        assert 0.0 < fu5 < fu3 < fu1 <= 1.0

    def test_metabolite_formation_tracked(self):
        assay = MicrosomesAssay(f_metabolite=1.0)
        res = assay.simulate(clint_ul_min_mg=20.0, c0_uM=1.0, t_end_min=120.0)
        # Metabolite should accumulate (peak > 0)
        assert res.c_metabolite.max() > 0.0

    def test_high_c0_warning(self):
        assay = self._assay()
        res = assay.simulate(clint_ul_min_mg=5.0, c0_uM=50.0)
        assert any("linear range" in w for w in res.warnings)

    def test_assay_type_is_microsomes(self):
        assay = self._assay()
        res = assay.simulate(clint_ul_min_mg=5.0)
        assert res.assay_type == "microsomes"


# ---------------------------------------------------------------------------
# HepatocyteAssay
# ---------------------------------------------------------------------------


class TestHepatocyteAssay:
    """Suspended hepatocyte metabolic stability simulator."""

    def _assay(self, **kw) -> HepatocyteAssay:
        defaults = dict(cell_density_million_per_mL=1.0, volume_uL=200.0, viability_fraction=0.9)
        defaults.update(kw)
        return HepatocyteAssay(**defaults)

    def test_parent_decays(self):
        assay = self._assay()
        res = assay.simulate(clint_ul_min_million=5.0, c0_uM=1.0, t_end_min=120.0)
        assert res.c_parent[-1] < res.c_parent[0]

    def test_zero_clint_no_depletion(self):
        assay = self._assay()
        res = assay.simulate(clint_ul_min_million=0.0, c0_uM=1.0, t_end_min=120.0)
        assert res.c_parent[-1] == pytest.approx(1.0, abs=1e-6)

    def test_lower_viability_slower_depletion(self):
        """Viable fraction scales the effective CLint."""
        assay_full = HepatocyteAssay(viability_fraction=1.0)
        assay_half = HepatocyteAssay(viability_fraction=0.5)
        clint = 5.0
        r_full = assay_full.simulate(clint_ul_min_million=clint)
        r_half = assay_half.simulate(clint_ul_min_million=clint)
        # Lower viability → slower depletion → longer t_half
        assert r_half.t_half_min > r_full.t_half_min

    def test_low_viability_emits_warning(self):
        assay = HepatocyteAssay(viability_fraction=0.7)
        res = assay.simulate(clint_ul_min_million=5.0)
        assert any("viability" in w.lower() for w in res.warnings)

    def test_clint_liver_scaling_hpgl(self):
        """Scale via HPGL × LIVER_WEIGHT_G."""
        assay = HepatocyteAssay(viability_fraction=1.0, fu_cell=1.0)
        clint = 3.0
        res = assay.simulate(clint_ul_min_million=clint)
        expected = clint * (HPGL / 1e6) * LIVER_WEIGHT_G / 1e6 * 60.0   # L/h
        assert res.clint_liver_L_per_h == pytest.approx(expected, rel=1e-4)

    def test_assay_type_is_hepatocytes(self):
        assay = self._assay()
        res = assay.simulate(clint_ul_min_million=5.0)
        assert res.assay_type == "hepatocytes"


# ---------------------------------------------------------------------------
# EquilibriumOccupancy
# ---------------------------------------------------------------------------


class TestEquilibriumOccupancy:
    """Langmuir equilibrium receptor occupancy."""

    def _cp(self, value: float, n: int = 101) -> tuple[np.ndarray, np.ndarray]:
        t = np.linspace(0.0, 10.0, n)
        cp = np.full(n, value)
        return t, cp

    def test_at_kd_occupancy_is_0_5(self):
        kd = 2.0
        model = EquilibriumOccupancy(kd_mg_L=kd)
        t, cp = self._cp(kd)
        res = model.compute(t, cp)
        assert res.occupancy[0] == pytest.approx(0.5, rel=1e-6)

    def test_zero_concentration_zero_occupancy(self):
        model = EquilibriumOccupancy(kd_mg_L=1.0)
        t, cp = self._cp(0.0)
        res = model.compute(t, cp)
        assert np.allclose(res.occupancy, 0.0)

    def test_very_high_concentration_near_full_occupancy(self):
        kd = 0.1
        model = EquilibriumOccupancy(kd_mg_L=kd)
        t, cp = self._cp(1000.0)
        res = model.compute(t, cp)
        assert res.occupancy[0] > 0.99

    def test_peak_occupancy_at_cmax(self):
        model = EquilibriumOccupancy(kd_mg_L=1.0)
        t = np.linspace(0.0, 24.0, 241)
        cp = np.exp(-0.3 * t) * 5.0   # decaying profile, peak at t=0
        res = model.compute(t, cp)
        assert res.tpeak_h == pytest.approx(t[0], abs=0.2)

    def test_auc_occupancy_positive(self):
        model = EquilibriumOccupancy(kd_mg_L=1.0)
        t, cp = self._cp(2.0)
        res = model.compute(t, cp)
        assert res.auc_occupancy_h > 0.0

    def test_model_type_is_equilibrium(self):
        model = EquilibriumOccupancy(kd_mg_L=1.0)
        t, cp = self._cp(1.0)
        res = model.compute(t, cp)
        assert res.model_type == "equilibrium"

    def test_effect_with_operational_model(self):
        occ_model = EquilibriumOccupancy(kd_mg_L=1.0)
        op_model = OperationalModel(e0=0.0, emax=100.0, tau=1.0)
        t, cp = self._cp(1.0)   # Cp = Kd → occ = 0.5
        res = occ_model.compute(t, cp, operational_model=op_model)
        # E = 0 + 100 × 1 × 0.5 / (1 + 1 × 0.5) = 50/1.5 ≈ 33.3
        assert res.effect[0] == pytest.approx(100.0 * 0.5 / 1.5, rel=1e-4)

    def test_output_shapes(self):
        model = EquilibriumOccupancy(kd_mg_L=1.0)
        t, cp = self._cp(1.0, n=50)
        res = model.compute(t, cp)
        assert res.occupancy.shape == (50,)
        assert res.effect.shape == (50,)
        assert res.time_h.shape == (50,)


# ---------------------------------------------------------------------------
# KineticOccupancy
# ---------------------------------------------------------------------------


class TestKineticOccupancy:
    def test_initial_occupancy_is_zero(self):
        model = KineticOccupancy(kon_per_h_per_mg_L=1.0, koff_per_h=1.0)
        t = np.linspace(0.0, 10.0, 101)
        cp = np.ones(101)
        res = model.compute(t, cp, occ0=0.0)
        assert res.occupancy[0] == pytest.approx(0.0, abs=1e-6)

    def test_kd_property(self):
        model = KineticOccupancy(kon_per_h_per_mg_L=2.0, koff_per_h=1.0)
        assert model.kd_mg_L == pytest.approx(0.5, rel=1e-6)

    def test_residence_time_property(self):
        model = KineticOccupancy(kon_per_h_per_mg_L=1.0, koff_per_h=2.0)
        assert model.residence_time_h == pytest.approx(0.5, rel=1e-6)

    def test_steady_state_approaches_equilibrium(self):
        """At constant Cp, kinetic occupancy should converge to Kd-based prediction."""
        kd = 1.0
        cp_val = 2.0   # occ_ss = Cp/(Kd+Cp) = 2/3 ≈ 0.667
        model = KineticOccupancy(kon_per_h_per_mg_L=1.0, koff_per_h=kd)
        t = np.linspace(0.0, 50.0, 501)
        cp = np.full(501, cp_val)
        res = model.compute(t, cp)
        expected = cp_val / (kd + cp_val)
        assert res.occupancy[-1] == pytest.approx(expected, rel=0.02)

    def test_model_type_is_kinetic(self):
        model = KineticOccupancy(kon_per_h_per_mg_L=1.0, koff_per_h=1.0)
        t = np.linspace(0.0, 10.0, 101)
        cp = np.ones(101)
        res = model.compute(t, cp)
        assert res.model_type == "kinetic"

    def test_occupancy_bounded_0_to_1(self):
        model = KineticOccupancy(kon_per_h_per_mg_L=10.0, koff_per_h=0.1)
        t = np.linspace(0.0, 20.0, 201)
        cp = np.full(201, 100.0)
        res = model.compute(t, cp)
        assert np.all(res.occupancy >= -1e-6)
        assert np.all(res.occupancy <= 1.0 + 1e-6)


# ---------------------------------------------------------------------------
# OperationalModel
# ---------------------------------------------------------------------------


class TestOperationalModel:
    def test_zero_occupancy_returns_e0(self):
        model = OperationalModel(e0=5.0, emax=100.0, tau=1.0)
        effect = model.effect(np.array([0.0]))
        assert effect[0] == pytest.approx(5.0, rel=1e-6)

    def test_full_occupancy_returns_near_emax_for_high_tau(self):
        """At tau >> 1, effect at occ=1 approaches Emax."""
        model = OperationalModel(e0=0.0, emax=100.0, tau=100.0)
        effect = model.effect(np.array([1.0]))
        assert effect[0] > 99.0

    def test_partial_agonist_tau_lt_1_lower_emax(self):
        full = OperationalModel(e0=0.0, emax=100.0, tau=10.0)
        partial = OperationalModel(e0=0.0, emax=100.0, tau=0.3)
        occ = np.array([1.0])
        assert partial.effect(occ)[0] < full.effect(occ)[0]

    def test_functional_ec50_fraction_tau_1(self):
        """For tau=1: functional EC50 fraction = 1/(1+1) = 0.5."""
        model = OperationalModel(e0=0.0, emax=100.0, tau=1.0)
        assert model.functional_ec50_fraction == pytest.approx(0.5, rel=1e-6)

    def test_functional_ec50_fraction_decreases_with_tau(self):
        """Higher tau → lower occupancy needed for half-maximal effect."""
        m_low = OperationalModel(tau=0.5)
        m_high = OperationalModel(tau=5.0)
        assert m_high.functional_ec50_fraction < m_low.functional_ec50_fraction

    def test_effect_monotonically_increases_with_occupancy(self):
        model = OperationalModel(e0=0.0, emax=100.0, tau=2.0)
        occ = np.linspace(0.0, 1.0, 50)
        effect = model.effect(occ)
        assert np.all(np.diff(effect) >= 0)

    def test_output_shape_preserved(self):
        model = OperationalModel(tau=1.0)
        occ = np.linspace(0.0, 1.0, 20)
        assert model.effect(occ).shape == occ.shape


# ---------------------------------------------------------------------------
# EC50Extractor
# ---------------------------------------------------------------------------


class TestEC50Extractor:
    def test_ec50_in_plausible_range(self):
        """EC50 should be near Kd for tau=1 (full agonist, no amplification)."""
        kd = 1.0
        occ_model = EquilibriumOccupancy(kd_mg_L=kd)
        op_model = OperationalModel(e0=0.0, emax=100.0, tau=1.0)
        extractor = EC50Extractor(occupancy_model=occ_model, operational_model=op_model)
        # Cmax range spanning 3 decades around Kd
        cmax = np.array([0.01, 0.05, 0.1, 0.3, 0.5, 1.0, 2.0, 5.0, 10.0]) * kd
        doses = cmax * 100.0
        res = extractor.fit_from_cmax_array(cmax, doses)
        # EC50 should be within an order of magnitude of Kd
        assert 0.01 < res.ec50_mg_L < 100.0

    def test_ec50_result_has_required_fields(self):
        occ_model = EquilibriumOccupancy(kd_mg_L=0.5)
        extractor = EC50Extractor(occupancy_model=occ_model)
        cmax = np.logspace(-2, 1, 8)
        doses = cmax * 50.0
        res = extractor.fit_from_cmax_array(cmax, doses)
        assert hasattr(res, "ec50_mg_L")
        assert hasattr(res, "emax_effect")
        assert hasattr(res, "hill")
        assert hasattr(res, "r_squared")

    def test_effects_array_length_matches_doses(self):
        occ_model = EquilibriumOccupancy(kd_mg_L=1.0)
        extractor = EC50Extractor(occupancy_model=occ_model)
        cmax = np.logspace(-1, 2, 6)
        doses = cmax * 30.0
        res = extractor.fit_from_cmax_array(cmax, doses)
        assert len(res.effects_at_cmax) == len(doses)


# ---------------------------------------------------------------------------
# Caco2Assay
# ---------------------------------------------------------------------------


class TestCaco2Assay:
    def test_efflux_ratio_from_explicit_papp(self):
        """ER = Papp(BA) / Papp(AB) — verify with exact ratio."""
        papp_ab = 5e-6
        papp_ba = 20e-6
        assay = Caco2Assay(papp_ab_cm_s=papp_ab, papp_ba_cm_s=papp_ba)
        res = assay.simulate()
        assert res.efflux_ratio == pytest.approx(4.0, rel=1e-4)

    def test_passive_permeability_flag_er_near_1(self):
        papp = 10e-6
        assay = Caco2Assay(papp_ab_cm_s=papp, papp_ba_cm_s=papp)
        res = assay.simulate()
        assert "passive" in res.pgp_bcrp_flag

    def test_efflux_substrate_flag_er_5(self):
        assay = Caco2Assay(papp_ab_cm_s=2e-6, papp_ba_cm_s=10e-6)
        res = assay.simulate()
        assert "efflux" in res.pgp_bcrp_flag

    def test_strong_efflux_flag_er_gt_10(self):
        assay = Caco2Assay(papp_ab_cm_s=1e-6, papp_ba_cm_s=12e-6)
        res = assay.simulate()
        assert "strong" in res.pgp_bcrp_flag

    def test_mass_recovery_near_100_no_metabolism(self):
        """Without gut metabolism, mass should be conserved."""
        assay = Caco2Assay(papp_ab_cm_s=5e-6, papp_ba_cm_s=5e-6, clint_gut_ul_min_mg=0.0)
        res = assay.simulate(c0_apical_uM=10.0)
        assert res.mass_recovery_pct == pytest.approx(100.0, abs=1.0)

    def test_concentration_arrays_non_negative(self):
        assay = Caco2Assay(papp_ab_cm_s=5e-6, papp_ba_cm_s=5e-6)
        res = assay.simulate(c0_apical_uM=10.0)
        assert np.all(res.c_apical_uM >= 0.0)
        assert np.all(res.c_basolateral_uM >= 0.0)

    def test_time_array_length(self):
        n = 61
        assay = Caco2Assay()
        res = assay.simulate(t_end_min=60.0, n_timepoints=n)
        assert len(res.time_min) == n

    def test_low_papp_warning(self):
        assay = Caco2Assay(papp_ab_cm_s=1e-9, papp_ba_cm_s=1e-9)
        res = assay.simulate()
        assert any("permeability" in w.lower() for w in res.warnings)

    def test_estimate_papp_from_logP_tpsa_increases_with_logp(self):
        papp_low = Caco2Assay.estimate_papp_from_logP_TPSA(logP=1.0, tpsa=80.0)
        papp_high = Caco2Assay.estimate_papp_from_logP_TPSA(logP=4.0, tpsa=80.0)
        assert papp_high > papp_low

    def test_estimate_papp_from_logP_tpsa_decreases_with_tpsa(self):
        papp_low_tpsa = Caco2Assay.estimate_papp_from_logP_TPSA(logP=3.0, tpsa=40.0)
        papp_high_tpsa = Caco2Assay.estimate_papp_from_logP_TPSA(logP=3.0, tpsa=140.0)
        assert papp_low_tpsa > papp_high_tpsa


# ---------------------------------------------------------------------------
# Phase2Predictor (SMILES heuristic path — no RDKit required)
# ---------------------------------------------------------------------------


class TestPhase2Predictor:
    """Tests use the SMILES-heuristic fallback path so no RDKit is required.

    Results are intentionally imprecise; the tests verify direction / sign
    of predictions, not exact numerical values.
    """

    @pytest.fixture(autouse=True)
    def predictor(self):
        self.pred = Phase2Predictor()

    def test_predict_returns_phase2properties(self):
        res = self.pred.predict("CC")  # ethane — inert
        assert hasattr(res, "clint_ugt_ul_min_mg")
        assert hasattr(res, "clint_sult_ul_min_mg")
        assert hasattr(res, "clint_mao_ul_min_mg")
        assert hasattr(res, "clint_phase2_total_ul_min_mg")

    def test_dominant_pathway_is_string(self):
        res = self.pred.predict("CC")
        assert isinstance(res.dominant_pathway, str)

    def test_confidence_is_valid(self):
        res = self.pred.predict("CC")
        assert res.confidence in ("high", "low")

    def test_inert_molecule_all_near_zero(self):
        """Hexane has no UGT/SULT/MAO-relevant groups."""
        res = self.pred.predict("CCCCCC")
        assert res.clint_phase2_total_ul_min_mg < 1.0

    def test_probabilities_bounded_0_to_1(self):
        for smiles in ["CC", "CCO", "CCN", "Cc1ccccc1O"]:
            res = self.pred.predict(smiles)
            assert 0.0 <= res.ugt_substrate_prob <= 1.0
            assert 0.0 <= res.sult_substrate_prob <= 1.0
            assert 0.0 <= res.mao_substrate_prob <= 1.0

    def test_clint_values_non_negative(self):
        for smiles in ["CC", "CCO", "CCN", "Cc1ccccc1O", "NCC(O)c1ccc(O)cc1"]:
            res = self.pred.predict(smiles)
            assert res.clint_ugt_ul_min_mg >= 0.0
            assert res.clint_sult_ul_min_mg >= 0.0
            assert res.clint_mao_ul_min_mg >= 0.0

    def test_total_equals_sum_of_pathways(self):
        res = self.pred.predict("CCO")
        total = round(
            res.clint_ugt_ul_min_mg + res.clint_sult_ul_min_mg + res.clint_mao_ul_min_mg, 4
        )
        assert res.clint_phase2_total_ul_min_mg == pytest.approx(total, abs=1e-3)

    def test_amine_molecule_has_nonzero_mao_prob(self):
        """Primary amine → positive MAO substrate probability."""
        # methylamine: simple primary amine
        res = self.pred.predict("CN")
        # With SMILES heuristic, N count > 0 → mao_prob > 0
        assert res.mao_substrate_prob > 0.0


# ---------------------------------------------------------------------------
# InVitroPipeline — integration tests
# ---------------------------------------------------------------------------


class TestInVitroPipeline:
    """Integration tests for the full 7-stage SMILES → Drug pipeline."""

    @pytest.fixture(scope="class")
    def caffeine_result(self):
        from omega_pbpk.in_vitro.pipeline import InVitroPipeline, InVitroRequest

        req = InVitroRequest(
            smiles="Cn1cnc2c1c(=O)n(C)c(=O)n2C",  # caffeine
            dose_mg=100.0,
            assay_type="hepatocyte",
            run_caco2=True,
            drug_name="caffeine",
        )
        return InVitroPipeline().run(req)

    def test_drug_object_created(self, caffeine_result):
        drug = caffeine_result.drug
        assert drug is not None
        assert drug.name == "caffeine"
        assert drug.mw > 100.0

    def test_drug_has_positive_mw(self, caffeine_result):
        assert caffeine_result.adme.mw > 0.0

    def test_clint_total_equals_phase1_plus_phase2(self, caffeine_result):
        expected = round(
            caffeine_result.clint_phase1_ul_min_mg + caffeine_result.clint_phase2_ul_min_mg,
            4,
        )
        assert caffeine_result.clint_total_ul_min_mg == pytest.approx(expected, abs=1e-3)

    def test_clh_is_positive(self, caffeine_result):
        assert caffeine_result.clh_l_per_h > 0.0

    def test_caco2_result_populated(self, caffeine_result):
        assert caffeine_result.caco2 is not None
        assert caffeine_result.caco2.efflux_ratio > 0.0

    def test_confidence_valid(self, caffeine_result):
        assert caffeine_result.confidence in ("high", "medium", "low")

    def test_assay_result_has_t_half(self, caffeine_result):
        assert caffeine_result.assay.t_half_min > 0.0

    def test_warnings_is_list(self, caffeine_result):
        assert isinstance(caffeine_result.warnings, list)

    def test_microsomes_assay_type(self):
        from omega_pbpk.in_vitro.pipeline import InVitroPipeline, InVitroRequest

        req = InVitroRequest(
            smiles="Cn1cnc2c1c(=O)n(C)c(=O)n2C",
            assay_type="microsomes",
            run_caco2=False,
        )
        result = InVitroPipeline().run(req)
        assert result.assay.assay_type == "microsomes"

    def test_no_caco2(self):
        from omega_pbpk.in_vitro.pipeline import InVitroPipeline, InVitroRequest

        req = InVitroRequest(
            smiles="Cn1cnc2c1c(=O)n(C)c(=O)n2C",
            run_caco2=False,
        )
        result = InVitroPipeline().run(req)
        assert result.caco2 is None

    def test_with_kd_receptor_result_populated(self):
        from omega_pbpk.in_vitro.pipeline import InVitroPipeline, InVitroRequest

        req = InVitroRequest(
            smiles="Cn1cnc2c1c(=O)n(C)c(=O)n2C",
            run_caco2=False,
            kd_mg_L=0.5,
            tau=2.0,
        )
        result = InVitroPipeline().run(req)
        assert result.receptor is not None
        assert result.receptor.ec50_mg_L > 0.0

    def test_without_kd_no_receptor_result(self):
        from omega_pbpk.in_vitro.pipeline import InVitroPipeline, InVitroRequest

        req = InVitroRequest(
            smiles="Cn1cnc2c1c(=O)n(C)c(=O)n2C",
            run_caco2=False,
            kd_mg_L=None,
        )
        result = InVitroPipeline().run(req)
        assert result.receptor is None

    def test_phase2_total_non_negative(self):
        from omega_pbpk.in_vitro.pipeline import InVitroPipeline, InVitroRequest

        req = InVitroRequest(smiles="CCCC", run_caco2=False)
        result = InVitroPipeline().run(req)
        assert result.clint_phase2_ul_min_mg >= 0.0

    def test_adme_properties_accessible(self, caffeine_result):
        adme = caffeine_result.adme
        assert hasattr(adme, "logP")
        assert hasattr(adme, "fup")
        assert hasattr(adme, "rbp")
        assert hasattr(adme, "peff")

    def test_phase2_properties_accessible(self, caffeine_result):
        p2 = caffeine_result.phase2
        assert hasattr(p2, "clint_ugt_ul_min_mg")
        assert hasattr(p2, "dominant_pathway")
        assert isinstance(p2.dominant_pathway, str)
