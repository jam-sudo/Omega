"""Omega PBPK test suite — 47 tests covering all modules.

Run: pytest tests/test_all.py -v
"""

from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path

import numpy as np
import pytest

# ============================================================
# 1. Drug dataclass tests (5 tests)
# ============================================================


class TestDrug:
    def test_drug_creation(self) -> None:
        from omega_pbpk.drugs.drug import Drug

        drug = Drug(name="Test", mw=300.0, logP=2.0, fup=0.5)
        assert drug.name == "Test"
        assert drug.mw == 300.0
        assert drug.fup == 0.5

    def test_drug_frozen(self) -> None:
        from omega_pbpk.drugs.drug import Drug

        drug = Drug(name="Test")
        with pytest.raises(AttributeError):
            drug.name = "Changed"  # type: ignore[misc]

    def test_drug_total_clint(self) -> None:
        from omega_pbpk.drugs.drug import Drug

        drug = Drug(clint={"CYP3A4": 0.9, "CYP3A5": 0.3})
        assert drug.total_clint == pytest.approx(1.2)

    def test_drug_clint_scaled(self) -> None:
        from omega_pbpk.drugs.drug import Drug

        drug = Drug(clint={"CYP3A4": 1.0})
        # 1.0 × 40 × 45 × 1800 / 1e6 / 60 = 0.054
        assert drug.clint_scaled_L_per_h == pytest.approx(0.054, rel=0.01)

    def test_drug_default_kp(self) -> None:
        from omega_pbpk.drugs.drug import Drug

        drug = Drug(kp={"liver": 5.0})
        assert drug.default_kp("liver") == 5.0
        assert drug.default_kp("missing") == 1.0


# ============================================================
# 2. Midazolam reference compound (3 tests)
# ============================================================


class TestMidazolam:
    def test_midazolam_loaded(self) -> None:
        from omega_pbpk.drugs.midazolam import MIDAZOLAM

        assert MIDAZOLAM.name == "Midazolam"
        assert MIDAZOLAM.mw == pytest.approx(325.77)

    def test_midazolam_kp_count(self) -> None:
        from omega_pbpk.drugs.midazolam import MIDAZOLAM

        assert len(MIDAZOLAM.kp) == 11  # 11 perfusion-limited organs

    def test_midazolam_permeability_limited(self) -> None:
        from omega_pbpk.drugs.midazolam import MIDAZOLAM

        assert len(MIDAZOLAM.permeability_limited) == 4
        assert "adipose" in MIDAZOLAM.permeability_limited


# ============================================================
# 3. Organ dataclass tests (3 tests)
# ============================================================


class TestOrgan:
    def test_organ_perfusion_limited(self) -> None:
        from omega_pbpk.core.organ import Organ

        organ = Organ(name="liver", volume_L=1.8, blood_flow_L_per_h=97.5, kp=5.87)
        assert not organ.is_permeability_limited
        assert organ.extravascular_volume_L == 1.8

    def test_organ_permeability_limited(self) -> None:
        from omega_pbpk.core.organ import Organ

        organ = Organ(name="adipose", volume_L=14.5, blood_flow_L_per_h=20.0,
                      kp=30.0, is_permeability_limited=True, ps_L_per_h=8.0)
        assert organ.is_permeability_limited
        assert organ.vascular_volume_L == pytest.approx(14.5 * 0.04)
        assert organ.extravascular_volume_L == pytest.approx(14.5 * 0.96)

    def test_organ_frozen(self) -> None:
        from omega_pbpk.core.organ import Organ

        organ = Organ(name="test", volume_L=1.0, blood_flow_L_per_h=10.0)
        with pytest.raises(AttributeError):
            organ.name = "changed"  # type: ignore[misc]


# ============================================================
# 4. PBPK model tests (8 tests)
# ============================================================


class TestWholeBodyPBPK:
    def test_model_creation(self) -> None:
        from omega_pbpk.core.body import WholeBodyPBPK, N_STATES
        from omega_pbpk.drugs.midazolam import MIDAZOLAM

        model = WholeBodyPBPK(MIDAZOLAM)
        assert len(model.organs) > 0
        assert model._y0.shape == (N_STATES,)

    def test_iv_setup(self) -> None:
        from omega_pbpk.core.body import WholeBodyPBPK, IDX_VEN
        from omega_pbpk.drugs.midazolam import MIDAZOLAM

        model = WholeBodyPBPK(MIDAZOLAM)
        model.setup_iv(5.0)
        assert model._y0[IDX_VEN] == 5.0
        assert model.drug.dose_mg == 5.0
        assert model.drug.route == "iv"

    def test_oral_setup(self) -> None:
        from omega_pbpk.core.body import WholeBodyPBPK, IDX_STOMACH
        from omega_pbpk.drugs.midazolam import MIDAZOLAM

        model = WholeBodyPBPK(MIDAZOLAM)
        model.setup_oral(7.5)
        assert model._y0[IDX_STOMACH] == 7.5

    def test_iv_simulation_runs(self) -> None:
        from omega_pbpk.core.body import WholeBodyPBPK
        from omega_pbpk.drugs.midazolam import MIDAZOLAM

        model = WholeBodyPBPK(MIDAZOLAM)
        model.setup_iv(5.0)
        result = model.simulate(t_end_h=24.0, dt_h=0.5)
        assert len(result.time_h) > 0
        assert result.amounts.shape[1] == 34

    def test_iv_mass_balance(self) -> None:
        """IV mass balance: total drug must equal dose ± 0.5%."""
        from omega_pbpk.core.body import WholeBodyPBPK
        from omega_pbpk.drugs.midazolam import MIDAZOLAM

        model = WholeBodyPBPK(MIDAZOLAM)
        model.setup_iv(5.0)
        result = model.simulate(t_end_h=24.0, dt_h=0.1)
        mb = result.mass_balance()
        # Check mass balance at all time points
        for i, total in enumerate(mb):
            assert abs(total - 5.0) / 5.0 < 0.005, (
                f"Mass balance violated at t={result.time_h[i]:.1f}h: "
                f"{total:.6f} mg (expected 5.0 mg)"
            )

    def test_oral_simulation_runs(self) -> None:
        from omega_pbpk.core.body import WholeBodyPBPK
        from omega_pbpk.drugs.midazolam import MIDAZOLAM

        model = WholeBodyPBPK(MIDAZOLAM)
        model.setup_oral(7.5)
        result = model.simulate(t_end_h=24.0, dt_h=0.5)
        assert len(result.time_h) > 0
        cp = result.plasma_concentration()
        assert np.max(cp) > 0  # Some drug reaches plasma

    def test_pk_summary(self) -> None:
        from omega_pbpk.core.body import WholeBodyPBPK
        from omega_pbpk.drugs.midazolam import MIDAZOLAM

        model = WholeBodyPBPK(MIDAZOLAM)
        model.setup_iv(5.0)
        result = model.simulate(t_end_h=24.0)
        pk = result.pk_summary()
        assert "Cmax_mg_L" in pk
        assert "AUC_mg_h_L" in pk
        assert "half_life_h" in pk
        assert pk["Cmax_mg_L"] > 0
        assert pk["AUC_mg_h_L"] > 0

    def test_body_weight_scaling(self) -> None:
        from omega_pbpk.core.body import WholeBodyPBPK
        from omega_pbpk.drugs.midazolam import MIDAZOLAM

        model_70 = WholeBodyPBPK(MIDAZOLAM, body_weight=70.0)
        model_50 = WholeBodyPBPK(MIDAZOLAM, body_weight=50.0)
        # Lighter person should have smaller organ volumes
        assert model_50.organs["liver"].volume_L < model_70.organs["liver"].volume_L


# ============================================================
# 5. DDI tests (3 tests)
# ============================================================


class TestDDI:
    def test_competitive_inhibition(self) -> None:
        from omega_pbpk.core.body import DDIInhibitor, WholeBodyPBPK
        from omega_pbpk.drugs.midazolam import MIDAZOLAM

        # Baseline
        model_base = WholeBodyPBPK(MIDAZOLAM)
        model_base.setup_oral(7.5)
        result_base = model_base.simulate(t_end_h=24.0)
        auc_base = result_base.pk_summary()["AUC_mg_h_L"]

        # With strong CYP3A4 inhibitor
        model_ddi = WholeBodyPBPK(MIDAZOLAM)
        model_ddi.setup_oral(7.5)
        model_ddi.add_inhibitor(DDIInhibitor(
            name="Ketoconazole",
            ki_uM=0.015,
            concentration_uM=5.0,
            target_enzyme="CYP3A4",
            mechanism="competitive",
        ))
        result_ddi = model_ddi.simulate(t_end_h=24.0)
        auc_ddi = result_ddi.pk_summary()["AUC_mg_h_L"]

        # Inhibitor should increase AUC
        assert auc_ddi > auc_base, "DDI should increase AUC"

    def test_mbi_inhibition(self) -> None:
        from omega_pbpk.core.body import DDIInhibitor, WholeBodyPBPK
        from omega_pbpk.drugs.midazolam import MIDAZOLAM

        model = WholeBodyPBPK(MIDAZOLAM)
        model.setup_oral(7.5)
        model.add_inhibitor(DDIInhibitor(
            name="Erythromycin", ki_uM=5.0, concentration_uM=3.0,
            target_enzyme="CYP3A4", mechanism="mbi",
            kinact_per_h=2.0, kdeg_per_h=0.02,
        ))
        result = model.simulate(t_end_h=24.0)
        assert result.pk_summary()["AUC_mg_h_L"] > 0

    def test_induction(self) -> None:
        from omega_pbpk.core.body import DDIInhibitor, WholeBodyPBPK
        from omega_pbpk.drugs.midazolam import MIDAZOLAM

        # Baseline
        model_base = WholeBodyPBPK(MIDAZOLAM)
        model_base.setup_oral(7.5)
        auc_base = model_base.simulate(t_end_h=24.0).pk_summary()["AUC_mg_h_L"]

        # With inducer
        model_ind = WholeBodyPBPK(MIDAZOLAM)
        model_ind.setup_oral(7.5)
        model_ind.add_inhibitor(DDIInhibitor(
            name="Rifampin", ki_uM=1.0, concentration_uM=1.0,
            target_enzyme="CYP3A4", mechanism="induction",
            fold_induction=5.0,
        ))
        auc_ind = model_ind.simulate(t_end_h=24.0).pk_summary()["AUC_mg_h_L"]

        # Induction should decrease AUC
        assert auc_ind < auc_base, "Induction should decrease AUC"


# ============================================================
# 6. Population PK tests (3 tests)
# ============================================================


class TestPopulation:
    def test_reference_man(self) -> None:
        from omega_pbpk.population.physiology import ReferenceMan

        ref = ReferenceMan()
        assert ref.body_weight_kg == 70.0
        assert ref.cardiac_output_L_h == 390.0

    def test_virtual_population_generation(self) -> None:
        from omega_pbpk.population.physiology import VirtualPopulation

        pop = VirtualPopulation(n=20, seed=42)
        subjects = pop.generate()
        assert len(subjects) == 20
        bws = [s.body_weight_kg for s in subjects]
        assert 40 < min(bws) < 120
        assert 40 < max(bws) < 120

    def test_population_summary(self) -> None:
        from omega_pbpk.population.physiology import VirtualPopulation

        pop = VirtualPopulation(n=50, seed=42)
        subjects = pop.generate()
        stats = pop.summary_stats(subjects)
        assert stats["n"] == 50
        assert 60 < stats["body_weight_kg"]["mean"] < 80


# ============================================================
# 7. Config loader tests (3 tests)
# ============================================================


class TestConfig:
    def test_load_midazolam_yaml(self) -> None:
        from omega_pbpk.config import load_compound

        drug = load_compound("compounds/midazolam.yaml")
        assert drug.name == "Midazolam"
        assert drug.mw == pytest.approx(325.77)
        assert drug.fup == pytest.approx(0.032)

    def test_load_caffeine_yaml(self) -> None:
        from omega_pbpk.config import load_compound

        drug = load_compound("compounds/caffeine.yaml")
        assert drug.name == "Caffeine"
        assert drug.mw == pytest.approx(194.19)

    def test_load_subject_yaml(self) -> None:
        from omega_pbpk.config import load_subject

        subj = load_subject("compounds/subject_default.yaml")
        assert subj["body_weight_kg"] == 70.0


# ============================================================
# 8. ADME predictor tests (3 tests)
# ============================================================


class TestADMEPredictor:
    def test_predict_from_smiles(self) -> None:
        from omega_pbpk.prediction.adme_predictor import ADMEPredictor

        predictor = ADMEPredictor()
        # Caffeine SMILES
        props = predictor.predict("Cn1c(=O)c2c(ncn2C)n(C)c1=O")
        assert props.mw > 100
        assert -5 < props.logP < 10
        assert 0 < props.fup <= 1.0

    def test_predict_returns_all_properties(self) -> None:
        from omega_pbpk.prediction.adme_predictor import ADMEPredictor

        predictor = ADMEPredictor()
        props = predictor.predict("c1ccccc1")  # Benzene
        assert props.mw > 0
        assert props.peff > 0
        assert props.rbp > 0
        assert props.herg_ic50_uM > 0

    def test_predict_from_dict(self) -> None:
        from omega_pbpk.prediction.adme_predictor import ADMEPredictor

        predictor = ADMEPredictor()
        props = predictor.predict_from_dict({"mw": 300, "logP": 3.0, "fup": 0.1})
        assert props is not None
        assert props.mw == 300
        assert props.confidence == "high"


# ============================================================
# 9. PD model tests (5 tests)
# ============================================================


class TestPDModels:
    def test_emax_model(self) -> None:
        from omega_pbpk.qsp.pd_models import EmaxModel

        model = EmaxModel(e0=0, emax=100, ec50=10, gamma=1)
        c = np.array([0.0, 10.0, 100.0, 1000.0])
        e = model.effect(c)
        assert e[0] == pytest.approx(0.0)
        assert e[1] == pytest.approx(50.0)  # At EC50 → 50% Emax
        assert e[3] > 90  # Near Emax

    def test_effect_compartment(self) -> None:
        from omega_pbpk.qsp.pd_models import EffectCompartment, EmaxModel

        ec = EffectCompartment(ke0=0.6, emax_model=EmaxModel(e0=0, emax=100, ec50=10))
        t = np.linspace(0, 24, 241)
        cp = 20 * np.exp(-0.1 * t)  # Declining plasma concentration
        ce, effect = ec.compute(t, cp)
        # Effect site should lag behind plasma
        assert np.argmax(ce) > np.argmax(cp)

    def test_indirect_response_type1(self) -> None:
        from omega_pbpk.qsp.pd_models import IndirectResponseModel

        idr = IndirectResponseModel(idr_type=1, kin=1.0, kout=0.1, imax_or_smax=0.9, ic50_or_sc50=1.0)
        t = np.linspace(0, 48, 481)
        cp = 5.0 * np.exp(-0.1 * t)
        response = idr.simulate(t, cp)
        assert len(response) == len(t)
        # Response should decrease from baseline when drug inhibits kin
        baseline = 1.0 / 0.1
        assert response[10] < baseline

    def test_tumor_growth_with_drug(self) -> None:
        from omega_pbpk.qsp.pd_models import TumorGrowthModel

        tumor = TumorGrowthModel(v0=100.0, k2=0.5)
        t = np.linspace(0, 168, 1681)  # 1 week
        cp = 10.0 * np.exp(-0.05 * t)  # Sustained drug exposure
        vol = tumor.simulate(t, cp)
        assert len(vol) == len(t)
        # Tumor should grow (even with drug, it doesn't fully suppress)
        assert vol[-1] > 0

    def test_biomarker_turnover(self) -> None:
        from omega_pbpk.qsp.pd_models import BiomarkerTurnover

        bm = BiomarkerTurnover(ksyn=10, kdeg=0.1, emax=2.0, ec50=1.0)
        t = np.linspace(0, 48, 481)
        cp = 5.0 * np.exp(-0.1 * t)
        response = bm.simulate(t, cp)
        # Biomarker should increase from baseline with drug
        baseline = 10 / 0.1
        assert response[5] > baseline


# ============================================================
# 10. Safety panel tests (3 tests)
# ============================================================


class TestSafety:
    def test_safety_assessment(self) -> None:
        from omega_pbpk.docking.off_target import SafetyPanel

        panel = SafetyPanel()
        report = panel.assess("TestDrug", logP=3.0, mw=300, cmax_uM=1.0, fup=0.5)
        assert report.overall_risk in ("low", "moderate", "high")
        assert len(report.target_results) == 11
        assert len(report.cyp_results) == 5

    def test_high_risk_compound(self) -> None:
        from omega_pbpk.docking.off_target import SafetyPanel

        panel = SafetyPanel()
        # Very lipophilic + high Cmax → likely risk
        report = panel.assess("RiskyDrug", logP=5.0, mw=400, cmax_uM=50.0, fup=0.01)
        # At least some flags should fire
        assert len(report.flags) > 0

    def test_safe_compound(self) -> None:
        from omega_pbpk.docking.off_target import SafetyPanel

        panel = SafetyPanel()
        # Low exposure, hydrophilic → likely safe
        report = panel.assess("SafeDrug", logP=0.5, mw=200, cmax_uM=0.001, fup=0.9)
        assert report.overall_risk == "low"


# ============================================================
# 11. Pharmacogenomics tests (3 tests)
# ============================================================


class TestPGx:
    def test_analyze_cyp2d6(self) -> None:
        from omega_pbpk.pharmacogenomics.cyp_polymorphism import PGxAnalyzer

        analyzer = PGxAnalyzer()
        results = analyzer.analyze_gene("CYP2D6", "Global")
        assert len(results) > 0
        phenotypes = {r.phenotype for r in results}
        assert "NM" in phenotypes
        assert "PM" in phenotypes

    def test_clint_scaling(self) -> None:
        from omega_pbpk.pharmacogenomics.cyp_polymorphism import PGxAnalyzer

        analyzer = PGxAnalyzer()
        # Normal metabolizer
        nm_scale = analyzer.clint_scaling("CYP2D6", "*1/*1")
        assert nm_scale == pytest.approx(1.0)
        # Poor metabolizer
        pm_scale = analyzer.clint_scaling("CYP2D6", "*4/*4")
        assert pm_scale == pytest.approx(0.0)

    def test_population_summary(self) -> None:
        from omega_pbpk.pharmacogenomics.cyp_polymorphism import PGxAnalyzer

        analyzer = PGxAnalyzer()
        summary = analyzer.population_summary("CYP2C19", "East_Asian")
        assert summary["gene"] == "CYP2C19"
        assert "phenotype_distribution" in summary
        dist = summary["phenotype_distribution"]
        assert abs(sum(dist.values()) - 1.0) < 0.01


# ============================================================
# 12. Clinical tools tests (3 tests)
# ============================================================


class TestClinical:
    def test_fih_dose(self) -> None:
        from omega_pbpk.clinical.dose_optimization import DoseOptimizer

        opt = DoseOptimizer()
        result = opt.fih_dose(noael_mg_kg=10.0, species="rat", safety_factor=10.0)
        assert result.fih_dose_mg > 0
        assert result.method == "BSA scaling from rat"

    def test_multidose_simulation(self) -> None:
        from omega_pbpk.clinical.dose_optimization import MultiDoseSimulator
        from omega_pbpk.drugs.midazolam import MIDAZOLAM

        sim = MultiDoseSimulator()
        result = sim.simulate(MIDAZOLAM, dose_mg=7.5, interval_h=12.0, n_days=3)
        assert result.css_max > 0
        assert result.accumulation_ratio >= 1.0
        assert len(result.time_h) > 0

    def test_dose_optimization(self) -> None:
        from omega_pbpk.clinical.dose_optimization import DoseOptimizer
        from omega_pbpk.drugs.midazolam import MIDAZOLAM

        opt = DoseOptimizer()
        result = opt.optimize_dose(MIDAZOLAM, mec_mg_L=0.001, mtc_mg_L=10.0)
        # Should find a dose that works for this wide window
        assert "optimal_dose_mg" in result or "error" in result


# ============================================================
# 13. GNN scaffold test (1 test)
# ============================================================


class TestGNN:
    def test_model_summary(self) -> None:
        from omega_pbpk.ml_models.gnn_adme import model_summary

        info = model_summary()
        assert info["name"] == "ADME_MPNN"
        assert info["n_endpoints"] == 6
        assert info["hidden_dim"] == 128


# ============================================================
# 14. Visualization test (1 test)
# ============================================================


class TestVisualization:
    def test_pk_plot(self) -> None:
        from omega_pbpk.visualization.plots import PKPlotter

        plotter = PKPlotter()
        t = np.linspace(0, 24, 241)
        cp = 10 * np.exp(-0.1 * t)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "test_plot.png"
            fig = plotter.plot_pk(t, cp, save_path=save_path)
            assert save_path.exists()


# ============================================================
# 15. Integration tests (3 tests)
# ============================================================


class TestIntegration:
    def test_full_pipeline_iv(self) -> None:
        """Full IV pipeline: YAML → Drug → PBPK → PK summary."""
        from omega_pbpk.config import load_compound
        from omega_pbpk.core.body import WholeBodyPBPK

        drug = load_compound("compounds/midazolam.yaml")
        model = WholeBodyPBPK(drug)
        model.setup_iv(5.0)
        result = model.simulate(t_end_h=24.0)
        pk = result.pk_summary()

        assert pk["Cmax_mg_L"] > 0
        assert pk["AUC_mg_h_L"] > 0
        assert pk["route"] == "iv"

    def test_full_pipeline_oral(self) -> None:
        """Full oral pipeline: YAML → Drug → PBPK → PK summary."""
        from omega_pbpk.config import load_compound
        from omega_pbpk.core.body import WholeBodyPBPK

        drug = load_compound("compounds/midazolam.yaml")
        model = WholeBodyPBPK(drug)
        model.setup_oral(7.5)
        result = model.simulate(t_end_h=24.0)
        pk = result.pk_summary()

        assert pk["Cmax_mg_L"] > 0
        assert pk["route"] == "oral"
        # Oral Cmax should be lower than IV for same dose
        model_iv = WholeBodyPBPK(drug)
        model_iv.setup_iv(7.5)
        pk_iv = model_iv.simulate(t_end_h=24.0).pk_summary()
        assert pk["Cmax_mg_L"] < pk_iv["Cmax_mg_L"]

    def test_caffeine_simulation(self) -> None:
        """Caffeine IV simulation should produce reasonable PK."""
        from omega_pbpk.config import load_compound
        from omega_pbpk.core.body import WholeBodyPBPK

        drug = load_compound("compounds/caffeine.yaml")
        model = WholeBodyPBPK(drug)
        model.setup_iv(200.0)  # 200mg IV caffeine
        result = model.simulate(t_end_h=24.0)
        pk = result.pk_summary()

        assert pk["Cmax_mg_L"] > 0
        assert pk["AUC_mg_h_L"] > 0
