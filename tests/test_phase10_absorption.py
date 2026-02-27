"""Phase 10 — Advanced absorption model tests.

Tests cover:
- GISegment data integrity
- AbsorptionModel: solubility_at_ph (Henderson-Hasselbalch)
- AbsorptionModel: peff_at_ph (ionisation correction)
- AbsorptionModel: segment_ka (dissolution factor, pH-dependency)
- AbsorptionModel: transit time multipliers (food effect)
- FoodEffect: gastric delay, bile-salt solubilisation
- WholeBodyPBPK: fed_state parameter works
- WholeBodyPBPK: fed state delays gastric emptying → delayed Tmax
- WholeBodyPBPK: ionisable acid/base show pH-dependent absorption
- WholeBodyPBPK: dissolution-limited drug (low solubility) has lower F
"""

import pytest

from omega_pbpk.core.absorption import (
    _GI_SEGMENT_MAP,
    GI_SEGMENTS,
    AbsorptionModel,
    FoodEffect,
)
from omega_pbpk.core.body import WholeBodyPBPK
from omega_pbpk.drugs.drug import Drug

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def neutral_drug():
    """Neutral lipophilic drug — pH-independent absorption."""
    return Drug(
        name="Neutral",
        mw=400.0,
        logP=2.5,
        pka=[],  # no ionisation
        drug_type="neutral",
        fup=0.3,
        rbp=1.0,
        clint_hepatic_L_per_h=3.0,
        clr_L_per_h=0.5,
        solubility_mg_mL=0.5,
        peff=3.0,
        particle_radius_um=25.0,
        route="oral",
    )


@pytest.fixture
def weak_acid():
    """Weak acid (pKa 4.5) — absorption favoured at low pH."""
    return Drug(
        name="WeakAcid",
        mw=350.0,
        logP=1.0,
        pka=[4.5],
        drug_type="monoprotic_acid",
        fup=0.5,
        rbp=1.0,
        clint_hepatic_L_per_h=2.0,
        solubility_mg_mL=1.0,
        peff=2.5,
        particle_radius_um=15.0,
        route="oral",
    )


@pytest.fixture
def weak_base():
    """Weak base (pKa 8.5) — absorption favoured at high pH."""
    return Drug(
        name="WeakBase",
        mw=380.0,
        logP=2.0,
        pka=[8.5],
        drug_type="monoprotic_base",
        fup=0.4,
        rbp=1.0,
        clint_hepatic_L_per_h=2.0,
        solubility_mg_mL=0.8,
        peff=2.0,
        particle_radius_um=20.0,
        route="oral",
    )


@pytest.fixture
def poorly_soluble():
    """BCS II drug — low solubility, dissolution-limited absorption."""
    return Drug(
        name="PoorlySoluble",
        mw=450.0,
        logP=4.0,
        pka=[],
        drug_type="neutral",
        fup=0.1,
        rbp=1.0,
        clint_hepatic_L_per_h=5.0,
        solubility_mg_mL=0.001,  # 1 µg/mL — very poorly soluble
        peff=5.0,  # high permeability
        particle_radius_um=50.0,
        route="oral",
    )


# ---------------------------------------------------------------------------
# GI_SEGMENTS data integrity
# ---------------------------------------------------------------------------


class TestGISegmentsData:
    def test_eight_segments(self):
        assert len(GI_SEGMENTS) == 8

    def test_all_segment_names_unique(self):
        names = [s.name for s in GI_SEGMENTS]
        assert len(names) == len(set(names))

    def test_all_ph_values_positive(self):
        for seg in GI_SEGMENTS:
            assert seg.ph_fasted > 0
            assert seg.ph_fed > 0

    def test_ph_increases_from_stomach_to_ileum(self):
        segs = {s.name: s for s in GI_SEGMENTS}
        assert segs["stomach"].ph_fasted < segs["duodenum"].ph_fasted
        assert segs["duodenum"].ph_fasted < segs["jejunum1"].ph_fasted
        assert segs["jejunum1"].ph_fasted <= segs["ileum1"].ph_fasted

    def test_fed_ph_differs_from_fasted_in_stomach(self):
        segs = _GI_SEGMENT_MAP
        assert segs["stomach"].ph_fed > segs["stomach"].ph_fasted

    def test_all_volumes_positive(self):
        for seg in GI_SEGMENTS:
            assert seg.volume_fasted_L > 0
            assert seg.volume_fed_L > 0

    def test_stomach_no_absorption(self):
        stomach = _GI_SEGMENT_MAP["stomach"]
        assert stomach.ka_fraction == 0.0

    def test_jejunum_highest_ka_fraction(self):
        segs = _GI_SEGMENT_MAP
        assert segs["jejunum1"].ka_fraction >= segs["ileum3"].ka_fraction
        assert segs["jejunum1"].ka_fraction >= segs["colon"].ka_fraction

    def test_lookup_by_name(self):
        for seg in GI_SEGMENTS:
            assert _GI_SEGMENT_MAP[seg.name] is seg


# ---------------------------------------------------------------------------
# AbsorptionModel.solubility_at_ph
# ---------------------------------------------------------------------------


class TestSolubilityAtPh:
    def test_neutral_drug_ph_independent(self, neutral_drug):
        am = AbsorptionModel(neutral_drug)
        s_low = am.solubility_at_ph(2.0)
        s_high = am.solubility_at_ph(8.0)
        # For neutral drug, small change due to bile (no bile w/o segment param)
        assert s_low == pytest.approx(s_high, rel=0.01)

    def test_weak_acid_higher_solubility_at_high_ph(self, weak_acid):
        am = AbsorptionModel(weak_acid)
        s_low = am.solubility_at_ph(1.5)  # stomach
        s_high = am.solubility_at_ph(7.0)  # ileum
        assert s_high > s_low

    def test_weak_acid_at_ph_equals_pka_doubled(self, weak_acid):
        """At pH = pKa: ionized fraction = 50%, S_pH = 2 × S0."""
        am = AbsorptionModel(weak_acid)
        s0 = weak_acid.solubility_mg_mL * 1000.0
        s_at_pka = am.solubility_at_ph(4.5)
        assert s_at_pka == pytest.approx(2 * s0, rel=0.01)

    def test_weak_base_higher_solubility_at_low_ph(self, weak_base):
        am = AbsorptionModel(weak_base)
        s_stomach = am.solubility_at_ph(1.5)
        s_ileum = am.solubility_at_ph(7.0)
        assert s_stomach > s_ileum

    def test_solubility_never_below_s0(self, weak_acid):
        am = AbsorptionModel(weak_acid)
        s0 = weak_acid.solubility_mg_mL * 1000.0
        for ph in [1.0, 2.0, 3.0, 4.5, 7.0, 9.0]:
            assert am.solubility_at_ph(ph) >= s0

    def test_bile_salt_increases_solubility_lipophilic(self):
        """Lipophilic drug (logP=4) should be solubilised by bile salts."""
        drug = Drug(
            name="Lipophilic",
            mw=400.0,
            logP=4.0,
            pka=[],
            drug_type="neutral",
            fup=0.1,
            solubility_mg_mL=0.01,
            peff=3.0,
        )
        am = AbsorptionModel(
            drug, fed_state=True, food_effect=FoodEffect(solubility_enhancement_per_logP=0.05)
        )
        seg_jejunum = _GI_SEGMENT_MAP["jejunum1"]
        s_no_bile = am.solubility_at_ph(6.0)  # no segment → no bile
        s_with_bile = am.solubility_at_ph(6.0, segment=seg_jejunum)
        assert s_with_bile > s_no_bile

    def test_hydrophilic_no_bile_enhancement(self):
        """logP < 2: no bile-salt solubilisation."""
        drug = Drug(
            name="Hydrophilic",
            mw=300.0,
            logP=0.5,
            pka=[],
            drug_type="neutral",
            fup=0.8,
            solubility_mg_mL=100.0,
            peff=2.0,
        )
        am = AbsorptionModel(
            drug, fed_state=True, food_effect=FoodEffect(solubility_enhancement_per_logP=0.05)
        )
        seg = _GI_SEGMENT_MAP["jejunum1"]
        s_no_bile = am.solubility_at_ph(6.0)
        s_with_bile = am.solubility_at_ph(6.0, segment=seg)
        # For logP ≤ 2, no enhancement (logP_factor = 0)
        assert s_with_bile == pytest.approx(s_no_bile, rel=1e-9)


# ---------------------------------------------------------------------------
# AbsorptionModel.peff_at_ph
# ---------------------------------------------------------------------------


class TestPeffAtPh:
    def test_neutral_drug_peff_constant(self, neutral_drug):
        am = AbsorptionModel(neutral_drug)
        peff_low = am.peff_at_ph(1.5)
        peff_high = am.peff_at_ph(8.0)
        assert peff_low == pytest.approx(peff_high, rel=1e-9)

    def test_weak_acid_higher_peff_at_low_ph(self, weak_acid):
        """Below pKa, weak acid is unionised → higher Peff."""
        am = AbsorptionModel(weak_acid)
        peff_below_pka = am.peff_at_ph(2.0)  # pH << pKa=4.5, mostly unionised
        peff_above_pka = am.peff_at_ph(7.0)  # pH >> pKa, mostly ionised
        assert peff_below_pka > peff_above_pka

    def test_weak_base_higher_peff_at_high_ph(self, weak_base):
        """Above pKa, weak base is unionised → higher Peff."""
        am = AbsorptionModel(weak_base)
        peff_high = am.peff_at_ph(10.0)  # pH >> pKa=8.5, mostly unionised
        peff_low = am.peff_at_ph(3.0)  # pH << pKa, mostly ionised
        assert peff_high > peff_low

    def test_peff_minimum_floor(self, weak_acid):
        """Even fully ionised drug has minimum Peff (10%)."""
        am = AbsorptionModel(weak_acid)
        # At pH 9 (>> pKa=4.5), mostly ionised but not zero
        peff_min = am.peff_at_ph(9.0)
        assert peff_min >= weak_acid.peff * 0.09  # at least near 10%

    def test_peff_max_equals_drug_peff(self, weak_acid):
        """At pH << pKa, Peff approaches full (unionised)."""
        am = AbsorptionModel(weak_acid)
        peff_unionised = am.peff_at_ph(0.5)
        assert peff_unionised == pytest.approx(weak_acid.peff, rel=0.01)


# ---------------------------------------------------------------------------
# AbsorptionModel.segment_ka
# ---------------------------------------------------------------------------


class TestSegmentKa:
    def test_stomach_ka_zero(self, neutral_drug):
        am = AbsorptionModel(neutral_drug)
        assert am.segment_ka("stomach") == pytest.approx(0.0, abs=1e-12)

    def test_ka_positive_for_jejunum(self, neutral_drug):
        am = AbsorptionModel(neutral_drug)
        ka = am.segment_ka("jejunum1")
        assert ka > 0.0

    def test_ka_decreases_colon_vs_jejunum(self, neutral_drug):
        am = AbsorptionModel(neutral_drug)
        ka_jej = am.segment_ka("jejunum1")
        ka_col = am.segment_ka("colon")
        assert ka_jej > ka_col

    def test_dissolution_limited_reduces_ka(self, poorly_soluble):
        """High dose of poorly soluble drug → dissolution factor reduces ka."""
        am = AbsorptionModel(poorly_soluble)
        ka_small = am.segment_ka("jejunum1", mass_mg=0.001)  # well below solubility
        ka_large = am.segment_ka("jejunum1", mass_mg=100.0)  # far above solubility
        assert ka_small > ka_large

    def test_ka_increases_with_peff(self):
        """Higher Peff → higher ka."""
        drug_lo = Drug(name="LoPeff", mw=300.0, peff=0.5, particle_radius_um=25.0)
        drug_hi = Drug(name="HiPeff", mw=300.0, peff=5.0, particle_radius_um=25.0)
        am_lo = AbsorptionModel(drug_lo)
        am_hi = AbsorptionModel(drug_hi)
        assert am_hi.segment_ka("jejunum1") > am_lo.segment_ka("jejunum1")

    def test_ka_increases_with_smaller_particles(self):
        """Smaller particles → larger surface area → faster dissolution → higher ka."""
        drug_big = Drug(name="BigParticle", mw=300.0, peff=2.0, particle_radius_um=100.0)
        drug_small = Drug(name="SmallParticle", mw=300.0, peff=2.0, particle_radius_um=10.0)
        am_big = AbsorptionModel(drug_big)
        am_small = AbsorptionModel(drug_small)
        assert am_small.segment_ka("jejunum1") > am_big.segment_ka("jejunum1")

    def test_all_ka_returns_eight_values(self, neutral_drug):
        am = AbsorptionModel(neutral_drug)
        kas = am.all_ka()
        assert len(kas) == 8

    def test_all_ka_non_negative(self, neutral_drug):
        am = AbsorptionModel(neutral_drug)
        for name, ka in am.all_ka().items():
            assert ka >= 0.0, f"Negative ka for {name}: {ka}"

    def test_supersaturation_increases_effective_ka(self, poorly_soluble):
        """Supersaturation ratio > 1 allows ka to be less reduced by dissolution."""
        am_no_ss = AbsorptionModel(poorly_soluble)
        fe_ss = FoodEffect(supersaturation_ratio=5.0)
        am_ss = AbsorptionModel(poorly_soluble, food_effect=fe_ss)
        # With supersaturation, more drug can be dissolved → higher ka for same mass
        ka_no_ss = am_no_ss.segment_ka("jejunum1", mass_mg=10.0)
        ka_ss = am_ss.segment_ka("jejunum1", mass_mg=10.0)
        assert ka_ss >= ka_no_ss


# ---------------------------------------------------------------------------
# AbsorptionModel.transit_time_multiplier
# ---------------------------------------------------------------------------


class TestTransitTimeMultiplier:
    def test_fasted_all_ones(self, neutral_drug):
        am = AbsorptionModel(neutral_drug, fed_state=False)
        mults = am.transit_time_multiplier()
        for _name, m in mults.items():
            assert m == pytest.approx(1.0, abs=1e-9)

    def test_fed_stomach_delayed(self, neutral_drug):
        fe = FoodEffect(gastric_delay_factor=2.5)
        am = AbsorptionModel(neutral_drug, fed_state=True, food_effect=fe)
        mults = am.transit_time_multiplier()
        assert mults["stomach"] == pytest.approx(2.5, rel=1e-9)

    def test_fed_intestine_factor_applied(self, neutral_drug):
        fe = FoodEffect(intestinal_transit_factor=1.2)
        am = AbsorptionModel(neutral_drug, fed_state=True, food_effect=fe)
        mults = am.transit_time_multiplier()
        assert mults["jejunum1"] == pytest.approx(1.2, rel=1e-9)

    def test_fed_default_intestinal_factor_one(self, neutral_drug):
        am = AbsorptionModel(neutral_drug, fed_state=True)
        mults = am.transit_time_multiplier()
        for name in ("duodenum", "jejunum1", "ileum1", "colon"):
            assert mults[name] == pytest.approx(1.0, rel=1e-9)


# ---------------------------------------------------------------------------
# WholeBodyPBPK integration
# ---------------------------------------------------------------------------


class TestFoodEffectIntegration:
    def _run(self, drug, fed=False, dose=100.0):
        model = WholeBodyPBPK(drug, body_weight=70.0, fed_state=fed)
        model.setup_oral(dose_mg=dose)
        return model.simulate(t_end_h=24.0)

    def test_model_runs_fasted(self, neutral_drug):
        result = self._run(neutral_drug, fed=False)
        assert result.pk_summary()["AUC_mg_h_L"] > 0

    def test_model_runs_fed(self, neutral_drug):
        result = self._run(neutral_drug, fed=True)
        assert result.pk_summary()["AUC_mg_h_L"] > 0

    def test_fed_delays_tmax_for_oral_drug(self):
        """Fed state (delayed gastric emptying) should shift Tmax later."""
        drug = Drug(
            name="FastAbsorb",
            mw=300.0,
            logP=2.0,
            pka=[],
            drug_type="neutral",
            fup=0.5,
            rbp=1.0,
            clint_hepatic_L_per_h=1.0,
            clr_L_per_h=0.0,
            solubility_mg_mL=10.0,
            peff=5.0,
            particle_radius_um=10.0,
            route="oral",
        )
        fe = FoodEffect(gastric_delay_factor=4.0)  # strong delay
        r_fasted = WholeBodyPBPK(drug, fed_state=False)
        r_fasted.setup_oral(100.0)
        res_fasted = r_fasted.simulate(t_end_h=12.0, dt_h=0.05)

        r_fed = WholeBodyPBPK(drug, fed_state=True, food_effect=fe)
        r_fed.setup_oral(100.0)
        res_fed = r_fed.simulate(t_end_h=12.0, dt_h=0.05)

        tmax_fasted = res_fasted.pk_summary()["Tmax_h"]
        tmax_fed = res_fed.pk_summary()["Tmax_h"]
        assert tmax_fed >= tmax_fasted  # fed state: delayed gastric emptying → later Tmax

    def test_poorly_soluble_lower_auc_than_well_soluble(self):
        """BCS II drug (low solubility) should achieve lower AUC than BCS I."""
        base = dict(
            mw=400.0,
            logP=3.0,
            pka=[],
            drug_type="neutral",
            fup=0.3,
            rbp=1.0,
            clint_hepatic_L_per_h=2.0,
            clr_L_per_h=0.0,
            peff=5.0,
            particle_radius_um=20.0,
            route="oral",
        )
        well_soluble = Drug(name="BCS_I", solubility_mg_mL=10.0, **base)
        poorly_soluble = Drug(name="BCS_II", solubility_mg_mL=0.002, **base)

        r_well = WholeBodyPBPK(well_soluble, fed_state=False)
        r_well.setup_oral(100.0)
        auc_well = r_well.simulate(t_end_h=24.0).pk_summary()["AUC_mg_h_L"]

        r_poor = WholeBodyPBPK(poorly_soluble, fed_state=False)
        r_poor.setup_oral(100.0)
        auc_poor = r_poor.simulate(t_end_h=24.0).pk_summary()["AUC_mg_h_L"]

        assert auc_well > auc_poor

    def test_weak_acid_ph_effect_on_absorption(self, weak_acid):
        """A weak acid absorbs differently at gastric pH 1.5 vs 4.5 (fed)."""
        r_fasted = WholeBodyPBPK(weak_acid, fed_state=False)
        r_fasted.setup_oral(100.0)
        pk_fasted = r_fasted.simulate(t_end_h=24.0).pk_summary()

        r_fed = WholeBodyPBPK(weak_acid, fed_state=True)
        r_fed.setup_oral(100.0)
        pk_fed = r_fed.simulate(t_end_h=24.0).pk_summary()

        # Both should run without error and produce non-zero AUC
        assert pk_fasted["AUC_mg_h_L"] > 0
        assert pk_fed["AUC_mg_h_L"] > 0
        # Fasted state has lower gastric pH → more unionised acid → different absorption
        # (either higher or lower AUC depending on balance of effects)
        assert pk_fasted["AUC_mg_h_L"] != pytest.approx(pk_fed["AUC_mg_h_L"], rel=0.001)

    def test_mass_balance_with_advanced_absorption(self, neutral_drug):
        """Mass balance should be conserved under advanced absorption model."""
        model = WholeBodyPBPK(neutral_drug, fed_state=False)
        model.setup_oral(100.0)
        result = model.simulate(t_end_h=48.0)
        total_at_end = result.amounts[-1].sum()
        assert total_at_end <= 100.0 * 1.05  # ≤ dose + 5% tolerance

    def test_fed_state_attribute_stored(self, neutral_drug):
        model = WholeBodyPBPK(neutral_drug, fed_state=True)
        assert model.fed_state is True

    def test_fasted_state_attribute_stored(self, neutral_drug):
        model = WholeBodyPBPK(neutral_drug, fed_state=False)
        assert model.fed_state is False

    def test_absorption_model_attached(self, neutral_drug):
        model = WholeBodyPBPK(neutral_drug)
        assert isinstance(model._absorption, AbsorptionModel)

    def test_transit_multipliers_computed(self, neutral_drug):
        model = WholeBodyPBPK(neutral_drug)
        assert isinstance(model._transit_multipliers, dict)
        assert "stomach" in model._transit_multipliers
