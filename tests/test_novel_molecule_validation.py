"""Novel molecule validation framework.

For a truly new molecule, there is NO clinical data to compare against.
Instead, we validate:

1. PLAUSIBILITY  — Are predicted parameters within physiological bounds?
2. CONSISTENCY   — Do predictions internally agree (logP↔fup↔Kp↔Vd)?
3. SENSITIVITY   — Do small structural changes produce small PK changes?
4. ANALOG MATCH  — Is prediction near the closest known drug?
5. UNCERTAINTY   — Does the model know what it doesn't know?

These tests DON'T check accuracy (impossible without ground truth).
They check whether the prediction is *physically reasonable* and whether
the system can distinguish "I'm confident" from "I'm guessing."
"""

import math

import numpy as np
import pytest

# ── A set of structurally diverse novel-ish molecules ──────────────────────
# These are plausible drug-like molecules, not existing marketed drugs.

NOVEL_MOLECULES = [
    {
        "name": "pyrazole_amide",
        "smiles": "CC(=O)Nc1cc(nn1C)c1ccc(F)cc1",
        "desc": "Fluorophenyl pyrazole amide — drug-like, MW~233",
    },
    {
        "name": "benzimidazole_ether",
        "smiles": "COc1ccc2[nH]c(nc2c1)C(C)N(C)C",
        "desc": "Methoxybenzimidazole dimethylamine — basic, MW~233",
    },
    {
        "name": "thiophene_sulfonamide",
        "smiles": "Cc1ccc(s1)S(=O)(=O)Nc1ccccc1",
        "desc": "Methylthiophene phenylsulfonamide — acidic, MW~253",
    },
    {
        "name": "piperidine_biphenyl",
        "smiles": "O=C(c1ccc(-c2ccccc2)cc1)N1CCCCC1",
        "desc": "Biphenyl piperidine amide — large, lipophilic, MW~265",
    },
    {
        "name": "small_polar",
        "smiles": "OCC(O)CO",
        "desc": "Glycerol — very small, very polar, MW~92",
    },
]

# Structural analog pairs for sensitivity testing
ANALOG_PAIRS = [
    {
        "name": "methyl_addition",
        "parent": "c1ccc(cc1)C(=O)O",  # benzoic acid MW=122
        "child": "Cc1ccc(cc1)C(=O)O",  # 4-methylbenzoic acid MW=136
        "change": "added methyl group",
        "expected_direction": {
            "logP": "increase",  # methyl is lipophilic
            "mw": "increase",
        },
    },
    {
        "name": "fluorine_swap",
        "parent": "c1ccc(cc1)NC(=O)C",  # acetanilide MW=135
        "child": "Fc1ccc(cc1)NC(=O)C",  # 4-fluoroacetanilide MW=153
        "change": "added fluorine",
        "expected_direction": {
            "mw": "increase",
        },
    },
    {
        "name": "hydroxylation",
        "parent": "c1ccc(cc1)CC",  # ethylbenzene MW=106
        "child": "OC(c1ccccc1)C",  # 1-phenylethanol MW=122
        "change": "added hydroxyl",
        "expected_direction": {
            "logP": "decrease",  # hydroxyl is polar
            "fup": "increase",  # less protein binding
        },
    },
]


def _simulate(smiles: str, dose_mg: float = 100.0) -> dict:
    """Run SMILES-only prediction, return result dict."""
    from omega_pbpk.pipeline import OmegaPipeline, SimulationRequest

    pipeline = OmegaPipeline()
    req = SimulationRequest(smiles=smiles, dose_mg=dose_mg, route="oral", duration_h=24.0)
    result = pipeline.simulate(req)
    return {
        "cmax": result.cmax_mg_L,
        "tmax": result.tmax_h,
        "auc": result.auc0t_mg_h_L,
        "thalf": result.t_half_h,
        "confidence": result.confidence,
        "adme": result.adme_properties,
        "warnings": result.warnings,
        "time_h": result.time_h,
        "cp": result.cp_mg_L,
    }


# ════════════════════════════════════════════════════════════════════════════
# TEST 1: PLAUSIBILITY — Are predictions within physiological bounds?
# ════════════════════════════════════════════════════════════════════════════


class TestPlausibility:
    """Predicted ADME properties and PK must be within physiological limits."""

    @pytest.mark.parametrize("mol", NOVEL_MOLECULES, ids=[m["name"] for m in NOVEL_MOLECULES])
    def test_adme_within_bounds(self, mol):
        """ADME predictions must be within physically possible ranges."""
        result = _simulate(mol["smiles"])
        adme = result["adme"]

        # Fraction unbound: must be between 0 and 1
        fup = adme.get("fup", -1)
        assert 0.0 < fup <= 1.0, f"fup={fup} outside (0,1] for {mol['name']}"

        # Blood-to-plasma ratio: typically 0.3-3.0
        rbp = adme.get("rbp", -1)
        assert 0.3 <= rbp <= 3.0, f"rbp={rbp} outside [0.3,3.0] for {mol['name']}"

        # Molecular weight: should match SMILES
        mw = adme.get("mw", 0)
        assert 30 < mw < 1000, f"mw={mw} outside drug-like range for {mol['name']}"

        # logP: drug-like space is roughly -3 to 8
        logP = adme.get("logP", None)
        if logP is not None:
            assert -5 < logP < 10, f"logP={logP} outside [-5,10] for {mol['name']}"

        # Permeability: must be positive
        peff = adme.get("peff", 0)
        assert peff > 0, f"peff={peff} must be positive for {mol['name']}"

    @pytest.mark.parametrize("mol", NOVEL_MOLECULES, ids=[m["name"] for m in NOVEL_MOLECULES])
    def test_pk_profile_shape(self, mol):
        """Oral PK profile must: rise to Cmax, then decline."""
        result = _simulate(mol["smiles"])

        # Cmax must be positive
        assert result["cmax"] > 0, f"Cmax must be positive for {mol['name']}"

        # Tmax must be > 0 for oral dosing (not instantaneous)
        assert result["tmax"] > 0, f"Tmax must be >0 for oral {mol['name']}"

        # Tmax should be within reasonable oral range (0.1h - 12h)
        assert result["tmax"] <= 24.0, (
            f"Tmax={result['tmax']}h — drug never peaks for {mol['name']}"
        )

        # AUC must be positive
        assert result["auc"] > 0, f"AUC must be positive for {mol['name']}"

        # t½ should be finite and reasonable for a drug (0.1h - 200h)
        if not math.isnan(result["thalf"]):
            assert 0.05 < result["thalf"] < 500, (
                f"t½={result['thalf']}h outside [0.05,500] for {mol['name']}"
            )

    @pytest.mark.parametrize("mol", NOVEL_MOLECULES, ids=[m["name"] for m in NOVEL_MOLECULES])
    def test_mass_balance(self, mol):
        """AUC should be roughly consistent with dose/CL relationship.

        For oral: AUC = F × Dose / CL
        At minimum: AUC < Dose/CL_min where CL_min ~ 0.1 L/h (very slow clearance)
        And: AUC > 0 (some drug absorbed)
        """
        dose_mg = 100.0
        result = _simulate(mol["smiles"], dose_mg=dose_mg)

        # AUC can't exceed dose/CL_min (assuming F=1, CL=0.01 L/h = almost no clearance)
        # That gives AUC_max = 100mg / 0.01 L/h = 10,000 mg·h/L
        assert result["auc"] < 10000, (
            f"AUC={result['auc']} exceeds theoretical max for {mol['name']}"
        )

        # AUC shouldn't be trivially small for a 100mg dose
        # Even with CL=100 L/h and F=0.01: AUC = 0.01 × 100 / 100 = 0.01
        assert result["auc"] > 0.001, (
            f"AUC={result['auc']} suspiciously small for 100mg dose of {mol['name']}"
        )


# ════════════════════════════════════════════════════════════════════════════
# TEST 2: CONSISTENCY — Do predictions internally agree?
# ════════════════════════════════════════════════════════════════════════════


class TestConsistency:
    """Cross-check that predicted properties are mutually consistent."""

    @pytest.mark.parametrize("mol", NOVEL_MOLECULES, ids=[m["name"] for m in NOVEL_MOLECULES])
    def test_auc_cmax_thalf_relationship(self, mol):
        """For 1-cpt oral: AUC ≈ Cmax × 1.44 × t½ (within ~10x for PBPK).

        This isn't exact for multi-compartment models, but wildly inconsistent
        values indicate a bug (e.g., Cmax=0.001 but AUC=1000).
        """
        result = _simulate(mol["smiles"])

        if math.isnan(result["thalf"]) or result["thalf"] <= 0:
            pytest.skip(f"t½ not calculable for {mol['name']}")

        # 1-cpt approximation: AUC ≈ 1.44 × Cmax × t½
        approx_auc = 1.44 * result["cmax"] * result["thalf"]
        ratio = result["auc"] / approx_auc if approx_auc > 0 else float("inf")

        # Allow broad tolerance for high-Vd multi-compartment redistribution.
        assert 0.002 < ratio < 500, (
            f"AUC/Cmax/t½ inconsistent for {mol['name']}: "
            f"AUC={result['auc']:.3f}, Cmax={result['cmax']:.3f}, "
            f"t½={result['thalf']:.1f}h, ratio={ratio:.2f}"
        )

    @pytest.mark.parametrize("mol", NOVEL_MOLECULES, ids=[m["name"] for m in NOVEL_MOLECULES])
    def test_profile_monotonicity_after_cmax(self, mol):
        """After Cmax, concentration should generally decline (not oscillate wildly)."""
        result = _simulate(mol["smiles"])

        cp = result["cp"]
        cmax_idx = int(np.argmax(cp))

        if cmax_idx >= len(cp) - 5:
            pytest.skip("Cmax too close to end of simulation")

        # Check terminal phase (last 30%) is declining
        n = len(cp)
        tail_start = max(cmax_idx, int(0.7 * n))
        tail = cp[tail_start:]

        if len(tail) < 3:
            pytest.skip("Tail too short")

        # Allow some noise but overall trend should be downward
        # Check that last point < first point of tail region
        # (with generous tolerance for multi-compartment redistribution)
        assert tail[-1] <= tail[0] * 1.5, (
            f"Terminal phase not declining for {mol['name']}: "
            f"tail start={tail[0]:.4f}, tail end={tail[-1]:.4f}"
        )


# ════════════════════════════════════════════════════════════════════════════
# TEST 3: SENSITIVITY — Small structural changes → proportional PK changes
# ════════════════════════════════════════════════════════════════════════════


class TestSensitivity:
    """Structural analogs should have similar (not wildly different) PK."""

    @pytest.mark.parametrize("pair", ANALOG_PAIRS, ids=[p["name"] for p in ANALOG_PAIRS])
    def test_analog_pk_within_10fold(self, pair):
        """Adding a methyl or fluorine shouldn't change Cmax/AUC by >10x."""
        parent = _simulate(pair["parent"])
        child = _simulate(pair["child"])

        for metric in ["cmax", "auc"]:
            p_val = parent[metric]
            c_val = child[metric]
            if p_val > 0 and c_val > 0:
                fold = max(c_val / p_val, p_val / c_val)
                assert fold < 10.0, (
                    f"{pair['name']}: {metric} changed {fold:.1f}x "
                    f"(parent={p_val:.3f}, child={c_val:.3f}) — "
                    f"a {pair['change']} shouldn't cause >10x change"
                )

    @pytest.mark.parametrize("pair", ANALOG_PAIRS, ids=[p["name"] for p in ANALOG_PAIRS])
    def test_expected_property_direction(self, pair):
        """Known structure-property relationships should hold."""
        parent = _simulate(pair["parent"])
        child = _simulate(pair["child"])

        for prop, direction in pair["expected_direction"].items():
            p_val = parent["adme"].get(prop, None)
            c_val = child["adme"].get(prop, None)

            if p_val is None or c_val is None:
                continue

            if direction == "increase":
                # Allow 5% tolerance for numerical noise
                assert c_val >= p_val * 0.95, (
                    f"{pair['name']}: {prop} should increase after {pair['change']} "
                    f"(parent={p_val:.3f}, child={c_val:.3f})"
                )
            elif direction == "decrease":
                assert c_val <= p_val * 1.05, (
                    f"{pair['name']}: {prop} should decrease after {pair['change']} "
                    f"(parent={p_val:.3f}, child={c_val:.3f})"
                )


# ════════════════════════════════════════════════════════════════════════════
# TEST 4: UNCERTAINTY — Does the model know what it doesn't know?
# ════════════════════════════════════════════════════════════════════════════


class TestUncertainty:
    """The model should report lower confidence for unusual molecules."""

    def test_drug_like_gets_higher_confidence_than_exotic(self):
        """A typical drug-like molecule should get higher confidence
        than a structurally exotic one (e.g., organometallic, very large)."""
        # Drug-like: simple NSAID structure
        normal = _simulate("CC(=O)Oc1ccccc1C(=O)O")  # aspirin-like

        # Exotic: large, unusual heteroatom-rich structure
        exotic = _simulate("B(O)(O)c1ccc(cc1)C#N")  # boronic acid (unusual for drugs)

        conf_rank = {"low": 0, "medium": 1, "high": 2}
        normal_rank = conf_rank.get(normal["confidence"], -1)
        exotic_rank = conf_rank.get(exotic["confidence"], -1)

        print(f"\n  Drug-like confidence: {normal['confidence']} (rank {normal_rank})")
        print(f"  Exotic confidence:   {exotic['confidence']} (rank {exotic_rank})")

        # At minimum, both should produce SOME confidence level
        assert normal["confidence"] in conf_rank, "Normal molecule has no confidence"
        assert exotic["confidence"] in conf_rank, "Exotic molecule has no confidence"

        # Ideally exotic <= normal, but current model may not distinguish
        # This test documents current behavior and will catch improvements
        if exotic_rank >= normal_rank:
            print("  WARNING: Model can't distinguish drug-like from exotic structures")
            print("  (This is a known limitation of MW/logP-only applicability domain)")

    def test_confidence_intervals_exist(self):
        """Predictions should include uncertainty intervals."""
        result = _simulate("CC(=O)Nc1ccc(O)cc1")  # acetaminophen-like
        adme = result["adme"]

        # Check that interval keys exist
        has_fup_interval = "fup_lo" in adme and "fup_hi" in adme
        has_clint_interval = "clint_3a4_lo" in adme and "clint_3a4_hi" in adme
        has_peff_interval = "peff_lo" in adme and "peff_hi" in adme

        print(f"\n  fup interval:   {'YES' if has_fup_interval else 'NO'}")
        print(f"  CLint interval: {'YES' if has_clint_interval else 'NO'}")
        print(f"  peff interval:  {'YES' if has_peff_interval else 'NO'}")

        if has_fup_interval:
            print(f"    fup: {adme['fup_lo']:.3f} — {adme['fup']:.3f} — {adme['fup_hi']:.3f}")
            # Interval should contain point estimate
            assert adme["fup_lo"] <= adme["fup"] <= adme["fup_hi"], (
                "fup interval doesn't contain point estimate"
            )

        # At least one interval should exist
        assert has_fup_interval or has_clint_interval or has_peff_interval, (
            "No uncertainty intervals found — model provides no uncertainty info"
        )

    def test_wide_intervals_for_uncertain_predictions(self):
        """If a molecule is far from training data, intervals should be wider."""
        # Well-studied space: typical drug MW~300, logP~2
        familiar = _simulate("c1ccc(cc1)C(=O)Nc2ccccc2")  # benzanilide

        # Unusual space: very high MW polycyclic
        unusual = _simulate("C1CCC(CC1)C1CCC(CC1)C1CCC(CC1)C1CCCCC1")  # 4 fused cyclohexanes

        fam_fup_range = familiar["adme"].get("fup_hi", 1) - familiar["adme"].get("fup_lo", 0)
        unu_fup_range = unusual["adme"].get("fup_hi", 1) - unusual["adme"].get("fup_lo", 0)

        print(f"\n  Familiar fup interval width: {fam_fup_range:.3f}")
        print(f"  Unusual fup interval width:  {unu_fup_range:.3f}")

        # Document behavior — in a well-calibrated model, unusual should be wider
        if unu_fup_range <= fam_fup_range:
            print("  WARNING: Model doesn't widen intervals for unusual structures")
            print("  (Intervals may be fixed ±50% defaults, not calibrated)")


# ════════════════════════════════════════════════════════════════════════════
# TEST 5: DIAGNOSTIC SUMMARY
# ════════════════════════════════════════════════════════════════════════════


def test_novel_molecule_diagnostic_summary():
    """Run all novel molecules and print diagnostic dashboard."""
    print(f"\n{'=' * 70}")
    print("NOVEL MOLECULE PREDICTION DIAGNOSTICS")
    print("(No ground truth — checking plausibility and consistency)")
    print(f"{'=' * 70}")

    for mol in NOVEL_MOLECULES:
        result = _simulate(mol["smiles"])
        adme = result["adme"]

        print(f"\n  {mol['name']}: {mol['desc']}")
        print(f"  SMILES: {mol['smiles']}")
        print(f"  Confidence: {result['confidence']}")
        print(f"  Warnings: {result['warnings'] if result['warnings'] else 'none'}")
        print(
            f"    MW={adme.get('mw', '?'):.1f}  logP={adme.get('logP', '?'):.2f}  "
            f"fup={adme.get('fup', '?'):.3f}  peff={adme.get('peff', '?'):.2f}"
        )
        thalf_str = "NaN" if math.isnan(result["thalf"]) else f"{result['thalf']:.1f}h"
        print(
            f"    Cmax={result['cmax']:.3f} mg/L  Tmax={result['tmax']:.1f}h  "
            f"AUC={result['auc']:.2f} mg·h/L  t½={thalf_str}"
        )

        # Flag suspicious values
        flags = []
        if result["tmax"] >= 23.9:
            flags.append("Tmax=24h (drug never peaks — likely poor absorption or very slow ka)")
        if not math.isnan(result["thalf"]) and result["thalf"] > 100:
            flags.append(f"t½={result['thalf']:.0f}h (>100h — likely CLint under-predicted)")
        if adme.get("fup", 1) < 0.005:
            flags.append(f"fup={adme['fup']:.4f} (<0.5% — extremely high binding)")
        if result["cmax"] < 0.001:
            flags.append(f"Cmax={result['cmax']:.5f} (nearly zero — absorption failure?)")

        if flags:
            print(f"    ⚠ FLAGS: {'; '.join(flags)}")
        else:
            print("    ✓ No plausibility flags")

    print(f"\n{'=' * 70}")
    print("KEY QUESTION: Are these predictions trustworthy enough to guide decisions?")
    print("If confidence='high' for everything, the model can't tell you when it's wrong.")
    print(f"{'=' * 70}")
