"""Integration tests: end-to-end SMILES -> simulation pipeline.

Tests cover:
  1. ADMEPredictor: property prediction from SMILES
  2. RDKitFeaturizer: molecular descriptor extraction
  3. Species physiology: human and rat physiological parameters
  4. Pediatric ontogeny: age-dependent CYP3A4 scaling
"""

from __future__ import annotations

import math

import pytest

# ---------------------------------------------------------------------------
# SMILES constants
# ---------------------------------------------------------------------------

MIDAZOLAM_SMILES = "CN1C(=O)CN=C(c2ccccc2Cl)c2cc(F)ccc21"
CAFFEINE_SMILES = "Cn1cnc2c1c(=O)n(C)c(=O)n2C"


# ---------------------------------------------------------------------------
# 1–2. ADMEPredictor tests
# ---------------------------------------------------------------------------


def _get_adme_predictor():
    """Import ADMEPredictor, skip if rdkit is not available."""
    pytest.importorskip("rdkit", reason="rdkit not installed")
    from omega_pbpk.prediction.adme_predictor import ADMEPredictor

    return ADMEPredictor()


@pytest.mark.integration
class TestADMEPredictorReturnsValidProperties:
    def test_adme_predictor_returns_valid_properties(self):
        """All predicted ADME fields are finite positive floats; key ranges checked."""
        predictor = _get_adme_predictor()
        props = predictor.predict(MIDAZOLAM_SMILES)

        # Every numeric field must be finite
        for field_name in (
            "mw",
            "logP",
            "logS",
            "peff",
            "fup",
            "rbp",
            "clint_3a4",
            "clint_2d6",
            "herg_ic50_uM",
        ):
            val = getattr(props, field_name)
            assert math.isfinite(val), f"{field_name} is not finite: {val}"

        # Specific range checks
        assert props.mw > 100, f"mw too small: {props.mw}"
        assert 0 < props.fup < 1, f"fup out of range: {props.fup}"
        assert props.herg_ic50_uM > 0, f"herg_ic50_uM not positive: {props.herg_ic50_uM}"

    def test_adme_predictor_confidence_field(self):
        """Confidence is one of the recognised levels."""
        predictor = _get_adme_predictor()
        props = predictor.predict(MIDAZOLAM_SMILES)
        assert props.confidence in {"low", "medium", "high"}, (
            f"Unexpected confidence value: {props.confidence}"
        )

    def test_adme_predictor_different_drugs_give_different_results(self):
        """Midazolam and caffeine should produce different logP predictions."""
        predictor = _get_adme_predictor()
        props_mdz = predictor.predict(MIDAZOLAM_SMILES)
        props_caf = predictor.predict(CAFFEINE_SMILES)
        assert props_mdz.logP != props_caf.logP, "logP should differ between midazolam and caffeine"


# ---------------------------------------------------------------------------
# 3–4. RDKitFeaturizer tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestRDKitFeaturizer:
    def test_rdkit_featurizer_shape(self):
        """Featurized descriptor vector has at least 10 elements."""
        pytest.importorskip("rdkit", reason="rdkit not installed")
        features_mod = pytest.importorskip(
            "omega_pbpk.features",
            reason="omega_pbpk.features module not yet implemented",
        )
        RDKitFeaturizer = features_mod.RDKitFeaturizer
        featurizer = RDKitFeaturizer()
        descriptors = featurizer.featurize(MIDAZOLAM_SMILES)
        assert descriptors.shape[0] >= 10, f"Expected >= 10 descriptors, got {descriptors.shape[0]}"

    def test_rdkit_featurizer_batch(self):
        """Batch featurization returns shape (n_smiles, n_features)."""
        pytest.importorskip("rdkit", reason="rdkit not installed")
        features_mod = pytest.importorskip(
            "omega_pbpk.features",
            reason="omega_pbpk.features module not yet implemented",
        )
        RDKitFeaturizer = features_mod.RDKitFeaturizer
        featurizer = RDKitFeaturizer()
        smiles_list = [
            MIDAZOLAM_SMILES,
            CAFFEINE_SMILES,
            "CC(=O)Oc1ccccc1C(=O)O",  # aspirin
        ]
        result = featurizer.featurize_batch(smiles_list)
        assert result.shape[0] == 3, f"Expected 3 rows, got {result.shape[0]}"
        assert result.shape[1] >= 10, f"Expected >= 10 features, got {result.shape[1]}"


# ---------------------------------------------------------------------------
# 5–7. Species physiology tests
# ---------------------------------------------------------------------------


def _get_species_physiology(species: str):
    """Import and call get_species_physiology; skip if not yet available."""
    try:
        from omega_pbpk.population.physiology import get_species_physiology
    except ImportError:
        pytest.skip(
            "get_species_physiology not yet implemented in omega_pbpk.population.physiology"
        )
    return get_species_physiology(species)


@pytest.mark.integration
class TestSpeciesPhysiology:
    def test_species_physiology_human(self):
        """Human physiology returns a dict with all positive numeric values."""
        phys = _get_species_physiology("human")
        assert isinstance(phys, dict), "Expected a dict"
        assert len(phys) > 0, "Physiology dict should not be empty"
        for key, val in phys.items():
            assert isinstance(val, (int, float)), f"{key} is not numeric: {val}"
            assert val > 0, f"{key} must be positive, got {val}"

    def test_species_physiology_rat_different_from_human(self):
        """Rat and human body weights should differ."""
        human = _get_species_physiology("human")
        rat = _get_species_physiology("rat")
        assert human["body_weight_kg"] != rat["body_weight_kg"], (
            "Human and rat body weights should differ"
        )

    def test_species_physiology_unknown_raises(self):
        """Unknown species name raises ValueError."""
        try:
            from omega_pbpk.population.physiology import get_species_physiology
        except ImportError:
            pytest.skip("get_species_physiology not yet implemented")
        with pytest.raises(ValueError):
            get_species_physiology("alien")


# ---------------------------------------------------------------------------
# 8–9. Pediatric ontogeny tests
# ---------------------------------------------------------------------------


def _get_pediatric_scaling():
    """Import get_pediatric_scaling; skip if not yet available."""
    try:
        from omega_pbpk.clinical.ontogeny import get_pediatric_scaling
    except ImportError:
        pytest.skip("get_pediatric_scaling not yet available in omega_pbpk.clinical.ontogeny")
    return get_pediatric_scaling


@pytest.mark.integration
class TestPediatricOntogeny:
    def test_pediatric_ontogeny_neonate_lower_than_adult(self):
        """Neonate CYP3A4 activity should be lower than adult."""
        get_pediatric_scaling = _get_pediatric_scaling()
        neonate = get_pediatric_scaling(age_years=0.1, weight_kg=3.0)
        adult = get_pediatric_scaling(age_years=30.0, weight_kg=70.0)
        assert neonate["cyp3a4"] < adult["cyp3a4"], (
            f"Neonate CYP3A4 ({neonate['cyp3a4']}) should be < adult ({adult['cyp3a4']})"
        )

    def test_pediatric_ontogeny_adult_cyp3a4_near_one(self):
        """Adult (30 years, 70 kg) CYP3A4 scaling factor should be near 1.0."""
        get_pediatric_scaling = _get_pediatric_scaling()
        adult = get_pediatric_scaling(age_years=30.0, weight_kg=70.0)
        assert abs(adult["cyp3a4"] - 1.0) <= 0.10, (
            f"Adult CYP3A4 should be ~1.0, got {adult['cyp3a4']}"
        )
