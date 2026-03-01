"""Transporter substrate/inhibitor classifier plugin."""

from __future__ import annotations

from omega_pbpk.contracts.drug_spec import DrugSpec
from omega_pbpk.plugins.base import PluginBase


class TransporterPlugin(PluginBase):
    """Wraps TransporterClassifier as a PluginBase for pipeline integration."""

    name = "transporter_classifier"
    provides = frozenset({"transporter_profile"})

    def __init__(self) -> None:
        from omega_pbpk.prediction.transporter_classifier import TransporterClassifier

        self._classifier = TransporterClassifier()

    def predict(self, spec: DrugSpec) -> dict:
        """Return transporter profile dict keyed by transporter name."""
        if spec.smiles:
            profile = self._classifier.predict(spec.smiles)
        else:
            charge = spec.compound_type or "neutral"
            profile = self._classifier.predict_from_properties(spec.mw, spec.logP, charge)

        return {
            "transporter_profile": {
                t: {
                    "substrate_prob": p.substrate_prob,
                    "inhibitor_prob": p.inhibitor_prob,
                    "km_uM": p.km_uM,
                    "ic50_uM": p.ic50_uM if p.ic50_uM != float("inf") else 9999.0,
                }
                for t, p in profile.profiles.items()
            }
        }

    def confidence(self, spec: DrugSpec) -> float:
        """Return prediction confidence: 0.5 with SMILES, 0.3 without."""
        return 0.5 if spec.smiles else 0.3
