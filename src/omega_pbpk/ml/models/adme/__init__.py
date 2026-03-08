"""ADME prediction models -- ML-based replacements for polynomial predictor."""

from abc import ABC, abstractmethod

from omega_pbpk.prediction.adme_predictor import ADMEProperties


class MLADMEPredictor(ABC):
    """Abstract base class for ML-based ADME predictors.

    All ML ADME models must implement this interface to be swappable
    with the existing polynomial predictor.
    """

    @abstractmethod
    def predict(self, smiles: str) -> ADMEProperties:
        """Predict ADME properties from a SMILES string.

        Args:
            smiles: Canonical SMILES string for the compound.

        Returns:
            ADMEProperties with all 9 properties, confidence, and intervals.
        """
        ...

    @abstractmethod
    def predict_batch(self, smiles_list: list[str]) -> list[ADMEProperties]:
        """Predict ADME properties for multiple compounds.

        Args:
            smiles_list: List of SMILES strings.

        Returns:
            List of ADMEProperties, one per input SMILES.
        """
        ...
