"""Data loaders and datasets for ML training."""


def __getattr__(name: str):
    """Lazy imports to avoid hard dependency on requests at collection time."""
    _loader_exports = {"PKDBLoader", "FDALabelExtractor", "TDCLoader"}
    _dataset_exports = {
        "ClinicalPKDataset",
        "run_clinical_data_pipeline",
        "validate_dataset",
        "validate_pk_record",
    }

    if name in _loader_exports:
        from omega_pbpk.ml.data.loaders import (
            FDALabelExtractor,
            PKDBLoader,
            TDCLoader,
        )

        return {
            "PKDBLoader": PKDBLoader,
            "FDALabelExtractor": FDALabelExtractor,
            "TDCLoader": TDCLoader,
        }[name]

    if name in _dataset_exports:
        from omega_pbpk.ml.data.datasets import (
            ClinicalPKDataset,
            run_clinical_data_pipeline,
            validate_dataset,
            validate_pk_record,
        )

        return {
            "ClinicalPKDataset": ClinicalPKDataset,
            "run_clinical_data_pipeline": run_clinical_data_pipeline,
            "validate_dataset": validate_dataset,
            "validate_pk_record": validate_pk_record,
        }[name]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "PKDBLoader",
    "FDALabelExtractor",
    "TDCLoader",
    "ClinicalPKDataset",
    "run_clinical_data_pipeline",
    "validate_dataset",
    "validate_pk_record",
]
