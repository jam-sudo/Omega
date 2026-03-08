"""Data loaders and datasets for ML training."""


def __getattr__(name: str):
    """Lazy imports to avoid hard dependency on requests at collection time."""
    _exports = {"PKDBLoader", "FDALabelExtractor", "TDCLoader"}
    if name in _exports:
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
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["PKDBLoader", "FDALabelExtractor", "TDCLoader"]
