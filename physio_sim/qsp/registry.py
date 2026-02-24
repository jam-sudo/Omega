from __future__ import annotations

from collections.abc import Callable

from physio_sim.qsp.base import BaseQSPModel

_MODEL_REGISTRY: dict[str, BaseQSPModel] = {}


def register_qsp_model(name: str) -> Callable[[type[BaseQSPModel]], type[BaseQSPModel]]:
    def _inner(model_cls: type[BaseQSPModel]) -> type[BaseQSPModel]:
        _MODEL_REGISTRY[name] = model_cls()
        return model_cls

    return _inner


def get_qsp_model(name: str) -> BaseQSPModel:
    _ensure_builtin_models_registered()
    try:
        return _MODEL_REGISTRY[name]
    except KeyError as exc:
        msg = f"Unknown QSP model '{name}'. Available: {', '.join(list_qsp_models())}"
        raise ValueError(msg) from exc


def list_qsp_models() -> list[str]:
    _ensure_builtin_models_registered()
    return sorted(_MODEL_REGISTRY)


def _ensure_builtin_models_registered() -> None:
    # Import side effects register built-in models.
    import physio_sim.qsp.models  # noqa: F401
