import logging
import sys
import types
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib


logger = logging.getLogger(__name__)


def _to_dense(matrix):
    """Compatibility helper for older serialized pipelines."""
    return matrix.toarray() if hasattr(matrix, "toarray") else matrix


def _register_pickle_compat_symbols() -> None:
    """
    Some models were serialized when helper functions lived under __main__/__mp_main__.
    Register those symbols so joblib can deserialize safely.
    """
    for module_name in ("__main__", "__mp_main__"):
        module = sys.modules.get(module_name)
        if module is None:
            module = types.ModuleType(module_name)
            sys.modules[module_name] = module
        if not hasattr(module, "_to_dense"):
            setattr(module, "_to_dense", _to_dense)


@lru_cache(maxsize=1)
def get_model() -> Any:
    model_path = Path(__file__).resolve().parents[3] / "ml-service" / "models" / "drug_response_model.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at: {model_path}")
    _register_pickle_compat_symbols()
    logger.info("Loading model from %s", model_path)
    return joblib.load(model_path)
