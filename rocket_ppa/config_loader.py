"""Helpers for loading editable Python configuration dictionaries."""

from __future__ import annotations

import importlib
import os
from collections.abc import Mapping
from types import SimpleNamespace
DEFAULT_CONFIG_MODULE = "rocket_ppa.run_config"


def load_config(name: str) -> SimpleNamespace:
    module_name = os.environ.get("ROCKET_PPA_CONFIG_MODULE", DEFAULT_CONFIG_MODULE)
    module = importlib.import_module(module_name)
    raw = getattr(module, name)
    if not isinstance(raw, Mapping):
        raise TypeError(f"{module_name}.{name} must be a mapping")
    return SimpleNamespace(**dict(raw))


def require_keys(config: SimpleNamespace, keys: tuple[str, ...]) -> None:
    missing = [key for key in keys if not hasattr(config, key)]
    if missing:
        raise ValueError(f"missing required config keys: {', '.join(missing)}")
