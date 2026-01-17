from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Mapping, Optional, Sequence



def _as_namespace(value: Any) -> SimpleNamespace:
    if isinstance(value, SimpleNamespace):
        return value
    if isinstance(value, Mapping):
        return SimpleNamespace(**value)
    if hasattr(value, "__dict__"):
        return SimpleNamespace(**value.__dict__)
    return SimpleNamespace(value=value)



