from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .compat import dataclass_slots


def _deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_update(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    inherits = data.pop("inherits", None)
    if inherits:
        parent = _load_yaml((path.parent / inherits).resolve() if not Path(inherits).is_absolute() else Path(inherits))
        return _deep_update(parent, data)
    return data


@dataclass_slots()
class SystemConfig:
    raw: dict[str, Any]
    root: Path

    @property
    def frame(self) -> dict[str, Any]:
        return self.raw["frame"]

    @property
    def pulse(self) -> dict[str, Any]:
        return self.raw["pulse"]

    @property
    def channel(self) -> dict[str, Any]:
        return self.raw["channel"]

    @property
    def estimation(self) -> dict[str, Any]:
        return self.raw["estimation"]

    @property
    def detection(self) -> dict[str, Any]:
        return self.raw["detection"]

    @property
    def seed(self) -> int:
        return int(self.raw["seed"])

    @property
    def M(self) -> int:
        return int(self.frame["M"])

    @property
    def N(self) -> int:
        return int(self.frame["N"])

    @property
    def Q(self) -> int:
        return self.M * self.N

    @property
    def tau_p(self) -> float:
        return float(self.frame["tau_p_s"])

    @property
    def nu_p(self) -> float:
        return float(self.frame["nu_p_hz"])

    @property
    def T(self) -> float:
        return float(self.frame["T_s"])

    @property
    def B(self) -> float:
        return float(self.frame["B_hz"])

    @property
    def q(self) -> int:
        return int(self.frame["q"])


def load_config(path: str | Path) -> SystemConfig:
    resolved = Path(path).resolve()
    return SystemConfig(raw=_load_yaml(resolved), root=resolved.parent.parent)
