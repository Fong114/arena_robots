"""The acoustic emission profile every robot starts from."""

from __future__ import annotations

from collections.abc import Mapping

import yaml
from ament_index_python.packages import get_package_share_path


def default_profile() -> dict[str, float]:
    """The parameters of `config/acoustic_profile.yaml`."""
    path = get_package_share_path("arena_robots") / "config" / "acoustic_profile.yaml"
    return {str(key): float(value) for key, value in yaml.safe_load(path.read_text()).items()}


def effective_profile(acoustics: Mapping[str, object] | None) -> dict[str, float]:
    """The default profile with a robot's `acoustics` block laid over it."""
    return {**default_profile(), **{str(key): float(value) for key, value in (acoustics or {}).items() if isinstance(value, int | float)}}
