"""Drive controller limit flags and the tolerance of the arm joint limit check."""

from __future__ import annotations

import functools
import typing
from collections.abc import Iterator
from pathlib import Path

import yaml

if typing.TYPE_CHECKING:
    from arena_robots.Robot import RobotView

AXES = {'linear': 'linear.x', 'lateral': 'linear.y', 'angular': 'angular.z'}
LIMIT_FLAGS = ('has_velocity_limits', 'has_acceleration_limits')
JOINT_LIMIT_TOLERANCE = 1e-6


def _controllers(path: Path) -> Iterator[tuple[str, dict[str, typing.Any]]]:
    data = yaml.safe_load(path.read_text()) or {}
    for name, section in data.items():
        if name == 'controller_manager' or not isinstance(section, dict):
            continue
        params = section.get('ros__parameters')
        if isinstance(params, dict):
            yield str(name), params


@functools.lru_cache(maxsize=128)
def _declares_limit_keys(view_path: str) -> bool:
    if (Path(view_path) / 'champ' / 'gait.yaml').is_file():
        return True
    flags = {f'{prefix}.{flag}' for prefix in AXES.values() for flag in LIMIT_FLAGS}
    return any(params.get(flag) is True for flag in flags for file in sorted(Path(view_path).glob('control*.yaml')) for _, params in _controllers(file))


def declares_limit_keys(view: RobotView) -> bool:
    """Whether the robot's drive controller sets a velocity or acceleration limit flag, or a CHAMP gait carries its speeds."""
    return _declares_limit_keys(str(view.path))
