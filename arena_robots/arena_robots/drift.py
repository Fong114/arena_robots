"""Robot values written in two places, compared with their owner: controller limits, wheel geometry, gait limits and joint limits."""

from __future__ import annotations

import functools
import math
import typing
from collections.abc import Iterator
from pathlib import Path

import yaml

if typing.TYPE_CHECKING:
    from arena_robots.caps import ArmSpec, LimitsSpec
    from arena_robots.Robot import RobotView

AXES = {'linear': 'linear.x', 'lateral': 'linear.y', 'angular': 'angular.z'}
LIMIT_SUFFIXES = ('min_velocity', 'max_velocity', 'min_acceleration', 'max_acceleration')
GAIT_KEYS = {'linear': 'max_linear_velocity_x', 'lateral': 'max_linear_velocity_y', 'angular': 'max_angular_velocity_z'}
WHEEL_RELATIVE_TOLERANCE = 0.01
WHEEL_ABSOLUTE_TOLERANCE = 1e-4
JOINT_LIMIT_TOLERANCE = 1e-6


def touches_limits(path: str) -> bool:
    """Whether a cap override path is `mobile.limits` or a wildcard over the mobile cap."""
    segments = path.split('.')
    return segments[0] == 'mobile' and len(segments) > 1 and segments[1] in ('limits', '*')


def control_files(view: RobotView) -> list[Path]:
    """The robot's `control.yaml` and its per-simulator `control.<sim>.yaml` overlays."""
    return sorted(view.path.glob('control*.yaml'))


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
    keys = {f'{prefix}.{suffix}' for prefix in AXES.values() for suffix in LIMIT_SUFFIXES}
    return any(keys & set(params) for file in sorted(Path(view_path).glob('control*.yaml')) for _, params in _controllers(file))


def declares_limit_keys(view: RobotView) -> bool:
    """Whether the robot's drive controller declares velocity or acceleration limit parameters, or a CHAMP gait carries its speeds."""
    return _declares_limit_keys(str(view.path))


def limit_drift(view: RobotView, limits: LimitsSpec | None) -> list[str]:
    """Controller limit literals of every `control*.yaml` that differ from `limits`."""
    if limits is None:
        return []
    out: list[str] = []
    for file in control_files(view):
        for name, params in _controllers(file):
            for axis, prefix in AXES.items():
                expected: dict[str, float] = {}
                if axis in limits.velocity:
                    expected['min_velocity'] = limits.velocity[axis].min
                    expected['max_velocity'] = limits.velocity[axis].max
                if axis in limits.acceleration:
                    expected['min_acceleration'] = -limits.acceleration[axis]
                    expected['max_acceleration'] = limits.acceleration[axis]
                for suffix, want in expected.items():
                    key = f'{prefix}.{suffix}'
                    if key in params and math.isfinite(want) and not math.isclose(float(params[key]), want, rel_tol=1e-9, abs_tol=1e-9):
                        out.append(f'{file.name} {name}: {key} {params[key]} differs from mobile.limits {want}')
    return out


def wheel_drift(view: RobotView, urdf: str, base_link: str) -> list[str]:
    """Wheel radius and track width literals of `control.yaml` that differ from the URDF's wheel joints."""
    from arena_robots import urdf_facts

    path = view.path / 'control.yaml'
    if not path.is_file():
        return []
    out: list[str] = []
    for name, params in _controllers(path):
        left, right = params.get('left_wheel_names'), params.get('right_wheel_names')
        names = params.get('wheel_joint_names')
        if isinstance(left, list) and isinstance(right, list):
            measured = urdf_facts.wheel_geometry(urdf, base_link, [str(n) for n in left], [str(n) for n in right])
            keys = ('wheel_radius', 'wheel_separation')
        elif isinstance(names, list) and names and 'wheel_radius' in params:
            pair = [str(n) for n in names[:2]]
            measured = urdf_facts.wheel_geometry(urdf, base_link, pair[:1], pair[1:] or pair[:1])
            keys = ('wheel_radius',)
        else:
            continue
        if measured is None:
            continue
        for key in keys:
            if key in params and not math.isclose(float(params[key]), measured[key], rel_tol=WHEEL_RELATIVE_TOLERANCE, abs_tol=WHEEL_ABSOLUTE_TOLERANCE):
                out.append(f'control.yaml {name}: {key} {params[key]} differs from the URDF {measured[key]:.4g}')
    return out


def gait_limits(limits: LimitsSpec) -> dict[str, float]:
    """CHAMP's symmetric `max_*` gait keys from the larger magnitude of each velocity axis of `limits`."""
    return {GAIT_KEYS[axis]: max(abs(r.min), abs(r.max)) for axis, r in limits.velocity.items() if axis in GAIT_KEYS and math.isfinite(r.min) and math.isfinite(r.max)}


def gait_drift(view: RobotView, limits: LimitsSpec | None) -> list[str]:
    """CHAMP gait speed literals of `champ/gait.yaml` that differ from `limits`."""
    path = view.path / 'champ' / 'gait.yaml'
    if limits is None or not path.is_file():
        return []
    gait = yaml.safe_load(path.read_text())['/**']['ros__parameters']['gait']
    return [f'champ/gait.yaml: {key} {gait[key]} differs from mobile.limits {want}' for key, want in gait_limits(limits).items() if key in gait and not math.isclose(float(gait[key]), want, rel_tol=1e-9, abs_tol=1e-9)]


def joint_drift(instance: str, arm: ArmSpec, urdf: str) -> list[str]:
    """Joint limit file values of an arm that exceed the URDF `<limit>` of the same joint."""
    from arena_robots import urdf_facts

    limits = urdf_facts.joint_limits(urdf)
    out: list[str] = []
    for joint, entry in arm.joint_limits.items():
        bounds = limits.get(joint)
        if bounds is None:
            continue
        for key, bound in (('max_velocity', bounds.velocity), ('max_effort', bounds.effort)):
            value = entry.get(key)
            if bound is not None and isinstance(value, int | float) and float(value) > bound * (1 + JOINT_LIMIT_TOLERANCE):
                out.append(f'arm {instance} joint {joint}: joint_limits {key} {value} exceeds the URDF limit {bound}')
    return out
