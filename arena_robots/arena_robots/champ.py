"""CHAMP parameter files of a quadruped instance: joint and link maps from its URDF, gait speeds from its mobile limits."""

from __future__ import annotations

import math
import typing
import xml.etree.ElementTree as ET
from pathlib import Path

import yaml

if typing.TYPE_CHECKING:
    from arena_robots.caps import RobotCaps

LEGS = ('left_front', 'right_front', 'left_hind', 'right_hind')
SPEED_KEYS = {'linear': 'max_linear_velocity_x', 'lateral': 'max_linear_velocity_y', 'angular': 'max_angular_velocity_z'}
_REVOLUTE_PER_LEG = 3


def _translation(joint: ET.Element) -> tuple[float, float, float]:
    origin = joint.find('origin')
    x, y, z = (float(v) for v in (origin.get('xyz', '0 0 0') if origin is not None else '0 0 0').split())
    return x, y, z


def _end(joint: ET.Element, tag: str) -> str:
    end = joint.find(tag)
    if end is None:
        raise ValueError(f'joint {joint.attrib["name"]!r} has no <{tag}> link')
    return end.attrib['link']


def leg_maps(urdf: str, base_frame: str) -> tuple[dict[str, typing.Any], dict[str, typing.Any]]:
    """CHAMP's `joints_map` and `links_map` of the four legs below `base_frame`, each a chain of three revolute joints ending in a fixed foot joint."""
    root = ET.fromstring(urdf)
    by_parent: dict[str, list[ET.Element]] = {}
    for joint in root.findall('joint'):
        by_parent.setdefault(_end(joint, 'parent'), []).append(joint)

    legs: dict[str, tuple[list[str], list[str]]] = {}
    offset_at: dict[str, tuple[float, float, float]] = {}

    def walk(link: str, path: list[ET.Element], offset: tuple[float, float, float]) -> None:
        children = by_parent.get(link, [])
        revolute = [j for j in path if j.get('type') == 'revolute']
        if not children and path[-1:] and path[-1].get('type') == 'fixed' and len(revolute) == _REVOLUTE_PER_LEG:
            chain = [*revolute, path[-1]]
            hip_offset = offset_at[revolute[0].attrib['name']]
            name = f'{"left" if hip_offset[1] > 0 else "right"}_{"front" if hip_offset[0] > 0 else "hind"}'
            if name in legs:
                raise ValueError(f'{base_frame}: two legs classify as {name}, hip joints {legs[name][0][0]} and {chain[0].get("name")}')
            legs[name] = ([j.attrib['name'] for j in chain], [_end(j, 'child') for j in chain])
        for joint in children:
            x, y, z = _translation(joint)
            shifted = (offset[0] + x, offset[1] + y, offset[2] + z)
            offset_at[joint.attrib['name']] = shifted
            walk(_end(joint, 'child'), [*path, joint], shifted)

    walk(base_frame, [], (0.0, 0.0, 0.0))
    missing = [name for name in LEGS if name not in legs]
    if missing:
        raise ValueError(f'{base_frame}: no leg of three revolute joints and a fixed foot joint for {missing}')
    joints = {name: legs[name][0] for name in LEGS}
    links = {'base': base_frame, **{name: legs[name][1] for name in LEGS}}
    return joints, links


def _write(suffix: str, parameters: dict[str, typing.Any]) -> Path:
    from arena_robots.served_urdf import content_addressed

    return content_addressed(yaml.safe_dump({'/**': {'ros__parameters': parameters}}, sort_keys=False).encode(), f'.{suffix}.yaml', stem='champ_')


def served_champ_files(caps: RobotCaps) -> dict[str, Path]:
    """The `joints_map`, `links_map` and `gait_config` parameter files of an instance, written under `<tempdir>/arena_robots/` by content."""
    assert caps.view is not None
    joints, links = leg_maps(caps.rendered_urdf(), caps.body.base_frame)
    gait = yaml.safe_load((caps.view.path / 'champ' / 'gait.yaml').read_text())['/**']['ros__parameters']['gait']
    limits = caps.mobile.limits
    assert limits is not None
    for axis, key in SPEED_KEYS.items():
        bound = limits.velocity[axis]
        if not math.isfinite(bound.min) or not math.isfinite(bound.max):
            raise ValueError(f'{caps.view.path.name}: mobile.limits.velocity.{axis} must be finite for the CHAMP gait, got [{bound.min}, {bound.max}]')
        gait[key] = max(abs(bound.min), abs(bound.max))
    return {
        'joints_map': _write('joints', {'joints_map': joints}),
        'links_map': _write('links', {'links_map': links}),
        'gait_config': _write('gait', {'gait': gait}),
    }
