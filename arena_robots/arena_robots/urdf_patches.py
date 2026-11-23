"""Corrections of vendor URDFs: typed patches read from `urdf_patches.yaml` and applied to a rendered robot."""

from __future__ import annotations

import math
import typing
import xml.etree.ElementTree as ET
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import attrs
import numpy as np
import yaml
from ament_index_python.packages import get_package_share_path

if typing.TYPE_CHECKING:
    from arena_robots.assembly import ResolvedAssembly
    from arena_robots.Robot import RobotView

FILE_NAME = 'urdf_patches.yaml'
LIMIT_KEYS = ('lower', 'upper', 'velocity', 'effort')
_INERTIA_KEYS = ('ixx', 'ixy', 'ixz', 'iyy', 'iyz', 'izz')


@attrs.frozen(kw_only=True)
class Patch:
    reason: str
    source: str


@attrs.frozen(kw_only=True)
class LinkMass(Patch):
    link: str
    mass: float


@attrs.frozen(kw_only=True)
class TotalMass(Patch):
    mass: float


@attrs.frozen(kw_only=True)
class JointLimit(Patch):
    joint: str
    limit: Mapping[str, float]


@attrs.frozen(kw_only=True)
class InertiaScale(Patch):
    link: str
    factor: float


@attrs.frozen(kw_only=True)
class InertiaFromCollision(Patch):
    link: str


@attrs.frozen(kw_only=True)
class AddCollision(Patch):
    link: str
    box: tuple[float, float, float]
    xyz: tuple[float, float, float]
    rpy: tuple[float, float, float]


def _positive(value: object, key: str, path: Path) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise ValueError(f"{path}: {key} must be a positive number, got {value!r}")
    return float(value)


def _triple(value: object, key: str, path: Path, *, positive: bool = False) -> tuple[float, float, float]:
    if not isinstance(value, list) or len(value) != 3 or any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in value):
        raise ValueError(f"{path}: {key} must be a list of three numbers, got {value!r}")
    if positive and any(item <= 0 for item in value):
        raise ValueError(f"{path}: {key} must be positive, got {value!r}")
    return (float(value[0]), float(value[1]), float(value[2]))


def _parse_collision(link: object, body: object, path: Path, meta: Mapping[str, str]) -> AddCollision:
    if not isinstance(body, dict) or not {'box'} <= set(body) <= {'box', 'xyz', 'rpy'}:
        raise ValueError(f"{path}: collision of link {link!r} must be a mapping of box with optional xyz and rpy, got {body!r}")
    return AddCollision(
        link=str(link),
        box=_triple(body['box'], 'box', path, positive=True),
        xyz=_triple(body.get('xyz', [0, 0, 0]), 'xyz', path),
        rpy=_triple(body.get('rpy', [0, 0, 0]), 'rpy', path),
        **meta,
    )


def _parse_entry(entry: object, path: Path) -> Patch:
    if not isinstance(entry, dict):
        raise ValueError(f"{path}: a patch must be a mapping, got {entry!r}")
    meta: dict[str, str] = {}
    for key in ('reason', 'source'):
        value = entry.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{path}: patch {entry!r} needs a {key}")
        meta[key] = value
    body = {key: value for key, value in entry.items() if key not in meta}
    keys = frozenset(body)
    if keys == {'link', 'mass'}:
        return LinkMass(link=str(body['link']), mass=_positive(body['mass'], 'mass', path), **meta)
    if keys == {'link', 'inertia_scale'}:
        return InertiaScale(link=str(body['link']), factor=_positive(body['inertia_scale'], 'inertia_scale', path), **meta)
    if keys == {'link', 'inertia'}:
        if body['inertia'] != 'from_collision':
            raise ValueError(f"{path}: inertia of link {body['link']!r} must be from_collision, got {body['inertia']!r}")
        return InertiaFromCollision(link=str(body['link']), **meta)
    if keys == {'link', 'collision'}:
        return _parse_collision(body['link'], body['collision'], path, meta)
    if keys == {'total_mass'}:
        return TotalMass(mass=_positive(body['total_mass'], 'total_mass', path), **meta)
    if keys == {'joint', 'limit'}:
        limit = body['limit']
        if not isinstance(limit, dict) or not limit or not set(limit) <= set(LIMIT_KEYS):
            raise ValueError(f"{path}: limit of joint {body['joint']!r} must be a mapping of {list(LIMIT_KEYS)}, got {limit!r}")
        return JointLimit(joint=str(body['joint']), limit={key: float(value) for key, value in limit.items()}, **meta)
    raise ValueError(f"{path}: patch {entry!r} is none of link+mass, link+inertia_scale, link+inertia, link+collision, total_mass, joint+limit")


def load(path: Path, variant: str | None = None) -> list[Patch]:
    """The patches of `path` (those of `variant` in a family file), empty when the file does not exist."""
    if not path.is_file():
        return []
    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict) or not set(data) <= {'patches', 'variants'} or {'patches', 'variants'} <= set(data):
        raise ValueError(f"{path}: must be a mapping with either 'patches' or 'variants'")
    if 'variants' in data:
        variants = data['variants']
        if not isinstance(variants, dict):
            raise ValueError(f"{path}: 'variants' must be a mapping of variant to patch list")
        entries = variants.get(variant) or []
    else:
        entries = data['patches'] or []
    if not isinstance(entries, list):
        raise ValueError(f"{path}: patches must be a list, got {type(entries).__name__}")
    return [_parse_entry(entry, path) for entry in entries]


def _find(elements: Mapping[str, ET.Element], name: str, kind: str, path: Path) -> ET.Element:
    if name not in elements:
        raise ValueError(f"{path}: {kind} {name!r} does not exist in the rendered URDF")
    return elements[name]


def _scale_inertia(link: ET.Element, factor: float) -> None:
    for inertia in link.findall('inertial/inertia'):
        for key in _INERTIA_KEYS:
            if key in inertia.attrib:
                inertia.set(key, _format(float(inertia.attrib[key]) * factor))


def _rotation(rpy: Sequence[float]) -> np.ndarray:
    roll, pitch, yaw = rpy
    rx = np.array([[1, 0, 0], [0, math.cos(roll), -math.sin(roll)], [0, math.sin(roll), math.cos(roll)]])
    ry = np.array([[math.cos(pitch), 0, math.sin(pitch)], [0, 1, 0], [-math.sin(pitch), 0, math.cos(pitch)]])
    rz = np.array([[math.cos(yaw), -math.sin(yaw), 0], [math.sin(yaw), math.cos(yaw), 0], [0, 0, 1]])
    return rz @ ry @ rx


def _numbers(text: str | None, count: int = 3) -> list[float]:
    return [float(token) for token in text.split()] if text is not None and text.split() else [0.0] * count


def _primitive_moments(shape: ET.Element, mass: float) -> list[float] | None:
    if shape.tag == 'box':
        x, y, z = _numbers(shape.get('size'))
        return [mass / 12 * (y * y + z * z), mass / 12 * (x * x + z * z), mass / 12 * (x * x + y * y)]
    if shape.tag == 'cylinder':
        radius, length = float(shape.get('radius', 0.0)), float(shape.get('length', 0.0))
        side = mass * (3 * radius * radius + length * length) / 12
        return [side, side, mass * radius * radius / 2]
    if shape.tag == 'sphere':
        radius = float(shape.get('radius', 0.0))
        return [2 * mass * radius * radius / 5] * 3
    return None


def _inertia_from_collision(link: ET.Element, path: Path) -> None:
    name = link.get('name')
    mass = float(_mass_element(link, path).attrib['value'])
    shapes = [(collision, geometry[0]) for collision in link.findall('collision') for geometry in collision.findall('geometry') if len(geometry)]
    moments = _primitive_moments(shapes[0][1], mass) if len(shapes) == 1 else None
    if moments is None or mass <= 0:
        raise ValueError(f"{path}: link {name!r} needs a mass above 0 and one box, cylinder or sphere collision to take its inertia from")
    collision = shapes[0][0]
    origin = collision.find('origin')
    xyz = _numbers(origin.get('xyz') if origin is not None else None)
    rpy = _numbers(origin.get('rpy') if origin is not None else None)
    rotation = _rotation(rpy)
    tensor = rotation @ np.diag(moments) @ rotation.T
    tensor[np.abs(tensor) < 1e-12 * np.abs(tensor).max()] = 0.0
    inertial = link.find('inertial')
    assert inertial is not None
    frame = inertial.find('origin')
    if frame is None:
        frame = ET.Element('origin')
        inertial.insert(0, frame)
    frame.set('xyz', ' '.join(_format(value) for value in xyz))
    frame.set('rpy', '0 0 0')
    inertia = inertial.find('inertia')
    if inertia is None:
        inertia = ET.SubElement(inertial, 'inertia')
    for key, (row, column) in zip(_INERTIA_KEYS, ((0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)), strict=True):
        inertia.set(key, _format(float(tensor[row, column]) + 0.0))


def _add_collision(link: ET.Element, patch: AddCollision) -> None:
    collision = ET.Element('collision')
    ET.SubElement(collision, 'origin', {'xyz': ' '.join(_format(value) for value in patch.xyz), 'rpy': ' '.join(_format(value) for value in patch.rpy)})
    geometry = ET.SubElement(collision, 'geometry')
    ET.SubElement(geometry, 'box', {'size': ' '.join(_format(value) for value in patch.box)})
    last = max((index for index, child in enumerate(link) if child.tag in ('visual', 'collision')), default=-1)
    link.insert(last + 1, collision)


def _format(value: float) -> str:
    return f'{value:.12g}'


def _mass_element(link: ET.Element, path: Path) -> ET.Element:
    mass = link.find('inertial/mass')
    if mass is None:
        raise ValueError(f"{path}: link {link.get('name')!r} has no inertial mass")
    return mass


def _scale_link(link: ET.Element, factor: float) -> None:
    mass = link.find('inertial/mass')
    if mass is not None:
        mass.set('value', _format(float(mass.attrib['value']) * factor))
    _scale_inertia(link, factor)


def apply_patches(root: ET.Element, patches: Sequence[Patch], *, path: Path, prefix: str, owns: Callable[[str], bool]) -> None:
    """Applies `patches` to the rendered `root`, names resolved under `prefix`, `total_mass` scaling the links `owns` accepts."""
    links = {str(link.get('name')): link for link in root.findall('link')}
    joints = {str(joint.get('name')): joint for joint in root.findall('joint')}
    for patch in patches:
        if isinstance(patch, LinkMass):
            link = _find(links, prefix + patch.link, 'link', path)
            mass = _mass_element(link, path)
            current = float(mass.attrib['value'])
            if current <= 0:
                raise ValueError(f"{path}: link {prefix + patch.link!r} has mass {current}, nothing to scale the inertia from")
            _scale_link(link, patch.mass / current)
        elif isinstance(patch, InertiaScale):
            _scale_inertia(_find(links, prefix + patch.link, 'link', path), patch.factor)
        elif isinstance(patch, InertiaFromCollision):
            _inertia_from_collision(_find(links, prefix + patch.link, 'link', path), path)
        elif isinstance(patch, AddCollision):
            _add_collision(_find(links, prefix + patch.link, 'link', path), patch)
        elif isinstance(patch, JointLimit):
            joint = _find(joints, prefix + patch.joint, 'joint', path)
            limit = joint.find('limit')
            if limit is None:
                limit = ET.SubElement(joint, 'limit')
            for key, value in patch.limit.items():
                limit.set(key, _format(value))
        elif isinstance(patch, TotalMass):
            owned = [link for name, link in links.items() if owns(name) and link.find('inertial/mass') is not None]
            current = sum(float(_mass_element(link, path).attrib['value']) for link in owned)
            if current <= 0:
                raise ValueError(f"{path}: total_mass finds no link mass to scale")
            for link in owned:
                _scale_link(link, patch.mass / current)


def _owner_index(name: str, prefixes: Sequence[str], chassis_links: frozenset[str]) -> int | None:
    if name in chassis_links:
        return None
    best: int | None = None
    for index, prefix in enumerate(prefixes):
        if name.startswith(prefix) and (best is None or len(prefix) > len(prefixes[best])):
            best = index
    return best


def owners(view: RobotView, resolved: ResolvedAssembly | None) -> Callable[[str], int | None]:
    """The index into `resolved.placements` of the placement a rendered link belongs to, None for the chassis."""
    placements = resolved.placements if resolved is not None else []
    prefix = view.assembly.prefix if view.assembly is not None else ''
    prefixes = [f"{prefix}{placement.mount.frame or placement.mount.name}_" for placement in placements]
    chassis_links = frozenset(f"{prefix}{placement.mount.parent}" for placement in placements if placement.mount.chained_parent is None)
    return lambda name: _owner_index(name, prefixes, chassis_links)


def apply(root: ET.Element, view: RobotView, resolved: ResolvedAssembly | None) -> None:
    """Applies the patch files of the placed components, then the robot's own, to the rendered `root`."""
    from arena_robots.catalog import Catalog

    placements = resolved.placements if resolved is not None else []
    prefix = view.assembly.prefix if view.assembly is not None else ''
    prefixes = [f"{prefix}{placement.mount.frame or placement.mount.name}_" for placement in placements]
    owner_of = owners(view, resolved)
    catalog = Catalog()
    components = get_package_share_path('arena_robots') / 'components'
    for index, placement in enumerate(placements):
        owner = catalog.owner(placement.type, placement.variant)
        if owner is None:
            continue
        path = components / placement.type / owner / FILE_NAME
        apply_patches(root, load(path, placement.variant), path=path, prefix=prefixes[index], owns=lambda name, index=index: owner_of(name) == index)
    path = view.path / FILE_NAME
    apply_patches(root, load(path), path=path, prefix='', owns=lambda name: owner_of(name) is None)
