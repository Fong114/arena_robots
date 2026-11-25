"""Robots that are a base robot with fixed parts and cap overrides, read from robots/<name>/preset.yaml."""

from __future__ import annotations

import copy
import functools
import typing
from pathlib import Path

import attrs
import yaml

from arena_robots import ARENA_ROBOTS_DIR
from arena_robots.assembly import AssemblyError, AssemblySpec, build_request, resolve
from arena_robots.inline import parse_inline_override

FILE_NAME = 'preset.yaml'
_KEYS = frozenset({'robot', 'parts', 'caps'})


class PresetError(RuntimeError):
    """A preset file that is malformed or names a robot or part that does not exist."""


@attrs.frozen
class Preset:
    """A base robot with the parts and cap overrides applied under the caller's own."""

    name: str
    robot: str
    parts: dict[str, list[str]] = attrs.field(factory=dict)
    caps: dict[str, typing.Any] = attrs.field(factory=dict)

    @classmethod
    def parse(cls, name: str, data: object) -> Preset:
        if not isinstance(data, dict):
            raise PresetError(f"preset {name!r}: expected a mapping with 'robot', 'parts' and 'caps', got {data!r}")
        unknown = sorted(set(data) - _KEYS)
        if unknown:
            raise PresetError(f"preset {name!r}: unknown keys {unknown}, expected {sorted(_KEYS)}")
        robot = data.get('robot')
        if not isinstance(robot, str) or not robot:
            raise PresetError(f"preset {name!r}: 'robot' must name the base robot")
        parts_raw = data.get('parts') or {}
        caps_raw = data.get('caps') or {}
        if not isinstance(parts_raw, dict) or not isinstance(caps_raw, dict):
            raise PresetError(f"preset {name!r}: 'parts' and 'caps' must be mappings")
        parts = {str(key): [str(item) for item in value] if isinstance(value, list) else [str(value)] for key, value in parts_raw.items()}
        caps: dict[str, typing.Any] = {}
        for key, value in caps_raw.items():
            cap, _, tail = str(key).partition('.')
            if not cap or not tail or not all(tail.split('.')) or tail == 'adapter':
                raise PresetError(f"preset {name!r}: cap override {key!r} is not '<cap>.<path>'")
            try:
                caps[str(key)] = parse_inline_override(value) if isinstance(value, str) or (isinstance(value, list) and all(isinstance(item, str) for item in value)) else value
            except ValueError as exc:
                raise PresetError(f"preset {name!r}: {key!r}: {exc}") from exc
        return cls(name=name, robot=robot, parts=parts, caps=caps)

    @classmethod
    def read(cls, path: Path) -> Preset:
        return cls.parse(path.parent.name, yaml.safe_load(path.read_text()))

    def spec(self) -> AssemblySpec:
        """The base robot's assembly."""
        from arena_robots.Robot import RobotIdentifier

        try:
            view = RobotIdentifier(self.robot).resolve_sync()
        except FileNotFoundError as exc:
            raise PresetError(f"preset {self.name!r}: base robot {self.robot!r} does not exist") from exc
        if not view.is_definition:
            raise PresetError(f"preset {self.name!r}: base {self.robot!r} is not a robot definition")
        if view.assembly is None:
            raise PresetError(f"preset {self.name!r}: base robot {self.robot!r} has no assembly.yaml")
        return view.assembly

    def validate(self) -> None:
        """Raise when the base robot or a part does not exist or cannot be placed."""
        from arena_robots.catalog import Catalog

        spec = self.spec()
        try:
            request, cleared_sockets, cleared_types = build_request(spec, self.parts)
            resolve(spec, request, cleared_sockets=cleared_sockets, cleared_types=cleared_types)
            catalog = Catalog()
            for type_, items in request.items():
                for item in items:
                    catalog.get(type_, item.variant)
        except (AssemblyError, RuntimeError) as exc:
            raise PresetError(f"preset {self.name!r}: {exc}") from exc

    def merged(self, parts: dict[str, list[str]], caps: dict[str, typing.Any]) -> tuple[dict[str, list[str]], dict[str, typing.Any]]:
        """The preset's parts and caps under the caller's, which win wherever both touch the same socket, type or cap path."""
        spec = self.spec()
        kept_parts = {lhs: list(values) for lhs, values in self.parts.items() if not _superseded(spec, lhs, values, parts)}
        kept_caps = {path: copy.deepcopy(value) for path, value in self.caps.items() if not any(path == other or path.startswith(f'{other}.') or other.startswith(f'{path}.') for other in caps)}
        return {**kept_parts, **parts}, {**kept_caps, **caps}


def _type_of(spec: AssemblySpec, lhs: str, value: str) -> str | None:
    """The part type a directive places, ``None`` when its socket accepts several and the value names no type."""
    if lhs not in spec.mounts:
        return lhs
    if '/' in value:
        return value.partition('/')[0]
    accepts = spec.mounts[lhs].accepts
    return accepts[0] if len(accepts) == 1 else None


def _superseded(spec: AssemblySpec, lhs: str, values: list[str], caller: dict[str, list[str]]) -> bool:
    """Whether the caller's directives replace a preset directive of the same key, type or socket."""
    if lhs in caller:
        return True
    type_ = _type_of(spec, lhs, values[0].partition('[')[0])
    if lhs in spec.mounts:
        return type_ in caller
    return any(other in spec.mounts and lhs in spec.mounts[other].accepts for other in caller)


@functools.cache
def lookup(name: str) -> Preset | None:
    """The validated preset of that name, ``None`` when no robot directory of it holds a ``preset.yaml``."""
    if not name or '/' in name:
        return None
    path = Path(ARENA_ROBOTS_DIR) / 'robots' / name / FILE_NAME
    if not path.is_file():
        return None
    preset = Preset.read(path)
    preset.validate()
    return preset


def expand(robot: str, parts: dict[str, list[str]], caps: dict[str, typing.Any]) -> tuple[str, dict[str, list[str]], dict[str, typing.Any]]:
    """The base robot and the merged parts and caps of ``robot``, unchanged for a robot that is no preset."""
    preset = lookup(robot)
    if preset is None:
        return robot, parts, caps
    merged_parts, merged_caps = preset.merged(parts, caps)
    return preset.robot, merged_parts, merged_caps
