"""The override paths an instance's effective caps expose, and completion of partial `robot:=` entries over them."""

from __future__ import annotations

import functools
import typing
from collections.abc import Iterable, Iterator, Mapping

import attrs

from arena_robots import assembly as assembly_mod
from arena_robots.caps import FACT_BLOCKS, CapConfig, RobotCaps, adapter_blocks, simulator_names, simulators_honoring, typed_fields
from arena_robots.inline import split_top_level
from arena_robots.Robot import RobotIdentifier, RobotView
from arena_robots.SetupFile import Config

Kind = typing.Literal['value', 'list', 'map', 'unset', 'arg']

_INSTANCE_CAPS = ('arm', 'lift', 'gripper')
_SUFFIXES: dict[Kind, tuple[str, ...]] = {'value': ('=',), 'list': ('=',), 'map': ('.',), 'unset': ('=', '.'), 'arg': ('=',)}
_NOT_OVERRIDABLE = ('body.mass', 'body.mass_urdf', 'body.outline', 'mobile.laser')
_COMPLETION_ERRORS = (ValueError, RuntimeError, FileNotFoundError, KeyError)


@attrs.frozen
class OverridePath:
    """One addressable value of an instance's effective caps."""

    path: str
    value: object
    kind: Kind
    provenance: str = 'file'
    part: str | None = None
    relative: str | None = None
    simulators: frozenset[str] = frozenset()

    @property
    def overridable(self) -> bool:
        return not any(self.path == head or self.path.startswith(f'{head}.') for head in _NOT_OVERRIDABLE)


def _kind(value: object) -> Kind:
    if isinstance(value, dict):
        return 'map'
    return 'list' if isinstance(value, list) else 'value'


def _walk(prefix: str, data: Mapping[str, object]) -> Iterator[tuple[str, object, Kind]]:
    for key, value in data.items():
        name = str(key)
        if '.' in name:
            continue
        yield f'{prefix}{name}', value, _kind(value)
        if isinstance(value, dict):
            yield from _walk(f'{prefix}{name}.', value)


def _relative(spec: CapConfig, cap: str) -> list[tuple[str, object, Kind]]:
    entries = list(_walk('', spec.raw))
    if cap in FACT_BLOCKS:
        return entries
    unset = sorted((typed_fields(type(spec)) | adapter_blocks(cap)) - {str(key) for key in spec.raw})
    return entries + [(name, None, 'unset') for name in unset]


def _arg_paths(caps: RobotCaps) -> list[OverridePath]:
    if caps.resolved is None or caps.catalog is None:
        return []
    return [
        OverridePath(f'{placement.type}.{placement.mount.name}.{arg}', placement.params.get(arg, default), 'arg', 'override' if arg in placement.args else 'component', placement.mount.name, arg, simulator_names())
        for placement in caps.resolved.placements
        for arg, default in caps.catalog.get(placement.type, placement.variant).args.items()
    ]


def override_paths(caps: RobotCaps) -> list[OverridePath]:
    """Every leaf, list and mapping of the effective caps and fact blocks, each with its provenance, plus the typed fields and adapter blocks the caps leave unset and the xacro arguments of the placed parts."""
    out: list[OverridePath] = []
    if 'mobile' in caps.available:
        out += [OverridePath(f'mobile.{relative}', value, kind, caps.provenance(f'mobile.{relative}'), simulators=simulators_honoring(f'mobile.{relative}', caps)) for relative, value, kind in _relative(caps.mobile, 'mobile')]
    placed = {(placement.type, placement.mount.name) for placement in caps.resolved.placements} if caps.resolved is not None else set()
    for cap in _INSTANCE_CAPS:
        for name, spec in (getattr(caps, cap) or {}).items():
            part = name if (cap, name) in placed else None
            out += [OverridePath(f'{cap}.{name}.{relative}', value, kind, caps.provenance(f'{cap}.{name}.{relative}'), part, relative if part else None, simulators_honoring(f'{cap}.{name}.{relative}', caps)) for relative, value, kind in _relative(spec, cap)]
    for block in FACT_BLOCKS:
        spec = getattr(caps, block)
        if spec is not None:
            out += [OverridePath(f'{block}.{relative}', value, kind, caps.provenance(f'{block}.{relative}'), simulators=simulators_honoring(f'{block}.{relative}', caps)) for relative, value, kind in _relative(spec, block)]
    return out + _arg_paths(caps)


def _split(text: str) -> tuple[list[str], str, int] | None:
    done: list[str] = []
    depth = 0
    start = 0
    for index, ch in enumerate(text):
        if ch == '[':
            depth += 1
        elif ch == ']':
            depth -= 1
            if depth < 0:
                return None
        elif ch == ',' and depth == 0:
            done.append(text[start:index])
            start = index + 1
    return done, text[start:], depth


def _entry_tokens(entry: str) -> tuple[str, list[str]]:
    model, bracket, rest = entry.strip().partition('[')
    if not bracket:
        return model.strip(), []
    if not rest.endswith(']'):
        raise ValueError(f"robot entry {entry!r}: missing closing ']'")
    scanned = _split(rest[:-1])
    if scanned is None or scanned[2] != 0:
        raise ValueError(f"robot entry {entry!r}: unbalanced brackets")
    return model.strip(), split_top_level(rest[:-1])


def _entry_dict(model: str, tokens: Iterable[str]) -> dict[str, object]:
    result: dict[str, object] = {'robot': model}
    for item in tokens:
        key, sep, value = item.partition('=')
        if not sep:
            try:
                count = int(item)
            except ValueError:
                raise ValueError(f"unknown token {item!r}, expected an integer count or key=value") from None
            if 'count' in result:
                raise ValueError(f"duplicate count in {item!r}")
            result['count'] = count
            continue
        key = key.strip()
        value = value.strip()
        parsed: str | list[str] = [v.strip() for v in split_top_level(value[1:-1])] if value.startswith('[') and value.endswith(']') else value
        cap, dot, tail = key.partition('.')
        if dot and cap and tail != 'adapter':
            if key in result:
                raise ValueError(f"duplicate cap override {key!r}")
            result[key] = parsed
        elif key in result:
            existing = result[key] if isinstance(result[key], list) else [result[key]]
            result[key] = [*typing.cast(list[object], existing), *(parsed if isinstance(parsed, list) else [parsed])]
        else:
            result[key] = parsed
    return result


@functools.lru_cache(maxsize=64)
def _view(model: str) -> RobotView:
    return RobotIdentifier(model).resolve_sync()


def _effective_caps(view: RobotView, config: Config) -> RobotCaps:
    spec = view.assembly
    if spec is None:
        if config.parts or config.frames:
            raise ValueError(f"morphology and frames need an assembly.yaml, and {view.name!r} has none")
        return view.effective_caps({}, caps=dict(config.caps))
    request, cleared_sockets, cleared_types = assembly_mod.build_request(spec, config.parts)
    resolved = assembly_mod.resolve(spec, request, cleared_sockets=cleared_sockets, cleared_types=cleared_types)
    unknown_frames = set(config.frames) - set(spec.mounts)
    if unknown_frames:
        raise ValueError(f"frames override targets unknown mount(s) {sorted(unknown_frames)}, declared mounts: {sorted(spec.mounts)}")
    resolved = assembly_mod.apply_frame_overrides(resolved, config.frames)
    return view.effective_caps(assembly_mod.placed_request(resolved), frames=config.frames, caps=dict(config.caps))


def entry_paths(entry: str) -> list[OverridePath]:
    """The override paths of one complete `robot:=` entry, with its own overrides applied."""
    model, tokens = _entry_tokens(entry)
    caps = _effective_caps(_view(model), Config.parse(_entry_dict(model, tokens))[0])
    caps.validate_overrides()
    return override_paths(caps)


@functools.lru_cache(maxsize=256)
def _instance(model: str, directives: tuple[tuple[str, str], ...]) -> tuple[tuple[OverridePath, ...], frozenset[tuple[str, str, str]]]:
    config = Config.parse(_entry_dict(model, [f'{key}={value}' for key, value in directives]))[0]
    caps = _effective_caps(_view(model), config)
    placements = frozenset((p.type, p.variant, p.mount.name) for p in caps.resolved.placements) if caps.resolved is not None else frozenset()
    return tuple(override_paths(caps)), placements


def _directives(tokens: Iterable[str]) -> tuple[tuple[str, str], ...]:
    out: list[tuple[str, str]] = []
    for token in tokens:
        key, sep, value = token.partition('=')
        key = key.strip()
        if not sep or ('.' in key and not key.startswith('frames.')):
            continue
        value = value.strip()
        out.append((key, value.partition('[')[0] or value))
    return tuple(out)


def _fold(token: str, entries: Iterable[tuple[str, Kind]]) -> list[str]:
    found: set[str] = set()
    for path, kind in entries:
        if not path.startswith(token):
            continue
        cut = path.find('.', len(token))
        if cut >= 0:
            found.add(path[: cut + 1])
        else:
            found.update(path + suffix for suffix in _SUFFIXES[kind])
    return sorted(found)


def _complete(entry: str, model: str, done: list[str], current: str, depth: int) -> list[str]:
    token = current.lstrip()
    directives = _directives(done)
    if depth == 0:
        if '=' in token or '[' in token:
            return []
        paths, _ = _instance(model, directives)
        return [entry[: len(entry) - len(token)] + candidate for candidate in _fold(token, [(p.path, p.kind) for p in paths if p.part is None and p.overridable])]
    if depth != 1:
        return []
    part_head, _, inner = token.partition('[')
    lhs, sep, value = part_head.partition('=')
    lhs = lhs.strip()
    if not sep or '.' in lhs:
        return []
    scanned = _split(inner)
    if scanned is None or scanned[2] != 0:
        return []
    inner_token = scanned[1].lstrip()
    if '=' in inner_token:
        return []
    value = value.strip()
    paths, placements = _instance(model, (*directives, (lhs, value)))
    spec = _view(model).assembly
    if spec is None:
        return []
    variant = value.rpartition('/')[2]
    mounts = {lhs} if lhs in spec.mounts else {mount for placed_type, placed_variant, mount in placements if placed_type == lhs and placed_variant == variant}
    relative: list[tuple[str, Kind]] = [(p.relative, p.kind) for p in paths if p.part in mounts and p.relative is not None and p.overridable]
    return [entry[: len(entry) - len(inner_token)] + candidate for candidate in _fold(inner_token, relative)]


def complete(entry: str) -> list[str]:
    """The full entries completing the token under the cursor of a partial `robot:=` entry, `[]` for an unknown model or an unparsable prefix."""
    model, bracket, body = entry.lstrip().partition('[')
    scanned = _split(body)
    if not bracket or scanned is None:
        return []
    try:
        return _complete(entry, model.strip(), *scanned)
    except _COMPLETION_ERRORS:
        return []
