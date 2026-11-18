"""The inline cap override grammar shared by `robot:=` entries, setup files and part brackets."""

from __future__ import annotations

import re
import typing

import yaml

SCALE_KEY = 'scale'
OFFSET_KEY = 'offset'
APPEND_KEY = 'append'
DELETE_KEY = 'delete'
WILDCARD = '*'

_NUMBER = r'(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?'
_AFFINE = re.compile(rf'\*(?P<scale>[+-]?{_NUMBER})(?P<offset>[+-]{_NUMBER})?')
_OFFSET = re.compile(rf'\+(?P<offset>[+-]?{_NUMBER})')


def is_affine(value: object) -> bool:
    """Whether a cap override value is ``{scale: a, offset: b}`` (either key may be left out), mapping ``v`` to ``a*v + b``."""
    return isinstance(value, dict) and bool(value) and set(value) <= {SCALE_KEY, OFFSET_KEY}


def is_append(value: object) -> bool:
    """Whether a cap override value is ``{append: [...]}``, extending a list."""
    return isinstance(value, dict) and set(value) == {APPEND_KEY} and isinstance(value[APPEND_KEY], list)


def is_delete(value: object) -> bool:
    """Whether a cap override value is ``{delete: true}``, removing a key or list element."""
    return isinstance(value, dict) and set(value) == {DELETE_KEY} and value[DELETE_KEY] is True


def split_top_level(text: str) -> list[str]:
    """Comma-split ``text`` at bracket depth 0, dropping blank tokens."""
    tokens: list[str] = []
    current: list[str] = []
    depth = 0
    for ch in text:
        if ch == '[':
            depth += 1
        elif ch == ']':
            depth -= 1
        if ch == ',' and depth == 0:
            tokens.append(''.join(current))
            current = []
        else:
            current.append(ch)
    tokens.append(''.join(current))
    return [t.strip() for t in tokens if t.strip()]


def parse_inline_override(value: str | list[str]) -> object:
    """Cap override value written inline: `*a`, `+b` or `*a+b` is affine, `+[...]` appends, `!delete` removes, anything else is YAML."""
    if isinstance(value, list):
        return [parse_inline_override(v) for v in value]
    if value == f'!{DELETE_KEY}':
        return {DELETE_KEY: True}
    if value.startswith('+['):
        items = yaml.safe_load(value[1:])
        if not isinstance(items, list):
            raise ValueError(f"cap override append {value!r}: expected '+[<item>,...]'")
        return {APPEND_KEY: items}
    if value.startswith(('*', '+')):
        affine = _AFFINE.fullmatch(value)
        if affine is not None:
            out: dict[str, object] = {SCALE_KEY: float(affine['scale'])}
            if affine['offset'] is not None:
                out[OFFSET_KEY] = float(affine['offset'])
            return out
        offset = _OFFSET.fullmatch(value)
        if offset is not None:
            return {OFFSET_KEY: float(offset['offset'])}
        raise ValueError(f"cap override {value!r}: expected '*<scale>', '+<offset>', '*<scale>+<offset>' or '+[<item>,...]'")
    return yaml.safe_load(value)


def format_inline_override(value: object) -> str:
    """Inverse of :func:`parse_inline_override` for one value."""
    if is_delete(value):
        return f'!{DELETE_KEY}'
    if is_append(value):
        return '+' + format_inline_override(typing.cast(dict[str, object], value)[APPEND_KEY])
    if is_affine(value):
        ops = typing.cast(dict[str, float], value)
        offset = ops.get(OFFSET_KEY)
        if SCALE_KEY not in ops:
            return f'+{offset}'
        scale = f'*{ops[SCALE_KEY]}'
        if offset is None:
            return scale
        return f'{scale}{offset:+}'
    if isinstance(value, list):
        return '[' + ','.join(format_inline_override(v) for v in value) + ']'
    if isinstance(value, dict):
        raise ValueError(f"cap override value {value!r}: a mapping has no inline form, override its fields one by one")
    return yaml.safe_dump(value, default_flow_style=True).removesuffix('\n...\n').strip()


def split_part(value: str) -> tuple[str, dict[str, object]]:
    """A part directive value `<variant>[<field>=<value>,...]` as the variant and its cap overrides."""
    head, bracket, rest = value.partition('[')
    if not bracket:
        return value, {}
    if not rest.endswith(']') or not head:
        raise ValueError(f"part {value!r}: expected '<variant>[<field>=<value>,...]'")
    caps: dict[str, object] = {}
    for item in split_top_level(rest[:-1]):
        key, sep, raw = item.partition('=')
        key = key.strip()
        if not sep or not key or not all(key.split('.')):
            raise ValueError(f"part {value!r}: override {item!r} is not '<field>=<value>'")
        if key in caps:
            raise ValueError(f"part {value!r}: duplicate override {key!r}")
        caps[key] = parse_inline_override(raw.strip())
    return head.strip(), caps


def format_part(variant: str, caps: typing.Mapping[str, object], params: typing.Mapping[str, object] | None = None) -> str:
    """Inverse of :func:`split_part`, the xacro arguments ``params`` leading the cap overrides."""
    entries = {**(params or {}), **caps}
    if not entries:
        return variant
    return f"{variant}[{','.join(f'{key}={format_inline_override(value)}' for key, value in entries.items())}]"
