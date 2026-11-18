"""One robot instance's request beyond its model, as passed between processes."""

from __future__ import annotations

import json
import typing

import attrs

from arena_robots.assembly import RequestPart

if typing.TYPE_CHECKING:
    from arena_robots.caps import RobotCaps
    from arena_robots.Robot import RobotView


@attrs.define(frozen=True)
class InstanceRequest:
    """Morphology parts (with their xacro arguments), mount frame overrides and cap overrides of one robot instance."""

    parts: dict[str, list[RequestPart]] = attrs.field(factory=dict)
    frames: dict[str, str] = attrs.field(factory=dict)
    caps: dict[str, object] = attrs.field(factory=dict)

    def caps_of(self, view: RobotView) -> RobotCaps:
        """The instance's effective caps on ``view``."""
        return view.effective_caps(self.parts, frames=self.frames, caps=self.caps)

    def dumps(self) -> str:
        return json.dumps(
            {
                'parts': {kind: [{'variant': p.variant, 'mount': p.mount, 'params': p.params, 'overrides': p.overrides, 'caps': p.caps} for p in parts] for kind, parts in self.parts.items()},
                'frames': self.frames,
                'caps': self.caps,
            }
        )

    @classmethod
    def loads(cls, text: str) -> InstanceRequest:
        data = json.loads(text) if text else {}
        return cls(
            parts={kind: [RequestPart(variant=p['variant'], mount=p.get('mount'), params=dict(p.get('params') or {}), overrides=dict(p.get('overrides') or {}), caps=dict(p.get('caps') or {})) for p in parts] for kind, parts in data.get('parts', {}).items()},
            frames={str(k): str(v) for k, v in data.get('frames', {}).items()},
            caps=dict(data.get('caps', {})),
        )
