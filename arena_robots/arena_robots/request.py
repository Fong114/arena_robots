"""One robot instance's request beyond its model, as passed between processes."""

from __future__ import annotations

import json
import typing

import attrs

from arena_robots import assembly
from arena_robots.assembly import RequestPart

if typing.TYPE_CHECKING:
    from arena_robots.caps import RobotCaps
    from arena_robots.Robot import RobotView
    from arena_robots.SetupFile import Config


@attrs.define(frozen=True)
class InstanceRequest:
    """Morphology parts (with their xacro arguments), mount frame overrides and cap overrides of one robot instance."""

    parts: dict[str, list[RequestPart]] = attrs.field(factory=dict)
    frames: dict[str, str] = attrs.field(factory=dict)
    caps: dict[str, object] = attrs.field(factory=dict)

    def caps_of(self, view: RobotView) -> RobotCaps:
        """The instance's effective caps on ``view``."""
        return view.effective_caps(self.parts, frames=self.frames, caps=self.caps)

    @classmethod
    def of(cls, view: RobotView, config: Config) -> InstanceRequest:
        """The instance of ``config`` on ``view``, its parts resolved against the view's assembly."""
        spec = view.assembly
        if spec is None:
            if config.parts or config.frames:
                raise ValueError(f"morphology and frames need an assembly.yaml, and {view.name!r} has none")
            return cls(caps=dict(config.caps))
        request, cleared_sockets, cleared_types = assembly.build_request(spec, config.parts)
        resolved = assembly.resolve(spec, request, cleared_sockets=cleared_sockets, cleared_types=cleared_types)
        unknown_frames = set(config.frames) - set(spec.mounts)
        if unknown_frames:
            raise ValueError(f"frames override targets unknown mount(s) {sorted(unknown_frames)}, declared mounts: {sorted(spec.mounts)}")
        resolved = assembly.apply_frame_overrides(resolved, config.frames)
        return cls(parts=assembly.placed_request(resolved), frames=dict(config.frames), caps=dict(config.caps))

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
