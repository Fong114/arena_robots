"""Fleet robots bound to their model-derived frames, odometry topics and drivetrain facts."""

from __future__ import annotations

import functools
import json
import math
import subprocess
import typing
from collections.abc import Mapping

import attrs
import yaml

from arena_robots.assembly import RequestPart, ResolvedAssembly
from arena_robots.caps import RobotCaps
from arena_robots.Robot import RobotIdentifier, RobotView

if typing.TYPE_CHECKING:
    from arena_runtime_msgs.msg import RobotFleet


def _join_frame(*parts: str) -> str:
    return "/".join(part.strip("/") for part in parts if part.strip("/"))


DEFAULT_ODOM_TOPIC_TEMPLATE = "{namespace}/{name}_velocity_controller/odom"

_MODEL_ERRORS = (OSError, LookupError, ValueError, TypeError, yaml.YAMLError, subprocess.CalledProcessError)


@attrs.frozen(kw_only=True)
class _ModelFacts:
    base_frame: str
    odom_topic: str
    wheel_separation_m: float | None
    max_linear_mps: float


def _wheel_separation(control: Mapping[str, typing.Any]) -> float | None:
    for name, value in control.items():
        if name == "controller_manager" or not isinstance(value, Mapping):
            continue
        params = value.get("ros__parameters")
        if not isinstance(params, Mapping) or "wheel_separation" not in params:
            continue
        effective = float(params["wheel_separation"]) * float(params.get("wheel_separation_multiplier", 1.0))
        return effective if effective > 0.0 and math.isfinite(effective) else None
    return None


PART_ARG_PREFIX = "param:"


def placement_params(resolved: ResolvedAssembly) -> dict[str, str]:
    """The `param:<type>@<mount>.<arg>` fleet params holding each placement's xacro arguments as JSON values."""
    return {f"{PART_ARG_PREFIX}{p.type}@{p.mount.name}.{arg}": json.dumps(value) for p in resolved.placements for arg, value in p.args.items()}


def _part_args(params: Mapping[str, str]) -> dict[tuple[str, str], dict[str, object]]:
    out: dict[tuple[str, str], dict[str, object]] = {}
    for key, value in params.items():
        if not key.startswith(PART_ARG_PREFIX):
            continue
        owner, _, arg = key.removeprefix(PART_ARG_PREFIX).rpartition(".")
        type_, _, mount = owner.partition("@")
        out.setdefault((type_, mount), {})[arg] = json.loads(value)
    return out


def fleet_parts(view: RobotView, params: Mapping[str, str]) -> dict[str, list[RequestPart]]:
    """The assembly request a fleet entry's `<type>@<mount>` and `param:<type>@<mount>.<arg>` params place, defaults' overrides kept."""
    spec = view.assembly
    if spec is None:
        return {}
    defaults = {(type_, part.mount, part.variant): part for type_, parts in spec.defaults.items() for part in parts}
    args = _part_args(params)
    request: dict[str, list[RequestPart]] = {}
    for key, variant in params.items():
        if key.startswith(PART_ARG_PREFIX):
            continue
        type_, sep, mount = key.partition("@")
        if not sep:
            continue
        default = defaults.get((type_, mount, variant)) or defaults.get((type_, None, variant))
        request.setdefault(type_, []).append(RequestPart(variant=variant, mount=mount, params=args.get((type_, mount), {}), overrides=dict(default.overrides) if default is not None else {}))
    return request


def fleet_cap_overrides(params: Mapping[str, str]) -> dict[str, typing.Any]:
    """The cap overrides a fleet entry carries as `cap:<path>` params holding JSON values."""
    return {key.removeprefix("cap:"): json.loads(value) for key, value in params.items() if key.startswith("cap:")}


@functools.cache
def _instance_caps(model: str, params: frozenset[tuple[str, str]]) -> RobotCaps:
    view = RobotIdentifier(model).resolve_sync()
    mapping = dict(params)
    return view.effective_caps(fleet_parts(view, mapping), caps=fleet_cap_overrides(mapping))


def instance_caps(model: str, params: Mapping[str, str]) -> RobotCaps:
    """The effective caps of a fleet entry's instance from its `<type>@<mount>`, `param:<type>@<mount>.<arg>` and `cap:<path>` params."""
    return _instance_caps(model, frozenset(params.items()))


@functools.cache
def _model_facts(model: str, params: frozenset[tuple[str, str]]) -> _ModelFacts:
    view = RobotIdentifier(model).resolve_sync()
    caps = _instance_caps(model, params)
    body = caps.body
    mobile = caps.mobile if "mobile" in caps.available else None
    try:
        separation = _wheel_separation(view.control)
    except _MODEL_ERRORS:
        separation = None
    return _ModelFacts(
        base_frame=body.base_frame.strip("/"),
        odom_topic=body.odom_topic.strip("/"),
        wheel_separation_m=separation,
        max_linear_mps=float(mobile.velocity_limits.linear.max) if mobile is not None and mobile.velocity_limits is not None else 0.0,
    )


@attrs.frozen(kw_only=True)
class RobotBinding:
    """A fleet robot with its model-derived frames, odometry topics and drivetrain facts."""

    name: str
    model: str
    namespace: str
    frame_prefix: str
    model_base_frame: str
    base_frame: str
    odom_topics: tuple[str, ...]
    wheel_separation_m: float | None
    max_linear_mps: float
    error: str = ""

    def frame(self, leaf: str) -> str:
        return _join_frame(self.frame_prefix, leaf)

    def mount(self, template: str) -> str:
        """Array mount frame of an array.mount_frame template: empty = base frame, {prefix} and {base_frame} expand, a bare leaf joins the prefix."""
        configured = template.format(prefix=self.frame_prefix, base_frame=self.model_base_frame).strip("/")
        if not configured:
            return self.base_frame
        if "/" in configured:
            return configured
        return self.frame(configured)

    @classmethod
    def resolve(cls, *, name: str, model: str, namespace: str, frame_prefix: str, params: Mapping[str, str] | None = None, odom_topic_template: str = DEFAULT_ODOM_TOPIC_TEMPLATE) -> RobotBinding:
        """Model lookups that fail fall back to base_link, no model odometry, no wheel separation and 0 m/s, with error set."""
        namespace = namespace.rstrip("/")
        prefix = frame_prefix.strip("/")
        error = ""
        try:
            facts = _model_facts(model, frozenset((params or {}).items()))
        except _MODEL_ERRORS as exc:
            error = f"could not resolve robot model {model!r}: {exc}"
            facts = _ModelFacts(base_frame="base_link", odom_topic="", wheel_separation_m=None, max_linear_mps=0.0)
        topics = [odom_topic_template.format(namespace=namespace, name=name), f"{namespace}/odom"]
        if facts.odom_topic:
            topics.append(f"{namespace}/{facts.odom_topic}")
        return cls(
            name=name,
            model=model,
            namespace=namespace,
            frame_prefix=prefix,
            model_base_frame=facts.base_frame,
            base_frame=_join_frame(prefix, facts.base_frame),
            odom_topics=tuple(dict.fromkeys(topic for topic in topics if topic)),
            wheel_separation_m=facts.wheel_separation_m,
            max_linear_mps=facts.max_linear_mps,
            error=error,
        )


def robot_bindings(fleet: RobotFleet, *, odom_topic_template: str = DEFAULT_ODOM_TOPIC_TEMPLATE) -> tuple[RobotBinding, ...]:
    """Every named fleet robot, in fleet order."""
    return tuple(
        RobotBinding.resolve(
            name=str(state.descriptor.name).strip(),
            model=str(state.descriptor.model),
            namespace=str(state.descriptor.ns),
            frame_prefix=str(state.descriptor.frame),
            params={item.key: item.value for item in state.params},
            odom_topic_template=odom_topic_template,
        )
        for state in fleet.robots
        if str(state.descriptor.name).strip()
    )


def bind_robot(fleet: RobotFleet, wanted: str = "", *, odom_topic_template: str = DEFAULT_ODOM_TOPIC_TEMPLATE) -> RobotBinding | None:
    """The fleet robot named wanted, or the first one when empty. None until the fleet carries it."""
    for state in fleet.robots:
        name = str(state.descriptor.name).strip()
        if name and (not wanted or name == wanted):
            return RobotBinding.resolve(
                name=name,
                model=str(state.descriptor.model),
                namespace=str(state.descriptor.ns),
                frame_prefix=str(state.descriptor.frame),
                params={item.key: item.value for item in state.params},
                odom_topic_template=odom_topic_template,
            )
    return None
