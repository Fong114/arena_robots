from __future__ import annotations

import functools
import os
import socket
import tempfile
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING, ClassVar

import attrs
import yaml

if TYPE_CHECKING:
    from arena_robots.caps import RobotCaps
    from arena_robots.Robot import RobotView
    from arena_robots.Sensor import SensorSpec
    from arena_robots.task_server_handlers import TaskHandler

from arena_rclpy_mixins.shared import Namespace
from launch import Action

from arena_robots.request import InstanceRequest
from arena_robots.task_kinds import TaskKind


class AdapterCapMismatch(RuntimeError):
    pass


def _to_frozenset_str(v: str | Iterable[str]) -> frozenset[str]:
    if isinstance(v, str):
        return frozenset({v})
    return frozenset(v)


@attrs.frozen
class BringupMeta:
    """Static metadata attached to a Bringup class."""

    requires: frozenset[str] = attrs.field(converter=_to_frozenset_str)
    cap: str = ""

    @classmethod
    def attach[T: type[Bringup]](cls, *, requires: str | Iterable[str], cap: str = "") -> Callable[[T], T]:
        meta = cls(requires=requires, cap=cap)

        def wrap(target: T) -> T:
            target._bringup_meta = meta
            return target

        return wrap


class Bringup(ABC):
    kind: ClassVar[str]
    _bringup_meta: ClassVar[BringupMeta]
    task_handlers: ClassVar[dict[TaskKind, Callable[[], type[TaskHandler]]]] = {}

    def __init__(self, robot: RobotView, namespace: str, *, frame: str = "", request: InstanceRequest | None = None) -> None:
        self.robot = robot
        self.namespace = Namespace(namespace)
        self.frame = frame
        self.request = request if request is not None else InstanceRequest()

    @property
    def parts(self) -> dict[str, list]:
        return self.request.parts

    @functools.cached_property
    def caps(self) -> RobotCaps:
        """The instance's effective caps: parts, frame overrides and cap overrides applied."""
        return self.request.caps_of(self.robot)

    def mobile_caps_launch_arguments(self) -> dict[str, str]:
        """`mobile_caps` pointing at the per-host, per-namespace temp file of the effective mobile cap when the request overrides it, else empty."""
        if not any(key.startswith("mobile.") for key in self.request.caps):
            return {}
        label = str(self.namespace).strip("/").replace("/", "_") or self.robot.name
        path = os.path.join(tempfile.gettempdir(), f"mobile_caps_{socket.gethostname()}_{label}.yaml")
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.caps.mobile.raw, f)
        return {"mobile_caps": path}

    @property
    def cap(self) -> str:
        return self._bringup_meta.cap

    @abstractmethod
    def _launch_actions(
        self,
        *,
        use_sim_time: bool = True,
        frame: str = "",
        arena_env_node: str = "",
        env_namespace: str = "",
        sensors: list[SensorSpec] | None = None,
    ) -> list[Action]: ...

    def telemetry_actions(self) -> list[Action]:
        import os

        from ament_index_python.packages import get_package_share_directory as get_share
        from launch_ros.actions import Node

        nodes = []

        power_params = self.robot.power_profile(self.parts)
        if power_params is not None:
            nodes.append(
                Node(
                    package='arena_robots',
                    executable='power_publisher',
                    name='power_publisher',
                    namespace=str(self.namespace),
                    parameters=[power_params],
                    output='screen',
                )
            )

        acoustics_yaml = os.path.join(get_share('arena_robots'), 'robots', self.robot.name, 'telemetry', 'acoustics.yaml')
        fallback_yaml = os.path.join(get_share('arena_robots'), 'config', 'acoustic_profile.yaml')

        if os.path.isfile(acoustics_yaml) or os.path.isfile(fallback_yaml):
            used_yaml = acoustics_yaml if os.path.isfile(acoustics_yaml) else fallback_yaml
            nodes.append(
                Node(
                    package='arena_robots',
                    executable='acoustics_publisher',
                    name='acoustics_publisher',
                    namespace=str(self.namespace),
                    parameters=[{'robot_name': self.robot.name, 'profile_path': used_yaml}],
                    output='screen',
                )
            )

        return nodes

    @property
    def requires(self) -> frozenset[str]:
        return self._bringup_meta.requires

    @property
    def accepts_task_kinds(self) -> frozenset[TaskKind]:
        return frozenset(self.task_handlers.keys())


def check_caps(bringup: Bringup) -> None:
    available = bringup.caps.available
    missing = bringup.requires - available
    if missing:
        raise AdapterCapMismatch(f"Bringup {bringup.kind!r} requires caps {sorted(missing)} but robot {bringup.robot.name!r} only advertises {sorted(available)}")


from arena_rclpy_mixins.registry import ClassRegistry

BRINGUPS: dict[str, ClassRegistry[str, type[Bringup]]] = {
    "mobile": ClassRegistry(),
    "arm": ClassRegistry(),
    "lift": ClassRegistry(),
    "gripper": ClassRegistry(),
}

from . import arm, gripper, lift, mobile  # noqa: F401, E402
