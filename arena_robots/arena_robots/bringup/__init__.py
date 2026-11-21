from __future__ import annotations

import functools
import json
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

from arena_robots import urdf_facts
from arena_robots.request import InstanceRequest
from arena_robots.Sensor import SensorType
from arena_robots.sensor_geometry import parse_sensor_geometry
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

    def _temp_yaml(self, stem: str, data: dict) -> str:
        label = str(self.namespace).strip("/").replace("/", "_") or self.robot.name
        path = os.path.join(tempfile.gettempdir(), f"{stem}_{socket.gethostname()}_{label}.yaml")
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f)
        return path

    def planner_launch_arguments(self) -> dict[str, str]:
        """`mobile_caps` (per-host, per-namespace temp file of the effective mobile cap with the laser block and wheelbase of the rendered URDF) and `sensors_json` (effective sensors with their geometry) for the nav2 launch files."""
        laser = self.caps.laser_block()
        geometry = parse_sensor_geometry(self.caps.rendered_urdf())
        sensors = self.robot.effective_sensors(self.parts, frames=self.caps.frames or None)
        controller = self.robot.drive_controller()
        wheelbase = None if controller is None else urdf_facts.wheelbase(self.caps.rendered_urdf(), controller[1], controller[2])
        return {
            "mobile_caps": self._temp_yaml("mobile_caps", {**self.caps.mobile.raw, **({} if laser is None else {"laser": laser}), **({} if wheelbase is None else {"wheelbase": wheelbase})}),
            "sensors_json": json.dumps(
                [
                    {
                        "name": s.name,
                        "type": s.type.value if isinstance(s.type, SensorType) else str(s.type),
                        "topic": s.topic,
                        "frame": s.frame,
                        "sensor": s.sensor,
                        "geometry": attrs.asdict(geometry[s.sensor]) if s.sensor in geometry else None,
                    }
                    for s in sensors
                ]
            ),
        }

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
        from ament_index_python.packages import get_package_share_directory as get_share
        from launch_ros.actions import Node

        nodes = []

        power_params = self.robot.power_profile(self.caps)
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

        acoustics = self.caps.acoustics
        profile_path = self._temp_yaml("acoustics_profile", acoustics.raw) if acoustics is not None else os.path.join(get_share('arena_robots'), 'config', 'acoustic_profile.yaml')
        if os.path.isfile(profile_path):
            nodes.append(
                Node(
                    package='arena_robots',
                    executable='acoustics_publisher',
                    name='acoustics_publisher',
                    namespace=str(self.namespace),
                    parameters=[{'robot_name': self.robot.name, 'profile_path': profile_path}],
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
