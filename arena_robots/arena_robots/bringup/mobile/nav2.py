from __future__ import annotations

import os
from typing import ClassVar

from ament_index_python.packages import get_package_share_directory
from launch import Action
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare

from arena_robots.bringup import Bringup, BringupMeta
from arena_robots.bringup.mobile.rosnav_rl import agent_range_actions
from arena_robots.Sensor import SensorSpec, SensorType
from arena_robots.task_kinds import TaskKind

DEFAULT_INTER_PLANNER = "default"
NO_SPIN_INTER_PLANNER = "default_no_spin"  # the default tree of a base that cannot turn in place


def _load_goto_pose_nav2() -> type:
    from arena_robots.task_server_handlers.goto_pose.nav2 import GotoPoseHandlerNav2

    return GotoPoseHandlerNav2


def scan_topic(sensors: list[SensorSpec], namespace: str) -> str:
    """The first laserscan topic among the sensors, its namespace placeholder resolved, '' without one."""
    for sensor in sensors:
        if str(sensor.type) == SensorType.LASERSCAN.value:
            return sensor.topic.replace("${namespace}", namespace)
    return ""


@BringupMeta.attach(requires={"mobile"}, cap="mobile")
class Nav2Bringup(Bringup):
    kind = "nav2"
    task_handlers: ClassVar[dict] = {TaskKind.GOTO_POSE: _load_goto_pose_nav2}

    @property
    def native_action_name(self) -> str:
        return self.namespace("navigate_to_pose")

    @property
    def bt_node_name(self) -> str:
        return self.namespace("bt_navigator")

    def _launch_actions(
        self,
        *,
        use_sim_time: bool = True,
        frame: str = "",
        global_planner: str = "navfn",
        local_planner: str = "regulated_pure_pursuit",
        inter_planner: str = DEFAULT_INTER_PLANNER,
        train_mode: bool = False,
        arena_env_node: str = "",
        env_namespace: str = "",
        sensors: list[SensorSpec] | None = None,
        params_overlay: str = "",
        agent: str = "",
        mapping: str = "static",
        **_: object,
    ) -> list[Action]:
        agent_name = agent or str(self.caps.mobile.sub("rosnav_rl").get("agent", ""))
        if local_planner == "rosnav_rl" and not agent_name:
            raise ValueError(f"nav2 bringup for '{self.robot.name}' with local_planner=rosnav_rl missing required 'agent': set caps.yaml mobile 'rosnav_rl.agent' or pass robot.mobile.agent:=<name>")
        if inter_planner == DEFAULT_INTER_PLANNER and self.robot.caps.mobile.min_turning_radius > 0.0:
            inter_planner = NO_SPIN_INTER_PLANNER
        launch_file = PathJoinSubstitution(
            [
                FindPackageShare("arena_robots"),
                "launch",
                "adapters",
                "mobile",
                "nav2.launch.py",
            ]
        )
        if params_overlay and "/" not in params_overlay:
            params_overlay = os.path.join(get_package_share_directory("arena_robots"), "config", "nav2", "overlays", f"{params_overlay}.yaml")
        if params_overlay and not os.path.isfile(params_overlay):
            raise FileNotFoundError(f"params_overlay {params_overlay!r} does not exist")
        launch_arguments = {
            "robot": self.robot.name,
            "namespace": self.namespace,
            "use_sim_time": str(use_sim_time).lower(),
            "frame": frame,
            "base_frame": self.caps.body.base_frame,
            "global_planner": global_planner,
            "local_planner": local_planner,
            "inter_planner": inter_planner,
            "train_mode": str(train_mode).lower(),
            "arena_env_node": arena_env_node,
            "env_namespace": env_namespace,
            "params_overlay": params_overlay,
            "agent": agent_name,
            "mapping": mapping,
            "scan_topic": scan_topic(sensors or [], str(self.namespace)),
            **self.planner_launch_arguments(),
        }
        actions: list[Action] = [
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(launch_file),
                launch_arguments=launch_arguments.items(),
            )
        ]
        if local_planner == "rosnav_rl":
            actions.extend(agent_range_actions(self.robot.name, self.caps.mobile, agent_name))
        return actions
