from __future__ import annotations

import math
import os
from collections.abc import Mapping
from pathlib import Path
from typing import ClassVar

import yaml
from launch import Action
from launch.actions import IncludeLaunchDescription, LogInfo
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare

from arena_robots.bringup import Bringup, BringupMeta
from arena_robots.caps import MobileSpec
from arena_robots.task_kinds import TaskKind

AGENT_CONFIG_NAME = "training_config.yaml"


def _load_goto_pose_rosnav_rl() -> type:
    from arena_robots.task_server_handlers.goto_pose._passthrough import GotoPoseHandlerRosnavRl

    return GotoPoseHandlerRosnavRl


def agents_root(environ: Mapping[str, str]) -> Path | None:
    """The agents directory: ROSNAV_AGENTS_DIR, else ARENA_DATA_DIR/agents, None when neither is set."""
    if environ.get("ROSNAV_AGENTS_DIR"):
        return Path(environ["ROSNAV_AGENTS_DIR"]).expanduser()
    if environ.get("ARENA_DATA_DIR"):
        return Path(environ["ARENA_DATA_DIR"]).expanduser() / "agents"
    return None


def saved_action_ranges(config: Path) -> dict[str, tuple[float, float]]:
    """The saved agent's continuous mobile action ranges keyed linear, lateral (omnidirectional only) and angular, ValueError naming why they cannot be read."""
    try:
        data = yaml.safe_load(config.read_text())
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"{config} is unreadable: {exc}") from exc
    try:
        space = data["agent_config"]["action_space"]
        kind = space["type"]
        if kind == "differential_drive":
            names = {"linear": "linear_range", "angular": "angular_range"}
        elif kind == "omnidirectional":
            names = {"linear": "linear_range_x", "lateral": "linear_range_y", "angular": "angular_range"}
        else:
            raise ValueError(f"{config} has action space type {kind!r}, not a mobile one")
        return {axis: (float(space[key][0]), float(space[key][1])) for axis, key in names.items()}
    except (KeyError, TypeError, IndexError) as exc:
        raise ValueError(f"{config} carries no continuous mobile action ranges ({exc!r})") from exc


def range_mismatches(saved: Mapping[str, tuple[float, float]], mobile: MobileSpec) -> list[str]:
    """One description per axis where the saved agent range differs from the instance's velocity_limits."""
    limits = mobile.velocity_limits
    if limits is None:
        raise ValueError("the instance has no velocity_limits to compare against")
    wanted = {"linear": limits.linear, "angular": limits.angular}
    if "lateral" in saved:
        wanted["lateral"] = limits.lateral if limits.lateral is not None else limits.linear
    return [f"{axis}: agent [{saved[axis][0]}, {saved[axis][1]}] vs robot [{rng.min}, {rng.max}]" for axis, rng in wanted.items() if not (math.isclose(saved[axis][0], rng.min, abs_tol=1e-6) and math.isclose(saved[axis][1], rng.max, abs_tol=1e-6))]


def agent_range_actions(robot_name: str, mobile: MobileSpec, agent_name: str, environ: Mapping[str, str] = os.environ) -> list[Action]:
    """A log line when the saved agent's action ranges differ from the instance's velocity_limits or the check cannot run."""
    if not agent_name:
        return []
    root = agents_root(environ)
    if root is None:
        return [LogInfo(msg=f"robot {robot_name!r}: agent {agent_name!r} action range check skipped, neither ROSNAV_AGENTS_DIR nor ARENA_DATA_DIR is set")]
    try:
        mismatches = range_mismatches(saved_action_ranges(root / agent_name / AGENT_CONFIG_NAME), mobile)
    except ValueError as exc:
        return [LogInfo(msg=f"robot {robot_name!r}: agent {agent_name!r} action range check skipped: {exc}")]
    if not mismatches:
        return []
    return [LogInfo(msg=f"WARNING robot {robot_name!r}: agent {agent_name!r} was trained on action ranges that differ from the instance's velocity_limits ({'; '.join(mismatches)})")]


@BringupMeta.attach(requires={"mobile"}, cap="mobile")
class RosnavRlBringup(Bringup):
    kind = "rosnav_rl"
    task_handlers: ClassVar[dict] = {TaskKind.GOTO_POSE: _load_goto_pose_rosnav_rl}

    @property
    def _cfg(self) -> dict:
        return self.caps.mobile.sub("rosnav_rl")

    @property
    def goal_topic(self) -> str:
        return self.namespace("goal_pose")

    @property
    def cmd_vel_topic(self) -> str:
        return self.namespace("cmd_vel")

    @property
    def inference_node_name(self) -> str:
        return "rosnav_rl_inference"

    def _launch_actions(
        self,
        *,
        use_sim_time: bool = True,
        frame: str = "",
        arena_env_node: str = "",
        train_mode: bool = False,
        agent: str = "",
        control_rate: float | None = None,
        min_lookahead_dist: float | None = None,
        max_lookahead_dist: float | None = None,
        lookahead_time: float | None = None,
        **_: object,
    ) -> list[Action]:
        cfg = self._cfg
        agent_name = agent or str(cfg.get("agent", ""))
        if not agent_name and not train_mode:
            raise ValueError(f"rosnav_rl bringup for '{self.robot.name}' missing required 'agent': set caps.yaml mobile 'rosnav_rl.agent' or pass robot.mobile.agent:=<name>")
        rate = float(cfg.get("control_rate", 0.0)) if control_rate is None else float(control_rate)
        lo = float(cfg.get("min_lookahead_dist", 0.5)) if min_lookahead_dist is None else float(min_lookahead_dist)
        hi = float(cfg.get("max_lookahead_dist", 2.5)) if max_lookahead_dist is None else float(max_lookahead_dist)
        lt = float(cfg.get("lookahead_time", 1.5)) if lookahead_time is None else float(lookahead_time)

        launch_file = PathJoinSubstitution(
            [
                FindPackageShare("arena_robots"),
                "launch",
                "adapters",
                "mobile",
                "rosnav_rl.launch.py",
            ]
        )
        return [
            *agent_range_actions(self.robot.name, self.caps.mobile, agent_name),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(launch_file),
                launch_arguments={
                    "agent": agent_name,
                    "base_frame": self.caps.body.base_frame,
                    "control_rate": str(rate),
                    "frame": frame,
                    "lookahead_time": str(lt),
                    "max_lookahead_dist": str(hi),
                    "min_lookahead_dist": str(lo),
                    "namespace": self.namespace,
                    "node_name": self.inference_node_name,
                    "robot": self.robot.name,
                    "arena_env_node": arena_env_node,
                    "train_mode": str(train_mode).lower(),
                    "use_sim_time": str(use_sim_time).lower(),
                }.items(),
            ),
        ]
