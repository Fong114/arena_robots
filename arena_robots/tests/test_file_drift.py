from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from arena_robots import ARENA_ROBOTS_DIR, drift
from arena_robots.caps import simulators_honoring
from arena_robots.Robot import RobotIdentifier, RobotView

ROBOTS_DIR = Path(ARENA_ROBOTS_DIR) / "robots"
MODELS = sorted(p.parent.parent.name for p in ROBOTS_DIR.glob("*/caps/mobile.yaml"))
HALF = {"mobile.limits": {"scale": 0.5}}


AXES = {"linear": "linear.x", "lateral": "linear.y", "angular": "angular.z"}


def _controllers(view: RobotView) -> list[tuple[str, str, dict]]:
    out = []
    for file in sorted(view.path.glob("control*.yaml")):
        for name, section in (yaml.safe_load(file.read_text()) or {}).items():
            if name != "controller_manager" and isinstance(section, dict) and isinstance(section.get("ros__parameters"), dict):
                out.append((file.name, name, section["ros__parameters"]))
    return out


@pytest.mark.parametrize("model", MODELS)
def test_no_control_file_authors_a_limit_value(model: str):
    view = RobotIdentifier(model).resolve_sync()
    suffixes = ("min_velocity", "max_velocity", "min_acceleration", "max_acceleration")
    for file, name, params in _controllers(view):
        assert not [key for key in params if key.endswith(suffixes)], f"{file}: {name}"


@pytest.mark.parametrize("model", MODELS)
def test_every_flagged_controller_axis_has_a_mobile_limit(model: str):
    view = RobotIdentifier(model).resolve_sync()
    limits = view.caps.mobile.limits
    for file, name, params in _controllers(view):
        for axis, prefix in AXES.items():
            for kind in ("velocity", "acceleration"):
                if params.get(f"{prefix}.has_{kind}_limits") is True:
                    assert limits is not None
                    assert axis in getattr(limits, kind), f"{file}: {name} limits {axis} {kind} but mobile.limits has none"


class TestSimulatorReach:
    @pytest.mark.parametrize("model", ("mpo700", "rbvogui", "rskomnidirectional", "a1", "go1"))
    def test_swerve_omni_and_champ_robot_limits_reach_every_simulator(self, model: str):
        caps = RobotIdentifier(model).resolve_sync().effective_caps({})
        assert {"flatsim", "gazebo", "mujoco", "isaac"} <= simulators_honoring("mobile.limits.velocity.linear", caps)

    def test_a_robot_without_limit_keys_reaches_flatsim_only(self, tmp_path: Path):
        robot = tmp_path / "bare"
        robot.mkdir()
        (robot / "control.yaml").write_text("base_controller:\n  ros__parameters:\n    wheel_radius: 0.1\n    linear.x.max_velocity: 1.0\n")
        assert not drift.declares_limit_keys(SimpleNamespace(path=robot))  # pyright: ignore[reportArgumentType]

    def test_a_robot_with_a_limit_flag_reaches_every_simulator(self, tmp_path: Path):
        robot = tmp_path / "flagged"
        robot.mkdir()
        (robot / "control.yaml").write_text("base_controller:\n  ros__parameters:\n    linear.x.has_velocity_limits: true\n")
        assert drift.declares_limit_keys(SimpleNamespace(path=robot))  # pyright: ignore[reportArgumentType]

    def test_a_diff_drive_robot_limits_reach_the_drive_controllers(self):
        caps = RobotIdentifier("jackal").resolve_sync().effective_caps({})
        assert {"flatsim", "gazebo", "mujoco", "isaac"} <= simulators_honoring("mobile.limits.velocity.linear", caps)

    def test_without_caps_the_limits_reach_every_actuated_simulator(self):
        assert {"flatsim", "gazebo", "mujoco", "isaac"} <= simulators_honoring("mobile.limits.velocity.linear")

    def test_a_planner_path_of_a_swerve_robot_still_reaches_every_simulator(self):
        caps = RobotIdentifier("mpo700").resolve_sync().effective_caps({})
        assert "gazebo" in simulators_honoring("mobile.velocity_limits.linear", caps)

    def test_a_swerve_robot_limits_override_raises_no_reach_warning(self):
        caps = RobotIdentifier("mpo700").resolve_sync().effective_caps({}, caps=HALF)
        assert not any("mobile.limits" in warning for warning in caps.warnings("gazebo"))
        assert caps.warnings("flatsim") == []
