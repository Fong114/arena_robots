from __future__ import annotations

import socket
from pathlib import Path

import pytest
import yaml
from arena_robots import ARENA_ROBOTS_DIR, drift
from arena_robots.caps import simulators_honoring
from arena_robots.Robot import RobotIdentifier

ROBOTS_DIR = Path(ARENA_ROBOTS_DIR) / "robots"
GOLDEN_DIR = Path(__file__).parent / "golden"
MODELS = sorted(p.parent.parent.name for p in ROBOTS_DIR.glob("*/caps/mobile.yaml"))
GOLDEN = sorted(p.name.removesuffix("_default.urdf") for p in GOLDEN_DIR.glob("*_default.urdf"))
HALF = {"mobile.limits": {"scale": 0.5}}


def _golden(robot: str) -> str:
    return (GOLDEN_DIR / f"{robot}_default.urdf").read_text()


def _gait(path: str) -> dict[str, float]:
    return yaml.safe_load(Path(path).read_text())["/**"]["ros__parameters"]["gait"]


@pytest.mark.parametrize("robot", [name for name in GOLDEN if name != "boxer"])
def test_file_laser_equals_the_rendered_scan_sensor(robot: str):
    assert drift.laser_drift(RobotIdentifier(robot).resolve_sync(), _golden(robot)) == []


def test_boxer_rear_lidar_differs_from_the_single_file_block():
    found = drift.laser_drift(RobotIdentifier("boxer").resolve_sync(), _golden("boxer"))
    assert found
    assert all("differs from the URDF scan sensor" in message for message in found)


@pytest.mark.parametrize("model", MODELS)
def test_every_control_literal_equals_the_mobile_limits(model: str):
    view = RobotIdentifier(model).resolve_sync()
    assert drift.limit_drift(view, view.caps.mobile.limits) == []


def test_a_control_literal_that_differs_from_the_limits_is_listed():
    view = RobotIdentifier("jackal").resolve_sync()
    limits = view.effective_caps({}, caps={"mobile.limits.velocity.linear": [-1.0, 1.0]}).mobile.limits
    found = drift.limit_drift(view, limits)
    assert any("linear.x.max_velocity 2.0 differs from mobile.limits 1.0" in message for message in found)
    assert any("linear.x.min_velocity -2.0 differs from mobile.limits -1.0" in message for message in found)


def test_rbvogui_wheel_radius_differs_from_the_urdf():
    view = RobotIdentifier("rbvogui").resolve_sync()
    found = drift.wheel_drift(view, _golden("rbvogui"), view.caps.body.base_frame)
    assert any("wheel_radius 0.1125 differs from the URDF" in message for message in found)


def test_jackal_wheel_geometry_equals_the_urdf():
    view = RobotIdentifier("jackal").resolve_sync()
    assert drift.wheel_drift(view, _golden("jackal"), view.caps.body.base_frame) == []


@pytest.mark.parametrize("model", ("a1", "go1"))
def test_gait_file_equals_the_mobile_limits(model: str):
    view = RobotIdentifier(model).resolve_sync()
    assert drift.gait_drift(view, view.caps.mobile.limits) == []


def test_a_gait_that_differs_from_the_limits_is_listed():
    view = RobotIdentifier("a1").resolve_sync()
    limits = view.effective_caps({}, caps=HALF).mobile.limits
    assert len(drift.gait_drift(view, limits)) == 3


class TestChampGaitFile:
    def test_none_without_an_override(self):
        assert RobotIdentifier("a1").resolve_sync().effective_caps({}).champ_gait_file("env_0_a1") is None

    def test_none_when_the_override_leaves_the_limits_alone(self):
        assert RobotIdentifier("a1").resolve_sync().effective_caps({}, caps={"mobile.radius": 0.5}).champ_gait_file("env_0_a1") is None

    def test_none_for_a_robot_without_a_gait_file(self):
        assert RobotIdentifier("jackal").resolve_sync().effective_caps({}, caps=HALF).champ_gait_file("env_0_jackal") is None

    @pytest.mark.parametrize("model, expected", [("a1", (0.25, 0.125, 0.5)), ("go1", (0.25, 0.125, 0.5))])
    def test_half_limits_halve_the_gait_speeds(self, model: str, expected: tuple[float, float, float]):
        path = RobotIdentifier(model).resolve_sync().effective_caps({}, caps=HALF).champ_gait_file("env_0_a1")
        assert path is not None
        gait = _gait(path)
        assert (gait["max_linear_velocity_x"], gait["max_linear_velocity_y"], gait["max_angular_velocity_z"]) == pytest.approx(expected)

    def test_the_file_is_named_by_host_and_label_and_keeps_the_other_gait_keys(self):
        view = RobotIdentifier("a1").resolve_sync()
        path = view.effective_caps({}, caps=HALF).champ_gait_file("env_0/a1")
        assert path is not None
        assert Path(path).name == f"champ_gait_{socket.gethostname()}_env_0_a1.yaml"
        authored = _gait(str(view.path / "champ" / "gait.yaml"))
        assert {k: v for k, v in _gait(path).items() if not k.startswith("max_")} == {k: v for k, v in authored.items() if not k.startswith("max_")}


class TestSimulatorReach:
    def test_a_swerve_robot_limits_reach_flatsim_only(self):
        caps = RobotIdentifier("mpo700").resolve_sync().effective_caps({})
        reach = simulators_honoring("mobile.limits.velocity.linear", caps)
        assert reach == frozenset({"flatsim"})
        assert "gazebo" not in reach

    def test_a_diff_drive_robot_limits_reach_the_drive_controllers(self):
        caps = RobotIdentifier("jackal").resolve_sync().effective_caps({})
        assert {"flatsim", "gazebo", "mujoco", "isaac"} <= simulators_honoring("mobile.limits.velocity.linear", caps)

    def test_without_caps_the_limits_reach_every_actuated_simulator(self):
        assert {"flatsim", "gazebo", "mujoco", "isaac"} <= simulators_honoring("mobile.limits.velocity.linear")

    def test_a_planner_path_of_a_swerve_robot_still_reaches_every_simulator(self):
        caps = RobotIdentifier("mpo700").resolve_sync().effective_caps({})
        assert "gazebo" in simulators_honoring("mobile.velocity_limits.linear", caps)

    def test_the_override_warning_of_a_swerve_robot_names_flatsim(self):
        caps = RobotIdentifier("mpo700").resolve_sync().effective_caps({}, caps=HALF)
        assert caps.warnings("gazebo") == ["cap override 'mobile.limits' has no effect in sim:=gazebo, honored by flatsim"]
        assert caps.warnings("flatsim") == []
