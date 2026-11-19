from __future__ import annotations

import socket
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from arena_robots import ARENA_ROBOTS_DIR, drift
from arena_robots.caps import simulators_honoring
from arena_robots.Robot import RobotIdentifier

ROBOTS_DIR = Path(ARENA_ROBOTS_DIR) / "robots"
GOLDEN_DIR = Path(__file__).parent / "golden"
MODELS = sorted(p.parent.parent.name for p in ROBOTS_DIR.glob("*/caps/mobile.yaml"))
HALF = {"mobile.limits": {"scale": 0.5}}


def _golden(robot: str) -> str:
    return (GOLDEN_DIR / f"{robot}_default.urdf").read_text()


def _gait(path: str) -> dict[str, float]:
    return yaml.safe_load(Path(path).read_text())["/**"]["ros__parameters"]["gait"]


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


@pytest.mark.parametrize("robot", sorted(p.name.removesuffix("_default.urdf") for p in GOLDEN_DIR.glob("*_default.urdf")))
def test_control_wheel_geometry_equals_the_urdf(robot: str):
    view = RobotIdentifier(robot).resolve_sync()
    assert drift.wheel_drift(view, _golden(robot), view.caps.body.base_frame) == []


def test_a_wheel_radius_that_differs_from_the_urdf_is_listed():
    view = RobotIdentifier("rbvogui").resolve_sync()
    urdf = _golden("rbvogui").replace('radius="0.1165"', 'radius="0.125"')
    found = drift.wheel_drift(view, urdf, view.caps.body.base_frame)
    assert any("wheel_radius 0.1165 differs from the URDF 0.125" in message for message in found)


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
    @pytest.mark.parametrize("model", ("mpo700", "rbvogui", "rbvogui_plus", "rskomnidirectional", "a1", "go1"))
    def test_swerve_omni_and_champ_robot_limits_reach_every_simulator(self, model: str):
        caps = RobotIdentifier(model).resolve_sync().effective_caps({})
        assert {"flatsim", "gazebo", "mujoco", "isaac"} <= simulators_honoring("mobile.limits.velocity.linear", caps)

    def test_a_robot_without_limit_keys_reaches_flatsim_only(self, tmp_path: Path):
        robot = tmp_path / "bare"
        robot.mkdir()
        (robot / "control.yaml").write_text("base_controller:\n  ros__parameters:\n    wheel_radius: 0.1\n")
        assert not drift.declares_limit_keys(SimpleNamespace(path=robot))  # pyright: ignore[reportArgumentType]

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
