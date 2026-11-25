from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from arena_robots import ARENA_ROBOTS_DIR
from arena_robots.caps import RobotCaps, simulator_names, simulators_honoring
from arena_robots.Robot import RobotIdentifier

ROBOTS_DIR = Path(ARENA_ROBOTS_DIR) / "robots"
MODELS = sorted(p.parent.name for p in ROBOTS_DIR.glob("*/caps.yaml"))
FIXED = ("ridgeback", "dingo", "rbsummit", "rbtheron")
MOBILE = {
    "radius": 0.3,
    "velocity_limits": {"linear": [-1.0, 1.0], "angular": [-2.0, 2.0]},
    "acceleration_limits": {"linear": 2.0, "angular": 3.0},
    "planar": {"velocity": {"linear": [-1.0, 1.0], "angular": [-2.0, 2.0]}, "acceleration": {"linear": 5.0, "angular": 5.0}},
}


def _caps(tmp_path: Path, overrides: dict[str, object] | None = None, **mobile: object) -> RobotCaps:
    robot_dir = tmp_path / "robot"
    robot_dir.mkdir(parents=True)
    (robot_dir / "caps.yaml").write_text(yaml.safe_dump({"mobile": {**MOBILE, **mobile}}))
    return RobotCaps(robot_dir, overrides=overrides or {})


@pytest.mark.parametrize("model", FIXED)
def test_fixed_robots_plan_within_the_drive_limits_their_controllers_receive(model: str):
    mobile = RobotIdentifier(model).resolve_sync().caps.mobile
    limits, drive = mobile.velocity_limits, mobile.limits
    assert limits is not None
    assert drive is not None
    assert limits.linear.max <= drive.velocity["linear"].max + 1e-9
    assert limits.angular.max <= drive.velocity["angular"].max + 1e-9


def test_diff_drive_ridgeback_plans_no_lateral_motion():
    limits = RobotIdentifier("ridgeback").resolve_sync().caps.mobile.velocity_limits
    assert limits is not None
    assert limits.lateral is None


@pytest.mark.parametrize("model", MODELS)
def test_shipped_robot_caps_raise_no_warnings(model: str):
    assert RobotIdentifier(model).resolve_sync().caps.warnings() == []


class TestClamp:
    def test_axis_without_explicit_limits_is_clamped(self, tmp_path: Path):
        caps = _caps(tmp_path, velocity_limits={"linear": [-3.0, 3.0], "angular": [-2.0, 2.0]})
        limits = caps.mobile.velocity_limits
        assert limits is not None
        assert (limits.linear.min, limits.linear.max) == (-1.0, 1.0)
        assert caps.warnings() == ["mobile: velocity_limits.linear [-3.0, 3.0] exceeds limits.velocity.linear [-1.0, 1.0], clamped to [-1.0, 1.0]"]

    def test_override_raising_a_planner_limit_is_clamped_and_warns(self, tmp_path: Path):
        caps = _caps(tmp_path, {"mobile.velocity_limits.angular": [-4.0, 4.0]})
        limits = caps.mobile.velocity_limits
        assert limits is not None
        assert limits.angular.max == pytest.approx(2.0)
        assert len(caps.warnings()) == 1

    def test_dict_form_keeps_its_form(self, tmp_path: Path):
        caps = _caps(tmp_path, velocity_limits={"linear": {"min": -2.0, "max": 2.0}, "angular": [-2.0, 2.0]})
        assert caps.mobile.raw["velocity_limits"]["linear"] == {"min": -1.0, "max": 1.0}


class TestSanity:
    def test_reversed_range_warns(self, tmp_path: Path):
        caps = _caps(tmp_path, {"mobile.limits.velocity.linear": [1.0, -1.0]})
        assert any("limits.velocity.linear has min 1.0 above max -1.0" in w for w in caps.warnings())

    def test_non_positive_acceleration_warns(self, tmp_path: Path):
        caps = _caps(tmp_path, {"mobile.acceleration_limits.linear": 0.0})
        assert "mobile: acceleration_limits.linear is 0.0, not positive" in caps.warnings()

    def test_non_positive_radius_warns(self, tmp_path: Path):
        caps = _caps(tmp_path, {"mobile.radius": -0.1})
        assert "mobile: radius is -0.1, not positive" in caps.warnings()

    def test_plausible_caps_raise_nothing(self, tmp_path: Path):
        assert _caps(tmp_path).warnings() == []


class TestSimulatorReach:
    def test_planner_facing_paths_reach_every_simulator(self):
        assert simulators_honoring("mobile.velocity_limits.linear") == simulator_names()
        assert simulators_honoring("arm.top.stall_time") == simulator_names()

    def test_limits_reach_the_actuated_simulators(self):
        assert simulators_honoring("mobile.limits.velocity.linear") == frozenset({"flatsim", "gazebo", "mujoco", "isaac"})

    def test_laser_override_raises(self, tmp_path: Path):
        caps = _caps(tmp_path, {"mobile.laser.range": 5.0})
        with pytest.raises(ValueError, match="rendered URDF"):
            caps.mobile

    def test_planar_dynamics_override_warns_outside_flatsim(self, tmp_path: Path):
        caps = _caps(tmp_path, {"mobile.planar.lag": 0.2})
        assert caps.warnings("flatsim") == []
        assert len(caps.warnings("mujoco")) == 1

    def test_limits_override_warns_in_the_dummy_simulator(self, tmp_path: Path):
        caps = _caps(tmp_path, {"mobile.limits": {"scale": 0.5}})
        assert caps.warnings("gazebo") == []
        assert len(caps.warnings("dummy")) == 1
