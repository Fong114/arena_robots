from __future__ import annotations

from pathlib import Path

import launch
import pytest
import yaml
from arena_rclpy_mixins.yaml_replace import YAMLReplacer
from arena_robots.nav2 import Nav2KinematicsDerivedYAML

PKG = Path(__file__).resolve().parents[2] / "arena_robots"
ROBOTS_DIR = PKG / "robots"
NAV2_YAML = PKG / "config" / "nav2" / "nav2.yaml"


def _smoother(robot: str, tmp_path: Path) -> dict:
    mobile = tmp_path / "mobile.yaml"
    mobile.write_text(yaml.safe_dump(yaml.safe_load((ROBOTS_DIR / robot / "caps.yaml").read_text())["mobile"]))
    derived = Nav2KinematicsDerivedYAML(str(mobile)).perform(launch.LaunchContext())
    variables = yaml.safe_load(Path(derived).read_text())
    block = yaml.safe_load(NAV2_YAML.read_text())["velocity_smoother"]["ros__parameters"]
    return YAMLReplacer(variables).replace(block)


def _all_floats(block: dict) -> None:
    for key in ("max_velocity", "min_velocity", "max_accel", "max_decel"):
        assert len(block[key]) == 3, key
        assert all(type(v) is float for v in block[key]), (key, block[key])


@pytest.mark.parametrize(
    ("robot", "max_velocity", "min_velocity", "max_accel", "max_decel"),
    [
        ("jackal", [2.0, 0.0, 4.0], [-2.0, 0.0, -4.0], [2.5, 0.0, 3.2], [-2.5, 0.0, -3.2]),
        ("mpo700", [0.8, 0.8, 1.0], [-0.8, -0.8, -1.0], [0.25, 0.25, 0.8], [-0.25, -0.25, -0.8]),
        ("husky", [1.0, 0.0, 2.0], [-1.0, 0.0, -2.0], [2.5, 0.0, 3.2], [-2.5, 0.0, -3.2]),
        ("rskomnidirectional", [0.5, 0.0, 1.5], [-0.5, 0.0, -1.5], [2.5, 0.0, 3.2], [-2.5, 0.0, -3.2]),
    ],
)
def test_velocity_smoother_follows_caps(robot, max_velocity, min_velocity, max_accel, max_decel, tmp_path):
    block = _smoother(robot, tmp_path)
    _all_floats(block)
    assert block["max_velocity"] == max_velocity
    assert block["min_velocity"] == min_velocity
    assert block["max_accel"] == max_accel
    assert block["max_decel"] == max_decel


def _smoother_of(mobile: dict, tmp_path: Path) -> dict:
    path = tmp_path / "mobile.yaml"
    path.write_text(yaml.safe_dump(mobile))
    variables = yaml.safe_load(Path(Nav2KinematicsDerivedYAML(str(path)).perform(launch.LaunchContext())).read_text())
    return YAMLReplacer(variables).replace(yaml.safe_load(NAV2_YAML.read_text())["velocity_smoother"]["ros__parameters"])


def test_a_robot_with_lateral_limits_gets_them_on_the_lateral_axis(tmp_path):
    block = _smoother_of(
        {"velocity_limits": {"linear": [-1.0, 1.0], "angular": [-2.0, 2.0], "lateral": [-0.4, 0.6]}, "acceleration_limits": {"linear": 2.0, "angular": 3.0, "lateral": 1.5}},
        tmp_path,
    )
    _all_floats(block)
    assert block["max_velocity"] == [1.0, 0.6, 2.0]
    assert block["min_velocity"] == [-1.0, -0.4, -2.0]
    assert block["max_accel"] == [2.0, 1.5, 3.0]
    assert block["max_decel"] == [-2.0, -1.5, -3.0]


def test_lateral_acceleration_falls_back_to_the_linear_one_when_only_the_velocity_is_lateral(tmp_path):
    block = _smoother_of(
        {"velocity_limits": {"linear": [-1.0, 1.0], "angular": [-2.0, 2.0], "lateral": [-0.4, 0.4]}, "acceleration_limits": {"linear": 2.0, "angular": 3.0}},
        tmp_path,
    )
    assert block["max_accel"] == [2.0, 2.0, 3.0]
    assert block["max_decel"] == [-2.0, -2.0, -3.0]


@pytest.mark.parametrize("robot", ["jackal", "husky", "tiago", "rbsummit"])
def test_a_robot_without_lateral_limits_gets_zero_on_every_lateral_entry(robot, tmp_path):
    block = _smoother(robot, tmp_path)
    for key in ("max_velocity", "min_velocity", "max_accel", "max_decel"):
        assert block[key][1] == 0.0, key
