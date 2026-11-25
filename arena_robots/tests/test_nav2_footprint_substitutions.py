from __future__ import annotations

from pathlib import Path

import launch
import pytest
import yaml
from arena_rclpy_mixins.yaml_replace import YAMLReplacer
from arena_robots.bringup.mobile.nav2 import Nav2Bringup
from arena_robots.caps import stringify_float_matrix
from arena_robots.nav2 import Nav2CollisionDerivedYAML, Nav2KinematicsDerivedYAML, SensorsDerivedYAML
from arena_robots.Robot import RobotIdentifier

NAV2 = Path(__file__).resolve().parents[2] / "arena_robots" / "config" / "nav2"


def _variables(robot: str) -> dict:
    pytest.importorskip("xacro")
    arguments = Nav2Bringup(RobotIdentifier(robot).resolve_sync(), f"/env_0/{robot}").planner_launch_arguments()
    mobile = arguments["mobile_caps"]
    context = launch.LaunchContext()
    variables = yaml.safe_load(Path(mobile).read_text())
    for derived in (Nav2CollisionDerivedYAML(mobile), Nav2KinematicsDerivedYAML(mobile), SensorsDerivedYAML(mobile, arguments["sensors_json"])):
        variables.update(yaml.safe_load(Path(derived.perform(context)).read_text()))
    return {"namespace": "/env_0/robot", **variables}


def _hateb(robot: str) -> dict:
    config = yaml.safe_load((NAV2 / "controllers" / "hateb" / "controller_config.yaml").read_text())
    plugin = YAMLReplacer(_variables(robot)).replace(config)["controller_plugins_dict"]["FollowPath"]
    return plugin


@pytest.mark.parametrize("robot", ["jackal", "tiago", "rskomnidirectional", "a1"])
def test_hateb_models_the_footprint_and_radius_of_the_robot(robot):
    caps = RobotIdentifier(robot).resolve_sync().effective_caps({})
    plugin = _hateb(robot)
    assert plugin["footprint_model"]["radius"] == pytest.approx(caps.mobile.radius)
    assert plugin["footprint_model"]["vertices"] == stringify_float_matrix(caps.mobile.footprint)


def test_hateb_takes_the_wheelbase_of_a_four_wheel_base_from_its_urdf():
    assert _hateb("jackal")["robot"]["wheelbase"] == pytest.approx(0.262)


def test_hateb_keeps_its_default_wheelbase_for_a_base_without_a_wheel_pair_along_x():
    assert _hateb("tiago")["robot"]["wheelbase"] == pytest.approx(1.0)


def test_slam_maps_out_to_the_range_of_the_primary_lidar():
    laser = RobotIdentifier("tiago").resolve_sync().effective_caps({}).mobile.laser
    assert laser is not None
    slam = YAMLReplacer({"namespace": "ns", "scan_topic": "scan", **_variables("tiago")}).replace(yaml.safe_load((NAV2 / "slam.yaml").read_text()))
    assert slam["slam_toolbox"]["ros__parameters"]["max_laser_range"] == pytest.approx(laser.range)
    assert laser.range != pytest.approx(12.0)
