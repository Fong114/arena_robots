from __future__ import annotations

from pathlib import Path

import pytest
from arena_robots.Robot import RobotIdentifier
from arena_robots.Sensor import SensorSpec, fill_frames, plugin_bridge_rows
from arena_robots.urdf_facts import sensor_frames

GOLDEN_DIR = Path(__file__).parent / "golden"
GOLDEN_ROBOTS = sorted(p.name.removesuffix("_default.urdf") for p in GOLDEN_DIR.glob("*_default.urdf"))
POSE_ROW = ("/model/r/pose", "pose", "geometry_msgs/msg/PoseStamped", "gz.msgs.Pose")
ODOMETRY_ROW = ("/model/r/odometry", "odom", "nav_msgs/msg/Odometry", "gz.msgs.Odometry")


def _golden(robot: str) -> str:
    return (GOLDEN_DIR / f"{robot}_default.urdf").read_text()


def _model(*plugins: str) -> str:
    return "<robot>" + "".join(f"<gazebo>{plugin}</gazebo>" for plugin in plugins) + "</robot>"


def _plugin(filename: str, **fields: str) -> str:
    return f'<plugin filename="{filename}" name="p">' + "".join(f"<{key}>{value}</{key}>" for key, value in fields.items()) + "</plugin>"


@pytest.mark.parametrize("robot", GOLDEN_ROBOTS)
def test_model_params_sensors_state_no_frame(robot):
    assert all(spec.frame == "" for spec in RobotIdentifier(robot).resolve_sync().model_params.sensors)


@pytest.mark.parametrize("robot", GOLDEN_ROBOTS)
def test_effective_sensor_frames_are_the_frames_of_their_urdf_sensors(robot):
    frames = sensor_frames(_golden(robot))
    view = RobotIdentifier(robot).resolve_sync()
    assert {s.name: s.frame for s in view.effective_sensors({})} == {s.name: frames[s.sensor] for s in view.effective_sensors({})}


@pytest.mark.parametrize("robot", GOLDEN_ROBOTS)
def test_filled_model_params_frames_are_the_effective_frames(robot):
    view = RobotIdentifier(robot).resolve_sync()
    assert fill_frames(view.model_params.sensors, lambda: _golden(robot)) == view.effective_sensors({})


def test_gz_frame_id_wins_over_the_reference_link_and_loses_its_leading_slash():
    urdf = '<robot><gazebo reference="link_a"><sensor name="s1" type="imu"/><sensor name="s2" type="imu"><gz_frame_id>/other</gz_frame_id></sensor></gazebo></robot>'
    assert sensor_frames(urdf) == {"s1": "link_a", "s2": "other"}


def test_fill_frames_keeps_a_stated_frame():
    specs = [SensorSpec(name="a", type="imu", topic="t", frame="kept", sensor="s1"), SensorSpec(name="b", type="imu", topic="t", sensor="s1")]
    urdf = '<robot><gazebo reference="link_a"><sensor name="s1" type="imu"/></gazebo></robot>'
    assert [s.frame for s in fill_frames(specs, lambda: urdf)] == ["kept", "link_a"]
    assert specs[1].frame == ""


def test_fill_frames_rejects_a_sensor_the_urdf_lacks():
    with pytest.raises(ValueError, match="not in the URDF"):
        fill_frames([SensorSpec(name="a", type="imu", topic="t", sensor="gone")], lambda: "<robot/>")


@pytest.mark.parametrize("robot", GOLDEN_ROBOTS)
def test_every_robot_bridges_its_pose_and_nothing_else_from_plugins(robot):
    assert plugin_bridge_rows(_golden(robot), "/model/r") == [POSE_ROW]


def test_odometry_publisher_and_diff_drive_plugins_bridge_the_default_odometry_topic():
    for filename in ("gz-sim-odometry-publisher-system", "gz-sim-diff-drive-system"):
        assert plugin_bridge_rows(_model(_plugin(filename)), "/model/r") == [ODOMETRY_ROW]


def test_an_odometry_plugin_with_its_own_topic_and_a_pose_vector_publisher_bridge_nothing():
    urdf = _model(_plugin("gz-sim-diff-drive-system", odom_topic="odom"), _plugin("gz-sim-pose-publisher-system", publish_model_pose="true", use_pose_vector_msg="true"))
    assert plugin_bridge_rows(urdf, "/model/r") == []


def test_plugins_on_a_reference_link_are_not_the_models():
    urdf = f'<robot><gazebo reference="l">{_plugin("gz-sim-pose-publisher-system", publish_model_pose="true")}</gazebo></robot>'
    assert plugin_bridge_rows(urdf, "/model/r") == []
