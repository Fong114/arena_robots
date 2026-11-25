from __future__ import annotations

import xml.etree.ElementTree as ET

import pytest
from arena_robots import urdf_facts
from arena_robots.caps import RobotCaps, simulators_honoring
from arena_robots.override_paths import override_paths
from arena_robots.Robot import RobotIdentifier

_LEG_JOINTS = [f"{leg}_{part}_joint" for leg in ("FL", "FR", "RL", "RR") for part in ("hip", "thigh", "calf")]

_URDF = """<robot name="r">
  <link name="base"><inertial><mass value="5.0"/></inertial></link>
  <link name="arm"><inertial><mass value="2.0"/></inertial></link>
  <link name="slide"><inertial><mass value="10.0"/></inertial></link>
  <link name="wheel"/>
  <joint name="shoulder" type="revolute"><parent link="base"/><child link="arm"/><limit effort="40" lower="-1" upper="1" velocity="1"/></joint>
  <joint name="lift" type="prismatic"><parent link="base"/><child link="slide"/><limit effort="100" lower="0" upper="0.3" velocity="0.1"/></joint>
  <joint name="spin" type="continuous"><parent link="base"/><child link="wheel"/></joint>
  <ros2_control name="r" type="system">
    <hardware><plugin>gz_ros2_control/GazeboSimSystem</plugin></hardware>
    <joint name="shoulder"><command_interface name="position"/><state_interface name="position"/></joint>
    <joint name="lift"><command_interface name="position"/><state_interface name="position"/></joint>
    <joint name="spin"><command_interface name="velocity"/><state_interface name="velocity"/></joint>
  </ros2_control>
</robot>"""


@pytest.fixture
def xacro() -> None:
    pytest.importorskip("xacro", reason="xacro is not importable, run under the Arena container")


def _go1(caps: dict[str, object] | None = None) -> RobotCaps:
    return RobotIdentifier("go1").resolve_sync().effective_caps({}, caps=caps)


def test_position_drives_size_revolute_gains_from_the_effort_limit():
    shoulder = urdf_facts.position_drives(_URDF)["shoulder"]
    assert shoulder == {"kp": 4000.0, "kd": 200.0, "effort": 40.0, "command": "position"}


def test_position_drives_size_prismatic_gains_for_the_carried_mass():
    lift = urdf_facts.position_drives(_URDF)["lift"]
    stiffness = 10.0 * 9.81 / 0.002
    assert lift["kp"] == pytest.approx(stiffness)
    assert lift["kd"] == pytest.approx(2.0 * (stiffness * 10.0) ** 0.5)
    assert lift["effort"] == 100.0


def test_position_drives_skip_velocity_commanded_joints():
    assert set(urdf_facts.position_drives(_URDF)) == {"shoulder", "lift"}


def test_go1_drives_cover_every_leg_joint_at_the_derived_gains(xacro):
    drives = _go1().drives
    assert drives is not None
    assert sorted(drives.raw) == sorted(_LEG_JOINTS)
    calf = drives.joint("FL_calf_joint")
    assert (calf.kp, calf.kd, calf.effort, calf.command) == (pytest.approx(3555.0), pytest.approx(177.75), pytest.approx(35.55), "position")


def test_go1_drive_overrides_reach_every_leg_joint(xacro):
    caps = _go1({"drives.*.command": "effort", "drives.*.kp": 20, "drives.FL_calf_joint.kd": {"scale": 2.0}})
    drives = caps.drives
    assert drives is not None
    assert drives.commanded_by("effort") == list(drives.raw)
    assert drives.joint("FR_hip_joint").kp == 20.0
    assert drives.joint("FL_calf_joint").kd == pytest.approx(355.5)
    assert caps.provenance("drives.FR_hip_joint.kp") == "override"
    assert _go1().provenance("drives.FR_hip_joint.kp") == "urdf"


def test_drive_overrides_reject_an_unknown_command_and_joint(xacro):
    with pytest.raises(ValueError, match="command"):
        _go1({"drives.FL_hip_joint.command": "torque"}).drives
    with pytest.raises(ValueError, match="unknown field"):
        _go1({"drives.FL_toe_joint.kp": 10}).drives


def test_drive_gains_reach_gazebo_only_in_effort_mode(xacro):
    assert simulators_honoring("drives.*.kp", _go1()) == frozenset({"mujoco", "isaac"})
    assert simulators_honoring("drives.*.kp", _go1({"drives.*.command": "effort"})) == frozenset({"gazebo", "mujoco", "isaac"})
    assert simulators_honoring("drives.*.command", _go1()) == frozenset({"gazebo", "mujoco", "isaac"})


def test_drives_are_listed_as_override_paths(xacro):
    paths = {path.path: path for path in override_paths(_go1())}
    assert paths["drives.FL_hip_joint.kp"].provenance == "urdf"
    assert "flatsim" not in paths["drives.FL_hip_joint.command"].simulators


def test_served_urdf_carries_the_gains_and_swaps_effort_commanded_joints(xacro):
    caps = _go1({"drives.FL_calf_joint.command": "effort", "drives.FL_calf_joint.kp": 30})
    drives = caps.drives
    assert drives is not None
    root = ET.fromstring(caps.served_urdf(drives=drives.all()).path.read_text())
    joints = {joint.get("name"): joint for control in root.iter("ros2_control") for joint in control.findall("joint")}
    calf, thigh = joints["FL_calf_joint"], joints["FL_thigh_joint"]
    assert {param.get("name"): float(param.text or "") for param in calf.findall("param")} == {"kp": 30.0, "kd": pytest.approx(177.75), "effort": pytest.approx(35.55)}
    assert [command.get("name") for command in calf.findall("command_interface")] == ["effort"]
    assert [command.get("name") for command in thigh.findall("command_interface")] == ["position"]


def test_effort_commanded_legs_under_a_position_controller_fail_validation(xacro):
    with pytest.raises(ValueError, match="does not command effort"):
        _go1({"drives.*.command": "effort"}).validate_overrides()
    _go1({"drives.*.kp": 20}).validate_overrides()
