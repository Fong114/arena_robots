from __future__ import annotations

import hashlib
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest
from arena_robots.assembly import RequestPart
from arena_robots.catalog import render_wrapper_xacro
from arena_robots.Robot import RobotIdentifier, RobotView
from arena_robots.served_urdf import (
    _ensure_effort_state,
    _lock_passive_joints,
    _patch_sensor_topics,
    _strip_sensors,
    _strip_unregistered_ros2_control,
    canonical,
    serve,
)

URDF = """
<robot name="x">
  <ros2_control name="chassis" type="system">
    <hardware><plugin>gz_ros2_control/GazeboSimSystem</plugin></hardware>
  </ros2_control>
  <ros2_control name="left_wheel_controller" type="system">
    <hardware><plugin>gazebo_ros2_control/GazeboSystem</plugin></hardware>
  </ros2_control>
  <ros2_control name="arm" type="system">
    <hardware><plugin>ur_robot_driver/URPositionHardwareInterface</plugin></hardware>
  </ros2_control>
  <ros2_control name="broken" type="system">
    <hardware/>
  </ros2_control>
  <gazebo>
    <plugin filename="libgazebo_ros2_control.so" name="gazebo_ros2_control">
      <parameters>/opt/ros/jazzy/share/irobot_create_control/config/control.yaml</parameters>
    </plugin>
  </gazebo>
  <gazebo reference="imu_joint">
    <preserveFixedJoint>true</preserveFixedJoint>
  </gazebo>
</robot>
"""

REGISTERED = frozenset({"ur_robot_driver/URPositionHardwareInterface"})


def test_strips_unregistered_keeps_registered_and_sentinel():
    root = ET.fromstring(URDF)
    _strip_unregistered_ros2_control(root, REGISTERED)
    names = [el.attrib["name"] for el in root.findall("ros2_control")]
    assert names == ["chassis", "arm"]


def test_classic_gazebo_host_plugin_removed_with_empty_wrapper():
    root = ET.fromstring(URDF)
    _strip_unregistered_ros2_control(root, REGISTERED)
    plugins = [p for g in root.findall("gazebo") for p in g.findall("plugin")]
    assert not plugins
    assert root.find("gazebo/preserveFixedJoint") is not None


def test_empty_registered_set_is_noop():
    root = ET.fromstring(URDF)
    _strip_unregistered_ros2_control(root, frozenset())
    assert len(root.findall("ros2_control")) == 4
    assert root.find("gazebo/plugin") is not None


JOINTS_URDF = """
<robot name="x">
  <joint name="wheel_drop_left_joint" type="prismatic">
    <parent link="base_link"/>
    <child link="wheel_drop_left"/>
    <axis xyz="0 1 0"/>
    <limit effort="0" lower="0" upper="0.03" velocity="0"/>
    <dynamics damping="50" friction="0.1"/>
  </joint>
  <joint name="left_wheel_joint" type="continuous">
    <parent link="base_link"/>
    <child link="left_wheel"/>
    <axis xyz="0 1 0"/>
  </joint>
  <joint name="arm_joint" type="revolute">
    <parent link="base_link"/>
    <child link="arm"/>
    <axis xyz="0 0 1"/>
    <limit effort="10" lower="-1" upper="1" velocity="1"/>
  </joint>
</robot>
"""


def test_zero_effort_joint_locked_others_untouched():
    root = ET.fromstring(JOINTS_URDF)
    _lock_passive_joints(root)
    drop = root.find("joint[@name='wheel_drop_left_joint']")
    assert drop.attrib["type"] == "fixed"
    assert drop.find("axis") is None
    assert drop.find("limit") is None
    assert drop.find("dynamics") is None
    assert drop.find("parent") is not None
    assert root.find("joint[@name='left_wheel_joint']").attrib["type"] == "continuous"
    assert root.find("joint[@name='arm_joint']").attrib["type"] == "revolute"


def test_patch_sensor_topics_sets_existing_child_text():
    root = ET.fromstring('<robot><gazebo><sensor name="gpu_lidar" type="gpu_lidar"><topic>old</topic></sensor></gazebo></robot>')
    _patch_sensor_topics(root, [("gpu_lidar", "topic", "/model/env_0/jackal/scan")])
    sensor = root.find(".//sensor[@name='gpu_lidar']")
    assert sensor.find("topic").text == "/model/env_0/jackal/scan"


def test_patch_sensor_topics_creates_nested_child():
    root = ET.fromstring('<robot><gazebo><sensor name="cam"></sensor></gazebo></robot>')
    _patch_sensor_topics(root, [("cam", "camera/camera_info_topic", "/model/env_0/jackal/camera_info")])
    sensor = root.find(".//sensor[@name='cam']")
    assert sensor.find("camera/camera_info_topic").text == "/model/env_0/jackal/camera_info"


def test_patch_sensor_topics_missing_sensor_is_noop():
    root = ET.fromstring('<robot><gazebo><sensor name="other"></sensor></gazebo></robot>')
    _patch_sensor_topics(root, [("missing", "topic", "value")])
    assert root.find(".//sensor[@name='missing']") is None
    assert root.find(".//sensor[@name='other']/topic") is None


SENSORS_URDF = """
<robot name="x">
  <link name="base_link"/>
  <gazebo reference="base_link">
    <sensor name="lidar" type="gpu_lidar"/>
    <sensor name="cam" type="camera"/>
    <sensor name="imu" type="imu"/>
  </gazebo>
</robot>
"""


@pytest.mark.parametrize(
    ("tokens", "remaining"),
    [
        ({"no_lidar"}, {"cam", "imu"}),
        ({"no_camera"}, {"lidar", "imu"}),
        ({"no_lidar", "no_camera"}, {"imu"}),
        ({"unknown"}, {"lidar", "cam", "imu"}),
        (set(), {"lidar", "cam", "imu"}),
    ],
)
def test_strip_sensors_removes_the_types_of_each_optim_token(tokens, remaining):
    root = ET.fromstring(SENSORS_URDF)
    _strip_sensors(root, tokens)
    assert {sensor.get("name") for sensor in root.iter("sensor")} == remaining


def test_ensure_effort_state_adds_it_once():
    root = ET.fromstring('<robot><ros2_control name="c" type="system"><joint name="j"><state_interface name="position"/></joint></ros2_control></robot>')
    _ensure_effort_state(root)
    _ensure_effort_state(root)
    assert [s.get("name") for s in root.find("ros2_control/joint").findall("state_interface")] == ["position", "effort"]


def _tmp_robot(tmp_path: Path, name: str, xacro_text: str) -> RobotView:
    urdf_dir = tmp_path / name / "urdf"
    urdf_dir.mkdir(parents=True)
    (urdf_dir / f"{name}.urdf.xacro").write_text(xacro_text)
    return RobotView(tmp_path / name)


def test_serve_patches_sensor_topics_and_writes_the_file(tmp_path):
    pytest.importorskip("xacro")
    view = _tmp_robot(
        tmp_path,
        "wrapped_model",
        """<?xml version="1.0"?>
<robot name="wrapped_model" xmlns:xacro="http://www.ros.org/wiki/xacro">
  <link name="base_link"/>
  <gazebo>
    <sensor name="lidar" type="gpu_lidar">
      <topic>old_topic</topic>
    </sensor>
  </gazebo>
</robot>
""",
    )
    served = serve(view, {}, sensor_topic_patches=[("lidar", "topic", "/model/env_0/robot/scan")])
    assert "/model/env_0/robot/scan" in served.description
    assert "old_topic" not in served.description
    assert served.path.suffix == ".urdf"
    assert served.path.read_text() == served.description
    assert served.path.stem == hashlib.sha256(served.description.encode()).hexdigest()


def test_serve_keeps_gz_namespace_prefix(tmp_path):
    pytest.importorskip("xacro")
    view = _tmp_robot(
        tmp_path,
        "mecanum_model",
        """<?xml version="1.0"?>
<robot name="mecanum_model" xmlns:xacro="http://www.ros.org/wiki/xacro" xmlns:gz="http://gazebosim.org/schema">
  <link name="base_footprint"/>
  <gazebo reference="wheel_link">
    <collision>
      <surface>
        <friction>
          <ode>
            <fdir1 gz:expressed_in="base_footprint">0.70710678 0.70710678 0</fdir1>
          </ode>
        </friction>
      </surface>
    </collision>
  </gazebo>
</robot>
""",
    )
    served = serve(view, {})
    assert 'gz:expressed_in="base_footprint"' in served.description
    assert "ns0:" not in served.description


def test_serve_passes_xacro_args_json_encoded(tmp_path):
    pytest.importorskip("xacro")
    view = _tmp_robot(
        tmp_path,
        "argy",
        """<?xml version="1.0"?>
<robot name="argy" xmlns:xacro="http://www.ros.org/wiki/xacro">
  <xacro:arg name="namespace" default=""/>
  <xacro:arg name="extra" default=""/>
  <xacro:arg name="skipped" default="kept"/>
  <link name="$(arg namespace)base"/>
  <link name="extra">
    <note text='$(arg extra)'/>
  </link>
  <link name="$(arg skipped)"/>
</robot>
""",
    )
    served = serve(view, {}, xacro_args={"namespace": "env_0/", "extra": {"a": 1}, "skipped": None})
    root = ET.fromstring(served.description)
    assert [link.get("name") for link in root.findall("link")] == ["env_0/base", "extra", "kept"]
    assert '{"a": 1}' in root.find("link/note").get("text")


def test_serve_makes_relative_and_package_mesh_uris_absolute(tmp_path):
    pytest.importorskip("xacro")
    view = _tmp_robot(
        tmp_path,
        "meshy",
        """<?xml version="1.0"?>
<robot name="meshy" xmlns:xacro="http://www.ros.org/wiki/xacro">
  <link name="base"><visual><geometry><mesh filename="meshes/a.stl"/></geometry></visual></link>
</robot>
""",
    )
    served = serve(view, {})
    mesh = ET.fromstring(served.description).find(".//mesh")
    assert mesh.get("filename") == f"file://{tmp_path / 'meshy' / 'urdf' / 'meshes' / 'a.stl'}"


def _old_render(view: RobotView, parts, frames=None) -> ET.Element:
    import xacro
    from arena_robots.assembly import apply_frame_overrides

    resolved = view._resolved(parts)
    if resolved is None:
        return ET.fromstring(xacro.process_file(str(view.path / "urdf" / f"{view.name}.urdf.xacro")).toxml())
    resolved = apply_frame_overrides(resolved, frames or {})
    wrapper = Path(tempfile.mkstemp(suffix=".urdf.xacro")[1])
    try:
        wrapper.write_text(render_wrapper_xacro(view, resolved))
        return ET.fromstring(xacro.process_file(str(wrapper)).toxml())
    finally:
        wrapper.unlink()


def _text(value: ET.Element | str) -> str:
    return value if isinstance(value, str) else ET.tostring(value, encoding="unicode")


def _same_xml(left: ET.Element | str, right: ET.Element | str) -> bool:
    return ET.canonicalize(_text(left), strip_text=True) == ET.canonicalize(_text(right), strip_text=True)


def _structural(root: ET.Element) -> ET.Element:
    _lock_passive_joints(root)
    _ensure_effort_state(root)
    return root


@pytest.mark.parametrize(
    "parts",
    [{}, {"lidar": [RequestPart(variant="rplidar", mount="top")]}],
    ids=["defaults", "placed_lidar"],
)
def test_canonical_is_the_plain_render_plus_the_structural_steps(parts, require_installed):
    pytest.importorskip("xacro")
    require_installed("jackal")
    view = RobotIdentifier("jackal").resolve_sync()
    assert _same_xml(canonical(view, parts), _structural(_old_render(view, parts)))


def test_canonical_places_the_requested_lidar(require_installed):
    pytest.importorskip("xacro")
    require_installed("jackal")
    view = RobotIdentifier("jackal").resolve_sync()
    placed = canonical(view, {"lidar": [RequestPart(variant="rplidar", mount="top")]})
    assert placed != canonical(view, {})


def test_canonical_is_memoized_per_request(require_installed):
    pytest.importorskip("xacro")
    require_installed("jackal")
    view = RobotIdentifier("jackal").resolve_sync()
    assert canonical(view, {}) is canonical(view, {})


def test_canonical_keeps_mesh_uris_and_serve_makes_them_absolute(require_installed):
    pytest.importorskip("xacro")
    require_installed("jackal")
    view = RobotIdentifier("jackal").resolve_sync()
    plain = [mesh.get("filename") for mesh in ET.fromstring(canonical(view, {})).iter("mesh")]
    assert plain
    assert any(not name.startswith("file://") for name in plain)
    served = [mesh.get("filename") for mesh in ET.fromstring(serve(view, {}).description).iter("mesh")]
    assert len(served) == len(plain)
    for name in served:
        assert name.startswith("file://")
        assert Path(name.removeprefix("file://")).exists(), name


@pytest.mark.parametrize("optim", [(), ("no_lidar",), ("no_camera", "no_lidar")])
def test_serve_strips_sensors_per_optim(optim, require_installed):
    pytest.importorskip("xacro")
    require_installed("jackal")
    view = RobotIdentifier("jackal").resolve_sync()
    types = {sensor.get("type") for sensor in ET.fromstring(serve(view, {}, optim=optim).description).iter("sensor")}
    assert not ({"ray", "gpu_lidar"} & types) if "no_lidar" in optim else True
    assert not ({"camera", "depth", "rgbd_camera"} & types) if "no_camera" in optim else True
    if not optim:
        assert {"gpu_lidar", "ray"} & types


def test_canonical_injects_the_assembly_control_joints(require_installed):
    pytest.importorskip("xacro")
    require_installed("jackal")
    view = RobotIdentifier("jackal").resolve_sync()
    root = ET.fromstring(canonical(view, {"arm": [RequestPart(variant="ur5e", mount="top")]}))
    tags = root.findall("ros2_control")
    assert len(tags) == 1
    assert [plugin.text for plugin in tags[0].iter("plugin")] == ["gz_ros2_control/GazeboSimSystem"]
    names = {joint.get("name", "") for joint in tags[0].findall("joint")}
    assert any("shoulder_pan" in name for name in names)
    assert all(any(state.get("name") == "effort" for state in joint.findall("state_interface")) for joint in tags[0].findall("joint"))


def test_robot_caps_serve_without_arguments(require_installed):
    pytest.importorskip("xacro")
    require_installed("jackal")
    view = RobotIdentifier("jackal").resolve_sync()
    caps = view.effective_caps({})
    served = caps.served_urdf()
    assert served.path.is_file()
    assert caps.rendered_urdf() == canonical(view, {})
