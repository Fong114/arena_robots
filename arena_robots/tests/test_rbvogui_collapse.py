"""``rbvogui_plus`` expands to ``rbvogui[arm=ur5e]``, whose composed URDF carries the UR5e
arm on chassis_link at z 0.235 and one merged ros2_control tag holding the arm joints."""

from __future__ import annotations

import shutil
import subprocess
import xml.etree.ElementTree as ET
from collections.abc import Callable
from pathlib import Path

import pytest
from arena_robots.assembly import build_request, resolve
from arena_robots.catalog import Catalog, render_control_joints, render_wrapper_xacro
from arena_robots.Robot import RobotIdentifier
from arena_robots.served_urdf import _inject_ros2_control_joints
from arena_robots.SetupFile import Config

COMPONENTS_ROOT = Path(__file__).resolve().parent.parent / "components"
ARM_JOINTS = tuple(f"robot_arm_{name}_joint" for name in ("shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "wrist_3"))

_XACRO = shutil.which("xacro")


def _render(path: Path) -> ET.Element:
    rendered = subprocess.run([_XACRO, str(path)], capture_output=True, text=True, check=True).stdout
    return ET.fromstring(rendered)


def _compose(model: str, parts: dict[str, list[str]], directory: Path) -> ET.Element:
    view = RobotIdentifier(model).resolve_sync()
    assert view.assembly is not None
    request, cleared_sockets, cleared_types = build_request(view.assembly, parts)
    resolved = resolve(view.assembly, request, cleared_sockets=cleared_sockets, cleared_types=cleared_types)
    catalog = Catalog(root=COMPONENTS_ROOT)
    wrapper_path = directory / f"{model}.urdf.xacro"
    wrapper_path.write_text(render_wrapper_xacro(view, resolved, catalog=catalog))
    composed = _render(wrapper_path)
    _inject_ros2_control_joints(composed, render_control_joints(resolved, catalog, prefix=view.assembly.prefix))
    return composed


@pytest.mark.skipif(_XACRO is None, reason="xacro CLI not on PATH; run under the Arena container (bash arena -c pytest)")
class TestRbvoguiPlusIsRbvoguiWithArm:
    @pytest.fixture(scope="class")
    def rendered(self, tmp_path_factory: pytest.TempPathFactory, require_installed: Callable[[str], None]) -> tuple[ET.Element, ET.Element]:
        require_installed("rbvogui")
        require_installed("arm/ur")
        config = Config.parse("rbvogui_plus")[0]
        directory = tmp_path_factory.mktemp("rbvogui_plus")
        return _compose(config.robot, config.parts, directory), _compose("rbvogui", {}, directory)

    def test_arm_base_sits_on_the_chassis_deck(self, rendered: tuple[ET.Element, ET.Element]) -> None:
        plus, _ = rendered
        joint = next(joint for joint in plus.findall("joint") if joint.find("child").get("link") == "robot_arm_base_link")
        assert joint.get("type") == "fixed"
        assert joint.find("parent").get("link") == "robot_chassis_link"
        assert tuple(float(value) for value in joint.find("origin").get("xyz", "").split()) == (0.0, 0.0, 0.235)

    def test_one_merged_ros2_control_tag_with_the_arm_joints(self, rendered: tuple[ET.Element, ET.Element]) -> None:
        plus, base = rendered
        tags = plus.findall("ros2_control")
        assert len(tags) == 1
        joints = {joint.get("name", "") for joint in tags[0].iter("joint")}
        base_joints = {joint.get("name", "") for joint in base.findall("ros2_control")[0].iter("joint")}
        assert joints == base_joints | set(ARM_JOINTS)
        assert [plugin.text for plugin in tags[0].iter("plugin")] == ["gz_ros2_control/GazeboSimSystem"]

    def test_arm_adds_no_gz_sensors(self, rendered: tuple[ET.Element, ET.Element]) -> None:
        plus, base = rendered

        def inventory(root: ET.Element) -> set[tuple[str, str]]:
            return {(sensor.get("name", ""), sensor.get("type", "")) for gz in root.iter("gazebo") for sensor in gz.findall("sensor")}

        assert inventory(plus) == inventory(base)
