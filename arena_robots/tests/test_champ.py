from __future__ import annotations

import pytest
import yaml
from arena_robots.champ import leg_maps, served_champ_files
from arena_robots.Robot import RobotIdentifier

HALF = {"mobile.limits": {"scale": 0.5}}

PREFIXES = {"left_front": "FL", "right_front": "FR", "left_hind": "RL", "right_hind": "RR"}


def _leg(prefix: str, x: float, y: float) -> str:
    return f"""
  <link name="{prefix}_hip"/><link name="{prefix}_shoulder"/><link name="{prefix}_thigh"/><link name="{prefix}_calf"/><link name="{prefix}_foot"/>
  <joint name="{prefix}_hip_joint" type="revolute"><origin xyz="{x} {y} 0"/><parent link="body"/><child link="{prefix}_hip"/></joint>
  <joint name="{prefix}_hip_fixed" type="fixed"><origin xyz="0 {y / 4} 0"/><parent link="{prefix}_hip"/><child link="{prefix}_shoulder"/></joint>
  <joint name="{prefix}_thigh_joint" type="revolute"><parent link="{prefix}_shoulder"/><child link="{prefix}_thigh"/></joint>
  <joint name="{prefix}_calf_joint" type="revolute"><parent link="{prefix}_thigh"/><child link="{prefix}_calf"/></joint>
  <joint name="{prefix}_foot_fixed" type="fixed"><parent link="{prefix}_calf"/><child link="{prefix}_foot"/></joint>"""


def _urdf(legs: dict[str, tuple[float, float]]) -> str:
    body = "".join(_leg(prefix, x, y) for prefix, (x, y) in legs.items())
    return f"""<robot name="q">
  <link name="base"/><link name="body"/><link name="camera"/>
  <joint name="floating_base" type="fixed"><parent link="base"/><child link="body"/></joint>
  <joint name="camera_joint" type="fixed"><origin xyz="0.3 0 0.1"/><parent link="body"/><child link="camera"/></joint>
  <ros2_control name="c" type="system"><joint name="ros2_control_only"/></ros2_control>{body}
</robot>"""


QUADRUPED = {"A": (0.2, -0.05), "B": (-0.2, 0.05), "C": (-0.2, -0.05), "D": (0.2, 0.05)}


def test_leg_maps_classify_the_legs_by_hip_origin_and_list_each_chain():
    joints, links = leg_maps(_urdf(QUADRUPED), "base")
    assert list(joints) == ["left_front", "right_front", "left_hind", "right_hind"]
    assert joints["left_front"] == ["D_hip_joint", "D_thigh_joint", "D_calf_joint", "D_foot_fixed"]
    assert joints["right_front"][0] == "A_hip_joint"
    assert joints["left_hind"][0] == "B_hip_joint"
    assert joints["right_hind"][0] == "C_hip_joint"
    assert list(links) == ["base", "left_front", "right_front", "left_hind", "right_hind"]
    assert links["base"] == "base"
    assert links["left_front"] == ["D_hip", "D_thigh", "D_calf", "D_foot"]


def test_leg_maps_reject_two_legs_in_one_corner():
    with pytest.raises(ValueError, match="two legs classify as left_front"):
        leg_maps(_urdf({**QUADRUPED, "E": (0.3, 0.06)}), "base")


def test_leg_maps_reject_a_missing_leg():
    with pytest.raises(ValueError, match="right_hind"):
        leg_maps(_urdf({k: v for k, v in QUADRUPED.items() if k != "C"}), "base")


def _parameters(path) -> dict:
    return yaml.safe_load(path.read_text())["/**"]["ros__parameters"]


@pytest.fixture
def champ_robot(request, require_installed):
    pytest.importorskip("xacro")
    require_installed(request.param)
    return RobotIdentifier(request.param).resolve_sync()


@pytest.mark.parametrize("champ_robot", ["a1", "go1"], indirect=True)
class TestServedChampFiles:
    def test_the_maps_name_the_four_legs_of_the_rendered_urdf(self, champ_robot):
        files = served_champ_files(champ_robot.effective_caps({}))
        joints = _parameters(files["joints_map"])["joints_map"]
        links = _parameters(files["links_map"])["links_map"]
        assert joints == {name: [f"{p}_hip_joint", f"{p}_thigh_joint", f"{p}_calf_joint", f"{p}_foot_fixed"] for name, p in PREFIXES.items()}
        assert links == {"base": "base", **{name: [f"{p}_hip", f"{p}_thigh", f"{p}_calf", f"{p}_foot"] for name, p in PREFIXES.items()}}

    def test_the_gait_speeds_are_the_mobile_limits(self, champ_robot):
        gait = _parameters(served_champ_files(champ_robot.effective_caps({}))["gait_config"])["gait"]
        assert (gait["max_linear_velocity_x"], gait["max_linear_velocity_y"], gait["max_angular_velocity_z"]) == pytest.approx((0.5, 0.25, 1.0))

    def test_half_limits_halve_the_gait_speeds(self, champ_robot):
        gait = _parameters(served_champ_files(champ_robot.effective_caps({}, caps=HALF))["gait_config"])["gait"]
        assert (gait["max_linear_velocity_x"], gait["max_linear_velocity_y"], gait["max_angular_velocity_z"]) == pytest.approx((0.25, 0.125, 0.5))

    def test_the_gait_keeps_the_authored_keys(self, champ_robot):
        authored = _parameters(champ_robot.path / "champ" / "gait.yaml")["gait"]
        assert not any(key.startswith("max_") for key in authored)
        served = _parameters(served_champ_files(champ_robot.effective_caps({}))["gait_config"])["gait"]
        assert {k: v for k, v in served.items() if not k.startswith("max_")} == authored

    def test_the_files_are_named_by_content(self, champ_robot):
        caps = champ_robot.effective_caps({})
        assert served_champ_files(caps) == served_champ_files(caps)
        assert served_champ_files(champ_robot.effective_caps({}, caps=HALF))["gait_config"] != served_champ_files(caps)["gait_config"]
        assert served_champ_files(champ_robot.effective_caps({}, caps=HALF))["joints_map"] == served_champ_files(caps)["joints_map"]
