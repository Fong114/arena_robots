from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from arena_robots import urdf_facts
from arena_robots.assembly import RequestPart
from arena_robots.caps import RobotCaps, moveit_joint_limits, srdf_document, srdf_group, srdf_group_states, srdf_joint_states
from arena_robots.Robot import RobotIdentifier
from arena_robots.urdf_facts import JointLimit

GOLDEN_DIR = Path(__file__).parent / "golden"
TIAGO_ARM_JOINTS = [f"arm_{i}_joint" for i in range(1, 8)]
SRDF = """<robot xmlns:xacro="http://www.ros.org/wiki/xacro">
  <xacro:arg name="prefix" default="p_"/>
  <xacro:arg name="mount" default="m"/>
  <group name="$(arg mount)_arm">
    <chain base_link="$(arg prefix)base" tip_link="$(arg prefix)tip"/>
  </group>
  <group name="$(arg mount)_hand">
    <joint name="$(arg prefix)finger"/>
  </group>
  <group_state name="stow" group="$(arg mount)_arm">
    <joint name="$(arg prefix)j1" value="0.5"/>
    <joint name="$(arg prefix)j2" value="-1.0"/>
  </group_state>
  <group_state name="open" group="$(arg mount)_hand">
    <joint name="$(arg prefix)finger" value="0.04"/>
  </group_state>
  <group_state name="close" group="$(arg mount)_hand">
    <joint name="$(arg prefix)finger" value="0.0"/>
  </group_state>
</robot>
"""
URDF = """<robot name="r">
  <link name="chassis"/><link name="base"/><link name="l1"/><link name="l2"/><link name="tip"/><link name="finger_link"/>
  <joint name="mount" type="fixed"><parent link="chassis"/><child link="base"/></joint>
  <joint name="j1" type="revolute"><parent link="base"/><child link="l1"/></joint>
  <joint name="j2" type="continuous"><parent link="l1"/><child link="l2"/></joint>
  <joint name="flange" type="fixed"><parent link="l2"/><child link="tip"/></joint>
  <joint name="side" type="revolute"><parent link="base"/><child link="finger_link"/></joint>
</robot>"""


@pytest.fixture
def srdf_package(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    pytest.importorskip("xacro", reason="xacro is not importable, run under the Arena container")
    package = tmp_path / "pkg"
    package.mkdir()
    (package / "arm.srdf.xacro").write_text(SRDF)
    monkeypatch.setattr("ament_index_python.packages.get_package_share_path", lambda name: package)
    return package


def _robot(tmp_path: Path, caps: dict) -> RobotCaps:
    robot_dir = tmp_path / "robot"
    robot_dir.mkdir()
    (robot_dir / "caps.yaml").write_text(yaml.dump(caps))
    return RobotCaps(robot_dir=robot_dir)


def _moveit(**extra: object) -> dict:
    return {"planning_group": "m_arm", "srdf": {"package": "pkg", "path": "arm.srdf.xacro"}, "args": {"prefix": "p_", "mount": "m"}, **extra}


class TestSrdfReaders:
    def test_the_document_is_rendered_with_the_arguments(self, srdf_package: Path) -> None:
        root = srdf_document({"srdf": {"package": "pkg", "path": "arm.srdf.xacro"}}, {"prefix": "q_", "mount": "x"})
        assert root is not None
        assert srdf_group(root, "x_arm") == {"base_link": "q_base", "tip_link": "q_tip"}

    def test_a_block_without_an_srdf_has_no_document(self) -> None:
        assert srdf_document({"planning_group": "arm"}, {}) is None

    def test_the_group_lists_its_chain_links_and_joints(self, srdf_package: Path) -> None:
        root = srdf_document(_moveit(), {"prefix": "p_", "mount": "m"})
        assert root is not None
        assert srdf_group(root, "m_arm") == {"base_link": "p_base", "tip_link": "p_tip"}
        assert srdf_group(root, "m_hand") == {"chain": ["p_finger"]}

    def test_an_unknown_group_raises(self, srdf_package: Path) -> None:
        root = srdf_document(_moveit(), {})
        assert root is not None
        with pytest.raises(ValueError, match="no <group name='nope'>"):
            srdf_group(root, "nope")

    def test_group_states_are_the_named_poses_of_one_group(self, srdf_package: Path) -> None:
        root = srdf_document(_moveit(), {"prefix": "p_", "mount": "m"})
        assert root is not None
        assert srdf_group_states(root, "m_arm") == {"stow": {"p_j1": 0.5, "p_j2": -1.0}}
        assert srdf_group_states(root, "none") == {}

    def test_a_joint_takes_its_value_in_each_named_state(self, srdf_package: Path) -> None:
        root = srdf_document(_moveit(), {"prefix": "p_", "mount": "m"})
        assert root is not None
        assert srdf_joint_states(root, "p_finger", ("open", "close")) == {"open": 0.04, "close": 0.0}
        assert srdf_joint_states(root, "p_j1", ("open", "close")) == {}


class TestChainJoints:
    def test_the_moving_joints_between_the_links_come_base_first(self) -> None:
        assert urdf_facts.chain_joints(URDF, "base", "tip") == ["j1", "j2"]

    def test_a_chain_from_the_chassis_skips_the_fixed_mount(self) -> None:
        assert urdf_facts.chain_joints(URDF, "chassis", "tip") == ["j1", "j2"]

    def test_a_tip_that_is_the_base_has_no_joints(self) -> None:
        assert urdf_facts.chain_joints(URDF, "base", "base") == []

    def test_a_tip_outside_the_base_subtree_raises(self) -> None:
        with pytest.raises(ValueError, match="not below"):
            urdf_facts.chain_joints(URDF, "l1", "finger_link")

    def test_the_tiago_arm_is_its_seven_joints(self) -> None:
        urdf = (GOLDEN_DIR / "tiago_default.urdf").read_text()
        assert urdf_facts.chain_joints(urdf, "torso_lift_link", "arm_tool_link") == TIAGO_ARM_JOINTS


class TestArmDerivation:
    def test_links_and_poses_come_from_the_srdf(self, tmp_path: Path, srdf_package: Path) -> None:
        caps = _robot(tmp_path, {"arm": {"arm": {"controller": "c", "moveit": _moveit()}}})
        arm = caps.arm["arm"]
        assert (arm.base_link, arm.tip_link) == ("p_base", "p_tip")
        assert arm.named_poses == {"stow": {"p_j1": 0.5, "p_j2": -1.0}}
        assert caps.provenance("arm.arm.tip_link") == "derived"
        assert caps.provenance("arm.arm.named_poses.stow.joints.p_j1") == "derived"
        assert caps.provenance("arm.arm.controller") == "file"
        with pytest.raises(ValueError, match="no chain"):
            arm.chain

    def test_what_the_file_authors_wins_over_the_srdf(self, tmp_path: Path, srdf_package: Path) -> None:
        entry = {"controller": "c", "base_link": "mine", "tip_link": "tool", "chain": ["a", "b"], "named_poses": {"stow": {"joints": {"a": 1.0}}}, "moveit": _moveit()}
        caps = _robot(tmp_path, {"arm": {"arm": entry}})
        arm = caps.arm["arm"]
        assert (arm.base_link, arm.tip_link, arm.chain) == ("mine", "tool", ["a", "b"])
        assert arm.named_poses == {"stow": {"a": 1.0}}
        assert caps.provenance("arm.arm.tip_link") == "file"

    def test_authored_poses_are_added_to_the_srdf_poses(self, tmp_path: Path, srdf_package: Path) -> None:
        entry = {"controller": "c", "named_poses": {"wave": {"joints": {"p_j1": 2.0}}}, "moveit": _moveit()}
        arm = _robot(tmp_path, {"arm": {"arm": entry}}).arm["arm"]
        assert arm.named_poses == {"stow": {"p_j1": 0.5, "p_j2": -1.0}, "wave": {"p_j1": 2.0}}

    def test_the_chain_is_the_srdf_joint_list_without_a_urdf(self, tmp_path: Path, srdf_package: Path) -> None:
        entry = {"controller": "c", "moveit": _moveit(planning_group="m_hand")}
        arm = _robot(tmp_path, {"arm": {"arm": entry}}).arm["arm"]
        assert arm.chain == ["p_finger"]

    def test_an_arm_without_an_srdf_still_needs_its_links(self, tmp_path: Path) -> None:
        arm = _robot(tmp_path, {"arm": {"arm": {"controller": "c"}}}).arm["arm"]
        with pytest.raises(ValueError, match="no base_link"):
            arm.base_link

    def test_a_planning_group_the_srdf_lacks_raises(self, tmp_path: Path, srdf_package: Path) -> None:
        entry = {"controller": "c", "moveit": _moveit(planning_group="nope")}
        with pytest.raises(ValueError, match="no <group name='nope'>"):
            _robot(tmp_path, {"arm": {"arm": entry}}).arm


class TestGripperDerivation:
    def test_open_and_close_are_the_srdf_values_of_the_joint(self, tmp_path: Path, srdf_package: Path) -> None:
        entry = {"joint": "p_finger", "controller": "c", "moveit": _moveit()}
        caps = _robot(tmp_path, {"gripper": {"hand": entry}})
        assert caps.gripper["hand"].positions == {"open": 0.04, "close": 0.0}
        assert caps.provenance("gripper.hand.positions.open") == "derived"

    def test_an_authored_position_wins_and_the_other_is_derived(self, tmp_path: Path, srdf_package: Path) -> None:
        entry = {"joint": "p_finger", "controller": "c", "positions": {"open": 0.02}, "moveit": _moveit()}
        caps = _robot(tmp_path, {"gripper": {"hand": entry}})
        assert caps.gripper["hand"].positions == {"open": 0.02, "close": 0.0}
        assert caps.provenance("gripper.hand.positions.open") == "file"
        assert caps.provenance("gripper.hand.positions.close") == "derived"

    def test_a_joint_the_srdf_does_not_set_leaves_the_positions_missing(self, tmp_path: Path, srdf_package: Path) -> None:
        entry = {"joint": "other", "controller": "c", "moveit": _moveit()}
        with pytest.raises(ValueError, match="missing 'positions'"):
            _robot(tmp_path, {"gripper": {"hand": entry}}).gripper["hand"].positions


class TestInstalledRobots:
    @pytest.fixture(autouse=True)
    def _rendering(self) -> None:
        pytest.importorskip("xacro", reason="xacro is not importable, run under the Arena container")

    def test_tiago_takes_its_arm_and_gripper_facts_from_its_srdf(self) -> None:
        caps = RobotIdentifier("tiago").resolve_sync().effective_caps({})
        arm = caps.arm["arm"]
        assert (arm.base_link, arm.tip_link, arm.chain) == ("torso_lift_link", "arm_tool_link", TIAGO_ARM_JOINTS)
        assert set(arm.named_poses) == {"stow", "ready", "carry", "wave_up", "wave_l", "wave_r"}
        assert [arm.named_poses["stow"][joint] for joint in TIAGO_ARM_JOINTS] == [0.50, -1.34, -0.48, 1.94, -1.49, 1.37, 0.0]
        assert [arm.named_poses["ready"][joint] for joint in TIAGO_ARM_JOINTS] == [0.21, -0.2, -2.2, 1.15, -1.57, 0.2, 0.0]
        assert caps.gripper["gripper"].positions == {"open": 0.044, "close": 0.0}
        assert caps.provenance("arm.arm.chain") == "derived"
        assert caps.provenance("arm.arm.named_poses.stow.joints.arm_1_joint") == "derived"
        assert caps.provenance("arm.arm.named_poses.wave_up.joints.arm_1_joint") == "file"

    @pytest.mark.requires_installed("arm/fr3", "gripper/franka_hand")
    def test_the_fr3_and_its_hand_take_their_facts_from_their_srdf(self) -> None:
        parts = {"arm": [RequestPart(variant="fr3", mount="top")], "gripper": [RequestPart(variant="franka_hand", mount="top_tool")]}
        caps = RobotIdentifier("jackal").resolve_sync().effective_caps(parts)
        arm = caps.arm["top"]
        joints = [f"top_fr3_joint{i}" for i in range(1, 8)]
        assert (arm.base_link, arm.tip_link, arm.chain) == ("chassis_link", "top_fr3_link8", joints)
        assert [arm.named_poses["ready"][joint] for joint in joints] == [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]
        assert caps.gripper["top_tool"].positions == {"open": 0.04, "close": 0.0}

    @pytest.mark.requires_installed("arm/ur", "gripper/robotiq_2f_85")
    def test_the_robotiq_gripper_takes_its_positions_from_its_srdf(self) -> None:
        parts = {"arm": [RequestPart(variant="ur5e", mount="top")], "gripper": [RequestPart(variant="robotiq_2f_85", mount="top_tool")]}
        caps = RobotIdentifier("jackal").resolve_sync().effective_caps(parts)
        assert caps.gripper["top_tool"].positions == {"open": 0.0, "close": 0.8}


class TestUrdfLimitFlags:
    def test_a_joint_without_a_position_range_gets_the_flag_off(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        (tmp_path / "limits.yaml").write_text(yaml.dump({"joint_limits": {"wrist": {"has_acceleration_limits": True, "max_acceleration": 5.0}}}))
        monkeypatch.setattr("ament_index_python.packages.get_package_share_path", lambda name: tmp_path)
        raw = {"moveit": {"joint_limits": {"package": "pkg", "path": "limits.yaml"}}}
        joints = moveit_joint_limits(raw, lambda: {"wrist": JointLimit(None, None, 3.0, 28.0)})
        assert joints["wrist"] == {"has_acceleration_limits": True, "max_acceleration": 5.0, "has_velocity_limits": True, "max_velocity": 3.0, "has_position_limits": False, "has_effort_limits": True, "max_effort": 28.0}
