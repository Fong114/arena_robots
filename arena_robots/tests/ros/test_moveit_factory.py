"""Tests for arena_robots.moveit_factory's `instance: str | None` parameter on
`build_moveit_params` and the allocation-derived rendering of rbvogui[arm=ur5e], the
robot `rbvogui_plus` expands to (SRDF composition + joint_limits)."""

from __future__ import annotations

import shutil
import xml.etree.ElementTree as ET
from collections.abc import Callable

import pytest

from arena_robots.assembly import RequestPart
from arena_robots.request import InstanceRequest

_XACRO = shutil.which("xacro")
_RBVOGUI_ARM = InstanceRequest(parts={"arm": [RequestPart(variant="ur5e", mount="arm0")]})
ARM_JOINTS = tuple(f"robot_arm_{name}_joint" for name in ("shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "wrist_3"))


class _SrdfLinksExist:
    def test_every_srdf_link_exists_in_urdf(self, params: dict) -> None:
        urdf_links = {link.get("name") for link in ET.fromstring(params["robot_description"]).iter("link")}
        srdf = ET.fromstring(params["robot_description_semantic"])
        srdf_links: set[str] = set()
        for chain in (c for g in srdf.iter("group") for c in g.findall("chain")):
            srdf_links |= {chain.get("base_link", ""), chain.get("tip_link", "")}
        for dc in srdf.iter("disable_collisions"):
            srdf_links |= {dc.get("link1", ""), dc.get("link2", "")}
        srdf_links.discard("")
        assert not (srdf_links - urdf_links), f"SRDF references links absent from URDF: {sorted(srdf_links - urdf_links)}"


_LINK = '<link name="{}"><collision><geometry><sphere radius="0.1"/></geometry></collision></link>'
_JOINT = '<joint name="{0}_{1}" type="{2}"><parent link="{0}"/><child link="{1}"/></joint>'


class TestDerivedDisableCollisions:
    """base - frame (no collision) - cover - arm are fixed, wheel and finger turn, tip is fixed to finger."""

    @pytest.fixture(scope="class")
    def pairs(self) -> set[frozenset]:
        from arena_robots.moveit_factory import _disable_rigid_and_adjacent

        links = "".join(_LINK.format(n) for n in ("base", "cover", "arm", "wheel", "finger", "tip")) + '<link name="frame"/>'
        edges = [("base", "frame", "fixed"), ("frame", "cover", "fixed"), ("cover", "arm", "fixed"), ("base", "wheel", "continuous"), ("arm", "finger", "revolute"), ("finger", "tip", "fixed")]
        control = '<ros2_control name="c" type="system"><joint name="base_wheel"/></ros2_control>'
        urdf = f"<robot name='r'>{links}{''.join(_JOINT.format(*e) for e in edges)}{control}</robot>"
        srdf = "<robot name='r'><disable_collisions link1='arm' link2='base' reason='User'/></robot>"
        out = [frozenset((d.get("link1", ""), d.get("link2", ""))) for d in ET.fromstring(_disable_rigid_and_adjacent(srdf, urdf)).iter("disable_collisions")]
        assert len(out) == len(set(out))
        return set(out)

    def test_fixed_chain_is_disabled_across_links_without_collision(self, pairs: set[frozenset]) -> None:
        assert {frozenset(p) for p in (("base", "cover"), ("base", "arm"), ("cover", "arm"), ("finger", "tip"))} <= pairs

    def test_joint_neighbors_are_disabled(self, pairs: set[frozenset]) -> None:
        assert {frozenset(("base", "wheel")), frozenset(("arm", "finger"))} <= pairs

    def test_bodies_that_move_against_each_other_stay_checked(self, pairs: set[frozenset]) -> None:
        assert not pairs & {frozenset(p) for p in (("wheel", "finger"), ("base", "finger"), ("base", "tip"), ("wheel", "arm"))}


class TestSelectArm:
    """`_select_arm` is the pure selection rule shared by `build_moveit_params`;
    it returns the `(key, entry)` pair so callers can render per-placement
    SRDF/joint_limits off the mount name."""

    def test_none_selects_sole_instance(self):
        from arena_robots.moveit_factory import _select_arm

        arms = {"arm": "the_arm"}
        assert _select_arm("robot", arms, None) == ("arm", "the_arm")

    def test_none_with_multiple_raises_value_error_listing_instances(self):
        from arena_robots.moveit_factory import _select_arm

        arms = {"arm0": "a0", "arm1": "a1"}
        with pytest.raises(ValueError, match=r"arm0.*arm1|arm1.*arm0"):
            _select_arm("robot", arms, None)

    def test_named_instance_selects_it(self):
        from arena_robots.moveit_factory import _select_arm

        arms = {"arm0": "a0", "arm1": "a1"}
        assert _select_arm("robot", arms, "arm1") == ("arm1", "a1")

    def test_unknown_instance_raises_key_error_listing_instances(self):
        from arena_robots.moveit_factory import _select_arm

        arms = {"arm0": "a0", "arm1": "a1"}
        with pytest.raises(KeyError, match=r"arm0.*arm1|arm1.*arm0"):
            _select_arm("robot", arms, "bogus")


class TestBuildMoveitParamsInstanceAgreement:
    """rbvogui_plus is single-arm (instance "arm0"): instance=None and the
    explicit instance="arm0" must build byte-identical params."""

    def test_none_and_explicit_instance_agree(self, require_installed: Callable[[str], None]):
        from arena_robots.moveit_factory import build_moveit_params

        require_installed("rbvogui")
        require_installed("arm/ur")
        default_params = build_moveit_params("rbvogui", request=_RBVOGUI_ARM)
        explicit_params = build_moveit_params("rbvogui", request=_RBVOGUI_ARM, instance="arm0")
        assert default_params is not None
        assert default_params == explicit_params

    @pytest.mark.requires_installed("rbvogui", "arm/ur")
    def test_unknown_instance_raises_key_error(self):
        from arena_robots.moveit_factory import build_moveit_params

        with pytest.raises(KeyError, match="arm0"):
            build_moveit_params("rbvogui", request=_RBVOGUI_ARM, instance="bogus")


def _disabled_pairs(srdf_text: str) -> frozenset:
    return frozenset(frozenset((dc.get("link1", ""), dc.get("link2", ""))) for dc in ET.fromstring(srdf_text).iter("disable_collisions"))


def _arm_link_names(urdf_text: str) -> frozenset:
    root = ET.fromstring(urdf_text)
    return frozenset(link.get("name", "") for link in root.iter("link") if "arm" in link.get("name", ""))


@pytest.mark.skipif(_XACRO is None, reason="xacro CLI not on PATH; run under the Arena container (bash arena -c pytest)")
class TestBuildMoveitParamsOfRbvoguiPlus(_SrdfLinksExist):
    """Moveit params for rbvogui[arm=ur5e], what `rbvogui_plus` expands to: the arm
    component's SRDF fragment composed with the chassis residual fragment."""

    @pytest.fixture(scope="class")
    def params(self, require_installed: Callable[[str], None]) -> dict:
        from arena_robots.moveit_factory import build_moveit_params

        require_installed("rbvogui")
        require_installed("arm/ur")
        p = build_moveit_params("rbvogui", tf_prefix="env_0/rbvogui/", request=_RBVOGUI_ARM)
        assert p is not None
        return p

    def test_srdf_disables_every_derived_pair(self, params: dict) -> None:
        from arena_robots.moveit_factory import _disable_rigid_and_adjacent

        srdf = params["robot_description_semantic"]
        assert _disabled_pairs(srdf) == _disabled_pairs(_disable_rigid_and_adjacent(srdf, params["robot_description"]))

    def test_srdf_carries_the_arm_group_and_the_chassis_residual(self, params: dict) -> None:
        root = ET.fromstring(params["robot_description_semantic"])
        assert "arm_manipulator" in {g.get("name") for g in root.iter("group")}
        assert any("front_right_base_wheel" in (dc.get("link1", "") + dc.get("link2", "")) for dc in root.iter("disable_collisions"))

    def test_joint_limits_cover_the_arm_joints(self, params: dict) -> None:
        assert set(ARM_JOINTS) <= set(params["robot_description_planning"]["joint_limits"])

    def test_robot_description_arm_link_set(self, params: dict) -> None:
        links = {f"env_0/rbvogui/robot_arm_{name}" for name in ("base_link", "shoulder_link", "upper_arm_link", "forearm_link", "wrist_1_link", "wrist_2_link", "wrist_3_link", "tool0")}
        assert links <= _arm_link_names(params["robot_description"])


@pytest.mark.skipif(_XACRO is None, reason="xacro CLI not on PATH; run under the Arena container (bash arena -c pytest)")
class TestBuildMoveitParamsZeroPrefixChassis(_SrdfLinksExist):
    """jackal declares ``prefix: ""`` (no Robotnik robot_ convention). Composing an arm
    onto it must render the SRDF with the SAME empty prefix as the URDF, so the planning
    group's chain and disable_collisions reference links that actually exist in the URDF.
    Regression for the xacro-CLI empty-``prefix:=`` fallback to the fragment default
    robot_ (``_compose_srdf`` now renders via the xacro python API, which honors "")."""

    @pytest.fixture(scope="class")
    def params(self, require_installed: Callable[[str], None]) -> dict:
        from arena_robots.assembly import RequestPart
        from arena_robots.moveit_factory import build_moveit_params

        require_installed("arm/ur")
        p = build_moveit_params("jackal", tf_prefix="", request=InstanceRequest(parts={"arm": [RequestPart(variant="ur5e", mount="top")]}), instance="top")
        assert p is not None
        return p

    def test_srdf_carries_no_robot_prefix(self, params: dict) -> None:
        root = ET.fromstring(params["robot_description_semantic"])
        for chain in (c for g in root.iter("group") for c in g.findall("chain")):
            assert chain.get("base_link", "").startswith("top_")
            assert not chain.get("base_link", "").startswith("robot_")
            assert not chain.get("tip_link", "").startswith("robot_")


@pytest.mark.skipif(_XACRO is None, reason="xacro CLI not on PATH; run under the Arena container (bash arena -c pytest)")
class TestBuildMoveitParamsWithGripper(_SrdfLinksExist):
    """A gripper chained onto the arm's tip (jackal ``top_tool``) merges its
    SRDF fragment into the composed document: gripper group + end_effector + collision
    disables, all referencing links that exist in the wrapper-rendered URDF."""

    @pytest.fixture(scope="class")
    def params(self, require_installed: Callable[[str], None]) -> dict:
        from arena_robots.assembly import RequestPart
        from arena_robots.moveit_factory import build_moveit_params

        require_installed("arm/ur")
        require_installed("gripper/robotiq_2f_85")
        p = build_moveit_params(
            "jackal",
            tf_prefix="",
            request=InstanceRequest(parts={"arm": [RequestPart(variant="ur5e", mount="top")], "gripper": [RequestPart(variant="robotiq_2f_85", mount="top_tool")]}),
            instance="top",
        )
        assert p is not None
        return p

    def test_gripper_group_and_end_effector_present(self, params: dict) -> None:
        srdf = ET.fromstring(params["robot_description_semantic"])
        assert "top_tool" in {g.get("name") for g in srdf.iter("group")}
        ee = next(iter(srdf.iter("end_effector")), None)
        assert ee is not None
        assert ee.get("parent_link") == "top_tool0"
        assert ee.get("parent_group") == "top_manipulator"

    def test_wrist_adjacency_disabled(self, params: dict) -> None:
        srdf = ET.fromstring(params["robot_description_semantic"])
        pairs = {frozenset({dc.get("link1"), dc.get("link2")}) for dc in srdf.iter("disable_collisions")}
        assert frozenset({"top_wrist_3_link", "top_tool_robotiq_85_base_link"}) in pairs

    def test_tf_prefix_reaches_end_effector_parent_link(self, require_installed: Callable[[str], None]) -> None:
        from arena_robots.assembly import RequestPart
        from arena_robots.moveit_factory import build_moveit_params

        require_installed("arm/ur")
        require_installed("gripper/robotiq_2f_85")
        p = build_moveit_params(
            "jackal",
            tf_prefix="env_0/jackal/",
            request=InstanceRequest(parts={"arm": [RequestPart(variant="ur5e", mount="top")], "gripper": [RequestPart(variant="robotiq_2f_85", mount="top_tool")]}),
            instance="top",
        )
        ee = next(iter(ET.fromstring(p["robot_description_semantic"]).iter("end_effector")))
        assert ee.get("parent_link") == "env_0/jackal/top_tool0"


@pytest.mark.skipif(_XACRO is None, reason="xacro CLI not on PATH; run under the Arena container (bash arena -c pytest)")
class TestBuildMoveitParamsSmallUrComponents(_SrdfLinksExist):
    """ur3e/ur3 resolve through the `arm/ur` family component (catalog `variants:`
    fallback); `${variant}` threads the ur_type into the URDF/config and the per-variant
    joint_limits path. This guards that a family variant with no dedicated dir composes a
    SRDF whose links exist in the variant-parametrized URDF (zero-prefix jackal[top=arm/<v>])."""

    @pytest.fixture(scope="class", params=["ur3e", "ur3"])
    def params(self, request: pytest.FixtureRequest, require_installed: Callable[[str], None]) -> dict:
        from arena_robots.assembly import RequestPart
        from arena_robots.moveit_factory import build_moveit_params

        require_installed("arm/ur")
        p = build_moveit_params("jackal", tf_prefix="", request=InstanceRequest(parts={"arm": [RequestPart(variant=request.param, mount="top")]}), instance="top")
        assert p is not None
        return p


_PILZ = "pilz_industrial_motion_planner"
_CARTESIAN_KEYS = {"max_trans_vel", "max_trans_acc", "max_trans_dec", "max_rot_vel"}


def _assert_linear_planning_available(params: dict) -> None:
    assert {"ompl", _PILZ} <= set(params["planning_pipelines"])
    assert params["default_planning_pipeline"] == "ompl"
    assert set(params["robot_description_planning"]["cartesian_limits"]) == _CARTESIAN_KEYS
    for joint, limits in params["robot_description_planning"]["joint_limits"].items():
        assert not [key for key, value in limits.items() if type(value) is int], joint


def _group_chains(params: dict) -> dict[str, tuple[str, str]]:
    srdf = ET.fromstring(params["robot_description_semantic"])
    return {g.get("name", ""): (c.get("base_link", ""), c.get("tip_link", "")) for g in srdf.iter("group") for c in g.findall("chain")}


def _group_states(params: dict, group: str) -> set[str]:
    srdf = ET.fromstring(params["robot_description_semantic"])
    return {gs.get("name", "") for gs in srdf.iter("group_state") if gs.get("group") == group}


def _controller_joints(params: dict, controller: str) -> list[str]:
    mgr = params["moveit_simple_controller_manager"]
    assert mgr["controller_names"] == [controller]
    assert mgr[controller]["type"] == "FollowJointTrajectory"
    return mgr[controller]["joints"]


@pytest.mark.skipif(_XACRO is None, reason="xacro CLI not on PATH; run under the Arena container (bash arena -c pytest)")
class TestBuildMoveitParamsFr3(_SrdfLinksExist):
    """FR3 on rbkairos has no MoveIt config package: srdf, joint limits, KDL and controller come from Arena-owned files."""

    _JOINTS = [f"robot_arm_fr3_joint{i}" for i in range(1, 8)]

    @pytest.fixture(scope="class")
    def params(self, require_installed: Callable[[str], None]) -> dict:
        from arena_robots.assembly import RequestPart
        from arena_robots.moveit_factory import build_moveit_params

        require_installed("arm/fr3")
        require_installed("gripper/franka_hand")
        p = build_moveit_params(
            "rbkairos",
            tf_prefix="",
            request=InstanceRequest(parts={"arm": [RequestPart(variant="fr3", mount="arm_mount")], "gripper": [RequestPart(variant="franka_hand", mount="gripper_mount")]}),
        )
        assert p is not None
        return p

    def test_planning_group_chain(self, params: dict) -> None:
        assert _group_chains(params)["arm_manipulator"] == ("robot_arm_fr3_link0", "robot_arm_fr3_link8")

    def test_named_states_match_caps(self, params: dict) -> None:
        assert _group_states(params, "arm_manipulator") == {"stow", "ready"}

    def test_joint_limits_cover_arm_joints(self, params: dict) -> None:
        assert set(self._JOINTS) <= set(params["robot_description_planning"]["joint_limits"])

    def test_kinematics_solver_set_on_group(self, params: dict) -> None:
        assert params["robot_description_kinematics"]["arm_manipulator"]["kinematics_solver"] == "kdl_kinematics_plugin/KDLKinematicsPlugin"

    def test_controller_names_arm_controller(self, params: dict) -> None:
        assert _controller_joints(params, "arm_controller") == self._JOINTS

    def test_ompl_is_default_pipeline(self, params: dict) -> None:
        assert params["default_planning_pipeline"] == "ompl"
        assert "planner_configs" in params["ompl"]

    def test_hand_end_effector_on_arm_group(self, params: dict) -> None:
        ee = next(iter(ET.fromstring(params["robot_description_semantic"]).iter("end_effector")))
        assert ee.get("parent_group") == "arm_manipulator"
        assert ee.get("parent_link") == "robot_arm_fr3_link8"


@pytest.mark.skipif(_XACRO is None, reason="xacro CLI not on PATH; run under the Arena container (bash arena -c pytest)")
class TestBuildMoveitParamsJackalFr3:
    """FR3 on jackal's top mount."""

    @pytest.fixture(scope="class")
    def params(self, require_installed: Callable[[str], None]) -> dict:
        from arena_robots.assembly import RequestPart
        from arena_robots.moveit_factory import build_moveit_params

        require_installed("arm/fr3")
        p = build_moveit_params("jackal", tf_prefix="", request=InstanceRequest(parts={"arm": [RequestPart(variant="fr3", mount="top")]}), instance="top")
        assert p is not None
        return p

    def test_ompl_and_pilz_pipelines_with_cartesian_limits(self, params: dict) -> None:
        _assert_linear_planning_available(params)


@pytest.mark.skipif(_XACRO is None, reason="xacro CLI not on PATH; run under the Arena container (bash arena -c pytest)")
class TestBuildMoveitParamsTiago(_SrdfLinksExist):
    """TIAGo's native arm: group `arm` from torso_lift_link to arm_tool_link, limits from the URDF."""

    _JOINTS = [f"arm_{i}_joint" for i in range(1, 8)]

    @pytest.fixture(scope="class")
    def params(self) -> dict:
        from arena_robots.moveit_factory import build_moveit_params

        p = build_moveit_params("tiago")
        assert p is not None
        return p

    def test_planner_checks_motion_at_a_fine_resolution(self, params: dict) -> None:
        """OMPL validates each motion segment densely enough that the final path check finds no missed contact."""
        assert params["ompl"]["arm"]["longest_valid_segment_fraction"] == 0.002

    def test_planning_group_chain(self, params: dict) -> None:
        assert _group_chains(params)["arm"] == ("torso_lift_link", "arm_tool_link")

    def test_named_states_match_caps(self, params: dict) -> None:
        assert _group_states(params, "arm") == {"stow", "ready", "carry"}

    def test_joint_limits_cover_arm_joints(self, params: dict) -> None:
        assert set(self._JOINTS) <= set(params["robot_description_planning"]["joint_limits"])

    def test_joint_limits_match_urdf(self, params: dict) -> None:
        urdf = ET.fromstring(params["robot_description"])
        limits = params["robot_description_planning"]["joint_limits"]
        for joint in (j for j in urdf.findall("joint") if j.get("name") in self._JOINTS):
            lim = joint.find("limit")
            assert lim is not None
            assert limits[joint.get("name")]["max_position"] == pytest.approx(float(lim.get("upper", "nan")))
            assert limits[joint.get("name")]["min_position"] == pytest.approx(float(lim.get("lower", "nan")))

    def test_kinematics_solver_set_on_group(self, params: dict) -> None:
        assert params["robot_description_kinematics"]["arm"]["kinematics_solver"] == "kdl_kinematics_plugin/KDLKinematicsPlugin"

    def test_controller_names_arm_controller(self, params: dict) -> None:
        assert _controller_joints(params, "arm_controller") == self._JOINTS

    def test_ompl_and_pilz_pipelines_with_cartesian_limits(self, params: dict) -> None:
        _assert_linear_planning_available(params)

    def test_tf_prefix_reaches_chain(self) -> None:
        from arena_robots.moveit_factory import build_moveit_params

        p = build_moveit_params("tiago", tf_prefix="env_0/tiago/")
        assert p is not None
        assert _group_chains(p)["arm"] == ("env_0/tiago/torso_lift_link", "env_0/tiago/arm_tool_link")


@pytest.mark.skipif(_XACRO is None, reason="xacro CLI not on PATH; run under the Arena container (bash arena -c pytest)")
class TestBuildMoveitParamsUrKeepsWorking:
    """UR keeps its config package; the factory adds the arm controller mapping and a solver on the planning group."""

    @pytest.fixture(scope="class")
    def params(self, require_installed: Callable[[str], None]) -> dict:
        from arena_robots.moveit_factory import build_moveit_params

        require_installed("rbvogui")
        require_installed("arm/ur")
        p = build_moveit_params("rbvogui", tf_prefix="robot_", request=_RBVOGUI_ARM)
        assert p is not None
        return p

    def test_planning_group_chain(self, params: dict) -> None:
        assert "arm_manipulator" in _group_chains(params)

    def test_joint_limits_cover_arm_joints(self, params: dict) -> None:
        limits = params["robot_description_planning"]["joint_limits"]
        assert set(ARM_JOINTS) <= set(limits)

    def test_kinematics_solver_set_on_group(self, params: dict) -> None:
        assert params["robot_description_kinematics"]["arm_manipulator"]["kinematics_solver"] == "kdl_kinematics_plugin/KDLKinematicsPlugin"

    def test_controller_names_arm_controller(self, params: dict) -> None:
        joints = _controller_joints(params, "arm_controller")
        assert joints[0] == "robot_arm_shoulder_pan_joint"
        assert len(joints) == 6
