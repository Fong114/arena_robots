"""Tests for the components/arm/ur family component. One templated `arm/ur` component
serves the whole UR family via the catalog's `variants:` fallback; `${variant}` threads
the ur_type into attach name, ur_description config dir, and the per-variant MoveIt
joint_limits path. `effective_caps` consumes the `caps:` block via
`RobotCaps`/`_substitute_keys`."""

from __future__ import annotations

import copy
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest
import yaml
from arena_rclpy_mixins.yaml_replace import YAMLReplacer
from arena_robots.assembly import RequestPart
from arena_robots.caps import ArmSpec
from arena_robots.catalog import Catalog, ComponentSpec
from arena_robots.Robot import RobotIdentifier

COMPONENTS_ROOT = Path(__file__).resolve().parent.parent / "components"
FAMILY = COMPONENTS_ROOT / "arm" / "ur"
ARM_JOINTS = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
UR_VARIANTS = ["ur3", "ur3e", "ur5", "ur5e", "ur7e", "ur8long", "ur10", "ur10e", "ur12e", "ur15", "ur16e", "ur18", "ur20", "ur30"]


class TestFamilyComponentSpec:
    """The single `arm/ur/component.yaml` parses and declares the family via `variants:`;
    its variant-specific surfaces (attach name, ur_type, joint_limits path) stay templated
    on `${variant}` in the un-rendered spec."""

    @pytest.fixture(scope="class")
    def spec(self) -> ComponentSpec:
        return ComponentSpec.from_yaml(FAMILY / "component.yaml")

    def test_declares_ur_family_variants(self, spec: ComponentSpec) -> None:
        assert spec.variants == UR_VARIANTS

    def test_attach_and_urtype_templated_on_variant(self, spec: ComponentSpec) -> None:
        assert spec.xacro_include == "$(find ur_description)/urdf/ur_macro.xacro"
        assert spec.xacro_macro == "ur_robot"
        assert spec.attach["name"] == "${variant}"
        assert spec.attach["joint_limits_parameters_file"] == "$(find ur_description)/config/${variant}/joint_limits.yaml"
        assert spec.caps["moveit"]["args"]["ur_type"] == "${variant}"

    def test_ros2_control_and_control_blocks(self, spec: ComponentSpec) -> None:
        names = [j["name"] for j in spec.ros2_control_joints]
        assert names == [f"${{prefix}}${{mount}}_{j}" for j in ARM_JOINTS]
        assert spec.control["controller"] == "${mount}_controller"
        assert spec.control["ros__parameters"]["state_publish_rate"] == 50.0

    def test_exports_tip_frame_for_gripper_chaining(self, spec: ComponentSpec) -> None:
        assert spec.frames == {"tip": "${prefix}${mount}_tool0"}

    def test_caps_moveit_paths_point_at_shared_family_files(self, spec: ComponentSpec) -> None:
        caps = spec.caps
        assert caps["base_link"] == "${prefix}${parent}"
        assert "tip_link" not in caps
        assert "chain" not in caps
        assert caps["moveit"]["planning_group"] == "${mount}_manipulator"
        assert caps["moveit"]["srdf"]["path"] == "components/arm/ur/srdf/ur.srdf.xacro"
        assert caps["moveit"]["joint_limits"]["path"] == "components/arm/ur/joint_limits.yaml"
        assert set(caps["named_poses"]) == {"wave_up", "wave_l", "wave_r"}


class TestCatalogFamilyResolution:
    """A variant with no dedicated dir resolves against the family component whose
    `variants:` names it (catalog family-fallback); every UR variant maps to `arm/ur`."""

    @pytest.fixture(scope="class")
    def catalog(self) -> Catalog:
        return Catalog(root=COMPONENTS_ROOT)

    @pytest.mark.parametrize("variant", UR_VARIANTS)
    def test_every_variant_resolves_to_family(self, catalog: Catalog, variant: str) -> None:
        spec = catalog.get("arm", variant)
        assert variant in spec.variants
        assert "tip_link" not in spec.caps

    def test_resolution_is_cached(self, catalog: Catalog) -> None:
        assert catalog.get("arm", "ur5e") is catalog.get("arm", "ur5e")

    def test_unknown_variant_lists_family_variants(self, catalog: Catalog) -> None:
        with pytest.raises(RuntimeError) as excinfo:
            catalog.get("arm", "ur999")
        msg = str(excinfo.value)
        assert "arm/ur999" in msg
        assert "ur5e" in msg and "ur10e" in msg


class TestRenderedCapsIsArmSpecCompatible:
    """Rendering caps with a concrete mount/variant (as `effective_caps` does)
    must produce a dict `ArmSpec` (caps.py) can front, including mount-substituted
    `named_poses` keys (`_substitute_keys` second pass over the value-rendered tree)."""

    def _rendered_caps(self, variant: str) -> dict:
        from arena_robots.caps import _substitute_keys

        spec = Catalog(root=COMPONENTS_ROOT).get("arm", variant)
        context = {"mount": "arm0", "variant": variant, "prefix": "robot_", "parent": "chassis_link"}
        rendered = YAMLReplacer(context).replace(copy.deepcopy(spec.caps))
        return _substitute_keys(rendered, context)

    @pytest.mark.parametrize("variant", ["ur3e", "ur5e", "ur10e"])
    def test_scalar_fields_resolve(self, variant: str) -> None:
        caps = self._rendered_caps(variant)
        arm = ArmSpec(path=Path("test"), raw=caps, name="arm0")
        assert arm.base_link == "robot_chassis_link"
        assert arm.controller == "arm0_controller"
        assert arm.planning_group == "arm0_manipulator"
        assert arm.workspace is not None

    def test_named_poses_keys_are_mount_substituted(self) -> None:
        caps = self._rendered_caps("ur5e")
        arm = ArmSpec(path=Path("test"), raw=caps, name="arm0")
        wave_joints = arm.named_poses["wave_up"]
        assert "robot_arm0_shoulder_pan_joint" in wave_joints
        assert "${prefix}${mount}_shoulder_pan_joint" not in wave_joints


class TestSrdfFragment:
    """The shared `arm/ur/srdf/ur.srdf.xacro` is UR-type-agnostic (link/joint names are
    identical across the family); only prefix/mount/parent are xacro args."""

    def test_parses_as_xml(self) -> None:
        root = ET.parse(FAMILY / "srdf" / "ur.srdf.xacro").getroot()
        group = root.find("group")
        assert group.get("name") == "$(arg mount)_manipulator"
        assert [state.get("name") for state in root.findall("group_state")] == ["stow", "ready"]
        # 8 intra-arm + 2 mount-adjacency (chassis_link vs arm base_link/base_link_inertia)
        assert len(root.findall("disable_collisions")) == 10

    @pytest.mark.requires_installed("arm/ur")
    @pytest.mark.parametrize("pose", ["stow", "ready"])
    def test_group_states_are_the_named_poses_of_the_arm(self, pose: str) -> None:
        rc = RobotIdentifier("jackal").resolve_sync().effective_caps({"arm": [RequestPart(variant="ur5e", mount="top")]})
        joints = [f"top_{joint}" for joint in ARM_JOINTS]
        assert [rc.arm["top"].named_poses[pose][joint] for joint in joints] == POSES[pose]


class TestJointLimits:
    """The family `arm/ur/joint_limits.yaml` authors the acceleration only, `max_velocity`,
    the position range and `max_effort` of each variant are its ur_description URDF limits."""

    def _file(self) -> dict:
        return yaml.safe_load((FAMILY / "joint_limits.yaml").read_text())["joint_limits"]

    def _limits(self, variant: str) -> dict:
        rc = RobotIdentifier("jackal").resolve_sync().effective_caps({"arm": [RequestPart(variant=variant, mount="top")]})
        return {joint.removeprefix("top_"): entry for joint, entry in rc.arm["top"].joint_limits.items()}

    def test_the_file_is_one_acceleration_entry_per_templated_joint(self) -> None:
        limits = self._file()
        assert set(limits) == {f"${{prefix}}${{mount}}_{j}" for j in ARM_JOINTS}
        for entry in limits.values():
            assert entry == {"has_acceleration_limits": True, "max_acceleration": 15.0}

    @pytest.mark.requires_installed("arm/ur")
    @pytest.mark.parametrize("variant", UR_VARIANTS)
    def test_each_variant_has_six_joints_with_urdf_limits_and_the_family_acceleration(self, variant: str) -> None:
        limits = self._limits(variant)
        assert set(limits) == set(ARM_JOINTS)
        for entry in limits.values():
            assert entry["has_velocity_limits"] is True
            assert entry["has_effort_limits"] is True
            assert entry["max_acceleration"] == 15.0

    @pytest.mark.requires_installed("arm/ur")
    def test_ur5e_moves_every_joint_at_its_vendor_180_degrees_per_second(self) -> None:
        limits = self._limits("ur5e")
        for joint in ARM_JOINTS:
            assert limits[joint]["max_velocity"] == pytest.approx(math.pi)

    @pytest.mark.requires_installed("arm/ur")
    def test_ur10e_differs_from_ur5e(self) -> None:
        """ur10e shoulders are slower (120deg/s)."""
        ur5e, ur10e = self._limits("ur5e"), self._limits("ur10e")
        assert ur5e != ur10e
        assert ur10e["shoulder_pan_joint"]["max_velocity"] == pytest.approx(math.radians(120))
        assert ur10e["wrist_1_joint"]["max_velocity"] == pytest.approx(math.radians(180))

    @pytest.mark.requires_installed("arm/ur")
    def test_cb_ur5_follows_ur_description_not_ur5e(self) -> None:
        """CB-series ur5 takes ur_description's 180deg/s wrists (the URDF/controller value)."""
        assert self._limits("ur5")["wrist_1_joint"]["max_velocity"] == pytest.approx(math.pi)

    @pytest.mark.requires_installed("arm/ur")
    def test_a_continuous_wrist_has_no_position_limits(self) -> None:
        assert self._limits("ur3")["wrist_3_joint"]["has_position_limits"] is False
        assert self._limits("ur5e")["wrist_3_joint"]["has_position_limits"] is True


POSES = {
    "stow": [0.0, -1.57, 1.57, -1.57, -1.57, 0.0],
    "ready": [0.0, -1.0, 1.0, -1.5, -1.57, 0.0],
    "wave_up": [0.0, -0.5, 1.2, -2.3, -1.57, 0.0],
    "wave_l": [0.0, -0.5, 1.2, -2.3, -1.57, -0.6],
    "wave_r": [0.0, -0.5, 1.2, -2.3, -1.57, 0.6],
}


class TestPresetArmCaps:
    """The ur5e `arm` caps of each `*_plus` preset keep the joint names, controller, base link and named poses of the robots they replaced."""

    @pytest.mark.parametrize(
        ("preset", "mount", "prefix", "base_link"),
        [
            pytest.param("rbvogui_plus", "arm0", "robot_", "robot_chassis_link", marks=pytest.mark.requires_installed("rbvogui")),
            pytest.param("rbkairos_plus", "arm_mount", "robot_", "robot_top_cover", marks=pytest.mark.requires_installed("rbkairos")),
            pytest.param("ridgeback_plus", "arm_mount", "", "chassis_link", marks=pytest.mark.requires_installed("ridgeback")),
        ],
    )
    @pytest.mark.requires_installed("arm/ur")
    def test_the_expanded_instance_carries_the_replaced_robots_arm_facts(self, preset: str, mount: str, prefix: str, base_link: str) -> None:
        from arena_robots.request import InstanceRequest
        from arena_robots.Robot import RobotIdentifier
        from arena_robots.SetupFile import Config

        config = Config.parse(preset)[0]
        view = RobotIdentifier(config.robot).resolve_sync()
        arm = InstanceRequest.of(view, config).caps_of(view).arm[mount].raw

        joints = [f"{prefix}arm_{joint}" for joint in ARM_JOINTS]
        assert arm["base_link"] == base_link
        assert arm["tip_link"] == f"{prefix}arm_tool0"
        assert arm["chain"] == joints
        assert arm["controller"] == "arm_controller"
        assert arm["workspace"] == {"type": "box", "frame": base_link, "min": [0.2, -0.5, 0.2], "max": [0.7, 0.5, 1.0]}
        assert {name: [pose["joints"][joint] for joint in joints] for name, pose in arm["named_poses"].items()} == POSES


class TestEffectiveCapsHonorsMountFrame:
    """A mount's ``frame`` (identity stem) drives caps content via ``catalog._frame_stem``,
    not the mount's addressing ``name``; the instance dict key stays the addressing name."""

    def test_tip_link_and_poses_use_frame_not_name(self) -> None:
        from arena_robots.assembly import Mount, Placement, ResolvedAssembly
        from arena_robots.caps import RobotCaps

        mount = Mount(name="arm", parent="chassis_link", xyz=(0.0, 0.0, 0.235), accepts=("arm",), frame="arm0")
        resolved = ResolvedAssembly(placements=[Placement(type="arm", variant="ur5e", mount=mount)])
        catalog = Catalog(root=COMPONENTS_ROOT)
        rc = RobotCaps(robot_dir=Path("/dev/null"), resolved=resolved, catalog=catalog)

        arms = rc.arm
        assert set(arms) == {"arm"}
        arm = arms["arm"].raw
        assert arm["tip_link"] == "robot_arm0_tool0"
        assert arm["named_poses"]["stow"]["joints"]["robot_arm0_shoulder_pan_joint"] == 0.0


class TestEffectiveCapsHonorsChassisPrefix:
    """A zero-prefix chassis (e.g. jackal, ``assembly.yaml`` ``prefix: ""``) must render
    arm caps without the ``robot_`` default baked in."""

    def test_zero_prefix_chassis_renders_unprefixed(self) -> None:
        from arena_robots.assembly import Mount, Placement, ResolvedAssembly
        from arena_robots.caps import RobotCaps

        mount = Mount(name="arm", parent="chassis_link", xyz=(0.0, 0.0, 0.235), accepts=("arm",))
        resolved = ResolvedAssembly(placements=[Placement(type="arm", variant="ur5e", mount=mount)])
        catalog = Catalog(root=COMPONENTS_ROOT)
        rc = RobotCaps(robot_dir=Path("/dev/null"), resolved=resolved, catalog=catalog, prefix="")

        arm = rc.arm["arm"].raw
        assert arm["base_link"] == "chassis_link"
        assert arm["tip_link"] == "arm_tool0"
        assert arm["named_poses"]["stow"]["joints"]["arm_shoulder_pan_joint"] == 0.0


@pytest.mark.requires_installed("arm/ur")
class TestChainFromTheUrdf:
    @pytest.mark.parametrize("variant", UR_VARIANTS)
    def test_the_chain_is_the_six_arm_joints_between_the_srdf_links(self, variant: str) -> None:
        rc = RobotIdentifier("jackal").resolve_sync().effective_caps({"arm": [RequestPart(variant=variant, mount="top")]})
        arm = rc.arm["top"]
        assert arm.chain == [f"top_{joint}" for joint in ARM_JOINTS]
        assert arm.tip_link == "top_tool0"
        assert arm.base_link == "chassis_link"

    def test_the_urdf_and_srdf_leaves_have_derived_provenance(self) -> None:
        rc = RobotIdentifier("jackal").resolve_sync().effective_caps({"arm": [RequestPart(variant="ur5e", mount="top")]})
        assert rc.provenance("arm.top.chain") == "derived"
        assert rc.provenance("arm.top.tip_link") == "derived"
        assert rc.provenance("arm.top.named_poses.stow.joints.top_elbow_joint") == "derived"
        assert rc.provenance("arm.top.named_poses.wave_up.joints.top_elbow_joint") == "component"
        assert rc.provenance("arm.top.base_link") == "component"
