from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from arena_robots.assembly import AssemblyError, Mount, Placement, RequestPart, ResolvedAssembly, build_request
from arena_robots.caps import RobotCaps
from arena_robots.catalog import Catalog
from arena_robots.request import InstanceRequest
from arena_robots.Robot import RobotIdentifier, RobotView
from arena_robots.inline import format_inline_override, format_part, parse_inline_override, split_part
from arena_robots.SetupFile import Config

COMPONENTS_ROOT = Path(__file__).resolve().parent.parent / "components"

MOBILE = {
    "odom_frame": "odom",
    "radius": 0.3,
    "velocity_limits": {"linear": [-1.0, 2.0], "angular": [-3.0, 3.0]},
    "planar": {"velocity": {"linear": [-1.0, 2.0], "angular": [-3.0, 3.0]}, "acceleration": {"linear": 10.0, "angular": 20.0}},
    "nav2": {"controller": "dwb"},
}
ARM = {"left": {"controller": "left_controller", "stall_time": 1.0}, "right": {"controller": "right_controller", "stall_time": 1.5}}


def _robot_dir(tmp_path: Path, **files: dict) -> Path:
    robot_dir = tmp_path / "robot"
    robot_dir.mkdir(parents=True)
    (robot_dir / "caps.yaml").write_text(yaml.safe_dump(files))
    return robot_dir


def _view(tmp_path: Path, **files: dict) -> RobotView:
    robot_dir = _robot_dir(tmp_path, **files)
    (robot_dir / "model_params.yaml").write_text(yaml.safe_dump({"robot_model": "robot"}))
    return RobotView(robot_dir)


class TestSetupGrammar:
    def test_dotted_cap_key_is_an_override(self):
        (config,) = Config.parse({"robot": "jackal", "mobile.velocity_limits.linear": [-0.5, 1.0], "mobile.adapter": "nav2"})
        assert config.caps == {"mobile.velocity_limits.linear": [-0.5, 1.0]}
        assert config.adapters == {"mobile": "nav2"}

    def test_string_values_go_through_the_inline_grammar(self):
        (config,) = Config.parse({"robot": "jackal", "mobile.radius": "*1.2", "mobile.velocity_limits.linear": "[-0.5,1.0]", "mobile.odom_frame": "odom", "mobile.laser.range": ["5.0"]})
        assert config.caps == {"mobile.radius": {"scale": 1.2}, "mobile.velocity_limits.linear": [-0.5, 1.0], "mobile.odom_frame": "odom", "mobile.laser.range": [5.0]}

    def test_bad_string_scale_names_the_key(self):
        with pytest.raises(RuntimeError, match="'mobile.radius'"):
            Config.parse({"robot": "jackal", "mobile.radius": "*wide"})

    def test_scale_override_kept_verbatim(self):
        (config,) = Config.parse({"robot": "jackal", "mobile.acceleration_limits": {"scale": 0.5}})
        assert config.caps == {"mobile.acceleration_limits": {"scale": 0.5}}

    def test_count_copies_overrides_independently(self):
        first, second = Config.parse({"robot": "jackal", "count": 2, "mobile.radius": 0.4})
        first.caps["mobile.radius"] = 1.0
        assert second.caps == {"mobile.radius": 0.4}

    @pytest.mark.parametrize("bad_key", ["mobile.", ".radius", "mobile..radius"])
    def test_malformed_dotted_key_raises(self, bad_key: str):
        with pytest.raises(RuntimeError, match="dotted key"):
            Config.parse({"robot": "jackal", bad_key: 1})

    def test_dotted_frames_key_is_a_frame_override(self):
        (config,) = Config.parse({"robot": "jackal", "frames.top": "custom_top", "frames": {"front": "custom_front"}})
        assert config.frames == {"top": "custom_top", "front": "custom_front"}
        assert config.caps == {}

    def test_caps_is_reserved(self):
        with pytest.raises(RuntimeError, match="reserved"):
            Config.parse({"robot": "jackal", "caps": "x"})


class TestInlineValues:
    @pytest.mark.parametrize(
        ("raw", "value"),
        [("0.5", 0.5), ("*0.5", {"scale": 0.5}), (["-0.5", "1.0"], [-0.5, 1.0]), (["[0.1,0.2]", "[0.3,0.4]"], [[0.1, 0.2], [0.3, 0.4]]), ("true", True), ("dwb", "dwb")],
    )
    def test_parse(self, raw: str | list[str], value: object):
        assert parse_inline_override(raw) == value

    @pytest.mark.parametrize(
        ("value", "text"),
        [(0.5, "0.5"), ({"scale": 0.25}, "*0.25"), ([-0.5, 1.0], "[-0.5,1.0]"), ([[0.1, 0.2], [0.3, 0.4]], "[[0.1,0.2],[0.3,0.4]]"), (True, "true"), ("dwb", "dwb"), (float("inf"), ".inf")],
    )
    def test_format(self, value: object, text: str):
        assert format_inline_override(value) == text

    def test_bad_scale_raises(self):
        with pytest.raises(ValueError, match="scale"):
            parse_inline_override("*fast")

    def test_mapping_has_no_inline_form(self):
        with pytest.raises(ValueError, match="no inline form"):
            format_inline_override({"linear": [0.0, 1.0]})


class TestMobileOverlay:
    def test_set_reaches_typed_accessor(self, tmp_path: Path):
        caps = RobotCaps(_robot_dir(tmp_path, mobile=MOBILE), overrides={"mobile.velocity_limits.linear": [-0.5, 0.5]})
        limits = caps.mobile.velocity_limits
        assert limits is not None
        assert (limits.linear.min, limits.linear.max) == (-0.5, 0.5)
        assert (limits.angular.min, limits.angular.max) == (-3.0, 3.0)

    def test_scale_multiplies_every_number_below_the_path(self, tmp_path: Path):
        caps = RobotCaps(_robot_dir(tmp_path, mobile=MOBILE), overrides={"mobile.limits.velocity": {"scale": 0.5}})
        planar = caps.mobile.planar
        assert planar is not None
        assert (planar.velocity["linear"].min, planar.velocity["linear"].max) == (-0.5, 1.0)
        assert (planar.velocity["angular"].min, planar.velocity["angular"].max) == (-1.5, 1.5)

    def test_unset_typed_field_can_be_set(self, tmp_path: Path):
        caps = RobotCaps(_robot_dir(tmp_path, mobile=MOBILE), overrides={"mobile.acceleration_limits.linear": 1.5, "mobile.acceleration_limits.angular": 2.0})
        limits = caps.mobile.acceleration_limits
        assert limits is not None
        assert (limits.linear, limits.angular) == (1.5, 2.0)

    def test_adapter_sub_block_is_overridden(self, tmp_path: Path):
        caps = RobotCaps(_robot_dir(tmp_path, mobile=MOBILE), overrides={"mobile.nav2.controller": "mppi"})
        assert caps.mobile.sub("nav2") == {"controller": "mppi"}

    def test_adapter_block_absent_from_the_file_is_created(self, tmp_path: Path):
        caps = RobotCaps(_robot_dir(tmp_path, mobile=MOBILE), overrides={"mobile.meshnav.planner": "cvp"})
        assert caps.mobile.sub("meshnav") == {"planner": "cvp"}

    def test_unknown_field_raises(self, tmp_path: Path):
        caps = RobotCaps(_robot_dir(tmp_path, mobile=MOBILE), overrides={"mobile.velocty_limits.linear": [0.0, 1.0]})
        with pytest.raises(ValueError, match="unknown field"):
            caps.mobile.raw

    def test_source_is_kept_out_of_the_cap_data_and_exposed_on_the_spec(self, tmp_path: Path):
        caps = RobotCaps(_robot_dir(tmp_path, mobile={**MOBILE, "source": "https://example.test/sheet"}, arm={"source": "https://example.test/arm", **ARM}))
        assert caps.mobile.source == "https://example.test/sheet"
        assert "source" not in caps.mobile.raw
        assert caps.arm["left"].source == "https://example.test/arm"
        assert sorted(caps.arm) == ["left", "right"]
        assert "source" not in caps.effective()["mobile"]

    def test_source_does_not_change_the_caps_hash(self, tmp_path: Path):
        plain = RobotCaps(_robot_dir(tmp_path / "plain", mobile=MOBILE))
        cited = RobotCaps(_robot_dir(tmp_path / "cited", mobile={**MOBILE, "source": "https://example.test/sheet"}))
        assert plain.caps_hash() == cited.caps_hash()

    def test_source_must_be_a_string(self, tmp_path: Path):
        caps = RobotCaps(_robot_dir(tmp_path, mobile={**MOBILE, "source": ["a"]}))
        with pytest.raises(ValueError, match="'source' must be a string"):
            caps.mobile

    def test_source_cannot_be_overridden(self, tmp_path: Path):
        caps = RobotCaps(_robot_dir(tmp_path, mobile={**MOBILE, "source": "https://example.test/sheet"}), overrides={"mobile.source": "elsewhere"})
        with pytest.raises(ValueError, match="unknown field 'source'"):
            caps.mobile.raw

    def test_scaling_unset_value_raises(self, tmp_path: Path):
        caps = RobotCaps(_robot_dir(tmp_path, mobile=MOBILE), overrides={"mobile.footprint_padding": {"scale": 2.0}})
        with pytest.raises(ValueError, match="cannot scale or offset an unset value"):
            caps.mobile.raw

    def test_scaling_text_raises(self, tmp_path: Path):
        caps = RobotCaps(_robot_dir(tmp_path, mobile=MOBILE), overrides={"mobile.odom_frame": {"scale": 2.0}})
        with pytest.raises(ValueError, match="cannot scale"):
            caps.mobile.raw

    def test_file_on_disk_and_other_views_unchanged(self, tmp_path: Path):
        robot_dir = _robot_dir(tmp_path, mobile=MOBILE)
        overridden = RobotCaps(robot_dir, overrides={"mobile.radius": 0.9})
        plain = RobotCaps(robot_dir)
        assert overridden.mobile.radius == 0.9
        assert plain.mobile.radius == 0.3
        assert yaml.safe_load((robot_dir / "caps.yaml").read_text())["mobile"] == MOBILE


class TestInstanceOverlay:
    def test_file_instance_overridden(self, tmp_path: Path):
        caps = RobotCaps(_robot_dir(tmp_path, mobile=MOBILE, arm=ARM), overrides={"arm.right.stall_time": 3.0})
        arms = caps.arm
        assert arms is not None
        assert arms["right"].stall_time == 3.0
        assert arms["left"].stall_time == 1.0

    def test_unknown_file_instance_raises(self, tmp_path: Path):
        caps = RobotCaps(_robot_dir(tmp_path, mobile=MOBILE, arm=ARM), overrides={"arm.middle.stall_time": 3.0})
        with pytest.raises(ValueError, match="unknown arm instance 'middle'"):
            caps.arm

    def test_instance_override_needs_a_field(self, tmp_path: Path):
        caps = RobotCaps(_robot_dir(tmp_path, mobile=MOBILE, arm=ARM), overrides={"arm.right": 3.0})
        with pytest.raises(ValueError, match="<instance>.<field>"):
            caps.arm

    def test_placement_instance_overridden(self):
        mount = Mount(name="arm", parent="chassis_link", xyz=(0.0, 0.0, 0.235), accepts=("arm",))
        resolved = ResolvedAssembly(placements=[Placement(type="arm", variant="ur5e", mount=mount)])
        catalog = Catalog(root=COMPONENTS_ROOT)
        caps = RobotCaps(robot_dir=Path("/dev/null"), resolved=resolved, catalog=catalog, overrides={"arm.arm.stall_time": 4.0})
        plain = RobotCaps(robot_dir=Path("/dev/null"), resolved=resolved, catalog=catalog)
        assert caps.arm is not None and plain.arm is not None
        assert caps.arm["arm"].stall_time == 4.0
        assert plain.arm["arm"].stall_time != 4.0

    def test_unknown_placement_instance_raises(self):
        mount = Mount(name="arm", parent="chassis_link", xyz=(0.0, 0.0, 0.235), accepts=("arm",))
        resolved = ResolvedAssembly(placements=[Placement(type="arm", variant="ur5e", mount=mount)])
        caps = RobotCaps(robot_dir=Path("/dev/null"), resolved=resolved, catalog=Catalog(root=COMPONENTS_ROOT), overrides={"arm.top.stall_time": 4.0})
        with pytest.raises(ValueError, match=r"unknown arm instance\(s\) \['top'\]"):
            caps.arm


class TestValidate:
    def test_valid_overrides_pass(self, tmp_path: Path):
        RobotCaps(_robot_dir(tmp_path, mobile=MOBILE, arm=ARM), overrides={"mobile.velocity_limits.linear": [0.0, 1.0], "arm.left.stall_time": 2.0}).validate_overrides()

    def test_unadvertised_cap_raises(self, tmp_path: Path):
        with pytest.raises(ValueError, match="does not advertise"):
            RobotCaps(_robot_dir(tmp_path, mobile=MOBILE), overrides={"lift.torso.limits": [0.0, 1.0]}).validate_overrides()

    def test_bad_typed_value_raises(self, tmp_path: Path):
        with pytest.raises(ValueError):
            RobotCaps(_robot_dir(tmp_path, mobile=MOBILE), overrides={"mobile.velocity_limits.linear": "fast"}).validate_overrides()


class TestEffectiveCaps:
    def test_no_overrides_reads_the_model_caps_through_the_view(self, tmp_path: Path):
        view = _view(tmp_path, mobile=MOBILE)
        caps = view.effective_caps({})
        assert caps.view is view
        assert caps.robot_dir == view.caps.robot_dir
        assert caps.overrides == {}
        assert caps.mobile.raw == view.caps.mobile.raw

    def test_overrides_on_robot_without_assembly_get_a_fresh_view(self, tmp_path: Path):
        view = _view(tmp_path, mobile=MOBILE)
        caps = view.effective_caps({}, caps={"mobile.radius": 0.5})
        assert caps is not view.caps
        assert caps.mobile.radius == 0.5
        assert view.caps.mobile.radius == 0.3

    def test_two_instances_stay_independent(self, tmp_path: Path):
        view = _view(tmp_path, mobile=MOBILE)
        slow = view.effective_caps({}, caps={"mobile.radius": 0.5})
        fast = view.effective_caps({}, caps={"mobile.radius": 0.7})
        assert (slow.mobile.radius, fast.mobile.radius, view.caps.mobile.radius) == (0.5, 0.7, 0.3)


class TestInstanceRequest:
    def test_json_round_trip(self):
        request = InstanceRequest(
            parts={"arm": [RequestPart(variant="ur5e", mount="top", overrides={"lidar": "front"})]},
            frames={"top": "custom_top"},
            caps={"mobile.radius": 0.5, "mobile.limits.velocity": {"scale": 0.5}},
        )
        assert InstanceRequest.loads(request.dumps()) == request
        assert json.loads(request.dumps())["caps"] == request.caps

    def test_empty_text_is_the_empty_request(self):
        assert InstanceRequest.loads("") == InstanceRequest()

    def test_caps_of_applies_overrides(self, tmp_path: Path):
        view = _view(tmp_path, mobile=MOBILE)
        assert InstanceRequest(caps={"mobile.radius": 0.6}).caps_of(view).mobile.radius == 0.6


class TestPartBrackets:
    @pytest.mark.parametrize(
        ("raw", "variant", "caps"),
        [
            ("arm/fr3", "arm/fr3", {}),
            ("arm/fr3[stall_time=2.0]", "arm/fr3", {"stall_time": 2.0}),
            ("franka_hand[max_effort=*0.5,positions.open=0.04]", "franka_hand", {"max_effort": {"scale": 0.5}, "positions.open": 0.04}),
            ("arm/fr3[workspace.max=[0.6,0.4,0.9]]", "arm/fr3", {"workspace.max": [0.6, 0.4, 0.9]}),
        ],
    )
    def test_split(self, raw: str, variant: str, caps: dict):
        assert split_part(raw) == (variant, caps)

    @pytest.mark.parametrize("raw", ["arm/fr3[stall_time=2.0", "[stall_time=2.0]", "arm/fr3[stall_time]", "arm/fr3[stall_time=1,stall_time=2]", "arm/fr3[.x=1]"])
    def test_malformed_raises(self, raw: str):
        with pytest.raises(ValueError, match="part"):
            split_part(raw)

    def test_format_round_trips(self):
        caps = {"stall_time": 2.0, "max_effort": {"scale": 0.5}, "workspace.max": [0.6, 0.4, 0.9]}
        assert split_part(format_part("arm/fr3", caps)) == ("arm/fr3", caps)
        assert format_part("arm/fr3", {}) == "arm/fr3"

    def test_build_request_carries_part_caps(self):
        view = RobotIdentifier("jackal").resolve_sync()
        assert view.assembly is not None
        request, _, _ = build_request(view.assembly, {"top": ["arm/fr3[stall_time=2.0]"], "gripper": ["franka_hand[max_effort=*0.5]"]})
        assert request["arm"][0].caps == {"stall_time": 2.0}
        assert request["gripper"][0].caps == {"max_effort": {"scale": 0.5}}

    def test_clearing_takes_no_overrides(self):
        view = RobotIdentifier("jackal").resolve_sync()
        assert view.assembly is not None
        with pytest.raises(AssemblyError, match="'none' takes no cap overrides"):
            build_request(view.assembly, {"front": ["none[range=1.0]"]})

    @pytest.mark.requires_installed("arm/fr3", "gripper/franka_hand")
    def test_placed_part_overrides_reach_its_instance(self):
        view = RobotIdentifier("jackal").resolve_sync()
        assert view.assembly is not None
        request, sockets, types = build_request(view.assembly, {"top": ["arm/fr3[stall_time=2.0]"], "top_tool": ["gripper/franka_hand[max_effort=*0.5]"]})
        plain_request, _, _ = build_request(view.assembly, {"top": ["arm/fr3"], "top_tool": ["gripper/franka_hand"]})
        caps = view.effective_caps(request)
        plain = view.effective_caps(plain_request)
        assert caps.arm is not None and caps.gripper is not None and plain.gripper is not None
        assert caps.arm["top"].stall_time == 2.0
        assert caps.gripper["top_tool"].max_effort == pytest.approx(plain.gripper["top_tool"].max_effort * 0.5)
        assert caps.all_overrides == {"arm.top.stall_time": 2.0, "gripper.top_tool.max_effort": {"scale": 0.5}}
        caps.validate_overrides()

    def test_robot_level_override_of_a_placed_part_points_at_the_brackets(self):
        view = RobotIdentifier("jackal").resolve_sync()
        assert view.assembly is not None
        request, _, _ = build_request(view.assembly, {"top": ["arm/fr3"]})
        with pytest.raises(ValueError, match=r"top=arm/fr3\[stall_time=\.\.\.\]"):
            view.effective_caps(request, caps={"arm.top.stall_time": 2.0}).validate_overrides()

    def test_instance_request_json_keeps_part_caps(self):
        request = InstanceRequest(parts={"arm": [RequestPart(variant="fr3", mount="top", caps={"stall_time": 2.0})]})
        assert InstanceRequest.loads(request.dumps()) == request
