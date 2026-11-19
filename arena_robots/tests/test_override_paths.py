from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from arena_robots.caps import MobileSpec, adapter_blocks, typed_fields
from arena_robots.override_paths import complete, entry_paths, override_paths
from arena_robots.Robot import RobotIdentifier

ROBOTS_DIR = Path(__file__).resolve().parent.parent / "robots"


def _by_path(entry: str) -> dict:
    return {item.path: item for item in entry_paths(entry)}


class TestOverridePaths:
    def test_jackal_mobile_leaf_list_and_map_carry_the_effective_values(self):
        authored = yaml.safe_load((ROBOTS_DIR / "jackal" / "caps" / "mobile.yaml").read_text())
        found = {item.path: item for item in override_paths(RobotIdentifier("jackal").resolve_sync().caps)}
        assert found["mobile.radius"].value == authored["radius"] and found["mobile.radius"].kind == "value"
        assert found["mobile.velocity_limits.linear"].value == authored["velocity_limits"]["linear"]
        assert found["mobile.velocity_limits.linear"].kind == "list"
        assert found["mobile.velocity_limits"].kind == "map"
        assert found["mobile.velocity_limits"].value == authored["velocity_limits"]
        assert all(item.part is None and item.relative is None for item in found.values())

    def test_a_list_is_one_entry_and_not_one_per_element(self):
        found = {item.path for item in override_paths(RobotIdentifier("jackal").resolve_sync().caps)}
        assert not any(path.startswith("mobile.velocity_limits.linear.") for path in found)
        assert not any(path.startswith("mobile.footprint.") for path in found)

    def test_adapter_blocks_and_typed_fields_the_file_leaves_out_are_unset(self):
        caps = RobotIdentifier("jackal").resolve_sync().caps
        found = {item.path: item for item in override_paths(caps)}
        raw = set(caps.mobile.raw)
        unset_blocks = adapter_blocks("mobile") - raw
        assert unset_blocks
        for block in unset_blocks:
            assert found[f"mobile.{block}"].kind == "unset" and found[f"mobile.{block}"].value is None
        for field in typed_fields(MobileSpec) - raw - {"laser"}:
            assert found[f"mobile.{field}"].kind == "unset"
        assert found["mobile.laser"].kind == "map"
        for key in raw:
            assert found[f"mobile.{key}"].kind != "unset"

    def test_an_override_shows_as_the_effective_value(self):
        found = _by_path("jackal[mobile.velocity_limits.linear=[-0.5,1.0]]")
        assert found["mobile.velocity_limits.linear"].value == [-0.5, 1.0]

    def test_placed_part_paths_carry_the_mount_and_the_bracket_relative_path(self):
        found = _by_path("jackal[top=arm/ur5e]")
        arm = {path: item for path, item in found.items() if path.startswith("arm.")}
        assert arm
        assert all(item.part == "top" and item.path == f"arm.top.{item.relative}" for item in arm.values())
        assert arm["arm.top.controller"].kind == "value" and arm["arm.top.controller"].relative == "controller"
        assert arm["arm.top.stall_time"].relative == "stall_time"
        assert found["mobile.radius"].part is None

    def test_part_overrides_inside_brackets_change_the_effective_value(self):
        found = _by_path("jackal[top=arm/ur5e[stall_time=2.0]]")
        assert found["arm.top.stall_time"].value == 2.0 and found["arm.top.stall_time"].kind == "value"

    def test_static_instances_are_addressed_by_instance_name_without_a_part(self):
        found = _by_path("tiago")
        instance_paths = [item for item in found.values() if item.path.split(".")[0] in {"arm", "lift", "gripper"}]
        assert {".".join(item.path.split(".")[:2]) for item in instance_paths} == {"arm.arm", "lift.torso", "gripper.gripper"}
        assert all(item.part is None and item.relative is None for item in instance_paths)
        assert found["lift.torso.joint"].value == "torso_lift_joint"

    def test_entry_paths_rejects_an_unknown_override_path(self):
        with pytest.raises(ValueError, match="unknown field 'velocty_limits'"):
            entry_paths("jackal[mobile.velocty_limits.linear=[0,1]]")

    def test_entry_paths_rejects_an_unclosed_bracket(self):
        with pytest.raises(ValueError, match="closing"):
            entry_paths("jackal[mobile.radius=0.3")

    def test_entry_paths_of_an_unknown_model_raises_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            entry_paths("no_such_robot")

    def test_every_path_is_listed_once(self):
        found = [item.path for item in entry_paths("jackal[top=arm/ur5e]")]
        assert len(set(found)) == len(found)


class TestComplete:
    def test_robot_level_key_folds_to_the_next_group(self):
        assert complete("jackal[mobile.vel") == ["jackal[mobile.velocity_limits."]

    def test_robot_level_leaf_and_list_complete_to_an_assignment(self):
        assert complete("jackal[mobile.velocity_limits.l") == ["jackal[mobile.velocity_limits.linear="]
        assert complete("jackal[mobile.rad") == ["jackal[mobile.radius="]

    def test_first_segment_completes_to_the_cap(self):
        assert complete("jackal[2,mob") == ["jackal[2,mobile."]

    def test_unset_typed_field_offers_assignment_and_group(self):
        field = sorted(typed_fields(MobileSpec) - set(RobotIdentifier("jackal").resolve_sync().caps.mobile.raw))[0]
        found = complete(f"jackal[mobile.{field}")
        assert {f"jackal[mobile.{field}=", f"jackal[mobile.{field}."} <= set(found)

    def test_unset_adapter_block_completes_as_a_group(self):
        caps = RobotIdentifier("jackal").resolve_sync().caps
        block = sorted(adapter_blocks("mobile") - set(caps.mobile.raw))[0]
        assert f"jackal[mobile.{block}." in complete(f"jackal[mobile.{block}")

    def test_static_instances_complete_by_instance_name(self):
        assert complete("tiago[arm.") == ["tiago[arm.arm."]
        assert "tiago[lift.torso.joint=" in complete("tiago[lift.torso.j")

    def test_placed_part_paths_are_not_offered_at_robot_level(self):
        assert complete("jackal[top=arm/ur5e,arm.") == []

    def test_inside_part_brackets_completes_cap_instance_relative_paths(self):
        found = complete("jackal[top=arm/ur5e[st")
        assert "jackal[top=arm/ur5e[stall_time=" in found
        assert all(candidate.startswith("jackal[top=arm/ur5e[st") for candidate in found)

    def test_inside_part_brackets_after_a_finished_override(self):
        found = complete("jackal[top=arm/ur5e[stall_time=2.0,con")
        assert "jackal[top=arm/ur5e[stall_time=2.0,controller=" in found
        assert all(candidate.startswith("jackal[top=arm/ur5e[stall_time=2.0,con") for candidate in found)

    def test_type_shorthand_part_completes_like_a_mount_directive(self):
        assert "jackal[arm=ur5e[stall_time=" in complete("jackal[arm=ur5e[stall_t")

    def test_after_a_finished_part_robot_level_keys_complete_again(self):
        assert complete("jackal[top=arm/ur5e[stall_time=2.0],mob") == ["jackal[top=arm/ur5e[stall_time=2.0],mobile."]

    def test_whitespace_after_a_comma_is_kept(self):
        assert complete("jackal[2, mobile.rad") == ["jackal[2, mobile.radius="]

    @pytest.mark.parametrize(
        "entry",
        [
            "no_such_robot[mobile.",
            "jackal",
            "",
            "jackal[top=nonsense/x[st",
            "jackal[top=arm/ur5e[stall_time=2.0]",
            "jackal[mobile.radius=",
            "jackal[mobile.velocity_limits.linear=[1,",
            "jackal[top=arm/ur5e[stall_time=",
            "jackal[a=b]]",
        ],
    )
    def test_nothing_to_complete_returns_empty(self, entry: str):
        assert complete(entry) == []

    def test_repeated_completion_is_stable(self):
        assert complete("jackal[top=arm/ur5e[st") == complete("jackal[top=arm/ur5e[st")
