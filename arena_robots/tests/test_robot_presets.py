from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from arena_robots.preset import FILE_NAME, Preset, PresetError, expand, lookup
from arena_robots.request import InstanceRequest
from arena_robots.Robot import RobotIdentifier
from arena_robots.SetupFile import Config

ROBOTS_DIR = Path(__file__).resolve().parents[1] / "robots"
PRESETS = {"rbvogui_plus": ("rbvogui", "arm0"), "rbkairos_plus": ("rbkairos", "arm_mount"), "ridgeback_plus": ("ridgeback", "arm_mount")}


class TestShippedPresets:
    @pytest.mark.parametrize("name", sorted(PRESETS))
    def test_a_preset_directory_holds_only_its_preset_file(self, name: str) -> None:
        assert [path.name for path in (ROBOTS_DIR / name).iterdir()] == [FILE_NAME]

    @pytest.mark.parametrize(("name", "base"), [(name, base) for name, (base, _) in PRESETS.items()])
    def test_the_preset_names_its_base_robot_and_places_an_ur5e(self, name: str, base: str) -> None:
        preset = lookup(name)
        assert preset is not None
        assert (preset.robot, preset.parts, preset.caps) == (base, {"arm": ["ur5e"]}, {})

    @pytest.mark.parametrize(("name", "base", "mount"), [(name, base, mount) for name, (base, mount) in PRESETS.items()])
    def test_the_expanded_instance_places_the_ur5e_on_the_arm_mount(self, name: str, base: str, mount: str) -> None:
        (config,) = Config.parse(name)
        view = RobotIdentifier(config.robot).resolve_sync()
        request = InstanceRequest.of(view, config)
        assert config.robot == base
        assert [(part.variant, part.mount) for part in request.parts["arm"]] == [("ur5e", mount)]
        assert set(request.caps_of(view).arm) == {mount}

    @pytest.mark.parametrize("name", sorted(PRESETS))
    def test_the_view_of_a_preset_carries_the_preset_and_is_no_definition(self, name: str) -> None:
        view = RobotIdentifier(name).resolve_sync()
        assert view.preset == lookup(name)
        assert not view.is_definition

    def test_the_robot_listing_includes_every_preset(self) -> None:
        assert set(PRESETS) <= {identifier.shortname for identifier in RobotIdentifier.listall()}

    def test_a_robot_that_is_no_preset_expands_to_itself(self) -> None:
        parts, caps = {"lidar": ["jackal_default"]}, {"mobile.velocity_limits.linear": [-0.5, 1.0]}
        assert expand("jackal", parts, caps) == ("jackal", parts, caps)
        assert lookup("jackal") is None


class TestCallerBracketsWin:
    def test_the_same_type_replaces_the_preset_part(self) -> None:
        (config,) = Config.parse({"robot": "rbvogui_plus", "arm": "ur10"})
        assert config.parts == {"arm": ["ur10"]}

    def test_a_socket_of_the_preset_type_replaces_the_preset_part(self) -> None:
        (config,) = Config.parse({"robot": "rbvogui_plus", "arm0": "arm/ur3"})
        assert config.parts == {"arm0": ["arm/ur3"]}

    def test_clearing_the_socket_drops_the_preset_part(self) -> None:
        (config,) = Config.parse({"robot": "rbvogui_plus", "arm0": "none"})
        assert config.parts == {"arm0": ["none"]}

    def test_a_part_of_another_type_keeps_the_preset_part(self) -> None:
        (config,) = Config.parse({"robot": "rbvogui_plus", "front": "lidar/sick_s300"})
        assert config.parts == {"arm": ["ur5e"], "front": ["lidar/sick_s300"]}

    def test_the_preset_does_not_share_its_lists_with_the_instances(self) -> None:
        first, second = Config.parse({"robot": "rbvogui_plus", "count": 2})
        first.parts["arm"].append("ur3")
        assert second.parts == {"arm": ["ur5e"]}
        assert lookup("rbvogui_plus").parts == {"arm": ["ur5e"]}

    def test_the_resolved_instance_is_the_callers_arm(self) -> None:
        (config,) = Config.parse({"robot": "ridgeback_plus", "arm": "ur10"})
        view = RobotIdentifier(config.robot).resolve_sync()
        assert [part.variant for part in InstanceRequest.of(view, config).parts["arm"]] == ["ur10"]


class TestPresetCaps:
    PRESET = Preset.parse("slow_vogui", {"robot": "rbvogui", "parts": {"arm": "ur5e"}, "caps": {"mobile.velocity_limits.linear": [-1.0, 1.0], "mobile.planar.lag": 0.1}})

    def test_the_preset_caps_apply_when_the_caller_names_none(self) -> None:
        assert self.PRESET.merged({}, {}) == ({"arm": ["ur5e"]}, {"mobile.velocity_limits.linear": [-1.0, 1.0], "mobile.planar.lag": 0.1})

    def test_a_caller_cap_replaces_the_same_and_the_enclosing_and_nested_paths(self) -> None:
        _, caps = self.PRESET.merged({}, {"mobile.velocity_limits": {"linear": [-0.5, 0.5]}})
        assert caps == {"mobile.planar.lag": 0.1, "mobile.velocity_limits": {"linear": [-0.5, 0.5]}}

    def test_inline_values_parse_like_setup_entries(self) -> None:
        preset = Preset.parse("scaled", {"robot": "rbvogui", "caps": {"mobile.velocity_limits.linear": "*0.5"}})
        assert preset.caps == {"mobile.velocity_limits.linear": {"scale": 0.5}}


class TestInvalidPresets:
    def test_an_unknown_base_robot_raises(self) -> None:
        with pytest.raises(PresetError, match="no_such_robot"):
            Preset.parse("bad", {"robot": "no_such_robot"}).validate()

    def test_a_base_that_is_a_preset_raises(self) -> None:
        with pytest.raises(PresetError, match="not a robot definition"):
            Preset.parse("bad", {"robot": "rbvogui_plus"}).validate()

    def test_an_unknown_variant_raises(self) -> None:
        with pytest.raises(PresetError, match="nonesuch"):
            Preset.parse("bad", {"robot": "rbvogui", "parts": {"arm": "nonesuch"}}).validate()

    def test_a_socket_or_type_the_base_lacks_raises(self) -> None:
        with pytest.raises(PresetError, match="unknown 'wing'"):
            Preset.parse("bad", {"robot": "rbvogui", "parts": {"wing": "ur5e"}}).validate()

    def test_more_parts_than_sockets_raises(self) -> None:
        with pytest.raises(PresetError):
            Preset.parse("bad", {"robot": "rbvogui", "parts": {"arm": ["ur5e", "ur10"]}}).validate()

    @pytest.mark.parametrize(
        "data",
        [
            {"parts": {"arm": "ur5e"}},
            {"robot": "rbvogui", "frames": {"arm0": "x"}},
            {"robot": "rbvogui", "caps": {"mobile": 1}},
            {"robot": "rbvogui", "caps": {"mobile.adapter": "drl"}},
            ["rbvogui"],
        ],
    )
    def test_a_malformed_file_raises(self, data: object) -> None:
        with pytest.raises(PresetError):
            Preset.parse("bad", data)

    def test_the_file_loads_from_yaml_text(self, tmp_path: Path) -> None:
        directory = tmp_path / "heavy_vogui"
        directory.mkdir()
        (directory / FILE_NAME).write_text(yaml.safe_dump({"robot": "rbvogui", "parts": {"arm": "ur10"}}))
        preset = Preset.read(directory / FILE_NAME)
        assert (preset.name, preset.robot, preset.parts) == ("heavy_vogui", "rbvogui", {"arm": ["ur10"]})
        preset.validate()
