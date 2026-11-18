from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from arena_robots.caps import RobotCaps
from arena_robots.inline import format_inline_override, parse_inline_override
from arena_robots.Robot import RobotIdentifier

MOBILE = {
    "odom_frame": "odom",
    "radius": 0.3,
    "footprint": [[0.3, 0.2], [0.3, -0.2], [-0.3, -0.2], [-0.3, 0.2]],
    "velocity_limits": {"linear": [-1.0, 2.0], "angular": [-3.0, 3.0]},
    "acceleration_limits": {"linear": 2.5, "angular": 4.0},
    "planar": {"velocity": {"linear": [-1.0, 1.5], "angular": [-2.0, 2.0]}, "acceleration": {"linear": 10.0, "angular": 3.0}, "gain": {"angular": 0.9}, "lag": 0.1},
    "nav2": {"controller": "dwb"},
}
ARM = {"left": {"controller": "left_controller", "stall_time": 1.0}, "right": {"controller": "right_controller", "stall_time": 1.5}}


def _caps(tmp_path: Path, overrides: dict[str, object], **files: dict) -> RobotCaps:
    caps_dir = tmp_path / "robot" / "caps"
    caps_dir.mkdir(parents=True)
    for cap, data in {"mobile": MOBILE, **files}.items():
        (caps_dir / f"{cap}.yaml").write_text(yaml.safe_dump(data))
    return RobotCaps(caps_dir, overrides=overrides)


class TestInlineOperations:
    @pytest.mark.parametrize(
        ("text", "value"),
        [
            ("*0.5", {"scale": 0.5}),
            ("+0.05", {"offset": 0.05}),
            ("+-0.05", {"offset": -0.05}),
            ("*2-1", {"scale": 2.0, "offset": -1.0}),
            ("*1e-1+3", {"scale": 0.1, "offset": 3.0}),
            ("+[[0.1,0.2]]", {"append": [[0.1, 0.2]]}),
            ("!delete", {"delete": True}),
            ("-0.5", -0.5),
        ],
    )
    def test_inline_text_parses_to_its_operation(self, text: str, value: object):
        assert parse_inline_override(text) == value

    @pytest.mark.parametrize("value", [{"scale": 0.5}, {"offset": 0.05}, {"scale": 2.0, "offset": -1.0}, {"append": [1, 2]}, {"delete": True}])
    def test_formatted_operation_parses_back(self, value: object):
        assert parse_inline_override(format_inline_override(value)) == value

    def test_malformed_affine_is_rejected(self):
        with pytest.raises(ValueError, match=r"\*<scale>"):
            parse_inline_override("*fast")


class TestAffine:
    def test_offset_shifts_a_number(self, tmp_path: Path):
        assert _caps(tmp_path, {"mobile.radius": {"offset": 0.05}}).mobile.radius == pytest.approx(0.35)

    def test_scale_and_offset_compose_as_scale_first(self, tmp_path: Path):
        assert _caps(tmp_path, {"mobile.radius": {"scale": 2.0, "offset": -0.1}}).mobile.radius == pytest.approx(0.5)

    def test_offset_on_a_range_shifts_both_ends(self, tmp_path: Path):
        limits = _caps(tmp_path, {"mobile.velocity_limits.linear": {"offset": 0.5}}).mobile.velocity_limits
        assert limits is not None
        assert (limits.linear.min, limits.linear.max) == pytest.approx((-0.5, 2.5))

    def test_affine_on_an_unset_value_is_rejected(self, tmp_path: Path):
        with pytest.raises(ValueError, match="cannot scale or offset an unset value"):
            _caps(tmp_path, {"mobile.inflation_radius": {"offset": 0.1}}).validate_overrides()

    def test_affine_on_a_string_is_rejected(self, tmp_path: Path):
        with pytest.raises(ValueError, match="cannot scale or offset 'odom'"):
            _caps(tmp_path, {"mobile.odom_frame": {"scale": 2.0}}).validate_overrides()


class TestListAndMapEdits:
    def test_index_sets_one_list_element(self, tmp_path: Path):
        assert _caps(tmp_path, {"mobile.footprint.0": [0.4, 0.2]}).mobile.raw["footprint"][0] == [0.4, 0.2]

    def test_negative_index_counts_from_the_end(self, tmp_path: Path):
        assert _caps(tmp_path, {"mobile.footprint.-1.0": -0.4}).mobile.raw["footprint"][-1] == [-0.4, 0.2]

    def test_index_out_of_range_is_rejected(self, tmp_path: Path):
        with pytest.raises(ValueError, match="index 4 out of range for a list of 4"):
            _caps(tmp_path, {"mobile.footprint.4": [0.0, 0.0]}).validate_overrides()

    def test_name_on_a_list_is_rejected(self, tmp_path: Path):
        with pytest.raises(ValueError, match="indexes a list"):
            _caps(tmp_path, {"mobile.footprint.front": [0.0, 0.0]}).validate_overrides()

    def test_append_extends_a_list(self, tmp_path: Path):
        footprint = _caps(tmp_path, {"mobile.footprint": {"append": [[0.0, 0.3]]}}).mobile.raw["footprint"]
        assert len(footprint) == 5
        assert footprint[-1] == [0.0, 0.3]

    def test_append_to_a_mapping_is_rejected(self, tmp_path: Path):
        with pytest.raises(ValueError, match="append needs a list"):
            _caps(tmp_path, {"mobile.nav2": {"append": [1]}}).validate_overrides()

    def test_delete_removes_a_map_key(self, tmp_path: Path):
        assert "controller" not in _caps(tmp_path, {"mobile.nav2.controller": {"delete": True}}).mobile.raw["nav2"]

    def test_delete_removes_a_list_element(self, tmp_path: Path):
        assert len(_caps(tmp_path, {"mobile.footprint.1": {"delete": True}}).mobile.raw["footprint"]) == 3

    def test_delete_of_an_unset_key_is_rejected(self, tmp_path: Path):
        with pytest.raises(ValueError, match="cannot delete an unset value"):
            _caps(tmp_path, {"mobile.nav2.planner": {"delete": True}}).validate_overrides()

    def test_set_adds_a_new_key_in_an_existing_mapping(self, tmp_path: Path):
        assert _caps(tmp_path, {"mobile.nav2.planner": "smac"}).mobile.raw["nav2"] == {"controller": "dwb", "planner": "smac"}


class TestWildcards:
    def test_star_matches_every_map_key(self, tmp_path: Path):
        limits = _caps(tmp_path, {"mobile.velocity_limits.*": {"scale": 0.5}}).mobile.velocity_limits
        assert limits is not None
        assert (limits.linear.max, limits.angular.max) == pytest.approx((1.0, 1.5))

    def test_star_matches_every_list_element(self, tmp_path: Path):
        footprint = _caps(tmp_path, {"mobile.footprint.*.0": {"scale": 2.0}}).mobile.raw["footprint"]
        assert [point[0] for point in footprint] == pytest.approx([0.6, 0.6, -0.6, -0.6])

    def test_star_delete_empties_a_list(self, tmp_path: Path):
        assert _caps(tmp_path, {"mobile.footprint.*": {"delete": True}}).mobile.raw["footprint"] == []

    def test_star_matching_nothing_is_rejected(self, tmp_path: Path):
        with pytest.raises(ValueError, match="matches nothing"):
            _caps(tmp_path, {"mobile.footprint.*": {"delete": True}, "mobile.footprint.*.0": 1.0}).validate_overrides()

    def test_star_instance_reaches_every_static_instance(self, tmp_path: Path):
        arms = _caps(tmp_path, {"arm.*.stall_time": {"scale": 2.0}}, arm=ARM).arm
        assert arms is not None
        assert (arms["left"].raw["stall_time"], arms["right"].raw["stall_time"]) == pytest.approx((2.0, 3.0))

    def test_named_instance_applies_after_the_star(self, tmp_path: Path):
        arms = _caps(tmp_path, {"arm.*.stall_time": 4.0, "arm.left.stall_time": {"offset": 1.0}}, arm=ARM).arm
        assert arms is not None
        assert (arms["left"].raw["stall_time"], arms["right"].raw["stall_time"]) == pytest.approx((5.0, 4.0))

    def test_star_instance_validates(self, tmp_path: Path):
        _caps(tmp_path, {"arm.*.stall_time": 2.0}, arm=ARM).validate_overrides()

    def test_star_on_a_placed_part_reaches_every_part_of_that_type(self):
        view = RobotIdentifier("jackal").resolve_sync()
        assert view.assembly is not None
        from arena_robots.assembly import build_request

        request = build_request(view.assembly, {"top": ["arm/ur5e"]})[0]
        caps = view.effective_caps(request, caps={"arm.*.stall_time": 3.0})
        arms = caps.arm
        assert arms is not None
        assert arms["top"].raw["stall_time"] == pytest.approx(3.0)


class TestLimits:
    def test_limits_are_materialized_from_planar(self, tmp_path: Path):
        mobile = _caps(tmp_path, {}).mobile
        assert mobile.raw["limits"] == {"velocity": {"linear": [-1.0, 1.5], "angular": [-2.0, 2.0]}, "acceleration": {"linear": 10.0, "angular": 3.0}}
        assert "velocity" not in mobile.raw["planar"]
        assert mobile.raw["planar"]["gain"] == {"angular": 0.9}

    def test_model_planner_limits_stay_unclamped(self, tmp_path: Path):
        mobile = _caps(tmp_path, {}).mobile
        assert mobile.velocity_limits is not None and mobile.acceleration_limits is not None
        assert mobile.velocity_limits.linear.max == pytest.approx(2.0)
        assert mobile.acceleration_limits.angular == pytest.approx(4.0)

    def test_planar_reads_its_velocity_from_limits(self, tmp_path: Path):
        planar = _caps(tmp_path, {"mobile.limits.velocity.linear": [-0.5, 0.5]}).mobile.planar
        assert planar is not None
        assert (planar.velocity["linear"].min, planar.velocity["linear"].max) == (-0.5, 0.5)
        assert planar.gain == {"angular": 0.9}
        assert planar.lag == pytest.approx(0.1)

    def test_limits_override_clamps_the_planner_limits_on_that_axis(self, tmp_path: Path):
        mobile = _caps(tmp_path, {"mobile.limits.velocity.linear": [-0.5, 0.5]}).mobile
        assert mobile.velocity_limits is not None
        assert (mobile.velocity_limits.linear.min, mobile.velocity_limits.linear.max) == (-0.5, 0.5)
        assert (mobile.velocity_limits.angular.min, mobile.velocity_limits.angular.max) == (-3.0, 3.0)

    def test_scaled_limits_clamp_every_axis(self, tmp_path: Path):
        mobile = _caps(tmp_path, {"mobile.limits": {"scale": 0.5}}).mobile
        assert mobile.velocity_limits is not None and mobile.acceleration_limits is not None
        assert (mobile.velocity_limits.linear.max, mobile.velocity_limits.angular.max) == pytest.approx((0.75, 1.0))
        assert (mobile.acceleration_limits.linear, mobile.acceleration_limits.angular) == pytest.approx((2.5, 1.5))

    def test_wider_limits_leave_narrower_planner_limits(self, tmp_path: Path):
        mobile = _caps(tmp_path, {"mobile.limits.velocity.linear": [-5.0, 5.0]}).mobile
        assert mobile.velocity_limits is not None
        assert mobile.velocity_limits.linear.max == pytest.approx(2.0)

    def test_declared_limits_win_over_planar_and_clamp(self, tmp_path: Path):
        mobile = _caps(tmp_path, {}, mobile={**MOBILE, "limits": {"velocity": {"linear": [-0.4, 0.4]}}}).mobile
        assert mobile.limits is not None and mobile.velocity_limits is not None
        assert mobile.limits.velocity["linear"].max == pytest.approx(0.4)
        assert mobile.limits.velocity["angular"].max == pytest.approx(2.0)
        assert mobile.velocity_limits.linear.max == pytest.approx(0.4)
        assert mobile.velocity_limits.angular.max == pytest.approx(3.0)

    def test_planar_velocity_override_is_rejected(self, tmp_path: Path):
        with pytest.raises(ValueError, match=r"override 'limits\.velocity\.linear'"):
            _caps(tmp_path, {"mobile.planar.velocity.linear": [-0.5, 0.5]}).validate_overrides()

    def test_planar_acceleration_block_override_points_at_limits(self, tmp_path: Path):
        with pytest.raises(ValueError, match=r"override 'limits\.acceleration'"):
            _caps(tmp_path, {"mobile.planar.acceleration": {"scale": 0.5}}).validate_overrides()

    def test_limits_typed_view_from_an_unnormalized_file(self):
        mobile = RobotIdentifier("jackal").resolve_sync().caps.mobile
        assert mobile.limits is not None and mobile.planar is not None
        assert mobile.limits.velocity["linear"].max == pytest.approx(mobile.planar.velocity["linear"].max)

    def test_limits_creatable_on_a_robot_without_planar(self, tmp_path: Path):
        mobile = _caps(tmp_path, {"mobile.limits.velocity.linear": [-0.5, 0.5]}, mobile={k: v for k, v in MOBILE.items() if k != "planar"}).mobile
        assert mobile.limits is not None and mobile.planar is not None and mobile.velocity_limits is not None
        assert mobile.planar.velocity["linear"].max == pytest.approx(0.5)
        assert mobile.velocity_limits.linear.max == pytest.approx(0.5)
