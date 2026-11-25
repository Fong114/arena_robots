from __future__ import annotations

import json
from collections.abc import Callable

import pytest
from arena_robots import urdf_facts
from arena_robots.assembly import AssemblyError, ResolvedAssembly, build_request, placed_request, resolve
from arena_robots.caps import RobotCaps
from arena_robots.fleet import fleet_parts, instance_caps, placement_params
from arena_robots.inline import format_part, split_part
from arena_robots.request import InstanceRequest
from arena_robots.Robot import RobotIdentifier, RobotView

BRACKET = {"base_front": ["lidar/arena_default[max_range=8]"]}
FLEET = {"lidar@base_front": "arena_default", "param:lidar@base_front.max_range": "8"}


def _tiago() -> RobotView:
    return RobotIdentifier("tiago").resolve_sync()


def _placed(view: RobotView, directives: dict[str, list[str]]) -> ResolvedAssembly:
    assert view.assembly is not None
    request, cleared_sockets, cleared_types = build_request(view.assembly, directives)
    return resolve(view.assembly, request, cleared_sockets=cleared_sockets, cleared_types=cleared_types)


def _caps_with_args(view: RobotView) -> RobotCaps:
    return view.effective_caps(placed_request(_placed(view, BRACKET)))


class TestBracketKeys:
    def test_a_declared_xacro_argument_becomes_a_part_param(self):
        view = _tiago()
        assert view.assembly is not None
        request, _, _ = build_request(view.assembly, BRACKET)
        (part,) = request["lidar"]
        assert part.params == {"max_range": 8}
        assert part.caps == {}

    def test_an_undeclared_key_stays_a_cap_override(self):
        view = RobotIdentifier("jackal").resolve_sync()
        assert view.assembly is not None
        request, _, _ = build_request(view.assembly, {"top": ["arm/ur5e[stall_time=2.0,safety_limits=true]"]})
        (part,) = request["arm"]
        assert part.params == {"safety_limits": True}
        assert part.caps == {"stall_time": 2.0}

    def test_an_operation_on_an_argument_is_rejected(self):
        view = _tiago()
        assert view.assembly is not None
        with pytest.raises(AssemblyError, match="takes one value"):
            build_request(view.assembly, {"base_front": ["lidar/arena_default[max_range=*2]"]})

    def test_an_argument_of_an_unknown_variant_names_the_variants(self):
        view = _tiago()
        assert view.assembly is not None
        with pytest.raises(AssemblyError, match="not found"):
            build_request(view.assembly, {"base_front": ["lidar/no_such_lidar[max_range=8]"]})

    def test_the_placement_keeps_the_defaults_params_beside_the_argument(self):
        (placement,) = (p for p in _placed(_tiago(), BRACKET).placements if p.type == "lidar")
        assert placement.args == {"max_range": 8}
        assert placement.params["max_range"] == 8
        assert placement.params["samples"] == 401

    def test_a_placement_without_brackets_carries_no_arguments(self):
        assert all(p.args == {} for p in _placed(_tiago(), {}).placements)


class TestRoundTrips:
    def test_format_part_writes_what_split_part_reads(self):
        text = format_part("arena_default", {"mobile.radius": 0.3}, {"max_range": 8})
        assert text == "arena_default[max_range=8,mobile.radius=0.3]"
        assert split_part(text) == ("arena_default", {"max_range": 8, "mobile.radius": 0.3})

    def test_instance_request_keeps_the_arguments(self):
        request = InstanceRequest(parts=placed_request(_placed(_tiago(), BRACKET)))
        (part,) = InstanceRequest.loads(request.dumps()).parts["lidar"]
        assert part.params == {"max_range": 8}
        assert json.loads(request.dumps())["parts"]["lidar"][0]["params"] == {"max_range": 8}

    def test_fleet_params_carry_the_arguments_as_json_entries(self):
        assert placement_params(_placed(_tiago(), BRACKET)) == {"param:lidar@base_front.max_range": "8"}
        assert placement_params(_placed(_tiago(), {})) == {}

    def test_fleet_parts_rebuild_the_arguments(self):
        parts = fleet_parts(_tiago(), FLEET)
        (lidar,) = parts["lidar"]
        assert (lidar.variant, lidar.mount, lidar.params) == ("arena_default", "base_front", {"max_range": 8})

    def test_fleet_parts_ignore_the_argument_entries_as_parts(self):
        assert set(fleet_parts(_tiago(), FLEET)) == {"lidar"}

    def test_the_rebuilt_placement_equals_the_requested_one(self):
        view = _tiago()
        rebuilt = resolve(view.assembly, fleet_parts(view, FLEET))
        (lidar,) = (p for p in rebuilt.placements if p.type == "lidar")
        assert lidar.args == {"max_range": 8}
        assert lidar.params["samples"] == 401

    def test_snapshot_request_names_the_arguments(self):
        view = _tiago()
        rebuilt = view.effective_caps(placed_request(_placed(view, BRACKET)))
        assert rebuilt.snapshot()["request"]["params"] == {"lidar@base_front": {"max_range": 8}}
        assert view.effective_caps({}).snapshot()["request"]["params"] == {}


class TestRenderedLaser:
    @pytest.fixture(autouse=True)
    def _rendering(self, require_installed: Callable[[str], None]) -> None:
        pytest.importorskip("xacro", reason="xacro is not importable, run under the Arena container")
        require_installed("tiago")

    def test_the_argument_reaches_the_effective_laser(self):
        laser = _caps_with_args(_tiago()).mobile.laser
        assert laser is not None
        assert laser.range == pytest.approx(8.0)
        assert laser.num_beams == 401
        assert laser.angle.min == pytest.approx(-1.74533)
        assert laser.angle.increment == pytest.approx((laser.angle.max - laser.angle.min) / 400)

    def test_the_laser_provenance_is_the_urdf(self):
        view = _tiago()
        assert _caps_with_args(view).provenance("mobile.laser.range") == "urdf"
        assert view.effective_caps({}).provenance("mobile.laser.range") == "urdf"

    def test_a_part_without_arguments_has_the_default_urdf_laser(self):
        view = _tiago()
        laser = view.effective_caps({}).mobile.laser
        assert laser is not None
        assert laser.range == pytest.approx(25.0)
        assert laser == view.caps.mobile.laser

    def test_the_fleet_params_rebuild_the_same_effective_caps(self):
        view = _tiago()
        assert instance_caps("tiago", FLEET).caps_hash() == _caps_with_args(view).caps_hash()
        assert instance_caps("tiago", FLEET).mobile.laser.range == pytest.approx(8.0)

    def test_the_rendered_urdf_carries_the_argument(self):
        block = urdf_facts.laser_block(_caps_with_args(_tiago()).rendered_urdf())
        assert block is not None
        assert block["range"] == pytest.approx(8.0)


class TestLaserOverride:
    def test_a_laser_override_raises_and_names_the_bracket_form(self):
        caps = _tiago().effective_caps({}, caps={"mobile.laser.range": 5.0})
        with pytest.raises(ValueError, match=r"base_front=lidar/arena_default\[max_range=\.\.\.\]"):
            caps.validate_overrides()

    def test_a_laser_override_raises_on_a_robot_without_lidar_parts_with_arguments(self):
        caps = RobotIdentifier("jackal").resolve_sync().effective_caps({}, caps={"mobile.laser": {"scale": 0.5}})
        with pytest.raises(ValueError, match="rendered URDF"):
            caps.validate_overrides()
