from __future__ import annotations

import yaml

import pytest
from arena_robots import fleet
from arena_robots.assembly import RequestPart
from arena_robots.caps import RobotCaps
from arena_robots.catalog import Catalog
from arena_robots.override_paths import entry_paths
from arena_robots.request import InstanceRequest
from arena_robots.Robot import RobotIdentifier, RobotView


def _jackal() -> RobotView:
    return RobotIdentifier("jackal").resolve_sync()


def _jackal_caps(overrides: dict[str, object] | None = None) -> RobotCaps:
    return _jackal().effective_caps({}, caps=overrides)


def _jackal_with_ur5e() -> RobotCaps:
    request = InstanceRequest(parts={"arm": [RequestPart(variant="ur5e", mount="top")]})
    return request.caps_of(_jackal())


def test_jackal_body_carries_the_frame_and_the_control_topics():
    body = _jackal_caps().body
    assert body is not None
    assert body.base_frame == "base_link"
    assert body.cmd_vel_topic == "jackal_velocity_controller/cmd_vel"
    assert body.odom_topic == "jackal_velocity_controller/odom"


def test_jackal_body_mass_base_is_the_model_params_value():
    body = _jackal_caps().body
    assert body is not None
    assert body.mass_base == pytest.approx(17.0)


def test_jackal_power_holds_the_power_system_without_the_model_name():
    power = _jackal_caps().power
    assert power is not None
    assert power.power_system["rolling_resistance_crr"] == pytest.approx(0.035)
    assert power.power_system["robot_mass_kg"] == pytest.approx(17.0)
    assert "robot_model" not in power.raw


def test_jackal_acoustics_holds_the_telemetry_profile():
    acoustics = _jackal_caps().acoustics
    assert acoustics is not None
    assert acoustics.raw["L_base_0"] == pytest.approx(40.0)


def test_placed_default_parts_add_their_component_mass_to_the_total():
    caps = _jackal_caps()
    body = caps.body
    assert body is not None
    assert body.mass_parts["lidar.front"] == pytest.approx(Catalog().get("lidar", "jackal_default").mass["kg"])
    assert body.mass_total == pytest.approx(17.0 + sum(body.mass_parts.values()))
    assert body.mass_total > body.mass_base


def test_ur5e_component_mass_enters_the_parts_and_the_total():
    kg = Catalog().get("arm", "ur5e").mass.get("kg")
    if kg is None:
        pytest.skip("arm/ur5e declares no mass in its component.yaml, so there is no part mass to assert")
    body = _jackal_with_ur5e().body
    assert body is not None
    assert body.mass_parts["arm.top"] == pytest.approx(kg)
    assert body.mass_total == pytest.approx(17.0 + sum(body.mass_parts.values()))


def test_power_override_applies_to_the_power_system():
    caps = _jackal_caps({"power.power_system.rolling_resistance_crr": 0.05})
    caps.validate_overrides()
    assert caps.power.power_system["rolling_resistance_crr"] == pytest.approx(0.05)
    assert _jackal_caps().power.power_system["rolling_resistance_crr"] == pytest.approx(0.035)


def test_body_mass_override_raises():
    caps = _jackal_caps({"body.mass.base": 30.0})
    with pytest.raises(ValueError, match="not overridable"):
        caps.validate_overrides()


def test_body_frame_override_applies():
    caps = _jackal_caps({"body.base_frame": "chassis"})
    caps.validate_overrides()
    assert caps.body.base_frame == "chassis"


def test_power_profile_takes_the_power_model_mass():
    view = _jackal()
    caps = view.effective_caps({})
    profile = view.power_profile(caps)
    assert profile is not None
    assert profile["robot_mass_kg"] == pytest.approx(17.0)
    assert profile["rolling_resistance_crr"] == pytest.approx(0.035)


def test_power_profile_follows_a_power_override():
    view = _jackal()
    profile = view.power_profile(view.effective_caps({}, caps={"power.power_system.rolling_resistance_crr": 0.05}))
    assert profile is not None
    assert profile["rolling_resistance_crr"] == pytest.approx(0.05)


def test_fleet_facts_come_from_the_body():
    facts = fleet._model_facts("jackal", frozenset())
    assert facts.base_frame == "base_link"
    assert facts.odom_topic == "jackal_velocity_controller/odom"


def test_snapshot_holds_the_format_model_request_effective_provenance_and_hash():
    caps = _jackal_caps({"mobile.radius": 0.5})
    snapshot = caps.snapshot()
    assert snapshot["format"] == "arena-caps/1"
    assert snapshot["model"] == "jackal"
    assert snapshot["request"]["overrides"] == {"mobile.radius": 0.5}
    assert snapshot["request"]["parts"]["lidar@front"] == "jackal_default"
    assert snapshot["effective"]["mobile"]["radius"] == 0.5
    assert {"body", "power", "acoustics"} <= set(snapshot["effective"])
    assert snapshot["provenance"]["mobile.radius"] == "override"
    assert snapshot["hash"] == caps.caps_hash()
    assert yaml.safe_load(yaml.safe_dump(snapshot)) == snapshot


def test_hash_is_stable_across_two_resolves_and_changes_with_an_override():
    plain = _jackal_caps().caps_hash()
    assert RobotIdentifier("jackal").resolve_sync().effective_caps({}).caps_hash() == plain
    overridden = _jackal_caps({"mobile.radius": 0.5})
    assert overridden.caps_hash() != plain
    assert _jackal_caps({"mobile.radius": 0.5}).caps_hash() == overridden.caps_hash()


def test_hash_changes_with_a_fact_block_override():
    assert _jackal_caps({"power.power_system.rolling_resistance_crr": 0.05}).caps_hash() != _jackal_caps().caps_hash()


def test_provenance_of_an_override_path():
    assert _jackal_caps({"mobile.radius": 0.5}).provenance("mobile.radius") == "override"
    assert _jackal_caps().provenance("mobile.radius") == "file"


def test_provenance_of_a_component_path():
    assert _jackal_with_ur5e().provenance("arm.top.stall_time") == "component"


def test_provenance_of_the_derived_body_mass_total_and_parts():
    caps = _jackal_caps()
    assert caps.provenance("body.mass.total") == "derived"
    assert caps.provenance("body.mass.parts.lidar.front") == "derived"
    assert caps.provenance("body.mass.base") == "file"


def test_provenance_of_mobile_limits_derived_when_the_file_declares_none():
    caps = _jackal_caps()
    assert caps.provenance("mobile.limits.velocity.linear") == "derived"
    assert _jackal_caps({"mobile.limits.velocity.linear": [-0.5, 0.5]}).provenance("mobile.limits.velocity.linear") == "override"


def test_override_paths_list_the_fact_blocks_with_provenance():
    found = {item.path: item for item in entry_paths("jackal[power.power_system.rolling_resistance_crr=0.05]")}
    assert found["body.base_frame"].value == "base_link" and found["body.base_frame"].provenance == "file"
    assert found["body.mass.total"].provenance == "derived"
    assert found["power.power_system.rolling_resistance_crr"].value == pytest.approx(0.05)
    assert found["power.power_system.rolling_resistance_crr"].provenance == "override"
    assert "acoustics.L_base_0" in found


def test_body_mass_paths_are_not_overridable_and_stay_out_of_completion():
    from arena_robots.override_paths import complete

    found = {item.path: item for item in entry_paths("jackal")}
    assert not found["body.mass"].overridable
    assert not found["body.mass.base"].overridable
    assert found["body.base_frame"].overridable
    completions = complete("jackal[body.")
    assert "jackal[body.base_frame=" in completions
    assert not any(entry.startswith("jackal[body.mass") for entry in completions)
