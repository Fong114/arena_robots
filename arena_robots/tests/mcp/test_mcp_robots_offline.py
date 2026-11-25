from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

import pytest
import yaml

pytest.importorskip("arena_mcp")

from arena_mcp.core import ToolError, cli
from arena_robots import mcp
from arena_robots.caps import simulator_names
from arena_robots.Robot import RobotView

ROBOTS_DIR = Path(__file__).resolve().parents[2] / "robots"


def _model_param(model: str):
    preset = ROBOTS_DIR / model / "preset.yaml"
    if not preset.is_file():
        return model
    return pytest.param(model, marks=pytest.mark.requires_installed(yaml.safe_load(preset.read_text())["robot"], "arm/ur"))


def test_catalog_lists_every_robot_directory_with_its_caps():
    catalog = mcp.robots_catalog()
    assert [entry["model"] for entry in catalog] == sorted(path.parent.name for path in [*ROBOTS_DIR.glob("*/model_params.yaml"), *ROBOTS_DIR.glob("*/preset.yaml")])
    jackal = next(entry for entry in catalog if entry["model"] == "jackal")
    assert "error" not in jackal
    assert jackal["caps"] == ["mobile"]
    assert "laserscan" in jackal["sensor_types"]
    assert jackal["summary"].startswith("mobile (non-holonomic, radius 0.3338")
    assert "preset" not in jackal


@pytest.mark.requires_installed("rbkairos", "arm/ur")
def test_catalog_lists_a_preset_with_its_base_robot_and_parts():
    kairos = next(entry for entry in mcp.robots_catalog() if entry["model"] == "rbkairos_plus")
    assert "arm x1" in kairos["summary"]
    assert kairos["preset"] == {"robot": "rbkairos", "parts": {"arm": ["ur5e"]}, "caps": {}}


def test_jackal_definition_carries_limits_sensors_and_only_the_goto_endpoint():
    definition = mcp.robots_definition("jackal")
    authored = yaml.safe_load((ROBOTS_DIR / "jackal" / "caps.yaml").read_text())["mobile"]
    assert definition["mobile"]["radius"] == pytest.approx(0.3338, abs=2e-3)
    assert max(abs(x) for x, _ in definition["mobile"]["footprint"]) == pytest.approx(0.2553, abs=5e-3)
    assert "sensor_frame" not in definition["mobile"]
    limits = definition["mobile"]["velocity_limits"]
    assert [limits["linear"]["min"], limits["linear"]["max"]] == authored["velocity_limits"]["linear"]
    assert definition["mobile"]["planar"]["acceleration"]["lateral"] is None
    assert definition["control"]["cmd_vel_topic"] == "jackal_velocity_controller/cmd_vel"
    sensors = {sensor["name"]: sensor for sensor in definition["sensors"]}
    assert sensors["lidar"]["type"] == "laserscan"
    assert sensors["lidar"]["topic"].startswith("${namespace}/")
    assert set(definition["actions"]["endpoints"]) == {"goto_pose"}
    assert definition["actions"]["endpoints"]["goto_pose"]["action"] == "arena_robots_msgs/action/GotoPose"
    nav2 = definition["actions"]["bringups"]["mobile"]["nav2"]
    assert nav2["compatible"] and "goto_pose" in nav2["task_kinds"]
    assert "lidar" in definition["assembly"]["variants"]
    assert "jackal_default" in definition["assembly"]["variants"]["lidar"]


@pytest.mark.requires_installed("rbkairos", "arm/ur")
def test_arm_robot_definition_lists_named_poses_gestures_and_arm_endpoints():
    definition = mcp.robots_definition("rbkairos_plus")
    assert definition["model"] == "rbkairos_plus"
    assert definition["preset"]["robot"] == "rbkairos"
    arm = definition["arm"]["arm_mount"]
    assert arm["named_poses"] == ["ready", "stow", "wave_l", "wave_r", "wave_up"]
    assert arm["raw"]["controller"] == "arm_controller"
    assert arm["gestures"], "the shared gesture library is empty"
    assert all(gesture["playable"] == (not gesture["missing_poses"]) for gesture in arm["gestures"].values() if "error" not in gesture)
    assert set(definition["actions"]["endpoints"]) == {"goto_pose", "reach_pose", "play_gesture"}
    assert any(sensor["type"] == "image" for sensor in definition["sensors"])


@pytest.mark.parametrize("model", [_model_param(model) for model in sorted(path.parent.name for path in [*ROBOTS_DIR.glob("*/model_params.yaml"), *ROBOTS_DIR.glob("*/preset.yaml")])])
def test_every_model_resolves_to_a_json_definition(model):
    definition = mcp.robots_definition(model)
    assert definition["model"] == model and definition["caps"]
    json.dumps(definition, allow_nan=False)
    bringup_errors = {cap: {kind: entry["error"] for kind, entry in kinds.items() if "error" in entry} for cap, kinds in definition["actions"]["bringups"].items()}
    assert all(not errors for errors in bringup_errors.values()), bringup_errors


def test_unknown_model_names_the_known_models():
    with pytest.raises(ToolError) as error:
        mcp.robots_definition("no_such_robot")
    assert "jackal" in error.value.hint


def test_definition_findings_flag_a_broken_robot(tmp_path):
    robot = tmp_path / "broken"
    robot.mkdir()
    params = {
        "robot_model": "other",
        "base_frame": "base_link",
        "sensors": [{"name": "sonar", "type": "sonar", "topic": "/sonar", "frame": "base_link"}],
        "control": {"mode": "ros2_control", "controllers": []},
    }
    (robot / "model_params.yaml").write_text(yaml.safe_dump(params))
    findings = mcp._definition_findings(RobotView(robot))
    by_check = {(finding["check"], finding["severity"]) for finding in findings}
    assert ("model_params", "warning") in by_check
    assert ("caps", "info") in by_check
    messages = " ".join(finding["message"] for finding in findings)
    assert "'sonar'" in messages and "${namespace}" in messages
    assert "no controllers" in messages
    assert "control.yaml not found" in messages


def test_definition_findings_are_empty_for_jackal():
    assert mcp._definition_findings(mcp._view("jackal")) == []


def test_live_sensors_resolve_the_fleet_parts_into_namespaced_topics():
    entry = {"name": "jackal", "model": "jackal", "ns": "/arena/env_0/arena_env_node/jackal", "params": {"lidar@front": "jackal_default", "imu@imu": "jackal_default"}}
    sensors = mcp._live_sensors(entry)
    assert {sensor["type"] for sensor in sensors} >= {"laserscan", "imu"}
    assert all(sensor["topic"].startswith(entry["ns"] + "/") for sensor in sensors)


def test_live_sensors_name_the_robot_when_its_parts_do_not_resolve():
    entry = {"name": "scout", "model": "jackal", "ns": "/arena/env_0/arena_env_node/scout", "params": {"lidar@front": "no_such_lidar"}}
    with pytest.raises(ToolError, match=r"sensors of scout \(jackal\) do not resolve .*lidar/no_such_lidar"):
        mcp._live_sensors(entry)


@pytest.fixture
def fake_arena(tmp_path):
    arena = Path(os.environ["ARENA_DIR"]) if "ARENA_DIR" in os.environ else None
    if not cli.sourced() or arena is None or not (arena / "_meta" / "arena_cli").is_dir():
        pytest.skip("needs a shell that sourced arena")
    (tmp_path / "_meta").symlink_to(arena / "_meta")
    urdf = tmp_path / "arena_robots" / "arena_robots" / "robots" / "ghost" / "urdf"
    urdf.mkdir(parents=True)
    (urdf / "ghost.urdf").write_text('<robot name="ghost"><link name="base"><visual><geometry><mesh filename="package://arena_robots/robots/ghost/meshes/body.stl"/></geometry></visual></link></robot>\n')
    os.environ["ARENA_DIR"] = str(tmp_path)
    yield tmp_path
    os.environ["ARENA_DIR"] = str(arena)


def test_uri_check_turns_the_cli_failure_listing_into_findings(fake_arena):
    uri = mcp._uri_check(include_uninitialized=False)
    assert uri["missing"] == [{"owner": "ghost", "uri": "robots/ghost/meshes/body.stl", "sources": ["arena_robots/robots/ghost/urdf/ghost.urdf"]}]
    assert "feature robots check" in uri["command"]


def test_check_of_jackal_runs_the_cli_and_reports_no_definition_errors():
    if not cli.sourced():
        pytest.skip("needs a shell that sourced arena")
    report = mcp.robots_check("jackal")
    assert report["models"] == ["jackal"]
    assert [finding for finding in report["findings"] if finding["check"] != "mesh_uri"] == []
    assert all(finding["robot"] == "jackal" or "/" in finding["robot"] for finding in report["findings"])


def test_preflight_needs_models_or_all_and_names_the_known_models():
    with pytest.raises(ToolError, match="not both or neither"):
        asyncio.run(mcp.robots_preflight())
    with pytest.raises(ToolError, match="not both or neither"):
        asyncio.run(mcp.robots_preflight(["jackal"], all_models=True))
    with pytest.raises(ToolError, match="unknown robot models: ghost") as unknown:
        asyncio.run(mcp.robots_preflight(["ghost"]))
    assert "jackal" in unknown.value.hint


def test_lift_and_gripper_tools_expose_their_schemas():
    from mcp.server.mcpserver import MCPServer

    server = MCPServer(name="test")
    mcp.domain.install(server)
    tools = {tool.name: tool.input_schema for tool in asyncio.run(server.list_tools())}
    lift = tools["robots_lift"]
    assert lift["required"] == ["height"]
    assert set(lift["properties"]) == {"height", "env_id", "robot", "instance"}
    gripper = tools["robots_gripper"]
    assert "required" not in gripper
    assert set(gripper["properties"]) == {"env_id", "robot", "command", "position", "max_effort", "instance"}
    assert gripper["properties"]["max_effort"]["minimum"] == 0


def test_gripper_needs_exactly_one_of_command_and_position():
    for kwargs in ({}, {"command": "open", "position": 0.01}):
        with pytest.raises(ToolError, match="exactly one"):
            asyncio.run(mcp.robots_gripper(**kwargs))


def test_lift_rejects_a_non_finite_height():
    with pytest.raises(ToolError, match="finite"):
        asyncio.run(mcp.robots_lift(float("nan")))


def _entry(model: str, caps: list[str], params: dict[str, str] | None = None) -> dict:
    return {"name": model, "model": model, "ns": f"/env_0/{model}", "caps": [{"cap": cap, "adapter": "direct"} for cap in caps], "params": params or {}}


def test_actuator_instance_resolves_the_only_instance_of_the_cap():
    entry = _entry("tiago", ["lift", "gripper"])
    name, spec = mcp._actuator_spec(entry, "lift", "")
    assert (name, spec.controller) == ("torso", "torso_controller")
    assert mcp._actuator_spec(entry, "gripper", "gripper")[1].positions == {"open": 0.044, "close": 0.0}


def test_actuator_instance_resolves_a_component_placed_gripper_from_the_live_parts():
    entry = _entry("rbkairos", ["arm", "gripper"], {"arm@arm_mount": "fr3", "gripper@gripper_mount": "franka_hand"})
    name, spec = mcp._actuator_spec(entry, "gripper", "")
    assert (name, spec.joint, spec.positions) == ("gripper_mount", "robot_gripper_fr3_finger_joint1", {"open": 0.04, "close": 0.0})


def test_actuator_instance_errors_name_the_choices():
    entry = _entry("tiago", ["lift"])
    with pytest.raises(ToolError, match="no lift instance 'x'") as unknown:
        mcp._actuator_spec(entry, "lift", "x")
    assert "torso" in (unknown.value.hint or "")
    with pytest.raises(ToolError, match="no gripper cap"):
        mcp._actuator_spec(_entry("jackal", []), "gripper", "")


def test_definition_with_a_set_override_returns_the_overridden_limits_and_leaves_the_model_alone():
    definition = mcp.robots_definition("jackal", caps={"mobile.velocity_limits.linear": [-0.5, 1.0]})
    limits = definition["mobile"]["velocity_limits"]
    assert [limits["linear"]["min"], limits["linear"]["max"]] == [-0.5, 1.0]
    authored = yaml.safe_load((ROBOTS_DIR / "jackal" / "caps.yaml").read_text())["mobile"]
    assert definition["mobile"]["radius"] == pytest.approx(0.3338, abs=2e-3)
    plain = mcp.robots_definition("jackal")["mobile"]["velocity_limits"]
    assert [plain["linear"]["min"], plain["linear"]["max"]] == authored["velocity_limits"]["linear"]


def test_definition_with_a_scale_override_multiplies_the_authored_limits():
    authored = yaml.safe_load((ROBOTS_DIR / "jackal" / "caps.yaml").read_text())["mobile"]["velocity_limits"]["linear"]
    limits = mcp.robots_definition("jackal", caps={"mobile.velocity_limits.linear": {"scale": 0.5}})["mobile"]["velocity_limits"]
    assert [limits["linear"]["min"], limits["linear"]["max"]] == [authored[0] * 0.5, authored[1] * 0.5]


def test_definition_with_parts_resolves_the_instance_sensors():
    definition = mcp.robots_definition("jackal", parts={"lidar": "jackal_default"})
    assert any(sensor["type"] == "laserscan" for sensor in definition["sensors"])


def test_definition_without_instance_arguments_equals_an_empty_instance_request():
    assert mcp.robots_definition("jackal") == mcp.robots_definition("jackal", parts={}, frames={}, caps={})


def test_definition_rejects_an_unknown_override_path_with_a_hint():
    with pytest.raises(ToolError, match="unknown field 'velocty_limits'") as error:
        mcp.robots_definition("jackal", caps={"mobile.velocty_limits.linear": [0.0, 1.0]})
    assert error.value.hint


def test_definition_rejects_frames_for_a_mount_the_model_lacks():
    with pytest.raises(ToolError, match="unknown mount"):
        mcp.robots_definition("jackal", frames={"no_such_mount": "x"})


def test_definition_findings_report_an_unknown_override_path():
    view = mcp._view("jackal")
    request = mcp._request(view, None, None, {"mobile.velocty_limits.linear": [0.0, 1.0]})
    findings = mcp._definition_findings(view, request)
    assert any(finding["check"] == "caps.overrides" and "velocty_limits" in finding["message"] for finding in findings)


def test_definition_findings_are_empty_for_a_valid_override():
    view = mcp._view("jackal")
    assert mcp._definition_findings(view, mcp._request(view, None, None, {"mobile.velocity_limits.linear": [-0.5, 1.0]})) == []


def test_request_rejects_an_adapter_key_as_a_cap_override():
    with pytest.raises(ValueError, match="cap overrides must be"):
        mcp._request(mcp._view("jackal"), None, None, {"mobile.adapter": "direct"})


def test_check_with_an_instance_needs_a_model():
    with pytest.raises(ToolError, match="pass model"):
        mcp.robots_check(caps={"mobile.radius": 0.3})


def test_definition_findings_report_an_override_of_the_wrong_type():
    findings = mcp._definition_findings(mcp._view("jackal"), mcp._request(mcp._view("jackal"), None, None, {"mobile.velocity_limits.linear": "fast"}))
    assert any(finding["check"] == "caps.overrides" and finding["severity"] == "error" for finding in findings)


def test_actuator_spec_honors_a_lift_override_from_the_fleet_params():
    entry = _entry("tiago", ["lift"], {"cap:lift.torso.limits": "[0.0, 0.2]"})
    _, spec = mcp._actuator_spec(entry, "lift", "")
    assert spec.raw["limits"] == [0.0, 0.2]
    assert mcp._actuator_spec(_entry("tiago", ["lift"]), "lift", "")[1].raw["limits"] == [0.0, 0.35]


def test_actuator_spec_honors_a_gripper_override_from_the_fleet_params():
    entry = _entry("tiago", ["gripper"], {"cap:gripper.gripper.positions.open": "0.03"})
    assert mcp._actuator_spec(entry, "gripper", "")[1].positions == {"open": 0.03, "close": 0.0}


def test_overrides_list_the_jackal_mobile_paths_with_their_effective_values_as_json():
    paths = mcp.robots_overrides("jackal")
    json.dumps(paths, allow_nan=False)
    authored = yaml.safe_load((ROBOTS_DIR / "jackal" / "caps.yaml").read_text())["mobile"]
    by_path = {entry["path"]: entry for entry in paths}
    assert by_path["mobile.velocity_limits.linear"] == {"path": "mobile.velocity_limits.linear", "kind": "list", "value": authored["velocity_limits"]["linear"], "provenance": "file", "part": None, "relative": None, "simulators": sorted(simulator_names()), "overridable": True}
    assert any(entry["kind"] == "unset" and entry["value"] is None for entry in paths)


@pytest.mark.requires_installed("arm/ur")
def test_overrides_with_parts_mark_the_placed_part_and_its_bracket_path():
    paths = mcp.robots_overrides("jackal", parts={"top": "arm/ur5e[stall_time=2.0]"})
    by_path = {entry["path"]: entry for entry in paths}
    assert by_path["arm.top.stall_time"]["part"] == "top"
    assert by_path["arm.top.stall_time"]["relative"] == "stall_time"
    assert by_path["arm.top.stall_time"]["value"] == 2.0


def test_overrides_with_caps_show_the_overridden_value():
    by_path = {entry["path"]: entry for entry in mcp.robots_overrides("jackal", caps={"mobile.velocity_limits.linear": [-0.5, 1.0]})}
    assert by_path["mobile.velocity_limits.linear"]["value"] == [-0.5, 1.0]


def test_overrides_rejects_an_unknown_override_path_with_a_hint():
    with pytest.raises(ToolError, match="unknown field 'velocty_limits'") as error:
        mcp.robots_overrides("jackal", caps={"mobile.velocty_limits.linear": [0.0, 1.0]})
    assert error.value.hint


def test_overrides_of_an_unknown_model_names_the_known_models():
    with pytest.raises(ToolError) as error:
        mcp.robots_overrides("no_such_robot")
    assert "jackal" in error.value.hint


def test_definition_findings_carry_no_mass_check_for_a_robot_whose_files_disagreed():
    findings = mcp._definition_findings(mcp._view("turtlebot"))
    assert [finding for finding in findings if finding["check"] == "caps.mass"] == []


def test_definition_findings_are_silent_on_the_mass_when_both_files_agree():
    findings = mcp._definition_findings(mcp._view("jackal"))
    assert [finding for finding in findings if finding["check"] == "caps.mass"] == []
