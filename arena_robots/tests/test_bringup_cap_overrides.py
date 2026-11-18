from __future__ import annotations

import socket
import tempfile
from pathlib import Path

import launch
import pytest
import yaml
from arena_robots.bringup.arm.moveit import MoveItArmBringup
from arena_robots.bringup.mobile.nav2 import Nav2Bringup
from arena_robots.bringup.mobile.rosnav_rl import (
    agent_range_actions,
    agents_root,
    range_mismatches,
    saved_action_ranges,
)
from arena_robots.caps import MobileSpec
from arena_robots.nav2 import Nav2CollisionDerivedYAML, Nav2KinematicsDerivedYAML
from arena_robots.request import InstanceRequest
from arena_robots.Robot import RobotView

MOBILE = {
    "odom_frame": "odom",
    "footprint": [[0.2, 0.1], [0.2, -0.1], [-0.2, -0.1], [-0.2, 0.1]],
    "footprint_padding": 0.1,
    "velocity_limits": {"linear": [-2.0, 2.0], "angular": [-4.0, 4.0]},
    "acceleration_limits": {"linear": 2.5, "angular": 3.2},
}
ARM = {"arm": {"controller": "arm_controller", "stall_time": 1.0, "moveit": {"package": "ur_moveit_config"}}}
NAMESPACE = "/env_0/robot"


def _view(tmp_path: Path, **caps: dict) -> RobotView:
    robot_dir = tmp_path / "robot"
    (robot_dir / "caps").mkdir(parents=True)
    (robot_dir / "model_params.yaml").write_text(yaml.safe_dump({"robot_model": "robot"}))
    for cap, data in caps.items():
        (robot_dir / "caps" / f"{cap}.yaml").write_text(yaml.safe_dump(data))
    return RobotView(robot_dir)


def _derived(substitution_cls: type, mobile_file: Path) -> dict:
    return yaml.safe_load(Path(substitution_cls(str(mobile_file)).perform(launch.LaunchContext())).read_text())


def _launch_args(bringup: Nav2Bringup) -> dict[str, str]:
    return dict(bringup._launch_actions()[0].launch_arguments)


def _mobile_file(bringup: Nav2Bringup) -> Path:
    return Path(bringup.mobile_caps_launch_arguments()["mobile_caps"])


def test_bringup_caps_carry_the_request_overrides(tmp_path: Path) -> None:
    view = _view(tmp_path, mobile=MOBILE)
    bringup = Nav2Bringup(view, NAMESPACE, request=InstanceRequest(caps={"mobile.velocity_limits.linear": [-0.5, 1.0]}))
    assert bringup.caps.mobile.raw["velocity_limits"]["linear"] == [-0.5, 1.0]
    assert view.caps.mobile.raw["velocity_limits"]["linear"] == [-2.0, 2.0]


def test_bringup_without_overrides_reads_the_model_caps(tmp_path: Path) -> None:
    view = _view(tmp_path, mobile=MOBILE)
    assert Nav2Bringup(view, NAMESPACE).caps is view.caps


def test_two_bringups_of_one_model_hold_independent_caps(tmp_path: Path) -> None:
    view = _view(tmp_path, mobile=MOBILE)
    slow = Nav2Bringup(view, "/env_0/slow", request=InstanceRequest(caps={"mobile.velocity_limits.linear": {"scale": 0.25}}))
    fast = Nav2Bringup(view, "/env_0/fast")
    assert slow.caps.mobile.velocity_limits.linear.max == 0.5
    assert fast.caps.mobile.velocity_limits.linear.max == 2.0


def test_nav2_kinematics_follow_the_overridden_mobile_file(tmp_path: Path) -> None:
    bringup = Nav2Bringup(_view(tmp_path, mobile=MOBILE), NAMESPACE, request=InstanceRequest(caps={"mobile.velocity_limits.linear": [-0.5, 1.0], "mobile.acceleration_limits": {"scale": 0.5}}))
    mobile_file = _mobile_file(bringup)
    try:
        derived = _derived(Nav2KinematicsDerivedYAML, mobile_file)
    finally:
        mobile_file.unlink()
    assert (derived["min_linear_vel"], derived["max_linear_vel"]) == (-0.5, 1.0)
    assert (derived["min_angular_vel"], derived["max_angular_vel"]) == (-4.0, 4.0)
    assert (derived["linear_acc"], derived["angular_acc"]) == (1.25, 1.6)


def test_nav2_collision_follows_the_overridden_mobile_file(tmp_path: Path) -> None:
    view = _view(tmp_path, mobile=MOBILE)
    plain = _derived(Nav2CollisionDerivedYAML, view.path / "caps" / "mobile.yaml")
    bringup = Nav2Bringup(view, NAMESPACE, request=InstanceRequest(caps={"mobile.footprint_padding": 0.3}))
    mobile_file = _mobile_file(bringup)
    try:
        derived = _derived(Nav2CollisionDerivedYAML, mobile_file)
    finally:
        mobile_file.unlink()
    assert derived["footprint_padding"] == 0.3
    assert derived["inflation_radius"] == pytest.approx(plain["inflation_radius"] + 0.2)


def test_nav2_launch_passes_the_effective_mobile_file_when_mobile_is_overridden(tmp_path: Path) -> None:
    bringup = Nav2Bringup(_view(tmp_path, mobile=MOBILE), NAMESPACE, request=InstanceRequest(caps={"mobile.velocity_limits.linear": [-0.5, 1.0]}))
    mobile_file = Path(_launch_args(bringup)["mobile_caps"])
    try:
        assert yaml.safe_load(mobile_file.read_text()) == bringup.caps.mobile.raw
        assert mobile_file.name == f"mobile_caps_{socket.gethostname()}_env_0_robot.yaml"
        assert mobile_file.parent == Path(tempfile.gettempdir())
    finally:
        mobile_file.unlink()


def test_nav2_launch_passes_no_mobile_file_without_mobile_overrides(tmp_path: Path) -> None:
    bringup = Nav2Bringup(_view(tmp_path, mobile=MOBILE), NAMESPACE)
    assert "mobile_caps" not in _launch_args(bringup)
    assert bringup.mobile_caps_launch_arguments() == {}


def test_moveit_arms_carry_the_instance_overrides(tmp_path: Path) -> None:
    bringup = MoveItArmBringup(_view(tmp_path, arm=ARM), NAMESPACE, request=InstanceRequest(caps={"arm.arm.stall_time": 2.0}))
    assert bringup.arms()["arm"].stall_time == 2.0


def test_moveit_launch_carries_the_whole_request_as_json(tmp_path: Path) -> None:
    request = InstanceRequest(frames={"mount": "base_link"}, caps={"arm.arm.stall_time": 2.0})
    (action,) = MoveItArmBringup(_view(tmp_path, arm=ARM), NAMESPACE, request=request)._launch_actions()
    launch_args = dict(action.launch_arguments)
    assert "parts_json" not in launch_args
    parsed = InstanceRequest.loads(launch_args["request_json"])
    assert parsed == request
    assert parsed.caps_of(_view(tmp_path / "again", arm=ARM)).arm["arm"].stall_time == 2.0


def test_empty_request_json_parses_to_the_default_request() -> None:
    assert InstanceRequest.loads("{}") == InstanceRequest()


def _agent_config(agents: Path, name: str, action_space: dict) -> Path:
    config = agents / name / "training_config.yaml"
    config.parent.mkdir(parents=True)
    config.write_text(yaml.safe_dump({"agent_config": {"name": name, "action_space": action_space}}))
    return config


def _mobile(**limits: list[float]) -> MobileSpec:
    return MobileSpec(path=Path("/robot/caps/mobile.yaml"), raw={"velocity_limits": {"linear": [-2.0, 2.0], "angular": [-4.0, 4.0], **limits}})


DIFFERENTIAL = {"type": "differential_drive", "linear_range": [-2.0, 2.0], "angular_range": [-4.0, 4.0]}
OMNIDIRECTIONAL = {"type": "omnidirectional", "linear_range_x": [-1.0, 1.0], "linear_range_y": [-0.5, 0.5], "angular_range": [-2.0, 2.0]}


def test_agents_root_prefers_rosnav_agents_dir_over_the_data_dir() -> None:
    assert agents_root({"ROSNAV_AGENTS_DIR": "/a", "ARENA_DATA_DIR": "/d"}) == Path("/a")
    assert agents_root({"ARENA_DATA_DIR": "/d"}) == Path("/d/agents")
    assert agents_root({}) is None


def test_saved_ranges_of_a_differential_agent(tmp_path: Path) -> None:
    config = _agent_config(tmp_path, "a", DIFFERENTIAL)
    assert saved_action_ranges(config) == {"linear": (-2.0, 2.0), "angular": (-4.0, 4.0)}


def test_saved_ranges_of_an_omnidirectional_agent_carry_the_lateral_axis(tmp_path: Path) -> None:
    config = _agent_config(tmp_path, "a", OMNIDIRECTIONAL)
    assert saved_action_ranges(config) == {"linear": (-1.0, 1.0), "lateral": (-0.5, 0.5), "angular": (-2.0, 2.0)}


def test_saved_ranges_of_a_missing_config_name_the_file(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="training_config.yaml"):
        saved_action_ranges(tmp_path / "a" / "training_config.yaml")


def test_saved_ranges_of_a_discrete_agent_are_unavailable(tmp_path: Path) -> None:
    config = _agent_config(tmp_path, "a", {"type": "differential_drive", "discrete_actions": [{"linear": 0.3, "angular": 0.0}]})
    with pytest.raises(ValueError, match="no continuous mobile action ranges"):
        saved_action_ranges(config)


def test_matching_ranges_have_no_mismatch() -> None:
    assert range_mismatches({"linear": (-2.0, 2.0), "angular": (-4.0, 4.0)}, _mobile()) == []


def test_mismatch_names_the_differing_axes_only() -> None:
    found = range_mismatches({"linear": (-0.5, 1.0), "angular": (-4.0, 4.0)}, _mobile())
    assert len(found) == 1
    assert found[0].startswith("linear")
    assert "[-0.5, 1.0]" in found[0]
    assert "[-2.0, 2.0]" in found[0]


def test_lateral_range_defaults_to_the_linear_one() -> None:
    saved = {"linear": (-2.0, 2.0), "lateral": (-2.0, 2.0), "angular": (-4.0, 4.0)}
    assert range_mismatches(saved, _mobile()) == []
    assert [m.split(":")[0] for m in range_mismatches(saved, _mobile(lateral=[-1.0, 1.0]))] == ["lateral"]


def _message(actions: list) -> str:
    (action,) = actions
    return "".join(token.text for token in action.msg)


def test_a_differing_agent_gets_a_warning_naming_both_ranges(tmp_path: Path) -> None:
    _agent_config(tmp_path, "a", {**DIFFERENTIAL, "linear_range": [-0.5, 1.0]})
    message = _message(agent_range_actions("jackal", _mobile(), "a", {"ROSNAV_AGENTS_DIR": str(tmp_path)}))
    assert message.startswith("WARNING")
    assert "[-0.5, 1.0]" in message
    assert "[-2.0, 2.0]" in message


def test_a_matching_agent_logs_nothing(tmp_path: Path) -> None:
    _agent_config(tmp_path, "a", DIFFERENTIAL)
    assert agent_range_actions("jackal", _mobile(), "a", {"ROSNAV_AGENTS_DIR": str(tmp_path)}) == []


def test_a_missing_agent_config_logs_that_the_check_was_skipped(tmp_path: Path) -> None:
    message = _message(agent_range_actions("jackal", _mobile(), "absent", {"ROSNAV_AGENTS_DIR": str(tmp_path)}))
    assert "check skipped" in message
    assert "training_config.yaml" in message


def test_an_unset_agents_directory_logs_that_the_check_was_skipped() -> None:
    assert "check skipped" in _message(agent_range_actions("jackal", _mobile(), "a", {}))


def test_no_agent_name_has_nothing_to_check() -> None:
    assert agent_range_actions("jackal", _mobile(), "", {}) == []
