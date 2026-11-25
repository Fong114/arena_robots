from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from arena_robots import ARENA_ROBOTS_DIR
from arena_robots.Robot import RobotView

ROBOTS_DIR = Path(ARENA_ROBOTS_DIR) / "robots"
SHIPPED = sorted(path.parent for path in ROBOTS_DIR.glob("*/model_params.yaml"))

WHEELS = {"fl": (0.3, 0.2), "fr": (0.3, -0.2), "rl": (-0.2, 0.2), "rr": (-0.2, -0.2)}
URDF = (
    '<robot name="bot" xmlns:xacro="http://www.ros.org/wiki/xacro"><link name="base_link"/>'
    + "".join(f'<link name="{name}_link"><collision><geometry><cylinder radius="0.1" length="0.05"/></geometry></collision></link><joint name="{name}" type="continuous"><parent link="base_link"/><child link="{name}_link"/><origin xyz="{x} {y} 0"/><axis xyz="0 1 0"/></joint>' for name, (x, y) in WHEELS.items())
    + "</robot>"
)
CONTROL = {
    "controller_manager": {"ros__parameters": {"update_rate": 50, "joint_state_broadcaster": {"type": "joint_state_broadcaster/JointStateBroadcaster"}, "drive": {"type": "diff_drive_controller/DiffDriveController"}}},
    "drive": {"ros__parameters": {"left_wheel_names": ["fl", "rl"], "right_wheel_names": ["fr", "rr"]}},
}


def _robot(tmp_path: Path, *, mobile: dict | None = None, control: dict | None = None, caps: dict | None = None, control_file: dict | None = CONTROL, urdf: bool = True) -> RobotView:
    root = tmp_path / "bot"
    (root / "urdf").mkdir(parents=True)
    if urdf:
        (root / "urdf" / "bot.urdf.xacro").write_text(URDF)
    params: dict = {"robot_model": "bot", "base_frame": "base_link"}
    if control_file is not None:
        (root / "control.yaml").write_text(yaml.safe_dump(control_file))
        params["control"] = {"mode": "ros2_control", "config": str(root / "control.yaml"), **(control or {})}
    (root / "model_params.yaml").write_text(yaml.safe_dump(params))
    (root / "caps.yaml").write_text(yaml.safe_dump({"mobile": {"odom_frame": "odom", **(mobile or {})}, **(caps or {})}))
    return RobotView(root)


@pytest.fixture
def xacro() -> None:
    pytest.importorskip("xacro", reason="xacro is not importable, run under the Arena container")


@pytest.mark.parametrize(("side", "expected"), [("front", 0.3), ("rear", -0.2), (0.15, 0.15)])
def test_the_pivot_is_a_wheel_axle_of_the_urdf_or_a_number(tmp_path, xacro, side, expected):
    view = _robot(tmp_path, mobile={"planar": {"pivot": side}})
    caps = view.effective_caps({})
    assert caps.mobile.planar.pivot == pytest.approx(expected)
    assert caps.provenance("mobile.planar.pivot") == ("file" if isinstance(side, float) else "urdf")


def test_a_planar_block_without_a_pivot_pivots_about_the_origin(tmp_path, xacro):
    assert _robot(tmp_path, mobile={"planar": {"gain": {"linear": 1.0}}}).effective_caps({}).mobile.planar.pivot == 0.0


def test_an_override_replaces_the_derived_pivot(tmp_path, xacro):
    view = _robot(tmp_path, mobile={"planar": {"pivot": "front"}})
    assert view.effective_caps({}, caps={"mobile.planar.pivot": 0.05}).mobile.planar.pivot == 0.05


def test_an_unknown_pivot_side_names_the_choices(tmp_path, xacro):
    with pytest.raises(ValueError, match="front"):
        _robot(tmp_path, mobile={"planar": {"pivot": "middle"}}).effective_caps({}).mobile


def test_a_pivot_side_needs_a_drive_controller(tmp_path, xacro):
    with pytest.raises(ValueError, match="drive controller"):
        _robot(tmp_path, mobile={"planar": {"pivot": "front"}}, control_file=None).effective_caps({}).mobile


def test_the_view_measures_the_wheel_axles_in_the_base_frame(tmp_path):
    assert _robot(tmp_path).wheel_axles(URDF) == [-0.2, -0.2, 0.3, 0.3]


def test_the_view_hands_the_drive_controller_the_base_frame(tmp_path):
    _, _, params = _robot(tmp_path).drive_controller()
    assert params["base_frame_id"] == "base_link"


def test_continuous_actions_default_to_the_velocity_limits(tmp_path):
    mobile = _robot(tmp_path, mobile={"velocity_limits": {"linear": [-1.0, 1.5], "angular": {"min": -2.0, "max": 2.0}}, "actions": {"discrete": []}}).effective_caps({}).mobile
    assert (mobile.actions.continuous.linear.min, mobile.actions.continuous.linear.max) == (-1.0, 1.5)
    assert (mobile.actions.continuous.angular.min, mobile.actions.continuous.angular.max) == (-2.0, 2.0)
    assert mobile.actions.continuous.lateral is None
    assert mobile.raw["actions"]["continuous"]["linear"] == {"min": -1.0, "max": 1.5}


def test_authored_continuous_actions_win_over_the_velocity_limits(tmp_path):
    authored = {"linear": {"min": -0.3, "max": 0.3}, "angular": {"min": -0.5, "max": 0.5}}
    mobile = _robot(tmp_path, mobile={"velocity_limits": {"linear": [-1.0, 1.0], "angular": [-2.0, 2.0]}, "actions": {"continuous": authored}}).effective_caps({}).mobile
    assert mobile.raw["actions"]["continuous"] == authored


def test_the_control_spec_takes_controllers_and_topics_from_the_control_config(tmp_path):
    spec = _robot(tmp_path).model_params.control
    assert sorted(spec.controllers) == ["drive", "joint_state_broadcaster"]
    assert (spec.cmd_vel_topic, spec.odom_topic) == ("drive/cmd_vel", "drive/odom")


def test_a_mecanum_controller_uses_its_reference_and_odometry_topics(tmp_path):
    control = {"controller_manager": {"ros__parameters": {"base": {"type": "mecanum_drive_controller/MecanumDriveController"}}}, "base": {"ros__parameters": {}}}
    spec = _robot(tmp_path, control_file=control).model_params.control
    assert (spec.cmd_vel_topic, spec.odom_topic) == ("base/reference", "base/odometry")


def test_authored_control_entries_win_over_the_derived_ones(tmp_path):
    spec = _robot(tmp_path, control={"controllers": ["drive"], "cmd_vel_topic": "cmd_vel_stamped"}).model_params.control
    assert (spec.controllers, spec.cmd_vel_topic, spec.odom_topic) == (("drive",), "cmd_vel_stamped", "drive/odom")


def test_a_control_config_without_a_drive_controller_leaves_the_default_topics(tmp_path):
    control = {"controller_manager": {"ros__parameters": {"group": {"type": "joint_trajectory_controller/JointTrajectoryController"}}}, "group": {"ros__parameters": {}}}
    spec = _robot(tmp_path, control_file=control).model_params.control
    assert (spec.cmd_vel_topic, spec.odom_topic) == ("cmd_vel", "odom")


def test_the_body_facts_carry_the_derived_topics(tmp_path):
    body = _robot(tmp_path).effective_caps({}).body
    assert (body.cmd_vel_topic, body.odom_topic) == ("drive/cmd_vel", "drive/odom")


def test_an_acoustics_section_overrides_the_default_profile_key_by_key(tmp_path):
    caps = _robot(tmp_path, caps={"acoustics": {"L_base_0": 50.0}}).effective_caps({})
    assert caps.acoustics.raw["L_base_0"] == 50.0
    assert caps.acoustics.raw["max_joint_torque_nm"] == 15.0


def test_the_power_profile_passes_the_torque_clamp_of_the_acoustics_block(tmp_path, xacro):
    view = _robot(tmp_path, caps={"power": {"power_system": {}}, "acoustics": {"max_joint_torque_nm": 9.0}})
    assert view.power_profile(view.effective_caps({}))["max_joint_torque_nm"] == 9.0
    plain = _robot(tmp_path / "plain", caps={"power": {"power_system": {}}})
    assert plain.power_profile(plain.effective_caps({}))["max_joint_torque_nm"] == 15.0


@pytest.mark.parametrize("model", ["jackal", "husky", "rbsummit", "ridgeback"])
def test_a_skid_steer_robot_pivots_about_the_axle_its_file_names(model, xacro):
    view = RobotView(ROBOTS_DIR / model)
    side = yaml.safe_load((ROBOTS_DIR / model / "caps.yaml").read_text())["mobile"]["planar"]["pivot"]
    axles = view.wheel_axles(view.effective_caps({}).rendered_urdf())
    assert view.effective_caps({}).mobile.planar.pivot == (axles[0] if side == "rear" else axles[-1])


@pytest.mark.parametrize("path", SHIPPED, ids=lambda path: path.name)
def test_a_model_params_file_repeats_nothing_its_control_config_or_assembly_states(path):
    params = yaml.safe_load((path / "model_params.yaml").read_text())
    assert "sensors" not in params
    assert not {"controllers", "odom_topic"} & set(params.get("control", {}))
    assert params.get("control", {}).get("cmd_vel_topic", "cmd_vel_stamped") == "cmd_vel_stamped"


@pytest.mark.parametrize("path", sorted(ROBOTS_DIR.glob("*/caps.yaml")), ids=lambda path: path.parent.name)
def test_a_caps_file_authors_no_copy_of_what_is_derived_or_defaulted(path):
    data = yaml.safe_load(path.read_text())
    mobile = data["mobile"]
    assert mobile.get("planar", {}).get("pivot", "front") != 0.0
    continuous = mobile.get("actions", {}).get("continuous")
    envelope = {axis: [bound["min"], bound["max"]] for axis, bound in (continuous or {}).items()}
    limits = {axis: list(bound.values()) if isinstance(bound, dict) else list(bound) for axis, bound in mobile.get("velocity_limits", {}).items()}
    assert continuous is None or envelope != limits
    assert "robot_model" not in data.get("power", {})
    assert "max_joint_torque_nm" not in data.get("acoustics", {})


def test_the_sick_s300_angle_default_is_the_declared_arg_and_a_request_replaces_it():
    from arena_robots.assembly import RequestPart, resolve
    from arena_robots.catalog import render_wrapper_xacro

    view = RobotView(ROBOTS_DIR / "mpo700")
    default = render_wrapper_xacro(view, resolve(view.assembly, {}))
    assert 'min_angle="-2.356194490192345"' in default
    assert 'max_angle="2.356194490192345"' in default
    placed = resolve(view.assembly, {}).placements[0]
    request = {"lidar": [RequestPart(variant="sick_s300", mount=placed.mount.name, params={"max_angle": 2.1})]}
    assert 'max_angle="2.1"' in render_wrapper_xacro(view, resolve(view.assembly, request))
