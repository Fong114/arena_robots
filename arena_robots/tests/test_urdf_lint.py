from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from arena_robots.Robot import RobotIdentifier, RobotView
from arena_robots.urdf_lint import Finding, lint

ROBOTS_DIR = Path(__file__).resolve().parents[1] / "robots"


def _box(name: str, mass: float, size: tuple[float, float, float] = (1.0, 0.5, 0.2), *, collision: bool = True) -> str:
    x, y, z = size
    tensor = (mass / 12 * (y * y + z * z), mass / 12 * (x * x + z * z), mass / 12 * (x * x + y * y))
    geometry = f'<collision><geometry><box size="{x} {y} {z}"/></geometry></collision>' if collision else ""
    return f'<link name="{name}"><inertial><mass value="{mass}"/><inertia ixx="{tensor[0]}" ixy="0" ixz="0" iyy="{tensor[1]}" iyz="0" izz="{tensor[2]}"/></inertial>{geometry}</link>'


def _wheel(name: str, *, round_collision: bool = True) -> str:
    shape = '<cylinder radius="0.1" length="0.05"/>' if round_collision else '<box size="0.2 0.2 0.05"/>'
    return f'<link name="{name}"><inertial><mass value="1"/><inertia ixx="0.003" ixy="0" ixz="0" iyy="0.003" iyz="0" izz="0.005"/></inertial><collision><geometry>{shape}</geometry></collision></link>'


def _joint(name: str, kind: str, parent: str, child: str, *, y: float = 0.0, limit: str | None = None) -> str:
    body = f"<limit {limit}/>" if limit is not None else ""
    return f'<joint name="{name}" type="{kind}"><parent link="{parent}"/><child link="{child}"/><origin xyz="0 {y} 0"/><axis xyz="0 1 0"/>{body}</joint>'


def _robot(
    tmp_path: Path,
    body: str,
    *,
    sensors: tuple[str, ...] = (),
    control: dict | None = None,
) -> RobotView:
    pytest.importorskip("xacro")
    name = "bot"
    root = tmp_path / name
    (root / "urdf").mkdir(parents=True)
    (root / "urdf" / f"{name}.urdf.xacro").write_text(f'<?xml version="1.0"?>\n<robot name="{name}" xmlns:xacro="http://www.ros.org/wiki/xacro">{body}</robot>\n')
    params: dict = {"robot_model": name, "base_frame": "base_link", "sensors": [{"name": f"s{i}", "type": "imu", "topic": f"${{namespace}}/s{i}", "frame": frame} for i, frame in enumerate(sensors)]}
    if control is not None:
        (root / "control.yaml").write_text(yaml.safe_dump(control))
        params["control"] = {"mode": "ros2_control", "controllers": [], "config": str(root / "control.yaml")}
    (root / "model_params.yaml").write_text(yaml.safe_dump(params))
    return RobotView(root)


def _errors(findings: list[Finding], check: str | None = None) -> list[Finding]:
    return [finding for finding in findings if finding.severity == "error" and (check is None or finding.check == check)]


def _diff_control(left: str = "lw", right: str = "rw") -> dict:
    return {
        "controller_manager": {"ros__parameters": {"drive": {"type": "diff_drive_controller/DiffDriveController"}}},
        "drive": {"ros__parameters": {"left_wheel_names": [left], "right_wheel_names": [right]}},
    }


def test_a_sound_robot_has_no_findings(tmp_path):
    view = _robot(tmp_path, _box("base_link", 10) + _wheel("lw_link") + _wheel("rw_link") + _joint("lw", "continuous", "base_link", "lw_link", y=0.3) + _joint("rw", "continuous", "base_link", "rw_link", y=-0.3), control=_diff_control())
    assert lint(view) == []


def test_a_tree_with_two_roots_and_a_missing_parent_is_a_structure_error(tmp_path):
    view = _robot(tmp_path, _box("base_link", 10) + _box("orphan", 1) + _box("lost", 1) + _joint("j", "fixed", "ghost", "lost"))
    messages = [finding.message for finding in _errors(lint(view), "structure")]
    assert any("ghost" in message for message in messages)
    assert any("2 root links" in message for message in messages)


def test_the_base_frame_must_be_a_link(tmp_path):
    view = _robot(tmp_path, _box("chassis", 10))
    assert any("base frame base_link" in finding.message for finding in _errors(lint(view), "structure"))


def test_a_moving_link_needs_mass_but_a_static_one_may_have_none(tmp_path):
    body = _box("base_link", 10) + _box("marker", 0, collision=False) + _box("arm", 0, collision=False) + _joint("fix", "fixed", "base_link", "marker") + _joint("turn", "revolute", "base_link", "arm", limit='lower="-1" upper="1" effort="5" velocity="1"')
    messages = [finding.message for finding in _errors(lint(_robot(tmp_path, body)), "inertia")]
    assert len(messages) == 1
    assert "arm" in messages[0]


def test_a_tensor_that_is_not_positive_definite_is_an_error(tmp_path):
    flat = '<link name="plate"><inertial><mass value="1"/><inertia ixx="0.1" ixy="0" ixz="0" iyy="0.1" iyz="0" izz="0"/></inertial></link>'
    view = _robot(tmp_path, _box("base_link", 10) + flat + _joint("fix", "fixed", "base_link", "plate"))
    assert any("not positive definite" in finding.message for finding in _errors(lint(view), "inertia"))


def test_principal_moments_must_satisfy_the_triangle_inequality(tmp_path):
    skew = '<link name="plate"><inertial><mass value="1"/><inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.5"/></inertial></link>'
    view = _robot(tmp_path, _box("base_link", 10) + skew + _joint("fix", "fixed", "base_link", "plate"))
    assert any("triangle inequality" in finding.message for finding in _errors(lint(view), "inertia"))


def test_a_trace_far_from_the_collision_shape_is_an_error(tmp_path):
    heavy = '<link name="cam"><inertial><mass value="1"/><inertia ixx="1" ixy="0" ixz="0" iyy="1" iyz="0" izz="1"/></inertial><collision><geometry><box size="0.1 0.1 0.1"/></geometry></collision></link>'
    body = _box("base_link", 10) + heavy + _joint("fix", "fixed", "base_link", "cam")
    assert any("collision" in finding.message for finding in _errors(lint(_robot(tmp_path, body)), "inertia"))


def test_a_wheel_without_a_round_collision_is_an_error(tmp_path):
    body = _box("base_link", 10) + _wheel("lw_link", round_collision=False) + _wheel("rw_link") + _joint("lw", "continuous", "base_link", "lw_link", y=0.3) + _joint("rw", "continuous", "base_link", "rw_link", y=-0.3)
    messages = [finding.message for finding in _errors(lint(_robot(tmp_path, body, control=_diff_control())), "wheels")]
    assert messages == ["wheel joint lw has no round collision on its child link"]


def test_a_wheel_joint_must_be_continuous_and_present(tmp_path):
    body = _box("base_link", 10) + _wheel("lw_link") + _joint("lw", "fixed", "base_link", "lw_link", y=0.3)
    messages = [finding.message for finding in _errors(lint(_robot(tmp_path, body, control=_diff_control())), "wheels")]
    assert "wheel joint lw is fixed, expected continuous" in messages
    assert "wheel joint rw is not in the URDF" in messages


@pytest.mark.parametrize(
    ("limit", "fragment"),
    [
        (None, "has no limit"),
        ('lower="-1" upper="1" velocity="1"', "effort None"),
        ('lower="-1" upper="1" effort="5" velocity="0"', "velocity 0.0"),
        ('lower="1" upper="-1" effort="5" velocity="1"', "exceeds upper"),
    ],
)
def test_a_revolute_joint_needs_a_complete_limit(tmp_path, limit, fragment):
    view = _robot(tmp_path, _box("base_link", 10) + _box("arm", 1) + _joint("turn", "revolute", "base_link", "arm", limit=limit))
    assert any(fragment in finding.message for finding in _errors(lint(view), "joints"))


def test_a_large_effort_is_not_an_error(tmp_path):
    body = _box("base_link", 10) + _box("arm", 1) + _joint("lift", "prismatic", "base_link", "arm", limit='lower="0" upper="1" effort="10000" velocity="1"')
    assert not _errors(lint(_robot(tmp_path, body)), "joints")


def test_a_sensor_frame_that_is_not_a_link_is_an_error(tmp_path):
    view = _robot(tmp_path, _box("base_link", 10), sensors=("base_link", "lidar_link"))
    assert [finding.message for finding in _errors(lint(view), "sensors")] == ["sensor frame lidar_link is not a link"]


def test_every_installed_robot_lints_without_errors(require_installed):
    pytest.importorskip("xacro")
    failures: list[str] = []
    for definition in sorted(ROBOTS_DIR.glob("*/model_params.yaml")):
        name = definition.parent.name
        try:
            require_installed(name)
        except pytest.skip.Exception:
            continue
        view = RobotIdentifier(name).resolve_sync()
        failures += [f"{name}: {finding.check}: {finding.message}" for finding in _errors(lint(view))]
    assert not failures, "\n".join(failures)
