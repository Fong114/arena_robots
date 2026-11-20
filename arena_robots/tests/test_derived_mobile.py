from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from arena_robots import ARENA_ROBOTS_DIR
from arena_robots.assembly import RequestPart
from arena_robots.request import InstanceRequest
from arena_robots.Robot import RobotIdentifier

ROBOTS_DIR = Path(ARENA_ROBOTS_DIR) / "robots"
DERIVED_KEYS = ("footprint", "radius", "is_holonomic", "sensor_frame")
HOLONOMIC = {"a1", "go1", "mpo700", "rbkairos", "rbrobout", "rbvogui", "rskomnidirectional"}
EXTENT_TOLERANCE = 5e-3

OUTLINES = [
    pytest.param("jackal", False, (-0.2553, 0.2553, -0.215, 0.215), 0.3338, id="diff"),
    pytest.param("rbkairos", True, (-0.457, 0.457, -0.465, 0.465), 0.584, id="mecanum"),
    pytest.param("mpo700", True, (-0.476, 0.476, -0.324, 0.324), 0.523, id="swerve"),
    pytest.param("rskomnidirectional", True, (-0.095, 0.063, -0.085, 0.085), 0.097, id="omni"),
    pytest.param("go1", True, (-0.188, 0.279, -0.047, 0.047), 0.279, id="legged"),
]


@pytest.fixture
def xacro() -> None:
    pytest.importorskip("xacro", reason="xacro is not importable, run under the Arena container")


@pytest.fixture
def ament() -> None:
    pytest.importorskip("ament_index_python", reason="the control config resolves package:// paths")


def _view(model: str):
    return RobotIdentifier(model).resolve_sync()


@pytest.mark.parametrize("path", sorted(ROBOTS_DIR.glob("*/caps.yaml")), ids=lambda path: path.parent.name)
def test_no_mobile_cap_file_authors_a_derived_key(path):
    assert not set(yaml.safe_load(path.read_text())["mobile"]) & set(DERIVED_KEYS)


@pytest.mark.parametrize("model", sorted(path.parent.name for path in ROBOTS_DIR.glob("*/caps.yaml")))
def test_holonomy_follows_the_drive_controller_or_the_gait(model, ament):
    assert _view(model).caps.holonomic() is (model in HOLONOMIC)


@pytest.mark.parametrize(("model", "holonomic", "extent", "circumradius"), OUTLINES)
def test_the_mobile_cap_derives_footprint_radius_and_holonomy(model, holonomic, extent, circumradius, xacro, ament):
    mobile = _view(model).effective_caps({}).mobile
    xs = [x for x, _ in mobile.footprint]
    ys = [y for _, y in mobile.footprint]
    assert (min(xs), max(xs), min(ys), max(ys)) == pytest.approx(extent, abs=EXTENT_TOLERANCE)
    assert mobile.radius == pytest.approx(circumradius, abs=EXTENT_TOLERANCE)
    assert mobile.radius == pytest.approx(max(x * x + y * y for x, y in mobile.footprint) ** 0.5, abs=1e-3)
    assert mobile.is_holonomic is holonomic


@pytest.mark.parametrize(("model", "holonomic"), [("rbrobout", True), ("rbsummit", False)])
def test_a_robot_whose_file_disagreed_with_its_controller_follows_the_controller(model, holonomic, xacro, ament):
    assert _view(model).effective_caps({}).mobile.is_holonomic is holonomic


def test_provenance_names_the_urdf_and_the_controller(xacro, ament):
    caps = _view("go1").effective_caps({})
    assert caps.provenance("mobile.footprint") == "urdf"
    assert caps.provenance("mobile.radius") == "urdf"
    assert caps.provenance("mobile.is_holonomic") == "derived"
    assert caps.provenance("mobile.footprint_padding") == "file"
    snapshot = caps.snapshot()
    assert snapshot["effective"]["mobile"]["radius"] == caps.mobile.radius
    assert snapshot["provenance"]["mobile.radius"] == "urdf"


def test_an_instance_override_wins_over_the_derived_values(xacro, ament):
    caps = _view("jackal").effective_caps({}, caps={"mobile.radius": 0.5, "mobile.is_holonomic": True, "mobile.footprint": [[0.1, 0.1], [0.1, -0.1], [-0.1, -0.1]]})
    assert caps.mobile.radius == 0.5
    assert caps.mobile.is_holonomic is True
    assert caps.mobile.footprint == [[0.1, 0.1], [0.1, -0.1], [-0.1, -0.1]]
    assert {caps.provenance(f"mobile.{key}") for key in ("radius", "is_holonomic", "footprint")} == {"override"}


def test_an_override_operation_applies_to_the_derived_value(xacro, ament):
    plain = _view("jackal").effective_caps({}).mobile.radius
    scaled = _view("jackal").effective_caps({}, caps={"mobile.radius": {"scale": 2.0}}).mobile.radius
    assert scaled == pytest.approx(2.0 * plain)


def test_the_tiago_torso_keeps_its_deliberate_limits_and_its_gripper_effort_is_the_urdf_effort(xacro):
    caps = _view("tiago").effective_caps({})
    assert caps.lift["torso"].limits == (0.0, 0.35)
    assert caps.provenance("lift.torso.limits") == "file"
    assert caps.gripper["gripper"].max_effort == 16.0
    assert caps.provenance("lift.torso.joint") == "file"
    assert caps.provenance("gripper.gripper.max_effort") == "urdf"


def test_a_placed_lift_takes_the_urdf_joint_range(xacro):
    caps = InstanceRequest(parts={"lift": [RequestPart(variant="ewellix_900mm", mount="lift0")]}).caps_of(_view("rbrobout"))
    assert caps.lift["lift0"].limits == pytest.approx((0.0, 0.9))
    assert caps.provenance("lift.lift0.limits") == "urdf"


def test_a_placed_gripper_takes_the_urdf_effort_unless_the_component_authors_it(xacro):
    robotiq = InstanceRequest(parts={"arm": [RequestPart(variant="ur5e", mount="top")], "gripper": [RequestPart(variant="robotiq_2f_85", mount="top_tool")]}).caps_of(_view("jackal"))
    assert robotiq.gripper["top_tool"].max_effort == 50.0
    assert robotiq.provenance("gripper.top_tool.max_effort") == "urdf"
    franka = InstanceRequest(parts={"arm": [RequestPart(variant="fr3", mount="top")], "gripper": [RequestPart(variant="franka_hand", mount="top_tool")]}).caps_of(_view("jackal"))
    assert franka.gripper["top_tool"].max_effort == 70.0
    assert franka.provenance("gripper.top_tool.max_effort") == "component"


def test_an_override_scales_the_urdf_default_effort(xacro):
    request = InstanceRequest(parts={"arm": [RequestPart(variant="ur5e", mount="top")], "gripper": [RequestPart(variant="robotiq_2f_85", mount="top_tool")]}, caps={"gripper.top_tool.max_effort": {"scale": 0.5}})
    caps = request.caps_of(_view("jackal"))
    assert caps.gripper["top_tool"].max_effort == pytest.approx(25.0)
    assert caps.provenance("gripper.top_tool.max_effort") == "override"
