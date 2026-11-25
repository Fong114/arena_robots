from __future__ import annotations

import logging
import math
import struct
from pathlib import Path

import pytest
import yaml
from arena_robots import urdf_facts
from arena_robots.urdf_facts import body_height, collision_outline, drive_controller, drive_geometry, joint_limits, laser_block, total_mass, wheelbase

PKG = Path(__file__).resolve().parents[2] / "arena_robots"
ROBOTS_DIR = PKG / "robots"
GOLDEN_DIR = Path(__file__).parent / "golden"
AUTHORED_LASERS = yaml.safe_load((GOLDEN_DIR / "mobile_laser_blocks.yaml").read_text())


def _golden(robot: str) -> str:
    return (GOLDEN_DIR / f"{robot}_default.urdf").read_text()


def _contains(polygon: list[tuple[float, float]], point: tuple[float, float]) -> bool:
    count = len(polygon)
    for i in range(count):
        ax, ay = polygon[i]
        bx, by = polygon[(i + 1) % count]
        if (bx - ax) * (point[1] - ay) - (by - ay) * (point[0] - ax) < -1e-9:
            return False
    return True


def _is_counterclockwise_convex(polygon: list[tuple[float, float]]) -> bool:
    count = len(polygon)
    for i in range(count):
        ax, ay = polygon[i]
        bx, by = polygon[(i + 1) % count]
        cx, cy = polygon[(i + 2) % count]
        if (bx - ax) * (cy - by) - (by - ay) * (cx - bx) <= 0:
            return False
    return True


def _write_binary_stl(path: Path, triangles: list[list[tuple[float, float, float]]]) -> None:
    data = struct.pack("<80sI", b"", len(triangles))
    for triangle in triangles:
        data += struct.pack("<12fH", 0, 0, 1, *[c for vertex in triangle for c in vertex], 0)
    path.write_bytes(data)


def _write_ascii_stl(path: Path, triangles: list[list[tuple[float, float, float]]]) -> None:
    lines = ["solid test"]
    for triangle in triangles:
        lines.append("facet normal 0 0 1")
        lines.append("outer loop")
        lines.extend(f"vertex {x} {y} {z}" for x, y, z in triangle)
        lines.append("endloop")
        lines.append("endfacet")
    lines.append("endsolid test")
    path.write_text("\n".join(lines))


def _urdf_with_mesh(mesh: Path, scale: str = "1 1 1") -> str:
    return f"""<robot name="probe">
  <link name="base_link">
    <collision><geometry><box size="1 0.5 0.2"/></geometry></collision>
  </link>
  <link name="mesh_link">
    <collision><geometry><mesh filename="file://{mesh}" scale="{scale}"/></geometry></collision>
  </link>
  <link name="arm_link">
    <collision><geometry><box size="2 2 2"/></geometry></collision>
  </link>
  <joint name="mesh_joint" type="fixed">
    <parent link="base_link"/><child link="mesh_link"/><origin xyz="1 0 0" rpy="0 0 0"/>
  </joint>
  <joint name="arm_joint" type="revolute">
    <parent link="base_link"/><child link="arm_link"/><origin xyz="0 5 0" rpy="0 0 0"/>
    <axis xyz="0 0 1"/><limit lower="-1" upper="1" velocity="1" effort="1"/>
  </joint>
</robot>
"""


_STRIP = [[(0.0, -0.1, 0.0), (0.5, -0.1, 0.0), (0.5, 0.1, 0.0)], [(0.0, -0.1, 0.0), (0.5, 0.1, 0.0), (0.0, 0.1, 0.0)]]


def test_jackal_joint_limits_cover_every_joint_and_leave_wheel_velocity_open() -> None:
    limits = joint_limits(_golden("jackal"))
    assert len(limits) == 13
    assert limits["front_left_wheel_joint"].velocity is None
    assert limits["base_link_joint"].effort is None


def test_tiago_wheel_joint_declares_velocity() -> None:
    assert joint_limits(_golden("tiago"))["wheel_left_joint"].velocity == pytest.approx(100.0)


def test_tiago_moveit_joint_limits_take_the_urdf_limits_and_keep_the_authored_acceleration() -> None:
    from arena_robots.caps import moveit_joint_limits
    from arena_robots.Robot import RobotIdentifier

    limits = joint_limits(_golden("tiago"))
    arm = RobotIdentifier("tiago").resolve_sync().caps.arm["arm"]
    moveit = moveit_joint_limits(arm.raw, lambda: limits)
    for name in ("arm_1_joint",):
        assert moveit[name]["max_velocity"] == pytest.approx(limits[name].velocity)
        assert moveit[name]["max_effort"] == pytest.approx(limits[name].effort)
        assert moveit[name]["min_position"] == pytest.approx(limits[name].lower)
        assert moveit[name]["max_position"] == pytest.approx(limits[name].upper)
        assert moveit[name]["max_acceleration"] == pytest.approx(2.0)


def test_jackal_laser_block_equals_the_formerly_authored_block() -> None:
    expected = AUTHORED_LASERS["jackal"]
    block = laser_block(_golden("jackal"))
    assert block is not None
    assert block["num_beams"] == expected["num_beams"] == 640
    assert block["range"] == pytest.approx(12.0)
    assert block["update_rate"] == pytest.approx(10.0)
    assert block["angle"]["min"] == pytest.approx(-3.14159)
    assert block["angle"]["max"] == pytest.approx(3.14159)
    assert block["angle"]["increment"] == pytest.approx(2 * math.pi / 640)
    assert block["angle"]["increment"] == pytest.approx(expected["angle"]["increment"], abs=1e-8)


def test_tiago_laser_block_uses_partial_fan_increment() -> None:
    expected = AUTHORED_LASERS["tiago"]
    block = laser_block(_golden("tiago"))
    assert block is not None
    assert block["num_beams"] == 401
    assert block["range"] == pytest.approx(25.0)
    assert block["angle"]["increment"] == pytest.approx((1.74533 * 2) / 400)
    assert block["angle"]["increment"] == pytest.approx(expected["angle"]["increment"], abs=1e-8)


def test_rbkairos_laser_block_selects_the_front_frame() -> None:
    expected = AUTHORED_LASERS["rbkairos"]
    block = laser_block(_golden("rbkairos"), "robot_front_laser_link")
    assert block is not None
    assert block["num_beams"] == expected["num_beams"] == 541
    assert block["range"] == pytest.approx(expected["range"])
    assert block["update_rate"] == pytest.approx(expected["update_rate"])
    assert block["angle"]["increment"] == pytest.approx(expected["angle"]["increment"], abs=1e-7)


def test_laser_block_of_unknown_frame_is_none() -> None:
    assert laser_block(_golden("jackal"), "no_such_link") is None


def test_total_mass_sums_link_inertials() -> None:
    assert total_mass(_golden("jackal")) == pytest.approx(17.101, abs=1e-3)
    assert total_mass(_golden("boxer")) == pytest.approx(107.0, abs=1e-3)


def _drive(robot: str) -> tuple[str, dict]:
    control = yaml.safe_load((ROBOTS_DIR / robot / "control.yaml").read_text())
    drive = drive_controller(control)
    assert drive is not None
    return drive[1], drive[2]


def _geometry(robot: str) -> dict:
    kind, params = _drive(robot)
    return drive_geometry(_golden(robot), kind, params)


def test_drive_controller_is_the_listed_wheel_controller() -> None:
    control = yaml.safe_load((ROBOTS_DIR / "jackal" / "control.yaml").read_text())
    name, kind, params = drive_controller(control)
    assert (name, kind) == ("jackal_velocity_controller", "diff_drive_controller/DiffDriveController")
    assert params["left_wheel_names"] == ["front_left_wheel_joint", "rear_left_wheel_joint"]


@pytest.mark.parametrize("robot", ("a1", "go1"))
def test_a_legged_robot_has_no_drive_controller(robot: str) -> None:
    assert drive_controller(yaml.safe_load((ROBOTS_DIR / robot / "control.yaml").read_text())) is None


DERIVED_KEYS = frozenset({"wheel_radius", "wheel_separation", "wheel_positions_x", "wheel_positions_y", "wheel_angles", "kinematics.wheels_radius", "kinematics.sum_of_robot_center_projection_on_X_Y_axis", "wheelbase", "traction_track_width", "steering_track_width", "traction_wheels_radius"})
WHEELED = sorted(path.name.removesuffix("_default.urdf") for path in GOLDEN_DIR.glob("*_default.urdf") if path.name not in ("a1_default.urdf", "go1_default.urdf"))


@pytest.mark.parametrize("robot", WHEELED)
def test_a_shipped_control_file_declares_no_derived_geometry(robot: str) -> None:
    _, params = _drive(robot)
    assert DERIVED_KEYS.isdisjoint(params)


@pytest.mark.parametrize("robot", WHEELED)
def test_every_shipped_drive_controller_has_derivable_geometry(robot: str) -> None:
    assert set(_geometry(robot)) <= DERIVED_KEYS


def test_jackal_diff_drive_geometry_is_its_wheel_joints() -> None:
    assert _geometry("jackal") == {"wheel_radius": 0.098, "wheel_separation": 0.37559}


def test_turtlebot_wheel_radius_is_the_collision_radius() -> None:
    assert _geometry("turtlebot")["wheel_radius"] == pytest.approx(0.03575)


def test_rbkairos_mecanum_projection_is_half_wheelbase_plus_half_track() -> None:
    assert _geometry("rbkairos") == {"kinematics.wheels_radius": 0.127, "kinematics.sum_of_robot_center_projection_on_X_Y_axis": 0.47428}


def test_rbrobout_mecanum_projection() -> None:
    assert _geometry("rbrobout")["kinematics.sum_of_robot_center_projection_on_X_Y_axis"] == pytest.approx(0.8445)


def test_rbvogui_swerve_positions_are_the_steering_joints() -> None:
    assert _geometry("rbvogui") == {"wheel_radius": 0.1165, "wheel_positions_x": [0.368, 0.368, -0.368, -0.368], "wheel_positions_y": [-0.235, 0.235, -0.235, 0.235]}


def test_mpo700_swerve_positions_are_the_caster_pivots_not_the_offset_wheels() -> None:
    geometry = _geometry("mpo700")
    assert geometry["wheel_positions_x"] == [0.24, 0.24, -0.24, -0.24]
    assert geometry["wheel_positions_y"] == [-0.18, 0.18, -0.18, 0.18]


def test_rskomnidirectional_omni_geometry_is_the_wheel_joint_frames() -> None:
    geometry = _geometry("rskomnidirectional")
    assert geometry["wheel_radius"] == pytest.approx(0.0345)
    assert geometry["wheel_positions_x"] == pytest.approx([0.0235875, 0.0235875, -0.0472125], abs=1e-6)
    assert geometry["wheel_positions_y"] == pytest.approx([0.0408763, -0.0408765, 0.0], abs=1e-6)
    assert geometry["wheel_angles"] == pytest.approx([5 * math.pi / 6, math.pi / 6, 3 * math.pi / 2], abs=1e-5)


def test_a_controller_without_a_base_frame_measures_in_the_root_link() -> None:
    kind, params = _drive("jackal")
    assert drive_geometry(_golden("jackal"), kind, {key: value for key, value in params.items() if key != "base_frame_id"}) == _geometry("jackal")


def test_a_base_frame_that_does_not_hold_the_wheels_raises() -> None:
    kind, params = _drive("jackal")
    with pytest.raises(ValueError, match="not below the base frame"):
        drive_geometry(_golden("jackal"), kind, {**params, "base_frame_id": "front_left_wheel_link"})


def test_a_missing_wheel_joint_raises() -> None:
    kind, params = _drive("jackal")
    with pytest.raises(ValueError, match="missing_joint"):
        drive_geometry(_golden("jackal"), kind, {**params, "left_wheel_names": ["missing_joint"]})


def test_wheels_of_different_radii_raise() -> None:
    kind, params = _drive("jackal")
    urdf = _golden("jackal").replace('radius="0.098"', 'radius="0.1"', 1)
    with pytest.raises(ValueError, match="disagree on the wheel radius"):
        drive_geometry(urdf, kind, params)


def test_mecanum_wheels_that_disagree_on_the_projection_raise() -> None:
    kind, params = _drive("rbkairos")
    urdf = _golden("rbkairos").replace('xyz="-0.21528 -0.259 0.0"', 'xyz="-0.31528 -0.259 0.0"', 1)
    with pytest.raises(ValueError, match="center projection"):
        drive_geometry(urdf, kind, params)


def test_jackal_outline_is_a_small_convex_polygon_around_the_origin() -> None:
    outline = collision_outline(_golden("jackal"), "base_link")
    assert outline is not None
    assert 3 <= len(outline) <= 8
    assert _is_counterclockwise_convex(outline)
    assert _contains(outline, (0.0, 0.0))
    xs = [x for x, _ in outline]
    ys = [y for _, y in outline]
    assert min(xs) <= -0.21
    assert max(xs) >= 0.21
    assert min(ys) <= -0.155
    assert max(ys) >= 0.155


def test_outline_includes_fixed_links_and_stl_and_excludes_links_behind_revolute_joints(tmp_path: Path) -> None:
    mesh = tmp_path / "strip.stl"
    _write_binary_stl(mesh, _STRIP)
    outline = collision_outline(_urdf_with_mesh(mesh), "base_link")
    assert outline is not None
    xs = [x for x, _ in outline]
    ys = [y for _, y in outline]
    assert max(xs) == pytest.approx(1.5)
    assert min(xs) == pytest.approx(-0.5)
    assert max(ys) == pytest.approx(0.25)
    assert min(ys) == pytest.approx(-0.25)


def test_outline_reads_ascii_stl_and_honors_mesh_scale(tmp_path: Path) -> None:
    mesh = tmp_path / "strip.stl"
    _write_ascii_stl(mesh, _STRIP)
    outline = collision_outline(_urdf_with_mesh(mesh, "2 1 1"), "base_link")
    assert outline is not None
    assert max(x for x, _ in outline) == pytest.approx(2.0)


def test_outline_skips_unknown_mesh_format_with_a_warning(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    mesh = tmp_path / "shape.obj"
    mesh.write_text("v 0 0 0\n")
    logger = logging.getLogger("arena_robots.urdf_facts")
    logger.addHandler(caplog.handler)
    try:
        with caplog.at_level("WARNING", logger=logger.name):
            outline = collision_outline(_urdf_with_mesh(mesh), "base_link")
    finally:
        logger.removeHandler(caplog.handler)
    assert outline is not None
    assert max(x for x, _ in outline) == pytest.approx(0.5)
    assert "shape.obj" in caplog.text


def test_outline_reduces_a_32_gon_to_at_most_eight_vertices(tmp_path: Path) -> None:
    ring = [(math.cos(2 * math.pi * i / 32), math.sin(2 * math.pi * i / 32), 0.0) for i in range(32)]
    triangles = [[(0.0, 0.0, 0.0), ring[i], ring[(i + 1) % 32]] for i in range(32)]
    mesh = tmp_path / "ring.stl"
    _write_ascii_stl(mesh, triangles)
    urdf = f'<robot name="ring"><link name="base_link"><collision><geometry><mesh filename="file://{mesh}"/></geometry></collision></link></robot>'
    outline = collision_outline(urdf, "base_link", max_vertices=8)
    assert outline is not None
    assert len(outline) == 8
    assert _is_counterclockwise_convex(outline)


def test_outline_without_collision_geometry_is_none() -> None:
    assert collision_outline('<robot name="empty"><link name="base_link"/></robot>', "base_link") is None


def test_reduction_encloses_every_hull_vertex():
    hull = [(math.cos(2 * math.pi * k / 32), 0.6 * math.sin(2 * math.pi * k / 32)) for k in range(32)]
    polygon = urdf_facts._reduce(hull, 8)
    assert len(polygon) == 8
    for point in hull:
        assert all(urdf_facts._cross(polygon[i - 1], polygon[i], point) >= -1e-9 for i in range(len(polygon)))


_HEIGHT_URDF = """<robot name="r">
  <link name="base_link"><collision><origin xyz="0 0 0.15"/><geometry><box size="0.4 0.3 0.2"/></geometry></collision></link>
  <link name="wheel"><collision><origin rpy="1.5708 0 0"/><geometry><cylinder radius="0.05" length="0.04"/></geometry></collision></link>
  <link name="caster"><collision><geometry><sphere radius="0.04"/></geometry></collision></link>
  <link name="mast"><collision><origin xyz="0 0 0.3"/><geometry><box size="0.1 0.1 0.6"/></geometry></collision></link>
  <joint name="wheel_joint" type="continuous"><parent link="base_link"/><child link="wheel"/><axis xyz="0 1 0"/></joint>
  <joint name="caster_joint" type="fixed"><parent link="base_link"/><child link="caster"/><origin xyz="0.1 0 -0.06"/></joint>
  <joint name="mast_joint" type="revolute"><parent link="base_link"/><child link="mast"/><origin xyz="0 0 0.25"/><axis xyz="0 0 1"/><limit lower="-1" upper="1" effort="1" velocity="1"/></joint>
</robot>"""


def test_body_height_spans_the_ground_contact_to_the_highest_collision_of_every_link() -> None:
    assert body_height(_HEIGHT_URDF, "base_link") == pytest.approx(0.95)


def test_body_height_reaches_a_sensor_frame_above_every_collision() -> None:
    urdf = _HEIGHT_URDF.replace(
        "</robot>",
        '<link name="lidar_link"/><joint name="lidar_joint" type="fixed"><parent link="base_link"/><child link="lidar_link"/><origin xyz="0 0 1.2"/></joint><gazebo reference="lidar_link"><sensor name="lidar" type="gpu_lidar"/></gazebo></robot>',
    )
    assert body_height(urdf, "base_link") == pytest.approx(1.3)


def test_body_height_counts_a_sphere_by_its_radius_not_its_center() -> None:
    urdf = _HEIGHT_URDF.replace('<link name="mast">', '<link name="mast_unused">').replace("mast_joint", "mast_unused_joint")
    assert body_height(urdf, "base_link") == pytest.approx(0.25 - (-0.1))


def test_body_height_of_a_robot_without_collision_geometry_is_none() -> None:
    assert body_height('<robot name="r"><link name="base_link"/></robot>', "base_link") is None
    assert body_height(_HEIGHT_URDF, "missing") is None


def test_jackal_body_height_reaches_its_lidar_plane() -> None:
    height = body_height(_golden("jackal"), "base_link")
    assert height == pytest.approx(0.0635 + 0.284, abs=1e-3)


def test_wheelbase_is_the_front_to_rear_distance_of_the_wheel_joints() -> None:
    assert wheelbase(_golden("jackal"), *_drive("jackal")) == pytest.approx(0.262)
    assert wheelbase(_golden("rbvogui"), *_drive("rbvogui")) == pytest.approx(0.736)


def test_a_two_wheel_base_has_no_wheelbase() -> None:
    assert wheelbase(_golden("tiago"), *_drive("tiago")) is None
