from __future__ import annotations

import logging
import math
import struct
from pathlib import Path

import pytest
import yaml
from arena_robots import urdf_facts
from arena_robots.urdf_facts import collision_outline, joint_limits, laser_block, total_mass, wheel_geometry

PKG = Path(__file__).resolve().parents[2] / "arena_robots"
ROBOTS_DIR = PKG / "robots"
GOLDEN_DIR = Path(__file__).parent / "golden"


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


def test_tiago_arm_joint_limits_equal_moveit_limits() -> None:
    moveit = yaml.safe_load((ROBOTS_DIR / "tiago" / "srdf" / "joint_limits.yaml").read_text())["joint_limits"]
    limits = joint_limits(_golden("tiago"))
    for name in ("arm_1_joint",):
        assert limits[name].velocity == pytest.approx(moveit[name]["max_velocity"])
        assert limits[name].effort == pytest.approx(moveit[name]["max_effort"])
        assert limits[name].lower == pytest.approx(moveit[name]["min_position"])
        assert limits[name].upper == pytest.approx(moveit[name]["max_position"])


def test_jackal_laser_block_equals_mobile_caps() -> None:
    expected = yaml.safe_load((ROBOTS_DIR / "jackal" / "caps" / "mobile.yaml").read_text())["laser"]
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
    expected = yaml.safe_load((ROBOTS_DIR / "tiago" / "caps" / "mobile.yaml").read_text())["laser"]
    block = laser_block(_golden("tiago"))
    assert block is not None
    assert block["num_beams"] == 401
    assert block["range"] == pytest.approx(25.0)
    assert block["angle"]["increment"] == pytest.approx((1.74533 * 2) / 400)
    assert block["angle"]["increment"] == pytest.approx(expected["angle"]["increment"], abs=1e-8)


def test_rbkairos_laser_block_selects_the_front_frame() -> None:
    expected = yaml.safe_load((ROBOTS_DIR / "rbkairos" / "caps" / "mobile.yaml").read_text())["laser"]
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


def test_jackal_wheel_geometry_equals_control_yaml() -> None:
    control = yaml.safe_load((ROBOTS_DIR / "jackal" / "control.yaml").read_text())["jackal_velocity_controller"]["ros__parameters"]
    facts = wheel_geometry(_golden("jackal"), "base_link", control["left_wheel_names"], control["right_wheel_names"])
    assert facts is not None
    assert facts["wheel_radius"] == pytest.approx(control["wheel_radius"]) == pytest.approx(0.098)
    assert facts["wheel_separation"] == pytest.approx(control["wheel_separation"]) == pytest.approx(0.37559)


def test_wheel_geometry_of_unknown_joint_is_none() -> None:
    assert wheel_geometry(_golden("jackal"), "base_link", ["missing_joint"], ["front_right_wheel_joint"]) is None


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
