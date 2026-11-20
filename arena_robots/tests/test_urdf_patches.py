from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest
import yaml
from arena_robots import urdf_facts, urdf_patches
from arena_robots.assembly import RequestPart
from arena_robots.Robot import RobotIdentifier, RobotView
from arena_robots.served_urdf import canonical, serve

COMPONENTS = Path(__file__).resolve().parents[1] / "components"
ROBOTS = Path(__file__).resolve().parents[1] / "robots"

XACRO = """<?xml version="1.0"?>
<robot name="{name}" xmlns:xacro="http://www.ros.org/wiki/xacro">
  <link name="base_link">
    <inertial><mass value="2.0"/><inertia ixx="0.4" ixy="0.0" ixz="0.0" iyy="0.5" iyz="0.0" izz="0.6"/></inertial>
  </link>
  <link name="arm_link">
    <inertial><mass value="1.0"/><inertia ixx="0.1" ixy="0.01" ixz="0.0" iyy="0.2" iyz="0.0" izz="0.3"/></inertial>
  </link>
  <joint name="arm_joint" type="revolute">
    <parent link="base_link"/>
    <child link="arm_link"/>
    <axis xyz="0 0 1"/>
    <limit lower="-1.0" upper="1.0" velocity="2.0" effort="10.0"/>
  </joint>
</robot>
"""


def _shapes(*geometries: str) -> str:
    collisions = "".join(f'<collision><origin xyz="{xyz}" rpy="{rpy}"/><geometry>{shape}</geometry></collision>' for shape, xyz, rpy in geometries)
    return f"""<?xml version="1.0"?>
<robot name="{{name}}" xmlns:xacro="http://www.ros.org/wiki/xacro">
  <link name="base_link">
    <inertial><origin xyz="0 0 0" rpy="0 0 0"/><mass value="12.0"/><inertia ixx="9" ixy="0" ixz="0" iyy="9" iyz="0" izz="9"/></inertial>
    {collisions}
  </link>
</robot>
"""


def _robot(tmp_path: Path, patches: str, xacro: str = XACRO) -> RobotView:
    pytest.importorskip("xacro")
    root = tmp_path / "patched"
    (root / "urdf").mkdir(parents=True)
    (root / "urdf" / "patched.urdf.xacro").write_text(xacro.format(name="patched"))
    (root / urdf_patches.FILE_NAME).write_text(patches)
    return RobotView(root)


def _links(view: RobotView) -> dict[str, ET.Element]:
    return {link.get("name", ""): link for link in ET.fromstring(canonical(view, {})).findall("link")}


def _mass(link: ET.Element) -> float:
    return float(link.find("inertial/mass").get("value"))


def _inertia(link: ET.Element) -> dict[str, float]:
    return {key: float(value) for key, value in link.find("inertial/inertia").attrib.items()}


def test_link_mass_sets_the_mass_and_scales_the_inertia_with_it(tmp_path: Path) -> None:
    view = _robot(tmp_path, "patches:\n  - {link: arm_link, mass: 3.0, reason: r, source: s}\n")
    links = _links(view)
    assert _mass(links["arm_link"]) == pytest.approx(3.0)
    assert _inertia(links["arm_link"]) == pytest.approx({"ixx": 0.3, "ixy": 0.03, "ixz": 0.0, "iyy": 0.6, "iyz": 0.0, "izz": 0.9})
    assert _mass(links["base_link"]) == 2.0


def test_total_mass_scales_every_link_of_the_robot_to_the_sum(tmp_path: Path) -> None:
    view = _robot(tmp_path, "patches:\n  - {total_mass: 1.5, reason: r, source: s}\n")
    links = _links(view)
    assert sum(_mass(link) for link in links.values()) == pytest.approx(1.5)
    assert _mass(links["base_link"]) == pytest.approx(1.0)
    assert _inertia(links["base_link"])["iyy"] == pytest.approx(0.25)
    assert _inertia(links["arm_link"])["ixy"] == pytest.approx(0.005)
    assert urdf_facts.total_mass(canonical(view, {})) == pytest.approx(1.5)


def test_joint_limit_sets_only_the_given_attributes(tmp_path: Path) -> None:
    view = _robot(tmp_path, "patches:\n  - {joint: arm_joint, limit: {velocity: 1.0, effort: 5.0}, reason: r, source: s}\n")
    limit = ET.fromstring(canonical(view, {})).find("joint[@name='arm_joint']/limit")
    assert (limit.get("lower"), limit.get("upper")) == ("-1.0", "1.0")
    assert (float(limit.get("velocity")), float(limit.get("effort"))) == (1.0, 5.0)


def test_inertia_scale_leaves_the_mass(tmp_path: Path) -> None:
    view = _robot(tmp_path, "patches:\n  - {link: base_link, inertia_scale: 0.5, reason: r, source: s}\n")
    link = _links(view)["base_link"]
    assert _mass(link) == 2.0
    assert _inertia(link)["izz"] == pytest.approx(0.3)


def test_patches_reach_the_served_urdf(tmp_path: Path) -> None:
    view = _robot(tmp_path, "patches:\n  - {total_mass: 1.5, reason: r, source: s}\n")
    served = ET.fromstring(serve(view, {}).description)
    assert sum(_mass(link) for link in served.findall("link")) == pytest.approx(1.5)


@pytest.mark.parametrize(
    ("patch", "name"),
    [
        ("link: ghost_link, mass: 1.0", "ghost_link"),
        ("link: ghost_link, inertia_scale: 2.0", "ghost_link"),
        ("link: ghost_link, inertia: from_collision", "ghost_link"),
        ("link: ghost_link, collision: {box: [1, 1, 1]}", "ghost_link"),
        ("joint: ghost_joint, limit: {effort: 1.0}", "ghost_joint"),
    ],
)
def test_a_missing_link_or_joint_raises_naming_the_file_and_the_name(tmp_path: Path, patch: str, name: str) -> None:
    view = _robot(tmp_path, "patches:\n  - {%s, reason: r, source: s}\n" % patch)
    with pytest.raises(ValueError, match=f"{urdf_patches.FILE_NAME}.*{name}"):
        canonical(view, {})


@pytest.mark.parametrize("missing", ["reason", "source"])
def test_a_patch_without_reason_or_source_raises(tmp_path: Path, missing: str) -> None:
    entry = {"total_mass": 1.5, "reason": "r", "source": "s"}
    del entry[missing]
    view = _robot(tmp_path, yaml.safe_dump({"patches": [entry]}))
    with pytest.raises(ValueError, match=missing):
        canonical(view, {})


def _variants() -> dict[str, float]:
    data = yaml.safe_load((COMPONENTS / "arm" / "ur" / urdf_patches.FILE_NAME).read_text())["variants"]
    return {variant: entries[0]["total_mass"] for variant, entries in data.items()}


def test_every_ur_variant_has_a_patch() -> None:
    family = yaml.safe_load((COMPONENTS / "arm" / "ur" / "component.yaml").read_text())["variants"]
    assert sorted(_variants()) == sorted(family)


@pytest.mark.parametrize(("variant", "mass"), sorted(_variants().items()))
def test_a_ur_arm_on_jackal_adds_its_patched_mass(variant: str, mass: float, require_installed) -> None:
    pytest.importorskip("xacro")
    require_installed("jackal")
    require_installed("arm/ur")
    view = RobotIdentifier("jackal").resolve_sync()
    bare = urdf_facts.total_mass(canonical(view, {}))
    armed = urdf_facts.total_mass(canonical(view, {"arm": [RequestPart(variant=variant, mount="top")]}))
    assert armed - bare == pytest.approx(mass, abs=1e-4)


def test_body_mass_of_jackal_with_a_ur5e_follows_the_patch(require_installed) -> None:
    pytest.importorskip("xacro")
    require_installed("jackal")
    require_installed("arm/ur")
    view = RobotIdentifier("jackal").resolve_sync()
    bare = view.effective_caps({}).body.mass
    armed = view.effective_caps({"arm": [RequestPart(variant="ur5e", mount="top")]}).body.mass
    assert armed - bare == pytest.approx(20.7, abs=1e-4)


def test_fr3_on_jackal_adds_its_patched_mass(require_installed) -> None:
    pytest.importorskip("xacro")
    require_installed("jackal")
    require_installed("arm/fr3")
    view = RobotIdentifier("jackal").resolve_sync()
    bare = urdf_facts.total_mass(canonical(view, {}))
    armed = urdf_facts.total_mass(canonical(view, {"arm": [RequestPart(variant="fr3", mount="top")]}))
    assert armed - bare == pytest.approx(18.3, abs=1e-4)


def test_turtlebot_weighs_its_vendor_mass_with_lidar_and_camera(require_installed) -> None:
    pytest.importorskip("xacro")
    require_installed("turtlebot")
    caps = RobotIdentifier("turtlebot").resolve_sync().effective_caps({})
    assert caps.body.mass == pytest.approx(3.945)


def test_ridgeback_plus_chassis_mass_follows_the_patch(require_installed) -> None:
    pytest.importorskip("xacro")
    require_installed("ridgeback_plus")
    view = RobotIdentifier("ridgeback_plus").resolve_sync()
    assert _mass(_links(view)["chassis_link"]) == pytest.approx(105.304)


BOX = ('<box size="1 2 3"/>', "0.1 0.2 0.3", "0 0 1.5707963267948966")
CYLINDER = ('<cylinder radius="0.1" length="0.4"/>', "0 0 0", "1.5707963267948966 0 0")
SPHERE = ('<sphere radius="0.2"/>', "0 0 0", "0 0 0")
FROM_COLLISION = "patches:\n  - {link: base_link, inertia: from_collision, reason: r, source: s}\n"


def test_inertia_from_collision_rotates_a_box_tensor_about_the_collision_origin(tmp_path: Path) -> None:
    link = _links(_robot(tmp_path, FROM_COLLISION, _shapes(BOX)))["base_link"]
    assert _inertia(link) == pytest.approx({"ixx": 10.0, "ixy": 0.0, "ixz": 0.0, "iyy": 13.0, "iyz": 0.0, "izz": 5.0})
    assert [float(value) for value in link.find("inertial/origin").get("xyz").split()] == pytest.approx([0.1, 0.2, 0.3])
    assert _mass(link) == 12.0


def test_inertia_from_collision_orients_a_cylinder_along_its_rotated_axis(tmp_path: Path) -> None:
    link = _links(_robot(tmp_path, FROM_COLLISION, _shapes(CYLINDER)))["base_link"]
    mass = _mass(link)
    assert _inertia(link) == pytest.approx({"ixx": mass * (3 * 0.01 + 0.16) / 12, "ixy": 0.0, "ixz": 0.0, "iyy": mass * 0.01 / 2, "iyz": 0.0, "izz": mass * (3 * 0.01 + 0.16) / 12}, abs=1e-9)


def test_inertia_from_collision_of_a_sphere_is_isotropic(tmp_path: Path) -> None:
    inertia = _inertia(_links(_robot(tmp_path, FROM_COLLISION, _shapes(SPHERE)))["base_link"])
    assert inertia == pytest.approx({"ixx": 2 / 5 * 12 * 0.04, "ixy": 0.0, "ixz": 0.0, "iyy": 2 / 5 * 12 * 0.04, "iyz": 0.0, "izz": 2 / 5 * 12 * 0.04})


def test_inertia_from_collision_off_axis_fills_the_products_of_inertia(tmp_path: Path) -> None:
    tilted = ('<box size="1 2 3"/>', "0 0 0", f"0 0 {math.pi / 4}")
    inertia = _inertia(_links(_robot(tmp_path, FROM_COLLISION, _shapes(tilted)))["base_link"])
    assert (inertia["ixx"], inertia["iyy"], inertia["izz"]) == pytest.approx((11.5, 11.5, 5.0))
    assert inertia["ixy"] == pytest.approx(1.5)


@pytest.mark.parametrize(
    "shapes",
    [(), (BOX, SPHERE), ('<mesh filename="package://x/y.stl"/>', "0 0 0", "0 0 0")],
    ids=["none", "two", "mesh"],
)
def test_inertia_from_collision_needs_a_single_primitive_collision(tmp_path: Path, shapes: tuple) -> None:
    geometries = (shapes,) if shapes and isinstance(shapes[0], str) else shapes
    view = _robot(tmp_path, FROM_COLLISION, _shapes(*geometries))
    with pytest.raises(ValueError, match=f"{urdf_patches.FILE_NAME}.*base_link.*collision"):
        canonical(view, {})


def test_inertia_other_than_from_collision_raises(tmp_path: Path) -> None:
    view = _robot(tmp_path, "patches:\n  - {link: base_link, inertia: guess, reason: r, source: s}\n")
    with pytest.raises(ValueError, match="from_collision"):
        canonical(view, {})


def test_a_collision_patch_adds_a_box_with_its_pose(tmp_path: Path) -> None:
    patch = "patches:\n  - {link: arm_link, collision: {box: [0.5, 0.6, 0.7], xyz: [0.1, 0.2, 0.3], rpy: [0, 0, 1.0]}, reason: r, source: s}\n"
    collision = _links(_robot(tmp_path, patch))["arm_link"].find("collision")
    assert [float(value) for value in collision.find("geometry/box").get("size").split()] == [0.5, 0.6, 0.7]
    assert [float(value) for value in collision.find("origin").get("xyz").split()] == [0.1, 0.2, 0.3]
    assert [float(value) for value in collision.find("origin").get("rpy").split()] == [0.0, 0.0, 1.0]


def test_a_collision_patch_reaches_the_ground_outline(tmp_path: Path) -> None:
    patch = "patches:\n  - {link: base_link, collision: {box: [2.0, 1.0, 0.1], xyz: [0, 0, 0.5]}, reason: r, source: s}\n"
    outline = urdf_facts.collision_outline(canonical(_robot(tmp_path, patch), {}), "base_link")
    assert max(x for x, _ in outline) - min(x for x, _ in outline) == pytest.approx(2.0)
    assert max(y for _, y in outline) - min(y for _, y in outline) == pytest.approx(1.0)


@pytest.mark.parametrize("collision", ["{box: [1, 1]}", "{box: [1, 1, -1]}", "{cylinder: [1, 1]}", "{box: [1, 1, 1], xyz: [0, 0]}", "box"])
def test_a_malformed_collision_patch_raises(tmp_path: Path, collision: str) -> None:
    view = _robot(tmp_path, "patches:\n  - {link: base_link, collision: %s, reason: r, source: s}\n" % collision)
    with pytest.raises(ValueError, match=urdf_patches.FILE_NAME):
        canonical(view, {})


@pytest.mark.parametrize(("robot", "length", "width"), [("dingo", 0.5512, 0.5172), ("ridgeback", 0.9326, 0.79), ("jackal", 0.5106, 0.43)])
def test_the_chassis_outline_reaches_the_bumpers_or_fenders(robot: str, length: float, width: float, require_installed) -> None:
    pytest.importorskip("xacro")
    require_installed(robot)
    view = RobotIdentifier(robot).resolve_sync()
    outline = urdf_facts.collision_outline(canonical(view, {}), view.model_params.base_frame, max_vertices=1_000_000)
    assert max(x for x, _ in outline) - min(x for x, _ in outline) == pytest.approx(length, abs=1e-3)
    assert max(y for _, y in outline) - min(y for _, y in outline) == pytest.approx(width, abs=1e-3)


@pytest.mark.parametrize("path", sorted([*COMPONENTS.glob(f"*/*/{urdf_patches.FILE_NAME}"), *ROBOTS.glob(f"*/{urdf_patches.FILE_NAME}")]), ids=lambda p: f"{p.parents[1].name}/{p.parent.name}")
def test_every_patch_file_parses(path: Path) -> None:
    variants = yaml.safe_load(path.read_text()).get("variants") or {None: None}
    for variant in variants:
        assert urdf_patches.load(path, variant)
