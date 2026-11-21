"""Facts read from a rendered URDF: joint limits, lidar block, mass, drive geometry, wheelbase and the ground outline and height of the base."""

from __future__ import annotations

import logging
import math
import struct
import typing
import xml.etree.ElementTree as ET
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import attrs
import numpy as np
from ament_index_python.packages import PackageNotFoundError, get_package_share_path

if typing.TYPE_CHECKING:
    from arena_robots.Robot import RobotView

_LOGGER = logging.getLogger(__name__)
_LIDAR_TYPES = frozenset({"gpu_lidar", "gpu_ray", "ray", "lidar"})
_FULL_TURN = 2.0 * math.pi
_ARC_SEGMENTS = 16
_TOLERANCE = 1e-3
_DECIMALS = 6
DIFF_DRIVE = "diff_drive_controller/DiffDriveController"
MECANUM_DRIVE = "mecanum_drive_controller/MecanumDriveController"
SWERVE_DRIVE = "arena_swerve_controller/SwerveController"
OMNI_DRIVE = "arena_omni_drive_controller/OmniDriveController"
DRIVE_CONTROLLER_TYPES = frozenset({DIFF_DRIVE, MECANUM_DRIVE, SWERVE_DRIVE, OMNI_DRIVE})
DRIVE_TOPICS = {DIFF_DRIVE: ("cmd_vel", "odom"), MECANUM_DRIVE: ("reference", "odometry"), SWERVE_DRIVE: ("cmd_vel", "odometry"), OMNI_DRIVE: ("cmd_vel", "odometry")}
DriveGeometry = dict[str, float | list[float]]
_MECANUM_JOINTS = ("front_left_wheel_command_joint_name", "front_right_wheel_command_joint_name", "rear_left_wheel_command_joint_name", "rear_right_wheel_command_joint_name")


@attrs.frozen
class JointLimit:
    lower: float | None
    upper: float | None
    velocity: float | None
    effort: float | None


def source(view: RobotView) -> Path:
    """The robot's own top-level URDF xacro."""
    return view.path / "urdf" / f"{view.name}.urdf.xacro"


def _floats(text: str | None, count: int, default: float = 0.0) -> list[float]:
    if text is None or not text.split():
        return [default] * count
    return [float(token) for token in text.split()]


def _optional_float(element: ET.Element, name: str) -> float | None:
    value = element.get(name)
    return None if value is None else float(value)


def _transform(origin: ET.Element | None) -> np.ndarray:
    matrix = np.eye(4)
    if origin is None:
        return matrix
    x, y, z = _floats(origin.get("xyz"), 3)
    roll, pitch, yaw = _floats(origin.get("rpy"), 3)
    rx = np.array([[1, 0, 0], [0, math.cos(roll), -math.sin(roll)], [0, math.sin(roll), math.cos(roll)]])
    ry = np.array([[math.cos(pitch), 0, math.sin(pitch)], [0, 1, 0], [-math.sin(pitch), 0, math.cos(pitch)]])
    rz = np.array([[math.cos(yaw), -math.sin(yaw), 0], [math.sin(yaw), math.cos(yaw), 0], [0, 0, 1]])
    matrix[:3, :3] = rz @ ry @ rx
    matrix[:3, 3] = (x, y, z)
    return matrix


def joint_limits(urdf: str) -> dict[str, JointLimit]:
    """Limits of every joint by name, None for the attributes a joint does not declare."""
    out: dict[str, JointLimit] = {}
    for joint in ET.fromstring(urdf).findall("joint"):
        name = joint.get("name")
        if name is None:
            continue
        limit = joint.find("limit")
        if limit is None:
            out[name] = JointLimit(None, None, None, None)
            continue
        out[name] = JointLimit(_optional_float(limit, "lower"), _optional_float(limit, "upper"), _optional_float(limit, "velocity"), _optional_float(limit, "effort"))
    return out


def chain_joints(urdf: str, base_link: str, tip_link: str) -> list[str]:
    """Names of the non-fixed joints between ``base_link`` and its descendant ``tip_link``, base first."""
    parents = {child.get("link"): (joint, parent.get("link")) for joint in ET.fromstring(urdf).findall("joint") if (child := joint.find("child")) is not None and (parent := joint.find("parent")) is not None}
    out: list[str] = []
    link = tip_link
    while link != base_link:
        if link not in parents:
            raise ValueError(f"link {tip_link!r} is not below {base_link!r}")
        joint, link = parents[link]
        if joint.get("type") != "fixed":
            out.append(str(joint.get("name")))
    return out[::-1]


def sensor_frames(urdf: str) -> dict[str, str]:
    """TF frame of every Gazebo sensor by sensor name, its `gz_frame_id` when set, else its `<gazebo reference>` link."""
    out: dict[str, str] = {}
    for gazebo in ET.fromstring(urdf).iter("gazebo"):
        for sensor in gazebo.findall("sensor"):
            name = sensor.get("name")
            frame = (sensor.findtext("gz_frame_id") or "").strip().lstrip("/") or gazebo.get("reference")
            if name is not None and frame:
                out[name] = frame
    return out


def model_plugins(urdf: str, *filenames: str) -> list[ET.Element]:
    """Plugins of the `<gazebo>` blocks without a reference (the model's own) whose filename is one of `filenames`."""
    return [plugin for gazebo in ET.fromstring(urdf).iter("gazebo") if gazebo.get("reference") is None for plugin in gazebo.findall("plugin") if plugin.get("filename") in filenames]


def _text_float(element: ET.Element, path: str) -> float | None:
    found = element.find(path)
    if found is None or found.text is None or not found.text.strip():
        return None
    return float(found.text)


def laser_block(urdf: str, frame: str | None = None) -> dict[str, object] | None:
    """The first ray or gpu_lidar sensor (on `frame` when given) as a `mobile.laser` mapping."""
    for gazebo in ET.fromstring(urdf).iter("gazebo"):
        for sensor in gazebo.findall("sensor"):
            if sensor.get("type") not in _LIDAR_TYPES:
                continue
            if frame is not None and gazebo.get("reference") != frame and (sensor.findtext("gz_frame_id") or "").strip().lstrip("/") != frame:
                continue
            block = sensor.find("lidar")
            if block is None:
                block = sensor.find("ray")
            if block is None:
                continue
            low = _text_float(block, "scan/horizontal/min_angle")
            high = _text_float(block, "scan/horizontal/max_angle")
            samples = _text_float(block, "scan/horizontal/samples")
            reach = _text_float(block, "range/max")
            rate = _text_float(sensor, "update_rate")
            if low is None or high is None or samples is None or reach is None or rate is None or samples < 2:
                continue
            count = int(samples)
            span = high - low
            increment = _FULL_TURN / count if abs(span - _FULL_TURN) < 1e-3 else span / (count - 1)
            return {"angle": {"min": low, "max": high, "increment": increment}, "num_beams": count, "range": reach, "update_rate": rate}
    return None


def total_mass(urdf: str) -> float:
    """Sum of the inertial masses of every link."""
    return sum(float(mass.get("value", 0.0)) for link in ET.fromstring(urdf).findall("link") for mass in link.findall("inertial/mass"))


def _joint_tree(root: ET.Element) -> dict[str, list[tuple[ET.Element, str]]]:
    children: dict[str, list[tuple[ET.Element, str]]] = {}
    for joint in root.findall("joint"):
        parent = joint.find("parent")
        child = joint.find("child")
        if parent is None or child is None:
            continue
        children.setdefault(str(parent.get("link")), []).append((joint, str(child.get("link"))))
    return children


def _link_poses(root: ET.Element, base_link: str, only: Callable[[ET.Element], bool] | None = None) -> dict[str, np.ndarray]:
    children = _joint_tree(root)
    poses = {base_link: np.eye(4)}
    queue = [base_link]
    while queue:
        parent = queue.pop()
        for joint, child in children.get(parent, []):
            if child in poses or (only is not None and not only(joint)):
                continue
            poses[child] = poses[parent] @ _transform(joint.find("origin"))
            queue.append(child)
    return poses


def _collision_radius(link: ET.Element) -> float | None:
    for collision in link.findall("collision"):
        for shape in ("cylinder", "sphere"):
            found = collision.find(f"geometry/{shape}")
            if found is not None and found.get("radius") is not None:
                return float(str(found.get("radius")))
    return None


def drive_controller(control: Mapping[str, Any]) -> tuple[str, str, dict[str, Any]] | None:
    """Name, type and ``ros__parameters`` of the first wheel drive controller the ``controller_manager`` section lists, None without one."""
    listed = (control.get("controller_manager") or {}).get("ros__parameters") or {}
    for name, entry in listed.items():
        kind = entry.get("type") if isinstance(entry, Mapping) else None
        if kind in DRIVE_CONTROLLER_TYPES:
            return str(name), str(kind), dict(control[name]["ros__parameters"])
    return None


def controller_names(control: Mapping[str, Any]) -> tuple[str, ...]:
    """Names of the controllers the ``controller_manager`` section lists, in file order."""
    listed = (control.get("controller_manager") or {}).get("ros__parameters") or {}
    return tuple(str(name) for name, entry in listed.items() if isinstance(entry, Mapping) and "type" in entry)


def wheel_joints(controller_type: str, params: Mapping[str, Any]) -> list[str]:
    """The wheel joint names a drive controller commands."""
    if controller_type == DIFF_DRIVE:
        return [str(name) for key in ("left_wheel_names", "right_wheel_names") for name in params[key]]
    if controller_type == MECANUM_DRIVE:
        return [str(params[key]) for key in _MECANUM_JOINTS]
    return [str(name) for name in params["wheel_joint_names"]]


def _joint_frames(root: ET.Element, base_link: str) -> dict[str, np.ndarray]:
    poses = _link_poses(root, base_link)
    return {str(joint.get("name")): poses[str(parent.get("link"))] @ _transform(joint.find("origin")) for joint in root.findall("joint") if (parent := joint.find("parent")) is not None and str(parent.get("link")) in poses}


def _wheel_radius(root: ET.Element, names: Sequence[str]) -> float:
    links = {str(link.get("name")): link for link in root.findall("link")}
    joints = {str(joint.get("name")): joint for joint in root.findall("joint")}
    radii: list[float] = []
    for name in names:
        child = joints[name].find("child")
        radius = _collision_radius(links[str(child.get("link"))]) if child is not None else None
        if radius is None:
            raise ValueError(f"wheel joint {name} has no round collision on its child link")
        radii.append(radius)
    if max(radii) - min(radii) > _TOLERANCE:
        raise ValueError(f"wheel joints {list(names)} disagree on the wheel radius: {radii}")
    return sum(radii) / len(radii)


def wheelbase(urdf: str, controller_type: str, params: Mapping[str, Any]) -> float | None:
    """Front to rear distance of a drive controller's wheel joints along the base x axis, None when they share one x."""
    root = ET.fromstring(urdf)
    frames = _joint_frames(root, str(params.get("base_frame_id") or _root_link(root)))
    xs = [float(frames[name][0, 3]) for name in wheel_joints(controller_type, params) if name in frames]
    extent = _rounded(max(xs) - min(xs)) if xs else 0.0
    return extent or None


def _rounded(value: float) -> float:
    return round(value, _DECIMALS) + 0.0


def wheel_axles(urdf: str, controller_type: str, params: Mapping[str, Any]) -> list[float]:
    """Sorted base x of every wheel joint of a drive controller, measured in its ``base_frame_id``."""
    root = ET.fromstring(urdf)
    frames = _joint_frames(root, str(params.get("base_frame_id") or _root_link(root)))
    return sorted(_rounded(float(frames[name][0, 3])) for name in wheel_joints(controller_type, params) if name in frames)


def drive_geometry(urdf: str, controller_type: str, params: Mapping[str, Any]) -> DriveGeometry:
    """Geometry parameters of one drive controller measured in its ``base_frame_id`` (the root link when it has none)."""
    root = ET.fromstring(urdf)
    base_link = str(params.get("base_frame_id") or _root_link(root))
    frames = _joint_frames(root, base_link)
    names = wheel_joints(controller_type, params)
    missing = [name for name in names if name not in frames]
    if missing:
        raise ValueError(f"wheel joints {missing} are not below the base frame {base_link}")
    radius = _rounded(_wheel_radius(root, names))
    if controller_type == DIFF_DRIVE:
        sides = [[float(frames[str(name)][1, 3]) for name in params[key]] for key in ("left_wheel_names", "right_wheel_names")]
        return {"wheel_radius": radius, "wheel_separation": _rounded(abs(sum(sides[0]) / len(sides[0]) - sum(sides[1]) / len(sides[1])))}
    if controller_type == MECANUM_DRIVE:
        sums = [abs(float(frames[name][0, 3])) + abs(float(frames[name][1, 3])) for name in names]
        if max(sums) - min(sums) > _TOLERANCE:
            raise ValueError(f"mecanum wheels disagree on the center projection: {sums}")
        return {"kinematics.wheels_radius": radius, "kinematics.sum_of_robot_center_projection_on_X_Y_axis": _rounded(sum(sums) / len(sums))}
    if controller_type == SWERVE_DRIVE:
        steering = [str(name) for name in params["steering_joint_names"]]
        if len(steering) != len(names) or any(name not in frames for name in steering):
            raise ValueError(f"steering joints {steering} do not match the wheel joints {names} below {base_link}")
        places = [frames[name] for name in steering]
    else:
        places = [frames[name] for name in names]
    geometry: DriveGeometry = {"wheel_radius": radius, "wheel_positions_x": [_rounded(float(pose[0, 3])) for pose in places], "wheel_positions_y": [_rounded(float(pose[1, 3])) for pose in places]}
    if controller_type == OMNI_DRIVE:
        joints = {str(joint.get("name")): joint for joint in root.findall("joint")}
        angles: list[float] = []
        for name in names:
            element = joints[name].find("axis")
            axis = np.array(_floats(element.get("xyz") if element is not None else "1 0 0", 3))
            along = frames[name][:3, :3] @ axis
            angles.append(_rounded((math.atan2(float(along[1]), float(along[0])) - math.pi / 2) % _FULL_TURN))
        geometry["wheel_angles"] = angles
    return geometry


def _root_link(root: ET.Element) -> str:
    children = {str(child.get("link")) for joint in root.findall("joint") if (child := joint.find("child")) is not None}
    return next(str(link.get("name")) for link in root.findall("link") if str(link.get("name")) not in children)


def _default_resolver(uri: str) -> Path | None:
    if uri.startswith("package://"):
        package, _, rest = uri[len("package://") :].partition("/")
        try:
            return get_package_share_path(package) / rest
        except PackageNotFoundError:
            return None
    if uri.startswith("file://"):
        return Path(uri[len("file://") :])
    path = Path(uri)
    return path if path.is_absolute() else None


def _stl_vertices(data: bytes) -> np.ndarray | None:
    if len(data) >= 84:
        count = struct.unpack_from("<I", data, 80)[0]
        if len(data) == 84 + 50 * count:
            if count == 0:
                return np.zeros((0, 3))
            raw = np.frombuffer(data, dtype=np.dtype([("normal", "<f4", 3), ("vertices", "<f4", (3, 3)), ("attr", "<u2")]), count=count, offset=84)
            return raw["vertices"].reshape(-1, 3).astype(float)
    try:
        text = data.decode("ascii")
    except UnicodeDecodeError:
        return None
    points = [[float(token) for token in line.split()[1:4]] for line in text.splitlines() if line.strip().startswith("vertex")]
    return np.array(points, dtype=float).reshape(-1, 3) if points else None


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _dae_vertices(data: bytes) -> np.ndarray | None:
    root = ET.fromstring(data)
    sources = {element.get("id"): element for element in root.iter() if _local_name(element.tag) == "source"}
    points: list[np.ndarray] = []
    for mesh in (element for element in root.iter() if _local_name(element.tag) == "mesh"):
        for vertices in (element for element in mesh if _local_name(element.tag) == "vertices"):
            for entry in (element for element in vertices if _local_name(element.tag) == "input" and element.get("semantic") == "POSITION"):
                source = sources.get(str(entry.get("source")).lstrip("#"))
                if source is None:
                    continue
                for array in (element for element in source if _local_name(element.tag) == "float_array"):
                    points.append(np.array((array.text or "").split(), dtype=float).reshape(-1, 3))
    return np.vstack(points) if points else None


def _mesh_vertices(uri: str, resolver: Callable[[str], Path | None]) -> np.ndarray | None:
    path = resolver(uri)
    if path is None or not path.is_file():
        _LOGGER.warning("collision mesh %s not found, skipped", path if path is not None else uri)
        return None
    suffix = path.suffix.lower()
    try:
        if suffix == ".stl":
            vertices = _stl_vertices(path.read_bytes())
        elif suffix == ".dae":
            vertices = _dae_vertices(path.read_bytes())
        else:
            _LOGGER.warning("collision mesh %s has an unsupported format, skipped", path)
            return None
    except (ValueError, ET.ParseError, struct.error):
        vertices = None
    if vertices is None:
        _LOGGER.warning("collision mesh %s could not be read, skipped", path)
    return vertices


def _circle(center: np.ndarray, radius: float) -> np.ndarray:
    angles = np.linspace(0.0, _FULL_TURN, _ARC_SEGMENTS, endpoint=False)
    return np.stack([center[0] + radius * np.cos(angles), center[1] + radius * np.sin(angles), np.full_like(angles, center[2])], axis=1)


def _geometry_points(geometry: ET.Element, pose: np.ndarray, resolver: Callable[[str], Path | None]) -> np.ndarray | None:
    box = geometry.find("box")
    cylinder = geometry.find("cylinder")
    sphere = geometry.find("sphere")
    mesh = geometry.find("mesh")
    local: np.ndarray | None
    if box is not None:
        sx, sy, sz = _floats(box.get("size"), 3)
        local = np.array([[x * sx / 2, y * sy / 2, z * sz / 2] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)])
    elif cylinder is not None:
        radius = float(cylinder.get("radius", 0.0))
        half = float(cylinder.get("length", 0.0)) / 2
        angles = np.linspace(0.0, _FULL_TURN, _ARC_SEGMENTS, endpoint=False)
        ring = np.stack([radius * np.cos(angles), radius * np.sin(angles)], axis=1)
        local = np.vstack([np.hstack([ring, np.full((_ARC_SEGMENTS, 1), -half)]), np.hstack([ring, np.full((_ARC_SEGMENTS, 1), half)])])
    elif sphere is not None:
        center = pose[:3, 3]
        return _circle(center, float(sphere.get("radius", 0.0)))
    elif mesh is not None:
        filename = mesh.get("filename")
        local = _mesh_vertices(filename, resolver) if filename else None
        if local is not None:
            local = local * np.array(_floats(mesh.get("scale"), 3, 1.0))
    else:
        return None
    if local is None or len(local) == 0:
        return None
    return (pose[:3, :3] @ local.T).T + pose[:3, 3]


def _cross(origin: tuple[float, float], a: tuple[float, float], b: tuple[float, float]) -> float:
    return (a[0] - origin[0]) * (b[1] - origin[1]) - (a[1] - origin[1]) * (b[0] - origin[0])


def _convex_hull(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    unique = sorted(set(points))
    if len(unique) < 3:
        return unique
    lower: list[tuple[float, float]] = []
    for point in unique:
        while len(lower) >= 2 and _cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)
    upper: list[tuple[float, float]] = []
    for point in reversed(unique):
        while len(upper) >= 2 and _cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)
    return lower[:-1] + upper[:-1]


def _meet(a: tuple[float, float], b: tuple[float, float], d: tuple[float, float], c: tuple[float, float]) -> tuple[float, float] | None:
    ux, uy = b[0] - a[0], b[1] - a[1]
    vx, vy = c[0] - d[0], c[1] - d[1]
    det = ux * vy - uy * vx
    if abs(det) < 1e-12:
        return None
    t = ((d[0] - a[0]) * vy - (d[1] - a[1]) * vx) / det
    x = (a[0] + t * ux, a[1] + t * uy)
    if (x[0] - b[0]) * ux + (x[1] - b[1]) * uy < 0 or (x[0] - c[0]) * vx + (x[1] - c[1]) * vy < 0:
        return None
    return x


def _reduce(hull: list[tuple[float, float]], max_vertices: int) -> list[tuple[float, float]]:
    """A convex polygon of at most ``max_vertices`` vertices enclosing ``hull``, each step trading one edge for the meeting point of its neighbors at the least added area."""
    polygon = list(hull)
    while len(polygon) > max(max_vertices, 3):
        n = len(polygon)
        best: tuple[float, int, tuple[float, float]] | None = None
        for i in range(n):
            a, b, c, d = polygon[i - 1], polygon[i], polygon[(i + 1) % n], polygon[(i + 2) % n]
            x = _meet(a, b, d, c)
            if x is None:
                continue
            added = abs(_cross(b, x, c)) / 2
            if best is None or added < best[0]:
                best = (added, i, x)
        if best is None:
            break
        _, i, x = best
        polygon[i] = x
        del polygon[(i + 1) % n]
    return polygon


def collision_outline(
    urdf: str,
    base_link: str,
    max_vertices: int = 8,
    mesh_resolver: Callable[[str], Path | None] | None = None,
) -> list[tuple[float, float]] | None:
    """Ground-plane convex hull of the base, its fixed-joint links and its wheels at zero joint position, counterclockwise and reduced to `max_vertices`."""
    resolver = mesh_resolver if mesh_resolver is not None else _default_resolver
    root = ET.fromstring(urdf)
    links = {str(link.get("name")): link for link in root.findall("link")}
    if base_link not in links:
        return None
    body = _link_poses(root, base_link, only=lambda joint: joint.get("type") == "fixed")
    poses = dict(body)
    children = _joint_tree(root)
    for parent, pose in body.items():
        for joint, child in children.get(parent, []):
            if joint.get("type") == "continuous" and child not in poses:
                poses[child] = pose @ _transform(joint.find("origin"))
    projected: list[tuple[float, float]] = []
    for name, pose in poses.items():
        link = links.get(name)
        if link is None:
            continue
        for collision in link.findall("collision"):
            geometry = collision.find("geometry")
            if geometry is None:
                continue
            points = _geometry_points(geometry, pose @ _transform(collision.find("origin")), resolver)
            if points is not None:
                projected.extend((float(x), float(y)) for x, y, _ in points)
    hull = _convex_hull(projected)
    if len(hull) < 3:
        return None
    return _reduce(hull, max_vertices)


def body_height(
    urdf: str,
    base_link: str,
    mesh_resolver: Callable[[str], Path | None] | None = None,
) -> float | None:
    """Vertical extent of the collision geometry and the sensor frames of every link below `base_link` at zero joint position."""
    resolver = mesh_resolver if mesh_resolver is not None else _default_resolver
    root = ET.fromstring(urdf)
    links = {str(link.get("name")): link for link in root.findall("link")}
    if base_link not in links:
        return None
    poses = _link_poses(root, base_link)
    heights: list[float] = []
    for name, pose in poses.items():
        link = links.get(name)
        if link is None:
            continue
        for collision in link.findall("collision"):
            geometry = collision.find("geometry")
            if geometry is None:
                continue
            placed = pose @ _transform(collision.find("origin"))
            sphere = geometry.find("sphere")
            if sphere is not None:
                radius = float(sphere.get("radius", 0.0))
                heights.extend((float(placed[2, 3]) - radius, float(placed[2, 3]) + radius))
                continue
            points = _geometry_points(geometry, placed, resolver)
            if points is not None:
                heights.extend(float(z) for z in points[:, 2])
    if not heights:
        return None
    heights.extend(float(poses[frame][2, 3]) for frame in set(sensor_frames(urdf).values()) if frame in poses)
    return _rounded(max(heights) - min(heights))
