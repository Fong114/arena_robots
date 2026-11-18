"""Facts read from a rendered URDF: joint limits, lidar block, mass, wheel geometry and the ground outline of the base."""

from __future__ import annotations

import functools
import json
import logging
import math
import struct
import typing
import xml.etree.ElementTree as ET
from collections.abc import Callable
from pathlib import Path

import attrs
import numpy as np
from ament_index_python.packages import PackageNotFoundError, get_package_share_path

if typing.TYPE_CHECKING:
    from arena_robots.assembly import RequestPart
    from arena_robots.Robot import RobotView

_LOGGER = logging.getLogger(__name__)
_LIDAR_TYPES = frozenset({"gpu_lidar", "gpu_ray", "ray", "lidar"})
_FULL_TURN = 2.0 * math.pi
_ARC_SEGMENTS = 16


@attrs.frozen
class JointLimit:
    lower: float | None
    upper: float | None
    velocity: float | None
    effort: float | None


@attrs.frozen
class _RenderRequest:
    key: str
    view: RobotView = attrs.field(eq=False)
    parts: dict[str, list[RequestPart]] = attrs.field(eq=False)
    frames: dict[str, str] | None = attrs.field(eq=False)


def _request_key(view: RobotView, parts: dict[str, list[RequestPart]], frames: dict[str, str] | None) -> str:
    canonical = {part_type: sorted(json.dumps([part.variant, part.mount, part.params, part.overrides], sort_keys=True, default=str) for part in items) for part_type, items in parts.items()}
    return json.dumps([str(view.path), canonical, sorted((frames or {}).items())], sort_keys=True, default=str)


@functools.lru_cache(maxsize=256)
def _render(request: _RenderRequest) -> str:
    import tempfile

    import xacro

    from arena_robots.assembly import apply_frame_overrides
    from arena_robots.catalog import render_wrapper_xacro

    view = request.view
    resolved = view._resolved(request.parts)
    if resolved is None:
        return xacro.process_file(str(view.path / "urdf" / f"{view.name}.urdf.xacro")).toxml()  # pyright: ignore[reportAttributeAccessIssue]
    resolved = apply_frame_overrides(resolved, request.frames or {})
    with tempfile.NamedTemporaryFile("w", suffix=".urdf.xacro", delete=False) as wrapper:
        wrapper.write(render_wrapper_xacro(view, resolved))
    try:
        return xacro.process_file(wrapper.name).toxml()  # pyright: ignore[reportAttributeAccessIssue]
    finally:
        Path(wrapper.name).unlink()


def rendered_urdf(view: RobotView, parts: dict[str, list[RequestPart]], frames: dict[str, str] | None = None) -> str:
    """The instance's URDF as the simulators render it, memoized per view, parts and frames."""
    return _render(_RenderRequest(_request_key(view, parts, frames), view, parts, frames))


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


def wheel_geometry(urdf: str, base_link: str, left: list[str], right: list[str]) -> dict[str, float] | None:
    """Wheel radius and track width from the named wheel joints, None when a joint or radius is missing."""
    if not left or not right:
        return None
    root = ET.fromstring(urdf)
    links = {str(link.get("name")): link for link in root.findall("link")}
    joints = {str(joint.get("name")): joint for joint in root.findall("joint")}
    poses = _link_poses(root, base_link)
    lateral: dict[str, list[float]] = {"left": [], "right": []}
    radii: list[float] = []
    for side, names in (("left", left), ("right", right)):
        for name in names:
            joint = joints.get(name)
            if joint is None:
                return None
            parent = joint.find("parent")
            child = joint.find("child")
            if parent is None or child is None or str(parent.get("link")) not in poses or str(child.get("link")) not in links:
                return None
            origin = poses[str(parent.get("link"))] @ _transform(joint.find("origin"))
            lateral[side].append(float(origin[1, 3]))
            radius = _collision_radius(links[str(child.get("link"))])
            if radius is None:
                return None
            radii.append(radius)
    separation = abs(sum(lateral["left"]) / len(lateral["left"]) - sum(lateral["right"]) / len(lateral["right"]))
    return {"wheel_radius": sum(radii) / len(radii), "wheel_separation": separation}


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
