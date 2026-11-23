"""The URDF of a robot instance: its canonical form for derived facts and the served file the simulators load."""

from __future__ import annotations

import functools
import hashlib
import json
import os
import sys
import tempfile
import typing
import xml.etree.ElementTree as ET
from collections.abc import Collection, Mapping, Sequence
from pathlib import Path

import attrs

if typing.TYPE_CHECKING:
    from arena_robots.assembly import RequestPart
    from arena_robots.caps import DriveSpec
    from arena_robots.Robot import RobotView

_PACKAGE_URI = "package://"

ET.register_namespace("gz", "http://gazebosim.org/schema")

_OPTIM_MAP: dict[str, frozenset[str]] = {
    'no_camera': frozenset({'camera', 'depth', 'rgbd_camera'}),
    'no_lidar': frozenset({'ray', 'gpu_lidar'}),
}


@attrs.frozen
class ServedURDF:
    description: str
    path: Path


def _strip_sensors(root: ET.Element, tokens: Collection[str]) -> None:
    disabled_types: set[str] = set()
    unknown: set[str] = set()
    for tok in tokens:
        m = _OPTIM_MAP.get(tok)
        if m is None:
            unknown.add(tok)
        else:
            disabled_types |= m
    if unknown:
        print(f"[urdf.optim] ignoring unknown token(s): {sorted(unknown)}", file=sys.stderr)
    if not disabled_types:
        return
    for parent in root.iter():
        for child in list(parent):
            if child.tag.rpartition('}')[-1] == 'sensor' and child.attrib.get('type') in disabled_types:
                parent.remove(child)


def _patch_sensor_topics(root: ET.Element, patches: Sequence[tuple[str, str, str]]) -> None:
    for sensor_name, child_path, value in patches:
        sensor = next(
            (elem for elem in root.iter() if elem.tag.rpartition('}')[-1] == 'sensor' and elem.attrib.get('name') == sensor_name),
            None,
        )
        if sensor is None:
            print(f"[urdf.sensor_topics] no <sensor name={sensor_name!r}> in this URDF variant, skipping patch", file=sys.stderr)
            continue
        node = sensor
        for part in child_path.split('/'):
            child = next((c for c in node if c.tag.rpartition('}')[-1] == part), None)
            if child is None:
                child = ET.SubElement(node, part)
            node = child
        node.text = value


def _ensure_effort_state(root: ET.Element) -> None:
    for control in root.iter():
        if control.tag.rpartition('}')[-1] != 'ros2_control':
            continue
        for joint in control:
            if joint.tag.rpartition('}')[-1] != 'joint':
                continue
            has_effort = any(child.tag.rpartition('}')[-1] == 'state_interface' and child.attrib.get('name') == 'effort' for child in joint)
            if not has_effort:
                ET.SubElement(joint, 'state_interface', {'name': 'effort'})


_GAZEBO_ROS2_CONTROL_PLUGIN = 'gz_ros2_control/GazeboSimSystem'
DRIVE_PARAMS = ('kp', 'kd', 'effort')


def _apply_drives(root: ET.Element, drives: Mapping[str, DriveSpec]) -> None:
    """Write each drive's gains and effort limit onto its ``<ros2_control>`` joint, and swap its position command interface for effort when the PD controller commands it."""
    for control in root.iter('ros2_control'):
        for joint in control.findall('joint'):
            drive = drives.get(str(joint.get('name')))
            if drive is None:
                continue
            for param in [el for el in joint.findall('param') if el.get('name') in DRIVE_PARAMS]:
                joint.remove(param)
            for name in DRIVE_PARAMS:
                ET.SubElement(joint, 'param', {'name': name}).text = repr(float(getattr(drive, name)))
            if drive.command == 'effort':
                for command in joint.findall('command_interface'):
                    if command.get('name') == 'position':
                        command.clear()
                        command.set('name', 'effort')


def _control_plugin_text(control: ET.Element) -> str | None:
    plugin = next((el for el in control.iter() if el.tag.rpartition('}')[-1] == 'plugin'), None)
    return plugin.text if plugin is not None else None


_LOCKABLE_JOINT_TYPES = frozenset({'prismatic', 'revolute'})
_JOINT_MOTION_CHILDREN = frozenset({'axis', 'limit', 'dynamics', 'calibration', 'safety_controller'})


def _lock_passive_joints(root: ET.Element) -> None:
    """Convert every ``<joint>`` with ``<limit effort="0">`` to a fixed joint.
    Zero effort makes a joint unactuatable per the URDF spec, so it only exists as
    passive compliance (e.g. create3's sprung wheel_drop suspension), and neither
    simulator models the spring: the joint collapses under gravity and the chassis
    beaches on its belly."""
    for joint in root.iter('joint'):
        if joint.attrib.get('type') not in _LOCKABLE_JOINT_TYPES:
            continue
        limit = joint.find('limit')
        if limit is None or float(limit.attrib.get('effort', 'inf')) != 0.0:
            continue
        joint.set('type', 'fixed')
        for child in [el for el in joint if el.tag in _JOINT_MOTION_CHILDREN]:
            joint.remove(child)
        print(f"[urdf.joints] locked passive zero-effort joint {joint.attrib.get('name')!r}", file=sys.stderr)


_HW_PLUGIN_RESOURCE = 'hardware_interface__pluginlib__plugin'


@functools.cache
def _registered_hardware_plugins() -> frozenset[str]:
    """Hardware plugin classes declared in the ament index, i.e. exactly the set a
    controller_manager in this environment can load."""
    try:
        from ament_index_python.resources import get_resource, get_resources
    except ModuleNotFoundError:
        return frozenset()
    names: set[str] = set()
    for pkg in get_resources(_HW_PLUGIN_RESOURCE):
        content, prefix = get_resource(_HW_PLUGIN_RESOURCE, pkg)
        for rel in filter(None, (line.strip() for line in content.splitlines())):
            try:
                manifest = ET.parse(os.path.join(prefix, rel))
            except (OSError, ET.ParseError) as e:
                print(f"[urdf.ros2_control] skipping unreadable plugin manifest of {pkg}: {e}", file=sys.stderr)
                continue
            for cls in manifest.getroot().iter('class'):
                name = cls.attrib.get('name') or cls.attrib.get('type')
                if name:
                    names.add(name)
    return frozenset(names)


def _strip_unregistered_ros2_control(root: ET.Element, registered: frozenset[str]) -> None:
    """Drop every top-level ``<ros2_control>`` block whose hardware plugin is not in
    ``registered`` (vendored descriptions leak gazebo-classic blocks that crash the
    external controller_manager), plus the classic ``libgazebo_ros2_control.so`` host
    plugin. The canonical ``gz_ros2_control/GazeboSimSystem`` block always survives:
    gazebo consumes it in-sim, the isaac adapter rewrites it. No-op when ``registered``
    is empty (no ament index to trust)."""
    if not registered:
        print("[urdf.ros2_control] no registered hardware plugins discovered, keeping all <ros2_control> blocks", file=sys.stderr)
        return
    keep = registered | {_GAZEBO_ROS2_CONTROL_PLUGIN}
    for control in [el for el in root if el.tag.rpartition('}')[-1] == 'ros2_control']:
        plugin = _control_plugin_text(control)
        if plugin is None or plugin.strip() not in keep:
            root.remove(control)
            print(f"[urdf.ros2_control] stripped <ros2_control name={control.attrib.get('name')!r}>: no loadable hardware plugin ({plugin!r})", file=sys.stderr)
    for gazebo in [el for el in root if el.tag.rpartition('}')[-1] == 'gazebo']:
        classic = [el for el in gazebo if el.tag.rpartition('}')[-1] == 'plugin' and el.attrib.get('filename') == 'libgazebo_ros2_control.so']
        if not classic:
            continue
        for plugin_el in classic:
            gazebo.remove(plugin_el)
        if len(gazebo) == 0:
            root.remove(gazebo)
        print("[urdf.ros2_control] stripped classic libgazebo_ros2_control.so plugin element", file=sys.stderr)


def _joint_element(joint: dict) -> ET.Element:
    el = ET.Element('joint', {'name': str(joint['name'])})
    if joint.get('mimic'):
        el.set('mimic', 'true')
    for iface in joint.get('command_interfaces', []):
        ET.SubElement(el, 'command_interface', {'name': str(iface)})
    for iface in joint.get('state_interfaces', []):
        state = ET.SubElement(el, 'state_interface', {'name': str(iface)})
        if iface == 'position' and joint.get('initial_value') is not None:
            ET.SubElement(state, 'param', {'name': 'initial_value'}).text = str(joint['initial_value'])
    return el


def _inject_ros2_control_joints(root: ET.Element, joints: Sequence[dict]) -> None:
    """Merge every top-level ``<ros2_control>`` tag in ``root`` into the one whose
    hardware plugin is ``gz_ros2_control/GazeboSimSystem`` (the chassis-emitted tag;
    arena_robots.catalog.render_wrapper_xacro renders the chassis and every arm
    placement's component normally, so a joint-bearing part's xacro may add its own
    native ``ros2_control`` tag too). Appends ``joints`` (a resolved arm placement's
    control-joint patch, ``arena_robots.catalog.render_control_joints``, computed from
    ``resolved_assembly`` independently of this URDF's rendered content) as ``<joint>``
    elements onto the chassis tag, then drops every other ``<ros2_control>`` tag (an
    arm component's own native tag, superseded by the injected patch). No-op when
    ``joints`` is empty."""
    if not joints:
        return
    control_tags = [el for el in root if el.tag.rpartition('}')[-1] == 'ros2_control']
    chassis_tag = next((tag for tag in control_tags if _control_plugin_text(tag) == _GAZEBO_ROS2_CONTROL_PLUGIN), None)
    if chassis_tag is None:
        raise RuntimeError(f"no <ros2_control> tag with <plugin>{_GAZEBO_ROS2_CONTROL_PLUGIN}</plugin> to merge the control-joint patch into")
    for joint in joints:
        chassis_tag.append(_joint_element(joint))
    for tag in control_tags:
        if tag is not chassis_tag:
            root.remove(tag)


def _absolute_mesh_uris(root: ET.Element, base_dir: str) -> None:
    for elem in root.iter():
        if 'filename' not in elem.attrib:
            continue
        if elem.tag.rpartition('}')[-1] == 'plugin':
            continue
        original_path = elem.attrib['filename']

        if original_path.startswith(_PACKAGE_URI):
            pkg, _, sub = original_path[len(_PACKAGE_URI) :].partition('/')
            abs_path: str | None = None
            try:
                from ament_index_python.packages import PackageNotFoundError, get_package_share_directory
            except ModuleNotFoundError:
                pass
            else:
                try:
                    abs_path = os.path.join(get_package_share_directory(pkg), sub)
                except PackageNotFoundError:
                    pass
            if abs_path is None:
                abs_path = os.path.abspath(os.path.join(base_dir, sub))
            elem.attrib['filename'] = f"file://{abs_path}"
            continue

        if original_path.startswith("file://"):
            continue

        if not os.path.isabs(original_path):
            elem.attrib['filename'] = f"file://{os.path.abspath(os.path.join(base_dir, original_path))}"
            continue

        elem.attrib['filename'] = f"file://{original_path}"


def _xacro_value(value: object) -> str:
    if attrs.has(type(value)):
        value = attrs.asdict(value)
    if isinstance(value, dict):
        return json.dumps(value)
    return str(value)


def render_xacro_file(path: Path, mappings: Mapping[str, object]) -> str:
    """The xacro file at ``path`` rendered in process with ``mappings`` as its arguments."""
    import xacro

    return xacro.process_file(str(path), mappings={k: str(v) for k, v in mappings.items()}).toxml()  # pyright: ignore[reportAttributeAccessIssue]


def _render_xacro(view: RobotView, resolved: object, xacro_args: Mapping[str, object]) -> ET.Element:
    import xacro

    from arena_robots import urdf_facts
    from arena_robots.catalog import render_wrapper_xacro

    mappings = {k: _xacro_value(v) for k, v in xacro_args.items() if v is not None}
    if resolved is None:
        return ET.fromstring(xacro.process_file(str(urdf_facts.source(view)), mappings=mappings).toxml())  # pyright: ignore[reportAttributeAccessIssue]
    with tempfile.NamedTemporaryFile("w", suffix=".urdf.xacro", delete=False) as wrapper:
        wrapper.write(render_wrapper_xacro(view, resolved))  # pyright: ignore[reportArgumentType]
    try:
        return ET.fromstring(xacro.process_file(wrapper.name, mappings=mappings).toxml())  # pyright: ignore[reportAttributeAccessIssue]
    finally:
        Path(wrapper.name).unlink()


def _control_joints(view: RobotView, resolved: object) -> list[dict[str, object]]:
    from arena_robots.catalog import Catalog, render_control_joints

    if resolved is None or view.assembly is None:
        return []
    return render_control_joints(resolved, Catalog(), prefix=view.assembly.prefix)  # pyright: ignore[reportArgumentType]


def _document(view: RobotView, parts: dict[str, list[RequestPart]], frames: dict[str, str] | None, xacro_args: Mapping[str, object]) -> tuple[ET.Element, list[dict[str, object]]]:
    from arena_robots import urdf_patches
    from arena_robots.assembly import apply_frame_overrides

    resolved = view._resolved(parts)
    if resolved is not None:
        resolved = apply_frame_overrides(resolved, frames or {})
    root = _render_xacro(view, resolved, xacro_args)
    urdf_patches.apply(root, view, resolved)
    return root, _control_joints(view, resolved)


@attrs.frozen
class _Request:
    key: str
    view: RobotView = attrs.field(eq=False)
    parts: dict[str, list[RequestPart]] = attrs.field(eq=False)
    frames: dict[str, str] | None = attrs.field(eq=False)


def _request_key(view: RobotView, parts: dict[str, list[RequestPart]], frames: dict[str, str] | None) -> str:
    canonical = {part_type: sorted(json.dumps([part.variant, part.mount, part.params, part.overrides], sort_keys=True, default=str) for part in items) for part_type, items in parts.items()}
    return json.dumps([str(view.path), canonical, sorted((frames or {}).items())], sort_keys=True, default=str)


@functools.lru_cache(maxsize=256)
def _canonical(request: _Request) -> str:
    root, joints = _document(request.view, request.parts, request.frames, {})
    _lock_passive_joints(root)
    _inject_ros2_control_joints(root, joints)
    _ensure_effort_state(root)
    return ET.tostring(root, encoding="unicode")


def canonical(view: RobotView, parts: dict[str, list[RequestPart]], frames: dict[str, str] | None = None) -> str:
    """The instance's URDF before simulator options, memoized per view, parts and frames."""
    return _canonical(_Request(_request_key(view, parts, frames), view, parts, frames))


def mass_split(view: RobotView, parts: dict[str, list[RequestPart]], frames: dict[str, str] | None = None) -> dict[str, float]:
    """Inertial mass in kg of the chassis, the placed components and the placed payload, summing to the instance's URDF mass."""
    return dict(_mass_split(_Request(_request_key(view, parts, frames), view, parts, frames)))


@functools.lru_cache(maxsize=256)
def _mass_split(request: _Request) -> tuple[tuple[str, float], ...]:
    from arena_robots import urdf_patches
    from arena_robots.assembly import apply_frame_overrides

    resolved = request.view._resolved(request.parts)
    if resolved is not None:
        resolved = apply_frame_overrides(resolved, request.frames or {})
    placements = resolved.placements if resolved is not None else []
    owner = urdf_patches.owners(request.view, resolved)
    split = {"base": 0.0, "components": 0.0, "payload": 0.0}
    for link in ET.fromstring(_canonical(request)).findall("link"):
        index = owner(str(link.get("name")))
        key = "base" if index is None else "payload" if placements[index].type == "payload" else "components"
        split[key] += sum(float(mass.get("value", 0.0)) for mass in link.findall("inertial/mass"))
    return (*split.items(), ("total", sum(split.values())))


def serve(
    view: RobotView,
    parts: dict[str, list[RequestPart]],
    frames: dict[str, str] | None = None,
    *,
    xacro_args: Mapping[str, object] | None = None,
    optim: Collection[str] = (),
    sensor_topic_patches: Sequence[tuple[str, str, str]] = (),
    drives: Mapping[str, DriveSpec] | None = None,
) -> ServedURDF:
    """The URDF a simulator loads, written under ``<tempdir>/arena_robots/<sha256>.urdf``."""
    root, joints = _document(view, parts, frames, xacro_args or {})
    _absolute_mesh_uris(root, str(view.path / "urdf"))
    if optim:
        _strip_sensors(root, optim)
    _lock_passive_joints(root)
    _strip_unregistered_ros2_control(root, _registered_hardware_plugins())
    _inject_ros2_control_joints(root, joints)
    _ensure_effort_state(root)
    _apply_drives(root, drives or {})
    _patch_sensor_topics(root, sensor_topic_patches)

    data = ET.tostring(root, encoding="utf-8", method="xml", xml_declaration=True)
    return ServedURDF(description=data.decode("utf-8"), path=content_addressed(data, ".urdf"))


def content_addressed(data: bytes, suffix: str, stem: str = "") -> Path:
    """The file ``<tempdir>/arena_robots/<stem><sha256 of data><suffix>`` holding ``data``, written once."""
    directory = Path(tempfile.gettempdir()) / "arena_robots"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{stem}{hashlib.sha256(data).hexdigest()}{suffix}"
    if not path.is_file():
        staging = path.with_suffix(f".{os.getpid()}.tmp")
        staging.write_bytes(data)
        staging.replace(path)
    return path
