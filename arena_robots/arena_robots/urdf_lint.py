"""Lint of the canonical URDF of a robot instance: structure, inertia, wheels, joint limits and sensor frames."""

from __future__ import annotations

import typing
import xml.etree.ElementTree as ET
from collections.abc import Mapping, Sequence
from typing import Any

import attrs
import numpy as np

if typing.TYPE_CHECKING:
    from arena_robots.assembly import RequestPart
    from arena_robots.Robot import RobotView

TRACE_FACTOR = 5.0
_TRACE_MIN_MASS = 0.05


class Finding(typing.NamedTuple):
    check: str
    severity: typing.Literal["error", "warning"]
    message: str


@attrs.frozen
class Subject:
    """Everything the lint reads besides the URDF."""

    base_frame: str
    drive: tuple[str, Mapping[str, Any]] | None = None
    sensor_frames: Sequence[str] = ()


def lint(view: RobotView, parts: dict[str, list[RequestPart]] | None = None, frames: dict[str, str] | None = None) -> list[Finding]:
    """Findings on `served_urdf.canonical(view, parts, frames)`."""
    from arena_robots import served_urdf

    request = parts or {}
    return lint_urdf(served_urdf.canonical(view, request, frames), _subject(view, request, frames))


def _subject(view: RobotView, parts: dict[str, list[RequestPart]], frames: dict[str, str] | None) -> Subject:
    controller = view.drive_controller() if view.model_params.control is not None else None
    return Subject(
        base_frame=view.model_params.base_frame,
        drive=None if controller is None else (controller[1], controller[2]),
        sensor_frames=[sensor.frame for sensor in view.effective_sensors(parts, frames=frames)],
    )


def lint_urdf(urdf: str, subject: Subject) -> list[Finding]:
    """Findings on a URDF document against what `subject` says about its robot."""
    root = ET.fromstring(urdf)
    findings = _structure(root, subject)
    if not any(finding.check == "structure" and finding.severity == "error" for finding in findings):
        findings += _inertia(root)
        findings += _wheels(root, subject)
        findings += _joints(root)
        findings += _sensors(root, subject)
    return findings


def _links(root: ET.Element) -> dict[str, ET.Element]:
    return {str(link.get("name")): link for link in root.findall("link")}


def _structure(root: ET.Element, subject: Subject) -> list[Finding]:
    findings: list[Finding] = []
    names = [str(link.get("name")) for link in root.findall("link")]
    links = set(names)
    for name in sorted({name for name in names if names.count(name) > 1}):
        findings.append(Finding("structure", "error", f"link {name} is defined more than once"))
    children: dict[str, str] = {}
    for joint in root.findall("joint"):
        name = joint.get("name")
        parent = joint.find("parent")
        child = joint.find("child")
        parent_name = None if parent is None else str(parent.get("link"))
        child_name = None if child is None else str(child.get("link"))
        if parent_name not in links:
            findings.append(Finding("structure", "error", f"joint {name} has the parent {parent_name}, which is not a link"))
        if child_name not in links:
            findings.append(Finding("structure", "error", f"joint {name} has the child {child_name}, which is not a link"))
        elif child_name in children:
            findings.append(Finding("structure", "error", f"link {child_name} is the child of more than one joint"))
        elif parent_name is not None:
            children[child_name] = parent_name
    roots = sorted(links - children.keys())
    if len(roots) != 1:
        findings.append(Finding("structure", "error", f"the URDF has {len(roots)} root links {roots}, expected one"))
    if subject.base_frame not in links:
        findings.append(Finding("structure", "error", f"the base frame {subject.base_frame} is not a link"))
    return findings


def _static_links(root: ET.Element) -> set[str]:
    joints = root.findall("joint")
    children = {str(child.get("link")) for joint in joints if (child := joint.find("child")) is not None}
    static = {str(link.get("name")) for link in root.findall("link")} - children
    grew = True
    while grew:
        grew = False
        for joint in joints:
            parent = joint.find("parent")
            child = joint.find("child")
            if joint.get("type") == "fixed" and parent is not None and child is not None and str(parent.get("link")) in static and str(child.get("link")) not in static:
                static.add(str(child.get("link")))
                grew = True
    return static


def _tensor(inertia: ET.Element) -> np.ndarray:
    def value(key: str) -> float:
        return float(inertia.get(key, 0.0))

    return np.array([[value("ixx"), value("ixy"), value("ixz")], [value("ixy"), value("iyy"), value("iyz")], [value("ixz"), value("iyz"), value("izz")]])


def _primitive_trace(link: ET.Element, mass: float) -> tuple[str, float] | None:
    shapes = [geometry[0] for collision in link.findall("collision") for geometry in collision.findall("geometry") if len(geometry)]
    if len(shapes) != 1:
        return None
    shape = shapes[0]
    if shape.tag == "box":
        x, y, z = (float(token) for token in str(shape.get("size")).split())
        return "box", mass * 2 * (x * x + y * y + z * z) / 12
    if shape.tag == "cylinder":
        radius, length = float(shape.get("radius", 0.0)), float(shape.get("length", 0.0))
        return "cylinder", mass * (2 * (3 * radius * radius + length * length) / 12 + radius * radius / 2)
    if shape.tag == "sphere":
        radius = float(shape.get("radius", 0.0))
        return "sphere", mass * 6 * radius * radius / 5
    return None


def _inertia(root: ET.Element) -> list[Finding]:
    findings: list[Finding] = []
    static = _static_links(root)
    for name, link in _links(root).items():
        inertial = link.find("inertial")
        if inertial is None:
            continue
        mass_element = inertial.find("mass")
        inertia = inertial.find("inertia")
        mass = float(mass_element.get("value", 0.0)) if mass_element is not None else 0.0
        issues: list[str] = []
        if mass <= 0.0 and name not in static:
            issues.append(f"mass {mass:g} is not above 0 on a link that moves")
        if mass > 0.0 and inertia is not None:
            eigenvalues = np.linalg.eigvalsh(_tensor(inertia))
            low, middle, high = (float(value) for value in eigenvalues)
            if low <= 0.0:
                issues.append(f"inertia tensor is not positive definite (smallest eigenvalue {low:.3g})")
            elif low + middle < high * (1 - 1e-6):
                issues.append(f"inertia principal moments {low:.3g} + {middle:.3g} < {high:.3g} violate the triangle inequality")
            elif mass > _TRACE_MIN_MASS:
                primitive = _primitive_trace(link, mass)
                trace = low + middle + high
                if primitive is not None and primitive[1] > 0.0 and not 1 / TRACE_FACTOR <= trace / primitive[1] <= TRACE_FACTOR:
                    issues.append(f"inertia trace {trace:.3g} is x{trace / primitive[1]:.2g} that of its {primitive[0]} collision ({primitive[1]:.3g})")
        for issue in issues:
            findings.append(Finding("inertia", "error", f"link {name}: {issue}"))
    return findings


def _wheels(root: ET.Element, subject: Subject) -> list[Finding]:
    if subject.drive is None:
        return []
    from arena_robots import urdf_facts

    kind, params = subject.drive
    try:
        names = urdf_facts.wheel_joints(kind, params)
    except KeyError as error:
        return [Finding("wheels", "error", f"the {kind} parameters lack {error}")]
    joints = {str(joint.get("name")): joint for joint in root.findall("joint")}
    links = _links(root)
    findings: list[Finding] = []
    for name in names:
        joint = joints.get(name)
        if joint is None:
            findings.append(Finding("wheels", "error", f"wheel joint {name} is not in the URDF"))
            continue
        if joint.get("type") != "continuous":
            findings.append(Finding("wheels", "error", f"wheel joint {name} is {joint.get('type')}, expected continuous"))
        child = joint.find("child")
        link = None if child is None else links.get(str(child.get("link")))
        if link is None or urdf_facts._collision_radius(link) is None:
            findings.append(Finding("wheels", "error", f"wheel joint {name} has no round collision on its child link"))
    return findings


def _joints(root: ET.Element) -> list[Finding]:
    from arena_robots import urdf_facts

    findings: list[Finding] = []
    limits = urdf_facts.joint_limits(ET.tostring(root, encoding="unicode"))
    for joint in root.findall("joint"):
        name = str(joint.get("name"))
        if joint.get("type") not in ("revolute", "prismatic"):
            continue
        issues: list[str] = []
        limit = limits[name]
        if joint.find("limit") is None:
            issues.append(f"{joint.get('type')} joint {name} has no limit")
        else:
            if limit.effort is None or limit.effort <= 0.0:
                issues.append(f"joint {name} effort {limit.effort} is not above 0")
            if limit.velocity is None or limit.velocity <= 0.0:
                issues.append(f"joint {name} velocity {limit.velocity} is not above 0")
            if limit.lower is not None and limit.upper is not None and limit.lower > limit.upper:
                issues.append(f"joint {name} lower {limit.lower:g} exceeds upper {limit.upper:g}")
        for issue in issues:
            findings.append(Finding("joints", "error", issue))
    return findings


def _sensors(root: ET.Element, subject: Subject) -> list[Finding]:
    links = _links(root)
    return [Finding("sensors", "error", f"sensor frame {frame} is not a link") for frame in dict.fromkeys(subject.sensor_frames) if frame not in links]
