from __future__ import annotations

import math
import shutil
from collections.abc import Callable
from pathlib import Path

import pytest
from arena_robots.assembly import RequestPart
from arena_robots.caps import ArmSpec, RobotCaps
from arena_robots.override_paths import override_paths
from arena_robots.Robot import RobotIdentifier

_XACRO = shutil.which("xacro")
_UR_MACRO = Path(__file__).resolve().parent.parent.parent / "deps" / "ur_description" / "urdf" / "ur_macro.xacro"
_JOINTS = [f"top_{name}_joint" for name in ("shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "wrist_3")]
_HALF = {"joint_limits.*.max_velocity": {"scale": 0.5}}


def _jackal_ur5e(**caps: object) -> RobotCaps:
    view = RobotIdentifier("jackal").resolve_sync()
    return view.effective_caps({"arm": [RequestPart(variant="ur5e", mount="top", caps=dict(caps))]})


def _arm(caps: RobotCaps) -> ArmSpec:
    arms = caps.arm
    assert arms is not None
    return arms["top"]


class TestJointLimits:
    def test_joints_are_the_component_file_keyed_by_rendered_joint_name(self):
        joints = _arm(_jackal_ur5e()).joint_limits
        assert sorted(joints) == sorted(_JOINTS)
        assert joints["top_shoulder_pan_joint"]["max_velocity"] == pytest.approx(math.pi)
        assert joints["top_wrist_1_joint"]["max_acceleration"] == pytest.approx(15.0)

    def test_wildcard_override_halves_every_max_velocity_and_nothing_else(self):
        plain = _arm(_jackal_ur5e()).joint_limits
        halved = _arm(_jackal_ur5e(**_HALF)).joint_limits
        assert sorted(halved) == sorted(_JOINTS)
        for joint in _JOINTS:
            assert halved[joint]["max_velocity"] == pytest.approx(plain[joint]["max_velocity"] * 0.5)
            assert halved[joint]["max_acceleration"] == plain[joint]["max_acceleration"]
            assert halved[joint]["max_effort"] == plain[joint]["max_effort"]

    def test_one_joint_takes_a_value_of_its_own(self):
        joints = _arm(_jackal_ur5e(**{"joint_limits.top_elbow_joint.max_velocity": 1.0})).joint_limits
        assert joints["top_elbow_joint"]["max_velocity"] == 1.0
        assert joints["top_shoulder_pan_joint"]["max_velocity"] == pytest.approx(math.pi)

    def test_an_unknown_joint_raises(self):
        with pytest.raises(ValueError, match="unknown field 'no_such_joint'"):
            _arm(_jackal_ur5e(**{"joint_limits.no_such_joint.max_velocity": 1.0}))

    def test_only_an_override_marks_the_joints_overridden(self):
        assert not _jackal_ur5e().joints_overridden("top")
        assert _jackal_ur5e(**_HALF).joints_overridden("top")

    def test_a_static_arm_exposes_its_file_limits(self):
        arm = RobotIdentifier("tiago").resolve_sync().effective_caps({}).arm
        assert arm is not None
        assert next(iter(arm.values())).joint_limits


class TestJointPaths:
    def test_every_joint_key_is_an_override_path_inside_the_part_brackets(self):
        found = {item.path: item for item in override_paths(_jackal_ur5e())}
        item = found["arm.top.joint_limits.top_elbow_joint.max_velocity"]
        assert item.part == "top"
        assert item.relative == "joint_limits.top_elbow_joint.max_velocity"
        assert item.provenance == "component"
        assert item.overridable

    def test_an_overridden_joint_key_has_the_override_provenance(self):
        found = {item.path: item for item in override_paths(_jackal_ur5e(**_HALF))}
        assert found["arm.top.joint_limits.top_elbow_joint.max_velocity"].provenance == "override"
        assert found["arm.top.joint_limits.top_elbow_joint.max_acceleration"].provenance == "component"

    def test_arm_arguments_are_listed_with_the_arg_kind(self):
        found = {item.path: item for item in override_paths(_jackal_ur5e())}
        item = found["arm.top.safety_limits"]
        assert (item.kind, item.part, item.relative) == ("arg", "top", "safety_limits")
        assert item.value is False


@pytest.mark.skipif(_XACRO is None, reason="xacro CLI not on PATH, run under the Arena container")
@pytest.mark.skipif(not _UR_MACRO.is_file(), reason="deps/ur_description not initialized, run `arena feature robots add arm/ur`")
class TestJointLimitsAgainstTheUrdf:
    @pytest.fixture(autouse=True)
    def _rendering(self, require_installed: Callable[[str], None]) -> None:
        pytest.importorskip("xacro", reason="xacro is not importable, run under the Arena container")
        require_installed("arm/ur")

    def test_a_tighter_override_validates(self):
        _jackal_ur5e(**_HALF).validate_overrides()

    def test_an_override_above_the_urdf_velocity_raises(self):
        caps = _jackal_ur5e(**{"joint_limits.top_shoulder_pan_joint.max_velocity": 10.0})
        with pytest.raises(ValueError, match=r"top_shoulder_pan_joint.*above the URDF limit"):
            caps.validate_overrides()

    def test_an_override_above_the_urdf_effort_raises(self):
        caps = _jackal_ur5e(**{"joint_limits.top_shoulder_pan_joint.max_effort": 1.0e6})
        with pytest.raises(ValueError, match="max_effort"):
            caps.validate_overrides()

    def test_a_file_value_above_the_urdf_is_not_an_override_error(self):
        caps = _jackal_ur5e(**{"joint_limits.top_shoulder_pan_joint.max_acceleration": 5.0})
        caps.validate_overrides()
