"""A per-instance override of an arm's joint limits reaches the MoveIt parameters build_moveit_params renders."""

from __future__ import annotations

import shutil
from collections.abc import Callable

import pytest
from arena_robots.assembly import RequestPart
from arena_robots.request import InstanceRequest

_XACRO = shutil.which("xacro")
_JOINTS = [f"top_{name}_joint" for name in ("shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "wrist_3")]


def _limits(require_installed: Callable[[str], None], **caps: object) -> dict[str, dict[str, object]]:
    from arena_robots.moveit_factory import build_moveit_params

    require_installed("arm/ur")
    params = build_moveit_params("jackal", tf_prefix="", request=InstanceRequest(parts={"arm": [RequestPart(variant="ur5e", mount="top", caps=dict(caps))]}), instance="top")
    assert params is not None
    return params["robot_description_planning"]["joint_limits"]


@pytest.mark.skipif(_XACRO is None, reason="xacro CLI not on PATH, run under the Arena container")
class TestMoveitJointOverrides:
    def test_half_velocity_override_halves_every_rendered_joint_limit(self, require_installed: Callable[[str], None]):
        plain = _limits(require_installed)
        halved = _limits(require_installed, **{"joint_limits.*.max_velocity": {"scale": 0.5}})
        for joint in _JOINTS:
            assert halved[joint]["max_velocity"] == pytest.approx(plain[joint]["max_velocity"] * 0.5)
            assert halved[joint]["max_acceleration"] == plain[joint]["max_acceleration"]

    def test_an_instance_without_a_joint_override_renders_the_component_file(self, require_installed: Callable[[str], None]):
        assert _limits(require_installed)["top_shoulder_pan_joint"]["max_velocity"] == pytest.approx(3.141592653589793)
