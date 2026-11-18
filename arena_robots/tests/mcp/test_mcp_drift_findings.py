from __future__ import annotations

import shutil
from collections.abc import Callable
from pathlib import Path

import pytest

pytest.importorskip("arena_mcp")

from arena_robots import mcp

_XACRO = shutil.which("xacro")
_UR_MACRO = Path(__file__).resolve().parents[3] / "deps" / "ur_description" / "urdf" / "ur_macro.xacro"


def _checks(model: str, parts: dict | None = None) -> dict[str, list[str]]:
    view = mcp._view(model)
    findings = mcp._definition_findings(view, mcp._request(view, parts, None, None))
    out: dict[str, list[str]] = {}
    for finding in findings:
        assert finding["severity"] == "warning", finding
        out.setdefault(finding["check"], []).append(finding["message"])
    return out


@pytest.fixture(autouse=True)
def _rendering() -> None:
    pytest.importorskip("xacro", reason="xacro is not importable, run under the Arena container")


def test_boxer_warns_about_its_rear_lidar_and_its_masses(require_installed: Callable[[str], None]):
    require_installed("boxer")
    checks = _checks("boxer")
    assert "laser" in checks
    assert len(checks["caps.mass"]) == 1
    assert "URDF 451" in checks["caps.mass"][0]
    assert "robot_mass_kg 115" in checks["caps.mass"][0]


def test_rbvogui_warns_about_its_wheel_radius(require_installed: Callable[[str], None]):
    require_installed("rbvogui")
    assert any("wheel_radius 0.1125" in message for message in _checks("rbvogui")["control.wheels"])


def test_jackal_raises_no_drift_finding(require_installed: Callable[[str], None]):
    require_installed("jackal")
    assert _checks("jackal") == {}


@pytest.mark.skipif(_XACRO is None, reason="xacro CLI not on PATH, run under the Arena container")
@pytest.mark.skipif(not _UR_MACRO.is_file(), reason="deps/ur_description not initialized, run `arena feature robots add arm/ur`")
def test_ur5e_wrists_exceed_the_urdf_velocity(require_installed: Callable[[str], None]):
    require_installed("arm/ur")
    messages = _checks("jackal", {"top": "arm/ur5e"})["joint_limits"]
    assert any("top_wrist_1_joint" in message and "max_velocity" in message for message in messages)
    assert not any("top_shoulder_pan_joint" in message for message in messages)
