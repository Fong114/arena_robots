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


def test_boxer_weighs_its_vendor_mass(require_installed: Callable[[str], None]):
    require_installed("boxer")
    assert "laser" not in _checks("boxer")
    assert mcp._view("boxer").effective_caps({}).body.mass == pytest.approx(107.0)


def test_turtlebot_power_model_takes_its_upstream_urdf_mass(require_installed: Callable[[str], None]):
    require_installed("turtlebot")
    view = mcp._view("turtlebot")
    profile = view.power_profile(view.effective_caps({}))
    assert profile is not None
    assert profile["robot_mass_kg"] == pytest.approx(4.454)


def test_rbvogui_wheels_match_the_urdf(require_installed: Callable[[str], None]):
    require_installed("rbvogui")
    assert "control.wheels" not in _checks("rbvogui")


def test_jackal_raises_no_drift_finding(require_installed: Callable[[str], None]):
    require_installed("jackal")
    assert _checks("jackal") == {}


@pytest.mark.skipif(_XACRO is None, reason="xacro CLI not on PATH, run under the Arena container")
@pytest.mark.skipif(not _UR_MACRO.is_file(), reason="deps/ur_description not initialized, run `arena feature robots add arm/ur`")
def test_ur5e_joint_limits_stay_within_the_urdf(require_installed: Callable[[str], None]):
    require_installed("arm/ur")
    assert "joint_limits" not in _checks("jackal", {"top": "arm/ur5e"})
