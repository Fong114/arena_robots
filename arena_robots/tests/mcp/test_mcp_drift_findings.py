from __future__ import annotations

from collections.abc import Callable

import pytest

pytest.importorskip("arena_mcp")

from arena_robots import mcp


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


def test_turtlebot_power_model_takes_its_patched_urdf_mass(require_installed: Callable[[str], None]):
    require_installed("turtlebot")
    view = mcp._view("turtlebot")
    profile = view.power_profile(view.effective_caps({}))
    assert profile is not None
    assert profile["robot_mass_kg"] == pytest.approx(3.945)


def test_rbvogui_swerve_geometry_comes_from_the_urdf(require_installed: Callable[[str], None]):
    require_installed("rbvogui")
    view = mcp._view("rbvogui")
    geometry = view.drive_geometry(view.effective_caps({}).rendered_urdf())
    assert geometry["wheel_radius"] == pytest.approx(0.1165)
    assert geometry["wheel_positions_y"] == [-0.235, 0.235, -0.235, 0.235]


def test_jackal_raises_no_drift_finding(require_installed: Callable[[str], None]):
    require_installed("jackal")
    assert _checks("jackal") == {}
