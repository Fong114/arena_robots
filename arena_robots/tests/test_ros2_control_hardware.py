from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import pytest
from arena_robots.Robot import RobotIdentifier
from arena_robots.served_urdf import serve

PKG = Path(__file__).resolve().parents[2] / "arena_robots"
ALL_ROBOTS = sorted(p.name for p in (PKG / "robots").iterdir() if (p / "model_params.yaml").is_file())


@pytest.mark.parametrize("robot", ALL_ROBOTS)
def test_every_ros2_control_block_runs_on_the_gazebo_sim_system(robot, require_installed):
    pytest.importorskip("xacro")
    require_installed(robot)
    view = RobotIdentifier(robot).resolve_sync()
    root = ET.fromstring(serve(view, {}).description)
    plugins = [el.findtext("hardware/plugin", "").strip() for el in root.findall("ros2_control")]
    assert plugins
    assert set(plugins) == {"gz_ros2_control/GazeboSimSystem"}, plugins
