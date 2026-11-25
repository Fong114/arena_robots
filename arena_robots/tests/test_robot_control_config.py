from __future__ import annotations

from pathlib import Path

import yaml
from arena_robots.Robot import RobotView

BASE = {
    "controller_manager": {"ros__parameters": {"update_rate": 50}},
    "base_controller": {
        "ros__parameters": {
            "wheel_separation": 0.4,
            "wheel_separation_multiplier": 1.5,
            "odom_frame_id": "odom",
        }
    },
}
GAZEBO_OVERLAY = {"base_controller": {"ros__parameters": {"wheel_separation_multiplier": 1.1}}}


def _robot(tmp_path: Path, *, overlay: bool) -> RobotView:
    root = tmp_path / "bot"
    root.mkdir()
    (root / "control.yaml").write_text(yaml.safe_dump(BASE))
    if overlay:
        (root / "control.gazebo.yaml").write_text(yaml.safe_dump(GAZEBO_OVERLAY))
    (root / "model_params.yaml").write_text(yaml.safe_dump({"control": {"mode": "ros2_control", "controllers": ["base_controller"], "config": str(root / "control.yaml")}}))
    return RobotView(root)


def test_sim_overlay_replaces_only_the_keys_it_names(tmp_path):
    loaded = _robot(tmp_path, overlay=True).control_config("gazebo")
    assert loaded["base_controller"]["ros__parameters"] == {"wheel_separation": 0.4, "wheel_separation_multiplier": 1.1, "odom_frame_id": "odom"}
    assert loaded["controller_manager"] == BASE["controller_manager"]


def test_other_sims_and_no_sim_ignore_the_overlay(tmp_path):
    view = _robot(tmp_path, overlay=True)
    assert view.control_config("mujoco") == BASE
    assert view.control_config() == BASE


def test_missing_overlay_loads_the_base_config(tmp_path):
    assert _robot(tmp_path, overlay=False).control_config("gazebo") == BASE
