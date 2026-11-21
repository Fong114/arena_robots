"""Tests for RobotView.effective_caps and effective_sensors (allocation-derived caps)."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml


def _write_robot(
    tmp_path: Path,
    *,
    model_params: dict,
    assembly: dict | None = None,
    mobile_cap: dict | None = None,
) -> Path:
    rd = tmp_path / "robot"
    rd.mkdir()
    (rd / "model_params.yaml").write_text(yaml.dump(model_params))
    if assembly is not None:
        (rd / "assembly.yaml").write_text(yaml.dump(assembly))
    if mobile_cap is not None:
        (rd / "caps.yaml").write_text(yaml.dump({"mobile": mobile_cap}))
    return rd


class TestEffectiveCapsNoAssembly:
    def test_no_assembly_reads_the_model_caps_through_the_view(self, tmp_path: Path):
        from arena_robots.Robot import RobotView

        rd = _write_robot(tmp_path, model_params={}, mobile_cap={"odom_frame": "odom"})
        view = RobotView(rd)
        assert view.assembly is None
        caps = view.effective_caps({})
        assert caps.view is view
        assert caps.robot_dir == view.caps.robot_dir
        assert caps.overrides == {}
        assert caps.mobile.raw == view.caps.mobile.raw == {"odom_frame": "odom"}


class TestEffectiveCapsRbtheron:
    """rbtheron declares sensors via assembly.yaml but declares no arm mount."""

    def test_available_unchanged_and_no_arm(self):
        from arena_robots.Robot import RobotIdentifier

        view = RobotIdentifier("rbtheron").resolve_sync()
        assert view.assembly is not None
        caps = view.effective_caps({})
        assert caps.available == view.caps.available
        assert caps.arm is None


class TestEffectiveSensorsOfAssemblyRobots:
    """An assembly renders its sensors, whatever model_params declares."""

    _ASSEMBLY = {
        "mounts": {"front_laser": {"parent": "base_link", "xyz": [0, 0, 0], "accepts": ["lidar"]}},
        "defaults": {"lidar": [{"variant": "sick_s300", "mount": "front_laser"}]},
    }

    def test_declared_sensors_do_not_join_the_rendered_ones(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]):
        from arena_robots.Robot import RobotView

        rd = _write_robot(
            tmp_path,
            model_params={"sensors": [{"name": "ghost_sensor", "type": "laserscan", "topic": "/scan", "frame": "ghost_frame"}]},
            assembly=self._ASSEMBLY,
        )
        view = RobotView(rd)
        assert [sensor.name for sensor in view.effective_sensors({})] == ["lidar", "lidar_points"]
        assert capsys.readouterr().err == ""

    def test_a_model_params_without_sensors_renders_the_same_sensors(self, tmp_path: Path):
        from arena_robots.Robot import RobotView

        rd = _write_robot(tmp_path, model_params={}, assembly=self._ASSEMBLY)
        assert [(sensor.name, sensor.topic) for sensor in RobotView(rd).effective_sensors({})] == [("lidar", "${namespace}/scan"), ("lidar_points", "${namespace}/scan/points")]
