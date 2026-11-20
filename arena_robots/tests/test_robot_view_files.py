from __future__ import annotations

import pytest
import yaml
from arena_robots.Robot import RobotView


def test_bridge_mappings_returns_the_rows_of_mappings_yaml(tmp_path):
    rows = [{"ros_topic_name": "scan", "gz_topic_name": "scan"}]
    (tmp_path / "mappings.yaml").write_text(yaml.safe_dump(rows))
    assert RobotView(tmp_path).bridge_mappings() == rows


def test_bridge_mappings_rejects_a_mapping_at_the_top_level(tmp_path):
    (tmp_path / "mappings.yaml").write_text(yaml.safe_dump({"a": 1}))
    with pytest.raises(ValueError):
        RobotView(tmp_path).bridge_mappings()


def test_bringup_launch_is_the_launch_file_when_it_exists(tmp_path):
    view = RobotView(tmp_path)
    assert view.bringup_launch is None
    (tmp_path / "launch").mkdir()
    (tmp_path / "launch" / "bringup.launch.py").write_text("")
    assert view.bringup_launch == tmp_path / "launch" / "bringup.launch.py"


def test_srdf_source_is_the_residual_fragment_when_it_exists(tmp_path):
    view = RobotView(tmp_path)
    assert view.srdf_source is None
    (tmp_path / "srdf").mkdir()
    (tmp_path / "srdf" / f"{tmp_path.name}.srdf.xacro").write_text("")
    assert view.srdf_source == tmp_path / "srdf" / f"{tmp_path.name}.srdf.xacro"
