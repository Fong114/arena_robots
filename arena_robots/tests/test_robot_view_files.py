from __future__ import annotations

from arena_robots.Robot import RobotView


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
