from __future__ import annotations

import configparser
from collections.abc import Callable
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]

_ROS_SKIP_REASON = "ROS2 not discoverable, source install/setup.bash to enable"


def _ros_available() -> bool:
    try:
        import rclpy  # noqa: F401
    except ImportError:
        return False
    return True


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if _ros_available():
        return
    skip = pytest.mark.skip(reason=_ROS_SKIP_REASON)
    for item in items:
        path = str(item.path)
        if "/tests/ros/" in path or "/tests/integration/" in path:
            item.add_marker(skip)


@pytest.fixture(scope="session", autouse=True)
def rclpy_context():
    try:
        import rclpy
        import rclpy.node
    except ImportError:
        yield None
        return
    rclpy.init()
    node = rclpy.node.Node("pytest_host")
    try:
        yield rclpy
    finally:
        node.destroy_node()
        rclpy.shutdown()


@pytest.fixture()
def write_urdf() -> Callable[..., None]:
    """Return a writer of `<robot_dir>/urdf/<name>.urdf.xacro` with one base link of `kg` kg."""
    pytest.importorskip("xacro", reason="xacro is not importable, run under the Arena container")

    def _write(robot_dir: Path, kg: float = 1.0) -> None:
        (robot_dir / "urdf").mkdir(exist_ok=True)
        (robot_dir / "urdf" / f"{robot_dir.name}.urdf.xacro").write_text(f'<?xml version="1.0"?>\n<robot name="{robot_dir.name}" xmlns:xacro="http://www.ros.org/wiki/xacro">\n  <link name="base_link"><inertial><mass value="{kg}"/><inertia ixx="1" ixy="0" ixz="0" iyy="1" iyz="0" izz="1"/></inertial></link>\n</robot>\n')

    return _write


def _submodules() -> dict[str, list[Path]]:
    cfg = configparser.ConfigParser()
    cfg.read(_REPO / ".gitmodules")
    out: dict[str, list[Path]] = {}
    for section in cfg.sections():
        path = cfg[section]["path"]
        names = cfg[section].get("robot", "").split() + cfg[section].get("component", "").split()
        if not names and path.startswith("arena_robots/robots/"):
            names = [path.split("/")[2]]
        for name in names:
            out.setdefault(name, []).append(_REPO / path)
    return out


@pytest.fixture(scope="session")
def require_installed() -> Callable[[str], None]:
    """Skips the test unless every submodule tagged for the robot or component is checked out."""
    submodules = _submodules()

    def require(name: str) -> None:
        if any(not path.is_dir() or not any(path.iterdir()) for path in submodules.get(name, [])):
            pytest.skip(f"{name} is not installed, run `arena feature robots add {name}`")

    return require
