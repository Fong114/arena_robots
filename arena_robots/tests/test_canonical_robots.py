from __future__ import annotations

import collections
from collections.abc import Callable
from pathlib import Path

import pytest
import yaml
from arena_robots.Robot import RobotIdentifier
from arena_robots.sensors import SENSOR_TYPES

ROBOTS_DIR = Path(__file__).resolve().parents[2] / "arena_robots" / "robots"


def _robot_dirs() -> list[Path]:
    return [p for p in ROBOTS_DIR.iterdir() if p.is_dir()]


def _model_params_files() -> list[Path]:
    return list(ROBOTS_DIR.rglob("model_params.yaml"))


@pytest.mark.parametrize("yaml_path", sorted(ROBOTS_DIR.glob("*/caps.yaml")), ids=lambda p: p.parent.name)
def test_every_mobile_cap_carries_a_planar_block(yaml_path: Path, require_installed: Callable[[str], None]) -> None:
    require_installed(yaml_path.parent.name)
    planar = RobotIdentifier(yaml_path.parent.name).resolve_sync().effective_caps({}).mobile.planar
    assert planar is not None, f"{yaml_path}: no planar block"
    assert set(planar.velocity) == {"linear", "lateral", "angular"}, f"{yaml_path}: planar.velocity must cover every axis"


def test_no_robot_named_auto() -> None:
    names = {p.name for p in _robot_dirs()}
    assert "auto" not in names, "'auto' is a reserved token and must not be used as a robot directory name"


@pytest.mark.parametrize("yaml_path", _model_params_files(), ids=lambda p: p.parent.name)
def test_sensor_types_are_canonical(yaml_path: Path, require_installed: Callable[[str], None]) -> None:
    require_installed(yaml_path.parent.name)
    for sensor in RobotIdentifier(yaml_path.parent.name).resolve_sync().effective_sensors({}):
        assert sensor.type in SENSOR_TYPES, f"{yaml_path.parent.name}: sensor type {sensor.type!r} is not in SENSOR_TYPES"


def test_priority_uniqueness_informational(capsys: pytest.CaptureFixture[str]) -> None:
    Sig = tuple[bool, frozenset[str]]
    groups: dict[Sig, list[tuple[str, int]]] = collections.defaultdict(list)

    for yaml_path in _model_params_files():
        with open(yaml_path) as f:
            data = yaml.safe_load(f)
        robot_name = yaml_path.parent.name
        is_holonomic = bool(data.get("is_holonomic", False))
        sensor_types: frozenset[str] = frozenset(str(s.type) for s in RobotIdentifier(robot_name).resolve_sync().effective_sensors({}))
        priority = int(data.get("priority", 0))
        sig: Sig = (is_holonomic, sensor_types)
        groups[sig].append((robot_name, priority))

    for sig, entries in groups.items():
        max_p = max(p for _, p in entries)
        top = [name for name, p in entries if p == max_p]
        if len(top) > 1:
            print(f"priority tie: holonomic={sig[0]} sensors={sorted(sig[1])}, robots with max priority {max_p}: {sorted(top)}")
