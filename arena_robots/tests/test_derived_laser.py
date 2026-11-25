from __future__ import annotations

import math
from collections.abc import Callable
from pathlib import Path

import pytest
import yaml
from arena_robots import ARENA_ROBOTS_DIR
from arena_robots.Robot import RobotIdentifier
from arena_robots.urdf_facts import laser_block

ROBOTS_DIR = Path(ARENA_ROBOTS_DIR) / "robots"
GOLDEN_DIR = Path(__file__).parent / "golden"
AUTHORED = yaml.safe_load((GOLDEN_DIR / "mobile_laser_blocks.yaml").read_text())
GOLDEN = sorted(p.name.removesuffix("_default.urdf") for p in GOLDEN_DIR.glob("*_default.urdf"))
WITHOUT_GOLDEN = sorted(set(AUTHORED) - set(GOLDEN))
ANGLE_TOLERANCE = 5e-3
INCREMENT_TOLERANCE = 1e-2
RANGE_TOLERANCE = 1e-3


def _golden(robot: str) -> str:
    return (GOLDEN_DIR / f"{robot}_default.urdf").read_text()


def _primary_frame(robot: str) -> str:
    sensors = RobotIdentifier(robot).resolve_sync().effective_sensors({})
    return next(sensor.frame for sensor in sensors if str(sensor.type) == "laserscan")


def _differences(block: dict[str, object], expected: dict[str, object]) -> list[str]:
    angle = block["angle"]
    want = expected["angle"]
    out: list[str] = []
    for key in ("min", "max"):
        if not math.isclose(float(angle[key]), float(want[key]), abs_tol=ANGLE_TOLERANCE):
            out.append(f"angle.{key} {angle[key]} vs {want[key]}")
    if not math.isclose(float(angle["increment"]), float(want["increment"]), rel_tol=INCREMENT_TOLERANCE):
        out.append(f"angle.increment {angle['increment']} vs {want['increment']}")
    if int(block["num_beams"]) != int(expected["num_beams"]):
        out.append(f"num_beams {block['num_beams']} vs {expected['num_beams']}")
    if not math.isclose(float(block["range"]), float(expected["range"]), rel_tol=RANGE_TOLERANCE):
        out.append(f"range {block['range']} vs {expected['range']}")
    if round(float(block["update_rate"])) != round(float(expected["update_rate"])):
        out.append(f"update_rate {block['update_rate']} vs {expected['update_rate']}")
    return out


def test_no_mobile_cap_file_authors_a_laser_block():
    for path in sorted(ROBOTS_DIR.glob("*/caps.yaml")):
        assert "laser" not in yaml.safe_load(path.read_text())["mobile"], path


@pytest.mark.parametrize("robot", GOLDEN)
def test_the_primary_lidar_of_the_golden_urdf_equals_the_formerly_authored_block(robot: str):
    block = laser_block(_golden(robot), _primary_frame(robot))
    assert block is not None
    assert _differences(block, AUTHORED[robot]) == []


def test_boxer_primary_lidar_is_the_front_laser_and_equals_the_formerly_authored_block():
    assert _primary_frame("boxer") == "front_laser"
    block = laser_block(_golden("boxer"), "front_laser")
    assert block is not None
    assert _differences(block, AUTHORED["boxer"]) == []


@pytest.mark.parametrize("robot", WITHOUT_GOLDEN)
def test_robots_without_a_golden_urdf_name_a_laserscan_sensor(robot: str):
    sensors = RobotIdentifier(robot).resolve_sync().effective_sensors({})
    assert any(str(sensor.type) == "laserscan" for sensor in sensors)


class TestRenderedCaps:
    @pytest.fixture(autouse=True)
    def _rendering(self) -> None:
        pytest.importorskip("xacro", reason="xacro is not importable, run under the Arena container")

    @pytest.mark.parametrize("robot", ["jackal", "tiago", "rbkairos", "boxer", *WITHOUT_GOLDEN])
    def test_the_mobile_cap_reads_the_laser_from_the_rendered_urdf(self, robot: str, require_installed: Callable[[str], None]):
        require_installed(robot)
        laser = RobotIdentifier(robot).resolve_sync().caps.mobile.laser
        assert laser is not None
        expected = AUTHORED[robot]
        assert laser.num_beams == expected["num_beams"]
        assert laser.range == pytest.approx(expected["range"], rel=RANGE_TOLERANCE)

    def test_effective_and_snapshot_include_the_derived_laser(self, require_installed: Callable[[str], None]):
        require_installed("jackal")
        caps = RobotIdentifier("jackal").resolve_sync().effective_caps({})
        assert _differences(caps.effective()["mobile"]["laser"], AUTHORED["jackal"]) == []
        assert caps.snapshot()["effective"]["mobile"]["laser"] == caps.effective()["mobile"]["laser"]
        assert caps.snapshot()["provenance"]["mobile.laser.range"] == "urdf"
