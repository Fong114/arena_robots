from __future__ import annotations

from pathlib import Path

import launch
import pytest
import yaml
from arena_rclpy_mixins.yaml_replace import YAMLReplacer
from arena_robots.caps import MobileSpec
from arena_robots.nav2 import Nav2CollisionDerivedYAML
from arena_robots.Robot import RobotIdentifier

PKG = Path(__file__).resolve().parents[2] / "arena_robots"
ROBOTS_DIR = PKG / "robots"
NAV2_YAML = PKG / "config" / "nav2" / "nav2.yaml"


def _inflation(robot: str, tmp_path: Path) -> tuple[float, float]:
    effective = tmp_path / "mobile.yaml"
    effective.write_text(yaml.safe_dump(RobotIdentifier(robot).resolve_sync().caps.mobile.raw))
    derived = Nav2CollisionDerivedYAML(str(effective)).perform(launch.LaunchContext())
    replacer = YAMLReplacer(yaml.safe_load(Path(derived).read_text()))
    nav2 = yaml.safe_load(NAV2_YAML.read_text())
    return tuple(replacer.replace(nav2[costmap][costmap]["ros__parameters"]["inflation_layer"])["inflation_radius"] for costmap in ("local_costmap", "global_costmap"))


@pytest.mark.parametrize(
    ("robot", "circumradius", "padding", "meshes"),
    [
        ("jackal", 0.3338, 0.0, False),
        ("ridgeback", 0.5519, 0.1, True),
        ("rbsummit", 0.465, 0.0, False),
        ("husky", 0.57, 0.0, False),
    ],
)
def test_both_costmaps_inflate_the_footprint_circumradius_plus_the_margin(robot, circumradius, padding, meshes, tmp_path, require_installed):
    if meshes:
        require_installed(robot)
    inflation = circumradius + padding + 0.4
    assert _inflation(robot, tmp_path) == pytest.approx((inflation, inflation), abs=5e-3)


def test_an_explicit_inflation_radius_wins():
    raw = {"radius": 0.3, "footprint": [[0.5, 0.5], [-0.5, -0.5]], "inflation_radius": 0.55}
    assert MobileSpec(path=Path("mobile.yaml"), raw=raw).inflation_radius == 0.55
