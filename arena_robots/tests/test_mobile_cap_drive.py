from __future__ import annotations

from pathlib import Path

import yaml
from arena_robots.caps import MobileSpec


def _spec(tmp_path: Path, **raw: object) -> MobileSpec:
    path = tmp_path / "mobile.yaml"
    path.write_text(yaml.dump(raw))
    return MobileSpec(path, raw)


def test_a_mobile_cap_drives_unless_it_says_otherwise(tmp_path: Path) -> None:
    assert _spec(tmp_path, radius=0.3).drive is True


def test_a_base_that_only_seats_the_robot_does_not_drive(tmp_path: Path) -> None:
    assert _spec(tmp_path, radius=0.3, drive=False).drive is False
