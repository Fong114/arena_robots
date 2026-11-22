from __future__ import annotations

from pathlib import Path

import pytest
from arena_robots.assembly import RequestPart
from arena_robots.Robot import RobotIdentifier, RobotView


def test_jackal_declared_mass_and_power_profile_add_mounted_payload_crate() -> None:
    view = RobotIdentifier.parse("jackal").resolve_sync()
    base_mass = view.declared_mass_kg({})
    assert base_mass is not None
    assert base_mass >= 18.431

    loaded_parts = {"payload": [RequestPart(variant="crate_19kg")]}
    loaded_mass = view.declared_mass_kg(loaded_parts)
    assert loaded_mass == pytest.approx(base_mass + 19.3)

    unloaded_split = view.mass_breakdown_kg({})
    loaded_split = view.mass_breakdown_kg(loaded_parts)
    assert unloaded_split is not None
    assert loaded_split is not None
    assert unloaded_split["base"] == pytest.approx(18.431)
    assert unloaded_split["components"] == pytest.approx(0.101)
    assert unloaded_split["payload"] == pytest.approx(0.0)
    assert loaded_split["base"] == pytest.approx(18.431)
    assert loaded_split["components"] == pytest.approx(unloaded_split["components"])
    assert loaded_split["payload"] == pytest.approx(19.3)
    assert loaded_split["total"] == pytest.approx(
        loaded_split["base"] + loaded_split["components"] + loaded_split["payload"]
    )

    base_profile = view.power_profile({})
    loaded_profile = view.power_profile(loaded_parts)
    assert base_profile is not None
    assert loaded_profile is not None
    assert base_profile["robot_mass_kg"] == pytest.approx(base_mass)
    assert loaded_profile["robot_mass_kg"] == pytest.approx(base_mass + 19.3)


def test_undeclared_robot_mass_returns_none(tmp_path: Path) -> None:
    (tmp_path / "model_params.yaml").write_text("base_frame: base_link\n")
    view = RobotView(tmp_path)
    assert view.declared_mass_kg({}) is None
