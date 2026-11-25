from __future__ import annotations

from pathlib import Path

import pytest
from arena_robots.assembly import RequestPart
from arena_robots.Robot import RobotIdentifier, RobotView

pytest.importorskip("xacro", reason="xacro is not importable, run under the Arena container")

_CRATE = {"payload": [RequestPart(variant="crate_19kg")]}


def test_jackal_mass_split_and_power_profile_add_the_mounted_payload_crate() -> None:
    view = RobotIdentifier.parse("jackal").resolve_sync()
    unloaded_caps, loaded_caps = view.effective_caps({}), view.effective_caps(_CRATE)
    unloaded, loaded = unloaded_caps.mass_split(), loaded_caps.mass_split()

    assert unloaded["base"] > 0.0
    assert unloaded["components"] > 0.0
    assert unloaded["payload"] == 0.0
    assert loaded["base"] == pytest.approx(unloaded["base"])
    assert loaded["components"] == pytest.approx(unloaded["components"])
    assert loaded["payload"] == pytest.approx(19.3)
    for split, caps in ((unloaded, unloaded_caps), (loaded, loaded_caps)):
        assert split["total"] == pytest.approx(split["base"] + split["components"] + split["payload"])
        assert split["total"] == pytest.approx(caps.body.mass)

    unloaded_profile, loaded_profile = view.power_profile(unloaded_caps), view.power_profile(loaded_caps)
    assert unloaded_profile is not None
    assert loaded_profile is not None
    assert loaded_profile["robot_mass_kg"] == pytest.approx(unloaded_profile["robot_mass_kg"] + 19.3)


def test_a_robot_without_a_urdf_states_no_mass(tmp_path: Path) -> None:
    (tmp_path / "model_params.yaml").write_text("base_frame: base_link\n")
    assert RobotView(tmp_path).effective_caps({}).body.mass is None
