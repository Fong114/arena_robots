from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from arena_robots.acoustic_profile import default_profile, effective_profile

PROFILE = Path(__file__).resolve().parents[2] / "arena_robots" / "config" / "acoustic_profile.yaml"


def test_the_default_profile_is_the_shared_acoustic_profile_file():
    assert default_profile() == {key: float(value) for key, value in yaml.safe_load(PROFILE.read_text()).items()}


def test_a_robots_acoustics_block_is_laid_over_the_default_profile():
    profile = effective_profile({"L_base_0": 51.0, "beta_0": 3})
    assert profile["L_base_0"] == pytest.approx(51.0)
    assert profile["beta_0"] == pytest.approx(3.0)
    assert profile["beta_1"] == pytest.approx(default_profile()["beta_1"])


def test_a_robot_without_an_acoustics_block_runs_the_default_profile():
    assert effective_profile(None) == default_profile()
