from __future__ import annotations

import json

import pytest
from arena_robots import fleet
from arena_robots.fleet import fleet_cap_overrides, instance_caps

HALF_SPEED = {"cap:mobile.limits.velocity": json.dumps({"scale": 0.5})}
WIDE_BODY = {"cap:mobile.radius": json.dumps(0.5)}


def test_instance_caps_without_params_are_the_model_caps():
    assert instance_caps("jackal", {}).mobile.radius == pytest.approx(0.267)


def test_instance_caps_apply_the_cap_params_as_overrides():
    mobile = instance_caps("jackal", WIDE_BODY).mobile
    assert mobile.radius == pytest.approx(0.5)
    assert instance_caps("jackal", {}).mobile.radius == pytest.approx(0.267)


def test_fleet_facts_of_an_overridden_instance_differ_from_the_model():
    model = fleet._model_facts("jackal", frozenset())
    slow = fleet._model_facts("jackal", frozenset(HALF_SPEED.items()))
    assert model.max_linear_mps == pytest.approx(2.0)
    assert slow.max_linear_mps == pytest.approx(1.0)
    assert slow.base_frame == model.base_frame
    assert slow.wheel_separation_m == model.wheel_separation_m


def test_facts_memo_separates_instances_of_one_model():
    plain = fleet._model_facts("jackal", frozenset())
    slow = fleet._model_facts("jackal", frozenset(HALF_SPEED.items()))
    assert plain is not slow
    assert fleet._model_facts("jackal", frozenset()) is plain
    assert fleet._model_facts("jackal", frozenset(HALF_SPEED.items())) is slow


def test_caps_memo_separates_instances_of_one_model():
    assert instance_caps("jackal", WIDE_BODY) is instance_caps("jackal", dict(WIDE_BODY))
    assert instance_caps("jackal", WIDE_BODY) is not instance_caps("jackal", {})


def test_binding_speed_follows_the_instance_params():
    plain = fleet.RobotBinding.resolve(name="a", model="jackal", namespace="/a", frame_prefix="a")
    slow = fleet.RobotBinding.resolve(name="b", model="jackal", namespace="/b", frame_prefix="b", params=HALF_SPEED)
    assert plain.max_linear_mps == pytest.approx(2.0)
    assert slow.max_linear_mps == pytest.approx(1.0)


def test_fleet_cap_overrides_parse_the_params_json():
    params = {"lidar@front": "jackal_default", "cap:mobile.velocity_limits.linear": "[-0.5, 1.0]", "cap:arm.arm.stall_time": "2.0", "cap:mobile.acceleration_limits": '{"scale": 0.5}'}
    assert fleet_cap_overrides(params) == {"mobile.velocity_limits.linear": [-0.5, 1.0], "arm.arm.stall_time": 2.0, "mobile.acceleration_limits": {"scale": 0.5}}
    assert fleet_cap_overrides({"lidar@front": "jackal_default"}) == {}
