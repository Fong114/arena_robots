"""Tests for arena_robots.task_kinds."""

from __future__ import annotations

import pytest


class TestTaskKind:
    def test_enum_value_stability(self):
        from arena_robots.task_kinds import TaskKind

        assert TaskKind.GOTO_POSE.value == "goto_pose"

    def test_enum_membership(self):
        from arena_robots.task_kinds import TaskKind

        kinds = list(TaskKind)
        assert TaskKind.GOTO_POSE in kinds


class TestPublicSuffix:
    def test_goto_pose_suffix(self):
        from arena_robots.task_kinds import PUBLIC_SUFFIX, TaskKind

        assert PUBLIC_SUFFIX[TaskKind.GOTO_POSE] == "goto_pose"

    def test_every_kind_but_grasp_has_a_public_endpoint(self):
        from arena_robots.task_kinds import PUBLIC_SUFFIX, TaskKind

        assert set(PUBLIC_SUFFIX) == set(TaskKind) - {TaskKind.GRASP}


class TestActionType:
    def test_goto_pose_returns_type(self):
        from arena_robots.task_kinds import TaskKind, action_type

        t = action_type(TaskKind.GOTO_POSE)
        assert t is not None
        assert callable(t)

    def test_unknown_raises_key_error(self):
        from arena_robots.task_kinds import action_type

        import enum

        FakeKind = enum.Enum("FakeKind", {"UNKNOWN": "unknown"})
        with pytest.raises(KeyError):
            action_type(FakeKind.UNKNOWN)


class TestEndpoint:
    def test_endpoint_wraps_namespace(self):
        from arena_robots.task_kinds import TaskKind, endpoint

        ep = endpoint("/robot1", TaskKind.GOTO_POSE)
        assert "goto_pose" in ep
        assert "robot1" in ep

    def test_endpoint_empty_namespace(self):
        from arena_robots.task_kinds import TaskKind, endpoint

        ep = endpoint("", TaskKind.GOTO_POSE)
        assert "goto_pose" in ep

    def test_endpoint_slash_namespace(self):
        from arena_robots.task_kinds import TaskKind, endpoint

        ep = endpoint("/", TaskKind.GOTO_POSE)
        assert "goto_pose" in ep


class TestGrasp:
    def test_grasp_is_served_by_the_gripper_cap(self):
        from arena_robots.task_kinds import TaskKind, cap_for_task_kind

        assert TaskKind.GRASP.value == "grasp"
        assert cap_for_task_kind(TaskKind.GRASP) == "gripper"

    def test_grasp_runs_on_the_gripper_controller_action(self):
        from control_msgs.action import GripperCommand

        from arena_robots.task_kinds import TaskKind, action_type

        assert action_type(TaskKind.GRASP) is GripperCommand
