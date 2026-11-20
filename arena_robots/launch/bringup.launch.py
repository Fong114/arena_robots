"""Standalone robot bringup launch (no task_generator)."""

from arena_rclpy_mixins.launch_argument import LaunchArgument
from arena_robots.bringup import check_caps, BRINGUPS
from arena_robots.request import InstanceRequest
from arena_robots.Robot import RobotIdentifier
from launch import LaunchDescription
from launch.actions import OpaqueFunction


def generate_launch_description():
    ld_items = []
    LaunchArgument.auto_append(ld_items)

    robot_arg = LaunchArgument('robot', description='Robot model name.')
    namespace_arg = LaunchArgument('namespace', description='ROS namespace of the robot.')
    bringup_arg = LaunchArgument('bringup', default_value='nav2', description='Bringup kind of the robot.')
    cap_arg = LaunchArgument('cap', default_value='mobile', description='Robot capability the bringup is for.')
    use_sim_time_arg = LaunchArgument('use_sim_time', default_value='true', description='Whether nodes follow the simulator clock.')
    frame_arg = LaunchArgument('frame', default_value='', description='Frame prefix of the robot.')

    def _compose(context, *args, **kwargs):
        name = robot_arg.substitution.perform(context)
        ns = namespace_arg.substitution.perform(context)
        kind = bringup_arg.substitution.perform(context)
        cap = cap_arg.substitution.perform(context)
        use_sim_time_str = use_sim_time_arg.substitution.perform(context)
        frame = frame_arg.substitution.perform(context)

        use_sim_time = use_sim_time_str.lower() in ('true', '1', 'yes')

        view = RobotIdentifier(name).resolve_sync()
        bringup = BRINGUPS[cap].get(kind)(robot=view, namespace=ns, request=InstanceRequest())
        check_caps(bringup)

        bringup_actions = bringup._launch_actions(use_sim_time=use_sim_time, frame=frame)

        actions = [*bringup_actions]
        actions.extend(bringup.telemetry_actions())

        return actions

    return LaunchDescription([*ld_items, OpaqueFunction(function=_compose)])
