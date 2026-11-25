"""MoveIt arm adapter launch."""

from arena_robots.moveit_factory import build_moveit_params
from arena_robots.request import InstanceRequest
from launch import LaunchDescription
from launch.actions import OpaqueFunction
from launch_ros.actions import Node, PushRosNamespace

from arena_rclpy_mixins.launch_argument import LaunchArgument


def generate_launch_description():
    ld_items = []
    LaunchArgument.auto_append(ld_items)

    robot = LaunchArgument("robot", description="Robot model name.")
    namespace = LaunchArgument("namespace", description="ROS namespace of the robot.")
    frame = LaunchArgument("frame", description="Frame prefix of the robot.")
    use_sim_time = LaunchArgument("use_sim_time", description="Whether nodes follow the simulator clock.")
    request_json = LaunchArgument("request_json", default_value="{}", description="JSON instance request (parts, frames, cap overrides) the MoveIt stack plans for.")
    instance = LaunchArgument("instance", default_value="", description="Name of the arm instance the MoveIt stack serves (empty = the only arm).")

    def launch_setup(context, *args, **kwargs):
        ns = namespace.substitution.perform(context)
        robot_name = robot.substitution.perform(context)
        sim = use_sim_time.substitution.perform(context).lower() == "true"
        request = InstanceRequest.loads(request_json.substitution.perform(context))
        instance_name = instance.substitution.perform(context) or None

        tf_prefix = frame.substitution.perform(context)

        moveit_params = build_moveit_params(robot_name, tf_prefix=tf_prefix, request=request, instance=instance_name)
        if moveit_params is None:
            raise ValueError(f"{robot_name}: missing 'arm' cap or caps.moveit block; cannot launch moveit adapter")

        move_group = Node(
            package="moveit_ros_move_group",
            executable="move_group",
            output="screen",
            parameters=[
                moveit_params,
                {"use_sim_time": sim},
            ],
        )

        return [PushRosNamespace(ns), move_group]

    return LaunchDescription([*ld_items, OpaqueFunction(function=launch_setup)])
