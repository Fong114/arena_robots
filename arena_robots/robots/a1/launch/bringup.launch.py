"""A1 bringup: CHAMP quadruped locomotion sidecar (cmd_vel -> leg joints).

Included by `RobotManager._launch_robot` once per robot spawn, inside the
already-pushed robot namespace, gated on this file's presence. It hosts the
open-source gait controller that
consumes the arena `cmd_vel` (Twist) and streams joint trajectories to the
`joint_group_position_controller` brought up from `control.yaml`. Sim-agnostic:
the same nodes drive the joints whether the controller_manager is backed by
Gazebo's gz_ros2_control or the MuJoCo or Isaac topic bridge.
"""

from arena_rclpy_mixins.launch_argument import LaunchArgument
from launch import LaunchDescription
from launch.substitutions import Command, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterFile


def generate_launch_description():
    use_sim_time = LaunchConfiguration('use_sim_time')

    declare_urdf_path = LaunchArgument('urdf_path', description='Served URDF file of the robot instance.')
    declare_namespace = LaunchArgument('namespace', default_value='', description='Robot namespace.')
    declare_base_frame = LaunchArgument('base_frame', default_value='base', description='Base link frame.')
    declare_use_sim_time = LaunchArgument('use_sim_time', default_value='true', description='Use the sim clock.')
    declare_joints_map = LaunchArgument('joints_map', description='CHAMP joints_map parameter file of the robot instance.')
    declare_links_map = LaunchArgument('links_map', description='CHAMP links_map parameter file of the robot instance.')
    declare_gait_config = LaunchArgument('gait_config', description='CHAMP gait parameter file of the robot instance.')

    urdf = {'urdf': Command(['cat ', LaunchConfiguration('urdf_path')])}

    # The gait engine: cmd_vel -> foot trajectories -> joint angles. arena owns
    # joint_states (joint_state_broadcaster) and odom TF (pose_to_tf from the sim
    # pose), so this only publishes joint control; foot contacts and odom TF are
    # left off to avoid double-publishing.
    quadruped_controller = Node(
        package='champ_base',
        executable='quadruped_controller_node',
        output='screen',
        parameters=[
            {'use_sim_time': use_sim_time},
            {'gazebo': True},
            {'publish_joint_states': False},
            {'publish_joint_control': True},
            {'publish_foot_contacts': False},
            {'publish_odom_tf': False},
            {'joint_controller_topic': 'joint_group_position_controller/joint_trajectory'},
            urdf,
            ParameterFile(LaunchConfiguration('joints_map')),
            ParameterFile(LaunchConfiguration('links_map')),
            ParameterFile(LaunchConfiguration('gait_config')),
        ],
        remappings=[('cmd_vel/smooth', 'cmd_vel')],
    )

    # Body-velocity / odometry estimate from the gait state on `odom/raw`, the sim
    # publishes the ground-truth `odom`.
    state_estimator = Node(
        package='champ_base',
        executable='state_estimation_node',
        output='screen',
        parameters=[
            {'use_sim_time': use_sim_time},
            {'orientation_from_imu': False},
            urdf,
            ParameterFile(LaunchConfiguration('joints_map')),
            ParameterFile(LaunchConfiguration('links_map')),
            ParameterFile(LaunchConfiguration('gait_config')),
        ],
    )

    return LaunchDescription(
        [
            declare_urdf_path,
            declare_namespace,
            declare_base_frame,
            declare_use_sim_time,
            declare_joints_map,
            declare_links_map,
            declare_gait_config,
            quadruped_controller,
            state_estimator,
        ]
    )
