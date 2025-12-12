"""
Mission launch file that optionally starts the entire navigation stack and then
runs the navigate_to_toilet node.
"""

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    pkg_share = get_package_share_directory('language_command_handler')

    start_stack_arg = DeclareLaunchArgument(
        'start_stack',
        default_value='true',
        description='When true, include start_navigation_stack.launch.py before running the mission node.'
    )

    start_stack_launch = os.path.join(
        pkg_share,
        'launch',
        'start_navigation_stack.launch.py'
    )
    start_stack = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(start_stack_launch),
        condition=IfCondition(LaunchConfiguration('start_stack'))
    )

    navigate_node = Node(
        package='language_command_handler',
        executable='go_to_pose.py',
        name='navigate_to_toilet',
        output='screen',
        parameters=[{
            'x': 9.0,
            'y': 11.0,
            'yaw': 0.0,
        }]
    )

    return LaunchDescription([
        start_stack_arg,
        start_stack,
        navigate_node,
    ])
