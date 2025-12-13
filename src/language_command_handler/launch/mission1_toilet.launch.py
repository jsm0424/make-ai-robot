import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    # 인자 선언
    use_sim_time_arg = DeclareLaunchArgument('use_sim_time', default_value='false')

    use_sim_time = LaunchConfiguration('use_sim_time')

    # 1. 일꾼 (Navigator)
    navigator_node = Node(
        package='language_command_handler',
        executable='smart_navigator.py',
        name='smart_navigator',
        output='screen',
        parameters=[{'use_sim_time': use_sim_time}]
    )

    # 2. 지휘관 (Controller)
    controller_node = Node(
        package='language_command_handler',
        executable='mission1_toilet_controller.py',
        name='mission1_toilet_controller',
        output='screen',
        parameters=[{
            'use_sim_time': use_sim_time
        }]
    )

    return LaunchDescription([
        use_sim_time_arg,
        navigator_node,
        controller_node
    ])