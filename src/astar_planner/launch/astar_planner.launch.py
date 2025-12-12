import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch.conditions import IfCondition

def generate_launch_description():  
    # Path Planner Node
    path_planner_node = Node(
        package='astar_planner',
        executable='path_planner_node',
        name='path_planner_node',
        output='screen',
    )

    return LaunchDescription([
        path_planner_node,
    ])
