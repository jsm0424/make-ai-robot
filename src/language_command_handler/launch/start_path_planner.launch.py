"""
Launch helper to start the path planner (A*).
"""

import os
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    planner_share = get_package_share_directory('astar_planner')
    planner_launch = os.path.join(
        planner_share,
        'launch',
        'astar_planner.launch.py'
    )

    planner_action = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(planner_launch)
    )

    return LaunchDescription([planner_action])
