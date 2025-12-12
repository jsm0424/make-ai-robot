"""
Launch helper to start the MPPI path tracker.
"""

import os
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    path_tracker_share = get_package_share_directory('path_tracker')
    path_tracker_launch = os.path.join(
        path_tracker_share,
        'launch',
        'path_tracker_launch.py'
    )

    path_tracker_action = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(path_tracker_launch)
    )

    return LaunchDescription([path_tracker_action])
