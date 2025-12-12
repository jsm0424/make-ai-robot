"""
Launch helper to start the perception pipeline.
"""

import os
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    perception_share = get_package_share_directory('perception')
    perception_launch = os.path.join(
        perception_share,
        'launch',
        'perception.launch.py'
    )

    perception_action = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(perception_launch)
    )

    return LaunchDescription([perception_action])
