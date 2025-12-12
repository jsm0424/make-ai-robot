"""
Launch helper to start the localization stack used by language_command_handler.
"""

import os
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    localization_share = get_package_share_directory('localization')
    localization_launch = os.path.join(
        localization_share,
        'launch',
        'global_localizer.launch.py'
    )

    localization_action = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(localization_launch)
    )

    return LaunchDescription([localization_action])
