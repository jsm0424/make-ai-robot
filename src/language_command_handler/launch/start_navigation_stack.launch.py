"""
Launch the full navigation stack (localization, path planner, path tracker, perception).
"""

import os
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory


def _include_from_package(package_name: str, launch_file: str) -> IncludeLaunchDescription:
    package_share = get_package_share_directory(package_name)
    launch_path = os.path.join(package_share, 'launch', launch_file)
    return IncludeLaunchDescription(PythonLaunchDescriptionSource(launch_path))


def generate_launch_description():
    odom_localization = _include_from_package('localization', 'odom_localizer.launch.py')
    global_localization = _include_from_package('localization', 'global_localizer.launch.py')
    planner = _include_from_package('astar_planner', 'astar_planner.launch.py')
    path_tracker = _include_from_package('path_tracker', 'path_tracker_launch.py')
    perception = _include_from_package('perception', 'perception.launch.py')

    return LaunchDescription([
        odom_localization,
        global_localization,
        planner,
        path_tracker,
        perception,
    ])
