import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():

    pkg_share = get_package_share_directory('language_command_handler')

    default_model_path = os.path.join(pkg_share, 'models', 'nurse_model.pt')

    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time', 
        default_value='true',
        description='Use simulation (Gazebo) clock if true'
    )
    
    use_sim_time_arg = DeclareLaunchArgument('use_sim_time', default_value='false')
    model_path_arg = DeclareLaunchArgument(
        'model_path', 
        default_value=default_model_path,
        description='Absolute path to the YOLO .pt file'
    )

    use_sim_time = LaunchConfiguration('use_sim_time')
    model_path = LaunchConfiguration('model_path')

    navigator_node = Node(
        package='language_command_handler',
        executable='smart_navigator.py',
        name='smart_navigator',
        output='screen',
        parameters=[{'use_sim_time': use_sim_time}]
    )

    mission_controller_node = Node(
        package='language_command_handler',
        executable='mission6_nurse_controller.py',
        name='mission6_nurse_controller',
        output='screen',
        parameters=[{
            'use_sim_time': use_sim_time,
            'yolo_model_path': model_path
        }]
    )

    return LaunchDescription([
        use_sim_time_arg,
        model_path_arg,
        navigator_node,
        mission_controller_node
    ])
