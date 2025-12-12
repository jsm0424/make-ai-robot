import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    # 1. 패키지 설치 경로 찾기
    pkg_share = get_package_share_directory('language_command_handler')
    
    # 2. 모델 파일 기본 경로 설정 (install/share/.../models/sign_model.pt)
    default_model_path = os.path.join(pkg_share, 'models', 'sign_model.pt')

    # 인자 선언
    use_sim_time_arg = DeclareLaunchArgument('use_sim_time', default_value='false')
    model_path_arg = DeclareLaunchArgument(
        'model_path', 
        default_value=default_model_path,
        description='Absolute path to the YOLO .pt file'
    )

    use_sim_time = LaunchConfiguration('use_sim_time')
    model_path = LaunchConfiguration('model_path')

    # 노드 설정
    navigator_node = Node(
        package='language_command_handler',
        executable='smart_navigator.py',
        name='smart_navigator',
        output='screen',
        parameters=[{'use_sim_time': use_sim_time}]
    )

    controller_node = Node(
        package='language_command_handler',
        executable='mission5_empty_room_controller.py',
        name='mission5_empty_room_controller',
        output='screen',
        parameters=[{
            'use_sim_time': use_sim_time,
            'model_path': model_path
        }]
    )

    return LaunchDescription([
        use_sim_time_arg,
        model_path_arg,
        navigator_node,
        controller_node
    ])