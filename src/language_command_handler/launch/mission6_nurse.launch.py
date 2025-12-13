import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    # 패키지 이름과 모델 파일명 확인 (사용자 환경에 맞게 수정!)
    package_name = 'language_command_handler' 
    model_filename = 'nurse_model.pt'
    
    # 모델 파일 경로 자동 탐색
    try:
        pkg_share = get_package_share_directory(package_name)
        default_model_path = os.path.join(pkg_share, 'models', model_filename)
    except:
        default_model_path = model_filename

    # 1. 인자 선언
    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time', 
        default_value='true',
        description='Use simulation (Gazebo) clock if true'
    )
    
    yolo_model_path_arg = DeclareLaunchArgument(
        'yolo_model_path',
        default_value=default_model_path, 
        description='Path to YOLO pt file'
    )

    use_sim_time = LaunchConfiguration('use_sim_time')
    yolo_model_path = LaunchConfiguration('yolo_model_path')

    # 2. 네비게이터 노드 (이동 담당)
    navigator_node = Node(
        package=package_name,
        executable='smart_navigator.py',
        name='smart_navigator',
        output='screen',
        parameters=[{'use_sim_time': use_sim_time}]
    )

    # 3. 미션 6 간호사 컨트롤러 (지휘 담당)
    mission_controller_node = Node(
        package=package_name,
        executable='mission6_nurse_controller.py',
        name='mission6_nurse_controller',
        output='screen',
        parameters=[{
            'use_sim_time': use_sim_time,
            'yolo_model_path': yolo_model_path
        }]
    )

    return LaunchDescription([
        use_sim_time_arg,
        yolo_model_path_arg,
        navigator_node,
        mission_controller_node
    ])
