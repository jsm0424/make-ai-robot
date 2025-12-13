import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time', default_value='true',
        description='Use simulation (Gazebo) clock'
    )
    
    # YOLO 모델 경로 (본인 경로로 꼭 수정!)
    yolo_model_path_arg = DeclareLaunchArgument(
        'yolo_model_path',
        default_value='/home/syw/yolo_models/yolov8n.pt', 
        description='Path to YOLO pt file'
    )

    use_sim_time = LaunchConfiguration('use_sim_time')
    yolo_model_path = LaunchConfiguration('yolo_model_path')

    # 이동 담당 노드
    navigator_node = Node(
        package='language_command_handler',
        executable='smart_navigator.py',
        name='smart_navigator',
        output='screen',
        parameters=[{'use_sim_time': use_sim_time}]
    )

    # 미션 6 컨트롤러 (새로 만든 파일명 확인)
    mission_controller_node = Node(
        package='language_command_handler',
        executable='mission6_nurse_controller.py',
        name='mission6_nurse_controller.py',
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
