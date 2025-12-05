import os
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory # <--- Key Import

def generate_launch_description():
    # 1. Find the path to the installed 'perception' package
    pkg_share = get_package_share_directory('perception')
    
    # 2. Build the path to the model file
    model_path = os.path.join(pkg_share, 'models', 'last.pt')

    return LaunchDescription([
        Node(
            package='perception',
            executable='perception_node',
            name='perception_node',
            output='screen',
            parameters=[
                # 3. Pass the dynamic path to the node
                {'model_path': model_path} 
            ]
        )
    ])