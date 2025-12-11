import os
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    pkg_share = get_package_share_directory('perception')
    
    # Ensure this path matches where you put your .pt file in the package
    model_path = os.path.join(pkg_share, 'models', 'food_model.pt')

    return LaunchDescription([
        Node(
            package='perception',
            executable='perception_node',
            name='perception_node',
            output='screen',
            parameters=[
                {'model_path': model_path},
                {'debug_mode': True} 
            ]
        )
    ])