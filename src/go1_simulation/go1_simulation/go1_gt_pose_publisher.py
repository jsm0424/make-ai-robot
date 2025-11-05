#!/usr/bin/env python3

""" 
This script reads from Gazebo's '/world/default/dynamic_pose/info' topic directly via CLI,
finds the 'go1' robot pose, and publishes it.

Parameters:
- comparison (bool, default=True): If True, publishes to '/go1_pose' and '/go1_pose_2d' without TF.
                                    If False, publishes to '/go1_pose_gt' and '/go1_pose_2d_gt' with TF.

Publishes:
- geometry_msgs/msg/PoseStamped (3D pose) to '/go1_pose' or '/go1_pose_gt'
- geometry_msgs/msg/Pose2D (x, y, theta) to '/go1_pose_2d' or '/go1_pose_2d_gt'
- TF transform from 'map' to 'odom' to 'base' frame (only when comparison=False)

Usage examples:
- Default (comparison=True): ros2 run go1_simulation go1_gt_pose_publisher
- With comparison=False: ros2 run go1_simulation go1_gt_pose_publisher --ros-args -p comparison:=false

"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose2D, PoseStamped, TransformStamped
from rosgraph_msgs.msg import Clock
from tf2_ros import TransformBroadcaster
import math
import subprocess
import threading
import re


class Go1GTPosePublisher(Node):
    """
    ROS 2 node that reads Gazebo's dynamic pose info topic via CLI,
    finds the 'go1' robot pose, and publishes it.
    Publishes both 3D pose (PoseStamped) and 2D pose (Pose2D).
    Broadcasts TF transform from map to odom to base frame only when comparison=False.
    """

    def __init__(self):
        super().__init__('go1_gt_pose_publisher')
        
        # Declare and get the 'comparison' parameter
        self.declare_parameter('comparison', True)
        self.comparison = self.get_parameter('comparison').get_parameter_value().bool_value
        
        # Determine topic names based on comparison parameter
        if self.comparison:
            self.topic_3d = '/go1_pose_gt'
            self.topic_2d = '/go1_pose_2d_gt'
        else:
            self.topic_3d = '/go1_pose'
            self.topic_2d = '/go1_pose_2d'
        
        # Store the latest simulation time from /clock
        self.current_sim_time = None
        self.clock_lock = threading.Lock()
        
        # Subscribe to /clock topic to get simulation time
        self.clock_subscription = self.create_subscription(
            Clock,
            '/clock',
            self.clock_callback,
            10
        )
        
        # Create TF broadcaster
        self.tf_broadcaster = TransformBroadcaster(self)
        
        # Create publisher for 3D pose
        self.publisher_3d = self.create_publisher(
            PoseStamped,
            self.topic_3d,
            10
        )
        
        # Create publisher for 2D pose
        self.publisher_2d = self.create_publisher(
            Pose2D,
            self.topic_2d,
            10
        )
        
        self.get_logger().info('Ground truth pose publisher node started')
        self.get_logger().info(f'Comparison mode: {self.comparison}')
        self.get_logger().info('Subscribing to /clock for simulation time')
        self.get_logger().info(f'Publishing to: {self.topic_3d} (PoseStamped: full 3D pose)')
        self.get_logger().info(f'Publishing to: {self.topic_2d} (Pose2D: x, y, theta)')
        self.get_logger().info('Reading from Gazebo topic: /world/default/dynamic_pose/info')
        if not self.comparison:
            self.get_logger().info('Broadcasting TF: map -> odom -> base')
        else:
            self.get_logger().info('TF broadcasting disabled (comparison mode)')
        
        # Start thread to read from gz topic
        self.running = True
        self.gz_thread = threading.Thread(target=self.read_gz_topic)
        self.gz_thread.daemon = True
        self.gz_thread.start()
    
    def clock_callback(self, msg):
        """
        Callback for /clock topic to store simulation time.
        
        Args:
            msg (Clock): Clock message with simulation time
        """
        with self.clock_lock:
            self.current_sim_time = msg.clock

    def quaternion_to_yaw(self, qx, qy, qz, qw):
        """
        Convert quaternion to yaw angle (theta).
        
        Args:
            qx, qy, qz, qw: Quaternion components
            
        Returns:
            float: Yaw angle in radians
        """
        # Convert quaternion to yaw using the formula:
        # yaw = atan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
        siny_cosp = 2.0 * (qw * qz + qx * qy)
        cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        return yaw

    def parse_pose_data(self, lines):
        """
        Parse pose data from a list of lines.
        
        Args:
            lines (list): List of lines from the gz topic output
            
        Returns:
            dict or None: Dictionary with name, position, and orientation, or None if parsing fails
        """
        try:
            data = {
                'name': None,
                'position': {'x': None, 'y': None, 'z': None},
                'orientation': {'x': None, 'y': None, 'z': None, 'w': None}
            }
            
            current_section = None
            
            for line in lines:
                line = line.strip()
                
                # Check for name
                if 'name:' in line:
                    match = re.search(r'name:\s*"([^"]+)"', line)
                    if match:
                        data['name'] = match.group(1)
                
                # Check for section headers
                elif line.startswith('position {'):
                    current_section = 'position'
                elif line.startswith('orientation {'):
                    current_section = 'orientation'
                elif line == '}':
                    current_section = None
                
                # Extract numeric values
                elif current_section and ':' in line:
                    parts = line.split(':')
                    if len(parts) == 2:
                        key = parts[0].strip()
                        value_str = parts[1].strip()
                        try:
                            value = float(value_str)
                            if current_section in data and key in data[current_section]:
                                data[current_section][key] = value
                        except ValueError:
                            pass
            
            # Verify all required fields are present
            if (data['name'] and 
                all(v is not None for v in data['position'].values()) and
                all(v is not None for v in data['orientation'].values())):
                return data
            
            return None
            
        except Exception as e:
            self.get_logger().debug(f'Error parsing pose data: {e}')
            return None

    def publish_pose_data(self, pose_data):
        """
        Publish pose data as ROS messages and TF.
        Publishes both 3D and 2D poses.
        Only broadcasts TF when comparison is False.
        
        Args:
            pose_data: Dictionary with position and orientation data
        """
        # Get simulation time from /clock topic
        with self.clock_lock:
            if self.current_sim_time is None:
                # If we haven't received clock yet, skip publishing
                return
            current_time = self.current_sim_time
        
        # Only broadcast TF transforms when comparison is False
        if not self.comparison:
            # Broadcast TF transform from map to odom
            t = TransformStamped()
            t.header.stamp = current_time
            t.header.frame_id = 'map'
            t.child_frame_id = 'odom'
            
            # Set translation
            t.transform.translation.x = 0.0
            t.transform.translation.y = 0.0
            t.transform.translation.z = 0.0
            
            # Set rotation (quaternion)
            t.transform.rotation.x = 0.0
            t.transform.rotation.y = 0.0
            t.transform.rotation.z = 0.0
            t.transform.rotation.w = 1.0

            # Broadcast the transform
            self.tf_broadcaster.sendTransform(t)        

            # Broadcast TF transform from odom to base
            t = TransformStamped()
            t.header.stamp = current_time
            t.header.frame_id = 'odom'
            t.child_frame_id = 'base'
            
            # Set translation
            t.transform.translation.x = pose_data['position']['x']
            t.transform.translation.y = pose_data['position']['y']
            t.transform.translation.z = pose_data['position']['z']
            
            # Set rotation (quaternion)
            t.transform.rotation.x = pose_data['orientation']['x']
            t.transform.rotation.y = pose_data['orientation']['y']
            t.transform.rotation.z = pose_data['orientation']['z']
            t.transform.rotation.w = pose_data['orientation']['w']        
            
            # Broadcast the transform
            self.tf_broadcaster.sendTransform(t)
        
        # Publish 3D pose (PoseStamped)
        pose_stamped = PoseStamped()
        pose_stamped.header.stamp = current_time
        pose_stamped.header.frame_id = 'map'
        
        pose_stamped.pose.position.x = pose_data['position']['x']
        pose_stamped.pose.position.y = pose_data['position']['y']
        pose_stamped.pose.position.z = pose_data['position']['z']
        
        pose_stamped.pose.orientation.x = pose_data['orientation']['x']
        pose_stamped.pose.orientation.y = pose_data['orientation']['y']
        pose_stamped.pose.orientation.z = pose_data['orientation']['z']
        pose_stamped.pose.orientation.w = pose_data['orientation']['w']
        
        self.publisher_3d.publish(pose_stamped)
        
        # Publish 2D pose (Pose2D)
        pose_2d = Pose2D()
        pose_2d.x = pose_data['position']['x']
        pose_2d.y = pose_data['position']['y']
        pose_2d.theta = self.quaternion_to_yaw(
            pose_data['orientation']['x'],
            pose_data['orientation']['y'],
            pose_data['orientation']['z'],
            pose_data['orientation']['w']
        )
        
        self.publisher_2d.publish(pose_2d)

    def read_gz_topic(self):
        """
        Read from the Gazebo topic using subprocess and parse the output.
        Runs in a separate thread.
        """
        try:
            # Start the gz topic echo command
            process = subprocess.Popen(
                ['gz', 'topic', '--echo', '--topic', '/world/default/dynamic_pose/info'],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1
            )
            
            self.get_logger().info('Started reading from Gazebo topic')
            
            current_pose_lines = []
            brace_count = 0
            in_pose_block = False
            msgs_received = 0
            
            while self.running and process.poll() is None:
                line = process.stdout.readline()
                if not line:
                    break
                
                stripped = line.strip()
                
                # Detect start of a pose block
                if stripped.startswith('pose {'):
                    in_pose_block = True
                    brace_count = 1
                    current_pose_lines = [line]
                elif in_pose_block:
                    current_pose_lines.append(line)
                    
                    # Count braces to find end of pose block
                    brace_count += stripped.count('{')
                    brace_count -= stripped.count('}')
                    
                    # When brace_count reaches 0, we have a complete pose block
                    if brace_count == 0:
                        in_pose_block = False
                        msgs_received += 1
                        
                        # Parse this pose block
                        pose_data = self.parse_pose_data(current_pose_lines)
                        
                        if pose_data and pose_data['name'] == 'go1':
                            # Publish the pose data
                            self.publish_pose_data(pose_data)
                            
                            # Log occasionally
                            if msgs_received % 100 == 0:
                                self.get_logger().info(
                                    f'Published go1 pose: x={pose_data["position"]["x"]:.3f}, '
                                    f'y={pose_data["position"]["y"]:.3f}, '
                                    f'z={pose_data["position"]["z"]:.3f}'
                                )
                        
                        # Reset for next pose
                        current_pose_lines = []
            
            process.terminate()
            
        except Exception as e:
            self.get_logger().error(f'Error reading gz topic: {e}')

    def destroy_node(self):
        """Clean up when node is destroyed."""
        self.running = False
        if hasattr(self, 'gz_thread'):
            self.gz_thread.join(timeout=1.0)
        super().destroy_node()


def main(args=None):
    """
    Main function to initialize and run the ROS 2 node.
    """
    rclpy.init(args=args)
    
    go1_gt_pose_publisher = Go1GTPosePublisher()
    
    try:
        rclpy.spin(go1_gt_pose_publisher)
    except KeyboardInterrupt:
        go1_gt_pose_publisher.get_logger().info('Node stopped by user')
    finally:
        go1_gt_pose_publisher.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()