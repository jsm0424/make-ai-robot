#!/usr/bin/env python3
"""
This code is for ROS2 node 'rotate'
This node will receive a target yaw angle via ROS parameter and rotate the robot to that angle.
It uses P-controller logic to align the robot's orientation.
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseStamped, Quaternion
import math
import time

class RotateNode(Node):
    """
    A ROS2 node that rotates the robot to a specific absolute Yaw angle
    """
    def __init__(self):
        super().__init__('rotate_node')
        
        # 1. Declare Parameter (Target Yaw in Radians)
        # Default is 0.0 (Facing East in standard map frame)
        self.declare_parameter('target_yaw', 0.0)
        self.target_yaw = self.get_parameter('target_yaw').value

        self.get_logger().info(f'🔄 Rotate Node Initialized. Target Yaw: {self.target_yaw:.2f} rad')

        # 2. Communication Setup
        self.publisher = self.create_publisher(Twist, '/cmd_vel', 10)
        
        self.pose_sub = self.create_subscription(
            PoseStamped, 
            '/go1_pose', 
            self.pose_callback, 
            10
        )
        
        self.robot_pose = None
        self.done = False
        
        # Control loop at 20Hz
        self.timer = self.create_timer(0.05, self.control_loop)

    def pose_callback(self, msg):
        self.robot_pose = msg

    def quaternion_to_yaw(self, q):
        """
        Convert Quaternion to Yaw angle (radians)
        """
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)

    def normalize_angle(self, angle):
        """
        Normalize angle to be within [-pi, pi]
        This ensures the robot rotates in the shortest direction.
        """
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle

    def control_loop(self):
        """
        Calculate error and publish angular velocity
        """
        if self.robot_pose is None:
            self.get_logger().warn('Waiting for /go1_pose...', throttle_duration_sec=2.0)
            return

        if self.done:
            return

        # 1. Get current Yaw
        current_q = self.robot_pose.pose.orientation
        current_yaw = self.quaternion_to_yaw(current_q)

        # 2. Calculate Error (Shortest path)
        error = self.normalize_angle(self.target_yaw - current_yaw)

        # 3. Check if reached (Tolerance: 0.05 rad approx 3 degrees)
        if abs(error) < 0.05:
            self.get_logger().info('✅ Rotation Complete.')
            self.stop_robot()
            self.done = True
            # Shutdown the node after completion so the next command can run
            raise SystemExit
            return

        # 4. P-Controller for Angular Velocity
        # Kp = 1.5 (Gain), Max Speed = 1.0 rad/s
        angular_z = 1.5 * error
        
        # Clamp velocity for safety
        max_speed = 1.0
        if angular_z > max_speed: angular_z = max_speed
        if angular_z < -max_speed: angular_z = -max_speed

        # 5. Publish Command
        twist = Twist()
        twist.linear.x = 0.0
        twist.angular.z = float(angular_z)
        self.publisher.publish(twist)

    def stop_robot(self):
        twist = Twist()
        self.publisher.publish(twist)

def main(args=None):
    rclpy.init(args=args)
    rotate_node = RotateNode()
    
    try:
        rclpy.spin(rotate_node)
    except SystemExit:
        # Expected exit when rotation is done
        pass
    except KeyboardInterrupt:
        pass
    finally:
        rotate_node.stop_robot()
        rotate_node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()