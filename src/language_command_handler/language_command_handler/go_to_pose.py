#!/usr/bin/env python3
"""
This code is for ROS2 node 'go_to_pose'
This node will control the robot to navigate to a specific (x, y, yaw) target.
It implements a 'Rotate-First' strategy to prevent backward walking.
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Twist, Quaternion
import math
import time

class GoToPoseNode(Node):
    """
    A ROS2 node that handles navigation to a specific pose
    """
    def __init__(self):
        super().__init__('go_to_pose')
        
        # 1. Declare & Get Parameters (Target Pose)
        self.declare_parameter('x', 0.0)
        self.declare_parameter('y', 0.0)
        self.declare_parameter('yaw', 0.0)

        self.target_x = self.get_parameter('x').value
        self.target_y = self.get_parameter('y').value
        self.target_yaw = self.get_parameter('yaw').value

        self.get_logger().info(f'🚀 Mission Initialized: Go to ({self.target_x}, {self.target_y})')

        # 2. Communication Setup
        self.pose_sub = self.create_subscription(PoseStamped, '/go1_pose', self.pose_callback, 10)
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.goal_pub = self.create_publisher(PoseStamped, '/goal_pose', 10)

        # 3. State Variables
        self.robot_pose = None
        self.mission_state = 0  # 0: Ready, 1: Rotate to Target, 2: Move, 3: Final Align, 4: Done

        # 4. Control Loop Timer (10Hz)
        self.timer = self.create_timer(0.1, self.mission_control_loop)

    def pose_callback(self, msg):
        """Callback to update robot pose"""
        self.robot_pose = msg

    def mission_control_loop(self):
        """
        Main control loop that manages the mission states
        """
        if self.robot_pose is None:
            self.get_logger().warn('Waiting for /go1_pose...', throttle_duration_sec=2.0)
            return

        # [State 0] Ready -> Start
        if self.mission_state == 0:
            self.mission_state = 1

        # [State 1] Initial Rotation (Face the target point)
        elif self.mission_state == 1:
            curr_x = self.robot_pose.pose.position.x
            curr_y = self.robot_pose.pose.position.y
            dx = self.target_x - curr_x
            dy = self.target_y - curr_y
            
            # If target is far enough (> 0.5m), rotate towards it first
            if math.hypot(dx, dy) > 0.5:
                target_heading = math.atan2(dy, dx)
                if self.rotate_to_angle(target_heading):
                    self.send_goal_to_planner()
                    self.mission_state = 2
            else:
                # If too close, just align to final yaw
                self.mission_state = 3

        # [State 2] Moving (Wait until arrival)
        elif self.mission_state == 2:
            curr_x = self.robot_pose.pose.position.x
            curr_y = self.robot_pose.pose.position.y
            dist = math.hypot(self.target_x - curr_x, self.target_y - curr_y)
            
            if dist < 0.4: # Arrival threshold
                self.get_logger().info('📍 Arrived. Aligning orientation...')
                self.mission_state = 3

        # [State 3] Final Alignment (Face the target yaw)
        elif self.mission_state == 3:
            if self.rotate_to_angle(self.target_yaw):
                self.get_logger().info('✅ Mission Complete.')
                self.mission_state = 4

        # [State 4] Shutdown
        elif self.mission_state == 4:
            raise SystemExit

    def rotate_to_angle(self, target_rad):
        """
        Helper: Rotates robot to absolute radian angle. Returns True when done.
        """
        curr_yaw = self.quaternion_to_yaw(self.robot_pose.pose.orientation)
        error = self.normalize_angle(target_rad - curr_yaw)

        if abs(error) < 0.08: # Tolerance ~5 deg
            self.cmd_vel_pub.publish(Twist()) # Stop
            return True

        cmd = Twist()
        cmd.angular.z = max(min(error * 1.5, 0.6), -0.6) # P-Control with limit
        self.cmd_vel_pub.publish(cmd)
        return False

    def send_goal_to_planner(self):
        """Helper: Publishes goal to A* planner"""
        goal = PoseStamped()
        goal.header.frame_id = "map"
        goal.header.stamp = self.get_clock().now().to_msg()
        goal.pose.position.x = self.target_x
        goal.pose.position.y = self.target_y
        goal.pose.orientation = self.yaw_to_quaternion(self.target_yaw)
        self.goal_pub.publish(goal)
        self.get_logger().info('🏃 Goal sent to planner!')

    def quaternion_to_yaw(self, q):
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)

    def yaw_to_quaternion(self, yaw):
        q = Quaternion()
        q.z = math.sin(yaw / 2.0)
        q.w = math.cos(yaw / 2.0)
        return q

    def normalize_angle(self, angle):
        while angle > math.pi: angle -= 2.0 * math.pi
        while angle < -math.pi: angle += 2.0 * math.pi
        return angle

def main(args=None):
    """
    Main function to initialize and run the ROS2 node
    """
    rclpy.init(args=args)
    node = GoToPoseNode()
    try:
        rclpy.spin(node)
    except SystemExit:
        pass
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()