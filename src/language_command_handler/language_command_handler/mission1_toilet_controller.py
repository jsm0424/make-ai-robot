#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String
import math


class Mission1Controller(Node):
    def __init__(self):
        super().__init__('mission1_controller')

        self.TOILET_POSE = {'x': -7.4, 'y': -21.7, 'yaw': 1.57}

        self.nav_pub = self.create_publisher(PoseStamped, '/navigator/input_pose', 10)
        self.speech_pub = self.create_publisher(String, '/robot_dog/speech', 10)
        self.nav_sub = self.create_subscription(String, '/navigator/status', self.status_callback, 10)
        self.pose_sub = self.create_subscription(PoseStamped, '/go1_pose', self.pose_callback, 10)

        self.step = 0
        self.start_delay = 0
        self.robot_pose_received = False

        self.create_timer(0.5, self.mission_loop)
        self.get_logger().info("🚽 Mission 1: Go to Toilet & Bark Started!")

    def pose_callback(self, msg):
        self.robot_pose_received = True
    
    def status_callback(self, msg):
        """Navigator State"""
        if msg.data == "ARRIVED":

            if self.step == 1:
                self.get_logger().info("✅ Found Toilet!")
                self.step = 2

    def send_nav_command(self, pose_dict):
        """Send Goal Pose to Navigator"""
        msg = PoseStamped()
        msg.header.frame_id = "map"
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.pose.position.x = float(pose_dict['x'])
        msg.pose.position.y = float(pose_dict['y'])
        yaw = float(pose_dict['yaw'])

        msg.pose.orientation.z = math.sin(yaw / 2.0)
        msg.pose.orientation.w = math.cos(yaw / 2.0)

        self.nav_pub.publish(msg)

    def mission_loop(self):
        if not self.robot_pose_received:
            self.get_logger().info("⏳ Waiting for robot pose...", throttle_duration_sec=2.0)
            return

        if self.start_delay < 6: 
            self.start_delay += 1
            if self.start_delay % 2 == 0:
                self.get_logger().info(f"⏳ System Warming up... {self.start_delay}/6")
            return
        
        # ===== mission logic =====

        if self.step == 0:

            if self.nav_pub.get_subscription_count() == 0:
                self.get_logger().info("📡 Waiting for Navigator connection...", throttle_duration_sec=1.0)
                return # 연결될 때까지 명령 안 보내고 리턴

            self.get_logger().info(f"🚀 Moving to Toilet: {self.TOILET_POSE}")
            self.send_nav_command(self.TOILET_POSE)
            self.step = 1

        elif self.step == 1:
            pass
        
        elif self.step == 2:
            self.get_logger().info("🐶 Barking at the Toilet!")
            msg = String()
            msg.data = "bark"
            self.speech_pub.publish(msg)
            
            self.step = 3 

        elif self.step == 3:
            self.get_logger().info("🎉 Mission 1 Complete!")
            self.step = 4

def main(args=None):
    rclpy.init(args=args)
    node = Mission1Controller()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()