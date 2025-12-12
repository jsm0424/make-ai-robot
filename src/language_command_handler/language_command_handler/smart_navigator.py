#!/usr/bin/env python3
# 파일명: smart_navigator.py

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Twist, Quaternion
from std_msgs.msg import String
import math
import time

class SmartNavigator(Node):
    def __init__(self):
        super().__init__('smart_navigator')
        
        # 통신 설정
        self.goal_sub = self.create_subscription(PoseStamped, '/navigator/input_pose', self.goal_callback, 10)
        self.status_pub = self.create_publisher(String, '/navigator/status', 10)
        
        self.pose_sub = self.create_subscription(PoseStamped, '/go1_pose', self.pose_callback, 10)
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.planner_pub = self.create_publisher(PoseStamped, '/goal_pose', 10)

        self.robot_pose = None
        self.current_goal = None
        
        # 상태 관리: 0(대기), 1(출발전회전), 2(이동중), 3(도착후정렬)
        self.nav_state = 0 
        
        self.create_timer(0.1, self.control_loop)
        self.get_logger().info("🤖 Smart Navigator Ready (With Final Alignment)")

    def pose_callback(self, msg):
        self.robot_pose = msg

    def goal_callback(self, msg):
        self.current_goal = msg
        self.nav_state = 1 # 명령 받으면 1단계(출발 전 회전)로 진입
        self.get_logger().info(f"📨 Command Received: Go to ({msg.pose.position.x:.2f}, {msg.pose.position.y:.2f})")
        self.publish_status("MOVING")

    def publish_status(self, status):
        msg = String()
        msg.data = status
        self.status_pub.publish(msg)

    # --- 유틸리티 ---
    def quaternion_to_yaw(self, q):
        return math.atan2(2.0*(q.w*q.z + q.x*q.y), 1.0-2.0*(q.y*q.y + q.z*q.z))

    def normalize_angle(self, angle):
        while angle > math.pi: angle -= 2.0 * math.pi
        while angle < -math.pi: angle += 2.0 * math.pi
        return angle

    def rotate_to_target_yaw(self, target_yaw):
        """특정 Yaw 각도를 바라보도록 회전"""
        if self.robot_pose is None: return False
        
        curr_yaw = self.quaternion_to_yaw(self.robot_pose.pose.orientation)
        error = self.normalize_angle(target_yaw - curr_yaw)
        
        # 오차가 3도(0.05rad) 이내면 성공
        if abs(error) < 0.05:
            self.cmd_vel_pub.publish(Twist()) # 정지
            return True
        
        # P제어 회전
        cmd = Twist()
        cmd.angular.z = max(min(error * 1.5, 0.6), -0.6)
        self.cmd_vel_pub.publish(cmd)
        return False

    def control_loop(self):
        if self.current_goal is None or self.robot_pose is None:
            return

        # 목표 좌표 및 각도 추출
        tx = self.current_goal.pose.position.x
        ty = self.current_goal.pose.position.y
        t_quat = self.current_goal.pose.orientation
        t_yaw = self.quaternion_to_yaw(t_quat)

        # [State 0] 대기 중
        if self.nav_state == 0:
            pass

        # [State 1] 출발 전 회전 (목표 지점을 바라봄)
        elif self.nav_state == 1:
            curr_x = self.robot_pose.pose.position.x
            curr_y = self.robot_pose.pose.position.y
            
            # 목표 지점까지의 각도 계산
            path_heading = math.atan2(ty - curr_y, tx - curr_x)
            dist = math.hypot(tx - curr_x, ty - curr_y)

            # 거리가 멀면 회전부터 하고, 가까우면 바로 정렬로 넘어감
            if dist > 0.5:
                if self.rotate_to_target_yaw(path_heading):
                    # 회전 끝 -> A* Planner에게 이동 명령 하달
                    self.planner_pub.publish(self.current_goal)
                    self.nav_state = 2
            else:
                self.nav_state = 3 # 바로 최종 정렬로

        # [State 2] 이동 중 (도착 확인)
        elif self.nav_state == 2:
            curr_x = self.robot_pose.pose.position.x
            curr_y = self.robot_pose.pose.position.y
            dist = math.hypot(tx - curr_x, ty - curr_y)

            # 30cm 이내 도착 시
            if dist < 0.3:
                self.get_logger().info("📍 Position Arrived. Starting Final Alignment...")
                self.nav_state = 3

        # [State 3] 도착 후 최종 정렬 (목표 Yaw 맞추기)
        elif self.nav_state == 3:
            # 여기서 계속 회전 함수를 호출해줘야 함!
            if self.rotate_to_target_yaw(t_yaw):
                self.get_logger().info("🏁 Final Alignment Done. Mission Complete!")
                self.publish_status("ARRIVED") # 지휘관에게 보고
                self.nav_state = 0 # 대기 상태로 복귀
                self.current_goal = None # 목표 초기화

def main(args=None):
    rclpy.init(args=args)
    node = SmartNavigator()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()