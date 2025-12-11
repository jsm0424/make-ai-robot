#!/usr/bin/env python3

"""
GO1 Smart Goal Sender Node (Rotate First, Then Move)

수정 사항:
- 목표 지점을 받으면 즉시 Path를 생성하지 않음
- 우선 목표 지점을 바라보도록 제자리 회전(Pivot Turn)을 수행함 (/cmd_vel 직접 제어)
- 회전이 완료되면 그제서야 /goal_pose를 발행하여 경로 주행 시작
"""

import math
import sys
import threading
import time

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Point, Quaternion, Twist


class MoveGo1(Node):
    def __init__(self):
        super().__init__('path_tracker_client')
        
        # State tracking
        self.robot_pose = None
        self.is_rotating = False # 회전 중인지 확인하는 플래그
        
        # Subscribe to robot pose
        self.pose_sub = self.create_subscription(
            PoseStamped,
            '/go1_pose',
            self.pose_callback,
            10
        )
        
        # [추가] 직접 회전을 시키기 위한 cmd_vel Publisher
        self.cmd_vel_pub = self.create_publisher(
            Twist,
            '/cmd_vel',
            10
        )
        
        # Goal Publisher
        self.goal_pub = self.create_publisher(
            PoseStamped,
            '/goal_pose',
            10
        )
        
        self.get_logger().info('=' * 60)
        self.get_logger().info('Smart Goal Sender Node Started')
        self.get_logger().info('Strategy: Rotate in place -> Then send goal')
        self.get_logger().info('Waiting for robot pose...')
        self.get_logger().info('=' * 60)
        
    def pose_callback(self, msg):
        self.robot_pose = msg
    
    def quaternion_to_yaw(self, q):
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)
    
    def yaw_to_quaternion(self, yaw):
        q = Quaternion()
        q.x = 0.0; q.y = 0.0
        q.z = math.sin(yaw / 2.0); q.w = math.cos(yaw / 2.0)
        return q

    def normalize_angle(self, angle):
        """각도를 -pi ~ pi 사이로 변환"""
        while angle > math.pi: angle -= 2.0 * math.pi
        while angle < -math.pi: angle += 2.0 * math.pi
        return angle

    def rotate_to_target(self, target_x, target_y):
        """
        목표 지점을 바라보도록 제자리 회전 수행
        """
        if self.robot_pose is None:
            return False

        # 1. 목표 방향 계산
        curr_x = self.robot_pose.pose.position.x
        curr_y = self.robot_pose.pose.position.y
        curr_yaw = self.quaternion_to_yaw(self.robot_pose.pose.orientation)

        dx = target_x - curr_x
        dy = target_y - curr_y
        target_head_yaw = math.atan2(dy, dx) # 목표를 바라보는 각도

        # 거리가 너무 가까우면(0.5m 이내) 회전 안 하고 그냥 리턴
        dist = math.sqrt(dx*dx + dy*dy)
        if dist < 0.5:
            self.get_logger().info("Target is too close, skipping rotation.")
            return True

        self.get_logger().info(f"Target is behind/side. Rotating to {math.degrees(target_head_yaw):.1f} deg first...")
        self.is_rotating = True

        # 2. 회전 루프 (P-Controller)
        rate = self.create_rate(10) # 10Hz
        while rclpy.ok():
            # 현재 Yaw 갱신
            if self.robot_pose is None: continue
            curr_yaw = self.quaternion_to_yaw(self.robot_pose.pose.orientation)
            
            # 오차 계산 (최단 회전 방향)
            yaw_error = self.normalize_angle(target_head_yaw - curr_yaw)
            
            # 오차가 5도(0.08 rad) 이내면 정지
            if abs(yaw_error) < 0.08:
                break
            
            # 회전 명령 생성
            cmd = Twist()
            # P-gain = 1.5, Max speed = 0.6 rad/s (안전하게)
            angular_z = 1.5 * yaw_error
            if angular_z > 0.6: angular_z = 0.6
            if angular_z < -0.6: angular_z = -0.6
            
            cmd.angular.z = float(angular_z)
            self.cmd_vel_pub.publish(cmd)
            
            # Python의 sleep 대신 rate.sleep() 사용이 원칙이나 스레드 환경 고려 time.sleep
            time.sleep(0.05) 

        # 3. 정지
        stop_cmd = Twist()
        self.cmd_vel_pub.publish(stop_cmd)
        time.sleep(0.5) # 안정화 대기
        self.is_rotating = False
        self.get_logger().info("Rotation complete. Sending path goal now.")
        return True

    def set_target(self, target_x, target_y, target_yaw):
        if self.robot_pose is None:
            self.get_logger().warn('Waiting for robot pose...')
            return

        # [단계 1] 제자리 회전 먼저 수행 (블로킹 함수)
        # 로봇이 목표 방향을 볼 때까지 여기서 멈춰있음
        self.rotate_to_target(target_x, target_y)

        # [단계 2] 회전이 끝나면 A* Planner에게 목표 전송
        goal_msg = PoseStamped()
        goal_msg.header.frame_id = "map"
        goal_msg.header.stamp = self.get_clock().now().to_msg()
        goal_msg.pose.position.x = target_x
        goal_msg.pose.position.y = target_y
        goal_msg.pose.position.z = 0.0
        goal_msg.pose.orientation = self.yaw_to_quaternion(target_yaw)

        self.goal_pub.publish(goal_msg)
        self.get_logger().info(f'Sent Goal: ({target_x}, {target_y})')


def input_thread(node):
    time.sleep(2.0)
    print("\n" + "=" * 60)
    print("Smart Move Go1 Interface")
    print("Input: x y yaw (e.g., 2.0 1.0 1.57)")
    print("=" * 60)
    
    while rclpy.ok():
        try:
            if node.is_rotating:
                time.sleep(1.0)
                continue

            user_input = input("\nEnter target (x y yaw): ").strip()
            if not user_input: continue
            
            parts = user_input.split()
            if len(parts) != 3:
                print("Error: Enter 3 numbers")
                continue
            
            tx = float(parts[0])
            ty = float(parts[1])
            tyaw = float(parts[2])
            
            # 입력을 받으면 로봇 제어 시작
            node.set_target(tx, ty, tyaw)
            
        except ValueError:
            print("Invalid number format")
        except:
            break

def main(args=None):
    rclpy.init(args=args)
    node = MoveGo1()
    
    input_thread_handle = threading.Thread(target=input_thread, args=(node,), daemon=True)
    input_thread_handle.start()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()