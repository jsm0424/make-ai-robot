#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Twist
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import math
import time

class MissionBoxController(Node):
    def __init__(self):
        super().__init__('mission_box_controller')

        # === [설정] ===
        self.OBSERVATION_POSE = {'x': 0.0, 'y': 7.8, 'yaw': 1.57}
        self.BOX_LOCATIONS = ['LEFT', 'CENTER', 'RIGHT']

        self.WAYPOINTS = {
            'LEFT':   [{'x': -3.0, 'y': 10.0, 'yaw': 1.57}],
            'CENTER': [{'x': 1.0, 'y': 9.0, 'yaw': 1.57}, {'x': 1.0, 'y': 15.0, 'yaw': 3.141592}],
            'RIGHT':  [{'x': 3.0, 'y': 10.0, 'yaw': 1.57}]
        }

        self.PUSH_READY_POSES = {
            'LEFT':   {'x': -3.0, 'y': 12.0, 'yaw': 0.0},
            'CENTER': {'x': 0.2,  'y': 15.0, 'yaw': -1.57},
            'RIGHT':  {'x': 3.0,  'y': 12.0, 'yaw': 3.141592}
        }

        self.GOAL_ZONE_POSES = {
            'LEFT':   {'x': -0.6, 'y': 12.0, 'yaw': 0.0}, 
            'CENTER': {'x': -0.2,  'y': 12.5, 'yaw': -1.57},
            'RIGHT':  {'x': 0.6,  'y': 12.0, 'yaw': 3.141592}
        }
        # ==============

        # 통신 설정
        self.nav_pub = self.create_publisher(PoseStamped, '/navigator/input_pose', 10)
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10) # 후진용 직접 제어
        self.nav_sub = self.create_subscription(String, '/navigator/status', self.status_callback, 10)
        
        self.pose_sub = self.create_subscription(PoseStamped, '/go1_pose', self.pose_callback, 10)
        self.img_sub = self.create_subscription(Image, '/camera_face/image', self.img_callback, 10)
        self.speech_pub = self.create_publisher(String, '/robot_dog/speech', 10)

        self.bridge = CvBridge()
        self.cv_image = None
        

        self.step = 0 
        self.target_box_loc = None 
        self.waypoint_queue = []   
        
        # 후진 거리 계산을 위한 변수
        self.start_back_pose = None
        self.start_delay = 0 # 워밍업 카운터
        self.robot_pose = None

        self.current_target_pose = None

        self.create_timer(0.5, self.mission_loop)
        self.get_logger().info("📦 Hybrid Box Controller Started!")

    def pose_callback(self, msg):
        self.robot_pose = msg

    def status_callback(self, msg):
        """Navigator 상태 처리"""
        if msg.data == "ARRIVED":

            dist_left = self.get_distance_to_target()
            if dist_left > 0.5:
                return
            
            # [Step 1 -> 2] 관측 위치 도착
            if self.step == 1: 
                self.step = 2
            
            # [Step 4] 웨이포인트 주행
            elif self.step == 4: 
                if self.waypoint_queue:
                    next_wp = self.waypoint_queue.pop(0)
                    self.get_logger().info(f"🚦 Next Waypoint: {next_wp}")
                    self.send_nav_command(next_wp)
                else:
                    self.step = 5 # 웨이포인트 끝 -> 준비 위치로 이동 명령 대기
            
            # [Step 5.5 -> 6] 준비 위치 도착 -> 밀기 시작
            elif self.step == 5.5: 
                self.get_logger().info("✅ Ready Pose Arrived. Starting Push!")
                self.step = 6 

            # [Step 6.5 -> 7] 밀기 완료(골인) -> 후진 준비
            elif self.step == 6.5:
                self.get_logger().info("✅ Box Push Complete! Preparing to Back up.")
                # 후진 시작 전 현재 위치 저장 (여기서부터 2m 잴 것임)
                self.start_back_pose = self.robot_pose 
                self.step = 7

    def img_callback(self, msg):
        try: self.cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except: pass

    def send_nav_command(self, pose_dict):
        """Navigator에게 목표 좌표 전송"""
        msg = PoseStamped()
        msg.header.frame_id = "map"
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.pose.position.x = float(pose_dict['x'])
        msg.pose.position.y = float(pose_dict['y'])
        yaw = float(pose_dict['yaw'])
        msg.pose.orientation.z = math.sin(yaw / 2.0)
        msg.pose.orientation.w = math.cos(yaw / 2.0)

        self.current_target_pose = pose_dict
        self.nav_pub.publish(msg)

    def get_distance_to_target(self):
        """현재 로봇 위치와 목표 지점 간의 거리 계산"""
        if self.robot_pose is None or self.current_target_pose is None:
            return 999.9 # 알 수 없음
        
        curr_x = self.robot_pose.pose.position.x
        curr_y = self.robot_pose.pose.position.y
        target_x = self.current_target_pose['x']
        target_y = self.current_target_pose['y']
        
        return math.hypot(target_x - curr_x, target_y - curr_y)

    def publish_cmd_vel(self, linear_x, angular_z=0.0):
        """직접 속도 제어 (후진용)"""
        msg = Twist()
        msg.linear.x = linear_x
        msg.angular.z = angular_z
        self.cmd_vel_pub.publish(msg)

    def analyze_image(self):
        # (이미지 분석 로직 유지)
        if self.cv_image is None: return None
        img = self.cv_image.copy()
        h, w, _ = img.shape
        regions = [img[0:h, 0:w//3], img[0:h, w//3:2*w//3], img[0:h, 2*w//3:w]]
        lower_brown = np.array([10, 100, 40])
        upper_brown = np.array([25, 255, 200])
        max_pixels = 0
        detected_idx = -1
        for i, region in enumerate(regions):
            hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, lower_brown, upper_brown)
            count = cv2.countNonZero(mask)
            if count > max_pixels and count > 300:
                max_pixels = count
                detected_idx = i
        if detected_idx != -1: return self.BOX_LOCATIONS[detected_idx]
        return None

    def mission_loop(self):
        # [안전장치] 로봇 통신 연결 대기 (Warm-up)
        if self.robot_pose is None:
            self.get_logger().info("⏳ Waiting for robot pose...", throttle_duration_sec=2.0)
            return
        
        # 4초간 대기하며 시스템 안정화 (시작하자마자 멈추는 현상 방지)
        if self.start_delay < 8: # 0.5초 * 8 = 4초
            self.start_delay += 1
            if self.start_delay % 2 == 0:
                self.get_logger().info(f"⏳ System Warming up... {self.start_delay}/8")
            return

        # ================= 미션 로직 시작 =================

        # [Step 0] 관측 위치 이동
        if self.step == 0:
            if self.nav_pub.get_subscription_count() == 0:
                self.get_logger().info("📡 Waiting for Navigator connection...", throttle_duration_sec=1.0)
                return # 연결될 때까지 명령 안 보내고 리턴
            
            self.get_logger().info("🚀 Moving to Observation Point")
            self.send_nav_command(self.OBSERVATION_POSE)
            self.step = 1

        elif self.step == 1: pass

        # [Step 2] 분석
        elif self.step == 2:
            time.sleep(1.0)
            self.step = 3

        # [Step 3] 박스 찾기
        elif self.step == 3:
            result = self.analyze_image()
            if result:
                self.target_box_loc = result
                self.get_logger().info(f"📦 Box Found: {self.target_box_loc}")
                self.waypoint_queue = list(self.WAYPOINTS[self.target_box_loc])
                
                if self.waypoint_queue:
                    self.send_nav_command(self.waypoint_queue.pop(0))
                    self.step = 4
                else:
                    self.step = 5
            else:
                self.get_logger().warn("Retrying detection...")

        elif self.step == 4: pass

        # [Step 5] 준비 위치(Push Ready)로 이동
        elif self.step == 5:
            target = self.PUSH_READY_POSES[self.target_box_loc]
            self.get_logger().info(f"📍 Moving to Ready Pose: {target}")
            self.send_nav_command(target)
            self.step = 5.5 # 대기 상태

        elif self.step == 5.5: pass

        # [Step 6] 박스 밀기 (다시 Navigator 사용!)
        elif self.step == 6:
            target = self.GOAL_ZONE_POSES[self.target_box_loc]
            self.get_logger().info(f"💪 PUSHING to Goal: {target}")
            self.send_nav_command(target)
            self.step = 6.5 # 도착 대기 상태

        elif self.step == 6.5: pass # Navigator가 가고 있음...

        # [Step 7] 2m 그대로 후진 (Direct Control)
        elif self.step == 7:
            if self.start_back_pose is None: return # 안전장치

            # 이동 거리 계산
            dist_moved = math.hypot(
                self.robot_pose.pose.position.x - self.start_back_pose.pose.position.x,
                self.robot_pose.pose.position.y - self.start_back_pose.pose.position.y
            )
            
            if dist_moved < 0.6:
                self.get_logger().info(f"🔙 Reversing... Moved: {dist_moved:.2f}m", throttle_duration_sec=1)
                # 속도 -0.3으로 후진 (회전 없이 직선 후진)
                self.publish_cmd_vel(-0.3, 0.0)
            else:
                # 2m 후진 완료
                self.publish_cmd_vel(0.0, 0.0) # 정지
                self.get_logger().info("✅ Backing Done.")
                self.step = 8

        # [Step 8] 종료
        elif self.step == 8:
            self.get_logger().info("🎉 Mission Complete! Bark!")
            msg = String()
            msg.data = "bark"
            self.speech_pub.publish(msg)
            self.step = 9

def main(args=None):
    rclpy.init(args=args)
    node = MissionBoxController()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()