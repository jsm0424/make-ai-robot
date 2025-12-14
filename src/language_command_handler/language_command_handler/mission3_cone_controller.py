#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Quaternion
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import math
import time

class MissionConeController(Node):
    def __init__(self):
        super().__init__('mission_cone_controller')

        # 찾을 색깔 파라미터
        self.declare_parameter('target_color', 'red')
        self.target_color = self.get_parameter('target_color').value

        # ================= 좌표 하드코딩 =================
        # 관측 위치 (3개가 다 보이는 곳)
        self.OBSERVATION_POSE = {'x': 1.263, 'y': 13.77, 'yaw': 1.57}
        
        # 콘 좌표 (왼쪽, 중앙, 오른쪽) - 실제 측정값으로 수정 필수
        self.CONE_POSES = [
            {'x': 0.2, 'y': 16.0, 'yaw': 1.57}, # Left
            {'x': 1.2, 'y': 16.0, 'yaw': 1.57}, # Center
            {'x': 2.2, 'y': 16.0, 'yaw': 1.57}  # Right
        ]
        # ===============================================

        # Smart Navigator와 통신
        self.nav_pub = self.create_publisher(PoseStamped, '/navigator/input_pose', 10)
        self.nav_sub = self.create_subscription(String, '/navigator/status', self.status_callback, 10)
        
        # 카메라 및 짖기 통신
        self.img_sub = self.create_subscription(Image, '/camera_face/image', self.img_callback, 10)
        self.speech_pub = self.create_publisher(String, '/robot_dog/speech', 10) # 짖기용

        self.bridge = CvBridge()
        self.cv_image = None
        
        # 상태: 0(준비), 1(관측이동), 2(관측도착), 3(분석), 4(최종이동), 5(완료)
        self.step = 0 
        self.create_timer(1.0, self.mission_loop)
        
        self.get_logger().info(f"🧠 Mission Controller Started. Target: {self.target_color}")

    def status_callback(self, msg):
        """Navigator가 도착했다고 알려줄 때 호출됨"""
        if msg.data == "ARRIVED":
            if self.step == 1: # 관측 위치 도착 완료
                self.get_logger().info("✅ Arrived at Observation Point.")
                self.step = 2
            elif self.step == 4: # 콘 앞 도착 완료
                self.get_logger().info("✅ Arrived at Cone.")
                self.step = 5

    def img_callback(self, msg):
        try:
            self.cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except: pass

    def send_nav_command(self, pose_dict):
        """Navigator에게 명령 전송"""
        msg = PoseStamped()
        msg.header.frame_id = "map"
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.pose.position.x = pose_dict['x']
        msg.pose.position.y = pose_dict['y']
        
        yaw = pose_dict['yaw']
        msg.pose.orientation.z = math.sin(yaw / 2.0)
        msg.pose.orientation.w = math.cos(yaw / 2.0)
        
        self.nav_pub.publish(msg)

    def analyze_image(self):
        """OpenCV로 3분할 색상 인식 (디버깅 강화판)"""
        if self.cv_image is None: return None

        img = self.cv_image.copy()
        h, w, _ = img.shape
        
        # [수정 1] 화면 전체 높이 사용 (바닥에 있는 콘을 보기 위해)
        regions = [img[0:h, 0:w//3], img[0:h, w//3:2*w//3], img[0:h, 2*w//3:w]]
        
        # [수정 2] HSV 범위 확장 (S, V 값을 낮춰서 어두운 색도 인식하게)
        colors_hsv = {
            # 빨강: Hue가 양쪽 끝에 걸침. Saturation/Value 최소값을 50으로 낮춤
            'red': [([0, 50, 50], [10, 255, 255]), ([170, 50, 50], [180, 255, 255])],
            # 초록
            'green': [([35, 50, 50], [85, 255, 255])],
            # 파랑
            'blue': [([100, 50, 50], [140, 255, 255])]
        }
        
        detected = []
        region_names = ['Left', 'Center', 'Right']

        self.get_logger().info("--- Image Analysis Start ---")

        for i, region in enumerate(regions):
            hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
            best_color = 'unknown'
            max_count = 0
            
            # 디버깅용 로그 문자열
            debug_str = f"Region {region_names[i]}: "
            
            for color_name, ranges in colors_hsv.items():
                mask = np.zeros(hsv.shape[:2], dtype="uint8")
                for (lower, upper) in ranges:
                    mask = cv2.bitwise_or(mask, cv2.inRange(hsv, np.array(lower), np.array(upper)))
                
                count = cv2.countNonZero(mask)
                debug_str += f"{color_name}={count} " # 각 색깔별 픽셀 수 출력
                
                # [수정 3] Threshold를 300 -> 50으로 대폭 낮춤 (일단 인식부터 되게)
                if count > max_count and count > 50: 
                    max_count = count
                    best_color = color_name
            
            self.get_logger().info(debug_str) # 로그 출력
            detected.append(best_color)
            
        self.get_logger().info(f"Result: {detected}")
        return detected

    def mission_loop(self):
        if self.robot_pose is None:
            self.get_logger().info("⏳ Waiting for robot pose...", throttle_duration_sec=2.0)
            return
        
        # 4초간 대기하며 시스템 안정화 (시작하자마자 멈추는 현상 방지)
        if self.start_delay < 8: # 0.5초 * 8 = 4초
            self.start_delay += 1
            if self.start_delay % 2 == 0:
                self.get_logger().info(f"⏳ System Warming up... {self.start_delay}/8")
            return
        
        # [Step 0] 시작 -> 관측 위치로 이동 명령
        if self.step == 0:
            if self.nav_pub.get_subscription_count() == 0:
                self.get_logger().info("📡 Waiting for Navigator connection...", throttle_duration_sec=1.0)
                return # 연결될 때까지 명령 안 보내고 리턴
            
            self.get_logger().info("Command: Move to Observation Point")
            self.send_nav_command(self.OBSERVATION_POSE)
            self.step = 1 # 이동 중 상태

        # [Step 1] 이동 중... (status_callback 대기)
        elif self.step == 1:
            pass 

        # [Step 2] 도착함 -> 카메라 안정화 대기
        elif self.step == 2:
            self.get_logger().info("👀 Stabilizing Camera...")
            time.sleep(2.0)
            self.step = 3

        # [Step 3] 이미지 분석 및 판단
        elif self.step == 3:
            colors = self.analyze_image()
            if colors:
                self.get_logger().info(f"🎨 Detected Colors (L-C-R): {colors}")
                try:
                    idx = colors.index(self.target_color)
                    self.get_logger().info(f"🎯 Target Found at index {idx}. Commanding Move!")
                    
                    # 최종 목표 좌표 전송
                    self.send_nav_command(self.CONE_POSES[idx])
                    self.step = 4
                except ValueError:
                    self.get_logger().warn(f"❌ Target {self.target_color} not found! Retrying...")
            else:
                self.get_logger().warn("No Image Received!")

        # [Step 4] 최종 이동 중...
        elif self.step == 4:
            pass

        # [Step 5] 완료 -> 짖기
        elif self.step == 5:
            self.get_logger().info("🎉 Mission Complete! BARK!")
            msg = String()
            msg.data = "bark"
            self.speech_pub.publish(msg)
            self.step = 6 # 진짜 끝

def main(args=None):
    rclpy.init(args=args)
    node = MissionConeController()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()