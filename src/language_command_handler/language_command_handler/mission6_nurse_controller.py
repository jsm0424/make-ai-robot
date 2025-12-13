#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Twist
from std_msgs.msg import String
from sensor_msgs.msg import Image, LaserScan
from cv_bridge import CvBridge
import cv2
import math
import time
import os
from ament_index_python.packages import get_package_share_directory

try:
    from ultralytics import YOLO
except ImportError:
    print("❌ Ultralytics library not found.")

class Mission6DynamicController(Node):
    def __init__(self):
        super().__init__('mission6_dynamic_controller')

        # === [설정] ===
        self.ROOM_ENTRANCE = {'x': -6.68, 'y': -24.92, 'yaw': -3.13}
        self.TARGET_RADIUS = 0.6  # 목표 유지 거리 (0.8m)
        self.SIDE_SPEED   = -0.3 # 공전 속도 (게걸음, 오른쪽)

        # 모델 경로 설정
        try:
            pkg_share = get_package_share_directory('language_command_handler')
            default_model_path = os.path.join(pkg_share, 'models', 'nurse_model.pt')
        except:
            default_model_path = 'src/language_command_handler/models/nurse_model.pt'
            
        self.declare_parameter('model_path', default_model_path)
        self.model = YOLO(self.get_parameter('model_path').value)

        # 통신 설정
        self.nav_pub = self.create_publisher(PoseStamped, '/navigator/input_pose', 10)
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        
        self.nav_sub = self.create_subscription(String, '/navigator/status', self.status_callback, 10)
        self.pose_sub = self.create_subscription(PoseStamped, '/go1_pose', self.pose_callback, 10)
        self.img_sub = self.create_subscription(Image, '/camera_face/image', self.img_callback, 10)
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.speech_pub = self.create_publisher(String, '/robot_dog/speech', 10)

        self.bridge = CvBridge()
        self.cv_image = None
        self.latest_scan = None
        
        # 로봇 상태
        self.rx = 0.0
        self.ry = 0.0
        self.ryaw = 0.0
        
        # 간호사 좌표 (계산 후 고정)
        self.nurse_x = 0.0
        self.nurse_y = 0.0
        
        # 0:시작 -> 1:문앞 -> 2:돌진 -> 3:좌표계산 -> 4:좌표기반공전 -> 5:완료
        self.state = 0
        self.orbit_start_time = 0.0

        self.create_timer(0.1, self.mission_loop)
        self.get_logger().info("🧠 Mission 6: Coordinate-Based Orbit Started!")

    def pose_callback(self, msg):
        self.rx = msg.pose.position.x
        self.ry = msg.pose.position.y
        q = msg.pose.orientation
        self.ryaw = math.atan2(2*(q.w*q.z + q.x*q.y), 1-2*(q.y*q.y + q.z*q.z))

    def scan_callback(self, msg):
        self.latest_scan = msg

    def status_callback(self, msg):
        if msg.data == "ARRIVED":
            if self.state == 1: 
                self.get_logger().info("🚪 문 앞 도착. 간호사 찾아 돌진!")
                self.state = 2

    def img_callback(self, msg):
        try:
            self.cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            # 디버깅용 화면 표시 (옵션)
            if self.model is not None:
                results = self.model(self.cv_image, verbose=False, conf=0.5)
                annotated_frame = results[0].plot()
                cv2.imshow("Mission5 Debug", annotated_frame)
                cv2.waitKey(1)
        except: pass

    def get_front_obstacle_dist(self):
        if self.latest_scan is None: return 99.9
        ranges = self.latest_scan.ranges
        valid_ranges = [r for r in ranges if 0.1 < r < 10.0]
        if not valid_ranges: return 99.9
        return min(valid_ranges)

    # --- [1] Visual Servoing (접근) ---
    def process_visual_servoing(self):
        if self.cv_image is None or self.model is None: return False, False
        
        results = self.model(self.cv_image, verbose=False)
        _, img_w, _ = self.cv_image.shape
        
        detected = False
        center_x = 0
        for r in results:
            for box in r.boxes:
                if int(box.cls[0]) == 0 and float(box.conf[0]) > 0.9:
                    x_c, _, _, _ = box.xywh[0].tolist()
                    center_x = x_c
                    detected = True
                    break
        
        if detected:
            error_x = (img_w / 2) - center_x
            ang_z = error_x * 0.005
            ang_z = max(min(ang_z, 0.5), -0.5)

            front_dist = self.get_front_obstacle_dist()
            cmd = Twist()
            cmd.angular.z = float(ang_z)
            
            # 목표 거리(0.8m)보다 멀면 전진
            if front_dist > self.TARGET_RADIUS:
                cmd.linear.x = 0.3
                is_arrived = False
                self.get_logger().info(f"🚀 접근 중... 거리: {front_dist:.2f}m")
            else:
                cmd.linear.x = 0.0
                is_arrived = True
                self.get_logger().info(f"🛑 목표 거리 도달! (거리: {front_dist:.2f}m)")

            self.cmd_vel_pub.publish(cmd)
            return True, is_arrived
            
        return False, False

    # --- [2] 간호사 좌표 특정 (Lock) ---
    def calculate_nurse_coordinates(self):
        """
        로봇이 멈춘 시점의 위치와 방향을 기준으로 간호사 좌표 확정
        """
        # 현재 로봇은 간호사를 정면으로 보고 있음
        # 간호사 위치 = 로봇위치 + (정면벡터 * 거리)
        dist = self.TARGET_RADIUS
        
        self.nurse_x = self.rx + dist * math.cos(self.ryaw)
        self.nurse_y = self.ry + dist * math.sin(self.ryaw)
        
        self.get_logger().info(f"📍 간호사 좌표 고정: ({self.nurse_x:.2f}, {self.nurse_y:.2f})")

    # --- [3] 좌표 기반 공전 (Feedback Control) ---
    def perform_coordinate_orbit(self):
        """
        간호사 좌표(Nx, Ny)를 중심으로 뱅글 돔.
        단순 속도 명령이 아니라, 현재 위치 오차를 보정하며 돔.
        """
        elapsed = (self.get_clock().now().nanoseconds / 1e9) - self.orbit_start_time
        if elapsed > 15.0: # 15초 동안 돌기 (약 한 바퀴 반)
            return True

        # 1. 간호사까지의 현재 거리와 각도 계산
        dx = self.nurse_x - self.rx
        dy = self.nurse_y - self.ry
        curr_dist = math.hypot(dx, dy)
        target_yaw = math.atan2(dy, dx) # 간호사를 바라보는 각도

        # 2. 제어량 계산
        cmd = Twist()

        # (A) 회전 제어 (Yaw Control): 항상 간호사를 바라보게
        yaw_error = target_yaw - self.ryaw
        # 각도 정규화 (-pi ~ pi)
        while yaw_error > math.pi: yaw_error -= 2*math.pi
        while yaw_error < -math.pi: yaw_error += 2*math.pi
        
        cmd.angular.z = yaw_error * 1.5 # P-Gain

        # (B) 거리 제어 (Distance Control): 0.8m 유지
        # 너무 멀면 전진(+), 너무 가까우면 후진(-)
        dist_error = curr_dist - self.TARGET_RADIUS
        cmd.linear.x = dist_error * 1.0 # P-Gain
        
        # (C) 공전 (Orbit): 옆으로 이동
        cmd.linear.y = self.SIDE_SPEED # -0.3 (오른쪽 이동)

        self.cmd_vel_pub.publish(cmd)
        
        self.get_logger().info(f"🔄 좌표 공전 중.. 거리오차: {dist_error:.2f}m / 각도오차: {yaw_error:.2f}rad", throttle_duration_sec=0.5)
        return False

    def send_nav_command(self, pose_dict):
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
        # [State 0] 문 앞으로 이동
        if self.state == 0:
            self.send_nav_command(self.ROOM_ENTRANCE)
            self.state = 1

        elif self.state == 1: pass 

        # [State 2] 돌진 (Visual Servoing)
        elif self.state == 2: 
            is_detected, is_arrived = self.process_visual_servoing()
            
            if is_detected:
                if is_arrived:
                    self.state = 3 # 도착 -> 좌표 계산으로 이동

            else:
                # 안 보이면 제자리 회전
                cmd = Twist()
                cmd.angular.z = 0.3
                self.cmd_vel_pub.publish(cmd)

        # [State 3] 간호사 좌표 확정
        elif self.state == 3:
            self.calculate_nurse_coordinates()
            self.orbit_start_time = self.get_clock().now().nanoseconds / 1e9
            self.state = 4
            self.get_logger().info("💫 좌표 기반 공전 시작!")

        # [State 4] 좌표 기반 피드백 주행
        elif self.state == 4:
            is_done = self.perform_coordinate_orbit()
            if is_done:
                self.cmd_vel_pub.publish(Twist()) # 정지
                self.state = 5

        # [State 5] 종료
        elif self.state == 5:
            self.get_logger().info("🎉 미션 완료! 멍멍!")
            self.speech_pub.publish(String(data="bark"))
            self.state = 6

def main(args=None):
    rclpy.init(args=args)
    node = Mission6DynamicController()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()