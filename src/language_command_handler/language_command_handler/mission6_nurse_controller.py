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

class Mission6NurseController(Node):
    def __init__(self):
        super().__init__('mission6_nurse_controller')

        # === [설정] ===
        self.ROOM_ENTRANCE = {'x': -6.68, 'y': -24.92, 'yaw': -3.13}
        
        # [거리 설정]
        self.APPROACH_DIST = 0.6  # 1차 접근 거리 (가까이 붙기)
        self.TARGET_RADIUS = 0.7  # 최종 공전 반지름 (뒤로 물러날 거리)
        self.SIDE_SPEED    = -0.3 # 공전 속도

        # 모델 경로
        try:
            pkg_share = get_package_share_directory('language_command_handler')
            default_model_path = os.path.join(pkg_share, 'models', 'nurse_model.pt')
        except:
            default_model_path = 'src/language_command_handler/models/nurse_model.pt'
            
        self.declare_parameter('model_path', default_model_path)
        self.model = YOLO(self.get_parameter('model_path').value)

        # 통신
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
        
        self.robot_pose = None
        self.rx = 0.0
        self.ry = 0.0
        self.ryaw = 0.0

        self.nurse_x = 0.0
        self.nurse_y = 0.0
        
        # 상태: 0(이동) -> 2(접근 0.6m) -> 2.5(후진 0.8m) -> 3(계산) -> 4(공전)
        self.state = 0
        self.orbit_start_time = 0.0
        
        # 안정성을 위한 Non-blocking Sleep 변수
        self.start_delay = 0 
        self.wait_until_time = 0.0

        self.create_timer(0.1, self.mission_loop)
        self.get_logger().info("🧠 Mission 6: Approach(0.6) -> Back(0.8) -> Orbit Started!")

    def pose_callback(self, msg):
        self.robot_pose = msg
        self.rx = msg.pose.position.x
        self.ry = msg.pose.position.y
        q = msg.pose.orientation
        self.ryaw = math.atan2(2*(q.w*q.z + q.x*q.y), 1-2*(q.y*q.y + q.z*q.z))

    def scan_callback(self, msg):
        self.latest_scan = msg

    def status_callback(self, msg):
        if msg.data == "ARRIVED":
            if self.is_sleeping(): return # 자는 중엔 무시
            if self.state == 1: 
                self.get_logger().info("🚪 문 앞 도착. 2초 대기 후 돌진.")
                self.set_sleep(2.0)
                self.state = 2

    def img_callback(self, msg):
        try:
            self.cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            if self.model is not None:
                results = self.model(self.cv_image, verbose=False, conf=0.5)
                annotated_frame = results[0].plot()
                cv2.imshow("Mission 6 Debug", annotated_frame)
                cv2.waitKey(1)
        except: pass

    # === [유틸리티] 안정적인 대기 함수 ===
    def set_sleep(self, seconds):
        self.wait_until_time = self.get_clock().now().nanoseconds / 1e9 + seconds
        # 대기 중에는 정지 명령
        self.cmd_vel_pub.publish(Twist())

    def is_sleeping(self):
        return (self.get_clock().now().nanoseconds / 1e9) < self.wait_until_time

    def get_front_obstacle_dist(self):
        if self.latest_scan is None: return 99.9
        ranges = self.latest_scan.ranges
        valid_ranges = [r for r in ranges if 0.1 < r < 10.0]
        if not valid_ranges: return 99.9
        return min(valid_ranges)
    
    # --- [1] Visual Servoing (0.6m 까지 접근) ---
    def process_visual_servoing(self):
        if self.cv_image is None or self.model is None: return False, False
        
        results = self.model(self.cv_image, verbose=False)
        _, img_w, _ = self.cv_image.shape
        
        detected = False
        center_x = 0
        for r in results:
            for box in r.boxes:
                if int(box.cls[0]) == 0 and float(box.conf[0]) > 0.8:
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
            
            # [수정] 0.6m 까지 접근
            if front_dist > self.APPROACH_DIST:
                cmd.linear.x = 1.0
                is_arrived = False
                # self.get_logger().info(f"🚀 접근 중... ({front_dist:.2f}m)")
            else:
                cmd.linear.x = 0.0
                is_arrived = True
                self.get_logger().info(f"🛑 1차 접근 완료 (0.6m 도달)!")

            self.cmd_vel_pub.publish(cmd)
            return True, is_arrived
            
        return False, False

    # --- [2] 간호사 좌표 확정 ---
    def calculate_nurse_coordinates(self):
        # 현재 거리는 약 0.8m (후진 완료 후이므로)
        dist = self.TARGET_RADIUS 
        self.nurse_x = self.rx + dist * math.cos(self.ryaw)
        self.nurse_y = self.ry + dist * math.sin(self.ryaw)
        self.get_logger().info(f"📍 간호사 좌표 Lock: ({self.nurse_x:.2f}, {self.nurse_y:.2f})")

    # --- [3] 좌표 기반 공전 ---
    def perform_coordinate_orbit(self):
        elapsed = (self.get_clock().now().nanoseconds / 1e9) - self.orbit_start_time
        if elapsed > 35.0: return True

        dx = self.nurse_x - self.rx
        dy = self.nurse_y - self.ry
        curr_dist = math.hypot(dx, dy)
        target_yaw = math.atan2(dy, dx)

        cmd = Twist()
        
        yaw_error = target_yaw - self.ryaw
        while yaw_error > math.pi: yaw_error -= 2*math.pi
        while yaw_error < -math.pi: yaw_error += 2*math.pi
        cmd.angular.z = yaw_error * 1.5 

        dist_error = curr_dist - self.TARGET_RADIUS
        cmd.linear.x = dist_error * 1.0 
        cmd.linear.y = self.SIDE_SPEED 

        self.cmd_vel_pub.publish(cmd)
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
        # [0] 안정성 체크: 자는 중이면 리턴
        if self.robot_pose is None:
            self.get_logger().info("⏳ Waiting for robot pose...", throttle_duration_sec=2.0)
            return
        
        # 4초간 대기하며 시스템 안정화 (시작하자마자 멈추는 현상 방지)
        if self.start_delay < 8: # 0.5초 * 8 = 4초
            self.start_delay += 1
            if self.start_delay % 2 == 0:
                self.get_logger().info(f"⏳ System Warming up... {self.start_delay}/8")
            return

        # [State 0] 문 앞으로 이동
        if self.state == 0:
            self.send_nav_command(self.ROOM_ENTRANCE)
            self.state = 1

        elif self.state == 1: pass 

        # [State 2] 0.6m 까지 접근
        elif self.state == 2: 
            is_detected, is_arrived = self.process_visual_servoing()
            
            if is_detected:
                if is_arrived: # 0.6m 도달함
                    self.get_logger().info("✅ 0.6m 도달. 1초 대기 후 후진.")
                    self.set_sleep(1.0)
                    self.state = 2.5 # 후진 상태로 이동

            else:
                cmd = Twist()
                cmd.angular.z = 0.3
                self.cmd_vel_pub.publish(cmd)

        # [State 2.5] 0.8m 까지 후진 (뒷걸음질)
        elif self.state == 2.5:
            front_dist = self.get_front_obstacle_dist()
            
            # 목표 거리(0.8m)보다 가까우면 뒤로 가라
            if front_dist < self.TARGET_RADIUS:
                cmd = Twist()
                cmd.linear.x = -0.15 # 천천히 후진
                self.cmd_vel_pub.publish(cmd)
                # self.get_logger().info(f"🔙 거리 벌리는 중... ({front_dist:.2f}m -> 0.80m)")
            else:
                # 0.8m 확보 완료
                self.cmd_vel_pub.publish(Twist()) # 정지
                self.get_logger().info("✅ 안전거리 0.8m 확보 완료. 2초 안정화.")
                self.set_sleep(2.0)
                self.state = 3

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
    node = Mission6NurseController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()