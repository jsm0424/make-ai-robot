#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import math
import time
import os
from ament_index_python.packages import get_package_share_directory

try:
    from ultralytics import YOLO
except ImportError:
    print("❌ Ultralytics library not found. Please install: pip install ultralytics")

class MissionEmptyRoomController(Node):
    def __init__(self):
        super().__init__('mission5_empty_room_controller')

        # 1. 모델 경로 설정
        try:
            pkg_share = get_package_share_directory('language_command_handler')
            default_model_path = os.path.join(pkg_share, 'models', 'sign_model.pt')
        except:
            default_model_path = 'models/sign_model.pt'

        self.declare_parameter('model_path', default_model_path)
        self.model_path = self.get_parameter('model_path').value

        # 2. YOLO 모델 로드
        self.get_logger().info(f"📂 Loading YOLO model from: {self.model_path}")
        try:
            self.model = YOLO(self.model_path)
            self.get_logger().info("✅ YOLO Model Loaded Successfully.")
        except Exception as e:
            self.get_logger().error(f"❌ Failed to load model: {e}")
            self.model = None

        # ================= 좌표 설정 (수정됨) =================
        # 1. 확인 위치 (Stop Sign 확인용)
        self.CHECK_POSE = {'x': 3.76, 'y': 8.62, 'yaw': 0.6687}

        # 2. [CASE 1: Sign 없을 때] -> 오른쪽 빈 방
        self.ROOM_ENTRY_POSE = {'x': 6.66, 'y': 11.41, 'yaw': 1.57} # 문 앞 정렬
        self.ROOM_INSIDE_POSE = {'x': 6.66, 'y': 13.3, 'yaw': 1.57}  # 방 안쪽

        # 3. [CASE 2: Sign 있을 때] -> 왼쪽 빈 방 (우회)
        self.DETOUR_WP1 = {'x': 2.0, 'y': 10.0, 'yaw': 2.356}        # 경유지 1
        self.DETOUR_ENTRY_POSE = {'x': -6.66, 'y': 11.41, 'yaw': 1.57} # 문 앞 정렬
        self.DETOUR_INSIDE_POSE = {'x': -6.66, 'y': 13.3, 'yaw': 1.57} # 방 안쪽
        # ======================================================

        # 통신 설정
        self.nav_pub = self.create_publisher(PoseStamped, '/navigator/input_pose', 10)
        self.nav_sub = self.create_subscription(String, '/navigator/status', self.status_callback, 10)
        self.img_sub = self.create_subscription(Image, '/camera_face/image', self.img_callback, 10)

        self.bridge = CvBridge()
        self.cv_image = None
        
        # 상태 관리 (State Machine)
        # 0: 시작 -> 확인 위치 이동
        # 1: 확인 위치 도착 -> 안정화
        # 2: 이미지 분석
        # [No Sign Branch]
        # 10: 문 앞(ROOM_ENTRY) 이동 중
        # 11: 문 앞 도착 -> 방 안(ROOM_INSIDE) 이동 중
        # [Sign Detected Branch]
        # 20: 우회지(DETOUR_WP1) 이동 중
        # 21: 우회지 도착 -> 문 앞(DETOUR_ENTRY) 이동 중
        # 22: 문 앞 도착 -> 방 안(DETOUR_INSIDE) 이동 중
        # 5: 최종 완료
        self.step = 0 
        self.is_nav_active = False 
        
        self.create_timer(1.0, self.mission_loop)
        self.get_logger().info("🧠 Mission 5 Controller Updated (Door Entry Logic Added).")

    def status_callback(self, msg):
        """Navigator 상태 수신"""
        if msg.data == "ARRIVED":
            self.is_nav_active = False # 명령 수행 완료, 다음 명령 가능 상태로 변경
            
            # [Step 0 -> 1] 확인 위치 도착
            if self.step == 0:
                self.get_logger().info("✅ Arrived at Check Point.")
                self.step = 1
            
            # [Step 10 -> 11] (No Sign) 문 앞 도착 -> 방 안으로 진입 명령
            elif self.step == 10:
                self.get_logger().info("✅ Arrived at Room Door. Entering...")
                self.step = 11

            # [Step 11 -> 5] (No Sign) 방 안 도착 -> 완료
            elif self.step == 11:
                self.get_logger().info("✅ Successfully Entered Empty Room.")
                self.step = 5

            # [Step 20 -> 21] (Sign) 경유지 도착 -> 문 앞으로 이동
            elif self.step == 20:
                self.get_logger().info("✅ Arrived at Detour Waypoint 1.")
                self.step = 21

            # [Step 21 -> 22] (Sign) 문 앞 도착 -> 방 안으로 진입
            elif self.step == 21:
                self.get_logger().info("✅ Arrived at Detour Room Door. Entering...")
                self.step = 22
            
            # [Step 22 -> 5] (Sign) 방 안 도착 -> 완료
            elif self.step == 22:
                self.get_logger().info("✅ Successfully Entered Detour Room.")
                self.step = 5

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

    def send_nav_command(self, pose_dict):
        """좌표 전송 함수"""
        msg = PoseStamped()
        msg.header.frame_id = "map"
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.pose.position.x = pose_dict['x']
        msg.pose.position.y = pose_dict['y']
        yaw = pose_dict['yaw']
        msg.pose.orientation.z = math.sin(yaw / 2.0)
        msg.pose.orientation.w = math.cos(yaw / 2.0)
        self.nav_pub.publish(msg)
        self.is_nav_active = True

    def detect_stop_sign(self):
        if self.cv_image is None or self.model is None: return False
        results = self.model(self.cv_image, verbose=False)
        for result in results:
            for box in result.boxes:
                if box.conf[0].item() > 0.5:
                    return True
        return False

    def mission_loop(self):
        # [Step 0] 시작 -> 확인 위치로 이동
        if self.step == 0 and not self.is_nav_active:
            self.get_logger().info("Command: Move to Check Point")
            self.send_nav_command(self.CHECK_POSE)

        # [Step 1] 도착함 -> 카메라 안정화 대기
        elif self.step == 1:
            self.get_logger().info("👀 Stabilizing Camera...")
            time.sleep(2.0)
            self.step = 2

        # [Step 2] 이미지 분석 및 판단
        elif self.step == 2:
            has_sign = self.detect_stop_sign()
            
            if has_sign:
                self.get_logger().info("🛑 STOP SIGN DETECTED! Taking Detour Path.")
                # 우회 경로 시작 (경유지 1로 이동)
                self.step = 20 
            else:
                self.get_logger().info("🟢 No Sign. Proceeding to Room.")
                # 일반 경로 시작 (문 앞으로 이동)
                self.step = 10

        # [Step 10] (No Sign) 문 앞으로 이동 명령
        elif self.step == 10 and not self.is_nav_active:
            self.get_logger().info("Command: Move to Door Front (Right Room)")
            self.send_nav_command(self.ROOM_ENTRY_POSE)

        # [Step 11] (No Sign) 방 안으로 이동 명령 (도착 후 실행됨)
        elif self.step == 11 and not self.is_nav_active:
            self.get_logger().info("Command: Enter Room (Straight Line)")
            self.send_nav_command(self.ROOM_INSIDE_POSE)

        # [Step 20] (Sign) 경유지로 이동 명령
        elif self.step == 20 and not self.is_nav_active:
            self.get_logger().info("Command: Move to Detour Waypoint 1")
            self.send_nav_command(self.DETOUR_WP1)

        # [Step 21] (Sign) 문 앞으로 이동 명령
        elif self.step == 21 and not self.is_nav_active:
            self.get_logger().info("Command: Move to Door Front (Left Room)")
            self.send_nav_command(self.DETOUR_ENTRY_POSE)

        # [Step 22] (Sign) 방 안으로 이동 명령
        elif self.step == 22 and not self.is_nav_active:
            self.get_logger().info("Command: Enter Detour Room (Straight Line)")
            self.send_nav_command(self.DETOUR_INSIDE_POSE)

        # [Step 5] 완료
        elif self.step == 5:
            self.get_logger().info("🎉 Mission 5 Complete!")
            self.step = 6 # 루프 종료

def main(args=None):
    rclpy.init(args=args)
    node = MissionEmptyRoomController()
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