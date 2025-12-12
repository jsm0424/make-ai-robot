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
from ament_index_python.packages import get_package_share_directory # 패키지 경로 찾기용

# YOLO 로드 (없으면 에러 처리)
try:
    from ultralytics import YOLO
except ImportError:
    print("❌ Ultralytics library not found. Please install: pip install ultralytics")

class MissionEmptyRoomController(Node):
    def __init__(self):
        super().__init__('mission5_empty_room_controller')

        # 1. 모델 경로 설정 (Launch 파일에서 받거나, 기본값은 패키지 내부 models 폴더)
        # 패키지 설치 경로를 자동으로 찾습니다.
        try:
            pkg_share = get_package_share_directory('language_command_handler')
            default_model_path = os.path.join(pkg_share, 'models', 'sign_model.pt')
        except:
            # 패키지를 못 찾을 경우 (테스트용)
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

        # ================= 좌표 하드코딩 =================
        self.CHECK_POSE = {'x': 3.76, 'y': 8.62, 'yaw': 0.6687}   # 확인 위치
        self.ROOM_POSE = {'x': 7.0, 'y': 14.0, 'yaw': 1.57}      # 빈 방 (Sign 없을 때)
        self.DETOUR_POSE_1 = {'x': 2.0, 'y': 10.0, 'yaw': 2.356} # 우회 1
        self.DETOUR_POSE_2 = {'x': -7.0, 'y': 14.0, 'yaw': 1.57} # 우회 2 (최종)
        # ===============================================

        # 통신 설정
        self.nav_pub = self.create_publisher(PoseStamped, '/navigator/input_pose', 10)
        self.nav_sub = self.create_subscription(String, '/navigator/status', self.status_callback, 10)
        self.img_sub = self.create_subscription(Image, '/camera_face/image', self.img_callback, 10)

        self.bridge = CvBridge()
        self.cv_image = None
        
        # 상태 관리
        self.step = 0 
        self.is_nav_active = False
        
        # 주기적 루프 실행
        self.create_timer(1.0, self.mission_loop)
        
        self.get_logger().info("🧠 Mission 5 Controller Started with Debug View.")

    def status_callback(self, msg):
        if msg.data == "ARRIVED":
            self.is_nav_active = False
            if self.step == 0:
                self.get_logger().info("✅ Arrived at Check Point.")
                self.step = 1
            elif self.step == 3:
                self.get_logger().info("✅ Arrived at Empty Room (No Sign).")
                self.step = 5
            elif self.step == 30:
                self.get_logger().info("✅ Arrived at Detour Waypoint 1.")
                self.step = 4
            elif self.step == 4:
                self.get_logger().info("✅ Arrived at Alternative Room.")
                self.step = 5

    def img_callback(self, msg):
        try:
            # 이미지 변환
            self.cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            
            # [디버깅 화면] 모델이 로드되어 있다면 실시간 추론 결과를 화면에 표시
            if self.model is not None:
                # YOLO 추론 (그림 그리기용, 속도를 위해 conf 낮게 설정 가능)
                results = self.model(self.cv_image, verbose=False, conf=0.5)
                
                # 결과 이미지를 가져옴 (박스가 그려진 이미지)
                annotated_frame = results[0].plot()
                
                # 화면에 띄우기
                cv2.imshow("Mission5 Debug: Stop Sign Detection", annotated_frame)
                cv2.waitKey(1)
                
        except Exception as e:
            self.get_logger().warn(f"Image Error: {e}")

    def send_nav_command(self, pose_dict):
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
        """판단용 YOLO 감지 함수"""
        if self.cv_image is None or self.model is None:
            return False

        # 추론 실행
        results = self.model(self.cv_image, verbose=False)
        
        detected = False
        for result in results:
            if len(result.boxes) > 0:
                for box in result.boxes:
                    conf = box.conf[0].item()
                    cls = int(box.cls[0].item())
                    
                    # Stop Sign 감지 조건 (confidence > 0.5)
                    # 만약 sign_model.pt가 stop sign만 학습했다면 cls 확인 불필요
                    if conf > 0.5: 
                        self.get_logger().info(f"🛑 Found Object! Conf: {conf:.2f}")
                        detected = True
                        break
        return detected

    def mission_loop(self):
        # Step 0: 이동
        if self.step == 0 and not self.is_nav_active:
            self.get_logger().info("Command: Move to Check Point")
            self.send_nav_command(self.CHECK_POSE)
            self.is_nav_active = True

        # Step 1: 안정화
        elif self.step == 1:
            self.get_logger().info("👀 Stabilizing Camera...")
            time.sleep(2.0)
            self.step = 2

        # Step 2: 판단
        elif self.step == 2:
            has_sign = self.detect_stop_sign()
            if has_sign:
                self.get_logger().info("🛑 STOP SIGN DETECTED! Detouring...")
                self.send_nav_command(self.DETOUR_POSE_1)
                self.step = 30
            else:
                self.get_logger().info("🟢 No Sign. Entering Room...")
                self.send_nav_command(self.ROOM_POSE)
                self.step = 3

        # Step 4: 최종 이동
        elif self.step == 4 and not self.is_nav_active:
            self.get_logger().info("Command: Move to Final Destination")
            self.send_nav_command(self.DETOUR_POSE_2)
            self.is_nav_active = True

        # Step 5: 완료
        elif self.step == 5:
            self.get_logger().info("🎉 Mission Complete!")
            self.step = 6

def main(args=None):
    rclpy.init(args=args)
    node = MissionEmptyRoomController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # 종료 시 cv창 닫기
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()