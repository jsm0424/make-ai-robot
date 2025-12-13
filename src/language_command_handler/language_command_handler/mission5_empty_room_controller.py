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

        # ================= 좌표 설정 =================
        self.CHECK_POSE = {'x': 3.76, 'y': 8.62, 'yaw': 0.6687}
        self.ROOM_ENTRY_POSE = {'x': 6.66, 'y': 11.41, 'yaw': 1.57} 
        self.ROOM_INSIDE_POSE = {'x': 6.66, 'y': 13.3, 'yaw': 1.57}  
        self.DETOUR_WP1 = {'x': 2.0, 'y': 10.0, 'yaw': 2.356}        
        self.DETOUR_ENTRY_POSE = {'x': -6.66, 'y': 11.41, 'yaw': 1.57} 
        self.DETOUR_INSIDE_POSE = {'x': -6.66, 'y': 13.3, 'yaw': 1.57} 
        # ============================================

        # 통신 설정
        self.nav_pub = self.create_publisher(PoseStamped, '/navigator/input_pose', 10)
        self.nav_sub = self.create_subscription(String, '/navigator/status', self.status_callback, 10)
        self.img_sub = self.create_subscription(Image, '/camera_face/image', self.img_callback, 10)

        self.bridge = CvBridge()
        self.cv_image = None
        
        # 상태 변수
        self.step = 0 
        self.is_nav_active = False 
        
        # [NEW] 감지 활성화 플래그 (True일 때만 YOLO 실행)
        self.detection_enabled = False

        self.create_timer(1.0, self.mission_loop)
        self.get_logger().info("🧠 Mission 5 Controller Started (Efficient Detection Mode).")

    def status_callback(self, msg):
        if msg.data == "ARRIVED":
            self.is_nav_active = False 
            
            if self.step == 0:
                self.get_logger().info("✅ Arrived at Check Point.")
                self.step = 1
            elif self.step == 10:
                self.get_logger().info("✅ Arrived at Room Door. Entering...")
                self.step = 11
            elif self.step == 11:
                self.get_logger().info("✅ Successfully Entered Empty Room.")
                self.step = 5
            elif self.step == 20:
                self.get_logger().info("✅ Arrived at Detour Waypoint 1.")
                self.step = 21
            elif self.step == 21:
                self.get_logger().info("✅ Arrived at Detour Room Door. Entering...")
                self.step = 22
            elif self.step == 22:
                self.get_logger().info("✅ Successfully Entered Detour Room.")
                self.step = 5

    def img_callback(self, msg):
        # [NEW] detection_enabled가 False면 이미지 처리 자체를 건너뜀 (CPU 절약)
        if not self.detection_enabled:
            return

        try:
            self.cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            
            # 디버깅 화면도 활성화 상태일 때만 갱신
            if self.model is not None:
                results = self.model(self.cv_image, verbose=False, conf=0.5)
                annotated_frame = results[0].plot()
                cv2.imshow("Mission5 Debug", annotated_frame)
                cv2.waitKey(1)
        except: pass

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
        # 이 함수 호출 시점에는 이미 detection_enabled가 True여야 함
        if self.cv_image is None or self.model is None: return False
        
        # img_callback에서 이미 최신 프레임을 받고 있으므로
        # 여기서는 가장 최근 프레임을 한 번 더 확실하게 추론 (또는 img_callback 결과를 저장해서 써도 됨)
        results = self.model(self.cv_image, verbose=False)
        for result in results:
            for box in result.boxes:
                if box.conf[0].item() > 0.5:
                    return True
        return False

    def mission_loop(self):
        # [Step 0] 이동 중에는 감지 꺼둠
        if self.step == 0 and not self.is_nav_active:
            self.detection_enabled = False # 확실하게 끄기
            self.get_logger().info("Command: Move to Check Point")
            self.send_nav_command(self.CHECK_POSE)

        # [Step 1] 도착 -> 카메라 안정화 및 감지 켜기
        elif self.step == 1:
            self.get_logger().info("👀 Stabilizing Camera & Enabling Detection...")
            self.detection_enabled = True # [NEW] 여기서부터 YOLO 실행 시작
            time.sleep(2.0)
            self.step = 2

        # [Step 2] 판단
        elif self.step == 2:
            has_sign = self.detect_stop_sign()
            
            # 판단 끝났으면 감지 끄기 (자원 절약 및 디버그 창 멈춤)
            self.detection_enabled = False 
            # 디버그 창 닫기 (선택 사항)
            cv2.destroyAllWindows() 

            if has_sign:
                self.get_logger().info("🛑 STOP SIGN DETECTED! Taking Detour Path.")
                self.step = 20 
            else:
                self.get_logger().info("🟢 No Sign. Proceeding to Room.")
                self.step = 10

        # [이후 단계들] 모두 detection_enabled = False 상태 유지
        elif self.step == 10 and not self.is_nav_active:
            self.get_logger().info("Command: Move to Door Front (Right Room)")
            self.send_nav_command(self.ROOM_ENTRY_POSE)

        elif self.step == 11 and not self.is_nav_active:
            self.get_logger().info("Command: Enter Room (Straight Line)")
            self.send_nav_command(self.ROOM_INSIDE_POSE)

        elif self.step == 20 and not self.is_nav_active:
            self.get_logger().info("Command: Move to Detour Waypoint 1")
            self.send_nav_command(self.DETOUR_WP1)

        elif self.step == 21 and not self.is_nav_active:
            self.get_logger().info("Command: Move to Door Front (Left Room)")
            self.send_nav_command(self.DETOUR_ENTRY_POSE)

        elif self.step == 22 and not self.is_nav_active:
            self.get_logger().info("Command: Enter Detour Room (Straight Line)")
            self.send_nav_command(self.DETOUR_INSIDE_POSE)

        elif self.step == 5:
            self.get_logger().info("🎉 Mission 5 Complete!")
            self.step = 6 

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
