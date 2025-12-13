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
    print("⚠️ ultralytics(YOLO) 라이브러리가 없습니다.")

class Mission6NurseController(Node):
    def __init__(self):
        super().__init__('mission6_nurse_controller')

        # === [1. 모델 경로 자동 설정] ===
        # 본인의 패키지 이름으로 꼭 수정하세요! (예: 'go1_simulation' 등)
        package_name = 'language_command_handler' 
        model_filename = 'yolov8n.pt' # 사용할 모델 파일명
        
        try:
            pkg_share = get_package_share_directory(package_name)
            default_model_path = os.path.join(pkg_share, 'models', model_filename)
        except:
            default_model_path = model_filename # 실패 시 상대 경로

        self.declare_parameter('yolo_model_path', default_model_path)
        self.yolo_path = self.get_parameter('yolo_model_path').value
        
        # YOLO 로드
        self.model = None
        try:
            self.model = YOLO(self.yolo_path)
            self.get_logger().info(f"✅ YOLO 로드 완료: {self.yolo_path}")
        except Exception as e:
            self.get_logger().error(f"❌ YOLO 로드 실패: {e}")

        # === [2. 거리 계산 상수 (Calibration)] ===
        # 공식: Distance = (Real_H * Focal_Length) / Image_H_Pixel
        self.REAL_NURSE_HEIGHT = 1.7  # 간호사 실제 키 (m)
        self.FOCAL_LENGTH = 600.0     # 카메라 초점거리 (환경에 따라 500~800 조절 필요)
        self.CAMERA_CENTER_X = 320.0  # 이미지 가로 해상도(640)의 절반

        # === [3. 좌표 및 경로] ===
        self.ROOM_ENTRANCE = {'x': -6.68, 'y': -24.92, 'yaw': -3.13}
        self.rectangle_path = []

        # === [4. 로봇 상태] ===
        self.current_x = 0.0
        self.current_y = 0.0
        self.current_yaw = 0.0
        self.is_localized = False 

        # === [5. 통신 설정] ===
        self.nav_pub = self.create_publisher(PoseStamped, '/navigator/input_pose', 10)
        self.nav_sub = self.create_subscription(String, '/navigator/status', self.status_callback, 10)
        
        # Localization 노드에서 발행하는 로봇 위치 (/go1_pose)
        self.pose_sub = self.create_subscription(PoseStamped, '/go1_pose', self.pose_callback, 10)
        
        self.img_sub = self.create_subscription(Image, '/camera_face/image', self.img_callback, 10)
        self.speech_pub = self.create_publisher(String, '/robot_dog/speech', 10)

        self.bridge = CvBridge()
        self.cv_image = None
        
        # === [6. 상태 머신 (FSM)] ===
        self.state = 0
        self.path_index = 0
        self.fail_count = 0 

        self.create_timer(0.5, self.mission_loop)
        self.get_logger().info("🧠 Mission 6 Nurse Controller Started (Distance Calc Ver)")

    # ---------------- 콜백 함수 ----------------
    def pose_callback(self, msg):
        """/go1_pose 토픽을 받아 로봇의 현재 위치 갱신"""
        self.current_x = msg.pose.position.x
        self.current_y = msg.pose.position.y
        q = msg.pose.orientation
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        self.current_yaw = math.atan2(siny_cosp, cosy_cosp)
        self.is_localized = True

    def status_callback(self, msg):
        """Navigator 도착 신호 처리"""
        if msg.data == "ARRIVED":
            if self.state == 1: 
                self.get_logger().info("✅ 문 앞 도착. 간호사 탐색 시작.")
                self.state = 2
            
            elif self.state == 4:
                self.get_logger().info(f"✅ 웨이포인트 {self.path_index + 1} 통과.")
                if self.path_index < 3:
                    self.path_index += 1
                    self.send_nav_command(self.rectangle_path[self.path_index])
                else:
                    self.state = 5

    def img_callback(self, msg):
        """카메라 영상 처리 및 디버깅 화면"""
        try:
            self.cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            if self.model is not None:
                # 디버깅용 화면 띄우기 (신뢰도 0.5 이상 표시)
                results = self.model(self.cv_image, verbose=False, conf=0.5)
                annotated_frame = results[0].plot()
                cv2.imshow("Nurse Detector Eye", annotated_frame)
                cv2.waitKey(1)
        except: pass

    # ---------------- 핵심 로직 ----------------
    def calculate_nurse_path(self, dist, angle_offset):
        """
        계산된 거리(dist)와 각도(angle_offset)를 이용해
        지도 상의 절대 좌표를 구하고 직사각형 경로 생성
        """
        # 1. 지도 기준 절대 각도 계산
        global_angle = self.current_yaw + angle_offset
        
        # 2. 간호사 절대 좌표 계산
        nurse_x = self.current_x + dist * math.cos(global_angle)
        nurse_y = self.current_y + dist * math.sin(global_angle)
        
        self.get_logger().info(f"📍 간호사 좌표 확정: ({nurse_x:.2f}, {nurse_y:.2f})")

        # 3. 직사각형 경로 생성 (간호사 중심 1.5m)
        # 반시계 방향: 우하 -> 우상 -> 좌상 -> 좌하
        offset = 1.5 
        p1 = {'x': nurse_x + offset, 'y': nurse_y - offset, 'yaw': 1.57}
        p2 = {'x': nurse_x + offset, 'y': nurse_y + offset, 'yaw': 3.14}
        p3 = {'x': nurse_x - offset, 'y': nurse_y + offset, 'yaw': -1.57}
        p4 = {'x': nurse_x - offset, 'y': nurse_y - offset, 'yaw': 0.0}

        self.rectangle_path = [p1, p2, p3, p4]

    def get_nurse_info(self):
        """YOLO로 인식 후 거리(m)와 각도(rad) 반환"""
        if self.cv_image is None or self.model is None: return False, 0.0, 0.0
        
        results = self.model(self.cv_image, verbose=False)
        
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                
                # 사람(0)이고 신뢰도가 충분할 때
                if cls_id == 0 and conf > 0.6: 
                    # === 거리 계산 ===
                    y1 = float(box.xyxy[0][1])
                    y2 = float(box.xyxy[0][3])
                    h_pixel = y2 - y1 # 박스 높이
                    
                    if h_pixel > 0:
                        distance = (self.REAL_NURSE_HEIGHT * self.FOCAL_LENGTH) / h_pixel
                    else:
                        distance = 2.0
                    
                    # === 각도 계산 ===
                    x1 = float(box.xyxy[0][0])
                    x2 = float(box.xyxy[0][2])
                    cx = (x1 + x2) / 2
                    
                    # 화면 중앙에서 벗어난 정도를 각도로 변환 (0.002는 픽셀->라디안 변환 상수)
                    angle_offset = (self.CAMERA_CENTER_X - cx) * 0.002 
                    
                    return True, distance, -angle_offset # 왼쪽이 +, 오른쪽이 -

        return False, 0.0, 0.0

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

    # ---------------- 메인 루프 ----------------
    def mission_loop(self):
        # [State 0] 시작 -> 문 앞 이동
        if self.state == 0:
            if not self.is_localized:
                if self.fail_count % 10 == 0:
                    self.get_logger().warn("⚠️ /go1_pose 데이터 대기 중... Localization 확인하세요.")
                self.fail_count += 1
                return
            
            self.get_logger().info("COMMAND: 문 앞으로 이동")
            self.send_nav_command(self.ROOM_ENTRANCE)
            self.state = 1
            self.fail_count = 0

        # [State 1] 이동 중
        elif self.state == 1:
            pass

        # [State 2] 문 앞 도착 -> 인식 및 거리 계산
        elif self.state == 2:
            found, dist, angle = self.get_nurse_info()
            
            if found:
                self.get_logger().info(f"✨ 인식 성공! 거리: {dist:.2f}m")
                
                # 거리 유효성 검사 (0.5m ~ 8m 사이만 인정)
                if 0.5 < dist < 8.0:
                    self.calculate_nurse_path(dist, angle)
                    self.state = 4
                    self.path_index = 0
                    self.send_nav_command(self.rectangle_path[0]) # 첫 번째 포인트로 출발
                    self.fail_count = 0
                else:
                    self.get_logger().warn(f"⚠️ 거리 이상 ({dist:.2f}m). 재시도...")
            else:
                self.fail_count += 1
                if self.fail_count % 4 == 0:
                    self.get_logger().warn("❌ 간호사 찾는 중... (화면 확인)")

        # [State 4] 직사각형 순회 중
        elif self.state == 4:
            pass

        # [State 5] 완료
        elif self.state == 5:
            self.get_logger().info("🎉 미션 6 완료!")
            msg = String()
            msg.data = "bark"
            self.speech_pub.publish(msg)
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
