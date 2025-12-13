#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import math
import time
import numpy as np

# YOLO 라이브러리 (설치 필요: pip install ultralytics)
try:
    from ultralytics import YOLO
except ImportError:
    print("⚠️ ultralytics(YOLO) 라이브러리가 없습니다. 설치해주세요.")

class Mission6DynamicController(Node):
    def __init__(self):
        super().__init__('mission6_dynamic_controller')

        # === [설정] 파라미터 ===
        self.declare_parameter('yolo_model_path', '')
        self.yolo_path = self.get_parameter('yolo_model_path').value
        
        # YOLO 모델 로드
        self.model = None
        if self.yolo_path:
            self.get_logger().info(f"🚀 Loading YOLO from: {self.yolo_path}")
            try:
                self.model = YOLO(self.yolo_path)
            except Exception as e:
                self.get_logger().error(f"❌ YOLO Load Failed: {e}")

        # === [좌표 설정] ===
        # 1. 방 입구 (여기는 고정)
        self.ROOM_ENTRANCE = {'x': -6.68, 'y': -24.92, 'yaw': -3.13}
        
        # 2. 직사각형 경로 (나중에 계산됨)
        self.rectangle_path = []

        # === [로봇 상태] ===
        self.current_x = 0.0
        self.current_y = 0.0
        self.current_yaw = 0.0
        self.is_localized = False # 로봇 위치 수신 여부

        # === [통신 설정] ===
        # 1. 내비게이션 명령 발행
        self.nav_pub = self.create_publisher(PoseStamped, '/navigator/input_pose', 10)
        
        # 2. 내비게이터 상태 구독
        self.nav_sub = self.create_subscription(String, '/navigator/status', self.status_callback, 10)
        
        # 3. 로봇 현재 위치 구독 (수정됨: /go1_pose 사용)
        # 주의: Localization 노드가 실행되어 있어야 데이터가 들어옵니다.
        self.pose_sub = self.create_subscription(
            PoseStamped, 
            '/go1_pose', 
            self.pose_callback, 
            10
        )

        # 4. 카메라 & 짖기
        self.img_sub = self.create_subscription(Image, '/camera_face/image', self.img_callback, 10)
        self.speech_pub = self.create_publisher(String, '/robot_dog/speech', 10)

        self.bridge = CvBridge()
        self.cv_image = None
        
        # === [상태 관리 FSM] ===
        # 0: 초기화 -> 문 앞 이동
        # 1: 문 앞 이동 중
        # 2: 문 앞 도착 -> 방 진입(직진) 하며 탐색
        # 3: 간호사 발견! -> 좌표 계산 및 경로 생성
        # 4: 직사각형 경로 순회
        # 5: 완료
        self.state = 0
        self.path_index = 0

        self.create_timer(0.5, self.mission_loop)
        self.get_logger().info("🧠 Dynamic Mission 6 Controller Started (using /go1_pose)!")

    # === [콜백 함수들] ===
    def pose_callback(self, msg):
        """
        /go1_pose (PoseStamped) 메시지를 받아서 로봇 상태 갱신
        """
        # PoseStamped는 Odometry와 달리 msg.pose.position으로 바로 접근합니다.
        self.current_x = msg.pose.position.x
        self.current_y = msg.pose.position.y
        
        # 쿼터니언 -> 오일러(Yaw) 변환
        q = msg.pose.orientation
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        self.current_yaw = math.atan2(siny_cosp, cosy_cosp)
        
        self.is_localized = True

    def status_callback(self, msg):
        """네비게이터 도착 신호 처리"""
        if msg.data == "ARRIVED":
            if self.state == 1: # 문 앞 도착
                self.get_logger().info("✅ 문 앞 도착. 방 안으로 진입하며 탐색 시작.")
                self.state = 2
            
            elif self.state == 4: # 직사각형 포인트 도착
                self.get_logger().info(f"✅ 웨이포인트 {self.path_index + 1} 통과.")
                if self.path_index < 3:
                    self.path_index += 1
                    self.send_nav_command(self.rectangle_path[self.path_index])
                else:
                    self.state = 5 # 모든 포인트 완료

    def img_callback(self, msg):
        try:
            self.cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except: pass

    # === [핵심 로직] ===
    def calculate_nurse_path(self):
        """
        현재 로봇 위치(/go1_pose) 기준으로 전방 2m 앞에 간호사가 있다고 가정하고,
        그 주변을 도는 직사각형 좌표 4개를 생성함.
        """
        # 1. 간호사 추정 위치 (로봇 기준 전방 2.0m)
        dist_to_nurse = 2.0
        nurse_x = self.current_x + dist_to_nurse * math.cos(self.current_yaw)
        nurse_y = self.current_y + dist_to_nurse * math.sin(self.current_yaw)
        
        self.get_logger().info(f"📍 간호사 위치 추정: ({nurse_x:.2f}, {nurse_y:.2f})")

        # 2. 직사각형 오프셋 (간호사 기준 동서남북 1.5m 거리)
        offset = 1.5 
        
        # P1, P2, P3, P4 생성 (반시계 방향)
        p1 = {'x': nurse_x + offset, 'y': nurse_y - offset, 'yaw': 1.57} # 우하단
        p2 = {'x': nurse_x + offset, 'y': nurse_y + offset, 'yaw': 3.14} # 우상단
        p3 = {'x': nurse_x - offset, 'y': nurse_y + offset, 'yaw': -1.57} # 좌상단
        p4 = {'x': nurse_x - offset, 'y': nurse_y - offset, 'yaw': 0.0}  # 좌하단

        self.rectangle_path = [p1, p2, p3, p4]
        self.get_logger().info(f"🗺️ 경로 생성 완료: {len(self.rectangle_path)}개 웨이포인트")

    def detect_nurse(self):
        """YOLO로 사람(간호사) 인식"""
        if self.cv_image is None or self.model is None: return False
        
        results = self.model(self.cv_image, verbose=False)
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                # COCO 데이터셋 0번 = person
                if cls_id == 0 and conf > 0.6: 
                    self.get_logger().info(f"👀 간호사 감지됨! (Conf: {conf:.2f})")
                    return True
        return False

    def send_nav_command(self, pose_dict):
        """좌표 명령 전송"""
        msg = PoseStamped()
        msg.header.frame_id = "map"
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.pose.position.x = pose_dict['x']
        msg.pose.position.y = pose_dict['y']
        
        yaw = pose_dict['yaw']
        msg.pose.orientation.z = math.sin(yaw / 2.0)
        msg.pose.orientation.w = math.cos(yaw / 2.0)
        
        self.nav_pub.publish(msg)

    # === [메인 루프] ===
    def mission_loop(self):
        # [State 0] 문 앞으로 출발
        if self.state == 0:
            if not self.is_localized:
                self.get_logger().warn("⚠️ /go1_pose 토픽 대기 중... Localization 노드를 실행하세요.")
                return
            
            self.get_logger().info("COMMAND: 문 앞으로 이동")
            self.send_nav_command(self.ROOM_ENTRANCE)
            self.state = 1

        # [State 1] 이동 중 (대기)
        elif self.state == 1:
            pass

        # [State 2] 방 안으로 조금씩 전진하며 탐색
        elif self.state == 2:
            if self.detect_nurse():
                self.get_logger().info("✨ 간호사 인식 성공! 좌표 계산 중...")
                self.calculate_nurse_path() # 현재 위치 기준으로 경로 생성
                
                # 첫 번째 포인트로 이동 명령
                self.state = 4
                self.path_index = 0
                self.send_nav_command(self.rectangle_path[0])
            else:
                pass 
                # 인식 안 될 경우의 동작 (예: 조금 더 전진) 추가 가능

        # [State 4] 직사각형 도는 중 (status_callback에서 처리)
        elif self.state == 4:
            pass

        # [State 5] 완료
        elif self.state == 5:
            self.get_logger().info("🎉 미션 완료! 멍멍!")
            msg = String()
            msg.data = "bark"
            self.speech_pub.publish(msg)
            self.state = 6

def main(args=None):
    rclpy.init(args=args)
    node = Mission6DynamicController()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
