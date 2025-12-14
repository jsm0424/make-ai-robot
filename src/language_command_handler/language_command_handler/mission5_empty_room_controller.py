#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import math
import os
from ament_index_python.packages import get_package_share_directory

try:
    from ultralytics import YOLO
except ImportError:
    print("❌ Ultralytics library not found.")

class MissionEmptyRoomController(Node):
    def __init__(self):
        super().__init__('mission5_empty_room_controller')

        # ================= [좌표 설정] =================
        self.OBSERVATION_POSE = {'x': 3.44, 'y': 9.18, 'yaw': 0.6}

        self.PATH_NO_SIGN = [
            {'x': 6.69, 'y': 10.6, 'yaw': 1.57},
            {'x': 6.69, 'y': 14.0, 'yaw': 1.57},
            {'x': 6.69, 'y': 10.66, 'yaw': -1.57},
            {'x': 0.0, 'y': 9.3, 'yaw': 3.0}
        ]

        self.PATH_WITH_SIGN = [
            {'x': 0.0, 'y': 9.3, 'yaw': 3.0},
            {'x': -6.69, 'y': 10.66, 'yaw': 1.57},
            {'x': -6.69, 'y': 14.0, 'yaw': 1.57},
            {'x': -6.69, 'y': 10.66, 'yaw': -1.57}
        ]
        # ===============================================

        try:
            pkg_share = get_package_share_directory('language_command_handler')
            model_path = os.path.join(pkg_share, 'models', 'sign_model.pt')
        except:
            model_path = 'src/language_command_handler/models/sign_model.pt'

        self.model = None
        try:
            self.model = YOLO(model_path)
            self.get_logger().info(f"✅ Model loaded: {model_path}")
        except:
            self.get_logger().error("❌ Failed to load YOLO model.")

        self.nav_pub = self.create_publisher(PoseStamped, '/navigator/input_pose', 10)
        self.nav_sub = self.create_subscription(String, '/navigator/status', self.status_callback, 10)
        self.img_sub = self.create_subscription(Image, '/camera_face/image', self.img_callback, 10)
        self.speech_pub = self.create_publisher(String, '/robot_dog/speech', 10)

        self.bridge = CvBridge()
        self.cv_image = None
        
        # 상태 관리
        self.state = 0
        self.nav_ready = True
        
        # [핵심 수정] 명령 전송 여부 확인 플래그
        self.goal_sent = False 
        
        # 감지 변수
        self.detect_start_time = 0.0
        self.sign_detect_count = 0
        self.is_detecting = False

        self.current_path = [] 
        self.path_index = 0    

        self.create_timer(0.5, self.control_loop)
        self.get_logger().info("🚀 Mission 5 Controller Started (Logic Fixed!)")

    def img_callback(self, msg):
        try:
            self.cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            if self.is_detecting and self.model is not None:
                results = self.model(self.cv_image, verbose=False, conf=0.6)
                found = False
                for r in results:
                    if len(r.boxes) > 0:
                        found = True
                        break
                if found:
                    self.sign_detect_count += 1
                
                annotated = results[0].plot()
                cv2.imshow("Mission5 Cam", annotated)
                cv2.waitKey(1)
            else:
                if self.cv_image is not None:
                    cv2.imshow("Mission5 Cam", self.cv_image)
                    cv2.waitKey(1)
        except Exception: pass

    def status_callback(self, msg):
        # 네비게이터가 도착했다고 하면 True로 변경
        if msg.data == "ARRIVED":
            self.nav_ready = True

    def send_nav_goal(self, pose_dict):
        # 이미 명령을 수행 중(False)이라면 중복 전송 방지
        if not self.nav_ready: return

        msg = PoseStamped()
        msg.header.frame_id = "map"
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.pose.position.x = float(pose_dict['x'])
        msg.pose.position.y = float(pose_dict['y'])
        yaw = float(pose_dict['yaw'])
        msg.pose.orientation.z = math.sin(yaw / 2.0)
        msg.pose.orientation.w = math.cos(yaw / 2.0)
        
        self.nav_pub.publish(msg)
        
        # 명령을 보냈으므로 '이동 중' 상태로 변경
        self.nav_ready = False 
        self.get_logger().info(f"📍 Moving to: {pose_dict['x']}, {pose_dict['y']}")

    def control_loop(self):
        if self.robot_pose is None:
            self.get_logger().info("⏳ Waiting for robot pose...", throttle_duration_sec=2.0)
            return
        
        # 4초간 대기하며 시스템 안정화 (시작하자마자 멈추는 현상 방지)
        if self.start_delay < 8: # 0.5초 * 8 = 4초
            self.start_delay += 1
            if self.start_delay % 2 == 0:
                self.get_logger().info(f"⏳ System Warming up... {self.start_delay}/8")
            return
        
        
        # [State 0] 시작
        if self.state == 0:
            if self.nav_pub.get_subscription_count() == 0:
                self.get_logger().info("📡 Waiting for Navigator connection...", throttle_duration_sec=1.0)
                return # 연결될 때까지 명령 안 보내고 리턴

            self.state = 1
            self.goal_sent = False # 초기화

        # [State 1] 관측 위치로 이동
        elif self.state == 1:
            # 1. 아직 명령을 안 보냈으면 -> 보낸다
            if not self.goal_sent:
                if self.nav_ready:
                    self.send_nav_goal(self.OBSERVATION_POSE)
                    self.goal_sent = True # "보냈음" 체크
            
            # 2. 명령을 보냈는데, 다시 nav_ready가 True가 되었다 -> 도착했다!
            else:
                if self.nav_ready: 
                    self.get_logger().info("👀 관측 위치 도착! 표지판 스캔 시작 (3초).")
                    self.state = 2
                    self.is_detecting = True
                    self.sign_detect_count = 0
                    self.detect_start_time = self.get_clock().now().nanoseconds / 1e9
                    self.goal_sent = False # 다음 단계를 위해 리셋

        # [State 2] 표지판 감지 및 판단
        elif self.state == 2:
            current_time = self.get_clock().now().nanoseconds / 1e9
            
            if current_time - self.detect_start_time > 3.0:
                self.is_detecting = False
                cv2.destroyAllWindows()
                
                if self.sign_detect_count > 5:
                    self.get_logger().info(f"🛑 STOP 감지됨 ({self.sign_detect_count}회) -> 왼쪽 방(Room 2) 우회")
                    self.current_path = self.PATH_WITH_SIGN
                else:
                    self.get_logger().info(f"🟢 표지판 없음 ({self.sign_detect_count}회) -> 오른쪽 방(Room 1) 진입")
                    self.current_path = self.PATH_NO_SIGN
                
                self.path_index = 0
                self.state = 3 
                self.goal_sent = False # 리셋

        # [State 3] 경로 주행
        elif self.state == 3:
            if self.path_index >= len(self.current_path):
                self.get_logger().info("🎉 미션 5 완료! 멍멍!")
                self.speech_pub.publish(String(data="bark"))
                self.state = 99
                return

            target_pose = self.current_path[self.path_index]

            # 1. 명령 안 보냈으면 -> 보냄
            if not self.goal_sent:
                if self.nav_ready:
                    self.get_logger().info(f"🚶 Step [{self.path_index + 1}/{len(self.current_path)}] 이동 시작")
                    self.send_nav_goal(target_pose)
                    self.goal_sent = True
            
            # 2. 도착했으면 -> 인덱스 증가 및 리셋
            else:
                if self.nav_ready:
                    self.get_logger().info(f"✅ Waypoint 도착.")
                    self.path_index += 1
                    self.goal_sent = False # 다음 웨이포인트를 위해 리셋

        elif self.state == 99:
            pass

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