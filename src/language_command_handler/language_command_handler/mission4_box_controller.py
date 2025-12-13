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

class MissionBoxController(Node):
    def __init__(self):
        super().__init__('mission_box_controller')

        # 1. 관측 위치 (박스 3군데가 다 보이는 곳)
        self.OBSERVATION_POSE = {'x': 0.0, 'y': 7.8, 'yaw': 1.57}

        # 2. 박스 후보 위치 (이미지 상: [왼쪽, 가운데, 오른쪽] 순서)
        # (로봇이 (0, 7.8)에서 Y축(북쪽)을 보고 있으므로, 왼쪽은 x < 0, 오른쪽은 x > 0 입니다)
        self.BOX_LOCATIONS = ['LEFT', 'CENTER', 'RIGHT']

        # 3. [핵심] Waypoints (박스 뒤로 돌아들어가기 위한 경유지들)
        # 박스를 치지 않기 위해 "옆으로 빠졌다가 -> 뒤로 가는" 경로를 설정합니다.
        self.WAYPOINTS = {
            'LEFT': [
                {'x': -3.0, 'y': 10.0, 'yaw': 1.57},
            ],
            'CENTER': [
                {'x': -1.0, 'y': 9.0, 'yaw': 1.57},
                {'x': -1.0, 'y': 15.0, 'yaw': 0},
            ],
            'RIGHT': [
                {'x': 3.0, 'y': 10.0, 'yaw': 1.57},
            ]
        }

        # 4. 밀기 준비 위치 (Push Ready Pose) - 박스 바로 뒤
        # 로봇이 박스를 바라보고 서야 합니다.
        self.PUSH_READY_POSES = {
            'LEFT':   {'x': -3.0, 'y': 12.0, 'yaw': 0}, # 박스(-2,12)의 뒤(y=13.5)에서 남쪽(-1.57)을 봄
            'CENTER': {'x': 0.0,  'y': 15.0, 'yaw': -1.57}, # 박스(0,14)의 뒤(y=15.5)
            'RIGHT':  {'x': 3.0,  'y': 12.0, 'yaw': 3.14}  # 박스(2,12)의 뒤(y=13.5)
        }

        # 5. 밀기 목표 위치 (빨간색 영역 안쪽 좌표)
        # 로봇이 박스를 밀고 들어갈 최종 좌표
        self.GOAL_ZONE_POSES = {
            'LEFT':   {'x': -0.8, 'y': 12.0, 'yaw': 0}, 
            'CENTER': {'x': 0.0,  'y': 12.5, 'yaw': -1.57},
            'RIGHT':  {'x': 0.8,  'y': 12.0, 'yaw': 3.14}
        }
        # ========================================================================================

        # 통신 설정
        self.nav_pub = self.create_publisher(PoseStamped, '/navigator/input_pose', 10)
        self.nav_sub = self.create_subscription(String, '/navigator/status', self.status_callback, 10)
        
        # 로봇 현재 위치 (후진 좌표 계산용)
        self.pose_sub = self.create_subscription(PoseStamped, '/go1_pose', self.pose_callback, 10)
        
        # 카메라 & 짖기
        self.img_sub = self.create_subscription(Image, '/camera_face/image', self.img_callback, 10)
        self.speech_pub = self.create_publisher(String, '/robot_dog/speech', 10)

        self.bridge = CvBridge()
        self.cv_image = None
        self.robot_pose = None

        # 상태 관리
        # 0:시작, 1:관측이동, 2:관측도착, 3:분석, 4:웨이포인트주행, 5:준비위치이동, 6:밀기, 7:후진, 8:종료
        self.step = 0 
        self.target_box_loc = None # 'LEFT', 'CENTER', 'RIGHT' 중 하나
        self.waypoint_queue = []   # 가야할 웨이포인트 목록

        self.create_timer(1.0, self.mission_loop)
        self.get_logger().info("📦 Box Mission Controller Started!")

    def pose_callback(self, msg):
        self.robot_pose = msg

    def status_callback(self, msg):
        """Navigator가 도착했다고 알려줄 때 호출"""
        if msg.data == "ARRIVED":
            if self.step == 1: # 관측 위치 도착
                self.step = 2
            
            elif self.step == 4: # 웨이포인트 하나 도착
                if self.waypoint_queue: # 남은 웨이포인트가 있으면
                    next_wp = self.waypoint_queue.pop(0)
                    self.get_logger().info(f"🚦 Moving to next waypoint: {next_wp}")
                    self.send_nav_command(next_wp)
                else: # 웨이포인트 다 돌았음 -> 준비 위치로
                    self.step = 5 
            
            elif self.step == 5: # 준비 위치 도착 (이제 밀기만 남음)
                self.step = 6
                
            elif self.step == 6: # 밀기 완료 (이제 후진)
                self.step = 7
            
            elif self.step == 7: # 후진 완료
                self.step = 8

    def img_callback(self, msg):
        try:
            self.cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except: pass

    def send_nav_command(self, pose_dict):
        """Navigator에게 좌표 전송"""
        msg = PoseStamped()
        msg.header.frame_id = "map"
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.pose.position.x = float(pose_dict['x'])
        msg.pose.position.y = float(pose_dict['y'])
        
        yaw = float(pose_dict['yaw'])
        msg.pose.orientation.z = math.sin(yaw / 2.0)
        msg.pose.orientation.w = math.cos(yaw / 2.0)
        
        self.nav_pub.publish(msg)

    def analyze_image(self):
        """박스(연한 갈색) 위치 찾기"""
        if self.cv_image is None: return None

        cv2.imwrite("/tmp/robot_view.jpg", self.cv_image)
        img = self.cv_image.copy()
        h, w, _ = img.shape
        
        # 전체 화면 높이 사용
        regions = [img[0:h, 0:w//3], img[0:h, w//3:2*w//3], img[0:h, 2*w//3:w]]
        
        lower_brown = np.array([10, 100, 40])   # 어두운 갈색까지 포함
        upper_brown = np.array([25, 255, 200])


        max_pixels = 0
        detected_idx = -1 # 0:Left, 1:Center, 2:Right
        
        debug_str = "Box Check: "

        for i, region in enumerate(regions):
            hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, lower_brown, upper_brown)
            count = cv2.countNonZero(mask)
            
            debug_str += f"Region{i}={count} "
            
            # 가장 픽셀 수가 많고 일정 이상인 곳 선택
            if count > max_pixels and count > 300:
                max_pixels = count
                detected_idx = i
        
        self.get_logger().info(debug_str)
        
        if detected_idx != -1:
            return self.BOX_LOCATIONS[detected_idx] # 'LEFT', 'CENTER', 'RIGHT' 반환
        return None

    def mission_loop(self):
        # [Step 0] 관측 위치로 이동
        if self.step == 0:
            self.get_logger().info("Command: Move to Observation Point")
            self.send_nav_command(self.OBSERVATION_POSE)
            self.step = 1

        # [Step 1] 이동 중 대기...
        elif self.step == 1: pass

        # [Step 2] 도착 -> 카메라 안정화
        elif self.step == 2:
            self.get_logger().info("👀 Analyzing Box Location...")
            time.sleep(2.0)
            self.step = 3

        # [Step 3] 박스 위치 판단
        elif self.step == 3:
            result = self.analyze_image()
            if result:
                self.target_box_loc = result
                self.get_logger().info(f"📦 Box found at: {self.target_box_loc}")
                
                # 웨이포인트 로드
                self.waypoint_queue = list(self.WAYPOINTS[self.target_box_loc]) # 복사해서 사용
                
                # 첫 번째 웨이포인트로 출발
                if self.waypoint_queue:
                    first_wp = self.waypoint_queue.pop(0)
                    self.get_logger().info("🚦 Starting Waypoint Navigation...")
                    self.send_nav_command(first_wp)
                    self.step = 4 # 웨이포인트 주행 모드
                else:
                    self.step = 5 # 웨이포인트 없으면 바로 준비 위치로
            else:
                self.get_logger().warn("Box not found! Retrying...")

        # [Step 4] 웨이포인트 주행 중 (status_callback이 처리)
        elif self.step == 4: pass

        # [Step 5] 준비 위치(Push Ready)로 이동
        elif self.step == 5:
            # 웨이포인트를 다 돌았으니 이제 박스 뒤로 이동
            target = self.PUSH_READY_POSES[self.target_box_loc]
            self.get_logger().info(f"ready to push at {target}")
            self.send_nav_command(target)
            # step은 status_callback이 바꿔줌 (도착하면 step 6)
            # self.step = 5.5 # 중복 전송 방지용 임시 상태

        # elif self.step == 5.5: pass

        # [Step 6] 박스 밀기 (Goal Zone으로 돌진)
        elif self.step == 6:
            target = self.GOAL_ZONE_POSES[self.target_box_loc]
            self.get_logger().info(f"💪 PUSHING BOX TO {target}")
            self.send_nav_command(target)
            # self.step = 6.5

        # elif self.step == 6.5: pass

        # [Step 7] 2m 후진
        elif self.step == 7:
            if self.robot_pose is None: return
            
            self.get_logger().info("🔙 Backing up 2 meters...")
            
            # 현재 위치에서 로봇이 바라보는 반대 방향으로 2m 좌표 계산
            curr_x = self.robot_pose.pose.position.x
            curr_y = self.robot_pose.pose.position.y
            
            # 쿼터니언 -> Yaw
            q = self.robot_pose.pose.orientation
            curr_yaw = math.atan2(2.0*(q.w*q.z + q.x*q.y), 1.0-2.0*(q.y*q.y + q.z*q.z))
            
            # 후진 좌표 계산: 현재 방향(yaw)의 반대편으로 2m
            back_x = curr_x - 2.0 * math.cos(curr_yaw)
            back_y = curr_y - 2.0 * math.sin(curr_yaw)
            
            back_pose = {'x': back_x, 'y': back_y, 'yaw': curr_yaw} # 방향은 유지
            
            self.send_nav_command(back_pose)
            # self.step = 7.5

        # elif self.step == 7.5: pass

        # [Step 8] 미션 완료
        elif self.step == 8:
            self.get_logger().info("🎉 Box Mission Complete! Bark!")
            msg = String()
            msg.data = "bark"
            self.speech_pub.publish(msg)
            self.step = 9 # 끝

def main(args=None):
    rclpy.init(args=args)
    node = MissionBoxController()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()