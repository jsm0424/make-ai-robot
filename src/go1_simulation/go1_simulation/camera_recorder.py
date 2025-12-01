#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import os
from datetime import datetime

class CameraRecorder(Node):
    """
    카메라 토픽을 구독하여 비디오 파일로 저장하는 노드입니다.
    YOLO 학습 데이터 수집용으로 사용됩니다.
    """

    def __init__(self):
        super().__init__('camera_recorder')

        # 파라미터 선언 (기본값 설정)
        # 실제 로봇의 토픽 이름에 맞춰서 수정하거나 ros2 run 시 변경 가능
        self.declare_parameter('image_topic', '/camera_top/image') 
        self.declare_parameter('fps', 30.0)
        self.declare_parameter('save_folder', 'test_videos')
        
        # 파라미터 가져오기
        self.image_topic = self.get_parameter('image_topic').get_parameter_value().string_value
        self.fps = self.get_parameter('fps').get_parameter_value().double_value
        folder_name = self.get_parameter('save_folder').get_parameter_value().string_value

        # CV Bridge 초기화
        self.bridge = CvBridge()
        
        # 비디오 저장 변수
        self.video_writer = None
        self.is_recording = False
        
        # 저장 경로 생성 (워크스페이스 최상위 기준)
        self.save_dir = os.path.join(os.getcwd(), folder_name)
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
            self.get_logger().info(f'폴더 생성됨: {self.save_dir}')

        # 카메라 구독
        self.subscription = self.create_subscription(
            Image,
            self.image_topic,
            self.image_callback,
            10
        )
        
        self.get_logger().info(f'Camera Recorder 시작됨.')
        self.get_logger().info(f'구독 중인 토픽: {self.image_topic}')
        self.get_logger().info(f'저장 경로: {self.save_dir}')
        self.get_logger().info('첫 번째 이미지가 들어오면 자동으로 녹화가 시작됩니다.')

    def image_callback(self, msg):
        try:
            # ROS 이미지를 OpenCV 포맷으로 변환
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            
            # 비디오 작성기가 없으면 초기화 (첫 프레임에서 해상도 결정)
            if self.video_writer is None:
                self.start_new_recording(cv_image)
            
            # 프레임 저장
            if self.is_recording:
                self.video_writer.write(cv_image)
                
                # (선택 사항) 녹화 중임을 터미널에 점으로 표시
                # print(".", end="", flush=True)

        except Exception as e:
            self.get_logger().error(f'이미지 처리 실패: {e}')

    def start_new_recording(self, image):
        """해상도에 맞춰 VideoWriter 초기화"""
        height, width, _ = image.shape
        
        # 파일명 생성 (날짜시간 포함)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"data_{timestamp}.avi"
        filepath = os.path.join(self.save_dir, filename)
        
        # 코덱 설정 (XVID는 호환성이 좋음)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        
        self.video_writer = cv2.VideoWriter(
            filepath, 
            fourcc, 
            self.fps, 
            (width, height)
        )
        
        self.is_recording = True
        self.get_logger().info(f'녹화 시작: {filename} ({width}x{height})')

    def destroy_node(self):
        """노드 종료 시 파일 저장"""
        if self.video_writer is not None:
            self.video_writer.release()
            self.get_logger().info('비디오 파일 저장 완료 및 종료.')
        
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = CameraRecorder()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('사용자에 의해 녹화 중지됨.')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()